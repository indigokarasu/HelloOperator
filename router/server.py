"""The routing server: OpenAI-compatible in, routed backend out (FR-1).

Selection order per request (spec 3.3): capability filter -> role
classification -> cascade position. The cached-affinity path does no
embedding call and no allocation-heavy work (NFR-1); the degenerate
single-model path skips selection entirely (NFR-7).
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

import aiohttp
from aiohttp import web

from .affinity import AffinityMap, SessionState, derive_session_key
from .capabilities import RequestProps, extract_props, model_ok
from .classify import Classifier
from .config import Config, ModelSpec, RoleSpec
from .escalate import (StreamCollector, calls_signature, missing_required_call,
                       refusal_matches, validate_tool_calls)
from .logging_ import DecisionLog

log = logging.getLogger("router.server")

HOP_BY_HOP = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
              "te", "trailers", "transfer-encoding", "upgrade", "content-length",
              "content-encoding", "host"}


@dataclass
class Decision:
    spec: Optional[ModelSpec] = None
    role: str = ""
    pos: int = 0
    kind: str = ""           # new | affinity | transition:<t> | escalation:<t> | pin | degenerate
    trigger: str = ""
    hops_to_charge: int = 0  # committed to the session only when the turn serves
    error_status: int = 0
    error_message: str = ""
    reasons: list[str] = field(default_factory=list)


def _err(status: int, message: str, code: str = "router_error") -> web.Response:
    return web.json_response(
        {"error": {"message": message, "type": code, "code": code}}, status=status)


class Router:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.affinity = AffinityMap(cfg.settings)
        self.decisions = DecisionLog(cfg.settings.decision_log)
        self.http: Optional[aiohttp.ClientSession] = None
        self.classifier: Optional[Classifier] = None
        self._sem: Optional[asyncio.Semaphore] = (
            asyncio.Semaphore(cfg.settings.max_inflight)
            if cfg.settings.max_inflight > 0 else None)

    # ------------------------------------------------------------ lifecycle

    async def startup(self, app: web.Application) -> None:
        self.http = aiohttp.ClientSession()
        self.classifier = Classifier(self.cfg, self.http)
        await self.classifier.prepare()

    async def cleanup(self, app: web.Application) -> None:
        if self.http:
            await self.http.close()

    # ------------------------------------------------------------ selection

    def _role_capable_positions(self, role: RoleSpec, props: RequestProps,
                                reasons: list[str]) -> list[int]:
        """Cascade positions whose model passes the capability filter and the
        role's hard requires."""
        out = []
        for i, mk in enumerate(role.cascade):
            spec = self.cfg.models[mk]
            if role.requires - spec.capabilities:
                reasons.append(f"model '{mk}' lacks role '{role.name}' "
                               f"requires {sorted(role.requires - spec.capabilities)}")
                continue
            ok, why = model_ok(spec, props, self.cfg.settings)
            if ok:
                out.append(i)
            else:
                reasons.append(why)
        return out

    def _eligible_roles(self, props: RequestProps,
                        reasons: list[str]) -> dict[str, list[int]]:
        out = {}
        for role in self.cfg.roles.values():
            positions = self._role_capable_positions(role, props, reasons)
            if positions:
                out[role.name] = positions
        return out

    def _pick_role(self, ranked: Optional[list[tuple[str, float]]],
                   eligible: dict[str, list[int]]) -> str:
        """First eligible role in classifier rank order; degraded -> the
        default role, then declaration order (FR-4, FR-9)."""
        if ranked:
            for name, _ in ranked:
                if name in eligible:
                    return name
        default = self.cfg.settings.default_role
        if default in eligible:
            return default
        return next(iter(eligible))

    async def select(self, props: RequestProps, st: Optional[SessionState],
                     pin: str, compacted_header: bool) -> Decision:
        cfg = self.cfg

        # FR-13 manual pin: bypasses classification, never capability filtering
        # (FR-3 MUST holds even for debugging pins).
        if pin:
            spec = cfg.model_by_key_or_id(pin)
            if spec is None:
                return Decision(error_status=404,
                                error_message=f"pinned model '{pin}' is not registered")
            ok, why = model_ok(spec, props, cfg.settings)
            if not ok:
                return Decision(error_status=400, error_message=why)
            return Decision(spec=spec, role=st.role if st else "", kind="pin",
                            trigger="pin")

        reasons: list[str] = []
        eligible = self._eligible_roles(props, reasons)
        if not eligible:
            unique = list(dict.fromkeys(reasons))
            return Decision(
                error_status=400,
                error_message="no registered model can serve this request: "
                              + "; ".join(unique),
                reasons=unique)

        margin = cfg.settings.classify_margin
        if cfg.settings.switch_cost == "high":
            margin *= 2.0  # high switch cost raises thresholds, not mechanisms (FR-6)

        # Per-session classification cache: mid tool chain the last *user*
        # message is unchanged turn to turn, so the affinity path skips the
        # embedding call entirely (NFR-1, AC-3).
        assert self.classifier is not None
        cls_hash = hashlib.sha1(props.last_user_text.encode()).hexdigest()[:16]
        if st is not None and st.last_cls_hash == cls_hash and st.last_cls_ranked:
            ranked = [tuple(t) for t in st.last_cls_ranked]
        else:
            ranked = await self.classifier.classify(props.last_user_text)
            if st is not None and ranked:
                st.last_cls_hash, st.last_cls_ranked = cls_hash, ranked
        scores = dict(ranked) if ranked else {}

        # ------------------------------------------------ new session
        if st is None or not st.role or st.role not in cfg.roles:
            role_name = self._pick_role(ranked, eligible)
            pos = eligible[role_name][0]
            return Decision(spec=cfg.models[cfg.roles[role_name].cascade[pos]],
                            role=role_name, pos=pos, kind="new", trigger="new-session")

        # ------------------------------------------------ existing session
        role = cfg.roles[st.role]
        current_positions = eligible.get(st.role, [])

        # Escalation armed by a previous turn's mechanical failure (FR-7/FR-8).
        # Consume-on-commit: the trigger is cleared and the hop charged in
        # _post_response_bookkeeping, only after the escalated turn is actually
        # served. If every backend then fails (502), the trigger survives and
        # the hop budget is untouched — spending it here would leave the
        # session sticky-wrong on the failed model with no budget left.
        if st.pending_escalation:
            trigger = st.pending_escalation
            if st.hops_used < cfg.settings.hop_limit:
                nxt = [p for p in current_positions if p > st.pos]
                if nxt:
                    return Decision(spec=cfg.models[role.cascade[nxt[0]]],
                                    role=st.role, pos=nxt[0],
                                    kind=f"escalation:{trigger}", trigger=trigger,
                                    hops_to_charge=1)
                log.info("session %s: escalation '%s' wanted but no capable model "
                         "beyond position %d in role '%s'", st.key, trigger, st.pos, st.role)
                st.pending_escalation = ""   # nowhere to go: drop, don't loop
            else:
                log.info("session %s: escalation '%s' suppressed by hop limit %d",
                         st.key, trigger, cfg.settings.hop_limit)
                st.pending_escalation = ""   # budget exhausted: drop

        # Capability break: the session's model can no longer serve the request
        # (media appeared mid-conversation). Moving is mandatory, not optional.
        if st.pos not in current_positions:
            if current_positions:
                pos = current_positions[0]
                return Decision(spec=cfg.models[role.cascade[pos]], role=st.role,
                                pos=pos, kind="transition:capability",
                                trigger="capability")
            role_name = self._pick_role(ranked, eligible)
            pos = eligible[role_name][0]
            return Decision(spec=cfg.models[cfg.roles[role_name].cascade[pos]],
                            role=role_name, pos=pos, kind="transition:capability",
                            trigger="capability")

        # Observable transitions that permit reclassification (FR-6).
        toolset_changed = props.tools_sig != st.tools_sig and st.turns > 0
        compacted = compacted_header or (
            st.est_tokens > 0 and props.est_tokens < st.est_tokens * 0.6
            and props.msg_count < st.msg_count)

        if toolset_changed or compacted:
            trigger = "toolset" if toolset_changed else "compaction"
            role_name = self._pick_role(ranked, eligible)
            if role_name != st.role:
                pos = eligible[role_name][0]
                return Decision(spec=cfg.models[cfg.roles[role_name].cascade[pos]],
                                role=role_name, pos=pos,
                                kind=f"transition:{trigger}", trigger=trigger)
            # Reclassified into the same role: hold the model (FR-6 "otherwise hold").

        # Classifier disagreement beyond margin (FR-6). Needs a live classifier.
        elif ranked:
            top_eligible = next((n for n, _ in ranked if n in eligible), None)
            if (top_eligible and top_eligible != st.role
                    and scores.get(top_eligible, 0.0) - scores.get(st.role, 0.0) > margin):
                pos = eligible[top_eligible][0]
                return Decision(spec=cfg.models[cfg.roles[top_eligible].cascade[pos]],
                                role=top_eligible, pos=pos,
                                kind="transition:classifier", trigger="classifier")

        # Hold: a 20-step tool chain stays on one model absent a trigger (FR-6).
        return Decision(spec=cfg.models[role.cascade[st.pos]], role=st.role,
                        pos=st.pos, kind="affinity", trigger="")

    # ------------------------------------------------------------ forwarding

    def _failover_candidates(self, decision: Decision,
                             props: RequestProps) -> list[tuple[ModelSpec, str, int]]:
        """Ordered (spec, role, pos): the chosen model, its remaining capable
        cascade, then capable models from other eligible roles (FR-9)."""
        out: list[tuple[ModelSpec, str, int]] = []
        seen: set[str] = set()

        def add(spec: ModelSpec, role: str, pos: int):
            if spec.key not in seen:
                seen.add(spec.key)
                out.append((spec, role, pos))

        assert decision.spec is not None
        add(decision.spec, decision.role, decision.pos)
        role = self.cfg.roles.get(decision.role)
        reasons: list[str] = []
        if role:
            for p in self._role_capable_positions(role, props, reasons):
                if p > decision.pos:
                    add(self.cfg.models[role.cascade[p]], decision.role, p)
        for rname, positions in self._eligible_roles(props, reasons).items():
            if rname == decision.role:
                continue
            for p in positions:
                add(self.cfg.models[self.cfg.roles[rname].cascade[p]], rname, p)
        # Last resort: earlier positions of the chosen role. An escalated
        # session whose stronger model is down is still better served by the
        # weaker live model than by a 502 (FR-9).
        if role:
            reasons2: list[str] = []
            for p in self._role_capable_positions(role, props, reasons2):
                if p < decision.pos:
                    add(self.cfg.models[role.cascade[p]], decision.role, p)
        return out

    def _backend_headers(self, request: web.Request, spec: ModelSpec) -> dict:
        headers = {}
        auth = request.headers.get("Authorization")
        if auth:
            headers["Authorization"] = auth
        elif spec.api_key:
            headers["Authorization"] = f"Bearer {spec.api_key}"
        return headers

    def _forward_body(self, body: dict, spec: ModelSpec) -> dict:
        out = dict(body)
        out["model"] = spec.id
        out.pop("session_id", None)  # router convention, not an OpenAI field
        return out

    async def _post_backend(self, request: web.Request, spec: ModelSpec,
                            body: dict, stream: bool = False) -> aiohttp.ClientResponse:
        assert self.http is not None
        if stream:
            # A healthy long generation must never be killed by a total-body
            # cap; streams are bounded by the gap BETWEEN chunks instead.
            timeout = aiohttp.ClientTimeout(
                total=None, connect=self.cfg.settings.connect_timeout_s,
                sock_read=self.cfg.settings.stream_idle_timeout_s)
        else:
            timeout = aiohttp.ClientTimeout(
                total=self.cfg.settings.request_timeout_s,
                connect=self.cfg.settings.connect_timeout_s)
        return await self.http.post(
            f"{spec.endpoint}/chat/completions",
            json=self._forward_body(body, spec),
            headers=self._backend_headers(request, spec), timeout=timeout)

    # ------------------------------------------------------------ handlers

    async def handle_models(self, request: web.Request) -> web.Response:
        data = [{"id": self.cfg.settings.logical_model, "object": "model",
                 "owned_by": "hermes-model-router"}]
        data += [{"id": k, "object": "model", "owned_by": "hermes-model-router-pin"}
                 for k in self.cfg.models]
        return web.json_response({"object": "list", "data": data})

    async def handle_health(self, request: web.Request) -> web.Response:
        return web.json_response({
            "status": "ok",
            "degenerate": self.cfg.degenerate,
            "classification": ("degraded" if (self.classifier is None
                                              or self.classifier.degraded) else "active"),
            "sessions": len(self.affinity),
        })

    async def handle_chat(self, request: web.Request) -> web.StreamResponse:
        t0 = time.monotonic()
        try:
            body = await request.json()
        except (ValueError, UnicodeDecodeError):
            return _err(400, "request body is not valid JSON")
        if not isinstance(body, dict) or not isinstance(body.get("messages"), list):
            return _err(400, "request must be a JSON object with a 'messages' list")

        cfg = self.cfg
        requested = body.get("model")
        pin = request.headers.get(cfg.settings.pin_header, "")
        if (requested and requested != cfg.settings.logical_model and not pin):
            if cfg.model_by_key_or_id(requested):
                pin = requested  # naming a registered model directly = a pin (FR-13)
            else:
                return _err(404, f"unknown model '{requested}'; this router serves "
                                 f"'{cfg.settings.logical_model}'", "model_not_found")

        props = extract_props(body, cfg.settings)

        # NFR-7: degenerate configs skip selection entirely. The capability
        # filter still runs — FR-3 forbids silently routing media to a
        # text-only model even when it is the only model.
        if cfg.degenerate and not pin:
            spec = next(iter(cfg.models.values()))
            ok, why = model_ok(spec, props, cfg.settings)
            if not ok:
                return _err(400, why)
            decision = Decision(spec=spec, role=next(iter(cfg.roles)), pos=0,
                                kind="degenerate")
            return await self._forward(request, body, props, decision, None, t0)

        session_key = derive_session_key(request.headers, body, cfg.settings)
        st = self.affinity.get(session_key)
        if st is not None and st.pinned and not pin:
            pin = st.pinned

        compacted = request.headers.get(cfg.settings.compaction_header, "") \
            .lower() in ("1", "true", "yes")
        decision = await self.select(props, st, pin, compacted)
        if decision.error_status:
            return _err(decision.error_status, decision.error_message)

        st = self.affinity.ensure(session_key)
        if pin:
            st.pinned = pin
            if decision.kind == "pin":
                log.info("session %s pinned to '%s' (FR-13; frequent use of "
                         "pins is a routing-failure signal, AC-1)", session_key, pin)
        return await self._forward(request, body, props, decision, st, t0)

    # ------------------------------------------------------------ forward path

    async def _forward(self, request: web.Request, body: dict, props: RequestProps,
                       decision: Decision, st: Optional[SessionState],
                       t0: float) -> web.StreamResponse:
        cfg = self.cfg
        routing_ms = round((time.monotonic() - t0) * 1000, 2)
        stream = bool(body.get("stream"))
        role_spec = cfg.roles.get(decision.role)
        buffered = bool(role_spec and role_spec.buffered and props.wants_tools)

        candidates = (self._failover_candidates(decision, props)
                      if not cfg.degenerate and decision.kind != "pin"
                      else [(decision.spec, decision.role, decision.pos)])

        if self._sem is not None:
            async with self._sem:
                resp = await self._try_candidates(request, body, props, decision,
                                                  st, candidates, stream, buffered,
                                                  routing_ms)
        else:
            resp = await self._try_candidates(request, body, props, decision, st,
                                              candidates, stream, buffered, routing_ms)
        return resp

    async def _try_candidates(self, request, body, props, decision, st,
                              candidates, stream, buffered, routing_ms):
        cfg = self.cfg
        last_error = "no candidates"
        for i, (spec, role, pos) in enumerate(candidates):
            if i > 0:
                decision = Decision(spec=spec, role=role, pos=pos,
                                    kind="failover", trigger="backend-unreachable")
                log.warning("failing over to '%s' (%s)", spec.key, last_error)
            try:
                if stream and not buffered:
                    return await self._forward_stream(request, body, props,
                                                      decision, st, routing_ms)
                return await self._forward_buffered(request, body, props, decision,
                                                    st, routing_ms,
                                                    emit_stream=stream)
            except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as e:
                # Safe to try the next candidate: _forward_stream never raises
                # once bytes have been written to the client (FR-8 commit).
                last_error = f"{spec.key}: {e.__class__.__name__}: {e}"
                continue
            except _RetryableStatus as e:
                last_error = f"{spec.key}: backend returned {e.status}"
                continue
        # FR-9: never silently fail without naming what went wrong.
        return _err(502, f"no backend could serve the request; last error: {last_error}",
                    "backends_unavailable")

    async def _forward_buffered(self, request, body, props, decision: Decision,
                                st: Optional[SessionState], routing_ms: float,
                                emit_stream: bool) -> web.StreamResponse:
        """Non-streaming backend call: validate before release, escalate
        in-request within the hop cap (the natural buffered path, FR-8).

        If an in-request escalation target turns out to be unreachable, the
        previously obtained (validation-failing) answer is served and the next
        turn is armed instead — an imperfect answer beats a 502 (FR-9)."""
        cfg = self.cfg
        spec = decision.spec
        assert spec is not None
        send_body = dict(body)
        send_body["stream"] = False

        payload: Optional[dict] = None
        message: dict = {}
        failures: list[str] = []
        prev = None   # (spec, decision, payload, message, failures) before a hop
        while True:
            try:
                client_resp = await self._post_backend(request, spec, send_body)
            except (aiohttp.ClientError, asyncio.TimeoutError, OSError):
                if prev is None:
                    raise               # first attempt: let _try_candidates fail over
                spec, decision, payload, message, failures = prev
                log.warning("in-request escalation target unreachable; serving "
                            "the previous answer and arming the next turn (FR-9)")
                break
            async with client_resp:
                if client_resp.status >= 500:
                    if prev is not None:
                        spec, decision, payload, message, failures = prev
                        break
                    raise _RetryableStatus(client_resp.status)
                try:
                    payload = await client_resp.json(content_type=None)
                except (ValueError, aiohttp.ClientError):
                    # HTML error page from a proxy, truncated body: the backend
                    # is not speaking the protocol — fail over (FR-9).
                    if prev is not None:
                        spec, decision, payload, message, failures = prev
                        break
                    raise _RetryableStatus(client_resp.status)
                if client_resp.status >= 400:
                    return web.json_response(payload, status=client_resp.status,
                                             headers=self._router_headers(
                                                 decision, routing_ms))
            message = (payload.get("choices") or [{}])[0].get("message") or {}
            failures = []
            if props.wants_tools:
                failures = validate_tool_calls(body.get("tools") or [], message)
                if missing_required_call(body, message):
                    failures.append("tool_choice required a call; none was emitted")
            if (failures and st is not None
                    and st.hops_used + decision.hops_to_charge < cfg.settings.hop_limit):
                nxt = self._next_capable(decision, props)
                if nxt is not None:
                    prev = (spec, decision, payload, message, failures)
                    log.info("in-request escalation (%s) -> '%s': %s",
                             decision.kind, nxt[0].key, "; ".join(failures))
                    spec = nxt[0]
                    decision = Decision(spec=spec, role=decision.role, pos=nxt[1],
                                        kind="escalation:tool-validation",
                                        trigger="tool-validation",
                                        hops_to_charge=decision.hops_to_charge + 1)
                    continue
            break

        self._post_response_bookkeeping(body, props, decision, st, message, failures)
        if failures and st is not None:
            # The served answer still fails validation (hop-capped, escalated
            # model failed too, or its backend was down): arm the next turn.
            st.pending_escalation = st.pending_escalation or "tool-validation"
        self._log_decision(st, decision, routing_ms, escalation_failures=failures)
        headers = self._router_headers(decision, routing_ms)
        assert payload is not None
        payload["router"] = {"model": spec.key, "backend_model": spec.id,
                             "role": decision.role, "pos": decision.pos,
                             "decision": decision.kind}
        if not emit_stream:
            return web.json_response(payload, headers=headers)
        # Buffered mode on a streaming request: validated, now emitted as a
        # short SSE stream (latency traded for retractability, FR-8).
        return await self._emit_as_stream(request, payload, headers)

    async def _forward_stream(self, request, body, props, decision: Decision,
                              st: Optional[SessionState],
                              routing_ms: float) -> web.StreamResponse:
        """Streaming relay. Once tokens flow the turn is committed; validation
        happens on the assembled message and arms escalation for the next turn
        (FR-8)."""
        spec = decision.spec
        assert spec is not None
        client_resp = await self._post_backend(request, spec, body, stream=True)
        if client_resp.status >= 500:
            client_resp.close()
            raise _RetryableStatus(client_resp.status)
        collector = StreamCollector()
        headers = self._router_headers(decision, routing_ms)
        if client_resp.status >= 400:
            try:
                payload = await client_resp.json(content_type=None)
            except (ValueError, aiohttp.ClientError):
                client_resp.close()
                raise _RetryableStatus(client_resp.status)
            client_resp.close()
            return web.json_response(payload, status=client_resp.status, headers=headers)

        resp = web.StreamResponse(status=client_resp.status)
        resp.headers["Content-Type"] = client_resp.headers.get(
            "Content-Type", "text/event-stream")
        for k, v in headers.items():
            resp.headers[k] = v
        # Everything before prepare() may raise: nothing has been written and
        # _try_candidates can safely fail over. From prepare() on, the turn is
        # COMMITTED (FR-8): a mid-relay backend death terminates the stream,
        # arms escalation for the next turn, and never raises upward — a retry
        # would write a second HTTP response into the open body.
        await resp.prepare(request)
        aborted = ""
        try:
            try:
                async for chunk in client_resp.content.iter_any():
                    await resp.write(chunk)
                    collector.feed(chunk)
            finally:
                client_resp.close()
            await resp.write_eof()
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError,
                ConnectionResetError) as e:
            aborted = f"{e.__class__.__name__}: {e}"
            log.warning("backend stream aborted mid-relay (%s); turn is committed, "
                        "arming escalation for the next turn (FR-8/FR-9)", aborted)
            try:
                await resp.write_eof()
            except Exception:
                pass  # client may already be gone; nothing left to salvage

        message = collector.assembled()
        failures = []
        if props.wants_tools:
            failures = validate_tool_calls(body.get("tools") or [], message)
            if missing_required_call(body, message):
                failures.append("tool_choice required a call; none was emitted")
        if aborted:
            failures.append(f"stream aborted mid-relay ({aborted})")
        # Bookkeeping first (it commits/clears the consumed escalation), then
        # arm the NEXT turn from this turn's failures.
        self._post_response_bookkeeping(body, props, decision, st, message, failures)
        if failures and st is not None:
            st.pending_escalation = st.pending_escalation or (
                "stream-abort" if aborted else "tool-validation")
            log.info("streamed turn failed; escalation armed for next turn "
                     "(FR-8): %s", "; ".join(failures))
        self._log_decision(st, decision, routing_ms, escalation_failures=failures)
        return resp

    async def _emit_as_stream(self, request, payload: dict,
                              headers: dict) -> web.StreamResponse:
        resp = web.StreamResponse()
        resp.headers["Content-Type"] = "text/event-stream"
        for k, v in headers.items():
            resp.headers[k] = v
        await resp.prepare(request)
        choice = (payload.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        chunk = {
            "id": payload.get("id", f"chatcmpl-{uuid.uuid4().hex[:12]}"),
            "object": "chat.completion.chunk",
            "created": payload.get("created", int(time.time())),
            "model": payload.get("model", ""),
            "router": payload.get("router", {}),
            "choices": [{"index": 0, "delta": message,
                         "finish_reason": choice.get("finish_reason")}],
        }
        await resp.write(b"data: " + json.dumps(chunk).encode() + b"\n\n")
        await resp.write(b"data: [DONE]\n\n")
        await resp.write_eof()
        return resp

    # ------------------------------------------------------------ bookkeeping

    def _next_capable(self, decision: Decision,
                      props: RequestProps) -> Optional[tuple[ModelSpec, int]]:
        role = self.cfg.roles.get(decision.role)
        if not role:
            return None
        reasons: list[str] = []
        for p in self._role_capable_positions(role, props, reasons):
            if p > decision.pos:
                return self.cfg.models[role.cascade[p]], p
        return None

    def _post_response_bookkeeping(self, body, props, decision: Decision,
                                   st: Optional[SessionState], message: dict,
                                   failures: list[str]) -> None:
        if st is None:
            return
        # Consume-on-commit (FR-7): the turn was actually served, so now the
        # escalation trigger that produced this decision is cleared and its
        # hops are charged. On total failure this never runs, so the trigger
        # survives for the next turn and no budget is burned.
        if decision.kind.startswith("escalation:"):
            st.hops_used += max(decision.hops_to_charge, 0)
            if st.pending_escalation == decision.trigger:
                st.pending_escalation = ""
        st.role = decision.role or st.role
        st.model_key = decision.spec.key if decision.spec else st.model_key
        st.pos = decision.pos
        st.tools_sig = props.tools_sig
        st.est_tokens = props.est_tokens
        st.msg_count = props.msg_count
        self.affinity.touch(st)

        # Identical call with identical arguments repeated past threshold ->
        # escalate next turn (FR-7, per-session recent-call memory).
        sig = calls_signature(message)
        if sig:
            if sig == st.last_call_sig:
                st.last_call_count += 1
            else:
                st.last_call_sig, st.last_call_count = sig, 1
            if st.last_call_count >= self.cfg.settings.repeat_call_threshold:
                st.pending_escalation = st.pending_escalation or "repeat-call"
                st.last_call_count = 0
        # Refusal detection: advisory only unless explicitly enabled (FR-7 MAY).
        markers = self.cfg.settings.refusal_markers
        if markers and refusal_matches(message.get("content") or "", markers):
            log.info("session %s: response matched a refusal marker (advisory)", st.key)
            if self.cfg.settings.refusal_escalate:
                st.pending_escalation = st.pending_escalation or "refusal"

    def _router_headers(self, decision: Decision, routing_ms: float) -> dict:
        spec = decision.spec
        return {
            "x-router-model": spec.key if spec else "",
            "x-router-backend-model": spec.id if spec else "",
            "x-router-role": decision.role,
            "x-router-pos": str(decision.pos),
            "x-router-decision": decision.kind,
            "x-router-latency-ms": str(routing_ms),
        }

    def _log_decision(self, st: Optional[SessionState], decision: Decision,
                      routing_ms: float, escalation_failures: list[str]) -> None:
        self.decisions.write(
            session=st.key if st else "",
            role=decision.role,
            model=decision.spec.key if decision.spec else "",
            backend_model=decision.spec.id if decision.spec else "",
            pos=decision.pos,
            decision=decision.kind,
            trigger=decision.trigger,
            routing_latency_ms=routing_ms,
            validation_failures=escalation_failures or None,
        )


class _RetryableStatus(Exception):
    def __init__(self, status: int):
        self.status = status


ROUTER_KEY = web.AppKey("router", "Router")


def build_app(cfg: Config) -> web.Application:
    router = Router(cfg)
    app = web.Application(client_max_size=256 * 1024 * 1024)
    app[ROUTER_KEY] = router
    app.on_startup.append(router.startup)
    app.on_cleanup.append(router.cleanup)
    app.router.add_post("/v1/chat/completions", router.handle_chat)
    app.router.add_get("/v1/models", router.handle_models)
    app.router.add_get("/healthz", router.handle_health)
    return app
