"""Mechanical escalation signals (FR-7) and streaming assembly (FR-8).

Escalation is triggered by observable events only — a tool call that fails to
parse against the supplied schema, a structurally missing tool call, an
identical call repeated past threshold — never by asking a model whether it is
struggling. Refusal string-matching is advisory and off by default (MAY).
"""
from __future__ import annotations

import hashlib
import json
from typing import Optional


def tool_schema_index(tools: list) -> dict[str, dict]:
    out = {}
    for t in tools or []:
        if isinstance(t, dict) and t.get("type") == "function":
            fn = t.get("function") or {}
            name = fn.get("name")
            if name:
                out[name] = fn.get("parameters") or {}
    return out


def validate_tool_calls(tools: list, message: dict) -> list[str]:
    """Failures for the emitted tool calls in an assistant message.

    Validation is deliberately structural: the call names a supplied tool, its
    arguments parse as JSON, and the schema's required top-level properties are
    present. Full JSON-Schema validation is out of scope; these three checks
    catch the failure modes that break harnesses (FR-7).
    """
    failures: list[str] = []
    index = tool_schema_index(tools)
    for i, tc in enumerate(message.get("tool_calls") or []):
        fn = (tc or {}).get("function") or {}
        name = fn.get("name") or ""
        if name not in index:
            failures.append(f"tool_calls[{i}] names unknown tool '{name}'")
            continue
        raw_args = fn.get("arguments", "")
        try:
            args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
        except (ValueError, TypeError):
            failures.append(f"tool_calls[{i}] '{name}': arguments are not valid JSON")
            continue
        if not isinstance(args, dict):
            failures.append(f"tool_calls[{i}] '{name}': arguments are not a JSON object")
            continue
        required = index[name].get("required") or []
        missing = [r for r in required if r not in args]
        if missing:
            failures.append(f"tool_calls[{i}] '{name}': missing required "
                            f"argument(s) {missing}")
    return failures


def missing_required_call(body: dict, message: dict) -> bool:
    """FR-7: no tool call where the request structurally required one."""
    tc = body.get("tool_choice")
    required = tc == "required" or (isinstance(tc, dict) and tc.get("type") == "function")
    return required and not (message.get("tool_calls") or [])


def calls_signature(message: dict) -> str:
    """Stable signature of the full emitted call set, for repeat detection."""
    calls = message.get("tool_calls") or []
    canon = sorted(
        (((c.get("function") or {}).get("name") or "",
          (c.get("function") or {}).get("arguments") or "")
         for c in calls))
    return hashlib.sha1(json.dumps(canon).encode()).hexdigest()[:16] if canon else ""


def refusal_matches(text: str, markers: list[str]) -> bool:
    """Advisory only; string-matching is brittle (FR-7 MAY, off by default)."""
    if not markers or not text:
        return False
    lowered = text.lower()
    return any(m.lower() in lowered for m in markers)


def stream_error(obj) -> Optional[dict]:
    """The error object of an SSE data payload, normalised to a dict, or None.
    OpenRouter streams ``{"error": {"code": 502, "message": ..., "metadata":
    {"error_type": "provider_unavailable"}}}`` after answering 200 when the
    provider behind a model fails."""
    err = obj.get("error") if isinstance(obj, dict) else None
    if not err:
        return None
    return err if isinstance(err, dict) else {"message": str(err)}


def stream_error_status(err: dict) -> int:
    """HTTP status an in-stream error stands for (its ``code`` when that is one)."""
    try:
        code = int((err or {}).get("code"))
    except (TypeError, ValueError):
        return 502
    return code if 400 <= code <= 599 else 502


def first_data_event(buf: bytes):
    """The first complete SSE ``data:`` payload in ``buf``: the parsed JSON, the
    string "[DONE]", ``{}`` for non-JSON data, or None while no complete data line
    has arrived. Comment lines (": OPENROUTER PROCESSING") are skipped."""
    for raw in buf.split(b"\n")[:-1]:          # the last piece may be incomplete
        line = raw.strip()
        if not line.startswith(b"data:"):
            continue
        payload = line[5:].strip()
        if payload == b"[DONE]":
            return "[DONE]"
        try:
            return json.loads(payload)
        except ValueError:
            return {}
    return None


class StreamCollector:
    """Reassembles an assistant message from SSE chunks as they relay through.

    Once tokens stream, the turn is committed (FR-8): this collector exists so
    the router can validate post-hoc and arm escalation for the *next* turn,
    not to retract anything.
    """

    def __init__(self):
        self.content: list[str] = []
        self.tool_calls: dict[int, dict] = {}
        self.finish_reason: Optional[str] = None
        self.model: str = ""
        self.usage: dict = {}
        self.error: Optional[dict] = None   # an error object the backend streamed
        self._buf = b""

    def feed(self, chunk: bytes) -> None:
        self._buf += chunk
        while b"\n" in self._buf:
            line, self._buf = self._buf.split(b"\n", 1)
            self._feed_line(line.strip())

    def _feed_line(self, line: bytes) -> None:
        if not line.startswith(b"data:"):
            return
        payload = line[5:].strip()
        if payload == b"[DONE]":
            return
        try:
            obj = json.loads(payload)
        except ValueError:
            return
        if isinstance(obj, dict) and obj.get("error"):
            self.error = stream_error(obj)
            return
        self.model = obj.get("model") or self.model
        # The final chunk carries settled token counts and, on OpenRouter, the
        # real cost. It is the ONLY truthful cost signal a streamed turn ever
        # produces, and it rides a chunk whose `choices` list is empty -- so it
        # has to be read here, not from the assembled message.
        _u = obj.get("usage")
        if isinstance(_u, dict) and _u:
            self.usage = _u
        for choice in obj.get("choices") or []:
            # Routing validation concerns the primary choice only; collecting
            # every choice index interleaves contents and concatenates
            # tool-call fragments across choices into invalid JSON.
            if choice.get("index", 0) != 0:
                continue
            if choice.get("finish_reason"):
                self.finish_reason = choice["finish_reason"]
            delta = choice.get("delta") or {}
            if delta.get("content"):
                self.content.append(delta["content"])
            for tc in delta.get("tool_calls") or []:
                idx = tc.get("index", 0)
                slot = self.tool_calls.setdefault(
                    idx, {"id": "", "type": "function",
                          "function": {"name": "", "arguments": ""}})
                if tc.get("id"):
                    slot["id"] = tc["id"]
                fn = tc.get("function") or {}
                if fn.get("name"):
                    slot["function"]["name"] = fn["name"]
                if fn.get("arguments"):
                    slot["function"]["arguments"] += fn["arguments"]

    def assembled(self) -> dict:
        msg: dict = {"role": "assistant", "content": "".join(self.content)}
        if self.tool_calls:
            msg["tool_calls"] = [self.tool_calls[i] for i in sorted(self.tool_calls)]
        return msg
