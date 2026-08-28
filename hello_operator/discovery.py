"""Model discovery (FR-15, spec 3.1a).

Three detection layers, in decreasing reliability:
  1. backend metadata (llama.cpp /props, Ollama /api/show, LM Studio
     /api/v0/models, vLLM /v1/models max_model_len; generic /v1/models as the
     weakest fallback)
  2. active probes — opt-in, lazy, serial (a probe against an on-demand model
     forces a full load, which is precisely the cost the user configured
     around)
  3. name heuristics — tie-breaking hints only, never the sole basis for a
     capability

Discovery must be able to fail entirely — a backend exposing nothing beyond
model ids — and leave the router fully usable on declarations alone.
"""
from __future__ import annotations

import base64
import json
import logging
import time
from typing import Any, Optional

import aiohttp

from .config import Config, ModelSpec

log = logging.getLogger("router.discovery")

# 1x1 transparent PNG for the vision probe.
_TINY_PNG = base64.b64encode(base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGNgYGBgAAAABQAB"
    "h6FO1AAAAABJRU5ErkJggg==")).decode()

NAME_HINTS = [
    (("-vl-", "-vl:", "vision", "llava", "qwen-vl", "pixtral"), "vision"),
    (("coder", "-code", "codestral", "deepseek-coder"), "code"),
    (("embed", "bge-", "gte-", "nomic-embed", "minilm"), "embeddings"),
    (("abliterated", "uncensored", "dolphin"), "unfiltered"),
]


def name_hints(model_id: str) -> set[str]:
    hints = set()
    low = model_id.lower()
    for needles, hint in NAME_HINTS:
        if any(n in low for n in needles):
            hints.add(hint)
    return hints


def _native_base(endpoint: str) -> str:
    """Strip a trailing /v1 to reach a backend's native API root."""
    return endpoint[:-3] if endpoint.endswith("/v1") else endpoint


async def _get_json(http: aiohttp.ClientSession, url: str, timeout: float = 5.0):
    try:
        async with http.get(url, timeout=aiohttp.ClientTimeout(total=timeout)) as r:
            if r.status != 200:
                return None
            return await r.json(content_type=None)
    except (aiohttp.ClientError, OSError, ValueError):
        return None


async def _post_json(http: aiohttp.ClientSession, url: str, payload: dict,
                     timeout: float = 10.0, headers: Optional[dict] = None):
    try:
        async with http.post(url, json=payload, headers=headers or {},
                             timeout=aiohttp.ClientTimeout(total=timeout)) as r:
            if r.status != 200:
                return None
            return await r.json(content_type=None)
    except (aiohttp.ClientError, OSError, ValueError):
        return None


async def list_backend_models(http: aiohttp.ClientSession, endpoint: str) -> Optional[list[dict]]:
    data = await _get_json(http, f"{endpoint}/models")
    if isinstance(data, dict) and isinstance(data.get("data"), list):
        return data["data"]
    return None


async def harvest(http: aiohttp.ClientSession, endpoint: str, model_id: str) -> dict:
    """Best-effort metadata for one model. Returns detected fields only —
    absent keys mean the backend exposed nothing for them."""
    detected: dict[str, Any] = {}
    caps: set[str] = set()
    native = _native_base(endpoint)

    # Ollama: /api/show carries an explicit capabilities field.
    show = await _post_json(http, f"{native}/api/show", {"name": model_id})
    if isinstance(show, dict) and ("model_info" in show or "capabilities" in show):
        detected["source"] = "ollama"
        for c in show.get("capabilities") or []:
            caps.add({"completion": "text", "vision": "vision", "tools": "tools",
                      "thinking": "reasoning-effort", "embedding": "embeddings"}.get(c, c))
        info = show.get("model_info") or {}
        for k, v in info.items():
            if k.endswith(".context_length") and isinstance(v, int):
                detected["context_window"] = v
                break
        details = show.get("details") or {}
        if details.get("family"):
            detected["family"] = details["family"]
        if details.get("quantization_level"):
            detected["quantization"] = details["quantization_level"]
        tags = await _get_json(http, f"{native}/api/tags")
        if isinstance(tags, dict):
            for m in tags.get("models") or []:
                if m.get("name") in (model_id, f"{model_id}:latest"):
                    detected["digest"] = m.get("digest", "")
                    break

    # LM Studio: /api/v0/models is per-model and typed.
    if "source" not in detected:
        lms = await _get_json(http, f"{native}/api/v0/models")
        if isinstance(lms, dict) and isinstance(lms.get("data"), list):
            for m in lms["data"]:
                if m.get("id") == model_id:
                    detected["source"] = "lmstudio"
                    if isinstance(m.get("max_context_length"), int):
                        detected["context_window"] = m["max_context_length"]
                    mtype = m.get("type", "")
                    if mtype == "vlm":
                        caps.update({"text", "vision"})
                    elif mtype == "embeddings":
                        caps.add("embeddings")
                    elif mtype == "llm":
                        caps.add("text")
                    if m.get("state") == "not-loaded":
                        detected["residency"] = "on-demand"
                    if m.get("quantization"):
                        detected["quantization"] = m["quantization"]
                    break

    # llama.cpp: /props is server-wide (one model per server).
    if "source" not in detected:
        props = await _get_json(http, f"{native}/props")
        if isinstance(props, dict) and ("default_generation_settings" in props
                                        or "model_path" in props):
            detected["source"] = "llama.cpp"
            dgs = props.get("default_generation_settings") or {}
            n_ctx = dgs.get("n_ctx")
            if isinstance(n_ctx, int):
                detected["context_window"] = n_ctx
            modalities = props.get("modalities") or {}
            if modalities.get("vision"):
                caps.add("vision")
            if modalities.get("audio"):
                caps.add("audio")
            caps.add("text")
            if props.get("model_path"):
                detected["family"] = str(props["model_path"]).rsplit("/", 1)[-1]

    # Generic /v1/models: weakest fallback. vLLM extends it with max_model_len.
    if "context_window" not in detected or "source" not in detected:
        models = await list_backend_models(http, endpoint)
        if models is not None:
            for m in models:
                if m.get("id") == model_id:
                    detected.setdefault("source", "openai-generic")
                    mml = m.get("max_model_len")           # vLLM extension
                    if isinstance(mml, int):
                        detected["context_window"] = mml
                        detected["source"] = "vllm"
                    cl = m.get("context_length")           # OpenRouter extension
                    if isinstance(cl, int) and "context_window" not in detected:
                        detected["context_window"] = cl
                        detected["source"] = "openrouter-style"
                    arch = m.get("architecture") or {}
                    mods = arch.get("input_modalities") or []
                    if "image" in mods:
                        detected.setdefault("capabilities", [])
                    break

    if caps:
        detected["capabilities"] = sorted(caps)
    return detected


# ---------------------------------------------------------------- probes ----

async def probe_tools(http: aiohttp.ClientSession, spec: ModelSpec) -> Optional[bool]:
    """Verified by response shape: does the model emit a parsable tool call
    when one is trivially appropriate?"""
    body = {
        "model": spec.id, "max_tokens": 64,
        "messages": [{"role": "user", "content": "What is the weather in Paris? Use the tool."}],
        "tools": [{"type": "function", "function": {
            "name": "get_weather",
            "parameters": {"type": "object",
                           "properties": {"city": {"type": "string"}},
                           "required": ["city"]}}}],
        "tool_choice": "auto",
    }
    resp = await _chat(http, spec, body)
    if resp is None:
        return None
    msg = (resp.get("choices") or [{}])[0].get("message") or {}
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function") or {}
        if fn.get("name"):
            try:
                json.loads(fn.get("arguments") or "{}")
                return True
            except ValueError:
                return False
    return False


async def probe_vision(http: aiohttp.ClientSession, spec: ModelSpec) -> Optional[bool]:
    body = {
        "model": spec.id, "max_tokens": 16,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "Reply with one word: what color dominates?"},
            {"type": "image_url",
             "image_url": {"url": f"data:image/png;base64,{_TINY_PNG}"}},
        ]}],
    }
    resp = await _chat(http, spec, body, ok_statuses=(200,))
    if resp is None:
        return False   # a vision-capable server accepts the request shape
    return bool((resp.get("choices") or []))


async def probe_json(http: aiohttp.ClientSession, spec: ModelSpec) -> Optional[bool]:
    body = {
        "model": spec.id, "max_tokens": 64,
        "messages": [{"role": "user", "content": 'Return {"ok": true} exactly.'}],
        "response_format": {"type": "json_object"},
    }
    resp = await _chat(http, spec, body)
    if resp is None:
        return None
    content = ((resp.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
    try:
        json.loads(content)
        return True
    except ValueError:
        return False


async def bench_speed(http: aiohttp.ClientSession, spec: ModelSpec) -> Optional[dict]:
    """Timed micro-benchmark at fixed token count. Callers must run these
    serially, never concurrently, to avoid contention-skewed numbers (3.1a)."""
    body = {"model": spec.id, "max_tokens": 48, "stream": True,
            "messages": [{"role": "user", "content": "Count from one to forty in words."}]}
    headers = {"Authorization": f"Bearer {spec.api_key}"} if spec.api_key else {}
    t0 = time.monotonic()
    ttft = None
    tokens = 0
    try:
        async with http.post(f"{spec.endpoint}/chat/completions", json=body,
                             headers=headers,
                             timeout=aiohttp.ClientTimeout(total=120)) as r:
            if r.status != 200:
                return None
            async for raw in r.content:
                line = raw.strip()
                if not line.startswith(b"data:") or line[5:].strip() == b"[DONE]":
                    continue
                try:
                    obj = json.loads(line[5:])
                except ValueError:
                    continue
                delta = (obj.get("choices") or [{}])[0].get("delta") or {}
                if delta.get("content"):
                    if ttft is None:
                        ttft = time.monotonic() - t0
                    tokens += 1
    except (aiohttp.ClientError, OSError):
        return None
    total = time.monotonic() - t0
    if ttft is None or tokens == 0:
        return None
    gen_time = max(total - ttft, 1e-6)
    tps = tokens / gen_time
    speed_class = "fast" if tps > 40 else ("mid" if tps > 12 else "slow")
    return {"ttft_s": round(ttft, 3), "tokens_per_s": round(tps, 1),
            "speed_class": speed_class}


async def _chat(http: aiohttp.ClientSession, spec: ModelSpec, body: dict,
                ok_statuses=(200,)) -> Optional[dict]:
    headers = {"Authorization": f"Bearer {spec.api_key}"} if spec.api_key else {}
    try:
        async with http.post(f"{spec.endpoint}/chat/completions", json=body,
                             headers=headers,
                             timeout=aiohttp.ClientTimeout(total=120)) as r:
            if r.status not in ok_statuses:
                return None
            return await r.json(content_type=None)
    except (aiohttp.ClientError, OSError, ValueError):
        return None


async def run_probes(http: aiohttp.ClientSession, spec: ModelSpec) -> dict:
    """All probes for one model, serially. Opt-in per model (3.1a)."""
    out: dict[str, Any] = {}
    caps = set()
    tools = await probe_tools(http, spec)
    if tools:
        caps.add("tools")
    vision = await probe_vision(http, spec)
    if vision:
        caps.add("vision")
    jsn = await probe_json(http, spec)
    if jsn:
        caps.add("json")
    if caps:
        caps.add("text")
        out["capabilities"] = sorted(caps)
    bench = await bench_speed(http, spec)
    if bench:
        out["speed_class"] = bench["speed_class"]
        out["bench"] = bench
    return out


# ------------------------------------------------------- cache and drift ----

def cache_load(cfg: Config) -> dict:
    try:
        return json.loads(cfg.state_path("discovery.json").read_text())
    except (OSError, ValueError):
        return {}


def cache_save(cfg: Config, cache: dict) -> None:
    try:
        cfg.state_path("discovery.json").write_text(json.dumps(cache, indent=1))
    except OSError as e:
        log.warning("could not persist discovery cache: %s", e)


def _cache_key(spec: ModelSpec, detected: dict) -> str:
    version = detected.get("digest") or detected.get("quantization") or ""
    return f"{spec.id}@{spec.endpoint}:{version}"


async def detect_all(http: aiohttp.ClientSession, cfg: Config,
                     probe: bool = False) -> dict[str, dict]:
    """Detected fields per model key.

    Metadata is harvested fresh every call — it is one cheap HTTP round-trip
    and discarding it in favor of a stale cache would hide backend changes.
    The cache exists for PROBE results, keyed on model id + digest where the
    backend exposes one, so probes re-run only when the model changes (3.1a).
    """
    cache = cache_load(cfg)
    out: dict[str, dict] = {}
    specs = list(cfg.models.values()) + ([cfg.embedding] if cfg.embedding else [])
    for spec in specs:
        detected = await harvest(http, spec.endpoint, spec.id)
        key = _cache_key(spec, detected)
        cached_probes = (cache.get(key) or {}).get("probes") or {}

        # Probe gate: opt-in flag, and never against an on-demand model — a
        # probe forces a full load, precisely the cost that configuration
        # avoids. Declared on-demand OR detected not-loaded both count, unless
        # the user set probe: true explicitly on the model (3.1a).
        costly = (spec.residency == "on-demand"
                  or detected.get("residency") == "on-demand"
                  or spec.location == "remote")
        want_probe = probe and (spec.probe is True
                                or (spec.probes_enabled and not costly))
        if want_probe:
            # Serial by construction: one model at a time in this loop.
            probes = await run_probes(http, spec)
        else:
            probes = cached_probes

        merged = dict(detected)
        if probes:
            # Probes can only see {text, tools, vision, json}; union with the
            # metadata layer's capabilities rather than clobbering them
            # (reasoning-effort/embeddings/audio come from metadata only).
            caps = set(detected.get("capabilities") or []) |                 set(probes.get("capabilities") or [])
            merged.update({k: v for k, v in probes.items() if k != "capabilities"})
            if caps:
                merged["capabilities"] = sorted(caps)
        cache[key] = {"ts": time.time(), "data": detected, "probes": probes}
        out[spec.key] = merged
    cache_save(cfg, cache)
    return out


async def drift_check(http: aiohttp.ClientSession, cfg: Config) -> list[str]:
    """Flag registry drift; never silently add or drop routing targets."""
    notes = []
    by_endpoint: dict[str, set[str]] = {}
    remote_endpoint: dict[str, bool] = {}
    for m in cfg.models.values():
        by_endpoint.setdefault(m.endpoint, set()).add(m.id)
        # Declaration wins here too (3.1): an endpoint counts as remote when
        # any of its registered models RESOLVES remote, not just when its
        # address looks remote.
        remote_endpoint[m.endpoint] = (remote_endpoint.get(m.endpoint, False)
                                       or m.location == "remote")
    for endpoint, ids in by_endpoint.items():
        served = await list_backend_models(http, endpoint)
        if served is None:
            continue
        served_ids = {m.get("id", "") for m in served}
        extra = served_ids - ids
        missing = ids - served_ids
        # A hosted provider's full catalog is not registry drift; for remote
        # endpoints only a registered model going missing is worth a flag.
        if extra and not remote_endpoint.get(endpoint, False):
            notes.append(f"drift: {endpoint} serves models not in the registry: "
                         f"{sorted(extra)}")
        if missing:
            notes.append(f"drift: registry models not currently served by "
                         f"{endpoint}: {sorted(missing)}")
    return notes


# ------------------------------------------------------------- proposal ----

async def generate_proposal(http: aiohttp.ClientSession, cfg: Config,
                            probe: bool = False) -> str:
    """A registry-and-roles proposal the user reviews and accepts or edits.
    Nothing routes on this until accepted (3.1a). Role assignments are
    heuristic defaults: detection can establish that a model can see images,
    not that it should front the vision cascade."""
    detected = await detect_all(http, cfg, probe=probe)
    lines = [
        "# Proposed by `hello-operator --discover` on "
        + time.strftime("%Y-%m-%d %H:%M:%S"),
        "# Review and edit. Nothing routes on this file until you adopt it as",
        "# your config. Every value is marked with its provenance; your own",
        "# declarations always win over anything detected here.",
        "",
        "models:",
    ]
    chat_candidates: list[tuple[str, str]] = []   # (speed_class, key)
    vision_candidates: list[str] = []
    code_candidates: list[str] = []
    unfiltered_candidates: list[str] = []
    embed_candidates: list[str] = []

    def _pick(field: str, spec, det: dict, probe_label: str = "detected"):
        """Declaration > detected > default — the 3.1 order applies to the
        proposal exactly as it does to routing."""
        declared = spec.provenance.get(field) == "declared"
        if declared:
            return getattr(spec, field), "declared"
        if det.get(field) is not None:
            return det[field], probe_label
        return getattr(spec, field), spec.provenance.get(field, "default")

    for key, spec in cfg.models.items():
        det = detected.get(key, {})
        hints = name_hints(spec.id)
        caps, caps_src = _pick("capabilities", spec, det)
        if isinstance(caps, set):
            caps = sorted(caps)
        ctx, ctx_src = _pick("context_window", spec, det)
        speed, speed_src = _pick("speed_class", spec, det, "detected (probe)")
        residency, _ = _pick("residency", spec, det)

        lines += [
            f"  {key}:",
            f"    id: {spec.id}",
            f"    endpoint: {spec.endpoint}",
            f"    capabilities: {json.dumps(caps)}   # {caps_src}"
            + (f"; name hints: {sorted(hints)}" if hints else ""),
            f"    context_window: {ctx}   # {ctx_src}"
            + (f" via {det['source']}" if det.get("source") else ""),
            f"    residency: {residency}",
            f"    speed_class: {speed}   # {speed_src}",
        ]

        if "embeddings" in set(caps) | hints:
            embed_candidates.append(key)
            continue
        if "vision" in caps:
            vision_candidates.append(key)
        if "unfiltered" in hints:
            unfiltered_candidates.append(key)
        if "code" in hints:
            code_candidates.append(key)
        if "tools" in caps or "text" in caps:
            chat_candidates.append((speed, key))

    speed_order = {"fast": 0, "mid": 1, "slow": 2}
    chat_candidates.sort(key=lambda t: speed_order.get(t[0], 1))
    chat = [k for _, k in chat_candidates]
    biggest = max(cfg.models, key=lambda k: detected.get(k, {}).get(
        "context_window", cfg.models[k].context_window), default=None)

    lines += ["", "roles:   # heuristic proposals — role assignment is a judgment (3.1a)"]
    if chat:
        cascade = chat[:1] + ([biggest] if biggest and biggest != chat[0] else [])
        lines += ["  chat:", f"    cascade: {json.dumps(cascade)}   # fastest first, "
                  "strongest as escalation — reorder to taste"]
    if code_candidates or biggest:
        lines += ["  code:", f"    cascade: {json.dumps(code_candidates or [biggest])}"
                  "   # name-hint / largest-context"]
    if vision_candidates:
        lines += ["  vision:", f"    cascade: {json.dumps(vision_candidates)}   # vision-capable"]
    if unfiltered_candidates:
        lines += ["  unfiltered:", f"    cascade: {json.dumps(unfiltered_candidates)}   # name hint only — verify"]
    if embed_candidates:
        lines += ["", f"embedding:   # candidate: {embed_candidates[0]}",
                  f"  id: {cfg.models[embed_candidates[0]].id}",
                  f"  endpoint: {cfg.models[embed_candidates[0]].endpoint}"]
    lines += ["", "routing:", "  default_role: chat" if chat else "  default_role: code"]
    return "\n".join(lines) + "\n"
