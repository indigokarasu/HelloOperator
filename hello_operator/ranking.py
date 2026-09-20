"""Daily free-model ranking (owner directive 2026-09-19).

Free tiers churn: models appear, and a model's free period ends without warning
(``tencent/hy3:free`` started returning 404 "free period has ended" while it sat
at cascade position 2, burning a hop on every request). So the free half of each
cascade is REDISCOVERED and RE-RANKED on a schedule rather than hand-maintained.

Per provider, in the configured provider order:
  1. discover  -- the provider's own catalogue, filtered to free ids that meet
     the context floor and (optionally) advertise tool support
  2. measure   -- a small deterministic probe suite; correctness is checkable
     offline, so a model that answers 200 with empty or wrong content ranks
     below one that works, instead of silently sitting at position 1
  3. rank      -- correctness, then throughput, then context window

Paid entries are never reordered and never dropped: they are the backstop the
owner chose, and they keep their relative order after the free block.

Nothing is adopted unless the rewritten config re-loads cleanly (FR-10): a
failed probe run leaves the previous cascade in place, because a stale ranking
still routes and a broken config does not.
"""
from __future__ import annotations

import json
import logging
import shutil
import statistics
import time
from typing import Any, Optional

import aiohttp
import yaml

log = logging.getLogger("router.ranking")

DEFAULTS = {
    "enabled": True,
    "min_context": 262144,
    "require_tools": True,
    "probe_max_tokens": 300,
    "request_timeout_s": 90,
    "provider_order": [],          # endpoint urls, most preferred first
}

# Deterministic probes: every answer is checkable without a judge model.
PROBES: list[tuple[str, str, Any, int]] = [
    # (name, prompt, checker, max_tokens). The budget is part of the test: a model
    # that spends its whole allowance on reasoning tokens and returns empty content
    # is unusable for ordinary calls, and answers HTTP 200 while doing it, so the
    # cascade would never fail over. Measured 2026-09-19: ling-3.0-flash-fin:free
    # returned '' with finish=length at 32/64/128 tokens and only answered at 300.
    ("terse", "Reply with exactly: ok",
     lambda t: t.strip().lower().strip(".") == "ok", 64),
    ("exact", "Reply with exactly: ok",
     lambda t: t.strip().lower().strip(".") == "ok", 300),
    ("math", "What is 17*23? Reply with only the number.",
     lambda t: "391" in t, 300),
    ("logic", "A bat and ball cost $1.10 together. The bat costs $1.00 more than "
              "the ball. How many cents is the ball? Reply with only the number.",
     lambda t: any(w == "5" for w in t.replace(",", " ").split()), 300),
    ("json", 'Return only this JSON and nothing else: {"a":1}',
     lambda t: '"a"' in t and "1" in t, 300),
]


def settings(raw_cfg: dict) -> dict:
    s = dict(DEFAULTS)
    s.update((raw_cfg or {}).get("ranking") or {})
    return s


def _ctx(entry: dict) -> int:
    return int(entry.get("context_length")
               or (entry.get("top_provider") or {}).get("context_length") or 0)


async def _catalogue(http: aiohttp.ClientSession, endpoint: str, api_key: str) -> list[dict]:
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    try:
        async with http.get(f"{endpoint.rstrip('/')}/models", headers=headers,
                            timeout=aiohttp.ClientTimeout(total=45)) as r:
            if r.status != 200:
                log.warning("ranking: %s /models -> HTTP %s", endpoint, r.status)
                return []
            return (await r.json()).get("data", []) or []
    except Exception as e:  # noqa: BLE001 - a dead provider must not break the run
        log.warning("ranking: %s catalogue unavailable: %s", endpoint, e)
        return []


async def _probe(http: aiohttp.ClientSession, endpoint: str, api_key: str,
                 model_id: str, cfg: dict) -> Optional[tuple[int, float, float]]:
    """(correct_count, median tok/s, median latency) or None when unusable."""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    score, tps, lat = 0, [], []
    for _name, prompt, check, budget in PROBES:
        body = {"model": model_id, "messages": [{"role": "user", "content": prompt}],
                "max_tokens": int(budget or cfg["probe_max_tokens"])}
        t0 = time.monotonic()
        try:
            async with http.post(f"{endpoint.rstrip('/')}/chat/completions", headers=headers,
                                 json=body,
                                 timeout=aiohttp.ClientTimeout(total=cfg["request_timeout_s"])) as r:
                if r.status != 200:
                    log.info("ranking: %s probe -> HTTP %s (dropped)", model_id, r.status)
                    return None
                j = await r.json()
        except Exception as e:  # noqa: BLE001
            log.info("ranking: %s probe failed: %s (dropped)", model_id, e)
            return None
        dt = max(time.monotonic() - t0, 1e-6)
        try:
            text = (j["choices"][0]["message"].get("content") or "")
        except (KeyError, IndexError, TypeError):
            return None
        if check(text):
            score += 1
        completion = int((j.get("usage") or {}).get("completion_tokens") or 0)
        lat.append(dt)
        if completion:
            tps.append(completion / dt)
    return score, (statistics.median(tps) if tps else 0.0), statistics.median(lat)


async def rank(http: aiohttp.ClientSession, raw_cfg: dict, keys: dict) -> dict:
    """Rank the free models of every configured provider. Returns
    {endpoint: [ {id, score, tok_s, latency, context}, ... ]} best first."""
    s = settings(raw_cfg)
    out: dict[str, list[dict]] = {}
    for endpoint in s["provider_order"]:
        api_key = keys.get(endpoint, "")
        rows = []
        for entry in await _catalogue(http, endpoint, api_key):
            mid = str(entry.get("id") or "")
            if not mid.endswith(":free"):
                continue
            if _ctx(entry) < int(s["min_context"]):
                continue
            if s["require_tools"] and "tools" not in (entry.get("supported_parameters") or []):
                # Nous does not publish supported_parameters; probe decides instead.
                if entry.get("supported_parameters") is not None:
                    continue
            measured = await _probe(http, endpoint, api_key, mid, s)
            if measured is None:
                continue
            score, tok_s, latency = measured
            rows.append({"id": mid, "score": score, "tok_s": round(tok_s, 1),
                         "latency": round(latency, 2), "context": _ctx(entry)})
        rows.sort(key=lambda r: (-r["score"], -r["tok_s"], -r["context"]))
        out[endpoint] = rows
        log.info("ranking: %s -> %d usable free model(s)", endpoint, len(rows))
    return out


def _key_for(endpoint: str, model_id: str, taken: set[str]) -> str:
    host = endpoint.split("//")[-1].split(".")[0][:8]
    slug = model_id.split("/")[-1].replace(":free", "").replace(".", "-")[:28]
    base = f"{host}-{slug}".strip("-")
    name, n = base, 2
    while name in taken:
        name, n = f"{base}-{n}", n + 1
    taken.add(name)
    return name


def apply_ranking(config_path: str, ranked: dict, raw_cfg: dict) -> tuple[bool, str]:
    """Rewrite the free half of every non-vision cascade. Returns (changed, note).

    Paid entries keep their order and follow the free block. The file is only
    replaced when the result parses and every cascade entry resolves to a model.
    """
    cfg = yaml.safe_load(open(config_path)) or {}
    models: dict = dict(cfg.get("models") or {})
    roles: dict = dict(cfg.get("roles") or {})
    # Keyed by (id, endpoint), never by id alone: the same model is offered by
    # more than one provider (inclusionai/ling-3.0-flash-fin:free is on both Nous
    # and OpenRouter), and an id-only map collapses those into one entry, which
    # would re-point a provider's row at the other provider's model on the next run.
    by_id_ep = {(v.get("id"), v.get("endpoint")): k
                for k, v in models.items() if isinstance(v, dict)}

    # model-first ordering (owner directive 2026-09-19): rank MODELS globally by
    # measured quality, then list each model's providers consecutively, best
    # provider first. A model's second provider is therefore one hop away rather
    # than a whole provider block away -- which is what makes the free-tier
    # breaker useful, since exhaustion is usually provider-wide.
    order = settings(raw_cfg)["provider_order"]
    variants: dict[str, list[tuple[str, dict]]] = {}
    for endpoint in order:
        for row in ranked.get(endpoint, []):
            variants.setdefault(row["id"], []).append((endpoint, row))

    def _best(vs):
        return max((v[1]["score"], v[1]["tok_s"], v[1]["context"]) for v in vs)

    free_keys: list[str] = []
    taken = set(models)
    for model_id, vs in sorted(variants.items(), key=lambda kv: _best(kv[1]), reverse=True):
        vs.sort(key=lambda ev: (-ev[1]["score"], -ev[1]["tok_s"],
                                order.index(ev[0]) if ev[0] in order else 99))
        for endpoint, row in vs:
            key = by_id_ep.get((model_id, endpoint))
            if key is None:
                key = _key_for(endpoint, model_id, taken)
                template = next((v for k, v in models.items()
                                 if isinstance(v, dict) and v.get("endpoint") == endpoint), {})
                models[key] = {"id": model_id, "endpoint": endpoint,
                               "capabilities": ["text", "tools"],
                               "context_window": row["context"]}
                if template.get("api_key"):
                    models[key]["api_key"] = template["api_key"]
            else:
                models[key]["context_window"] = row["context"] or models[key].get("context_window")
            free_keys.append(key)

    if not free_keys:
        # Every probe failing (bad/absent credentials, provider outage) must not be
        # mistaken for "the free tier is empty": applying that would rewrite every
        # cascade down to the paid tail and start spending money silently.
        return False, "refused: no usable free models discovered (previous cascade kept)"

    ranked_ids = {row["id"] for rows in ranked.values() for row in rows}
    changed = False
    for role, spec in roles.items():
        cascade = list((spec or {}).get("cascade") or [])
        if not cascade or role == "vision":
            continue
        paid_tail = [k for k in cascade
                     if k in models and models[k].get("id") not in ranked_ids
                     and not str(models[k].get("id", "")).endswith(":free")]
        new_cascade = free_keys + paid_tail
        if new_cascade != cascade:
            changed = True
        roles[role] = {**(spec or {}), "cascade": new_cascade}

    if not changed:
        return False, "cascade already optimal"
    cfg["models"], cfg["roles"] = models, roles
    for role, spec in roles.items():
        for key in spec.get("cascade", []):
            if key not in models:
                return False, f"refused: cascade entry '{key}' has no model definition"
    backup = f"{config_path}.bak.rank-{time.strftime('%Y%m%d-%H%M%S')}"
    shutil.copy2(config_path, backup)
    tmp = f"{config_path}.tmp.rank"
    with open(tmp, "w") as f:
        f.write("# Free block re-ranked by hello-operator --rank-free (daily timer).\n"
                "# Ordering: correctness, then throughput, then context; paid tail untouched.\n")
        yaml.safe_dump(cfg, f, sort_keys=False, width=100)
    try:
        yaml.safe_load(open(tmp))
    except Exception as e:  # noqa: BLE001
        return False, f"refused: rewritten config does not parse ({e})"
    import os
    os.replace(tmp, config_path)
    return True, f"applied; backup {backup}"


def format_report(ranked: dict) -> str:
    lines = []
    for endpoint, rows in ranked.items():
        lines.append(f"{endpoint}  ({len(rows)} usable free)")
        for i, r in enumerate(rows, 1):
            lines.append(f"  {i:>2}. {r['id']:<46} score={r['score']}/{len(PROBES)} "
                         f"{r['tok_s']:>7.1f} tok/s  p50={r['latency']:>5.2f}s  ctx={r['context']:,}")
    return "\n".join(lines) or "no free models discovered"
