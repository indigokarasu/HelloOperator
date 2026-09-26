"""JEV judgments over provider catalogue entries (owner directive 2026-09-25).

The daily ranking used to decide "free" and "usable" with rules written against
one provider's /models shape: a ':free' id suffix, then $0 pricing fields. Every
new shape needed another rule -- stealth models priced "0" with no suffix,
"-1" for router pricing, per-image or per-second prices, an expiry date on a
free preview. Instead each catalogue entry's attributes go to TypeSafe's JEV
(POST /v1/systemone) with yes/no (``noul``) questions:

  free      -- does this model cost nothing to use?
  meets     -- does it meet the configured requirement (by default: text or
               multimodal in, text or multimodal out)?
  image_in  -- does it accept images? (sets ``vision`` on a newly added model)

JEV answers each with a probability of yes. Verdicts are cached on a hash of the
attributes sent, so the daily run only asks about entries that are new or
changed. When JEV cannot answer (no key, network, rate limit) the entry is
judged by the old rules instead, and the run log says how many were.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

import aiohttp

log = logging.getLogger("router.jev")

DEFAULT_REQUIREMENT = (
    "The model accepts text input, alone or together with images, audio, video or "
    "files, and produces text output, alone or together with other media. It is a "
    "chat or instruction model, not an embedding, image-generation-only, "
    "speech-only or moderation-only model.")

DEFAULTS = {
    "enabled": True,
    "endpoint": "https://api.typesafe.ai/v1/systemone",
    "model": "jev-latest",
    "api_key": "${JEVER_API_KEY}",
    "threshold": 0.5,
    "concurrency": 6,
    "timeout_s": 30,
    "max_failures": 5,        # consecutive failures before the rest fall back to rules
    "cache_days": 14,
    "requirement": DEFAULT_REQUIREMENT,
}

# Bump when the questions change so cached verdicts are asked again.
QUESTIONS_VERSION = 1

ATTRIBUTE_KEYS = ("id", "name", "description", "pricing", "architecture", "context_length",
                  "supported_parameters", "top_provider", "per_request_limits",
                  "expiration_date")
PRICE_KEYS = ("prompt", "completion", "request", "image", "audio", "web_search",
              "internal_reasoning", "input_cache_read", "input_cache_write")


def settings(raw_cfg: dict) -> dict:
    s = dict(DEFAULTS)
    s.update(((raw_cfg or {}).get("ranking") or {}).get("jev") or {})
    key = os.path.expandvars(str(s.get("api_key") or ""))
    s["api_key"] = "" if key.startswith("$") else key
    s["enabled"] = bool(s["enabled"]) and bool(s["api_key"])
    return s


def attributes(entry: dict) -> dict:
    """The catalogue fields JEV judges from. Stable fields only, so the cache
    key changes when the model's terms change and not otherwise."""
    attrs = {k: entry[k] for k in ATTRIBUTE_KEYS if entry.get(k) not in (None, "", [], {})}
    if isinstance(attrs.get("description"), str):
        attrs["description"] = attrs["description"][:600]
    return attrs


def questions(requirement: str) -> dict:
    return {
        "free": {"type": "noul",
                 "instructions": "Is this model free to use: it costs nothing per request "
                                 "and per token?",
                 "criteria": {"true": "free: zero price, or explicitly offered free of charge",
                              "false": "paid: any non-zero price per token, request, image "
                                       "or second, or variable pricing"}},
        "meets": {"type": "noul",
                  "instructions": "Does this model meet the requirement? " + requirement,
                  "criteria": {"true": "meets the requirement",
                               "false": "does not meet it"}},
        "image_in": {"type": "noul",
                     "instructions": "Does this model accept image input?",
                     "criteria": {"true": "accepts images",
                                  "false": "text or other media only"}},
    }


def fingerprint(attrs: dict, s: dict) -> str:
    blob = json.dumps([QUESTIONS_VERSION, s["model"], s["requirement"], attrs],
                      sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(blob.encode()).hexdigest()


def listed_price(entry: dict) -> Optional[float]:
    """Largest non-zero price the catalogue lists, 0.0 when every listed price is
    zero, None when it lists none. '-1' (variable, router-chosen) counts as priced."""
    pricing = entry.get("pricing")
    if not isinstance(pricing, dict):
        return None
    seen, worst = False, 0.0
    for k in PRICE_KEYS:
        try:
            v = float(pricing[k])
        except (KeyError, TypeError, ValueError):
            continue
        seen = True
        if v != 0:
            worst = max(worst, abs(v))
    return worst if seen else None


def meets_by_rules(entry: dict) -> bool:
    """Fallback for 'meets': text among the inputs and among the outputs. An
    entry that publishes no modalities is assumed to be a text model."""
    arch = entry.get("architecture") or {}
    ins, outs = arch.get("input_modalities"), arch.get("output_modalities")
    return (not ins or "text" in ins) and (not outs or "text" in outs)


def decide(entry: dict, verdict: Optional[dict], s: dict, free_by_rules: bool) -> dict:
    """{free, meets, vision, by, note}. JEV decides when it answered; the rules
    decide when it did not. One override: a model the catalogue lists with a
    non-zero price is never treated as free, whatever JEV says, because the
    router would then send it free-tier traffic and bill for it."""
    if verdict is None:
        return {"free": free_by_rules, "meets": meets_by_rules(entry), "vision": None,
                "by": "rules", "note": ""}
    t = float(s["threshold"])
    free = verdict.get("free", 0.0) >= t
    note = ""
    price = listed_price(entry)
    if free and price:
        free = False
        note = f"JEV said free ({verdict.get('free')}) but a price of {price} is listed; kept as paid"
    elif not free and free_by_rules:
        note = f"JEV said paid ({verdict.get('free')}) although the catalogue lists it at $0"
    return {"free": free, "meets": verdict.get("meets", 0.0) >= t,
            "vision": verdict.get("image_in", 0.0) >= t, "by": "jev", "note": note}


def cache_load(path: Optional[str]) -> dict:
    if not path:
        return {}
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {}


def cache_save(path: Optional[str], cache: dict, s: dict) -> None:
    if not path:
        return
    horizon = time.time() - float(s["cache_days"]) * 86400
    kept = {k: v for k, v in cache.items() if float(v.get("ts", 0)) >= horizon}
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        tmp = f"{path}.tmp"
        Path(tmp).write_text(json.dumps(kept, indent=1))
        os.replace(tmp, path)
    except OSError as e:
        log.warning("jev: could not save verdict cache %s: %s", path, e)


async def ask(http: aiohttp.ClientSession, s: dict, attrs: dict) -> Optional[dict]:
    """One JEV call for one catalogue entry: {question: probability} or None."""
    body = {"state": {"model": attrs}, "model": s["model"],
            "questions": questions(s["requirement"])}
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {s['api_key']}"}
    try:
        async with http.post(s["endpoint"], json=body, headers=headers,
                             timeout=aiohttp.ClientTimeout(total=float(s["timeout_s"]))) as r:
            if r.status != 200:
                log.info("jev: %s -> HTTP %s", attrs.get("id"), r.status)
                return None
            data = await r.json(content_type=None)
    except Exception as e:  # noqa: BLE001 - JEV down must not stop the ranking
        log.info("jev: %s failed: %s", attrs.get("id"), e)
        return None
    answers = (data or {}).get("answers") or {}
    out = {}
    for q in ("free", "meets", "image_in"):
        try:
            out[q] = float((answers.get(q) or {})["noul"])
        except (KeyError, TypeError, ValueError):
            return None
    return out


async def judge(http: aiohttp.ClientSession, entries: list[dict], s: dict,
                cache: dict) -> tuple[dict[str, dict], dict]:
    """Verdicts for every entry JEV (or the cache) could answer, keyed by model id,
    plus run counts {asked, cached, failed, skipped}. Mutates ``cache``."""
    verdicts: dict[str, dict] = {}
    counts = {"asked": 0, "cached": 0, "failed": 0, "skipped": 0}
    todo = []
    for entry in entries:
        attrs = attributes(entry)
        fp = fingerprint(attrs, s)
        hit = cache.get(fp)
        if hit and isinstance(hit.get("verdict"), dict):
            verdicts[str(entry.get("id"))] = hit["verdict"]
            hit["ts"] = time.time()
            counts["cached"] += 1
        else:
            todo.append((entry, attrs, fp))

    sem = asyncio.Semaphore(max(1, int(s["concurrency"])))
    streak = {"n": 0}

    async def one(entry, attrs, fp):
        async with sem:
            # A dead key or an outage fails every call the same way; stop asking
            # rather than spend the whole run timing out, and let the rules judge.
            if streak["n"] >= int(s["max_failures"]):
                counts["skipped"] += 1
                return
            v = await ask(http, s, attrs)
        if v is None:
            streak["n"] += 1
            counts["failed"] += 1
            return
        streak["n"] = 0
        counts["asked"] += 1
        verdicts[str(entry.get("id"))] = v
        cache[fp] = {"ts": time.time(), "id": entry.get("id"), "verdict": v}

    await asyncio.gather(*(one(*t) for t in todo))
    return verdicts, counts
