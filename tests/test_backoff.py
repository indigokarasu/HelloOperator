"""Provider-stated waits and the last resort (2026-10-07).

Observed live: with OpenRouter's daily free cap spent on both accounts and Nous
rate-limiting its key, every failing turn re-sent ~30 requests -- the last
resort retried every free model, including ones that had refused moments
earlier in the same turn and ones whose provider had said when to come back.
Nous then answered "too many refused requests for this credential; back off".
"""
import json
import sys
import time
from pathlib import Path

from aiohttp import web

sys.path.insert(0, str(Path(__file__).parent))

from helpers import FakeBackend, RouterEnv, base_config, run  # noqa: E402

from hello_operator import server as server_mod  # noqa: E402

DAILY_CAP = {"error": {"message": "Rate limit exceeded: free-models-per-day-high-balance. ",
                       "code": 429, "metadata": {"headers": {
                           "X-RateLimit-Limit": "1000", "X-RateLimit-Remaining": "0",
                           "X-RateLimit-Reset": None}}}}
KEY_LIMIT = {"status": 429, "message": "Hold up for a bit, you've exceeded the rate "
                                       "limit on your API key."}
FAIR_SHARE = {"status": 429, "message": "You've reached this model's current fair-share "
                                        "rate limit.", "retry_after": 91}


def _daily_cap(reset_in_s=3600):
    body = json.loads(json.dumps(DAILY_CAP))
    reset_ms = int((time.time() + reset_in_s) * 1000)
    body["error"]["metadata"]["headers"]["X-RateLimit-Reset"] = str(reset_ms)
    return body


def _refuse(payload, status=429):
    return lambda body, idx: web.json_response(payload, status=status)


def _ok(text):
    return lambda body, idx: {"content": text}


def _env(tmp_path, ids):
    models = {f"m{i}": {"id": mid, "endpoint": "BACKEND", "capabilities": ["text", "tools"],
                        "context_window": 131072} for i, mid in enumerate(ids)}
    cfg = base_config(tmp_path, models=models,
                      roles={"work": {"cascade": list(models)}},
                      default_role="work", embedding=False)
    return cfg


# ---- reading the wait ------------------------------------------------------

def test_stated_wait_reads_each_provider_shape():
    sw = server_mod._stated_wait
    assert sw({"Retry-After": "30"}, "") == 30
    w = sw({}, json.dumps(_daily_cap(3600)))
    assert 3500 < w <= 3600, w                      # OpenRouter: epoch ms in metadata
    assert sw({}, json.dumps(FAIR_SHARE)) == 91      # Nous: retry_after field
    assert sw({}, '{"error": "too many refused requests for this credential; '
                  'back off before retrying"}') == server_mod._BACKOFF_S
    assert sw({}, json.dumps(KEY_LIMIT)) is None     # no duration given: no hold
    assert sw({}, json.dumps(_daily_cap(-60))) is None   # a reset in the past
    assert sw({"Retry-After": str(10 ** 7)}, "") == server_mod._MAX_WAIT_S


# ---- routing honours it ------------------------------------------------------

def test_a_stated_wait_is_honoured_by_the_last_resort_and_the_next_turn(tmp_path):
    """The daily cap says 'come back in an hour': nothing on that account is
    asked again this turn or the next, so a dead account costs one request."""
    async def scenario():
        backend = FakeBackend({"vendor/a:free": _refuse(_daily_cap()),
                               "vendor/b:free": _ok("b")})
        async with RouterEnv(tmp_path, _env(tmp_path, ["vendor/a:free", "vendor/b:free"]),
                             backend) as env:
            first = await env.chat([{"role": "user", "content": "hi"}], session="s1")
            second = await env.chat([{"role": "user", "content": "hi"}], session="s2")
        return first, second, dict(backend.per_model_calls)

    first, second, calls = run(scenario())
    assert first[0] == 502 and second[0] == 502
    assert calls == {"vendor/a:free": 1}, f"asked a backend that said to wait: {calls}"


def test_last_resort_stops_at_an_account_that_refuses_again(tmp_path):
    """A key-wide 429 with no duration: the breaker skips the account, the last
    resort asks ONE of its models, and when that refuses too it leaves the
    rest of the account alone."""
    ids = ["vendor/a:free", "vendor/b:free", "vendor/c:free"]

    async def scenario():
        backend = FakeBackend({i: _refuse(KEY_LIMIT) for i in ids})
        async with RouterEnv(tmp_path, _env(tmp_path, ids), backend) as env:
            server_mod.mark_free_exhausted(env.backend_base)   # an earlier 429
            out = await env.chat([{"role": "user", "content": "hi"}], session="s1")
        return out, sum(backend.per_model_calls.values())

    (status, payload, _), total = run(scenario())
    assert status == 502, payload
    assert total == 1, f"the last resort kept asking a refusing account: {total} calls"


def test_a_last_resort_that_serves_clears_the_breaker(tmp_path):
    """The breaker's 15 minutes are a guess; an answer proves it wrong, so the
    next turn routes normally instead of failing everything else first."""
    async def scenario():
        backend = FakeBackend({"vendor/a:free": _ok("a")})
        async with RouterEnv(tmp_path, _env(tmp_path, ["vendor/a:free"]), backend) as env:
            server_mod.mark_free_exhausted(env.backend_base)
            first = await env.chat([{"role": "user", "content": "hi"}], session="s1")
            second = await env.chat([{"role": "user", "content": "hi"}], session="s2")
        return first, second

    first, second = run(scenario())
    assert first[0] == 200 and first[1]["router"]["decision"] == "last-resort"
    assert second[0] == 200 and second[1]["router"]["decision"] != "last-resort"
    assert not server_mod._FREE_EXHAUSTED


def test_a_model_scoped_wait_holds_only_that_model(tmp_path):
    """Nous's fair-share 429 names one model and gives retry_after: that model
    waits 91 s, its siblings on the same key keep serving."""
    async def scenario():
        backend = FakeBackend({"vendor/a:free": _refuse(FAIR_SHARE),
                               "vendor/b:free": _ok("b")})
        async with RouterEnv(tmp_path, _env(tmp_path, ["vendor/a:free", "vendor/b:free"]),
                             backend) as env:
            first = await env.chat([{"role": "user", "content": "hi"}], session="s1")
            second = await env.chat([{"role": "user", "content": "hi"}], session="s2")
        return first, second, dict(backend.per_model_calls)

    first, second, calls = run(scenario())
    assert first[0] == 200 and second[0] == 200
    assert calls["vendor/a:free"] == 1, f"held model asked again: {calls}"
    assert calls["vendor/b:free"] == 2
    held = {k: v - time.monotonic() for k, v in server_mod._HOLD.items()}
    assert any(s == "model:vendor/a:free" and 85 < left <= 91 for (_, s), left in held.items()), \
        f"the stated 91 s was not honoured: {held}"
    assert not any(s == "free-quota" for (_, s) in held), "a model-scoped limit held the account"


def test_a_spent_free_quota_does_not_hold_stealth_models():
    """The daily cap is the ':free' quota's; a zero-priced stealth model on the
    same key has its own limits and keeps serving."""
    class Free:
        endpoint, id, key, free = "https://p.example/v1", "vendor/a:free", "a", False
        api_keys, api_key = ["k"], "k"

    class Sibling(Free):
        id, key = "vendor/b:free", "b"

    class Stealth(Free):
        id, key, free = "vendor/stealth", "s", True

    detail = json.dumps(_daily_cap(3600))
    server_mod.set_key_aside(Free, "k", 429, detail, server_mod._stated_wait({}, detail))
    assert server_mod.on_hold(Free, "k") and server_mod.on_hold(Sibling, "k")
    assert not server_mod.on_hold(Stealth, "k"), "a spent free cap held a stealth model"
