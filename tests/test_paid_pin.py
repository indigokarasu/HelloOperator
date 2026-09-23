"""Regression: a turn served by the PAID tail must not become the session's
sticky affinity position.

Affinity holds a session wherever it last landed, and the idle timeout only
counts *idle* time -- so one escalation (a backend hiccup, one oversized
request) used to pin every later turn of an active session onto the paid
model. Measured live on 2026-09-22: 787 of 860 paid calls were affinity
reuses, and they drained the daily budget by mid-morning, after which the
budget refused real work and cron jobs failed with HTTP 502.

The contract: after a paid turn, the session re-tries the top of its cascade.
If the free block still cannot serve it, the next turn escalates again on its
own -- correct, just not silent and not sticky.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from aiohttp import web

from helpers import CHAT_UTTERANCES, FakeBackend, RouterEnv, base_config, run


def _fail(body, idx):
    """A free head that answers, badly: any backend error triggers failover."""
    return web.json_response({"error": {"message": "free head is down"}}, status=500)


def _echo(name):
    return lambda body, idx: {"content": f"answer from {name}"}


def _cfg(tmp_path):
    return base_config(
        tmp_path,
        models={
            "free": {"id": "free-model:free", "endpoint": "BACKEND",
                     "capabilities": ["text", "tools", "json"],
                     "context_window": 32768, "speed_class": "fast"},
            "paid": {"id": "paid-model", "endpoint": "BACKEND",
                     "capabilities": ["text", "tools", "json"],
                     "context_window": 32768, "speed_class": "slow",
                     "price_in": 0.15, "price_out": 0.6},
        },
        roles={"chat": {"cascade": ["free", "paid"], "utterances": CHAT_UTTERANCES}},
        default_role="chat")


def test_paid_turn_does_not_pin_the_session(tmp_path):
    async def scenario():
        backend = FakeBackend({"paid-model": _echo("paid"), "free-model:free": _fail})
        async with RouterEnv(tmp_path, _cfg(tmp_path), backend) as env:
            msgs = [{"role": "user", "content": "hello let's chat"}]

            _, _, h1 = await env.chat(msgs, session="paidpin")
            assert h1["x-router-backend-model"] == "paid-model", h1
            free_calls_after_1 = backend.per_model_calls.get("free-model:free", 0)
            assert free_calls_after_1 >= 1, "turn 1 never tried the free head"

            _, _, h2 = await env.chat(msgs, session="paidpin")
            assert h2["x-router-backend-model"] == "paid-model", h2
            # The fix's observable: turn 2 walks the cascade again from the head.
            # Without it, turn 2 is an affinity hit straight on the paid model
            # and the free head is never contacted again.
            assert backend.per_model_calls.get("free-model:free", 0) > free_calls_after_1, \
                "turn 2 skipped the free head -- the paid pick became sticky"
            assert h2["x-router-decision"] != "affinity", \
                f"paid model served by affinity: {h2['x-router-decision']}"

    run(scenario())
