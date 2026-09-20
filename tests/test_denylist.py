"""Operator denylist: a banned model is never routed to, on any provider.

The ban must also survive the nightly re-ranking, or it lasts exactly one day.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from helpers import CHAT_UTTERANCES, CODE_UTTERANCES, FakeBackend, RouterEnv, base_config, run

from hello_operator.server import _denied


class _Spec:
    def __init__(self, id_, key=""):
        self.id = id_
        self.key = key


def _echo(name):
    return lambda body, idx: {"content": f"answer from {name}"}


def _two_model_cfg(tmp_path, **router_extra):
    models = {
        "fast": {"id": "vendor/fast-model:free", "endpoint": "BACKEND",
                 "capabilities": ["text", "tools", "json"], "context_window": 32768},
        "big": {"id": "vendor/big-model:free", "endpoint": "BACKEND",
                "capabilities": ["text", "tools", "json"], "context_window": 131072},
    }
    roles = {"chat": {"cascade": ["fast", "big"], "utterances": CHAT_UTTERANCES},
             "code": {"cascade": ["big"], "utterances": CODE_UTTERANCES}}
    return base_config(tmp_path, models=models, roles=roles,
                       default_role="chat", **router_extra)


# ----------------------------------------------------------------- the matcher

def test_denied_matches_substring_case_insensitively():
    spec = _Spec("inclusionai/ling-3.0-flash-sante:free", "openrout-ling-sante")
    assert _denied(spec, ["ling-3.0-flash-sante"]) is True
    assert _denied(spec, ["LING-3.0-FLASH-SANTE"]) is True
    assert _denied(spec, ["ling-3.0-flash-fin"]) is False


def test_denied_is_false_for_empty_list():
    assert _denied(_Spec("anything"), []) is False
    assert _denied(_Spec("anything"), None) is False


def test_denied_also_matches_the_registry_key():
    """One model served by two providers has two keys but one id; either may be named."""
    assert _denied(_Spec("vendor/m:free", "openrout-m"), ["openrout-m"]) is True


# ---------------------------------------------------------------- the routing

def test_denylisted_head_is_skipped_and_the_next_model_serves(tmp_path):
    async def scenario():
        backend = FakeBackend({"vendor/fast-model:free": _echo("fast"),
                               "vendor/big-model:free": _echo("big")})
        cfg = _two_model_cfg(tmp_path, denylist=["fast-model"])
        async with RouterEnv(tmp_path, cfg, backend) as env:
            return await env.chat([{"role": "user", "content": "hello chat"}])
    status, payload, _ = run(scenario())
    assert status == 200
    # the cascade head is banned, so the second model answers
    assert payload["router"]["backend_model"] == "vendor/big-model:free"


def test_without_the_denylist_the_head_still_serves(tmp_path):
    """Control: proves the test above is caused by the denylist and nothing else."""
    async def scenario():
        backend = FakeBackend({"vendor/fast-model:free": _echo("fast"),
                               "vendor/big-model:free": _echo("big")})
        async with RouterEnv(tmp_path, _two_model_cfg(tmp_path), backend) as env:
            return await env.chat([{"role": "user", "content": "hello chat"}])
    status, payload, _ = run(scenario())
    assert status == 200
    assert payload["router"]["backend_model"] == "vendor/fast-model:free"


def test_denying_every_model_fails_loudly_rather_than_serving_one(tmp_path):
    """A ban must never be silently ignored because it emptied the cascade."""
    async def scenario():
        backend = FakeBackend({"vendor/fast-model:free": _echo("fast"),
                               "vendor/big-model:free": _echo("big")})
        cfg = _two_model_cfg(tmp_path, denylist=["vendor/"])
        async with RouterEnv(tmp_path, cfg, backend) as env:
            return await env.chat([{"role": "user", "content": "hello chat"}])
    status, payload, _ = run(scenario())
    assert status >= 500
    assert "denylist" in str(payload).lower() or "error" in str(payload).lower()
