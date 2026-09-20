"""/v1/status: report the backend that ACTUALLY served, to the RIGHT caller.

The display surfaces render the model from config, so they always read
"hello-operator" while the router is active. These tests pin the endpoint that
makes the real backend visible -- and that it answers about the asking session,
not whichever cron turn happened to finish last.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from helpers import CHAT_UTTERANCES, FakeBackend, RouterEnv, base_config, run

from hello_operator.server import _provider_name

SID = "x-session-id"


class _Spec:
    def __init__(self, location=""):
        self.location = location


def _echo(name):
    return lambda body, idx: {"content": f"answer from {name}"}


def _cfg(tmp_path, **extra):
    """TWO models on purpose: a single-model registry makes the router degenerate
    (passthrough, no session state), which would make these tests vacuous."""
    models = {"fast": {"id": "vendor/fast-model:free", "endpoint": "BACKEND",
                       "capabilities": ["text", "tools", "json"], "context_window": 32768},
              "big": {"id": "vendor/big-model:free", "endpoint": "BACKEND",
                      "capabilities": ["text", "tools", "json"], "context_window": 131072}}
    roles = {"chat": {"cascade": ["fast", "big"], "utterances": CHAT_UTTERANCES}}
    return base_config(tmp_path, models=models, roles=roles, default_role="chat", **extra)


def _backend():
    return FakeBackend({"vendor/fast-model:free": _echo("fast"),
                        "vendor/big-model:free": _echo("big")})


async def _get(env, headers=None):
    async with env.client.get(f"{env.base}/v1/status", headers=headers or {}) as r:
        assert r.status == 200
        return await r.json()


# ------------------------------------------------------------ provider names

def test_provider_name_maps_known_hosts():
    assert _provider_name("https://openrouter.ai/api/v1") == "openrouter"
    assert _provider_name("https://inference-api.nousresearch.com/v1") == "nousresearch"
    assert _provider_name("https://model.inferx.net/endpoints/v1") == "inferx"


def test_provider_name_handles_local_and_junk():
    assert _provider_name("http://127.0.0.1:8080/v1") == "local"
    assert _provider_name("http://localhost:8800/v1") == "local"
    assert _provider_name("") == "unknown"
    assert _provider_name("not a url") == "unknown"


def test_provider_name_never_reports_an_ip_octet_as_a_provider():
    """A LAN backend used to render as the provider '1'."""
    assert _provider_name("http://192.168.1.50:8000/v1") == "192.168.1.50"
    assert _provider_name("http://10.0.0.5:11434/v1") == "10.0.0.5"


def test_provider_name_trusts_a_declared_local_location():
    assert _provider_name("http://192.168.1.50:8000/v1", _Spec("local")) == "local"


def test_provider_name_falls_back_to_second_level_domain():
    assert _provider_name("https://api.example.com/v1") == "example"


# ------------------------------------------------------------------ endpoint

def test_status_before_any_traffic(tmp_path):
    async def scenario():
        backend = _backend()
        async with RouterEnv(tmp_path, _cfg(tmp_path), backend) as env:
            return await _get(env)
    body = run(scenario())
    assert body["status"] == "ok"
    assert body["active"] is None
    assert body["last_any"] is None
    assert body["display"] == body["router"]
    assert body["cascade_head"]["model"] == "vendor/fast-model:free"


def test_status_reports_the_backend_that_served_this_session(tmp_path):
    async def scenario():
        backend = _backend()
        async with RouterEnv(tmp_path, _cfg(tmp_path), backend) as env:
            status, payload, _ = await env.chat(
                [{"role": "user", "content": "hello chat"}], session="s1")
            assert status == 200
            return payload, await _get(env, {SID: "s1"})
    payload, body = run(scenario())
    active = body["active"]
    assert active is not None
    assert active["model"] == "vendor/fast-model:free"
    assert active["age_s"] >= 0
    assert body["display"] == "HelloOperator/local/vendor/fast-model:free"
    # status and the response's own router block must never disagree
    assert payload["router"]["backend_model"] == active["model"]
    assert active["free"] is True


def test_status_does_not_answer_with_another_sessions_model(tmp_path):
    """The defect this endpoint must not have: session A asking, and being told
    about the cron job (session B) that merely finished most recently."""
    async def scenario():
        backend = _backend()
        async with RouterEnv(tmp_path, _cfg(tmp_path), backend) as env:
            await env.chat([{"role": "user", "content": "hello chat"}], session="cron-b")
            return await _get(env, {SID: "user-a"})
    body = run(scenario())
    assert body["active"] is None, "session A was told about session B's turn"
    assert body["last_any"]["model"] == "vendor/fast-model:free"
    # the process-wide value is still available, but honestly named
    assert body["last_any"] is not None
    assert body["last_any"]["model"] == "vendor/fast-model:free"
    assert body["display"] == body["router"]


def test_unidentified_caller_gets_last_any_but_no_active(tmp_path):
    async def scenario():
        backend = _backend()
        async with RouterEnv(tmp_path, _cfg(tmp_path), backend) as env:
            await env.chat([{"role": "user", "content": "hello chat"}], session="s1")
            return await _get(env)
    body = run(scenario())
    assert body["active"] is None
    assert body["last_any"]["model"] == "vendor/fast-model:free"


def test_each_session_gets_its_own_answer(tmp_path):
    """Two sessions on two different models: each must be told its own."""
    async def scenario():
        models = {
            "fast": {"id": "vendor/fast-model:free", "endpoint": "BACKEND",
                     "capabilities": ["text", "tools", "json"], "context_window": 32768},
            "big": {"id": "vendor/big-model:free", "endpoint": "BACKEND",
                    "capabilities": ["text", "tools", "json"], "context_window": 131072},
        }
        roles = {"chat": {"cascade": ["fast", "big"], "utterances": CHAT_UTTERANCES}}
        cfg = base_config(tmp_path, models=models, roles=roles, default_role="chat")
        backend = FakeBackend({"vendor/fast-model:free": _echo("fast"),
                               "vendor/big-model:free": _echo("big")})
        async with RouterEnv(tmp_path, cfg, backend) as env:
            await env.chat([{"role": "user", "content": "hello chat"}], session="s1")
            # pin s2 to the other model by naming it directly (FR-13)
            await env.chat([{"role": "user", "content": "hello chat"}],
                           session="s2", model="big")
            return await _get(env, {SID: "s1"}), await _get(env, {SID: "s2"})
    a, b = run(scenario())
    assert a["active"]["model"] == "vendor/fast-model:free"
    assert b["active"]["model"] == "vendor/big-model:free"
