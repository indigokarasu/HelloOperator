"""router.key_pools: several accounts' keys for one provider (owner directive 2026-09-25).

The keys live in the environment; the config names them. Each call starts on the
next key, and a refused key is set aside while the SAME model is retried on the
next account before the cascade moves on.
"""
import json
import sys
from pathlib import Path

import yaml
from aiohttp import web

sys.path.insert(0, str(Path(__file__).parent))

from helpers import CHAT_UTTERANCES, FakeBackend, RouterEnv, base_config, run  # noqa: E402

from hello_operator import config as config_mod  # noqa: E402
from hello_operator import server as server_mod  # noqa: E402


class KeyedBackend(FakeBackend):
    """Answers each Authorization header with a scripted status (default 200)."""

    def __init__(self, behaviors, refuse=None):
        super().__init__(behaviors)
        self.refuse = dict(refuse or {})
        self.auths: list[str] = []

    async def chat(self, request):
        auth = request.headers.get("Authorization", "").removeprefix("Bearer ")
        self.auths.append(auth)
        if auth in self.refuse:
            return web.json_response({"error": {"message": "refused"}},
                                     status=self.refuse[auth])
        return await super().chat(request)


def _reply(text):
    return lambda body, idx: {"content": text}


def _cfg(tmp_path, monkeypatch, model_id="vendor/stealth", free=True, **extra):
    monkeypatch.setenv("TEST_KEY_A", "key-a")
    monkeypatch.setenv("TEST_KEY_B", "key-b")
    models = {"m": {"id": model_id, "endpoint": "BACKEND", "free": free,
                    "capabilities": ["text", "tools"], "context_window": 131072}}
    roles = {"chat": {"cascade": ["m"], "utterances": CHAT_UTTERANCES}}
    return base_config(tmp_path, models=models, roles=roles, default_role="chat",
                       key_pools={"BACKEND": ["${TEST_KEY_A}", "${TEST_KEY_B}"]}, **extra)


def _reset():
    server_mod._KEY_NEXT.clear()
    server_mod._KEY_COOLDOWN.clear()
    server_mod._FREE_EXHAUSTED.clear()


def test_pool_resolves_from_env_and_drops_unset(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_KEY_A", "key-a")
    monkeypatch.delenv("TEST_KEY_UNSET", raising=False)
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump({
        "router": {"allow_remote_endpoints": True,
                   "key_pools": {"https://p.example/v1/": ["${TEST_KEY_A}", "${TEST_KEY_UNSET}"]}},
        "models": {"m": {"id": "x", "endpoint": "https://p.example/v1", "api_key": "own"},
                   "other": {"id": "y", "endpoint": "https://q.example/v1", "api_key": "solo"}},
        "roles": {"chat": {"cascade": ["m", "other"]}},
        "routing": {"default_role": "chat"}}))
    cfg = config_mod.load(str(p))
    assert cfg.models["m"].api_keys == ["key-a"] and cfg.models["m"].api_key == "key-a"
    assert cfg.models["other"].api_keys == ["solo"]
    assert any("TEST_KEY_UNSET" in w for w in cfg.warnings)
    assert not any("key-a" in w for w in cfg.warnings), "a key value leaked into a warning"


def test_literal_key_in_pool_is_flagged(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump({
        "router": {"allow_remote_endpoints": True,
                   "key_pools": {"https://p.example/v1": ["sk-literal"]}},
        "models": {"m": {"id": "x", "endpoint": "https://p.example/v1"}}}))
    cfg = config_mod.load(str(p))
    assert any("literal key" in w for w in cfg.warnings)
    assert not any("sk-literal" in w for w in cfg.warnings)


def test_calls_rotate_across_keys(tmp_path, monkeypatch):
    _reset()
    cfg = _cfg(tmp_path, monkeypatch)

    async def scenario():
        backend = KeyedBackend({"vendor/stealth": _reply("ok")})
        async with RouterEnv(tmp_path, cfg, backend) as env:
            for n in range(4):
                status, _, _ = await env.chat([{"role": "user", "content": "hello chat"}],
                                              session=f"s{n}")
                assert status == 200
        return backend.auths

    auths = run(scenario())
    assert auths == ["key-a", "key-b", "key-a", "key-b"], auths


def test_refused_key_retries_same_model_on_next_key_then_is_set_aside(tmp_path, monkeypatch):
    """Account A is capped: the turn is served by the same model on account B,
    and later calls skip A instead of spending a round-trip on it."""
    _reset()
    cfg = _cfg(tmp_path, monkeypatch)

    async def scenario():
        backend = KeyedBackend({"vendor/stealth": _reply("served")}, refuse={"key-a": 429})
        async with RouterEnv(tmp_path, cfg, backend) as env:
            first = await env.chat([{"role": "user", "content": "hello chat"}], session="s1")
            second = await env.chat([{"role": "user", "content": "hello chat"}], session="s2")
            third = await env.chat([{"role": "user", "content": "hello chat"}], session="s3")
            async with env.client.get(f"{env.base}/v1/status") as r:
                status = await r.json()
        return backend.auths, first, second, third, status

    auths, first, second, third, status = run(scenario())
    assert first[0] == 200 and first[1]["choices"][0]["message"]["content"] == "served"
    assert second[0] == 200 and third[0] == 200
    assert auths == ["key-a", "key-b", "key-b", "key-b"], auths
    assert status["last_any"]["key_slot"] == 2
    log = [json.loads(l) for l in (tmp_path / "decisions.jsonl").read_text().splitlines()]
    assert log[0].get("key_slot") == 2
    assert "key-a" not in json.dumps(status) and "key-b" not in json.dumps(status)


def test_every_key_refused_falls_through_to_the_last_resort(tmp_path, monkeypatch):
    _reset()
    cfg = _cfg(tmp_path, monkeypatch)

    backend_auths = []

    async def scenario2():
        backend = KeyedBackend({"vendor/stealth": _reply("x")},
                               refuse={"key-a": 429, "key-b": 429})
        async with RouterEnv(tmp_path, cfg, backend) as env:
            out = await env.chat([{"role": "user", "content": "hello chat"}], session="s")
        backend_auths.extend(backend.auths)
        return out

    status, payload, _ = run(scenario2())
    assert status == 502, payload
    # both accounts in rotation, then both again as the last resort
    assert sorted(backend_auths) == ["key-a", "key-a", "key-b", "key-b"], backend_auths


def test_paid_402_sets_aside_only_paid_calls_on_that_key():
    _reset()

    class S:
        endpoint, id, key, free = "https://p.example/v1", "vendor/paid", "p", False
        api_keys = ["key-a", "key-b"]
        api_key = "key-a"

    server_mod.set_key_aside(S, "key-a", 402)
    assert server_mod.key_cooling(S, "key-a") and not server_mod.key_cooling(S, "key-b")

    class F(S):
        id, key, free = "vendor/stealth", "f", True

    assert not server_mod.key_cooling(F, "key-a"), "a spent paid balance must not block free models"
    _reset()


def test_single_key_models_keep_their_old_behaviour():
    """No pool, no rotation state: a 429 on a stealth model does not set it aside."""
    _reset()

    class S:
        endpoint, id, key, free = "https://p.example/v1", "vendor/stealth", "s", True
        api_keys = ["only"]
        api_key = "only"

    server_mod.set_key_aside(S, "only", 429)
    assert not server_mod.key_cooling(S, "only")
    assert server_mod.keys_in_turn(S) == ["only"]
    _reset()


def test_ranking_probes_on_the_next_key_when_one_account_cannot(monkeypatch):
    """A model account A cannot reach today is probed on account B before it is
    written off as unprobeable."""
    import asyncio

    from hello_operator import ranking

    EP = "https://p.example/v1"
    seen = []

    async def fake_catalogue(http, endpoint, key):
        return [{"id": "vendor/stealth", "context_length": 262144,
                 "supported_parameters": ["tools"],
                 "pricing": {"prompt": "0", "completion": "0"}}]

    async def fake_probe(http, endpoint, key, mid, s):
        seen.append(key)
        return None if key == "key-a" else (5, 10.0, 1.0)

    async def fake_tools(http, endpoint, key, mid, s):
        return 2

    monkeypatch.setattr(ranking, "_catalogue", fake_catalogue)
    monkeypatch.setattr(ranking, "_probe", fake_probe)
    monkeypatch.setattr(ranking, "_probe_tools", fake_tools)
    monkeypatch.delenv("JEVER_API_KEY", raising=False)
    out = asyncio.run(ranking.rank(None, {"ranking": {"provider_order": [EP]}},
                                   {EP: ["key-a", "key-b"]}))
    assert [r["id"] for r in out[EP]] == ["vendor/stealth"] and seen == ["key-a", "key-b"]
