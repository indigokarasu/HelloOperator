"""A rotated credential is applied without restarting the router.

The Nous token lives 60 minutes and reached the router only through a restart.
Restarts cut live turns, so hello-operator-env-sync deferred and then skipped
them whenever turns were in flight, which on a busy router is nearly always:
the running process kept an expired token and every Nous model answered 401
(112 of them in one hour on 2026-09-29, with no restart for 100 minutes). With
router.env_files the router re-reads the files systemd loads its keys from,
whenever they change, and /healthz shows which key it holds so the sync can
confirm the rotation took without a restart.
"""
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from helpers import CHAT_UTTERANCES, FakeBackend, RouterEnv, base_config, run

from hello_operator.config import expand_refs


def _fp(value):
    return hashlib.sha256(value.encode()).hexdigest()[:12]


def _write_env(path, **values):
    path.write_text("".join(f"{k}={v}\n" for k, v in values.items()))


def _cfg(tmp_path, env_file, **models_extra):
    models = {"nous": {"id": "vendor/nous:free", "endpoint": "BACKEND",
                       "api_key": "${HO_TEST_NOUS_KEY}",
                       "capabilities": ["text"], "context_window": 131072}}
    models.update(models_extra)
    roles = {"chat": {"cascade": list(models), "utterances": CHAT_UTTERANCES}}
    return base_config(tmp_path, models=models, roles=roles, default_role="chat",
                       env_files=[str(env_file)])


def _echo(body, idx):
    return {"content": "ok"}


def test_rotated_key_is_used_without_a_restart(tmp_path, monkeypatch):
    env = tmp_path / "hello-operator.env"
    # systemd put the token in the process environment at start...
    monkeypatch.setenv("HO_TEST_NOUS_KEY", "token-old")
    _write_env(env, HO_TEST_NOUS_KEY="token-old")

    async def scenario():
        backend = FakeBackend({"vendor/nous:free": _echo})
        async with RouterEnv(tmp_path, _cfg(tmp_path, env), backend) as env_:
            s1, _, _ = await env_.chat([{"role": "user", "content": "hello chat"}])
            async with env_.client.get(f"{env_.base}/healthz") as r:
                before = (await r.json())["keys"]
            # ...then the token rotates: only the file changes.
            _write_env(env, HO_TEST_NOUS_KEY="token-new")
            async with env_.client.get(f"{env_.base}/healthz") as r:
                after = (await r.json())["keys"]
            s2, _, _ = await env_.chat([{"role": "user", "content": "hello chat again"}])
            return s1, s2, before, after, list(backend.auth)

    s1, s2, before, after, auth = run(scenario())
    assert (s1, s2) == (200, 200)
    assert auth == ["Bearer token-old", "Bearer token-new"]
    assert before == {"HO_TEST_NOUS_KEY": _fp("token-old")}
    assert after == {"HO_TEST_NOUS_KEY": _fp("token-new")}
    assert "token-new" not in str(after)   # a fingerprint, never the key


def test_key_pool_follows_the_file(tmp_path, monkeypatch):
    env = tmp_path / "keys.env"
    monkeypatch.setenv("HO_TEST_K1", "k1")
    monkeypatch.setenv("HO_TEST_K2", "k2-old")
    _write_env(env, HO_TEST_K1="k1", HO_TEST_K2="k2-old")
    cfg = _cfg(tmp_path, env)
    cfg["router"]["key_pools"] = {"BACKEND": ["${HO_TEST_K1}", "${HO_TEST_K2}"]}

    async def scenario():
        backend = FakeBackend({"vendor/nous:free": _echo})
        async with RouterEnv(tmp_path, cfg, backend) as env_:
            _write_env(env, HO_TEST_K1="k1", HO_TEST_K2="k2-new")
            for i in range(4):
                await env_.chat([{"role": "user", "content": f"hello chat {i}"}])
            return list(backend.auth)

    auth = run(scenario())
    assert set(auth) == {"Bearer k1", "Bearer k2-new"}


def test_missing_env_file_keeps_the_startup_keys(tmp_path, monkeypatch):
    """The keys file is optional in the unit (EnvironmentFile=-...): its absence
    must not blank a key the process already holds."""
    monkeypatch.setenv("HO_TEST_NOUS_KEY", "token-start")

    async def scenario():
        backend = FakeBackend({"vendor/nous:free": _echo})
        async with RouterEnv(tmp_path, _cfg(tmp_path, tmp_path / "absent.env"),
                             backend) as env_:
            status, _, _ = await env_.chat([{"role": "user", "content": "hello chat"}])
            return status, list(backend.auth)

    status, auth = run(scenario())
    assert status == 200
    assert auth == ["Bearer token-start"]


def test_expand_refs_matches_expandvars_for_unset_names():
    env = {"A": "1"}
    assert expand_refs("${A}", env) == "1"
    assert expand_refs("$A-x", env) == "1-x"
    assert expand_refs("${MISSING}", env) == "${MISSING}"   # left as-is, like expandvars
    assert expand_refs("literal", env) == "literal"
