"""Mixed local + cloud fleets: the remote opt-in gate, the role locality
fence (which failover must never cross), auth precedence, probe cost gating,
drift quieting for hosted catalogs, and offline resilience."""
import sys
from pathlib import Path

import aiohttp
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent))

from helpers import FakeBackend, RouterEnv, base_config, run
from hello_operator import config as config_mod
from hello_operator import discovery
from hello_operator.config import ConfigError, infer_location


def _echo(name):
    return lambda body, idx: {"content": f"answer from {name}"}


def _write(tmp_path, doc):
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(doc))
    return str(p)


# ------------------------------------------------------- location inference

def test_infer_location():
    local = ["http://127.0.0.1:8080/v1", "http://localhost:11434/v1",
             "http://192.168.1.5:8080/v1", "http://10.0.0.2:8000/v1",
             "http://172.16.0.9:1234/v1", "http://studio.local:1234/v1",
             "http://box.lan:8080/v1", "http://host.docker.internal:8080/v1"]
    remote = ["https://openrouter.ai/api/v1", "https://api.openai.com/v1",
              "https://api.anthropic.com/v1", "http://203.0.113.7:8080/v1",
              "https://my-vps.example.com/v1", "http://172.32.0.1/v1"]
    for e in local:
        assert infer_location(e) == "local", e
    for e in remote:
        assert infer_location(e) == "remote", e


# ------------------------------------------------------------ opt-in gate

def test_remote_endpoint_requires_explicit_opt_in(tmp_path):
    doc = {"models": {"cloud": {"id": "gpt-x",
                                "endpoint": "https://api.example-cloud.com/v1"}}}
    with pytest.raises(ConfigError, match="allow_remote_endpoints"):
        config_mod.load(_write(tmp_path, doc))
    # with the gate: loads, and warns that traffic leaves the machine
    doc["router"] = {"allow_remote_endpoints": True}
    cfg = config_mod.load(_write(tmp_path, doc))
    assert cfg.models["cloud"].location == "remote"
    assert any("leaves this machine" in w for w in cfg.warnings)


def test_declared_local_overrides_inference(tmp_path):
    # a tailnet/VPN host looks remote by address but the user knows better —
    # declaration wins, no gate required (spec 3.1 resolution order)
    doc = {"models": {"studio": {"id": "m", "location": "local",
                                 "endpoint": "https://studio.tailnet-name.ts.net/v1"}}}
    cfg = config_mod.load(_write(tmp_path, doc))
    assert cfg.models["studio"].location == "local"
    assert cfg.models["studio"].provenance["location"] == "declared"


def test_role_locality_fence_validated_at_startup(tmp_path):
    doc = {
        "router": {"allow_remote_endpoints": True},
        "models": {
            "loc": {"id": "loc", "endpoint": "http://127.0.0.1:9/v1"},
            "cld": {"id": "cld", "endpoint": "https://api.example-cloud.com/v1"},
        },
        "roles": {"private": {"cascade": ["loc", "cld"], "locality": "local"}},
        "routing": {"default_role": "private"},
    }
    with pytest.raises(ConfigError, match="locality 'local'"):
        config_mod.load(_write(tmp_path, doc))


# --------------------------------------------------------- routing behavior

def mixed_cfg(tmp_path, backend_placeholder="BACKEND"):
    """A 'private' local-only role and a 'burst' role fronted by cloud."""
    models = {
        "loc": {"id": "loc-model", "endpoint": backend_placeholder,
                "capabilities": ["text", "tools"]},
        "cld": {"id": "cld-model", "endpoint": backend_placeholder,
                "location": "remote",     # declared remote (fake runs on loopback)
                "capabilities": ["text", "tools"]},
    }
    roles = {
        "private": {"cascade": ["loc"], "locality": "local"},
        "burst": {"cascade": ["cld", "loc"]},
    }
    return base_config(tmp_path, models=models, roles=roles,
                       default_role="private", embedding=False,
                       allow_remote_endpoints=True)


def test_failover_never_crosses_the_locality_fence(tmp_path):
    async def scenario():
        backend = FakeBackend({"cld-model": _echo("cloud")})
        cfg = mixed_cfg(tmp_path)
        # the local model's endpoint is dead; the cloud model is alive
        cfg["models"]["loc"]["endpoint"] = "http://127.0.0.1:1/v1"
        async with RouterEnv(tmp_path, cfg, backend) as env:
            status, payload, _ = await env.chat(
                [{"role": "user", "content": "sensitive stuff"}], session="s1")
            # the ONLY live model is remote, but the session's role is fenced
            # local — a 502 naming the failure is correct; silently shipping
            # the content to the cloud is not.
            assert status == 502
            assert "loc" in payload["error"]["message"]
        assert backend.per_model_calls.get("cld-model") is None  # never called
    run(scenario())


def test_cloud_down_degrades_to_local(tmp_path):
    # offline resilience: the burst role fronts cloud, cloud is unreachable,
    # the local model in the same cascade serves (FR-9)
    async def scenario():
        backend = FakeBackend({"loc-model": _echo("local")})
        cfg = mixed_cfg(tmp_path)
        cfg["models"]["cld"]["endpoint"] = "http://127.0.0.1:1/v1"
        cfg["routing"]["default_role"] = "burst"
        async with RouterEnv(tmp_path, cfg, backend) as env:
            status, payload, headers = await env.chat(
                [{"role": "user", "content": "hi"}], session="s1")
            assert status == 200
            assert payload["choices"][0]["message"]["content"] == "answer from local"
            assert headers["x-router-decision"] == "failover"
    run(scenario())


def test_registry_api_key_beats_client_bearer(tmp_path):
    # the client's local-harness token must NOT be forwarded to a backend
    # with its own declared key
    async def scenario():
        seen = {}

        def capture(body, idx):
            return {"content": "ok"}

        backend = FakeBackend({"cld-model": capture, "loc-model": capture})

        orig_chat = backend.chat

        async def chat(request):
            body = await request.clone().json()
            seen[body.get("model")] = request.headers.get("Authorization", "")
            return await orig_chat(request)
        backend.chat = chat  # type: ignore[method-assign]

        cfg = mixed_cfg(tmp_path)
        cfg["models"]["cld"]["api_key"] = "sk-cloud-secret"
        cfg["routing"]["default_role"] = "burst"
        async with RouterEnv(tmp_path, cfg, backend) as env:
            await env.chat([{"role": "user", "content": "hi"}], session="s1",
                           headers={"Authorization": "Bearer local-harness-token"})
        # declared key won for the keyed backend
        assert seen.get("cld-model") == "Bearer sk-cloud-secret"
    run(scenario())


def test_client_bearer_still_forwards_to_keyless_backends(tmp_path):
    async def scenario():
        seen = {}
        backend = FakeBackend({"loc-model": lambda b, i: {"content": "ok"}})
        orig_chat = backend.chat

        async def chat(request):
            body = await request.clone().json()
            seen[body.get("model")] = request.headers.get("Authorization", "")
            return await orig_chat(request)
        backend.chat = chat  # type: ignore[method-assign]

        cfg = base_config(tmp_path,
                          models={"loc": {"id": "loc-model", "endpoint": "BACKEND"}},
                          roles={}, default_role="", embedding=False)
        cfg.pop("roles"), cfg.pop("routing")
        async with RouterEnv(tmp_path, cfg, backend) as env:
            await env.chat([{"role": "user", "content": "hi"}],
                           headers={"Authorization": "Bearer pass-through"})
        assert seen.get("loc-model") == "Bearer pass-through"
    run(scenario())


# ------------------------------------------------------ discovery behavior

def test_hosted_catalog_is_not_drift(tmp_path):
    # a remote endpoint serving 3 unregistered models produces no "extra"
    # noise; a registered model going missing is still flagged
    async def scenario():
        backend = FakeBackend({"served-a": _echo("a"), "catalog-1": _echo("x"),
                               "catalog-2": _echo("x")})
        base = await backend.start()
        try:
            doc = {
                "router": {"state_dir": str(tmp_path / "state"),
                           "allow_remote_endpoints": True},
                "models": {"a": {"id": "served-a", "endpoint": base,
                                 "location": "remote"},
                           "gone": {"id": "not-served", "endpoint": base,
                                    "location": "remote"}},
                "roles": {"work": {"cascade": ["a", "gone"]}},
                "routing": {"default_role": "work"},
            }
            cfg = config_mod.load(_write(tmp_path, doc))
            async with aiohttp.ClientSession() as http:
                notes = await discovery.drift_check(http, cfg)
        finally:
            await backend.stop()
        joined = "\n".join(notes)
        assert "catalog-1" not in joined         # hosted catalog: no extra noise
        assert "not-served" in joined            # missing still flagged
    run(scenario())


def test_openrouter_style_context_length_detected(tmp_path):
    async def scenario():
        from aiohttp import web
        backend = FakeBackend({"vendor/big-model": _echo("x")})

        async def models(request):
            return web.json_response({"object": "list", "data": [
                {"id": "vendor/big-model", "object": "model",
                 "context_length": 200000,
                 "architecture": {"input_modalities": ["text"]}}]})
        backend.models = models  # type: ignore[method-assign]
        base = await backend.start()
        try:
            async with aiohttp.ClientSession() as http:
                det = await discovery.harvest(http, base, "vendor/big-model")
        finally:
            await backend.stop()
        assert det["context_window"] == 200000
        assert det["source"] == "openrouter-style"
    run(scenario())


def test_remote_probes_default_off(tmp_path):
    from hello_operator.config import ModelSpec
    remote = ModelSpec(key="c", id="c", endpoint="https://api.x.com/v1",
                       location="remote")
    assert remote.probes_enabled is False
    opted = ModelSpec(key="c", id="c", endpoint="https://api.x.com/v1",
                      location="remote", probe=True)
    assert opted.probes_enabled is True
