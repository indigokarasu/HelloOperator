"""Discovery layer: metadata harvest, drift flagging, proposal generation."""
import sys
from pathlib import Path

import aiohttp

sys.path.insert(0, str(Path(__file__).parent))

from helpers import FakeBackend, base_config, run
from hello_operator import config as config_mod
from hello_operator import discovery


def _cfg(tmp_path, backend_base, native_models):
    import yaml
    doc = base_config(
        tmp_path,
        models={k: {"id": k, "endpoint": "BACKEND"} for k in native_models},
        roles={"work": {"cascade": list(native_models)}},
        default_role="work", embedding=False)
    text = yaml.safe_dump(doc).replace("BACKEND", backend_base)
    p = tmp_path / "c.yaml"
    p.write_text(text)
    return config_mod.load(str(p))


def test_harvest_ollama_metadata(tmp_path):
    async def scenario():
        backend = FakeBackend({"llama3:8b": lambda b, i: {"content": "x"}},
                              native="ollama")
        base = await backend.start()
        try:
            async with aiohttp.ClientSession() as http:
                det = await discovery.harvest(http, base, "llama3:8b")
        finally:
            await backend.stop()
        assert det["source"] == "ollama"
        assert det["context_window"] == 32768
        assert "tools" in det["capabilities"] and "text" in det["capabilities"]
        assert det["digest"] == "digest-llama3:8b"
    run(scenario())


def test_harvest_llamacpp_props(tmp_path):
    async def scenario():
        backend = FakeBackend({"local-gguf": lambda b, i: {"content": "x"}},
                              native="llamacpp")
        base = await backend.start()
        try:
            async with aiohttp.ClientSession() as http:
                det = await discovery.harvest(http, base, "local-gguf")
        finally:
            await backend.stop()
        assert det["source"] == "llama.cpp"
        assert det["context_window"] == 16384
        assert "text" in det["capabilities"]
    run(scenario())


def test_harvest_survives_a_silent_backend(tmp_path):
    # Discovery must be able to fail entirely and leave the router usable on
    # declarations alone (FR-15).
    async def scenario():
        async with aiohttp.ClientSession() as http:
            det = await discovery.harvest(http, "http://127.0.0.1:1/v1", "ghost")
        assert det == {}
    run(scenario())


def test_drift_flags_extra_and_missing(tmp_path):
    async def scenario():
        backend = FakeBackend({"served-a": lambda b, i: {"content": "x"},
                               "served-b": lambda b, i: {"content": "x"}})
        base = await backend.start()
        try:
            import yaml
            doc = base_config(
                tmp_path,
                models={"a": {"id": "served-a", "endpoint": "BACKEND"},
                        "gone": {"id": "not-served", "endpoint": "BACKEND"}},
                roles={"work": {"cascade": ["a", "gone"]}},
                default_role="work", embedding=False)
            text = yaml.safe_dump(doc).replace("BACKEND", base)
            p = tmp_path / "c.yaml"
            p.write_text(text)
            cfg = config_mod.load(str(p))
            async with aiohttp.ClientSession() as http:
                notes = await discovery.drift_check(http, cfg)
        finally:
            await backend.stop()
        joined = "\n".join(notes)
        assert "served-b" in joined        # served but not registered
        assert "not-served" in joined      # registered but not served
    run(scenario())


def test_proposal_contains_models_and_roles(tmp_path):
    async def scenario():
        backend = FakeBackend({"llama3:8b": lambda b, i: {"content": "x"}},
                              native="ollama")
        base = await backend.start()
        try:
            cfg = _cfg(tmp_path, base, ["llama3:8b"])
            async with aiohttp.ClientSession() as http:
                proposal = await discovery.generate_proposal(http, cfg)
        finally:
            await backend.stop()
        assert "models:" in proposal and "roles:" in proposal
        assert "context_window: 32768" in proposal      # detected, not default
        assert "detected" in proposal                   # provenance is visible
        assert "Nothing routes on this file" in proposal
    run(scenario())


def test_detection_cache_reused(tmp_path):
    async def scenario():
        backend = FakeBackend({"llama3:8b": lambda b, i: {"content": "x"}},
                              native="ollama")
        base = await backend.start()
        try:
            cfg = _cfg(tmp_path, base, ["llama3:8b"])
            async with aiohttp.ClientSession() as http:
                first = await discovery.detect_all(http, cfg)
                second = await discovery.detect_all(http, cfg)
        finally:
            await backend.stop()
        assert first["llama3:8b"]["context_window"] == 32768
        assert second == first
        cache = discovery.cache_load(cfg)
        assert any("digest-llama3:8b" in k for k in cache)  # keyed on digest
    run(scenario())


def test_name_hints_are_hints_only():
    assert "vision" in discovery.name_hints("qwen2-vl-7b")
    assert "embeddings" in discovery.name_hints("nomic-embed-text")
    assert "unfiltered" in discovery.name_hints("llama3-abliterated")
    assert discovery.name_hints("plainmodel") == set()
