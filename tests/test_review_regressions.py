"""Regression tests for the defects found in the pre-push adversarial review.
Each test names the defect it pins down; if one of these fails, that exact
bug is back."""
import asyncio
import json
import sys
from pathlib import Path

import aiohttp
import yaml
from aiohttp import web

sys.path.insert(0, str(Path(__file__).parent))

from helpers import (WEATHER_TOOL, FakeBackend, RouterEnv, bad_json_call,
                     base_config, good_call, run)
from hello_operator import config as config_mod
from hello_operator import discovery
from hello_operator.capabilities import extract_props
from hello_operator.config import Settings
from hello_operator.escalate import StreamCollector


def _echo(name):
    return lambda body, idx: {"content": f"answer from {name}"}


# ---- defect: shipped example config failed the router's own validation ----

def test_example_config_passes_validation():
    cfg = config_mod.load(str(Path(__file__).parent.parent / "config.example.yaml"))
    assert set(cfg.roles) == {"chat", "code", "vision"}
    assert "vision" in cfg.models["main-model"].capabilities


# ---- defect: base64 image payloads counted as prompt tokens (FR-3) ----

def test_media_payload_not_token_counted():
    big_data_uri = "data:image/png;base64," + "A" * 680_000  # ~500KB image
    body = {"messages": [{"role": "user", "content": [
        {"type": "text", "text": "what is this?"},
        {"type": "image_url", "image_url": {"url": big_data_uri}}]}]}
    p = extract_props(body, Settings())
    # flat per-media estimate, not 170k "tokens" of base64
    assert p.has_image
    assert p.est_tokens < 3000


# ---- defect: StreamCollector conflated n>1 choices into invalid JSON ----

def test_collector_ignores_non_primary_choices():
    col = StreamCollector()
    chunks = [
        {"choices": [{"index": 0, "delta": {"content": "A1"}},
                     {"index": 1, "delta": {"content": "B1"}}]},
        {"choices": [{"index": 1, "delta": {"tool_calls": [
            {"index": 0, "function": {"name": "g", "arguments": '{"b":1}'}}]}}]},
        {"choices": [{"index": 0, "delta": {"content": "A2"}}]},
    ]
    for c in chunks:
        col.feed(b"data: " + json.dumps(c).encode() + b"\n\n")
    msg = col.assembled()
    assert msg["content"] == "A1A2"          # no B1 interleaved
    assert "tool_calls" not in msg           # choice-1 fragments not adopted


# ---- defect: mid-stream backend death crashed the handler uncaught ----
# ---- and a caught retry wrote a second HTTP response into the open body ----

def test_backend_dying_mid_stream_commits_and_escalates_next_turn(tmp_path):
    def abort_mid_stream(body, idx):
        return {"__abort_stream__": True}

    async def scenario():
        backend = FakeBackend({"weak-model": abort_mid_stream,
                               "strong-model": good_call})
        # teach the fake backend to abort: patch its chat handler inline
        orig_chat = backend.chat

        async def chat(request):
            b = await request.json()
            backend.calls.append(b)
            model = b.get("model", "")
            idx = backend.per_model_calls.get(model, 0)
            backend.per_model_calls[model] = idx + 1
            if model == "weak-model" and b.get("stream"):
                resp = web.StreamResponse()
                resp.headers["Content-Type"] = "text/event-stream"
                await resp.prepare(request)
                chunk = {"id": "x", "object": "chat.completion.chunk",
                         "model": model,
                         "choices": [{"index": 0, "delta": {"content": "par"},
                                      "finish_reason": None}]}
                await resp.write(b"data: " + json.dumps(chunk).encode() + b"\n\n")
                # die abruptly: no terminal chunk, no [DONE]
                request.transport.close()
                return resp
            return await orig_chat(request)

        backend.chat = chat  # type: ignore[method-assign]
        models = {
            "weak": {"id": "weak-model", "endpoint": "BACKEND",
                     "capabilities": ["text", "tools"]},
            "strong": {"id": "strong-model", "endpoint": "BACKEND",
                       "capabilities": ["text", "tools"]},
        }
        cfg = base_config(tmp_path, models=models,
                          roles={"work": {"cascade": ["weak", "strong"]}},
                          default_role="work", embedding=False)
        # rebuild routes with the patched handler
        async with RouterEnv(tmp_path, cfg, backend) as env:
            # turn 1: stream aborts mid-relay. The router must return the
            # truncated stream (turn committed) — NOT crash, NOT write a
            # second response.
            status, payload, headers = await env.chat(
                [{"role": "user", "content": "hi"}], session="s1", stream=True)
            assert status == 200
            assert headers["x-router-model"] == "weak"
            # turn 2: the abort armed escalation
            status, payload, headers = await env.chat(
                [{"role": "user", "content": "again"}], session="s1")
            assert status == 200
            assert headers["x-router-model"] == "strong"
            assert headers["x-router-decision"].startswith("escalation:")
    run(scenario())


# ---- defect: escalation hop charged before the turn served (sticky-wrong) ----

def test_failed_escalation_keeps_trigger_and_hop_budget(tmp_path):
    async def scenario():
        backend = FakeBackend({"weak-model": bad_json_call, "strong-model": good_call})
        models = {
            "weak": {"id": "weak-model", "endpoint": "BACKEND",
                     "capabilities": ["text", "tools"]},
            # strong points at a dead endpoint: the escalation target is down
            "strong": {"id": "strong-model", "endpoint": "http://127.0.0.1:1/v1",
                       "capabilities": ["text", "tools"]},
        }
        cfg = base_config(tmp_path, models=models,
                          roles={"work": {"cascade": ["weak", "strong"]}},
                          default_role="work", embedding=False)
        async with RouterEnv(tmp_path, cfg, backend) as env:
            # turn 1 streams a bad tool call from weak -> escalation armed
            status, _, h = await env.chat(
                [{"role": "user", "content": "go"}], session="s1",
                stream=True, tools=WEATHER_TOOL)
            assert status == 200 and h["x-router-model"] == "weak"
            # turn 2: select escalates to strong (dead) -> failover must land
            # back on the live weak model, NOT 502 (FR-9 last resort)
            status, _, h = await env.chat(
                [{"role": "user", "content": "go"}], session="s1",
                tools=WEATHER_TOOL)
            assert status == 200
            assert h["x-router-model"] == "weak"
            assert h["x-router-decision"] == "failover"
    run(scenario())


# ---- defect: non-JSON backend body crashed instead of failing over ----

def test_html_response_fails_over(tmp_path):
    async def scenario():
        def html_page(body, idx):
            return web.Response(text="<html>bad gateway page</html>",
                                status=200, content_type="text/html")
        backend = FakeBackend({"broken-model": html_page, "live-model": _echo("live")})
        models = {
            "broken": {"id": "broken-model", "endpoint": "BACKEND"},
            "live": {"id": "live-model", "endpoint": "BACKEND"},
        }
        cfg = base_config(tmp_path, models=models,
                          roles={"work": {"cascade": ["broken", "live"]}},
                          default_role="work", embedding=False)
        async with RouterEnv(tmp_path, cfg, backend) as env:
            status, payload, headers = await env.chat(
                [{"role": "user", "content": "hi"}], session="s1")
            assert status == 200
            assert payload["choices"][0]["message"]["content"] == "answer from live"
            assert headers["x-router-decision"] == "failover"
    run(scenario())


# ---- defect: detection results discarded by serve mode (spec 3.1) ----

def test_detection_reaches_the_served_config(tmp_path):
    async def scenario():
        backend = FakeBackend({"llama3:8b": _echo("l3")}, native="ollama")
        base = await backend.start()
        try:
            doc = {
                "router": {"state_dir": str(tmp_path / "state")},
                "models": {"m": {"id": "llama3:8b", "endpoint": base}},
            }
            cfg_path = tmp_path / "c.yaml"
            cfg_path.write_text(yaml.safe_dump(doc))
            cfg = config_mod.load(str(cfg_path))
            # this is exactly what serve mode now does
            from hello_operator.__main__ import _pre_serve
            cfg2 = await _pre_serve(cfg)
        finally:
            await backend.stop()
        m = cfg2.models["m"]
        assert m.context_window == 32768                 # detected, not default 8192
        assert m.provenance["context_window"] == "detected"
        assert "tools" in m.capabilities                 # from Ollama capabilities
    run(scenario())


# ---- defect: --discover proposal let detection override declarations ----

def test_proposal_respects_declarations(tmp_path):
    async def scenario():
        backend = FakeBackend({"llama3:8b": _echo("l3")}, native="ollama")
        base = await backend.start()
        try:
            doc = {
                "router": {"state_dir": str(tmp_path / "state")},
                "models": {"m": {"id": "llama3:8b", "endpoint": base,
                                 "context_window": 999999}},   # declared
            }
            cfg_path = tmp_path / "c.yaml"
            cfg_path.write_text(yaml.safe_dump(doc))
            cfg = config_mod.load(str(cfg_path))
            async with aiohttp.ClientSession() as http:
                return await discovery.generate_proposal(http, cfg)
        finally:
            await backend.stop()

    proposal = run(scenario())
    assert "context_window: 999999" in proposal      # declaration wins
    assert "declared" in proposal


# ---- defect: fresh harvest results were discarded in favor of stale cache ----

def test_fresh_metadata_beats_stale_cache(tmp_path):
    async def scenario():
        backend = FakeBackend({"llama3:8b": _echo("l3")}, native="ollama")
        base = await backend.start()
        try:
            doc = {"router": {"state_dir": str(tmp_path / "state")},
                   "models": {"m": {"id": "llama3:8b", "endpoint": base}}}
            cfg_path = tmp_path / "c.yaml"
            cfg_path.write_text(yaml.safe_dump(doc))
            cfg = config_mod.load(str(cfg_path))
            async with aiohttp.ClientSession() as http:
                await discovery.detect_all(http, cfg)
            # poison the cache with wrong metadata under the same key
            cache = discovery.cache_load(cfg)
            for k in cache:
                cache[k]["data"]["context_window"] = 4
            discovery.cache_save(cfg, cache)
            async with aiohttp.ClientSession() as http:
                out = await discovery.detect_all(http, cfg)
        finally:
            await backend.stop()
        # fresh harvest must win over the poisoned cache
        assert out["m"]["context_window"] == 32768
    run(scenario())
