"""End-to-end routing behavior through real sockets: classification, affinity,
transitions, escalation, failover, degenerate mode, pins."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from helpers import (CHAT_UTTERANCES, CODE_UTTERANCES, WEATHER_TOOL, FakeBackend,
                     RouterEnv, bad_json_call, base_config, good_call, no_call, run)


def _echo(name):
    return lambda body, idx: {"content": f"answer from {name}"}


def two_model_cfg(tmp_path, **router_extra):
    """chat: [fast -> big], code: [big]; deterministic fake embeddings."""
    models = {
        "fast": {"id": "fast-model", "endpoint": "BACKEND",
                 "capabilities": ["text", "tools", "json"], "context_window": 32768,
                 "speed_class": "fast"},
        "big": {"id": "big-model", "endpoint": "BACKEND",
                "capabilities": ["text", "tools", "json", "vision"],
                "context_window": 131072, "speed_class": "slow"},
    }
    roles = {
        "chat": {"cascade": ["fast", "big"], "utterances": CHAT_UTTERANCES},
        "code": {"cascade": ["big"], "utterances": CODE_UTTERANCES},
    }
    return base_config(tmp_path, models=models, roles=roles,
                       default_role="chat", **router_extra)


# ------------------------------------------------------------- degenerate

def test_degenerate_passthrough(tmp_path):
    async def scenario():
        backend = FakeBackend({"solo-model": _echo("solo")})
        cfg = base_config(tmp_path,
                          models={"solo": {"id": "solo-model", "endpoint": "BACKEND"}},
                          roles={}, default_role="", embedding=False)
        cfg.pop("roles"), cfg.pop("routing")
        async with RouterEnv(tmp_path, cfg, backend) as env:
            status, payload, headers = await env.chat(
                [{"role": "user", "content": "hi"}])
            assert status == 200
            assert payload["choices"][0]["message"]["content"] == "answer from solo"
            assert headers["x-router-decision"] == "degenerate"
            assert payload["router"]["backend_model"] == "solo-model"
    run(scenario())


def test_degenerate_still_capability_filters(tmp_path):
    # FR-3: never silently route media to a text-only model, even the only one.
    async def scenario():
        backend = FakeBackend({"solo-model": _echo("solo")})
        cfg = base_config(tmp_path,
                          models={"solo": {"id": "solo-model", "endpoint": "BACKEND",
                                           "capabilities": ["text"]}},
                          roles={}, default_role="", embedding=False)
        cfg.pop("roles"), cfg.pop("routing")
        async with RouterEnv(tmp_path, cfg, backend) as env:
            status, payload, _ = await env.chat([
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": "u"}}]}])
            assert status == 400
            assert "vision" in payload["error"]["message"]
    run(scenario())


# -------------------------------------------------- classification + affinity

def test_classification_routes_by_role_and_affinity_holds(tmp_path):
    async def scenario():
        backend = FakeBackend({"fast-model": _echo("fast"), "big-model": _echo("big")})
        async with RouterEnv(tmp_path, two_model_cfg(tmp_path), backend) as env:
            # code-flavored message -> code role -> big
            status, payload, headers = await env.chat(
                [{"role": "user", "content": "please fix the bug in this code function"}],
                session="s1")
            assert status == 200
            assert headers["x-router-model"] == "big"
            assert headers["x-router-role"] == "code"
            assert headers["x-router-decision"] == "new"
            # neutral follow-up in the same session: hold (FR-5/FR-6)
            status, payload, headers = await env.chat(
                [{"role": "user", "content": "please fix the bug in this code function"},
                 {"role": "assistant", "content": "done"},
                 {"role": "user", "content": "thanks, and also this one"}],
                session="s1")
            assert headers["x-router-decision"] == "affinity"
            assert headers["x-router-model"] == "big"
            # chat-flavored message in a NEW session -> chat -> fast
            status, payload, headers = await env.chat(
                [{"role": "user", "content": "hello, chat about the weather?"}],
                session="s2")
            assert headers["x-router-role"] == "chat"
            assert headers["x-router-model"] == "fast"
    run(scenario())


def test_classifier_disagreement_transition(tmp_path):
    async def scenario():
        backend = FakeBackend({"fast-model": _echo("fast"), "big-model": _echo("big")})
        async with RouterEnv(tmp_path, two_model_cfg(tmp_path), backend) as env:
            _, _, h = await env.chat(
                [{"role": "user", "content": "hello, chat about the weather?"}],
                session="s1")
            assert h["x-router-role"] == "chat"
            # strong code signal mid-session, same toolset -> classifier transition
            _, _, h = await env.chat(
                [{"role": "user", "content": "hello, chat about the weather?"},
                 {"role": "assistant", "content": "sunny"},
                 {"role": "user",
                  "content": "fix the bug in this code function, debug the function"}],
                session="s1")
            assert h["x-router-decision"] == "transition:classifier"
            assert h["x-router-role"] == "code"
    run(scenario())


def test_toolset_change_is_a_transition(tmp_path):
    async def scenario():
        backend = FakeBackend({"fast-model": _echo("fast"), "big-model": _echo("big")})
        async with RouterEnv(tmp_path, two_model_cfg(tmp_path), backend) as env:
            await env.chat([{"role": "user", "content": "hello chat weather"}],
                           session="s1")
            _, _, h = await env.chat(
                [{"role": "user", "content": "hello chat weather"},
                 {"role": "assistant", "content": "hi"},
                 {"role": "user", "content": "now fix the code bug function"}],
                session="s1", tools=WEATHER_TOOL)
            assert h["x-router-decision"] == "transition:toolset"
            assert h["x-router-role"] == "code"
    run(scenario())


def test_compaction_header_is_a_transition(tmp_path):
    async def scenario():
        backend = FakeBackend({"fast-model": _echo("fast"), "big-model": _echo("big")})
        async with RouterEnv(tmp_path, two_model_cfg(tmp_path), backend) as env:
            await env.chat([{"role": "user", "content": "hello chat weather"}],
                           session="s1")
            _, _, h = await env.chat(
                [{"role": "user", "content": "summarized: fix the code bug function"}],
                session="s1", headers={"x-context-compacted": "true"})
            assert h["x-router-decision"] == "transition:compaction"
            assert h["x-router-role"] == "code"
    run(scenario())


def test_vision_request_excludes_text_only_models(tmp_path):
    async def scenario():
        backend = FakeBackend({"fast-model": _echo("fast"), "big-model": _echo("big")})
        async with RouterEnv(tmp_path, two_model_cfg(tmp_path), backend) as env:
            # image in history: fast (no vision) is filtered; chat cascade
            # falls through to big.
            _, _, h = await env.chat(
                [{"role": "user", "content": [
                    {"type": "text", "text": "hello chat weather, what is this"},
                    {"type": "image_url", "image_url": {"url": "u"}}]}],
                session="s1")
            assert h["x-router-model"] == "big"
    run(scenario())


# ------------------------------------------------------------- escalation

def esc_cfg(tmp_path, **router_extra):
    models = {
        "weak": {"id": "weak-model", "endpoint": "BACKEND",
                 "capabilities": ["text", "tools"], "context_window": 32768},
        "strong": {"id": "strong-model", "endpoint": "BACKEND",
                   "capabilities": ["text", "tools"], "context_window": 131072},
    }
    roles = {"work": {"cascade": ["weak", "strong"]}}
    return base_config(tmp_path, models=models, roles=roles, default_role="work",
                       embedding=False, **router_extra)


def test_nonstream_bad_tool_call_escalates_in_request(tmp_path):
    async def scenario():
        backend = FakeBackend({"weak-model": bad_json_call, "strong-model": good_call})
        async with RouterEnv(tmp_path, esc_cfg(tmp_path), backend) as env:
            status, payload, headers = await env.chat(
                [{"role": "user", "content": "weather in paris, use the tool"}],
                session="s1", tools=WEATHER_TOOL)
            assert status == 200
            # weak was tried, failed validation, strong served the turn
            assert backend.per_model_calls == {"weak-model": 1, "strong-model": 1}
            assert headers["x-router-model"] == "strong"
            assert headers["x-router-decision"] == "escalation:tool-validation"
            args = payload["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]
            json.loads(args)  # served answer is the valid one
    run(scenario())


def test_hop_limit_zero_suppresses_escalation(tmp_path):
    async def scenario():
        backend = FakeBackend({"weak-model": bad_json_call, "strong-model": good_call})
        async with RouterEnv(tmp_path, esc_cfg(tmp_path, hop_limit=0), backend) as env:
            status, payload, headers = await env.chat(
                [{"role": "user", "content": "weather in paris"}],
                session="s1", tools=WEATHER_TOOL)
            assert status == 200
            assert backend.per_model_calls == {"weak-model": 1}  # no hop
            assert headers["x-router-model"] == "weak"
    run(scenario())


def test_streamed_failure_commits_then_escalates_next_turn(tmp_path):
    async def scenario():
        backend = FakeBackend({"weak-model": bad_json_call, "strong-model": good_call})
        async with RouterEnv(tmp_path, esc_cfg(tmp_path), backend) as env:
            # streamed turn: committed even though the tool call is bad (FR-8)
            status, payload, headers = await env.chat(
                [{"role": "user", "content": "weather in paris"}],
                session="s1", stream=True, tools=WEATHER_TOOL)
            assert status == 200
            assert headers["x-router-model"] == "weak"
            # next turn in the session starts one cascade position down (FR-7)
            status, payload, headers = await env.chat(
                [{"role": "user", "content": "try again"}],
                session="s1", tools=WEATHER_TOOL)
            assert headers["x-router-model"] == "strong"
            assert headers["x-router-decision"] == "escalation:tool-validation"
    run(scenario())


def test_missing_required_call_escalates(tmp_path):
    async def scenario():
        backend = FakeBackend({"weak-model": no_call, "strong-model": good_call})
        async with RouterEnv(tmp_path, esc_cfg(tmp_path), backend) as env:
            status, payload, headers = await env.chat(
                [{"role": "user", "content": "weather in paris"}],
                session="s1", tools=WEATHER_TOOL, tool_choice="required")
            assert status == 200
            assert headers["x-router-model"] == "strong"
    run(scenario())


def test_repeat_identical_call_escalates(tmp_path):
    async def scenario():
        backend = FakeBackend({"weak-model": good_call, "strong-model": good_call})
        cfg = esc_cfg(tmp_path, repeat_call_threshold=2)
        async with RouterEnv(tmp_path, cfg, backend) as env:
            msgs = [{"role": "user", "content": "weather in paris"}]
            _, _, h1 = await env.chat(msgs, session="s1", tools=WEATHER_TOOL)
            assert h1["x-router-model"] == "weak"
            _, _, h2 = await env.chat(msgs, session="s1", tools=WEATHER_TOOL)
            assert h2["x-router-model"] == "weak"       # threshold reached here
            _, _, h3 = await env.chat(msgs, session="s1", tools=WEATHER_TOOL)
            assert h3["x-router-model"] == "strong"     # armed escalation fires
            assert h3["x-router-decision"] == "escalation:repeat-call"
    run(scenario())


# ---------------------------------------------------------------- failover

def test_backend_unreachable_fails_over(tmp_path):
    async def scenario():
        backend = FakeBackend({"live-model": _echo("live")})
        models = {
            "dead": {"id": "dead-model", "endpoint": "http://127.0.0.1:1/v1",
                     "capabilities": ["text"]},
            "live": {"id": "live-model", "endpoint": "BACKEND",
                     "capabilities": ["text"]},
        }
        roles = {"work": {"cascade": ["dead", "live"]}}
        cfg = base_config(tmp_path, models=models, roles=roles,
                          default_role="work", embedding=False)
        async with RouterEnv(tmp_path, cfg, backend) as env:
            status, payload, headers = await env.chat(
                [{"role": "user", "content": "hi"}], session="s1")
            assert status == 200
            assert payload["choices"][0]["message"]["content"] == "answer from live"
            assert headers["x-router-model"] == "live"
            assert headers["x-router-decision"] == "failover"
    run(scenario())


def test_all_backends_down_names_the_failure(tmp_path):
    async def scenario():
        backend = FakeBackend({})
        models = {"dead": {"id": "dead-model", "endpoint": "http://127.0.0.1:1/v1"}}
        cfg = base_config(tmp_path, models=models, roles={}, default_role="",
                          embedding=False)
        cfg.pop("roles"), cfg.pop("routing")
        async with RouterEnv(tmp_path, cfg, backend) as env:
            status, payload, _ = await env.chat(
                [{"role": "user", "content": "hi"}])
            assert status == 502
            assert "dead" in payload["error"]["message"]
    run(scenario())


# --------------------------------------------------------------------- pin

def test_pin_header_overrides_routing(tmp_path):
    async def scenario():
        backend = FakeBackend({"fast-model": _echo("fast"), "big-model": _echo("big")})
        async with RouterEnv(tmp_path, two_model_cfg(tmp_path), backend) as env:
            _, _, h = await env.chat(
                [{"role": "user", "content": "fix the code bug function"}],
                session="s1", headers={"x-router-pin": "fast"})
            assert h["x-router-model"] == "fast"
            assert h["x-router-decision"] == "pin"
            # pin persists for the session without re-sending the header
            _, _, h = await env.chat(
                [{"role": "user", "content": "more code bug fixing"}],
                session="s1")
            assert h["x-router-model"] == "fast"
    run(scenario())


def test_pin_never_bypasses_capability_filter(tmp_path):
    async def scenario():
        backend = FakeBackend({"fast-model": _echo("fast"), "big-model": _echo("big")})
        async with RouterEnv(tmp_path, two_model_cfg(tmp_path), backend) as env:
            status, payload, _ = await env.chat(
                [{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": "u"}}]}],
                session="s1", headers={"x-router-pin": "fast"})
            assert status == 400
            assert "vision" in payload["error"]["message"]
    run(scenario())


# ---------------------------------------------------------------- plumbing

def test_models_and_health_endpoints(tmp_path):
    async def scenario():
        backend = FakeBackend({"fast-model": _echo("fast"), "big-model": _echo("big")})
        async with RouterEnv(tmp_path, two_model_cfg(tmp_path), backend) as env:
            async with env.client.get(f"{env.base}/v1/models") as r:
                data = await r.json()
            ids = [m["id"] for m in data["data"]]
            assert "hermes-router" in ids and "fast" in ids and "big" in ids
            async with env.client.get(f"{env.base}/healthz") as r:
                h = await r.json()
            assert h["status"] == "ok" and h["classification"] == "active"
    run(scenario())


def test_unknown_model_is_a_clear_404(tmp_path):
    async def scenario():
        backend = FakeBackend({"fast-model": _echo("fast"), "big-model": _echo("big")})
        async with RouterEnv(tmp_path, two_model_cfg(tmp_path), backend) as env:
            status, payload, _ = await env.chat(
                [{"role": "user", "content": "hi"}], model="gpt-4o")
            assert status == 404
            assert "gpt-4o" in payload["error"]["message"]
    run(scenario())


def test_decision_log_written(tmp_path):
    async def scenario():
        backend = FakeBackend({"fast-model": _echo("fast"), "big-model": _echo("big")})
        async with RouterEnv(tmp_path, two_model_cfg(tmp_path), backend) as env:
            await env.chat([{"role": "user", "content": "hello chat weather"}],
                           session="s1")
        lines = [json.loads(l) for l in
                 (tmp_path / "decisions.jsonl").read_text().splitlines()]
        assert lines and lines[-1]["role"] == "chat"
        assert lines[-1]["decision"] == "new"
        assert "routing_latency_ms" in lines[-1]
    run(scenario())


def test_streaming_relay_preserves_content(tmp_path):
    async def scenario():
        backend = FakeBackend({"fast-model": _echo("fast"), "big-model": _echo("big")})
        async with RouterEnv(tmp_path, two_model_cfg(tmp_path), backend) as env:
            status, payload, headers = await env.chat(
                [{"role": "user", "content": "hello chat weather"}],
                session="s1", stream=True)
            assert status == 200
            assert payload["choices"][0]["message"]["content"] == "answer from fast"
            assert headers["x-router-model"] == "fast"
    run(scenario())


def test_tool_chain_classification_is_cached(tmp_path):
    # AC-3/NFR-1: mid tool chain the last user message is unchanged, so the
    # affinity path must not re-embed it every step.
    async def scenario():
        backend = FakeBackend({"fast-model": _echo("fast"), "big-model": _echo("big")})
        async with RouterEnv(tmp_path, two_model_cfg(tmp_path), backend) as env:
            msgs = [{"role": "user", "content": "hello chat weather"}]
            await env.chat(msgs, session="s1")
            embeds_after_first = sum(
                1 for c in backend.calls if "input" in c)  # embeddings bodies
            # ten "tool chain" turns: same user text, growing assistant history
            for i in range(10):
                msgs = msgs + [{"role": "assistant", "content": f"step {i}"},
                               {"role": "user", "content": "hello chat weather"}]
            # (send with the SAME last user text each time)
            for i in range(10):
                _, _, h = await env.chat(msgs, session="s1")
                assert h["x-router-decision"] == "affinity"
            embeds_total = sum(1 for c in backend.calls if "input" in c)
        # centroid prep + first classify happened; the 10 affinity turns added 0
        assert embeds_total == embeds_after_first
    run(scenario())
