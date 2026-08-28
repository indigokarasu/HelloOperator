"""Unit layer: config resolution/validation, capability extraction, tool-call
validation, stream reassembly, session key derivation."""
import json
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent))

from hello_operator import config as config_mod
from hello_operator.affinity import derive_session_key
from hello_operator.capabilities import extract_props, model_ok
from hello_operator.config import ConfigError, Settings
from hello_operator.escalate import (StreamCollector, calls_signature,
                             missing_required_call, validate_tool_calls)


def _write(tmp_path, doc):
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(doc))
    return str(p)


BASE = {
    "models": {"m1": {"id": "m1", "endpoint": "http://127.0.0.1:1/v1"}},
}


# ------------------------------------------------------------------ config

def test_degenerate_config_synthesizes_role(tmp_path):
    cfg = config_mod.load(_write(tmp_path, BASE))
    assert cfg.degenerate
    assert list(cfg.roles) == ["default"]
    assert cfg.settings.default_role == "default"


def test_multiple_models_require_roles(tmp_path):
    doc = {"models": {"a": {"id": "a", "endpoint": "http://127.0.0.1:1/v1"},
                      "b": {"id": "b", "endpoint": "http://127.0.0.1:1/v1"}}}
    with pytest.raises(ConfigError, match="no roles"):
        config_mod.load(_write(tmp_path, doc))


def test_cascade_must_reference_registered_models(tmp_path):
    doc = dict(BASE)
    doc["roles"] = {"chat": {"cascade": ["ghost"]}}
    doc["routing"] = {"default_role": "chat"}
    with pytest.raises(ConfigError, match="unregistered model 'ghost'"):
        config_mod.load(_write(tmp_path, doc))


def test_default_role_must_exist(tmp_path):
    doc = dict(BASE)
    doc["roles"] = {"chat": {"cascade": ["m1"]}}
    doc["routing"] = {"default_role": "nope"}
    with pytest.raises(ConfigError, match="default_role"):
        config_mod.load(_write(tmp_path, doc))


def test_non_loopback_requires_opt_in(tmp_path):
    doc = dict(BASE)
    doc["router"] = {"listen": "0.0.0.0:9999"}
    with pytest.raises(ConfigError, match="loopback"):
        config_mod.load(_write(tmp_path, doc))
    doc["router"]["allow_non_loopback"] = True
    cfg = config_mod.load(_write(tmp_path, doc))
    assert cfg.settings.listen_host == "0.0.0.0"


def test_role_requires_gates_position_zero(tmp_path):
    doc = dict(BASE)
    doc["roles"] = {"vision": {"cascade": ["m1"], "requires": ["vision"]}}
    doc["routing"] = {"default_role": "vision"}
    with pytest.raises(ConfigError, match="requires"):
        config_mod.load(_write(tmp_path, doc))


def test_declaration_wins_over_detection_with_warning(tmp_path):
    doc = {"models": {"m1": {"id": "m1", "endpoint": "http://127.0.0.1:1/v1",
                             "context_window": 1000000}}}
    cfg = config_mod.load(_write(tmp_path, doc),
                          detected_cache={"m1": {"context_window": 256000}})
    m = cfg.models["m1"]
    assert m.context_window == 1000000          # declaration always wins
    assert m.provenance["context_window"] == "declared"
    assert any("256000" in w for w in cfg.warnings)


def test_detected_fills_when_not_declared(tmp_path):
    cfg = config_mod.load(_write(tmp_path, BASE),
                          detected_cache={"m1": {"context_window": 32768,
                                                 "capabilities": ["text", "tools"]}})
    m = cfg.models["m1"]
    assert m.context_window == 32768
    assert m.provenance["context_window"] == "detected"
    assert m.capabilities == {"text", "tools"}


def test_default_provenance_when_nothing_known(tmp_path):
    cfg = config_mod.load(_write(tmp_path, BASE))
    m = cfg.models["m1"]
    assert m.provenance["context_window"] == "default"
    assert m.capabilities == {"text"}


# ------------------------------------------------------------ capabilities

def _props(body):
    return extract_props(body, Settings())


def test_image_detected_history_wide():
    body = {"messages": [
        {"role": "user", "content": [
            {"type": "text", "text": "look"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,x"}}]},
        {"role": "assistant", "content": "a cat"},
        {"role": "user", "content": "and now?"},
    ]}
    p = _props(body)
    assert p.has_image                     # media at turn 1 still constrains turn N
    assert p.last_user_text == "and now?"


def test_tools_signature_stable_and_order_independent():
    t1 = [{"type": "function", "function": {"name": "a"}},
          {"type": "function", "function": {"name": "b"}}]
    t2 = list(reversed(t1))
    p1 = _props({"messages": [], "tools": t1})
    p2 = _props({"messages": [], "tools": t2})
    assert p1.tools_sig == p2.tools_sig != ""


def test_context_window_filter():
    from hello_operator.config import ModelSpec
    spec = ModelSpec(key="s", id="s", endpoint="e", capabilities={"text"},
                     context_window=1024)
    big = {"messages": [{"role": "user", "content": "x" * 40000}]}
    ok, why = model_ok(spec, _props(big), Settings())
    assert not ok and "context window" in why


def test_vision_filter_names_constraint():
    from hello_operator.config import ModelSpec
    spec = ModelSpec(key="s", id="s", endpoint="e", capabilities={"text"},
                     context_window=8192)
    body = {"messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "u"}}]}]}
    ok, why = model_ok(spec, _props(body), Settings())
    assert not ok and "vision" in why


# ------------------------------------------------------------- escalation

TOOLS = [{"type": "function", "function": {
    "name": "f", "parameters": {"type": "object",
                                "properties": {"x": {"type": "string"}},
                                "required": ["x"]}}}]


def test_validate_bad_json_arguments():
    msg = {"tool_calls": [{"function": {"name": "f", "arguments": "{oops"}}]}
    assert any("not valid JSON" in f for f in validate_tool_calls(TOOLS, msg))


def test_validate_unknown_tool():
    msg = {"tool_calls": [{"function": {"name": "ghost", "arguments": "{}"}}]}
    assert any("unknown tool" in f for f in validate_tool_calls(TOOLS, msg))


def test_validate_missing_required_argument():
    msg = {"tool_calls": [{"function": {"name": "f", "arguments": "{}"}}]}
    assert any("missing required" in f for f in validate_tool_calls(TOOLS, msg))


def test_validate_good_call_passes():
    msg = {"tool_calls": [{"function": {"name": "f",
                                        "arguments": json.dumps({"x": "1"})}}]}
    assert validate_tool_calls(TOOLS, msg) == []


def test_missing_required_call():
    assert missing_required_call({"tool_choice": "required"}, {"content": "hi"})
    assert not missing_required_call({"tool_choice": "auto"}, {"content": "hi"})
    assert not missing_required_call(
        {"tool_choice": "required"},
        {"tool_calls": [{"function": {"name": "f", "arguments": "{}"}}]})


def test_calls_signature_stability():
    a = {"tool_calls": [{"function": {"name": "f", "arguments": '{"x":1}'}}]}
    b = {"tool_calls": [{"function": {"name": "f", "arguments": '{"x":1}'}}]}
    c = {"tool_calls": [{"function": {"name": "f", "arguments": '{"x":2}'}}]}
    assert calls_signature(a) == calls_signature(b) != calls_signature(c)
    assert calls_signature({"content": "hi"}) == ""


def test_stream_collector_reassembles_split_chunks():
    col = StreamCollector()
    chunks = [
        {"choices": [{"delta": {"content": "Hel"}}]},
        {"choices": [{"delta": {"content": "lo"}}]},
        {"choices": [{"delta": {"tool_calls": [
            {"index": 0, "id": "c1", "function": {"name": "f"}}]}}]},
        {"choices": [{"delta": {"tool_calls": [
            {"index": 0, "function": {"arguments": '{"x"'}}]}}]},
        {"choices": [{"delta": {"tool_calls": [
            {"index": 0, "function": {"arguments": ': "1"}'}}]}}]},
        {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
    ]
    blob = b"".join(b"data: " + json.dumps(c).encode() + b"\n\n" for c in chunks)
    # feed in awkward split positions to exercise buffering
    col.feed(blob[:17])
    col.feed(blob[17:60])
    col.feed(blob[60:])
    msg = col.assembled()
    assert msg["content"] == "Hello"
    assert msg["tool_calls"][0]["function"]["arguments"] == '{"x": "1"}'
    assert col.finish_reason == "tool_calls"


# ---------------------------------------------------------------- affinity

def test_session_key_priority():
    s = Settings()
    hdr = {"x-session-id": "abc"}
    assert derive_session_key(hdr, {"session_id": "zzz"}, s) == "hdr:abc"
    assert derive_session_key({}, {"session_id": "zzz"}, s) == "body:session_id:zzz"
    assert derive_session_key({}, {"user": "u1"}, s) == "body:user:u1"
    k1 = derive_session_key({}, {"messages": [{"role": "user", "content": "hi"}]}, s)
    k2 = derive_session_key({}, {"messages": [{"role": "user", "content": "hi"},
                                              {"role": "assistant", "content": "yo"}]}, s)
    assert k1 == k2 and k1.startswith("hist:")   # stable across appended turns
