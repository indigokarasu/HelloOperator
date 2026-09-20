"""Per-backend parameter normalisation.

Hermes sent reasoning_effort="medium"; OpenRouter accepts that string but
DeepSeek V4.1 rejects it, so the designated paid fallback answered HTTP 400
220 times and never served. These tests assert on the body the BACKEND
receives, not merely on a 200, because a passing status would not prove the
value was actually rewritten.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from helpers import CHAT_UTTERANCES, FakeBackend, RouterEnv, base_config, run


def _echo(body, idx):
    return {"content": "ok"}


def _cfg(tmp_path, model_extra=None):
    models = {"m": {"id": "vendor/model", "endpoint": "BACKEND",
                    "capabilities": ["text", "tools", "json"], "context_window": 131072}}
    if model_extra:
        models["m"].update(model_extra)
    models["other"] = {"id": "vendor/other", "endpoint": "BACKEND",
                       "capabilities": ["text", "tools", "json"], "context_window": 131072}
    roles = {"chat": {"cascade": ["m", "other"], "utterances": CHAT_UTTERANCES}}
    return base_config(tmp_path, models=models, roles=roles, default_role="chat")


def _send(tmp_path, cfg, **extra):
    async def scenario():
        backend = FakeBackend({"vendor/model": _echo, "vendor/other": _echo})
        async with RouterEnv(tmp_path, cfg, backend) as env:
            await env.chat([{"role": "user", "content": "hello chat"}], **extra)
        return backend.calls
    return run(scenario())


def test_mapped_value_is_rewritten_on_the_wire(tmp_path):
    cfg = _cfg(tmp_path, {"param_map": {"reasoning_effort": {"medium": "low"}}})
    calls = _send(tmp_path, cfg, reasoning_effort="medium")
    assert calls, "no request reached the backend"
    assert calls[0].get("reasoning_effort") == "low", calls[0].get("reasoning_effort")


def test_value_mapped_to_null_is_dropped_entirely(tmp_path):
    cfg = _cfg(tmp_path, {"param_map": {"reasoning_effort": {"none": None}}})
    calls = _send(tmp_path, cfg, reasoning_effort="none")
    assert "reasoning_effort" not in calls[0], calls[0]


def test_unmapped_value_passes_through_untouched(tmp_path):
    """A value the backend accepts must not be rewritten."""
    cfg = _cfg(tmp_path, {"param_map": {"reasoning_effort": {"medium": "low"}}})
    calls = _send(tmp_path, cfg, reasoning_effort="high")
    assert calls[0].get("reasoning_effort") == "high"


def test_drop_params_removes_a_rejected_parameter(tmp_path):
    cfg = _cfg(tmp_path, {"drop_params": ["reasoning_effort"]})
    calls = _send(tmp_path, cfg, reasoning_effort="medium")
    assert "reasoning_effort" not in calls[0], calls[0]


def test_a_model_without_normalisation_is_untouched(tmp_path):
    """Control: proves the rewrites above come from the config, not the router."""
    calls = _send(tmp_path, _cfg(tmp_path), reasoning_effort="medium")
    assert calls[0].get("reasoning_effort") == "medium"


def test_normalisation_does_not_disturb_the_rest_of_the_body(tmp_path):
    cfg = _cfg(tmp_path, {"param_map": {"reasoning_effort": {"medium": "low"}}})
    calls = _send(tmp_path, cfg, reasoning_effort="medium", temperature=0.3)
    assert calls[0].get("temperature") == 0.3
    assert calls[0].get("model") == "vendor/model"
    assert calls[0].get("messages")
