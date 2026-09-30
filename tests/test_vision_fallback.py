"""A rate-limited free vision model must fall over to the local one.

Owner directive 2026-09-29: trying free vision models first is fine, but a
screenshot must then land on the local model rather than fail. Hermes'
vision_analyze sends one user message (text + image) and no max_tokens. The
router's assumed 1024-token completion ruled out the local 2048-token moondream,
so the vision role had no eligible model, the image went to the one free vision
model in the chat cascade, and its 429 was the whole answer: "no backend could
serve the request".
"""
import sys
from pathlib import Path

from aiohttp import web

sys.path.insert(0, str(Path(__file__).parent))

from helpers import CHAT_UTTERANCES, FakeBackend, RouterEnv, base_config, run


def _rate_limited(body, idx):
    return web.json_response({"error": {"message": "Rate limit exceeded: free-models-per-day",
                                        "code": 429}}, status=429)


def _upstream_limited(body, idx):
    """OpenRouter when the provider behind ONE free model is throttling it."""
    return web.json_response({"error": {"message": "Provider returned error", "code": 429,
        "metadata": {"raw": "vendor/eye:free is temporarily rate-limited upstream. Please "
                            "retry shortly, or add your own key to accumulate your rate limits",
                     "provider_name": "Some Provider"}}}, status=429)


def _nous_fair_share(body, idx):
    return web.json_response({"status": 429, "reason": "rate_limited", "retry_after": 237673,
        "message": "You've reached this model's current fair-share rate limit. It adapts "
                   "to demand — retry after the indicated delay, or try an alternate model."},
        status=429)


def _cfg(tmp_path):
    models = {
        "text": {"id": "vendor/text:free", "endpoint": "BACKEND",
                 "capabilities": ["text", "tools"], "context_window": 131072},
        "free-eye": {"id": "vendor/eye:free", "endpoint": "BACKEND",
                     "capabilities": ["text", "vision"], "context_window": 131072},
        "moondream": {"id": "moondream", "endpoint": "BACKEND",
                      "capabilities": ["text", "vision"], "context_window": 2048},
    }
    roles = {
        "chat": {"cascade": ["text", "free-eye"], "utterances": CHAT_UTTERANCES},
        "vision": {"cascade": ["free-eye", "moondream"], "requires": ["vision"],
                   "utterances": ["describe this screenshot", "what is in this image"]},
    }
    return base_config(tmp_path, models=models, roles=roles, default_role="chat")


def _vision_call():
    """What Hermes' vision_analyze sends: no max_tokens, one image."""
    return [{"role": "user", "content": [
        {"type": "text", "text": "Describe this page in detail. Are card images visible "
                                 "or blank? Is the filter bar visible at the top?"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}}]}]


def test_rate_limited_free_vision_falls_over_to_local_model(tmp_path):
    async def scenario():
        backend = FakeBackend({
            "vendor/text:free": lambda b, i: {"content": "text model"},
            "vendor/eye:free": _rate_limited,
            "moondream": lambda b, i: {"content": "a newspaper-style events page"},
        })
        async with RouterEnv(tmp_path, _cfg(tmp_path), backend) as env:
            status, payload, headers = await env.chat(_vision_call())
            tried = [c["model"] for c in backend.calls]
        return status, payload, headers, tried

    status, payload, headers, tried = run(scenario())
    assert status == 200, payload
    assert headers["x-router-model"] == "moondream"
    assert payload["choices"][0]["message"]["content"] == "a newspaper-style events page"
    # Free first, local last.
    assert tried.index("vendor/eye:free") < tried.index("moondream")
    assert "vendor/text:free" not in tried


def _two_free_eyes(tmp_path):
    cfg = _cfg(tmp_path)
    cfg["models"]["free-eye-b"] = {"id": "vendor/eye-b:free", "endpoint": "BACKEND",
                                   "capabilities": ["text", "vision"],
                                   "context_window": 131072}
    cfg["roles"]["vision"]["cascade"] = ["free-eye", "free-eye-b", "moondream"]
    return cfg


def _ask_two_eyes(tmp_path, first_behaviour):
    async def scenario():
        backend = FakeBackend({
            "vendor/text:free": lambda b, i: {"content": "text model"},
            "vendor/eye:free": first_behaviour,
            "vendor/eye-b:free": lambda b, i: {"content": "photos and a FILTERS button"},
            "moondream": lambda b, i: {"content": "no card images visible"},
        })
        async with RouterEnv(tmp_path, _two_free_eyes(tmp_path), backend) as env:
            status, payload, headers = await env.chat(_vision_call())
            return status, headers, [c["model"] for c in backend.calls]
    return run(scenario())


def test_one_free_model_throttled_upstream_does_not_retire_its_siblings(tmp_path):
    """A 429 that says the limit is that MODEL's (its provider is throttling it)
    is not the account's free quota running out. Treating it as such skipped
    every other free model on the key for 15 minutes, so a throttled gemma sent
    screenshots to the local model while a free model that reads them well sat
    one position behind it."""
    for behaviour in (_upstream_limited, _nous_fair_share):
        status, headers, tried = _ask_two_eyes(tmp_path, behaviour)
        assert status == 200
        assert headers["x-router-model"] == "free-eye-b", (behaviour.__name__, tried)
        assert "moondream" not in tried


def test_account_free_cap_still_skips_the_other_free_models(tmp_path):
    """The account-wide daily cap is what the breaker is for: every free model
    on that key answers the same way, so they are skipped, not tried one by one."""
    status, headers, tried = _ask_two_eyes(tmp_path, _rate_limited)
    assert status == 200
    assert headers["x-router-model"] == "moondream"
    assert "vendor/eye-b:free" not in tried
