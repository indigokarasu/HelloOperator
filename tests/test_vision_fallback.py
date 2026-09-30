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
