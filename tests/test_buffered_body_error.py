"""Errors a backend puts in a BUFFERED body after answering 200 (incident
2026-09-30).

OpenRouter answers 200 and then sends {"error": {"code": 502, ...}} as the JSON
body when the provider behind a model fails. _forward_stream already fails over
on that shape (2026-09-25); the buffered path did not, so it was served as a
valid completion with choices=None and every auxiliary consumer raised
"missing choices[0].message". Lanes without a fallback_chain (tools.approval,
tools.vision_tools) refused deterministically instead of routing around it.
"""
import sys
from pathlib import Path

from aiohttp import web

sys.path.insert(0, str(Path(__file__).parent))

from helpers import CHAT_UTTERANCES, FakeBackend, RouterEnv, base_config, run


def _echo(name):
    return lambda body, idx: {"content": f"answer from {name}"}


def _errored_body(code, message):
    """A provider that answers 200 with an error object instead of choices."""
    def behave(body, idx):
        return web.json_response({"id": "gen-err", "choices": None, "created": None,
                                  "model": None, "error": {"code": code,
                                                          "message": message}})
    return behave


def _cfg(tmp_path):
    models = {"first": {"id": "vendor/first:free", "endpoint": "BACKEND",
                        "capabilities": ["text"], "context_window": 131072},
              "second": {"id": "vendor/second:free", "endpoint": "BACKEND",
                         "capabilities": ["text"], "context_window": 131072}}
    roles = {"chat": {"cascade": ["first", "second"], "utterances": CHAT_UTTERANCES}}
    return base_config(tmp_path, models=models, roles=roles, default_role="chat")


def _ask(tmp_path, first_behaviour):
    async def scenario():
        backend = FakeBackend({"vendor/first:free": first_behaviour,
                               "vendor/second:free": _echo("second")})
        async with RouterEnv(tmp_path, _cfg(tmp_path), backend) as env:
            return await env.chat([{"role": "user", "content": "hello chat"}])
    return run(scenario())


def test_buffered_200_with_error_body_fails_over(tmp_path):
    """choices=None from a 200 must not be served; the cascade moves on."""
    status, payload, _ = _ask(tmp_path, _errored_body(503, "no provider available"))
    assert status == 200, payload
    assert payload["router"]["backend_model"] == "vendor/second:free", payload
    assert payload["choices"][0]["message"]["content"] == "answer from second"


def test_buffered_error_body_without_code_fails_over(tmp_path):
    """An error object with no usable code is still a failure, not a completion."""
    status, payload, _ = _ask(tmp_path, _errored_body("nonsense", "upstream exploded"))
    assert status == 200, payload
    assert payload["router"]["backend_model"] == "vendor/second:free", payload


def test_buffered_clean_body_is_unaffected(tmp_path):
    """A normal completion carries no error key, so nothing changes."""
    status, payload, _ = _ask(tmp_path, _echo("first"))
    assert status == 200
    assert payload["router"]["backend_model"] == "vendor/first:free"