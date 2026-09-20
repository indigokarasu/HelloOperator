"""A cascade exists so one backend's problem is not the turn's problem.

402 and 429 already failed over. 401/403/404 did not, so a stale credential or a
withdrawn model failed the whole turn with ten healthy free models behind it.
"""
import sys
from pathlib import Path

from aiohttp import web

sys.path.insert(0, str(Path(__file__).parent))

from helpers import CHAT_UTTERANCES, FakeBackend, RouterEnv, base_config, run


def _echo(name):
    return lambda body, idx: {"content": f"answer from {name}"}


def _status(code, message):
    def behave(body, idx):
        return web.json_response({"error": {"message": message, "code": code}}, status=code)
    return behave


def _cfg(tmp_path):
    models = {"first": {"id": "vendor/first:free", "endpoint": "BACKEND",
                        "capabilities": ["text", "tools", "json"], "context_window": 131072},
              "second": {"id": "vendor/second:free", "endpoint": "BACKEND",
                         "capabilities": ["text", "tools", "json"], "context_window": 131072}}
    roles = {"chat": {"cascade": ["first", "second"], "utterances": CHAT_UTTERANCES}}
    return base_config(tmp_path, models=models, roles=roles, default_role="chat")


def _ask(tmp_path, first_behaviour):
    async def scenario():
        backend = FakeBackend({"vendor/first:free": first_behaviour,
                               "vendor/second:free": _echo("second")})
        async with RouterEnv(tmp_path, _cfg(tmp_path), backend) as env:
            return await env.chat([{"role": "user", "content": "hello chat"}])
    return run(scenario())


def test_401_fails_over_to_the_next_model():
    """A stale credential on one endpoint must not fail the turn."""


def test_stale_credential_401_fails_over(tmp_path):
    status, payload, _ = _ask(tmp_path, _status(401, "User not found."))
    assert status == 200, payload
    assert payload["router"]["backend_model"] == "vendor/second:free"


def test_withdrawn_model_404_fails_over(tmp_path):
    status, payload, _ = _ask(tmp_path, _status(404, "This model's free period has ended."))
    assert status == 200, payload
    assert payload["router"]["backend_model"] == "vendor/second:free"


def test_forbidden_403_fails_over(tmp_path):
    status, payload, _ = _ask(tmp_path, _status(403, "Forbidden"))
    assert status == 200, payload
    assert payload["router"]["backend_model"] == "vendor/second:free"


def test_a_bad_request_400_still_fails_fast(tmp_path):
    """400 means WE sent something wrong; retrying it on every model in the
    cascade would just repeat the same mistake N times and hide the cause."""
    status, payload, _ = _ask(tmp_path, _status(400, "Invalid option: reasoning_effort"))
    assert status != 200
    assert "second" not in str(payload), payload
