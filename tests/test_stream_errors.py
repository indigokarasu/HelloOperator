"""Errors a backend streams after answering 200 (incident 2026-09-25).

OpenRouter answers 200 and then sends {"error": {"code": 502, "message": "JSON
error injected into SSE stream", "metadata": {"error_type":
"provider_unavailable"}}} when the provider behind a model fails. Relayed, the
client got a failed turn, and affinity kept every retry on the same backend.
"""
import asyncio
import json
import sys
from pathlib import Path

from aiohttp import web

sys.path.insert(0, str(Path(__file__).parent))

from helpers import CHAT_UTTERANCES, FakeBackend, RouterEnv, base_config, run  # noqa: E402

OR_ERROR = (b'data: {"error": {"code": 502, "message": "JSON error injected into SSE stream",'
            b' "metadata": {"error_type": "provider_unavailable"}}}\n\n')


def _chunk(text):
    return (b"data: " + json.dumps({"choices": [{"index": 0, "delta": {"content": text},
                                                  "finish_reason": None}]}).encode() + b"\n\n")


class RawSSEBackend(FakeBackend):
    """Models listed in `raw` stream exactly the given (delay_s, bytes) pieces."""

    def __init__(self, behaviors, raw):
        super().__init__(behaviors)
        self.raw = raw

    async def chat(self, request):
        body = await request.json()
        pieces = self.raw.get(body.get("model"))
        if pieces is None:
            return await super().chat(request)
        self.calls.append(body)
        resp = web.StreamResponse()
        resp.headers["Content-Type"] = "text/event-stream"
        await resp.prepare(request)
        for delay, data in pieces:
            if delay:
                await asyncio.sleep(delay)
            await resp.write(data)
        await resp.write_eof()
        return resp


def _cfg(tmp_path, **router):
    models = {"stealth": {"id": "vendor/stealth", "endpoint": "BACKEND", "free": True,
                          "capabilities": ["text", "tools"], "context_window": 131072},
              "backup": {"id": "vendor/backup:free", "endpoint": "BACKEND",
                         "capabilities": ["text", "tools"], "context_window": 131072}}
    roles = {"chat": {"cascade": ["stealth", "backup"], "utterances": CHAT_UTTERANCES}}
    return base_config(tmp_path, models=models, roles=roles, default_role="chat", **router)


def _decisions(tmp_path):
    return [json.loads(l) for l in (tmp_path / "decisions.jsonl").read_text().splitlines()]


def test_error_as_first_event_fails_over_before_anything_is_sent(tmp_path):
    backend = RawSSEBackend({"vendor/backup:free": lambda b, i: {"content": "from backup"}},
                            {"vendor/stealth": [(0, b": OPENROUTER PROCESSING\n\n"),
                                                (0, OR_ERROR)]})

    async def scenario():
        async with RouterEnv(tmp_path, _cfg(tmp_path), backend) as env:
            return await env.chat([{"role": "user", "content": "hello chat"}],
                                  session="s", stream=True)

    status, payload, _ = run(scenario())
    assert status == 200
    assert payload["choices"][0]["message"]["content"] == "from backup", payload
    last = _decisions(tmp_path)[-1]
    assert last["model"] == "backup" and last["decision"] == "failover", last


def test_error_after_content_moves_the_next_turn_off_the_backend(tmp_path):
    """Committed turns cannot fail over, but the session must not stay put."""
    backend = RawSSEBackend({"vendor/backup:free": lambda b, i: {"content": "from backup"}},
                            {"vendor/stealth": [(0, _chunk("partial ")), (0, OR_ERROR)]})

    async def scenario():
        async with RouterEnv(tmp_path, _cfg(tmp_path), backend) as env:
            first = await env.chat([{"role": "user", "content": "hello chat"}],
                                   session="s", stream=True)
            second = await env.chat([{"role": "user", "content": "hello chat again"}],
                                    session="s", stream=True)
        return first, second

    first, second = run(scenario())
    assert first[0] == 200     # committed: the partial answer was already relayed
    assert second[1]["choices"][0]["message"]["content"] == "from backup", second
    last = _decisions(tmp_path)[-1]
    assert last["model"] == "backup" and last["decision"].startswith("escalation"), last


def test_slow_first_event_is_relayed_once_the_peek_window_ends(tmp_path):
    """The hold is bounded: a backend that is only slow still streams."""
    backend = RawSSEBackend({}, {"vendor/stealth": [(0.4, _chunk("slow but fine")),
                                                    (0, b"data: [DONE]\n\n")]})

    async def scenario():
        async with RouterEnv(tmp_path, _cfg(tmp_path, stream_peek_s=0.1), backend) as env:
            return await env.chat([{"role": "user", "content": "hello chat"}],
                                  session="s", stream=True)

    status, payload, _ = run(scenario())
    assert status == 200 and payload["choices"][0]["message"]["content"] == "slow but fine"


def test_normal_stream_is_unchanged(tmp_path):
    backend = RawSSEBackend({}, {"vendor/stealth": [(0, _chunk("hel")), (0, _chunk("lo")),
                                                    (0, b"data: [DONE]\n\n")]})

    async def scenario():
        async with RouterEnv(tmp_path, _cfg(tmp_path), backend) as env:
            return await env.chat([{"role": "user", "content": "hello chat"}],
                                  session="s", stream=True)

    status, payload, _ = run(scenario())
    assert status == 200 and payload["choices"][0]["message"]["content"] == "hello"
    assert _decisions(tmp_path)[-1]["model"] == "stealth"
