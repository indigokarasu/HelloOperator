"""Test scaffolding: a scriptable fake OpenAI-compatible backend and a context
manager that stands up fake backends + the router on ephemeral loopback ports.

The fake embedding endpoint is deterministic: a vector of marker-substring
counts, so tests control classification exactly.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import math
import socket
from pathlib import Path

import aiohttp
import yaml
from aiohttp import web

from hello_operator import config as config_mod
from hello_operator.server import build_app

MARKERS = ["code", "function", "bug", "fix", "chat", "hello", "weather", "story", "image"]

CODE_UTTERANCES = ["fix this code bug", "write a function for this code",
                   "debug this function bug fix"]
CHAT_UTTERANCES = ["hello let's chat", "chat about the weather hello",
                   "hello friend chat weather"]


def fake_embedding(text: str) -> list[float]:
    low = text.lower()
    v = [float(low.count(m)) for m in MARKERS] + [0.1]
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def _bound_socket() -> tuple[socket.socket, int]:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("127.0.0.1", 0))
    return s, s.getsockname()[1]


class FakeBackend:
    """One fake inference server. `behaviors` maps backend model id -> a
    callable(body, call_index) -> dict describing the assistant message:
      {"content": "..."} and/or {"tool_calls": [...]}
    The backend honors body["stream"] by emitting SSE chunks.
    """

    def __init__(self, behaviors: dict, native: str = ""):
        self.behaviors = behaviors
        self.native = native            # "", "ollama", "llamacpp", "lmstudio"
        self.calls: list[dict] = []     # every chat body received
        self.per_model_calls: dict[str, int] = {}
        self.runner = None
        self.port = 0

    # ------------------------------------------------------------ handlers

    async def chat(self, request: web.Request) -> web.StreamResponse:
        body = await request.json()
        self.calls.append(body)
        model = body.get("model", "")
        idx = self.per_model_calls.get(model, 0)
        self.per_model_calls[model] = idx + 1
        fn = self.behaviors.get(model)
        if fn is None:
            return web.json_response(
                {"error": {"message": f"unknown model {model}"}}, status=404)
        spec = fn(body, idx)
        if isinstance(spec, web.Response):
            return spec
        message = {"role": "assistant", "content": spec.get("content", "")}
        if spec.get("tool_calls"):
            message["tool_calls"] = spec["tool_calls"]
        finish = "tool_calls" if spec.get("tool_calls") else "stop"

        if body.get("stream"):
            resp = web.StreamResponse()
            resp.headers["Content-Type"] = "text/event-stream"
            await resp.prepare(request)
            deltas = []
            content = message.get("content") or ""
            # split content into two chunks to exercise reassembly
            half = max(1, len(content) // 2)
            if content:
                deltas.append({"content": content[:half]})
                deltas.append({"content": content[half:]})
            for i, tc in enumerate(message.get("tool_calls") or []):
                fn_part = tc.get("function", {})
                args = fn_part.get("arguments", "")
                mid = max(1, len(args) // 2)
                deltas.append({"tool_calls": [{"index": i, "id": tc.get("id", f"c{i}"),
                                               "function": {"name": fn_part.get("name", "")}}]})
                deltas.append({"tool_calls": [{"index": i,
                                               "function": {"arguments": args[:mid]}}]})
                deltas.append({"tool_calls": [{"index": i,
                                               "function": {"arguments": args[mid:]}}]})
            for d in deltas:
                chunk = {"id": "cc1", "object": "chat.completion.chunk", "model": model,
                         "choices": [{"index": 0, "delta": d, "finish_reason": None}]}
                await resp.write(b"data: " + json.dumps(chunk).encode() + b"\n\n")
            done = {"id": "cc1", "object": "chat.completion.chunk", "model": model,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": finish}]}
            await resp.write(b"data: " + json.dumps(done).encode() + b"\n\n")
            await resp.write(b"data: [DONE]\n\n")
            await resp.write_eof()
            return resp

        return web.json_response({
            "id": "cc1", "object": "chat.completion", "model": model,
            "choices": [{"index": 0, "message": message, "finish_reason": finish}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        })

    async def models(self, request: web.Request) -> web.Response:
        return web.json_response({"object": "list", "data": [
            {"id": mid, "object": "model"} for mid in self.behaviors]})

    async def embeddings(self, request: web.Request) -> web.Response:
        body = await request.json()
        inputs = body["input"]
        if isinstance(inputs, str):
            inputs = [inputs]
        return web.json_response({"object": "list", "data": [
            {"object": "embedding", "index": i, "embedding": fake_embedding(t)}
            for i, t in enumerate(inputs)]})

    async def ollama_show(self, request: web.Request) -> web.Response:
        body = await request.json()
        if body.get("name") not in self.behaviors:
            return web.json_response({}, status=404)
        return web.json_response({
            "capabilities": ["completion", "tools"],
            "model_info": {"general.architecture": "llama",
                           "llama.context_length": 32768},
            "details": {"family": "llama", "quantization_level": "Q4_K_M"},
        })

    async def ollama_tags(self, request: web.Request) -> web.Response:
        return web.json_response({"models": [
            {"name": mid, "digest": f"digest-{mid}"} for mid in self.behaviors]})

    async def llama_props(self, request: web.Request) -> web.Response:
        return web.json_response({
            "default_generation_settings": {"n_ctx": 16384},
            "model_path": "/models/test.gguf",
            "modalities": {"vision": False},
        })

    # ------------------------------------------------------------ lifecycle

    async def start(self) -> str:
        app = web.Application()
        app.router.add_post("/v1/chat/completions", self.chat)
        app.router.add_get("/v1/models", self.models)
        app.router.add_post("/v1/embeddings", self.embeddings)
        if self.native == "ollama":
            app.router.add_post("/api/show", self.ollama_show)
            app.router.add_get("/api/tags", self.ollama_tags)
        elif self.native == "llamacpp":
            app.router.add_get("/props", self.llama_props)
        self.runner = web.AppRunner(app)
        await self.runner.setup()
        sock, self.port = _bound_socket()
        site = web.SockSite(self.runner, sock)
        await site.start()
        return f"http://127.0.0.1:{self.port}/v1"

    async def stop(self) -> None:
        if self.runner:
            await self.runner.cleanup()


class RouterEnv:
    """Fake backend(s) + router server, all on ephemeral loopback ports."""

    def __init__(self, tmp_path: Path, config_dict: dict, backend: FakeBackend):
        self.tmp_path = tmp_path
        self.config_dict = config_dict
        self.backend = backend
        self.router_runner = None
        self.base = ""
        self.client: aiohttp.ClientSession | None = None

    async def __aenter__(self) -> "RouterEnv":
        backend_base = await self.backend.start()
        text = yaml.safe_dump(self.config_dict).replace("BACKEND", backend_base)
        cfg_path = self.tmp_path / "config.yaml"
        cfg_path.write_text(text)
        cfg = config_mod.load(str(cfg_path))
        app = build_app(cfg)
        self.router_runner = web.AppRunner(app)
        await self.router_runner.setup()
        sock, port = _bound_socket()
        site = web.SockSite(self.router_runner, sock)
        await site.start()
        self.base = f"http://127.0.0.1:{port}"
        self.client = aiohttp.ClientSession()
        return self

    async def __aexit__(self, *exc) -> None:
        if self.client:
            await self.client.close()
        if self.router_runner:
            await self.router_runner.cleanup()
        await self.backend.stop()

    async def chat(self, messages, session: str = "", stream: bool = False,
                   tools=None, tool_choice=None, headers: dict | None = None,
                   model: str = "hello-operator", **extra):
        body: dict = {"model": model, "messages": messages, "stream": stream}
        if tools is not None:
            body["tools"] = tools
        if tool_choice is not None:
            body["tool_choice"] = tool_choice
        body.update(extra)
        hdrs = dict(headers or {})
        if session:
            hdrs["x-session-id"] = session
        assert self.client is not None
        async with self.client.post(f"{self.base}/v1/chat/completions",
                                    json=body, headers=hdrs) as resp:
            raw = await resp.read()
            out_headers = dict(resp.headers)
            status = resp.status
        if stream and status == 200 and b"data:" in raw:
            payload = _assemble_sse(raw)
        else:
            payload = json.loads(raw) if raw else {}
        return status, payload, out_headers


def _assemble_sse(raw: bytes) -> dict:
    content = []
    tool_calls: dict[int, dict] = {}
    model = ""
    for line in raw.split(b"\n"):
        line = line.strip()
        if not line.startswith(b"data:") or line[5:].strip() == b"[DONE]":
            continue
        obj = json.loads(line[5:])
        model = obj.get("model", model)
        for ch in obj.get("choices", []):
            delta = ch.get("delta") or {}
            if delta.get("content"):
                content.append(delta["content"])
            for tc in delta.get("tool_calls") or []:
                slot = tool_calls.setdefault(tc.get("index", 0),
                                             {"function": {"name": "", "arguments": ""}})
                fnp = tc.get("function") or {}
                if fnp.get("name"):
                    slot["function"]["name"] = fnp["name"]
                if fnp.get("arguments"):
                    slot["function"]["arguments"] += fnp["arguments"]
    msg: dict = {"role": "assistant", "content": "".join(content)}
    if tool_calls:
        msg["tool_calls"] = [tool_calls[i] for i in sorted(tool_calls)]
    return {"model": model, "choices": [{"message": msg}]}


def base_config(tmp_path: Path, *, models: dict, roles: dict, default_role: str,
                embedding: bool = True, **router_extra) -> dict:
    router = {"listen": "127.0.0.1:0", "state_dir": str(tmp_path / "state"),
              "decision_log": str(tmp_path / "decisions.jsonl")}
    router.update(router_extra)
    cfg: dict = {"router": router, "models": models, "roles": roles,
                 "routing": {"default_role": default_role}}
    if embedding:
        cfg["embedding"] = {"id": "fake-embed", "endpoint": "BACKEND"}
    return cfg


WEATHER_TOOL = [{"type": "function", "function": {
    "name": "get_weather",
    "parameters": {"type": "object",
                   "properties": {"city": {"type": "string"}},
                   "required": ["city"]}}}]


def good_call(body, idx):
    return {"tool_calls": [{"id": "a1", "type": "function", "function": {
        "name": "get_weather", "arguments": json.dumps({"city": "Paris"})}}]}


def bad_json_call(body, idx):
    return {"tool_calls": [{"id": "a1", "type": "function", "function": {
        "name": "get_weather", "arguments": '{"city": "Par'}}]}


def no_call(body, idx):
    return {"content": "I would rather describe the weather in prose."}


def run(coro):
    return asyncio.run(coro)
