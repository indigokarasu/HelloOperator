"""A stop must finish in seconds, not hang until systemd's SIGKILL.

The router is restarted several times a day by design (the Nous token rotating,
the daily re-rank), and every restart is an outage for every Hermes profile.
aiohttp's run_app waited shutdown_timeout=60 s for each in-flight handler, then
waited another 60 s without cancelling it, so a stream with more than ~90 s left
held `systemctl stop` past TimeoutStopSec until SIGKILL: 21 of 165 stops in two
weeks, measured locally at 121.5 s. A stop now drains in-flight turns for
router.shutdown_drain_s and then cuts what is left.
"""
import asyncio
import signal
import socket
import sys
import time
from pathlib import Path

import aiohttp
import yaml

sys.path.insert(0, str(Path(__file__).parent))

from helpers import CHAT_UTTERANCES, FakeBackend, RouterEnv, base_config, run


def _slow_stream(n, delay):
    return lambda body, idx: {"chunks": [f"t{i} " for i in range(n)],
                              "chunk_delay_s": delay}


def _cfg(tmp_path, **router_extra):
    models = {"only": {"id": "vendor/slow:free", "endpoint": "BACKEND",
                       "capabilities": ["text"], "context_window": 32768}}
    roles = {"chat": {"cascade": ["only"], "utterances": CHAT_UTTERANCES}}
    return base_config(tmp_path, models=models, roles=roles, default_role="chat",
                       **router_extra)


def _body():
    return {"model": "hello-operator", "stream": True,
            "messages": [{"role": "user", "content": "hello chat"}]}


async def _read_rest(resp):
    """Everything after the first chunk, and whether the body ended cleanly."""
    out = []
    try:
        async for c in resp.content.iter_any():
            out.append(c)
        return b"".join(out), "eof"
    except aiohttp.ClientPayloadError as e:
        return b"".join(out), f"cut: {e}"


def test_stop_cuts_a_stream_that_outlives_the_drain(tmp_path):
    """A generation still running when the drain ends is cut, and the stop
    returns promptly instead of waiting for it."""
    async def scenario():
        backend = FakeBackend({"vendor/slow:free": _slow_stream(100, 0.2)})
        async with RouterEnv(tmp_path, _cfg(tmp_path, shutdown_drain_s=0.5),
                             backend) as env:
            async with env.client.post(f"{env.base}/v1/chat/completions",
                                       json=_body()) as resp:
                assert resp.status == 200
                first = await resp.content.readany()
                assert b"t0" in first
                reader = asyncio.create_task(_read_rest(resp))
                runner, env.router_runner = env.router_runner, None
                t0 = time.monotonic()
                await asyncio.wait_for(runner.cleanup(), 10)
                elapsed = time.monotonic() - t0
                rest, how = await asyncio.wait_for(reader, 5)
            return elapsed, rest, how

    elapsed, rest, how = run(scenario())
    assert elapsed < 3, f"stop took {elapsed:.1f}s with a 0.5 s drain"
    # The client must be able to tell the turn was cut. A clean end of body
    # without [DONE] would read as a finished, silently truncated answer.
    assert how.startswith("cut"), how
    assert b"[DONE]" not in rest


def test_stop_lets_a_stream_finish_inside_the_drain(tmp_path):
    """A turn that completes within the drain is delivered whole; the stop
    waits for it rather than cutting a healthy stream."""
    async def scenario():
        backend = FakeBackend({"vendor/slow:free": _slow_stream(5, 0.2)})
        async with RouterEnv(tmp_path, _cfg(tmp_path, shutdown_drain_s=5),
                             backend) as env:
            async with env.client.post(f"{env.base}/v1/chat/completions",
                                       json=_body()) as resp:
                first = await resp.content.readany()
                reader = asyncio.create_task(_read_rest(resp))
                runner, env.router_runner = env.router_runner, None
                t0 = time.monotonic()
                await asyncio.wait_for(runner.cleanup(), 10)
                elapsed = time.monotonic() - t0
                rest, how = await asyncio.wait_for(reader, 5)
            return elapsed, first + rest, how

    elapsed, raw, how = run(scenario())
    assert how == "eof", how
    assert b"data: [DONE]" in raw
    for i in range(5):
        assert f"t{i} ".encode() in raw
    assert elapsed < 4, f"stop took {elapsed:.1f}s for a 1 s stream"


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def test_sigterm_mid_stream_exits_within_seconds(tmp_path):
    """The production path: the real CLI under web.run_app, stopped by SIGTERM
    (what systemd sends) while a stream is in flight."""
    async def scenario():
        backend = FakeBackend({"vendor/slow:free": _slow_stream(300, 0.1)})
        backend_base = await backend.start()
        port = _free_port()
        cfg = _cfg(tmp_path, shutdown_drain_s=0.5)
        cfg["router"]["listen"] = f"127.0.0.1:{port}"
        path = tmp_path / "config.yaml"
        path.write_text(yaml.safe_dump(cfg).replace("BACKEND", backend_base))
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-m", "hello_operator", "-c", str(path),
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
        base = f"http://127.0.0.1:{port}"
        try:
            async with aiohttp.ClientSession() as http:
                for _ in range(200):
                    try:
                        async with http.get(f"{base}/healthz") as r:
                            if r.status == 200:
                                break
                    except aiohttp.ClientError:
                        await asyncio.sleep(0.05)
                async with http.post(f"{base}/v1/chat/completions",
                                     json=_body()) as resp:
                    assert resp.status == 200
                    await resp.content.readany()
                    reader = asyncio.create_task(_read_rest(resp))
                    t0 = time.monotonic()
                    proc.send_signal(signal.SIGTERM)
                    try:
                        rc = await asyncio.wait_for(proc.wait(), 15)
                    except asyncio.TimeoutError:
                        rc = None
                        proc.kill()
                        await proc.wait()
                    elapsed = time.monotonic() - t0
                    _, how = await asyncio.wait_for(reader, 5)
        finally:
            if proc.returncode is None:
                proc.kill()
                await proc.wait()
            assert proc.stdout is not None
            out = (await proc.stdout.read()).decode(errors="replace")
            await backend.stop()
        return rc, elapsed, how, out

    rc, elapsed, how, out = run(scenario())
    assert rc is not None, f"router still running 15 s after SIGTERM\n{out}"
    assert elapsed < 5, f"SIGTERM took {elapsed:.1f}s with a 0.5 s drain\n{out}"
    assert rc == 0, out
    assert how.startswith("cut"), how
    assert "Traceback" not in out, out
    assert "Unclosed client session" not in out, out
