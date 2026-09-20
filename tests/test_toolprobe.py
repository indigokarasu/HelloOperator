"""The ranking's tool probe.

Tool use is the capability an agent's model is useless without, and the one the
probe suite never tested. A model that answers prose where a tool call belongs
returns HTTP 200 while doing it, so nothing downstream notices.
"""
import sys
from pathlib import Path

import aiohttp

sys.path.insert(0, str(Path(__file__).parent))

from helpers import FakeBackend, bad_json_call, good_call, no_call, run

from hello_operator.ranking import _probe_tools

CFG = {"probe_max_tokens": 128, "request_timeout_s": 30}


def _score(behaviors, model_id="m"):
    async def scenario():
        backend = FakeBackend(behaviors)
        base = await backend.start()
        try:
            async with aiohttp.ClientSession() as http:
                return await _probe_tools(http, base, "", model_id, CFG)
        finally:
            await backend.stop()
    return run(scenario())


def _two_stage(final_text):
    """First call: the tool call. Second: the model's answer given the result."""
    def behave(body, idx):
        return good_call(body, idx) if idx == 0 else {"content": final_text}
    return behave


def test_model_that_completes_the_whole_loop_scores_2():
    assert _score({"m": _two_stage("It is 18C and raining in Paris.")}) == 2


def test_model_that_calls_but_ignores_the_result_scores_1():
    """Emitting a call is half the job; an agent loop needs the follow-up too."""
    assert _score({"m": _two_stage("I have no idea what the weather is.")}) == 1


def test_model_that_answers_in_prose_scores_0():
    assert _score({"m": no_call}) == 0


def test_model_that_emits_unparseable_arguments_scores_0():
    assert _score({"m": bad_json_call}) == 0


def test_model_that_calls_the_wrong_tool_scores_0():
    def wrong(body, idx):
        return {"tool_calls": [{"id": "a1", "type": "function", "function": {
            "name": "search_web", "arguments": '{"q": "paris weather"}'}}]}
    assert _score({"m": wrong}) == 0


def test_model_that_omits_the_required_argument_scores_0():
    def empty_args(body, idx):
        return {"tool_calls": [{"id": "a1", "type": "function", "function": {
            "name": "get_weather", "arguments": "{}"}}]}
    assert _score({"m": empty_args}) == 0


def test_unreachable_model_scores_0_instead_of_raising():
    """A dead model must not break the nightly ranking run."""
    assert _score({"other": good_call}, model_id="missing") == 0


def test_the_probe_actually_sends_the_tool_schema():
    """Guards against the probe silently degrading into a plain chat request."""
    async def scenario():
        backend = FakeBackend({"m": _two_stage("18C, rain")})
        base = await backend.start()
        try:
            async with aiohttp.ClientSession() as http:
                await _probe_tools(http, base, "", "m", CFG)
        finally:
            await backend.stop()
        return backend.calls
    calls = run(scenario())
    assert calls, "probe sent nothing"
    assert calls[0].get("tools"), "first probe request carried no tool schema"
    assert calls[0]["tools"][0]["function"]["name"] == "get_weather"
    # the follow-up must replay the tool result back to the model
    assert len(calls) == 2
    roles = [m.get("role") for m in calls[1]["messages"]]
    assert "tool" in roles, f"follow-up never sent a tool result: {roles}"
