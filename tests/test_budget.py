"""The spend ceiling, enforced where the money is actually spent.

Each test puts the PAID model first in the cascade, so a pass cannot be confused
with "the free one was chosen anyway".
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from helpers import CHAT_UTTERANCES, FakeBackend, RouterEnv, base_config, run

from hello_operator.budget import Budget, estimate_usd


class _Spec:
    def __init__(self, pin=0.0, pout=0.0):
        self.price_in, self.price_out = pin, pout


def _echo(name):
    return lambda body, idx: {"content": f"answer from {name}"}


def _cfg(tmp_path, **router):
    models = {
        # paid FIRST in the cascade on purpose
        "paid": {"id": "vendor/premium", "endpoint": "BACKEND",
                 "capabilities": ["text", "tools", "json"], "context_window": 131072,
                 "price_in": 10.0, "price_out": 50.0},
        "free": {"id": "vendor/cheap:free", "endpoint": "BACKEND",
                 "capabilities": ["text", "tools", "json"], "context_window": 131072},
    }
    roles = {"chat": {"cascade": ["paid", "free"], "utterances": CHAT_UTTERANCES}}
    return base_config(tmp_path, models=models, roles=roles, default_role="chat", **router)


def _ask(tmp_path, cfg):
    async def scenario():
        backend = FakeBackend({"vendor/premium": _echo("premium"),
                               "vendor/cheap:free": _echo("cheap")})
        async with RouterEnv(tmp_path, cfg, backend) as env:
            return await env.chat([{"role": "user", "content": "hello chat"}])
    return run(scenario())


# ------------------------------------------------------------- the estimate

def test_estimate_is_worst_case_on_max_tokens():
    """Hermes asks for 131,072 max_tokens; the estimate must assume it uses them."""
    est = estimate_usd(_Spec(10.0, 50.0), est_prompt_tokens=100_000, max_tokens=131_072)
    assert round(est, 4) == round((100_000 * 10 + 131_072 * 50) / 1e6, 4)
    assert est > 7.5


def test_estimate_is_zero_when_no_price_declared():
    assert estimate_usd(_Spec(), 100_000, 100_000) == 0.0


# ---------------------------------------------------------------- the ledger

def test_unreadable_ledger_fails_closed(tmp_path):
    """A budget that fails open is the incident it was written to prevent."""
    p = tmp_path / "budget.json"
    p.write_text("{ this is not json")
    b = Budget(str(p), 100.0)
    assert b.remaining() == 0.0
    assert b.would_exceed(0.01) is True


def test_ledger_survives_a_restart(tmp_path):
    p = str(tmp_path / "budget.json")
    Budget(p, 10.0).record(4.0)
    assert round(Budget(p, 10.0).remaining(), 2) == 6.0


def test_disabled_budget_never_refuses(tmp_path):
    b = Budget(str(tmp_path / "b.json"), 0)
    assert b.enabled is False
    assert b.would_exceed(1_000_000) is False


# --------------------------------------------------------------- the routing

def test_paid_model_is_refused_when_it_would_exceed_the_budget(tmp_path):
    """The whole point: the expensive call never happens."""
    status, payload, _ = _ask(tmp_path, _cfg(tmp_path, budget_daily_usd=0.001))
    assert status == 200
    assert payload["router"]["backend_model"] == "vendor/cheap:free", \
        "a paid model was used despite the budget"


def test_paid_model_is_allowed_when_the_budget_covers_it(tmp_path):
    """Control: proves the refusal above is the budget and not something else."""
    status, payload, _ = _ask(tmp_path, _cfg(tmp_path, budget_daily_usd=10_000))
    assert status == 200
    assert payload["router"]["backend_model"] == "vendor/premium"


def test_no_budget_configured_keeps_the_old_behaviour(tmp_path):
    status, payload, _ = _ask(tmp_path, _cfg(tmp_path))
    assert status == 200
    assert payload["router"]["backend_model"] == "vendor/premium"

# ----------------------------------------------------- unknown cost != free

def _cfg_unpriced(tmp_path, **router):
    """A paid model with NO declared price, deliberately first in the cascade."""
    models = {
        "mystery": {"id": "vendor/mystery-premium", "endpoint": "BACKEND",
                    "capabilities": ["text", "tools", "json"], "context_window": 131072},
        "free": {"id": "vendor/cheap:free", "endpoint": "BACKEND",
                 "capabilities": ["text", "tools", "json"], "context_window": 131072},
    }
    roles = {"chat": {"cascade": ["mystery", "free"], "utterances": CHAT_UTTERANCES}}
    return base_config(tmp_path, models=models, roles=roles, default_role="chat", **router)


def _ask_unpriced(tmp_path, cfg):
    async def scenario():
        backend = FakeBackend({"vendor/mystery-premium": _echo("mystery"),
                               "vendor/cheap:free": _echo("cheap")})
        async with RouterEnv(tmp_path, cfg, backend) as env:
            return await env.chat([{"role": "user", "content": "hello chat"}])
    return run(scenario())


def test_unpriced_paid_model_is_refused_under_a_budget(tmp_path):
    """An unpriced model estimated $0.00 and sailed through the ceiling."""
    status, payload, _ = _ask_unpriced(tmp_path, _cfg_unpriced(tmp_path, budget_daily_usd=1000))
    assert status == 200
    assert payload["router"]["backend_model"] == "vendor/cheap:free", \
        "an unpriced paid model was used despite a budget being set"


def test_unpriced_paid_model_is_allowed_when_no_budget_is_set(tmp_path):
    """Control: the refusal is the budget, not a blanket ban on unpriced models."""
    status, payload, _ = _ask_unpriced(tmp_path, _cfg_unpriced(tmp_path))
    assert status == 200
    assert payload["router"]["backend_model"] == "vendor/mystery-premium"

# ------------------------------------------- streamed turns must also be charged

def _priced_cfg(tmp_path, **router):
    """Paid model first, prices declared, ledger written into tmp_path."""
    models = {
        "paid": {"id": "vendor/premium", "endpoint": "BACKEND",
                 "capabilities": ["text", "tools", "json"], "context_window": 131072,
                 "price_in": 10.0, "price_out": 50.0},
        "free": {"id": "vendor/cheap:free", "endpoint": "BACKEND",
                 "capabilities": ["text", "tools", "json"], "context_window": 131072},
    }
    roles = {"chat": {"cascade": ["paid", "free"], "utterances": CHAT_UTTERANCES}}
    return base_config(tmp_path, models=models, roles=roles, default_role="chat",
                       state_dir=str(tmp_path), **router)


def _spent(tmp_path):
    import json as _j
    p = tmp_path / "budget.json"
    if not p.exists():
        return 0.0
    try:
        return float(_j.loads(p.read_text()).get("spent_usd") or 0)
    except Exception:
        return 0.0


def _run_turn(tmp_path, cfg, stream):
    async def scenario():
        backend = FakeBackend({"vendor/premium": _echo("premium"),
                               "vendor/cheap:free": _echo("cheap")})
        async with RouterEnv(tmp_path, cfg, backend) as env:
            return await env.chat([{"role": "user", "content": "hello chat"}], stream=stream)
    return run(scenario())


def test_a_streamed_paid_turn_is_charged_to_the_budget(tmp_path):
    """The defect: cost was recorded only on the buffered path, so every streamed
    turn spent unmetered and the ceiling never bound. Hermes streams."""
    cfg = _priced_cfg(tmp_path, budget_daily_usd=10_000)
    status, payload, _ = _run_turn(tmp_path, cfg, stream=True)
    assert status == 200
    assert _spent(tmp_path) > 0, "a streamed paid turn was served without charging the budget"


def test_a_buffered_paid_turn_is_charged_to_the_budget(tmp_path):
    cfg = _priced_cfg(tmp_path, budget_daily_usd=10_000)
    status, payload, _ = _run_turn(tmp_path, cfg, stream=False)
    assert status == 200
    assert _spent(tmp_path) > 0


def test_a_free_turn_is_never_charged(tmp_path):
    """Control: only paid backends move the ledger."""
    models = {"free": {"id": "vendor/cheap:free", "endpoint": "BACKEND",
                       "capabilities": ["text", "tools", "json"], "context_window": 131072},
              "free2": {"id": "vendor/other:free", "endpoint": "BACKEND",
                        "capabilities": ["text", "tools", "json"], "context_window": 131072}}
    roles = {"chat": {"cascade": ["free", "free2"], "utterances": CHAT_UTTERANCES}}
    cfg = base_config(tmp_path, models=models, roles=roles, default_role="chat",
                      state_dir=str(tmp_path), budget_daily_usd=10_000)

    async def scenario():
        backend = FakeBackend({"vendor/cheap:free": _echo("cheap"),
                               "vendor/other:free": _echo("other")})
        async with RouterEnv(tmp_path, cfg, backend) as env:
            return await env.chat([{"role": "user", "content": "hello chat"}], stream=True)
    status, _, _ = run(scenario())
    assert status == 200
    assert _spent(tmp_path) == 0.0

# --------------------------------- the escalation hop must honour the budget

def test_escalation_hop_cannot_walk_around_the_budget(tmp_path):
    """The in-request escalation re-dispatches on its own, so it skipped the
    pre-flight gate -- and it escalates UPWARD in price. A budget the escalation
    path can walk around is not a budget."""
    from helpers import WEATHER_TOOL, bad_json_call
    models = {
        "free": {"id": "vendor/cheap:free", "endpoint": "BACKEND",
                 "capabilities": ["text", "tools", "json"], "context_window": 131072},
        "paid": {"id": "vendor/premium", "endpoint": "BACKEND",
                 "capabilities": ["text", "tools", "json"], "context_window": 131072,
                 "price_in": 10.0, "price_out": 50.0},
    }
    roles = {"chat": {"cascade": ["free", "paid"], "utterances": CHAT_UTTERANCES}}
    cfg = base_config(tmp_path, models=models, roles=roles, default_role="chat",
                      state_dir=str(tmp_path), budget_daily_usd=0.0001, hop_limit=2)

    async def scenario():
        # the free model emits an unparseable tool call, which is what arms the
        # in-request escalation toward the next (paid) cascade position
        backend = FakeBackend({"vendor/cheap:free": bad_json_call,
                               "vendor/premium": bad_json_call})
        async with RouterEnv(tmp_path, cfg, backend) as env:
            await env.chat([{"role": "user", "content": "weather in paris?"}],
                           tools=WEATHER_TOOL)
        return backend.per_model_calls
    calls = run(scenario())
    assert calls.get("vendor/premium", 0) == 0, (
        "escalation reached the paid model despite an exhausted budget: %s" % calls)
    assert calls.get("vendor/cheap:free", 0) >= 1


def test_escalation_hop_is_allowed_when_the_budget_covers_it(tmp_path):
    """Control: the block above must be the budget, not escalation being broken."""
    from helpers import WEATHER_TOOL, bad_json_call
    models = {
        "free": {"id": "vendor/cheap:free", "endpoint": "BACKEND",
                 "capabilities": ["text", "tools", "json"], "context_window": 131072},
        "paid": {"id": "vendor/premium", "endpoint": "BACKEND",
                 "capabilities": ["text", "tools", "json"], "context_window": 131072,
                 "price_in": 0.01, "price_out": 0.01},
    }
    roles = {"chat": {"cascade": ["free", "paid"], "utterances": CHAT_UTTERANCES}}
    cfg = base_config(tmp_path, models=models, roles=roles, default_role="chat",
                      state_dir=str(tmp_path), budget_daily_usd=10_000, hop_limit=2)

    async def scenario():
        backend = FakeBackend({"vendor/cheap:free": bad_json_call,
                               "vendor/premium": bad_json_call})
        async with RouterEnv(tmp_path, cfg, backend) as env:
            await env.chat([{"role": "user", "content": "weather in paris?"}],
                           tools=WEATHER_TOOL)
        return backend.per_model_calls
    calls = run(scenario())
    assert calls.get("vendor/premium", 0) >= 1, (
        "escalation never reached the paid model, so the test above proves nothing: %s" % calls)


# ------------------------- cost must be derived when the backend reports none

def test_cost_is_derived_from_tokens_when_the_backend_reports_none():
    """usage.cost is an OpenRouter extension; everyone else reports only tokens."""
    from hello_operator.budget import response_cost

    class _S:
        price_in, price_out = 10.0, 50.0

    payload = {"usage": {"prompt_tokens": 100_000, "completion_tokens": 2_000}}
    got = response_cost(payload, _S())
    assert round(got, 6) == round((100_000 * 10 + 2_000 * 50) / 1e6, 6), got


def test_reported_cost_wins_over_the_derived_one():
    from hello_operator.budget import response_cost

    class _S:
        price_in, price_out = 10.0, 50.0

    payload = {"usage": {"cost": 0.25, "prompt_tokens": 100_000, "completion_tokens": 2_000}}
    assert response_cost(payload, _S()) == 0.25


def test_no_price_and_no_reported_cost_is_zero():
    from hello_operator.budget import response_cost

    class _S:
        price_in, price_out = 0.0, 0.0

    assert response_cost({"usage": {"prompt_tokens": 100_000}}, _S()) == 0.0


# ------------------------------------------- the ledger books what was spent

def _cost_cfg(tmp_path, **router):
    """One paid model only, priced high, so the estimate and the reported cost
    are far apart and cannot be confused for each other."""
    models = {
        "paid": {"id": "vendor/premium", "endpoint": "BACKEND",
                 "capabilities": ["text", "tools", "json"], "context_window": 131072,
                 "price_in": 10.0, "price_out": 50.0},
    }
    roles = {"chat": {"cascade": ["paid"], "utterances": CHAT_UTTERANCES}}
    return base_config(tmp_path, models=models, roles=roles, default_role="chat",
                       **router)


def test_ledger_books_the_reported_cost_not_the_estimate(tmp_path):
    """On the buffered path the backend tells us what the turn actually cost.

    Booking the worst-case pre-flight estimate instead spends the daily ceiling
    on money that was never billed: 473 real calls booked $1.91 against cents
    actually spent, which retires the paid tier ~2/3 through the day for no
    reason. The estimate is the fallback for streamed turns, not the default.
    """
    REPORTED = 0.000012

    def priced(body, idx):
        return {"content": "answer",
                "usage": {"prompt_tokens": 2000, "completion_tokens": 40,
                          "total_tokens": 2040, "cost": REPORTED}}

    cfg = _cost_cfg(tmp_path, budget_daily_usd=10_000)

    async def scenario():
        backend = FakeBackend({"vendor/premium": priced})
        async with RouterEnv(tmp_path, cfg, backend) as env:
            return await env.chat([{"role": "user", "content": "hello chat"}],
                                  max_tokens=8192)
    status, payload, _ = run(scenario())
    assert status == 200
    assert payload["router"]["backend_model"] == "vendor/premium"

    ledger = json.loads((tmp_path / "state" / "budget.json").read_text())
    spent = ledger["spent_usd"]

    estimate = estimate_usd(_Spec(10.0, 50.0), 2000, 8192)
    assert estimate > REPORTED * 100, "test is meaningless unless the two differ"
    assert abs(spent - REPORTED) < 1e-9, (
        f"ledger booked ${spent:.6f}; the backend reported ${REPORTED:.6f} "
        f"(pre-flight estimate was ${estimate:.6f})")


def test_streamed_turn_books_the_reported_cost(tmp_path):
    """Hermes streams every turn, so this is the path that matters.

    With no cost signal the ledger falls back to the worst-case max_tokens
    estimate. Live, that booked $1.48 against $0.55 actually billed and then
    refused every request for the rest of the UTC day with
    "est $0.0090 exceeds the $0.0000 remaining".
    """
    REPORTED = 0.000012

    def priced(body, idx):
        return {"content": "answer",
                "usage": {"prompt_tokens": 2000, "completion_tokens": 40,
                          "total_tokens": 2040, "cost": REPORTED}}

    cfg = _cost_cfg(tmp_path, budget_daily_usd=10_000)

    async def scenario():
        backend = FakeBackend({"vendor/premium": priced})
        async with RouterEnv(tmp_path, cfg, backend) as env:
            return await env.chat([{"role": "user", "content": "hello chat"}],
                                  stream=True, max_tokens=8192)
    status, payload, _ = run(scenario())
    assert status == 200

    ledger = json.loads((tmp_path / "state" / "budget.json").read_text())
    spent = ledger["spent_usd"]
    estimate = estimate_usd(_Spec(10.0, 50.0), 2000, 8192)
    assert estimate > REPORTED * 100, "test is meaningless unless the two differ"
    assert abs(spent - REPORTED) < 1e-9, (
        f"streamed turn booked ${spent:.6f}; backend reported ${REPORTED:.6f} "
        f"(worst-case estimate was ${estimate:.6f})")

def test_budget_exhaustion_still_tries_free(tmp_path):
    """A spent budget must not take the free tier down with it.

    Both disqualifications at once -- free models inside their 429 cooldown,
    paid models refused by the daily ceiling -- is the live failure:
    "no backend could serve the request; last error: or-deepseek-v41-flash:
    refused, est $0.0090 exceeds the $0.0000 remaining". The cooldown has to be
    tripped here or the normal loop serves the free model and the last-resort
    pass is never reached, which would make this test pass vacuously.
    """
    from hello_operator import server as server_mod

    models = {
        "paid": {"id": "vendor/premium", "endpoint": "BACKEND",
                 "capabilities": ["text", "tools", "json"], "context_window": 131072,
                 "price_in": 10.0, "price_out": 50.0},
        "free": {"id": "vendor/cheap:free", "endpoint": "BACKEND",
                 "capabilities": ["text", "tools", "json"], "context_window": 131072},
    }
    roles = {"chat": {"cascade": ["paid", "free"], "utterances": CHAT_UTTERANCES}}
    cfg = base_config(tmp_path, models=models, roles=roles, default_role="chat",
                      budget_daily_usd=0.0000001)

    async def scenario():
        backend = FakeBackend({"vendor/premium": _echo("premium"),
                               "vendor/cheap:free": _echo("cheap")})
        async with RouterEnv(tmp_path, cfg, backend) as env:
            # put the free endpoint into its exhausted cooldown, as a 429 would
            server_mod.mark_free_exhausted(env.backend_base)
            return await env.chat([{"role": "user", "content": "hello chat"}],
                                  max_tokens=1024)
    try:
        status, payload, _ = run(scenario())
    finally:
        server_mod._FREE_EXHAUSTED.clear()   # never leak state into other tests

    assert status == 200, (
        f"budget exhausted + free in cooldown produced {status}; "
        f"a free retry costs nothing and beats failing the turn")
    assert payload["router"]["backend_model"] == "vendor/cheap:free"
    assert payload["router"]["decision"] == "last-resort", \
        "served, but not via the last-resort path this test exists to cover"
