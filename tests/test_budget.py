"""The spend ceiling, enforced where the money is actually spent.

Each test puts the PAID model first in the cascade, so a pass cannot be confused
with "the free one was chosen anyway".
"""
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
