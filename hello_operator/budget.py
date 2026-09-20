"""A spend ceiling the router enforces itself.

Every layer of this stack failed OPEN toward spending: a failure in the cheap
path escalated to the expensive path, and nothing anywhere knew what anything
cost. This module is the missing invariant -- never spend more than the operator
allowed, per UTC day -- enforced at the only place money is actually spent.

Two properties on purpose:

* PRE-FLIGHT. A paid candidate is rejected BEFORE the request is sent, using the
  operator-declared price and the request's own token estimate. Post-hoc
  accounting cannot stop the one 500k-token call that empties the account.
* FAIL CLOSED. An unreadable or corrupt ledger means "assume spent", not
  "assume free". The failure mode of a budget that fails open is the incident
  it was written to prevent.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from typing import Optional

log = logging.getLogger("router.budget")


def _utc_day(now: Optional[float] = None) -> str:
    return time.strftime("%Y-%m-%d", time.gmtime(now if now is not None else time.time()))


class Budget:
    """Daily USD ceiling with a durable ledger. daily_usd <= 0 disables it."""

    def __init__(self, path: str, daily_usd: float):
        self.daily_usd = float(daily_usd or 0)
        self.path = os.path.expanduser(path) if path else ""
        self._day = _utc_day()
        self._spent = 0.0
        self._broken = False
        self._load()

    # ---------------------------------------------------------------- state
    def _load(self) -> None:
        if not self.enabled or not self.path:
            return
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path) as fh:
                data = json.load(fh)
            day = str(data.get("day") or "")
            spent = float(data.get("spent_usd") or 0)
        except Exception as exc:  # noqa: BLE001
            # Fail CLOSED: a ledger we cannot read is treated as exhausted.
            self._broken = True
            log.error("budget ledger unreadable (%s); treating the day as spent", exc)
            return
        if day == self._day:
            self._spent = spent

    def _save(self) -> None:
        if not self.enabled or not self.path:
            return
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=os.path.dirname(self.path), prefix=".budget-")
            with os.fdopen(fd, "w") as fh:
                json.dump({"day": self._day, "spent_usd": round(self._spent, 6)}, fh)
            os.replace(tmp, self.path)          # atomic: never a half-written ledger
        except Exception as exc:  # noqa: BLE001
            log.warning("budget ledger write failed: %s", exc)

    def _roll(self) -> None:
        today = _utc_day()
        if today != self._day:
            self._day, self._spent, self._broken = today, 0.0, False
            self._save()

    # ------------------------------------------------------------- queries
    @property
    def enabled(self) -> bool:
        return self.daily_usd > 0

    def spent(self) -> float:
        self._roll()
        return self._spent

    def remaining(self) -> float:
        if not self.enabled:
            return float("inf")
        self._roll()
        if self._broken:
            return 0.0
        return max(0.0, self.daily_usd - self._spent)

    def would_exceed(self, est_usd: float) -> bool:
        """True when a call estimated at est_usd must not be made."""
        if not self.enabled:
            return False
        return float(est_usd or 0) > self.remaining()

    # -------------------------------------------------------------- update
    def record(self, usd: float) -> None:
        if not self.enabled:
            return
        self._roll()
        try:
            amount = float(usd or 0)
        except (TypeError, ValueError):
            return
        if amount <= 0:
            return
        self._spent += amount
        self._save()
        if self._spent >= self.daily_usd:
            log.warning("budget: daily ceiling $%.2f reached (spent $%.4f); "
                        "paid models are now refused until %s UTC",
                        self.daily_usd, self._spent, "00:00")


def estimate_usd(spec, est_prompt_tokens: int, max_tokens: int) -> float:
    """Worst-case cost of one call at the operator-declared price.

    Deliberately pessimistic: it assumes the model emits its entire max_tokens
    allowance. Hermes routinely asks for 131,072, and the whole point is to
    refuse that call before it happens rather than to discover it afterwards.
    """
    p_in = float(getattr(spec, "price_in", 0) or 0)
    p_out = float(getattr(spec, "price_out", 0) or 0)
    if p_in <= 0 and p_out <= 0:
        return 0.0
    return (max(0, int(est_prompt_tokens or 0)) * p_in
            + max(0, int(max_tokens or 0)) * p_out) / 1_000_000.0


def response_cost(payload: dict) -> float:
    """Actual cost OpenRouter reports on a response, when it reports one."""
    try:
        usage = (payload or {}).get("usage") or {}
        return float(usage.get("cost") or 0)
    except (AttributeError, TypeError, ValueError):
        return 0.0
