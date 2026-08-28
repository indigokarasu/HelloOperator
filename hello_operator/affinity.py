"""Session affinity (FR-5).

Keyed on a caller-supplied session identifier (header or request field), with
idle-timeout expiry. History-hash derivation is the documented-fragile
fallback (MAY): it survives simple continuations but breaks under compaction,
which is exactly when reclassification is cheapest anyway (FR-6).

Affinity is in-memory only. It is explicitly allowed to be lost (FR-14,
NFR-3): loss degrades to one reclassification per active session.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Optional

from .config import Settings


@dataclass
class SessionState:
    key: str
    role: str = ""
    model_key: str = ""
    pos: int = 0                    # cascade position (FR-7)
    hops_used: int = 0              # escalations consumed against hop_limit
    last_seen: float = 0.0
    tools_sig: str = ""
    est_tokens: int = 0
    msg_count: int = 0
    pending_escalation: str = ""    # trigger label; consumed at next turn (FR-8)
    last_call_sig: str = ""         # repeat-call detection (FR-7)
    last_call_count: int = 0
    pinned: str = ""                # FR-13 manual override
    turns: int = 0
    last_cls_hash: str = ""         # classification cache: same user text in a
    last_cls_ranked: list = field(default_factory=list)  # tool chain is not re-embedded (NFR-1)


class AffinityMap:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._sessions: dict[str, SessionState] = {}
        self._last_sweep = time.monotonic()

    def get(self, key: str) -> Optional[SessionState]:
        self._maybe_sweep()
        st = self._sessions.get(key)
        if st is None:
            return None
        if time.monotonic() - st.last_seen > self.settings.affinity_idle_timeout_s:
            del self._sessions[key]
            return None
        return st

    def ensure(self, key: str) -> SessionState:
        st = self.get(key)
        if st is None:
            st = SessionState(key=key)
            self._sessions[key] = st
        return st

    def touch(self, st: SessionState) -> None:
        st.last_seen = time.monotonic()
        st.turns += 1

    def _maybe_sweep(self) -> None:
        now = time.monotonic()
        if now - self._last_sweep < 300:
            return
        self._last_sweep = now
        cutoff = now - self.settings.affinity_idle_timeout_s
        for k in [k for k, s in self._sessions.items() if s.last_seen < cutoff]:
            del self._sessions[k]

    def __len__(self) -> int:
        return len(self._sessions)


def derive_session_key(headers, body: dict, settings: Settings) -> str:
    """Caller-supplied id wins; history hash is the fragile fallback (MAY)."""
    sid = headers.get(settings.session_header)
    if sid:
        return f"hdr:{sid}"
    for fld in ("session_id", "user"):
        v = body.get(fld)
        if v:
            return f"body:{fld}:{v}"
    # Fallback: hash of the conversation opening (first system + first user
    # message). Stable across appended turns; breaks under compaction.
    messages = body.get("messages") or []
    first_system = next((m for m in messages if isinstance(m, dict)
                         and m.get("role") == "system"), None)
    first_user = next((m for m in messages if isinstance(m, dict)
                       and m.get("role") == "user"), None)
    basis = json.dumps([first_system, first_user], sort_keys=True, default=str)
    return "hist:" + hashlib.sha1(basis.encode()).hexdigest()[:16]
