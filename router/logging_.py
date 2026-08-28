"""Decision logging (FR-11).

One JSONL line per routing decision: session, role, model, cascade position,
trigger, decision type, routing latency. Escalation rate per role is the
primary tuning signal; this file is where it comes from.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

log = logging.getLogger("router.decisions")


class DecisionLog:
    def __init__(self, path: str):
        self.path: Optional[Path] = None
        if path:
            p = Path(path).expanduser()
            p.parent.mkdir(parents=True, exist_ok=True)
            self.path = p

    def write(self, **fields) -> None:
        fields.setdefault("ts", round(time.time(), 3))
        if self.path is None:
            return
        try:
            with self.path.open("a") as f:
                f.write(json.dumps(fields, sort_keys=True) + "\n")
        except OSError as e:
            # Logging must never be the component that kills a run (FR-9).
            log.warning("decision log write failed: %s", e)
