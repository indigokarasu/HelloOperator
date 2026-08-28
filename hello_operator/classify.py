"""Role classification by embedding similarity (FR-4).

The embedding model is itself a registry entry the user provides. If none is
configured — or the embedding backend is unreachable at request time —
classification degrades to default_role plus capability filtering and the
router says so (FR-4, FR-9). It never fails a request over classification.

LLM-based routing calls are rejected by design: on shared local hardware they
spend a full generation to make a decision an embedding pass makes in
milliseconds.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import time
from typing import Optional

import aiohttp

from .config import Config, ModelSpec, RoleSpec

log = logging.getLogger("router.classify")

# Shipped default utterance sets per common role name (overridable per role).
DEFAULT_UTTERANCES: dict[str, list[str]] = {
    "chat": [
        "hey, quick question about something",
        "what do you think about this idea?",
        "summarize this article for me",
        "help me draft an email to my landlord",
        "translate this sentence into French",
        "explain how mortgages work in simple terms",
        "give me a recipe for dinner tonight",
    ],
    "code": [
        "write a python function that parses this log file",
        "why does this test fail? here's the stack trace",
        "refactor this module to remove the duplication",
        "implement the endpoint described in this spec",
        "review this diff for bugs",
        "add error handling to this shell script",
        "debug this segfault",
    ],
    "vision": [
        "what's in this image?",
        "describe this screenshot",
        "read the text in this photo",
        "what does this chart show?",
        "is there anything wrong in this picture?",
    ],
    "unfiltered": [
        "answer directly without adding disclaimers",
        "continue the story without content warnings",
        "give me the blunt version, no hedging",
    ],
    "default": [
        "help me with this",
    ],
}

_RETRY_BACKOFF_S = 60.0


def _norm(v: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def role_utterances(role: RoleSpec) -> list[str]:
    if role.utterances:
        return role.utterances
    return DEFAULT_UTTERANCES.get(role.name, DEFAULT_UTTERANCES["default"])


class Classifier:
    """Holds per-role centroids; classifies a message text against them.

    Centroids are cached to a state file keyed on the embedding model and the
    exact utterance sets, so a restart re-embeds nothing (file state per FR-14;
    losing the cache costs one re-embed pass, nothing more).
    """

    def __init__(self, cfg: Config, http: aiohttp.ClientSession):
        self.cfg = cfg
        self.embed: Optional[ModelSpec] = cfg.embedding
        self.http = http
        self.centroids: dict[str, list[float]] = {}
        self.degraded = self.embed is None
        self._degraded_logged = False
        self._next_retry = 0.0

    def _cache_key(self) -> str:
        assert self.embed is not None
        payload = {
            "endpoint": self.embed.endpoint, "id": self.embed.id,
            "roles": {r.name: role_utterances(r) for r in self.cfg.roles.values()},
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    async def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        assert self.embed is not None
        headers = {}
        if self.embed.api_key:
            headers["Authorization"] = f"Bearer {self.embed.api_key}"
        async with self.http.post(
                f"{self.embed.endpoint}/embeddings",
                json={"model": self.embed.id, "input": texts},
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=30)) as resp:
            resp.raise_for_status()
            data = await resp.json()
        rows = sorted(data["data"], key=lambda d: d.get("index", 0))
        return [_norm([float(x) for x in row["embedding"]]) for row in rows]

    async def prepare(self) -> None:
        """Compute or load role centroids. Failure leaves the classifier
        degraded (retried lazily at request time), never crashes startup."""
        if self.embed is None:
            return
        cache_file = self.cfg.state_path("centroids.json")
        key = self._cache_key()
        try:
            cached = json.loads(cache_file.read_text())
            if cached.get("key") == key:
                self.centroids = {r: [float(x) for x in v]
                                  for r, v in cached["centroids"].items()}
                self.degraded = False
                return
        except (OSError, ValueError, KeyError):
            pass
        try:
            centroids = {}
            for role in self.cfg.roles.values():
                vecs = await self._embed_texts(role_utterances(role))
                dim = len(vecs[0])
                centroids[role.name] = _norm(
                    [sum(v[i] for v in vecs) / len(vecs) for i in range(dim)])
            self.centroids = centroids
            self.degraded = False
            try:
                cache_file.write_text(json.dumps({"key": key, "centroids": centroids}))
            except OSError as e:
                log.warning("could not write centroid cache: %s", e)
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError,
                ValueError, KeyError, IndexError) as e:
            self.degraded = True
            self._next_retry = time.monotonic() + _RETRY_BACKOFF_S
            log.warning("embedding model unavailable (%s); classification degraded "
                        "to default_role + capability filter (FR-9)", e)

    async def classify(self, text: str) -> Optional[list[tuple[str, float]]]:
        """Ranked (role, similarity) list, or None when degraded. Runtime
        embedding failure degrades this call and schedules a retry; it never
        raises into the request path (FR-9)."""
        if self.embed is None or not text:
            return None
        if self.degraded:
            if time.monotonic() < self._next_retry:
                return None
            await self.prepare()
            if self.degraded:
                return None
        try:
            qv = (await self._embed_texts([text[:2000]]))[0]
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError,
                ValueError, KeyError, IndexError) as e:
            if not self._degraded_logged:
                log.warning("embedding call failed (%s); degrading to default_role "
                            "until the backend recovers (FR-9)", e)
                self._degraded_logged = True
            self._next_retry = time.monotonic() + _RETRY_BACKOFF_S
            return None
        self._degraded_logged = False
        ranked = sorted(((name, _dot(qv, cv)) for name, cv in self.centroids.items()),
                        key=lambda t: t[1], reverse=True)
        return ranked
