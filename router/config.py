"""Configuration loading, resolution, and validation.

Every model field resolves in a fixed order: user declaration > detected value
> heuristic default (spec 3.1). Each resolved field carries provenance
("declared" | "detected" | "default") so the operator can always see why the
router believes what it believes. A declaration always wins, even when
detection disagrees; disagreement is a logged warning, never a silent fix
(FR-10).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

KNOWN_CAPABILITIES = {"text", "vision", "audio", "tools", "json", "reasoning-effort", "embeddings"}
SPEED_CLASSES = {"fast", "mid", "slow"}
RESIDENCY = {"resident", "on-demand"}

# Role names whose conventional meaning implies a capability. Used only for
# warnings and for capability-filtering of eligible roles; a user can make the
# requirement explicit (and hard) with `requires:` on the role.
ROLE_IMPLIED_CAPS = {"vision": {"vision"}, "audio": {"audio"}}

# Heuristic defaults (provenance: "default")
DEFAULT_CONTEXT_WINDOW = 8192
DEFAULT_CAPABILITIES = {"text"}


class ConfigError(Exception):
    """Raised for configuration the router refuses to start on (FR-10)."""


@dataclass
class ModelSpec:
    key: str                      # registry key (what cascades reference)
    id: str                       # name passed to the backend
    endpoint: str                 # OpenAI-compatible base URL, e.g. http://127.0.0.1:8080/v1
    capabilities: set[str] = field(default_factory=lambda: set(DEFAULT_CAPABILITIES))
    context_window: int = DEFAULT_CONTEXT_WINDOW
    residency: str = "resident"
    speed_class: str = "mid"
    api_key: str = ""             # optional bearer token some backends want
    probe: Optional[bool] = None  # None => default: probes off for on-demand (3.1a)
    provenance: dict[str, str] = field(default_factory=dict)

    @property
    def probes_enabled(self) -> bool:
        if self.probe is not None:
            return self.probe
        return self.residency != "on-demand"


@dataclass
class RoleSpec:
    name: str
    cascade: list[str]                       # ordered model keys; position 0 is the role's main
    utterances: Optional[list[str]] = None   # None => shipped defaults (classify.py)
    requires: set[str] = field(default_factory=set)  # hard capability requirements
    buffered: bool = False                   # FR-8 optional buffered mode, per-role


@dataclass
class Settings:
    listen_host: str = "127.0.0.1"
    listen_port: int = 8800
    allow_non_loopback: bool = False
    logical_model: str = "hermes-router"
    session_header: str = "x-session-id"
    compaction_header: str = "x-context-compacted"
    pin_header: str = "x-router-pin"
    affinity_idle_timeout_s: float = 3600.0
    switch_cost: str = "low"              # low | high (FR-6)
    hop_limit: int = 1                    # FR-7 default
    classify_margin: float = 0.08
    max_inflight: int = 0                 # 0 = unlimited (NFR-5)
    request_timeout_s: float = 600.0      # total cap, non-streaming only
    stream_idle_timeout_s: float = 300.0  # inter-chunk cap for streams — a
                                          # healthy long generation must never
                                          # be killed by a total-body timeout
    connect_timeout_s: float = 10.0
    assumed_completion_tokens: int = 1024
    media_token_estimate: int = 1024      # flat per-image/audio-part estimate
    context_safety_margin: int = 256
    repeat_call_threshold: int = 3
    refusal_markers: list[str] = field(default_factory=list)  # MAY; empty = off
    refusal_escalate: bool = False        # advisory only unless explicitly enabled
    decision_log: str = ""                # path; empty disables (FR-11)
    state_dir: str = "~/.hermes-model-router"
    default_role: str = ""


@dataclass
class Config:
    settings: Settings
    models: dict[str, ModelSpec]
    roles: dict[str, RoleSpec]
    embedding: Optional[ModelSpec]
    warnings: list[str] = field(default_factory=list)
    source_path: str = ""

    @property
    def degenerate(self) -> bool:
        """One model, every cascade is just that model: NFR-7 fast path."""
        if len(self.models) != 1:
            return False
        only = next(iter(self.models))
        return all(r.cascade == [only] for r in self.roles.values())

    def model_by_key_or_id(self, name: str) -> Optional[ModelSpec]:
        if name in self.models:
            return self.models[name]
        for m in self.models.values():
            if m.id == name:
                return m
        return None

    def state_path(self, *parts: str) -> Path:
        p = Path(os.path.expanduser(self.settings.state_dir)).joinpath(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p


def _resolve_field(name: str, declared: Any, detected: Any, default: Any,
                   provenance: dict, warnings: list, model_key: str):
    """declared > detected > default, with provenance and disagreement warnings."""
    if declared is not None:
        provenance[name] = "declared"
        if detected is not None and detected != declared:
            warnings.append(
                f"model '{model_key}': declared {name}={declared!r} but backend "
                f"reports {detected!r}; proceeding on the declaration (FR-10)")
        return declared
    if detected is not None:
        provenance[name] = "detected"
        return detected
    provenance[name] = "default"
    return default


def _parse_model(key: str, raw: dict, detected: dict, warnings: list) -> ModelSpec:
    if not isinstance(raw, dict):
        raise ConfigError(f"model '{key}' must be a mapping")
    endpoint = raw.get("endpoint")
    if not endpoint:
        raise ConfigError(f"model '{key}' is missing required field 'endpoint'")
    prov: dict[str, str] = {"endpoint": "declared"}

    mid = _resolve_field("id", raw.get("id"), detected.get("id"), key, prov, warnings, key)

    caps_declared = raw.get("capabilities")
    caps_detected = detected.get("capabilities")
    caps = _resolve_field("capabilities",
                          set(caps_declared) if caps_declared is not None else None,
                          set(caps_detected) if caps_detected is not None else None,
                          set(DEFAULT_CAPABILITIES), prov, warnings, key)
    unknown = set(caps) - KNOWN_CAPABILITIES
    if unknown:
        warnings.append(f"model '{key}': unknown capabilities {sorted(unknown)} (kept verbatim)")

    ctx = _resolve_field("context_window", raw.get("context_window"),
                         detected.get("context_window"), DEFAULT_CONTEXT_WINDOW,
                         prov, warnings, key)
    residency = _resolve_field("residency", raw.get("residency"),
                               detected.get("residency"), "resident", prov, warnings, key)
    speed = _resolve_field("speed_class", raw.get("speed_class"),
                           detected.get("speed_class"), "mid", prov, warnings, key)

    if residency not in RESIDENCY:
        raise ConfigError(f"model '{key}': residency must be one of {sorted(RESIDENCY)}")
    if speed not in SPEED_CLASSES:
        raise ConfigError(f"model '{key}': speed_class must be one of {sorted(SPEED_CLASSES)}")
    if not isinstance(ctx, int) or ctx <= 0:
        raise ConfigError(f"model '{key}': context_window must be a positive integer")

    api_key = os.path.expandvars(str(raw.get("api_key", "") or ""))
    return ModelSpec(key=key, id=str(mid), endpoint=str(endpoint).rstrip("/"),
                     capabilities=set(caps), context_window=int(ctx),
                     residency=residency, speed_class=speed, api_key=api_key,
                     probe=raw.get("probe"), provenance=prov)


def _parse_settings(raw_router: dict, raw_routing: dict) -> Settings:
    s = Settings()
    listen = str(raw_router.get("listen", f"{s.listen_host}:{s.listen_port}"))
    if ":" not in listen:
        raise ConfigError(f"router.listen must be host:port, got {listen!r}")
    host, _, port = listen.rpartition(":")
    try:
        s.listen_port = int(port)
    except ValueError:
        raise ConfigError(f"router.listen port is not an integer: {listen!r}")
    s.listen_host = host

    for name in ("logical_model", "session_header", "compaction_header", "pin_header",
                 "switch_cost", "decision_log", "state_dir"):
        if name in raw_router:
            setattr(s, name, str(raw_router[name]))
    for name in ("affinity_idle_timeout_s", "classify_margin",
                 "request_timeout_s", "stream_idle_timeout_s", "connect_timeout_s"):
        if name in raw_router:
            setattr(s, name, float(raw_router[name]))
    for name in ("hop_limit", "max_inflight", "assumed_completion_tokens",
                 "media_token_estimate", "context_safety_margin", "repeat_call_threshold"):
        if name in raw_router:
            setattr(s, name, int(raw_router[name]))
    for name in ("allow_non_loopback", "refusal_escalate"):
        if name in raw_router:
            setattr(s, name, bool(raw_router[name]))
    if "refusal_markers" in raw_router:
        s.refusal_markers = [str(x) for x in (raw_router["refusal_markers"] or [])]

    if s.switch_cost not in ("low", "high"):
        raise ConfigError("router.switch_cost must be 'low' or 'high'")
    if s.hop_limit < 0:
        raise ConfigError("router.hop_limit must be >= 0")

    s.default_role = str(raw_routing.get("default_role", "") or "")
    return s


def load(path: str, detected_cache: Optional[dict[str, dict]] = None) -> Config:
    """Load, resolve, and validate a config file. Raises ConfigError on anything
    the router should refuse to start on; collects non-fatal issues in
    Config.warnings. `detected_cache` maps model key -> detected fields
    (discovery output); absent entries resolve declaration > default only.
    """
    p = Path(path)
    if not p.exists():
        raise ConfigError(f"config file not found: {path}")
    try:
        raw = yaml.safe_load(p.read_text()) or {}
    except yaml.YAMLError as e:
        raise ConfigError(f"config is not valid YAML: {e}")
    if not isinstance(raw, dict):
        raise ConfigError("config root must be a mapping")

    warnings: list[str] = []
    detected_cache = detected_cache or {}

    settings = _parse_settings(raw.get("router") or {}, raw.get("routing") or {})

    raw_models = raw.get("models") or {}
    if not raw_models:
        raise ConfigError("no models declared: the registry ('models:') is required")
    models: dict[str, ModelSpec] = {}
    for key, m in raw_models.items():
        models[str(key)] = _parse_model(str(key), m or {}, detected_cache.get(str(key), {}), warnings)

    embedding: Optional[ModelSpec] = None
    if raw.get("embedding"):
        embedding = _parse_model("embedding", raw["embedding"],
                                 detected_cache.get("embedding", {}), warnings)

    raw_roles = raw.get("roles") or {}
    roles: dict[str, RoleSpec] = {}
    for name, r in raw_roles.items():
        r = r or {}
        cascade = r.get("cascade") or []
        if not cascade:
            raise ConfigError(f"role '{name}' has an empty cascade")
        roles[str(name)] = RoleSpec(
            name=str(name),
            cascade=[str(x) for x in cascade],
            utterances=[str(u) for u in r["utterances"]] if r.get("utterances") else None,
            requires=set(r.get("requires") or []),
            buffered=bool(r.get("buffered", False)),
        )

    # Degenerate installs are first-class (spec 2.3): with exactly one model and
    # no roles, synthesize the single role rather than demanding ceremony.
    if not roles:
        if len(models) == 1:
            only = next(iter(models))
            roles["default"] = RoleSpec(name="default", cascade=[only])
            if not settings.default_role:
                settings.default_role = "default"
        else:
            raise ConfigError("multiple models declared but no roles: declare "
                              "'roles:' so the router knows who does what")
    if not settings.default_role:
        raise ConfigError("routing.default_role is required when roles are declared")
    if settings.default_role not in roles:
        raise ConfigError(f"routing.default_role '{settings.default_role}' is not a declared role")

    # FR-10: cascade entries must reference registered models.
    for role in roles.values():
        for mk in role.cascade:
            if mk not in models:
                raise ConfigError(f"role '{role.name}' cascade references "
                                  f"unregistered model '{mk}'")
        # Hard `requires` must be satisfiable by position 0.
        p0 = models[role.cascade[0]]
        missing = role.requires - p0.capabilities
        if missing:
            raise ConfigError(f"role '{role.name}' requires {sorted(missing)} but its "
                              f"position-0 model '{p0.key}' does not declare them")
        # Conventional-name warning (soft; the assignment is a judgment, 3.1a).
        implied = ROLE_IMPLIED_CAPS.get(role.name, set()) - p0.capabilities - role.requires
        if implied:
            warnings.append(f"role '{role.name}': name implies {sorted(implied)} but "
                            f"position-0 model '{p0.key}' does not declare them")

    # FR-12: loopback by default; anything else is explicit opt-in.
    if settings.listen_host not in ("127.0.0.1", "localhost", "::1") and not settings.allow_non_loopback:
        raise ConfigError(
            f"router.listen binds {settings.listen_host!r}, which is not loopback. "
            "The router fronts models with tool access; set "
            "router.allow_non_loopback: true to opt in explicitly (FR-12)")

    if embedding is None:
        warnings.append("no embedding model configured: role classification is "
                        "degraded to routing.default_role + capability filtering (FR-4)")

    for m in models.values():
        if m.residency == "on-demand" and m.probe:
            warnings.append(f"model '{m.key}': probes explicitly enabled on an "
                            f"on-demand model; each probe forces a full load (3.1a)")

    return Config(settings=settings, models=models, roles=roles,
                  embedding=embedding, warnings=warnings, source_path=str(p))


def provenance_table(cfg: Config) -> str:
    """Human-readable resolved-config summary with per-field provenance."""
    lines = []
    for m in list(cfg.models.values()) + ([cfg.embedding] if cfg.embedding else []):
        lines.append(f"model {m.key} ({m.endpoint})")
        for f_ in ("id", "capabilities", "context_window", "residency", "speed_class"):
            val = getattr(m, f_)
            if isinstance(val, set):
                val = sorted(val)
            lines.append(f"  {f_:<16} = {val!r:<40} [{m.provenance.get(f_, 'default')}]")
    return "\n".join(lines)
