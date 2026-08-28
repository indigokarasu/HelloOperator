# hermes-model-router

A local-first routing layer that automatically selects which of **your**
models handles a given conversation turn — no per-prompt, per-task, or
per-session model selection by the operator.

It presents one OpenAI-compatible endpoint. Hermes profiles (or any
OpenAI-compatible client) target one logical model name; the router selects
among the models you have configured, holds sessions to their model, and walks
an escalation cascade on mechanically observable failure.

Implements `docs/requirements-v0.4.md`. Design in one line: **the user's
models, not assumed models** — the router ships with no model opinions, and a
single-model config behaves like a direct connection.

## Install

```bash
pip install .
hermes-model-router -c config.yaml
```

Dependencies: `aiohttp`, `PyYAML`. No database daemon; all persistent state is
files under `state_dir`. Runs on macOS and Linux against any OpenAI-compatible
backend (llama.cpp, Ollama, LM Studio, vLLM, MLX server, ...).

## Quick start

1. Copy `config.example.yaml` to `config.yaml` and declare your models.
2. `hermes-model-router --check` — validates config, endpoint reachability,
   and shows every resolved field with its provenance
   (`declared` / `detected` / `default`). Fails loudly at startup, never
   silently at request time.
3. `hermes-model-router` — serve. Point your client at
   `http://127.0.0.1:8800/v1` with model `hermes-router`.

Minimal (degenerate) config — one model, no roles, near-zero overhead:

```yaml
models:
  main:
    id: qwen3-30b
    endpoint: http://127.0.0.1:8080/v1
```

Routing is something a config grows into, not a tax every install pays.

## Discovery

```bash
hermes-model-router --discover           # metadata only
hermes-model-router --discover --probe   # + opt-in active probes, run serially
```

Emits a proposed registry-and-roles config from backend metadata (llama.cpp
`/props`, Ollama `/api/show`, LM Studio `/api/v0/models`, vLLM
`max_model_len`), optional probes (tools / vision / JSON verified by response
shape; a timed micro-benchmark for `speed_class`), and name heuristics as
tie-breaking hints only. **Nothing routes on the proposal until you adopt it**
— and your declarations always win over detection; disagreement is a logged
warning, never a silent correction.

Probes default **off** for `residency: on-demand` models: probing one forces a
full model load, which is precisely the cost that configuration exists to
avoid.

At startup the router also flags **registry drift** — models a backend serves
that you haven't registered, and registered models a backend no longer serves.
It never silently adds or drops routing targets.

## How a request is routed

Selection order: **capability filter → role classification → cascade position.**

1. **Capability filter** (hard, zero inference cost): image/audio anywhere in
   the message history requires a model declaring `vision`/`audio`; tool
   definitions require `tools`; the estimated token count must fit the model's
   context window. If nothing survives, you get a clear error naming the
   constraint — media is never silently routed to a text-only model.
2. **Role classification**: the current message is scored by embedding
   similarity against per-role example utterances (shipped defaults for
   `chat` / `code` / `vision` / `unfiltered`, overridable per role). The
   embedding model is a registry entry you provide; without one,
   classification degrades to `default_role` + capability filtering and the
   router says so. LLM-based routing calls are rejected by design.
3. **Cascade position**: session affinity and escalation state decide how far
   down the role's cascade the session sits.

### Affinity and transitions

Once routed, a session follows its model (keyed on `x-session-id`, then
`session_id`/`user` in the body, then a fragile history hash). A 20-step tool
chain stays on one model. Reclassification happens only at observable
transitions:

| Transition | Why |
|---|---|
| Material tool-set change | Phase change (discussion → implementation) |
| Classifier disagreement beyond `classify_margin` | Intent shifted |
| Compaction (`x-context-compacted` header, or a large context drop) | The re-prefill is already being paid; switching is free |
| Escalation trigger | The current model demonstrably failed |

`switch_cost: high` raises the transition thresholds (for load-on-demand
installs where a switch costs a model load); it does not change the mechanism.

### Escalation

Cascades walk one position on **mechanically observable failure only** —
never on model self-assessment:

- a tool call whose arguments don't parse against the supplied schema
- no tool call where `tool_choice` structurally required one
- an identical call with identical arguments repeated past
  `repeat_call_threshold`
- (advisory, off by default) refusal string markers

Non-streaming responses are validated **before release** and retried one
cascade position down within the same request (`hop_limit` caps total hops per
session). Once a response **streams**, the turn is committed: validation runs
on the assembled stream and escalation applies to the next turn. A role with
`buffered: true` holds tool-call-bearing streaming responses for validation
first, trading latency for retractability.

### Degraded operation

The router must never be the component that kills an unattended run:

| Condition | Behavior |
|---|---|
| Embedding model down | `default_role` + capability filter; request proceeds |
| Every model filtered out | Clear error naming the constraint |
| Affinity lost (restart) | Reclassify; one re-prefill per active session |
| Backend unreachable | Next cascade position, then any eligible model; logged |

## Observability

- Response headers on every turn: `x-router-model`, `x-router-backend-model`,
  `x-router-role`, `x-router-pos`, `x-router-decision`, `x-router-latency-ms`.
  Non-streaming bodies also carry an ignorable `router` object.
- `decision_log`: one JSONL line per decision (session, role, model, cascade
  position, trigger, decision type, routing latency). **Escalation rate per
  role is the primary tuning signal** — a healthy band is roughly 5–30% of
  sessions; outside it, the cascade order is wrong.
- `GET /healthz`: mode, classification state, live session count.

## Manual override

Pin a session for debugging with the `x-router-pin: <model-key>` header (or by
naming a registered model directly in the `model` field). Pins never bypass
the capability filter. Frequent pinning is a routing-failure signal — the
design goal is zero pins in normal operation.

## Conventions, not dependencies

Hermes integration points are documented conventions so the router works with
any OpenAI-compatible harness unmodified:

- session identity: `x-session-id` header (or `session_id` / `user` body field)
- compaction signal: `x-context-compacted: true` header on the first request
  after a context compaction (a >40% context drop is also detected
  heuristically)

## Deploy

Examples for unattended restart under the host's service manager are in
`deploy/` (systemd unit for Linux, launchd plist for macOS). The router binds
loopback by default; it fronts models with tool access, so binding other
interfaces requires the explicit `allow_non_loopback: true`.

## Should this install route at all?

The null hypothesis is route-everything-to-main. Enable roles only when a
single model demonstrably falls short — latency on mechanical turns, refusals,
a missing capability. The degenerate config makes "install now, route later"
safe, and `AC-7` in the requirements is explicit: if routing is less reliable
than main-only, remove it. All escalation signals are mechanical, so
well-formed-but-wrong output triggers nothing — which argues for conservative
cascades (strong model at position 0, fast model as an explicit opt-in).

## Tests

```bash
pip install -e ".[dev]"
python -m pytest tests/
```

58 tests: config resolution/provenance, capability filtering, classification,
affinity, every transition and escalation trigger, streaming commit semantics
(including a backend dying mid-stream), failover (including HTML-speaking and
unreachable backends), discovery harvesting (Ollama / llama.cpp / generic),
drift, and the degenerate path — all through real loopback sockets against a
scriptable fake backend. `tests/test_review_regressions.py` pins down every
defect found in the pre-release adversarial review, one test per defect.
