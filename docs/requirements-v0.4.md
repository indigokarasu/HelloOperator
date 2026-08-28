# Hermes Model Router — Requirements

Version 0.4 (draft)
Date: 2026-08-27
Status: Requirements only. Not a build spec.
Supersedes: v0.3 (2026-08-27)

Change from v0.3: adds model discovery. The router detects context window, capabilities, and speed from backend metadata and optional probes, and proposes registry entries and role assignments as overridable defaults. Resolution order everywhere: user declaration > detected > heuristic default.

Change from v0.2: generalized from a single-appliance component to a distributable companion for any Hermes install. Models, roles, and cascades are user-defined configuration. The 512GB Mac Studio deployment becomes the reference install, not the design.

---

## 1. Purpose

A routing layer that automatically selects which of the user's models handles a given conversation turn, without per-prompt, per-task, or per-session model selection by the operator.

It presents an OpenAI-compatible endpoint. Hermes profiles (or any OpenAI-compatible client) target one logical model name; the router selects among the models the user has configured.

---

## 2. Design principles

1. **The user's models, not assumed models.** The router ships with no model opinions. Every model, its capabilities, and its position in a cascade is declared by the user. A user whose main model is text-only is as supported as one whose main model is multimodal.

2. **Roles and cascades are the unit of configuration.** A role is a routing target (what kind of work). A cascade is an ordered list of models serving that role (who does it, and who takes over on failure). Any role may have a cascade of length one.

3. **Degenerate configurations are first-class.** One model, one role, no cascade must work and add near-zero overhead. Routing is something a config grows into, not a tax every install pays.

4. **No hardware assumptions.** The reference install holds all models resident in 512GB of unified memory. Another user runs two models on 32GB with load-on-demand. Switching cost differs by an order of magnitude between these; the router must not hard-code either.

5. **Mechanical signals over model self-assessment.** Escalation and transitions are triggered by observable events (parse failure, tool-set change), never by asking a model whether it is struggling.

---

## 3. Concepts

### 3.1 Model registry

Each model the user runs is declared once:

| Field | Meaning |
|---|---|
| id | Name passed to the backend |
| endpoint | OpenAI-compatible base URL (llama.cpp, Ollama, LM Studio, MLX server, vLLM, etc.) |
| capabilities | Declared set: text, vision, audio, tools, json, reasoning-effort |
| context_window | Tokens |
| residency | resident, on-demand |
| speed_class | Coarse relative label (fast, mid, slow) used only for tie-breaking |

Every field resolves in a fixed order: user declaration > detected value > heuristic default. Each resolved field carries provenance (`declared`, `detected`, `default`) visible in config output and logs, so a user can always see why the router believes what it believes. A declaration always wins, even when detection disagrees; disagreement is logged as a warning, never silently corrected.

### 3.1a Discovery

Three detection layers, in decreasing reliability:

| Layer | Source | Yields |
|---|---|---|
| Backend metadata | llama.cpp `/props`, Ollama `/api/show` (including its capabilities field), LM Studio `/api/v0/models`, vLLM `/v1/models` `max_model_len`; generic `/v1/models` as the weakest fallback | context_window, capabilities where exposed, family, quantization, loaded state |
| Active probes (opt-in per model, lazy) | Trivial request with a tool definition; tiny image; response_format request; timed micro-benchmark at fixed token count run serially | tools, vision, json verified by response shape; speed_class from measured TTFT and tok/s |
| Name heuristics | Model id substrings ("-vl-", "coder", "embed", "abliterated") | Tie-breaking hints only; never sole basis for a capability |

Probe constraints: a probe against an on-demand model forces a full model load, which on constrained installs is precisely the cost the user configured around. Probes therefore default off for `residency: on-demand` entries and run only on explicit request or first natural use. Speed benchmarks run serially, never concurrently, to avoid contention-skewed numbers.

Discovery output is a generated registry-and-roles proposal the user reviews and accepts or edits (`--discover` writes it; nothing routes on it until accepted). Capabilities constrain role eligibility, but role assignment is a judgment: detection can establish that a model can see images, not that it should front the vision cascade. Proposed role assignments are heuristic defaults in the same resolution order as everything else.

Tracking: detection results are cached keyed on model id plus digest or version where the backend exposes one, and re-run when the key changes. The router watches backend model lists for additions and removals and flags registry drift at startup and in logs; it does not silently add or drop routing targets.

### 3.2 Roles

A role names a class of work and carries a cascade:

```yaml
roles:
  chat:
    cascade: [qwen-fast, main-model]
  code:
    cascade: [main-model]
  vision:
    cascade: [main-model]        # a multimodal main needs no separate vision model
  unfiltered:
    cascade: [qwen-abliterated]
routing:
  default_role: chat
```

Cascade position 0 is the role's main model. Positions 1..n are the escalation path, walked in order on escalation triggers, capped by a configurable hop limit (default 1).

The example above is one user. Another user's `vision` cascade might be `[llava-small, main-model]`; another has no `vision` role at all, and image-bearing requests fail capability filtering with a clear error. Both are valid installs.

### 3.3 Selection order

For each request: capability filter (which roles/models are even eligible) → role classification (which role this turn belongs to) → cascade position (affinity and escalation state decide how far down the cascade the session currently sits).

---

## 4. Functional requirements

Priority: MUST (v1), SHOULD (deferrable), MAY (optional).

### FR-1 — Transparent OpenAI-compatible interface (MUST)

One logical model name in; routing behind it. The response identifies the serving model in an ignorable field. Streaming supported (see FR-8).

### FR-2 — No operator annotation (MUST)

No per-prompt, per-task, or per-session model input. Configuration happens once, in the registry and roles file.

### FR-3 — Capability filtering (MUST)

Hard-filter candidates on observable request properties against declared capabilities:

| Property | Constraint |
|---|---|
| Image (or audio) anywhere in message history, not only the current turn | Candidate declares vision (or audio) |
| Tool definitions present | Candidate declares tools |
| Estimated token count | Candidate context_window accommodates it |

History-wide, because media introduced at turn 1 remains in context at turn 20. Zero inference cost.

If filtering eliminates every model in every eligible cascade, return a clear error naming the unsatisfiable constraint (FR-9). Never silently route media to a text-only model.

### FR-4 — Role classification (MUST)

Among surviving candidates, classify the current message into a role by embedding similarity against per-role example utterances supplied in config. The router ships with default utterance sets per common role name (chat, code, vision, unfiltered) that users can override.

The embedding model is itself a registry entry the user provides. If none is configured, classification degrades to `default_role` plus capability filtering, and the router says so at startup.

LLM-based routing calls are rejected by design: on shared local hardware they spend a full generation to make a decision an embedding pass makes in milliseconds.

### FR-5 — Session affinity (MUST)

Once a session is routed to a model, subsequent requests follow it until a transition (FR-6). Affinity is keyed on a caller-supplied session identifier (header or request field), with idle-timeout expiry. History-hash derivation is a fallback (MAY) and is fragile under compaction.

### FR-6 — Controlled transitions (MUST)

Reclassify only at observable transitions; otherwise hold:

| Transition | Rationale |
|---|---|
| Material tool-set change | Phase change (discussion → implementation) |
| Classifier disagreement beyond configured margin | Intent shifted |
| Context compaction signal | Re-prefill is already being paid; switching is free |
| Escalation trigger (FR-7) | Current model demonstrably failed |

A 20-step tool chain stays on one model absent a trigger.

Transition cost is configuration-dependent: on a resident-everything install it is one re-prefill; on a load-on-demand install it is re-prefill plus model load. The router exposes a per-install `switch_cost` hint (low, high) that biases how eagerly transitions fire. High switch cost raises thresholds; it does not change the mechanism.

### FR-7 — Cascade escalation (MUST)

Walk the session's cascade one position on mechanically observable failure:

| Signal | Detection |
|---|---|
| Tool call fails to parse against the supplied schema | Router validates emitted calls; requires per-model-family format support, which is real implementation cost |
| No tool call where the request required one | Structural check |
| Identical call with identical arguments repeated past threshold | Per-session recent-call memory |
| Refusal | Advisory only; string-matching is brittle. MAY, off by default |

Hop cap configurable, default 1. Never triggered by model self-assessment.

### FR-8 — Streaming behavior (MUST)

Once tokens stream, the turn is committed; escalation applies to the next turn. An optional buffered mode (SHOULD) may hold tool-call-bearing responses for validation before release, trading latency for retractability. Buffered mode is per-role config, since a batch role tolerates latency an interactive role cannot.

### FR-9 — Degraded operation (MUST)

| Condition | Behavior |
|---|---|
| Embedding model unavailable | default_role + capability filter; do not fail the request |
| All candidates filtered out | Clear error naming the constraint |
| Affinity map lost | Reclassify; cost is one re-prefill per active session |
| Selected backend unreachable | Next cascade position, or next eligible role; log; do not fail |

The router must never be the component that kills an unattended run.

### FR-10 — Config validation (MUST)

Validate at startup: cascade entries reference registered models; every cascade's declared role is satisfiable by at least its position-0 model; endpoints respond. Where detection is available and disagrees with a declaration (a declared 1M context the backend reports as 256k; declared vision the probe cannot confirm), warn with both values and provenance, and proceed on the declaration. Fail loudly at startup, never silently at request time. A `--check` mode validates without serving; `--discover` emits a proposed config from detection without serving.

### FR-15 — Model discovery (SHOULD)

Implement §3.1a: metadata harvesting on registration and startup (MUST within this FR), opt-in lazy probes (SHOULD), name heuristics as tie-breakers (MAY), proposal generation via `--discover` (SHOULD), and drift flagging when backend model lists change (SHOULD). Discovery must be able to fail entirely, on a backend exposing nothing beyond model ids, and leave the router fully usable on declarations alone.

### FR-11 — Decision logging (SHOULD)

Per decision: session id, role, model, cascade position, trigger, decision type, routing latency. Escalation rate per role is the primary tuning signal.

### FR-12 — Network binding (MUST)

Loopback by default; explicit opt-in to other interfaces. The router fronts models with tool access.

### FR-13 — Manual override (SHOULD)

Pin a session to a named model for debugging. Frequency of use is a failure signal (AC-1).

### FR-14 — Portability and packaging (MUST)

Runs on macOS and Linux against any OpenAI-compatible backend. Single config file. No database daemon; state that must persist (affinity is allowed to be lost, config is not) lives in files. Installable without Hermes-specific code; Hermes integration points (compaction signal, session id convention) are documented conventions, not hard dependencies, so the router works with other OpenAI-compatible harnesses unmodified.

---

## 5. Non-functional requirements

| ID | Requirement | Priority |
|---|---|---|
| NFR-1 | Cached-affinity path adds negligible latency; classification tens of milliseconds | MUST |
| NFR-2 | No outbound network calls | MUST |
| NFR-3 | Crash resilience: affinity loss degrades to reclassification | MUST |
| NFR-4 | No additional database daemon; minimal runtime footprint | SHOULD |
| NFR-5 | Configurable cap on concurrent in-flight requests across distinct models (bandwidth contention on shared-memory hosts) | SHOULD |
| NFR-6 | Unattended restart under the host's service manager (launchd, systemd) | MUST |
| NFR-7 | Degenerate config (one model, one role) measurably indistinguishable from a direct connection | MUST |

---

## 6. Non-goals

- Cost tracking, budgets, pricing. Router is local-first; cost policies are out of scope even where users mix in hosted endpoints.
- Multi-tenancy, RBAC.
- Model download, quantization, lifecycle, or load management. Residency is declared, not managed; backends own loading.
- Continuous or quality benchmarking. Discovery probes (§3.1a) establish capability presence and coarse speed once; the router does not score output quality, run eval suites, or re-benchmark on a schedule.
- Routing among non-conversational tool models (image gen, embed, rerank, ASR, TTS): those are tools invoked by the active model, not routing targets. The embedding model used for FR-4 is infrastructure, not a target.
- Task-level routing. Hermes Kanban routes tasks to profiles; this router operates below that. Both on the same traffic is a config error worth documenting (profiles should target the router uniformly if it is adopted).

---

## 7. Dependencies and open questions

Blocking, unverified:

1. Hermes profiles accept arbitrary base URL + logical model name (probable: Ollama/vLLM/llama.cpp are documented backends).
2. A stable per-session identifier is obtainable from the calling harness.
3. A compaction signal is obtainable, or the cheapest transition point is lost.

Open, ordered by consequence:

1. **Does a given install need routing at all.** The null hypothesis is route-everything-to-main. The router should ship with guidance: enable roles only when a single model demonstrably falls short (latency on mechanical turns, refusals, missing capability). Degenerate config (NFR-7) makes "install now, route later" safe.
2. **Fast-tier tool-call reliability on ablated models.** If parse rates are poor, optimistic cascades are unviable for that role and the right config is main-first.
3. **Re-prefill cost curves per install class** (resident vs on-demand) to calibrate `switch_cost`.
4. **Silent degradation.** All escalation signals are mechanical; well-formed-but-wrong output triggers nothing. This is the design's blind spot and argues for conservative default cascades (strong model at position 0, fast model as an explicit opt-in).
5. **Prior art.** One targeted search for session-affinity local gateways should precede a build.

---

## 8. Acceptance criteria

Measured on two installs: the reference (all-resident, multi-role) and a constrained one (two models, on-demand loading).

| # | Criterion | Measure |
|---|---|---|
| AC-1 | No manual model selection in normal operation | Zero FR-13 overrides over 30 days |
| AC-2 | Phase transition | Scripted discussion→implementation conversation switches exactly once, at the boundary |
| AC-3 | Chain affinity | 20-step tool chain, zero mid-chain switches absent a trigger |
| AC-4 | Unattended safety | 4-hour run, zero routing-attributable failures, on both installs |
| AC-5 | Overhead | Cached-affinity latency under 1% of median turn inference |
| AC-6 | Escalation rate sane per role | 5–30% of sessions; outside that band the cascade order is wrong |
| AC-7 | Reliability gate | Tool-call success under routing within noise of single-model operation. If routing is less reliable than main-only, remove it |
| AC-8 | Degenerate config | Single-model config: throughput and latency within noise of a direct backend connection |
| AC-9 | Portability | Same binary + different config runs both test installs unmodified |
| AC-10 | Discovery accuracy | On the reference install, `--discover` proposes context windows and capabilities matching ground truth for every model whose backend exposes metadata, and misdeclares nothing it probed |

Failure modes to watch: thrashing (repeated transitions in one conversation → raise margins), sticky-wrong (pinned to unsuitable model → add the missed signal), escalation loops (hop cap must hold), and silent degradation (undetectable by design; mitigated only by cascade order).

---

## 9. Sequencing (reference install)

1. Inference servers up, one model; Hermes migrated
2. Primary-tier benchmark
3. Null-hypothesis test: main-only across the real workload. Stop if sufficient
4. Remaining models resident; fast-tier tool-call bench. Fall back to main-first cascades if weak
5. Router, degenerate config first (NFR-7/AC-8), then roles

The generalization does not change the order: even as a distributable tool, its first proving ground is one install measured honestly.
