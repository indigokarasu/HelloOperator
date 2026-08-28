# HelloOperator

*"Hello, operator? Get me..." You dial one number. The switchboard does the rest.*

HelloOperator sits in front of the models you already run and picks which one
handles each conversation turn. Clients see a single OpenAI-compatible endpoint
with a single model name. Behind it, requests get matched to a role (chat, code,
vision, whatever you define), sessions stick with the model they started on, and
when a model fails in a way the router can actually observe, the next one in
line takes over.

It ships with no opinions about which models are good. You declare what you run
and it routes among that. Configure exactly one model and it behaves like a
plain proxy that stays out of the way.

The requirements doc it implements is at `docs/requirements-v0.4.md`.

## Install

```bash
pip install .
hello-operator -c config.yaml
```

Two dependencies: `aiohttp` and `PyYAML`. There is no database; persistent
state is a few files under `state_dir`. Works on macOS and Linux against
anything that speaks the OpenAI API: llama.cpp, Ollama, LM Studio, vLLM, an
MLX server, or a hosted provider.

## Quick start

1. Copy `config.example.yaml` to `config.yaml` and declare your models.
2. Run `hello-operator --check`. It validates the config, confirms your
   endpoints answer, and prints every resolved field along with where the
   value came from (declared, detected, or default). Bad config fails here at
   startup instead of surprising you at request time.
3. Run `hello-operator` and point your client at `http://127.0.0.1:8800/v1`
   with model `hello-operator`.

The smallest useful config is one model and nothing else:

```yaml
models:
  main:
    id: qwen3-30b
    endpoint: http://127.0.0.1:8080/v1
```

Roles and cascades are there when you want them. Start with the proxy, add
routing once a single model stops being enough.

## Discovery

```bash
hello-operator --discover           # metadata only
hello-operator --discover --probe   # also run the active probes, one at a time
```

`--discover` asks your backends what they know and writes out a proposed
config: context windows and capabilities from llama.cpp `/props`, Ollama
`/api/show`, LM Studio's `/api/v0/models`, or vLLM's `max_model_len`, plus
optional probes that verify tool calling, vision, and JSON mode by actually
trying them. A short timed run estimates each model's speed class. Name
heuristics ("-vl-", "coder", "embed") only break ties; they never assert a
capability on their own.

Nothing routes on the proposal until you adopt it as your config. And your own
declarations always beat detection. If a backend disagrees with something you
declared, you get a warning with both values, and the router proceeds on yours.

Probes are off by default for `residency: on-demand` models, because probing
one forces a full model load, which is usually the exact cost that setting
exists to avoid. Same for remote models, where every probe is a paid request.

At startup the router also flags registry drift: models a backend serves that
you never registered, and registered models a backend has stopped serving. It
tells you and changes nothing.

## How a request gets routed

Three steps, in order: capability filter, role classification, cascade
position.

1. **Capability filter.** An image or audio part anywhere in the message
   history requires a model that declares `vision` or `audio`. Tool
   definitions require `tools`. The estimated token count has to fit the
   model's context window. This is a hard filter with zero inference cost,
   and if nothing survives it, you get an error naming the exact constraint.
   Media never gets quietly routed to a text-only model.
2. **Role classification.** The current message is scored by embedding
   similarity against example utterances for each role. Defaults ship for
   `chat`, `code`, `vision`, and `unfiltered`; override them per role. The
   embedding model is a registry entry you provide. Without one,
   classification falls back to `default_role` plus the capability filter,
   and the router says so at startup. Using an LLM call to route was rejected
   on principle: on shared local hardware it spends a whole generation making
   a decision an embedding pass makes in milliseconds.
3. **Cascade position.** Session affinity and escalation state decide how far
   down the role's cascade the session currently sits.

### Affinity and transitions

Once a session is routed, it follows its model. The session key comes from the
`x-session-id` header, then `session_id` or `user` in the body, then a history
hash as a last resort (the hash is fragile under compaction, which is why it's
last). A twenty-step tool chain stays on one model the whole way through.

Reclassification only happens at points where something observable changed:

| Transition | Why |
|---|---|
| The tool set changed materially | Phase change, e.g. discussion to implementation |
| The classifier disagrees beyond `classify_margin` | Intent shifted |
| Compaction (`x-context-compacted` header, or a large context drop) | The re-prefill is being paid anyway, so switching is free |
| An escalation trigger fired | The current model demonstrably failed |

`switch_cost: high` raises those thresholds for installs where switching
means loading a model from disk. It only changes how eager the router is to
move; the decision logic stays the same.

### Escalation

Cascades walk one position on mechanically observable failure. The router
never asks a model whether it's struggling; it checks things it can verify:

- a tool call whose arguments fail to parse against the supplied schema
- no tool call where `tool_choice` required one
- the same call with the same arguments repeated past `repeat_call_threshold`
- refusal string markers, if you turn them on (advisory, off by default,
  because string matching is brittle)

Non-streaming responses are validated before release and retried one cascade
position down within the same request, up to `hop_limit`. Streaming is
different: once tokens reach the client the turn is committed, so validation
runs on the assembled stream and any escalation applies to the next turn. If
you want retractability for a tool-heavy role, set `buffered: true` on it and
the router will hold streaming responses for validation before releasing them.
That trades latency for the ability to swap models before the client sees
anything.

### When things break

The router should never be the reason an unattended run dies. The degraded
behaviors:

| Condition | Behavior |
|---|---|
| Embedding model down | `default_role` plus capability filter; the request proceeds |
| Every model filtered out | An error naming the constraint |
| Affinity lost (restart) | Reclassify; costs one re-prefill per active session |
| Backend unreachable, 5xx, or rate-capped | Next cascade position, then any eligible model inside the role's locality fence; logged |

## Observability

Every response carries headers saying what happened: `x-router-model`,
`x-router-backend-model`, `x-router-role`, `x-router-pos`,
`x-router-decision`, and `x-router-latency-ms`. Non-streaming bodies also get
an ignorable `router` object.

Set `decision_log` and you get one JSONL line per routing decision: session,
role, model, cascade position, trigger, decision type, latency. The number to
watch is escalation rate per role. Somewhere between 5% and 30% of sessions is
healthy. Much outside that band and your cascade order is probably wrong.

`GET /healthz` reports the mode, classification state, and live session count.

## Manual override

Pin a session with the `x-router-pin: <model-key>` header, or by naming a
registered model directly in the `model` field. Pins skip classification but
never skip the capability filter. If you find yourself pinning often, that's
the router telling you its config is wrong.

## Mixing cloud and local models

Hosted endpoints (OpenRouter, OpenAI, Anthropic's compatibility endpoint) are
just registry entries. Mixing them with local models raises questions that a
router shouldn't answer silently, so three rules govern it.

Remote is opt-in. Every model resolves a `location` of `local` or
`remote`, declared or inferred from the endpoint host. Loopback, RFC-1918
addresses, and `.local`/`.lan` names count as local. If anything resolves
remote, the router refuses to start until you set
`router.allow_remote_endpoints: true`, and once you do, it names exactly which
models' traffic leaves the machine. A tailnet host that looks remote can be
declared `location: local`, and the declaration wins.

Role locality is a fence that failover can't cross. A role with
`locality: local` only accepts local models in its cascade (checked at
startup), and a session routed under it will never fail over to a remote
model, even through another role's cascade. The constraint stays with the
conversation itself, whatever happens to be alive at the moment. So if the
only live model is remote and the role is fenced local, you get a 502
explaining why, instead of your conversation quietly leaving the machine.

And the registry's `api_key` beats the client's bearer token. Your local
harness token never gets forwarded to a hosted provider. Client auth passes
through only to backends with no declared key of their own.

Cost tracking stays out of scope, but the defaults are shaped by cost anyway:
probes stay off for remote models unless you opt in per model, and a hosted
catalog serving hundreds of models is never reported as drift. Only a
registered model going missing is.

There's a nice side effect for offline use. A cascade like
`[cloud-big, local-mid]` degrades to the local model the moment your uplink
dies, and the decision log records the failover. Quota exhaustion works the
same way: a 429 or 402 from a hosted free tier walks the cascade instead of
ending your turn.

## Conventions, not dependencies

The Hermes integration points are documented conventions, so the router works
with any OpenAI-compatible harness unmodified:

- session identity: the `x-session-id` header, or `session_id` / `user` in
  the body
- compaction signal: an `x-context-compacted: true` header on the first
  request after a context compaction (a large context drop is also detected
  heuristically)

## Deploy

`deploy/` has a systemd unit for Linux and a launchd plist for macOS, for
unattended restarts under the host's service manager. The router binds
loopback by default. It fronts models with tool access, so binding any other
interface requires the explicit `allow_non_loopback: true`.

## Should this install route at all?

Honest answer: maybe not. The null hypothesis is route-everything-to-main, and
the requirements doc holds routing to a hard bar (AC-7): if routing is less
reliable than a single model, remove it. Enable roles only when one model
demonstrably falls short, whether that's latency on mechanical turns,
refusals, or a missing capability. The single-model config makes "install now,
route later" safe.

One blind spot worth knowing about: every escalation signal is mechanical, so
output that is well-formed but wrong triggers nothing. That argues for
conservative cascades, with your strongest model at position 0 and the fast
model as a deliberate opt-in.

## Tests

```bash
pip install -e ".[dev]"
python -m pytest tests/
```

70 tests, all through real loopback sockets against a scriptable fake backend:
config resolution and provenance, capability filtering, classification,
affinity, every transition and escalation trigger, streaming commit semantics
(including a backend dying mid-stream), failover (including backends that
answer with HTML, go unreachable, or hit rate caps), discovery against Ollama,
llama.cpp, and generic endpoints, drift, mixed local/cloud fleets with the
locality fence, and the single-model path. `tests/test_review_regressions.py`
has one test per defect found in the pre-release adversarial review. If any
of those bugs comes back, a test fails.
