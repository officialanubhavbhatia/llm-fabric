# MyVista LLM Fabric — Architecture

> **On the engineering constitution.** The opening instruction for this repository
> referenced an attached engineering constitution to be treated as the permanent
> architecture and engineering specification. **That attachment never reached the
> session** — the cover note arrived twice, the document neither time. The
> workspace, `/tmp`, the home directory, and the artifact store were all searched.
>
> Phase 1 was therefore built against the architecture inferred from the
> repository description, with the stack chosen deliberately and recorded as a
> reversible decision in [`docs/adr/0001`](docs/adr/0001-language-and-runtime.md).
> Sections marked **Built** describe code that exists and is tested. Sections
> marked **Not built** are honest gaps. When the constitution arrives, commit it to
> `docs/constitution.md`; it overrides everything here, and
> [§7](#7-reconciling-with-the-constitution) states what each kind of
> disagreement would cost.

---

## 1. Repository inspection

Inspection was completed before any code was written.

**Starting state.** The GitHub repository `officialanubhavbhatia/llm-fabric` was
created empty on 2026-08-23, described as "LLM Gateway, Serving and Inferencing
Layers", public, default branch `main`, with no commits. The working tree held
only placeholder files written during this session. There was no application
code, build system, dependency manifest, test suite, CI configuration, or
infrastructure definition.

**Consequence.** Greenfield: nothing to preserve, no interface to keep
compatible, no migration to plan — and no inherited constraints to shape Phase 1
either, which is precisely what the constitution would have supplied.

---

## 2. The system

A **fabric** is a single control point that accepts inference traffic, decides
where each request should run, executes it against external provider APIs or
self-hosted model servers, and stays accountable for cost and reliability.

Its central claim is that **callers depend on one surface, not on providers.** A
client sends a request to the fabric and never encodes which vendor serves it.
That is what makes providers substitutable, and it is why the routing decision is
a first-class object with its own provenance rather than an implementation detail
buried in a proxy.

```
client
  │
  ├─ gateway      auth, limits, normalize            §3
  ├─ router       registry + policy → decision       §4
  ├─ serving      provider adapter | self-hosted     §5
  └─ observability  meter, trace, explain            §6
```

---

## 3. Gateway layer — Built

`src/llm_fabric/gateway/`, with the public schema isolated in
`src/llm_fabric/contract/`.

The stable public surface. Everything that must happen before a routing decision
is possible lives here.

| Endpoint | Purpose |
| --- | --- |
| `POST /v1/chat/completions` | Inference, buffered or streamed over SSE |
| `GET /v1/models`, `GET /v1/models/{id}` | Discovery, including aliases |
| `GET /v1/usage` | Usage totals and recent routing decisions |
| `GET /healthz`, `GET /readyz` | Liveness and readiness |
| `GET /docs` | Generated OpenAPI reference |

Three decisions worth stating:

**The dialect is OpenAI-compatible** so existing clients and SDKs can point at
the fabric without code changes — see
[`docs/adr/0002`](docs/adr/0002-openai-compatible-contract.md). Fields the fabric
does not yet act on are accepted rather than rejected, so unmodified clients do
not break; [`docs/CONTRACT.md`](docs/CONTRACT.md) records exactly which fields
are honoured and which are currently inert.

**Liveness and readiness are separate.** A fabric with an empty registry is alive
but cannot serve anything. Conflating the two would keep traffic arriving at a
gateway with nowhere to send it, so `/readyz` returns 503 when no model is
enabled.

**Every error uses one envelope** — `{"error": {message, type, request_id}}` —
including schema validation failures, which are reshaped from pydantic's format.
A client has exactly one error shape to parse.

Authentication is API-key based, accepting `Authorization: Bearer` or `x-api-key`,
compared in constant time. With no keys configured the gateway runs open, which
is a local-development affordance and is logged as a warning at startup so an
open deployment is never silent. Keys are never stored or logged — only a
truncated SHA-256 fingerprint reaches a metering record.

**Not built:** rate limiting, quotas, spend ceilings, and multi-tenant identity.
The layer is the right home for them and they are absent.

---

## 4. Routing layer — Built

`src/llm_fabric/router/`.

### The registry

`config/models.yaml`, loaded into `ModelRegistry`. Declarative on purpose:
adding a model, repricing one, or taking one out of rotation is a config change,
not a deploy. References are validated at load time, so a typo in a fallback
chain fails at startup rather than during a production failover.

Two entry kinds:

- a **model** maps a fabric-facing id onto one provider and that provider's own
  model name, with cost, context window, capabilities, and a fallback chain;
- an **alias** is a virtual id resolving to several models under a policy.
  `auto` is an alias.

Costs are USD per million tokens and are **operator-supplied inputs, not figures
the fabric measures**. The `cheapest` policy ranks candidates using exactly these
numbers, so a wrong price produces a wrong route. The shipped registry leaves
external-provider prices at zero and disabled, to be filled in from the
provider's current pricing page.

### Policies

`cheapest` orders by a blended price weighting output tokens more heavily than
input, since generation is the priced-heavier side for most providers. It is a
ranking heuristic, not a spend prediction. `declared` preserves registry order,
letting the operator's stated preference win.

**A latency-aware policy is deliberately absent.** Ordering by latency requires
per-backend latency measurement, and the fabric does not yet collect it. Shipping
the policy first would mean ranking on numbers that do not exist.

A pinned model is never reordered — the caller asked for it explicitly — but its
declared fallbacks trail it.

### Failover

Candidates are tried in policy order. Only a retryable failure advances to the
next candidate; a caller error stops immediately, because retrying a malformed
request just fails again more slowly. `max_attempts` bounds the whole chain.

**Failover is forbidden once a stream has produced its first byte.** The client
has already committed to a response it cannot un-see, and splicing a second
model's output onto it would corrupt the response rather than save it. The engine
enforces that boundary centrally instead of trusting each adapter —
[`docs/adr/0003`](docs/adr/0003-no-failover-after-first-streamed-byte.md).

**Not built:** circuit breaking, response caching, and semantic caching. Failure
handling is per-request; a backend that is down is rediscovered on every request
rather than tripped out of rotation.

---

## 5. Serving layer — Built

`src/llm_fabric/serving/`.

One `Provider` interface — `generate`, `stream`, `aclose` — over every backend.
Adding a backend means adding one subclass in `serving/adapters/`; neither the
gateway nor the router changes, because neither knows what a provider is beyond
that interface.

| Adapter | Notes |
| --- | --- |
| `mock` | No credentials. Performs **no inference**; returns text assembled from the request. Exists so the fabric runs and is testable out of the box, and can be told to fail on demand to exercise failover. |
| `openai` | Chat-completions API, requesting `stream_options.include_usage` so streamed responses still carry real token counts. |
| `anthropic` | Messages API. Absorbs the dialect differences — system prompt as a top-level field, `max_tokens` required — so they never leak upward. |

Adapters share one job beyond transport: **classifying failures**. Timeouts,
transport errors, and 429/5xx become retryable; a 4xx that is the caller's fault
does not. That classification is what the router's failover consumes, so it is
centralised in `adapters/_http.py` rather than repeated per provider.

**Not built:** self-hosted serving. No engine integration, replica pool
selection, or KV-cache-aware placement exists. This is the largest gap against
the "Serving and Inferencing Layers" in the repository description, and the
`Provider` interface is where it would attach.

---

## 6. Observability — Built

`src/llm_fabric/observability/`.

Every served request produces a `UsageRecord` naming the model requested, the
model actually used, the policy that chose it, token counts, cost at registry
prices, latency, and **every attempt made including the failures**. A route can
therefore be explained after the fact instead of guessed at. The same provenance
is returned to the caller in `x-fabric-*` headers, and in the final SSE chunk for
streamed responses, so a client sending `auto` always learns what served it.

Honesty is enforced structurally. When a backend reports no token counts, the
fabric estimates them with an explicit heuristic in `serving/tokens.py`, and the
record carries `cost_is_estimated: true`. `/v1/usage` reports how many requests
that applies to. An estimated cost is never presented as a measured one.

Logs are one JSON object per line, so a pipeline can query them without regex.

**Not built:** the metering sink is in-memory and bounded — **not durable**.
Records are lost on restart and visible only to the process that made them. There
is no persistence, no aggregation across replicas, and no distributed tracing.
`/v1/usage` states this in its own response rather than implying a durable ledger.

---

## 7. Reconciling with the constitution

What each kind of disagreement costs, stated plainly so the decision is informed:

| If the constitution mandates | Cost |
| --- | --- |
| A different language or runtime | Full rewrite. The layering and the contract survive as a design; the code does not. |
| A different API dialect | Contained. `contract/` is isolated from the layers behind it for this reason. |
| Different routing policies | Small. Policies are pure functions in one module with a registry. |
| A different registry schema | Small. One loader, validated in one place. |
| Self-hosted serving in Phase 1 | Additive. A new `Provider` implementation; no change above it. |
| Persistent metering | Contained. `MeteringSink` is already a protocol; the in-memory class is one implementation. |

Still needed from the specification: phase definitions and acceptance criteria,
the deployment target, whether self-hosted serving belongs in an early phase,
state dependencies, and the non-functional targets the design must actually meet.

---

## 8. Engineering rules in force

- **Authorship.** All production code is authored under Anubhav Bhatia. No AI or
  Cursor authorship, co-author trailers, or generated-by markers.
- **No fabrication.** No benchmark, metric, or capability claim appears unless it
  was actually produced by a reproducible run. This repository contains **no
  performance measurements of any kind** — no latency, throughput, or cost
  benchmarks have been run, and none are claimed. The only quantitative claim
  made anywhere is the test count, which is reproducible with `pytest`.
- **Unbuilt means labelled unbuilt**, as in §3 through §6 above.
- **Scope.** Work stays in this repository. No other project is modified.
