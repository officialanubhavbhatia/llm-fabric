# MyVista LLM Fabric — Architecture

> **On the engineering constitution.** The constitution arrived on 2026-08-23,
> after Phase 1 was built, and is committed verbatim at
> [`docs/constitution.md`](docs/constitution.md). **It is the authoritative
> specification and overrides this document.**
>
> Phase 1 was built against an architecture inferred from the repository
> description, because the specification had not yet reached the session. This
> document has not been rewritten to match the constitution; doing so before the
> code matches would replace an honest record with an aspirational one. Sections
> marked **Built** describe code that exists and is tested. Sections marked **Not
> built** are honest gaps. [§9](#9-reconciling-with-the-constitution) is the
> reconciliation: what is built to specification, what is not, and what is simply
> absent.
>
> **Phase 2 (identity and tenancy) is built** and is described in
> [§7](#7-identity-and-tenancy--built). It was built against the constitution
> rather than against inference, and it replaces the Phase 1 API-key stopgap.
>
> **Phase 3 (IntentOS) is built** and is described in
> [§8](#8-intentos--built). Its measured results — and the reasons they prove
> much less than they might appear to — are in
> [`docs/EVALUATIONS.md`](docs/EVALUATIONS.md).
>
> **Phase 4 (routing fabric) is built** and is described in
> [§4](#4-routing-fabric--built), which replaced the Phase 1 price-sorting
> router. Grades, capability vectors, the seven policies, health-aware routing,
> circuit breakers, the fallback graph, the auditable decision object and the
> route preview API are in the tree. A labelled planner-match eval exists
> (`route_match`, `policy_match` against the mock registry). That is a
> regression tripwire, not a routing-quality evaluation: it scores whether the
> planner hit a fixture label, not whether the route is a good one.
>
> **Phase 7 (observability command center) is built** and is described in
> [§6](#6-observability--built). OpenTelemetry spans cover the stages that run;
> Prometheus scrapes bounded-cardinality labels; Langfuse is an optional adapter;
> the Command Center serves every named view and marks unbuilt backends as
> unavailable rather than inventing series.
>
> **Phase 8 (evaluation platform) is built** and lives in `src/llm_fabric/eval/`.
> Suites, runs, comparisons and gates exist. DeepEval and lm-evaluation-harness
> are adapters that report unavailable when their packages are missing. Agent and
> safety eval suites are not built. Classification numbers remain the same
> self-authored bootstrap tripwire documented in
> [`docs/EVALUATIONS.md`](docs/EVALUATIONS.md).
>
> **Phase 9 (self-healing and drift) is built** and lives in `src/llm_fabric/heal/`.
> Model and provider health scores are aggregates of this process's
> `HealthTracker` observations. Drift compares two usage windows (or a stored
> baseline). Embedding, compiler context-length and safety-block drift stay
> unavailable. Remediation can open a held circuit breaker, shift traffic,
> roll back a remembered model spec, prompt version or pinned classifier,
> reduce a serving-path context ceiling, invalidate caches, raise an incident,
> or queue a learning job. Learning jobs are not trained or promoted without a
> passing evaluation. Authorization and safety policy cannot be mutated from
> this path.
>
> **Phase 10 (performance) extends the load harness.** Isolated in-process
> benches cover the gateway, auth, both intent caches, the classifier, the
> router, streaming and the full mock serving path. Ollama and vLLM inference
> stay unavailable. HTTP load remains `llm-fabric-load` against a running
> server. Techniques such as continuous batching and speculative decoding are
> not production defaults: their backends are unbuilt, and a gain has not been
> measured. Results are written as versioned JSON under `artifacts/bench/`.
>
> **Phase 11 (SDK) is built.** `from myvista import MyVista` is a thin HTTP
> client: sync, async, streaming, retries, timeouts, typed errors, and
> `x-fabric-request-id`. Chat and `responses.create` talk to the existing
> completions route. Classification, route preview, named eval (`ci`) and
> traces are real endpoints. Embeddings and agents raise `UnsupportedError`
> rather than inventing a backend. A TypeScript client lives in
> `sdk/typescript`. Ordinary inference does not require infrastructure config.
>
> **Phase 12 (production readiness) is a review, not a go-live.** The verdict
> and the suite results are in [`PRODUCTION_READINESS.md`](PRODUCTION_READINESS.md).
> This build is **not** production-ready: P0 and P1 issues remain.
>
> **Scope warning.** Phases 1 to 4 and 7 to 12 are a small fraction of the specified
> system. Phases 5 and 6 (context compiler, guardrails) are not built. Read §9
> before treating anything below as the target architecture.

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
either, which is precisely what the constitution would have supplied. It arrived
after Phase 1 was built; §9 records the resulting divergence.

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
  ├─ identity     validate token → principal         §7
  ├─ gateway      authorize, admit, normalize        §3
  ├─ intent       cascade → classification           §8
  ├─ router       plan → decision → execute          §4
  ├─ serving      provider adapter | self-hosted     §5
  ├─ tenancy      scoped storage, cache, quotas      §7
  └─ observability  meter, trace, explain            §6
```

The router now reads an `IntentClassification` when one is present, and can infer
a policy from it. Classification on the serving path is **off by default**
(`LLM_FABRIC_INTENT_CLASSIFICATION_ENABLED`), because putting a cascade in front
of every request adds the 0.58 ms p50 in-process cost measured in
[`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) §10. The extra HTTP cost of turning
it on has not been measured; §4 and §8 describe both halves of that wire.

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

**Liveness, readiness and admission are separate.**

- `/healthz` is process liveness. PostgreSQL, Redis or OTEL being down does
  not fail it. Conflating liveness with dependency health would restart pods
  during an outage they cannot fix.
- `/readyz` is whether this instance is safe to receive **new** production
  serving traffic. It is 503 when PostgreSQL or Redis (when this process uses
  them) is unhealthy, or when no enabled model has a constructible provider.
  OTEL is fail-soft and does not remove readiness. Individual inference
  backends belong to routing, not this signal.
- Admission control on `POST /v1/chat/completions` refuses new provider
  calls from the same cached health state. Readiness is a signal to
  infrastructure; requests can still arrive over keep-alive connections or
  directly to a pod. Admission is the correctness boundary.

`/readyz` does not open a new Postgres or Redis connection on every probe.
A background monitor updates cached state on a bounded interval; a serving-path
failure marks the dependency unhealthy immediately.

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

## 4. Routing fabric — Built

`src/llm_fabric/router/`. Rewritten in Phase 4 against the constitution's
*Routing engine* and *Model grading* sections.

Routing is split into two halves that never blur into each other. **Planning**
is pure: given a request, the registry and the current health snapshot, it
returns a `RoutePlan` and touches no provider. **Execution** takes that plan and
spends money against it. Everything explainable is decided in the first half,
which is what makes the preview endpoint able to answer honestly without
performing inference.

### Grades and capability vectors

`Grade00`–`Grade29` are **logical capability bands, not model names**. A grade is
a slot an operator maps deployments into; nothing in the code knows that
`Grade27` means any particular vendor's model, and that indirection is the point,
because it is what lets a model be replaced without rewriting a route.

`CapabilityVector` is a normalised set with implication: declaring
`json_schema` implies `structured_output`, so a requirement expressed either way
matches. Unknown capability names are preserved rather than dropped, since
silently discarding a requirement would widen a route instead of narrowing it.

### The registry

`config/models.yaml`, loaded into `ModelRegistry`, validated at load time so a
typo in a fallback chain fails at startup rather than during a production
failover. A deployment carries its grade, capability vector, context window,
prices, per-dimension quality scores, a declared performance profile, and a
`Placement` giving region, hardware and **locality**.

Locality is a privacy boundary, not a performance one: `local` is the fabric's
own machine, `private` is operator-controlled infrastructure reached over a
network, and `external` means a third party receives the prompt. **A deployment
that does not declare a locality is treated as `external`**, because the failure
mode of guessing the other way is sending a regulated tenant's prompt off-site.

Everything in the registry is **declared** — operator-supplied input, not
measurement. A wrong price produces a wrong route. Quality scores for real
deployments are left null rather than invented; the shipped registry leaves
external-provider prices at zero and disabled.

### Declared, observed, absent

Every feature the planner scores on is labelled with where it came from, and the
three labels are kept apart everywhere:

- **declared** — from the registry, i.e. what the operator asserted;
- **observed** — measured by this process from attempts it actually made;
- **absent** — no value, which is *not* zero.

`HealthTracker` supplies the observed half: EWMA latency and error rate, success
and failure counts, in-flight depth, and circuit state. A deployment nobody has
called reports absent rather than healthy.

The rule that follows is the one that keeps the scores honest: **if quality or
latency is missing for any eligible candidate, that feature is dropped for the
whole decision**, and the explanation says so. **Cost is different:** omitted
prices are unknown; `0.0` is known-zero. Unknown prices are excluded from the
cost comparison but do not erase ranking among deployments whose prices are
known, and they are not treated as free. If every feature drops, the planner
falls back to registry order and states that too. See `docs/COST-MODEL.md`.

### Policies

Seven, as specified: `quality_first`, `latency_first`, `cost_first`, `balanced`,
`local_only`, `private_only`, and `custom` weights. `declared` additionally
preserves registry order. Each is a weighting over the four features; the
weights are renormalised across whichever features survived, so a score stays
comparable in [0, 1].

`local_only` and `private_only` are **filters, not preferences**. They exclude on
locality before scoring, so no weighting can trade privacy away for price.

Precedence, strongest first: the tenant's pinned policy, then an explicit request
policy, then the alias's, then what the intent implies, then the default. The
tenant wins because a tenant policy is an administrative decision about that
tenant's own traffic — a caller cannot opt out of it, and
[`tests/security/test_routing_isolation.py`](tests/security/test_routing_isolation.py)
attacks that claim through every lever the API exposes.

A pinned model is never reordered; the caller named it, so ranking would override
the choice they made. Its declared fallbacks trail it.

### The decision object

Every route produces a `RoutePlan`: the resolved policy and where it came from,
the selected deployment, the ranked candidates with each feature's raw value,
source, weight and contribution, every excluded candidate with a typed reason
from a closed vocabulary, the fallback chain, the budget, and a prose
explanation. It also lists which planner inputs were **unavailable** — historical
quality and cache-hit probability have no source in this build, and the plan says
so rather than omitting them.

Expected quality, latency and cost are labelled estimates derived from declared
figures. They are not predictions and the payload says as much.

### Health, breakers and fallback

`BreakerPolicy` trips a deployment on consecutive failures or a sustained error
rate, holds it open for a cooldown, then admits a small number of half-open
probes before closing. An open circuit excludes a candidate at planning time, so
a backend that is down is no longer rediscovered on every request.

`FallbackGraph` is a directed, reason-labelled graph rather than a list. Edges
carry a typed `FallbackReason` — timeout, overloaded, context too large, filtered,
provider error — so a failover is traceable to why it happened. Traversal detects
cycles, and a `FallbackBudget` bounds depth, cumulative cost and elapsed latency,
because an unbounded fallback chain is a way to spend a tenant's money slowly.

**Failover remains forbidden once a stream has produced its first byte** —
[`docs/adr/0003`](docs/adr/0003-no-failover-after-first-streamed-byte.md).

### Synthetic fleet

`router/synthetic.py` builds 30 deployments, one per grade, with attributes
derived deterministically from the grade, plus a provider with programmable
failures and simulated latency. It exists so routing can be tested exhaustively
with no network and no paid API.

**Its numbers are fixtures, not measurements of any real model**, and nothing
derived from them is a benchmark of anything. Test names and the module docstring
both say so, because a fleet of plausible-looking quality scores is exactly the
sort of thing that later gets quoted as if it meant something.

**Not built:** response caching and semantic response caching. Health is measured
per process, so breaker state does not coordinate across replicas.

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

Each provider invocation is also a `UsageEvent` with a stable `event_id`
(`invocation_id`). When `LLM_FABRIC_DATABASE_URL` is set, those events are the
authoritative ledger in PostgreSQL `usage_events` and do not depend on which
worker handled the request. Redis may hold fast day counters; it is not the
source of truth. Semantics, crash windows and streaming rules are in
[`docs/USAGE_METERING.md`](docs/USAGE_METERING.md).

Honesty is enforced structurally. When a backend reports no token counts, the
fabric estimates them with an explicit heuristic in `serving/tokens.py`, and the
record carries `cost_is_estimated: true` while the invocation stores
`token_source` (`PROVIDER_MEASURED` | `LOCAL_TOKENIZER_ESTIMATE` | `DERIVED` |
`UNAVAILABLE`). `/v1/usage` reports both OpenAI-compatible request totals and
invocation ledger totals. An estimated cost is never presented as a measured one.

The OpenAI-compatible `usage` object on the response is the **final visible
model call**. Fallback and internal invocations are in the ledger and in
`invocations` on `/v1/usage`, not folded into that object.

Logs are one JSON object per line, so a pipeline can query them without regex.

When API keys are configured, `/v1/usage` is scoped to the calling client. Failed
requests are recorded as well as successful ones, with `error` set on the record,
when a provider attempt actually began. Authentication failures and input
guardrail blocks produce no invocation events.

Every served request also opens an OpenTelemetry span tree for the stages that
actually ran (`request`, `auth`, `intent` when enabled, `route`, `llm`). Unbuilt
stages — guardrails, context compilation, retrieval, tools, output validation,
evaluations — are listed as unavailable. They do not receive invented timings.

Prometheus metrics live at `/metrics` with a closed set of path labels and
capped model/provider/policy labels. Request ids, tenant ids and user ids are
never Prometheus labels.

Langfuse is reached through `HttpLangfuseAdapter` when host and keys are set,
and is a no-op otherwise. Export failure is logged and never fails a request.
Prompt content is not sent.

The Command Center is at `/command-center`, backed by
`/v1/observability/dashboards/{view}`. Every named view exists. Views whose
backend is not built (`kv_cache`, `batching`, `context`,
`threads`) return `available: false` and empty data. The `evals` view lists
in-process runs. The `drift` view reports signals this process can compute
and lists embedding, compiler context-length and safety-block drift as
unavailable. Agent and safety suites are absent; DeepEval and
lm-evaluation-harness stay empty until those packages are installed and mapped. Ollama and vLLM engine
metrics are consumed only when that adapter is present; neither adapter is
built, so those measurements stay unavailable. Ollama's supported set, when it
exists, will not include KV-cache utilisation — that is not something Ollama
exposes.

**Not built:** the span journal is in-memory and bounded — **not durable**.
OTLP export is configured only when `LLM_FABRIC_OTEL_EXPORTER_OTLP_ENDPOINT`
is set. There is no ClickHouse sink. Usage **is** durable when PostgreSQL is
configured; without a database URL the process still uses `InMemoryMeter`
(tests and local runs). `/v1/usage` and Command Center usage views state
whether they read the postgres ledger or a process-local buffer.

---

## 7. Identity and tenancy — Built

Phase 2. Security properties, threat coverage and known gaps are documented in
[`SECURITY.md`](SECURITY.md); this section covers structure and the reasoning
behind it.

### 7.1 Identity

`identity/` turns a credential into a `Principal`: a frozen record carrying
`tenant_id`, `user_id`, `subject`, `issuer`, `roles`, `scopes` and `project_id`.
Nothing downstream reads a header to learn who is calling.

Four sources implement one `TokenVerifier` protocol:

| Mode | Implementation | Use |
| --- | --- | --- |
| `oidc` | `OIDCTokenVerifier` — JWKS, cached, rate-limited refresh | Production |
| `api_key` | `ApiKeyVerifier` — operator-configured key → tenant | Machine callers |
| `dev` | `DevIdentityProvider` — issues *and* verifies HS256 | Local development |
| `disabled` | Anonymous principal, no scopes or roles | Development and test only |

`LLM_FABRIC_ENVIRONMENT` is `development`, `test`, or `production`. The value
must be set explicitly: unset, empty, and unknown values are a configuration
error. Production refuses to start without a complete identity source
(`validate_startup`). Production also refuses to serve until PostgreSQL and
Redis are reachable (`initialize_runtime` → `probe_distributed_state`). That
function is invoked from `create_app` and from the ASGI lifespan, so the CLI,
`uvicorn --factory`, and the container CMD cannot skip it. The development
issuer, anonymous bypass, and the
multi-worker escape hatch are startup failures in production, not warnings.
OIDC warms JWKS before the process becomes ready.

Every verifier is wrapped in `RevokingVerifier`, which consults a process-local
denylist (`jti` and credential fingerprint). Fleet-wide revocation is not
built; see `docs/AUTH_REVOCATION.md`.

The development issuer exists so local runs exercise the **real** authorization
path. The alternative — developing with authentication off — defers the
discovery of authorization bugs to staging.

Both real paths converge on `RawClaims.to_principal()`, so code downstream
cannot tell an OIDC caller from a development one.

### 7.2 The tenant scope

No repository accepts a bare tenant id. Every read and write takes a
`TenantScope`, constructible only from an authenticated `Principal`.

This is the whole design. Tenant isolation does not usually fail because
somebody wrote a wrong comparison; it fails because the tenant was an optional
argument and one code path forgot it. Here it cannot be forgotten, because there
is no overload that omits it.

### 7.3 Two independent lines of defence

`TenantScopedStore` enforces isolation twice:

1. **Namespacing** — records live in a per-tenant partition, and cache keys are
   fingerprinted with the tenant mixed in.
2. **Ownership re-check** — every record carries its owning `tenant_id`, and
   every read asserts it matches the requesting scope.

The second is redundant while the first is correct, which is exactly why it is
kept: the first is the one a refactor can quietly break. Both were verified by
mutation testing — breaking either alone is caught by the suite, and neither
alone is sufficient (see §7.7).

A cross-tenant read returns **404, not 403**. A 403 confirms the identifier
exists somewhere, which makes the endpoint an enumeration oracle.

### 7.4 Storage

`storage/` holds tenant-scoped repositories for conversations, traces, intent
examples, prompt definitions and evaluation datasets. Record shapes follow the
constitution's field lists so later phases extend behaviour rather than reshape
storage. Prompt versions are immutable once published.

**Backed by memory, not Postgres.** These classes are the interface a SQLAlchemy
implementation will satisfy. Nothing survives a restart.

### 7.5 Caches

`TenantScopedCache` is seven distinct caches — exact response, semantic
response, intent, embedding, retrieval, prompt, context artifact — each with its
own TTL, bound, invalidation and counters. They share only the isolation
guarantee, inherited from `TenantScopedStore`.

Keys are assembled from named discriminators rather than concatenated strings,
which is where "we forgot to include the policy version" bugs come from.

The provider prefix/KV cache named in the constitution is deliberately absent:
it lives inside vLLM or Ollama, the fabric does not control its keys, and
modelling it here would imply a guarantee this process cannot make.

### 7.6 Quotas and admission

`QuotaLedger` enforces per-tenant **and** per-user ceilings on requests per
minute, requests per day, tokens per day and spend per day. Both levels matter:
a tenant-wide limit does not protect one user from another inside that tenant,
and a per-user limit does not bound the tenant in aggregate.

Counters are fixed-window and bounded by tracked subjects rather than by
windows, because a tenant id is attacker-influenced and an unbounded ledger is a
memory-exhaustion vector.

**The ledger is in-process**, so limits apply per replica, not per fleet.

### 7.7 The middleware

`AuthenticationMiddleware` is raw ASGI, not Starlette's `BaseHTTPMiddleware`,
because the gateway streams SSE and `BaseHTTPMiddleware` wraps the response body
in an extra task — complicating cancellation and cleanup on exactly the path
where getting cleanup wrong leaks provider connections.

Order is fixed: identify the caller, resolve the tenant, then admit or shed.
Authentication runs *before* routing, so an unauthenticated caller cannot map
which endpoints exist.

### 7.8 Verification

298 tests pass, of which **106 are adversarial** and marked `isolation`. They run
as a dedicated CI job so a cross-tenant failure is legible on the checks list
rather than buried in a general log. `pytest -m isolation` exits non-zero when it
collects nothing, so deleting the marker fails CI rather than disabling the gate.

Every attack is paired with a control assertion proving the victim can still
reach their own data — without it, a fixture that stored nothing would make the
suite pass while proving nothing.

The two defences were confirmed independent by mutation: removing the tenant
filter from the store failed 21 tests while the cache tests still passed, and
removing the tenant from the cache key failed 7 tests while the store tests
still passed.

---

## 8. IntentOS — Built

Deciding what a prompt is *for*, so the router has something to route on. Built
against the constitution.

Every claim in this section is structural — what exists and how it is wired.
**Nothing here asserts that the classifier is accurate.** The measured numbers,
and the reasons not to over-read them, are in
[`INTENTOS_EVALUATION.md`](INTENTOS_EVALUATION.md). Architecture detail:
[`docs/INTENTOS.md`](docs/INTENTOS.md). A historical cascade run remains in
[`docs/EVALUATIONS.md`](docs/EVALUATIONS.md) §2.

### 8.1 The taxonomy

`IntentTaxonomy` is immutable once constructed. `evolve()` returns a new version
rather than editing the old one, and `TaxonomyRegistry` refuses to re-register a
version. This is not fastidiousness: every stored classification records the
taxonomy version that produced it, and that record is worthless if an intent's
meaning can change underneath it. Retiring marks a node rather than deleting it,
so a historical classification that named it can still be explained.

Hierarchy is carried in the id — `coding`, `coding.debug`,
`coding.debug.stacktrace` — so `domain`, `task` and `subtask` are derived rather
than separately maintained fields that can drift apart. Construction validates
parentage, rejects cycles, and bounds depth.

Each node carries what the constitution names, including `examples`,
`counterexamples` and `hard_negatives`, plus an `IntentProfile` giving the
routing-relevant shape — complexity, reasoning level, risk, latency tolerance —
so the cheap classifier layers can emit a complete classification without
inferring all of it per request.

The shipped taxonomy is **bootstrap material, not a settled ontology**: 15
domains and 3 sub-tasks, written as a starting point to be replaced by intents
derived from real traffic.

### 8.2 The cascade

```
L0 exact cache → L1 semantic cache → L2 rules → L3 embedding
→ L4 structured model → L5 escalation → abstain
```

Each layer returns only if its confidence clears that layer's threshold.
**Thresholds fall as the cascade deepens** (0.70 → 0.62 → 0.55 → 0.40), which
looks backwards until you see what a threshold is for: it is the price of
skipping everything below it. A regex must be near-certain to earn the right to
stop the cascade, because the layers it skips are better than it is. The
escalation layer has nothing better behind it, so demanding the same certainty
would only convert answers into abstentions.

L2 and L3 may agree at a lower floor (0.48) only when they name the **same**
intent and the runner-up is weak. That guard is what keeps multi-intent prompts
from becoming a confident single label.

Abstention is the floor. When nothing clears its bar the engine returns
`unknown` rather than the best of several bad guesses. The rejected best guess
is retained on the result, because "nearly certain but under the bar" and "no
idea" are different signals even though both abstain.

A classifier failure never becomes a request failure. A provider outage at L4
degrades to no opinion; the caller asked for a completion, not a classification.
Intent capability extras that would empty the candidate set are dropped; the
planner still honours hard request, alias, and tenant requirements.

### 8.3 The layers

| Layer | Implementation | Cost | Notes |
| --- | --- | --- | --- |
| L2 rules | Weighted regex, negative rules for hard negatives | free | Bounded scan; offline |
| L3 embedding | Nearest centroid over intent examples | free\* | Adapter over `EmbeddingProvider` |
| L4 structured | Model asked for validated JSON | metered | Hallucinated ids discarded, no repair |
| L5 escalation | Same class, stronger model, L3 shortlist | metered | Optional |

\* Free with the default `HashingEmbedder`, which is a **lexical** hashing
vectoriser, not a semantic model. It exists so the suite runs deterministically
offline. Where it is the default, that is a decision about determinism, not a
claim about quality; a real deployment plugs a trained model into the same
interface.

The structured classifier enforces three things strictly: the model may only
choose an intent that exists, it may abstain by returning `unknown`, and
malformed output yields no opinion rather than a repair attempt. Coaxing a
second answer out of a model that already failed the schema spends money to
raise the chance of a confidently wrong label.

### 8.4 The two intent caches

Kept separate because they carry different risk. An exact hit is a fact about an
identical prompt; a semantic hit is a *guess* that two different prompts want
the same thing, and a wrong guess silently misroutes a caller.

Both are keyed on every discriminator the constitution names — `tenant_id`,
`taxonomy_version`, `classifier_version`, `policy_version`, `language`,
`conversation_state_signature` — and `classifier_version` digests *every* layer's
version, so changing any layer invalidates the cache.

The semantic index is **partitioned by an exact-match discriminator signature**,
so similarity is only compared within one taxonomy version, classifier version,
policy version, language and conversation state. Comparing across them would let
a stale taxonomy answer a current question. A hit must clear both a similarity
threshold and a confidence threshold; either alone is insufficient.

Abstentions are never cached. An abstention is the outcome most likely to be
fixed by a taxonomy or threshold change, and caching it would keep serving the
old answer after the fix landed.

False hits cannot be detected at read time — that needs ground truth — so the
API is a reporting hook, and `false_hit_rate` divides by *reviewed* hits and
returns `None` when nothing has been reviewed. Dividing by total hits would
understate the rate by assuming every unreviewed hit was correct, which is
exactly the assumption the metric exists to test.

### 8.5 Tenancy

Intent data is tenant-scoped like everything else, through the same
`TenantScopedCache` and `TenantScopedStore` from §7. `tests/security/
test_intent_isolation.py` attacks it: replaying a victim's exact prompt, probing
with near neighbours against a semantic cache with **both thresholds set to
zero**, attempting cross-tenant invalidation and overwrite, and checking the
embedding cache is not shared. Each attack is paired with a control proving the
victim can still reach their own entry.

### 8.6 Tracing and metrics

Every classification returns an `IntentDecision` carrying a `LayerAttempt` per
layer: which intent it favoured, its confidence, the threshold it faced, whether
it was accepted, its latency and its cost. That is the audit trail for why a
prompt was classified as it was.

`trace_attributes()` never carries the prompt. Metric labels are layers, intents
and fixed histogram buckets — all bounded, per the constitution's prohibition on
unbounded-cardinality labels.

### 8.7 The learning loop, still incomplete

Low-confidence, ambiguous, abstained and layer-disagreement results are offered
to a bounded `candidate_sink`. Ingest redacts secret-shaped spans, hashes for
dedup, and stores a tenant-scoped **draft**. Shadow classification can sample
traffic without returning the candidate to the user.

**Nothing trains on them and nothing is promoted.** `promotion_blocked_reason`
refuses promotion without human review and passing eval gates. Canary, a
statistical gate, and a rollback artifact are not built. There is no live
self-training.

### 8.8 The benchmark

`llm-fabric-bench` scores a JSONL dataset and reports accuracy, macro/micro F1,
per-intent precision and recall, confusion matrix, top-k accuracy, expected
calibration error, Brier score, high-confidence routing precision at explicit
thresholds, abstention precision/recall/accuracy, unknown-intent recall,
measured semantic-cache false-hit rate, latency percentiles and classification
cost.

Two modes, kept separate because they measure different things: `classifier`
runs one cold pass; `cache` warms on prompts then scores their paraphrases.
Gates (`--min-accuracy` and friends) set exit codes, and **a gate whose metric
could not be measured fails** — "not measured" is never allowed to pass as "met
the bar".

Results: [`INTENTOS_EVALUATION.md`](INTENTOS_EVALUATION.md). The dataset is
self-authored and small, which that document states before it states any number.

---

## 9. Reconciling with the constitution

The constitution is at [`docs/constitution.md`](docs/constitution.md). This
section is the honest diff against it as of Phase 11.

### 9.1 Confirmed by the constitution

These Phase 1 decisions were guesses that the specification subsequently
ratified. They stand.

| Decision | Constitutional basis |
| --- | --- |
| Python 3.12, FastAPI, Pydantic v2, asyncio, httpx, pytest, ruff ([ADR 0001](docs/adr/0001-language-and-runtime.md)) | *Code quality* names exactly this stack. |
| OpenAI-compatible dialect at `/v1` ([ADR 0002](docs/adr/0002-openai-compatible-contract.md)) | *SDK* and *API versioning* require OpenAI compatibility and `/v1`. |
| No failover after the first streamed byte ([ADR 0003](docs/adr/0003-no-failover-after-first-streamed-byte.md)) | *Fallback graph* requires traced, bounded, loop-free fallback. |
| One `Provider` interface, adapters isolated | *Provider architecture*: register providers without modifying routing logic. |
| Estimated token counts labelled as estimates | *Role*: represent unmeasurable capabilities as unsupported, never synthesize. |
| OAuth2/OIDC identity from validated claims (Phase 2) | *Authentication*: identity comes from validated authentication claims. |
| `TenantScope` on every storage and cache operation (Phase 2) | *Multi-tenancy*: isolation is mandatory across every storage layer. |
| Adversarial cross-tenant suite gating CI (Phase 2) | *Multi-tenancy*: cross-tenant penetration tests are required. |
| Seven independently-configured caches (Phase 2) | *Caching*: distinct types with independent TTL, namespace, invalidation, metrics. |
| Per-tenant and per-user quotas (Phase 2) | *Multi-tenancy* and *Reliability*: admission control and tenant limits. |
| Immutable versioned taxonomy (Phase 3) | *Intent storage*: do not mutate historical taxonomy versions. |
| L0–L5 cascade with per-layer thresholds (Phase 3) | *IntentOS*: a lower layer may return only if its confidence threshold is satisfied. |
| `unknown` as a first-class result (Phase 3) | *IntentOS*: unknown intent is a valid and important result. |
| Separate exact and semantic intent caches (Phase 3) | *Intent cache*: separate TTL policies, similarity and confidence thresholds, false-hit statistics. |
| Candidate collection without auto-promotion (Phase 3) | *Intent learning loop*: do **not** automatically train and promote from production inputs. |
| Grades as logical bands, not model names (Phase 4) | *Model grading*: grades are logical classes; deployments map into them. |
| Auditable `RoutePlan` on every route (Phase 4) | *Routing engine*: no hidden routing decision; every decision explainable from stored features. |
| Reason-labelled, bounded, loop-free fallback graph (Phase 4) | *Fallback graph*: typed reasons, depth limits, cycle prevention, budget. |
| Circuit breakers over EWMA latency and error rate (Phase 4) | *Reliability*: circuit breakers and EWMA tracking per deployment. |
| Locality filters that scoring cannot override (Phase 4) | *Privacy*: local-only and private-only routing are constraints, not preferences. |
| Unmeasured features dropped rather than defaulted (Phase 4) | *Role*: represent unmeasurable capabilities as unsupported, never synthesize. |

### 9.2 Built, but not to specification

These exist and work, but are materially narrower than what is mandated.

| Area | Current | Constitution requires |
| --- | --- | --- |
| **Model registry** | Typed `ModelSpec` with grades, capabilities, quality, performance and placement in YAML, plus an evidence-bound `registered → probed → evaluated → shadow → approved` overlay. Production pin/auto routing requires artifact-bound approval; revision/digest mismatch fail-closes. | The same registry in Postgres, with live health, latency and queue signals fed from the fleet rather than one process |
| **Routing features** | Four: quality, latency, cost, health | Also historical quality, cache-hit probability, and queue depth sourced fleet-wide. The plan reports these as unavailable rather than defaulting them. |
| **Route health** | EWMA and breaker state measured by each process | Fleet-wide health, so a breaker tripped on one replica is known to the others |
| **Routing evals** | Labelled planner match against the mock registry (`route_match`, `policy_match`). Declared regret stays unavailable without quality/latency numbers. | A routing eval proving decision *quality*, not just that the planner hit a fixture label |
| **Tenant storage** | Interfaces enforced, backed by bounded in-memory stores | The same isolation, durably, across Postgres, Redis, ClickHouse, object and vector storage |
| **Quotas** | Per-tenant and per-user, in-process fixed windows | Fleet-wide enforcement, so limits do not multiply by replica count |
| **Adapters** | `openai`, `anthropic`, `mock` | `LiteLLM`, `Ollama`, OpenAI-compatible, vLLM-compatible. A direct Anthropic adapter is not in the mandated set. |
| **Reliability** | Bounded attempts, retryable-error classification, admission control, circuit breakers, EWMA latency/error tracking | Also backoff with jitter, load shedding, deadlines, cancellation propagation, graceful shutdown, idempotency |
| **Observability** | OpenTelemetry spans for built stages, in-process journal, bounded-cardinality Prometheus scrape, optional Langfuse adapter, Command Center UI | The same plus ClickHouse; spans for every constitution stage once those stages exist; fleet-wide aggregation |
| **Prompt registry** | Tenant-scoped versioned storage, published versions immutable | Promotion workflow, evaluation gating, model-family adapters |
| **Intent taxonomy** | 15 bootstrap domains, 3 sub-tasks, in-process | Taxonomy in Postgres, derived from real traffic, with promotion workflow |
| **Semantic intent cache** | Brute-force cosine over a bounded in-memory index | A vector store. The index is linear in entries per signature. |
| **Intent learning loop** | Redact, dedup, tenant-scoped draft store, sampled shadow. No auto-promotion. | Sanitisation → dedup → dataset → candidate → offline eval → shadow → canary → statistical gate → promotion, with a rollback artifact |
| **Intent confidence** | Heuristic scores, monotone and bounded. Temperature scaling is identity until a larger val set is fitted offline. Measured ECE on the v1 frozen set is 0.177. | *Calibrated* thresholds against datasets that are not self-authored |
| **Intent evals** | Every mandated metric computed by `llm-fabric-bench` | The same metrics against datasets that are not self-authored, plus maintained hard-negative sets |
| **Route consumption** | The planner reads an `IntentClassification` and can infer a policy from it, but gateway classification is off by default | Intent routed on by default, once the added latency has been measured |
| **Drift / self-healing** | In-process usage windows, held breakers, traffic overlay, remembered rollbacks, incidents and unevaluated learning jobs | Fleet-wide baselines, embedding drift, compiler context-length drift, safety-block drift, the rest of the learning-loop pipeline |

### 9.3 Not built at all

Entire mandated subsystems with no code in this repository:

control plane / data plane split · context compiler · guardrail engine (five
stages) · structured-output validation and bounded repair · economics subsystem
(the Command Center economics view is registry prices × tokens, not this) ·
agent and safety eval suites · DeepEval / lm-eval mapped runners (adapters exist,
packages are not installed) · chaos suite · ClickHouse,
object storage · native vLLM Python engine and engine `/metrics` scrape · Terraform.

IntentOS moved out of this list in Phase 3; §8 describes what was built and §9.2
records where it falls short of the specification. The load harness and
`BENCHMARKS.md` moved out when the throughput target was measured: the harness
is `llm-fabric-load` rather than k6, which the constitution permits, and it
covers seven of the nine workload classes it names — agent and real-generation
workloads are absent because the subsystems they would exercise are not built.

### 9.4 Standing constitutional violations

Facts about the current tree, recorded rather than quietly fixed:

1. **No control plane / data plane separation.** Everything is one process.
2. **Tenant isolation is not proven against a database.** The boundary is
   enforced and adversarially tested, but the backing store is memory. Postgres
   row-level security — the second line of defence a database can provide — does
   not exist, and nothing survives a restart.
3. **Quota enforcement is per replica.** The ledger is in-process, so limits
   multiply by the number of replicas.
4. **OpenTelemetry has no durable sink.** Spans exist for the stages that run
   and can be exported over OTLP when configured. There is no ClickHouse store,
   and unbuilt stages are not given empty timings.
5. **No `docker compose` stack.** `make dev` exists; the local Postgres, Redis
   and ClickHouse services it should depend on do not.
6. **`mypy` is strict only in `identity/`, `tenancy/`, `intent/`, `eval/`,
   `heal/` and `errors`.** The rest of the package is checked at default
   strictness.
7. **The server package is still `llm_fabric`.** The constitution's product
   import is `myvista`; that package now exists as the SDK. The gateway itself
   was not renamed.
8. **Gateway benchmarks cover only the mock provider.** The constitution's 500
   RPS target is met and exceeded — a single worker sustains 1,000 req/s, and
   saturates at 2,377 req/s on `chat-short` — but every workload runs against a
   provider that performs no inference, on a laptop, for at most 20 seconds. No
   real provider has ever been load tested. See
   [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md), whose §1 and §8 state the limits
   before and after the numbers.
9. **Intent classification is off on the serving path.** The planner reads a
   classification when given one, but the gateway does not produce one unless
   `LLM_FABRIC_INTENT_CLASSIFICATION_ENABLED` is set. The offline cascade is
   0.58 ms p50 in-process; the HTTP cost of running it on every request has
   not been measured.
10. **Intent confidence is uncalibrated.** The cascade gates on confidence
    scores whose measured expected calibration error is 0.175. The thresholds
    were deliberately *not* tuned against the evaluation dataset, so they remain
    documented judgements rather than fitted values.
11. **Circuit-breaker and health state is per replica.** `HealthTracker` observes
    only the attempts its own process made, so a deployment tripped out on one
    replica stays in rotation on the others.
12. **No routing-quality evaluation has been run.** A labelled planner-match
    suite scores `route_match` and `policy_match` against the mock registry.
    That is a CI tripwire. The constitution's regret, escalation and
    under/overpowered rates stay unavailable unless both sides already carry
    declared numbers; nothing here shows the routes are good ones.
13. **The synthetic fleet's figures are fixtures.** Its quality scores, prices
    and latencies were chosen to make the tests discriminating, and correspond to
    no real deployment. Nothing measured against it is a benchmark.
14. **Routing policy weights are judgements, not fitted values.** The seven
    policies' weightings were chosen to express the intent of each policy; they
    were not tuned against any outcome data, because none has been collected.
15. **Throughput and correctness cannot both be scaled up here without
    distributed state.** Quotas, breakers and caches need Redis. Authoritative
    usage needs PostgreSQL. Multi-worker without both is refused unless
    explicitly acknowledged (forbidden in production).

### 9.5 Cost of the corrections

| Correction | Cost |
| --- | --- |
| Registry into Postgres | Contained. `ModelRegistry` is the seam the planner reads through, and `ModelSpec` is already the typed row a table would hold. |
| Fleet-wide health | Contained. `HealthTracker` is the interface the planner and engine use; its counters move to Redis and the breaker logic is unchanged. |
| Routing-quality eval suite | Additive. Labelled planner match exists. `RoutePlan` is already a complete, serialisable record of a decision, which is the input a quality suite needs. |
| Postgres / Redis / ClickHouse | Contained by design. `TenantScopedStore` is the interface; the repositories and the adversarial suite should survive the swap unchanged, which is the main thing Phase 2 bought. |
| Fleet-wide quotas | Replace the ledger's counters with Redis. The `QuotaLedger` surface holds. |
| OpenTelemetry | Contained. `TraceContext` already carries W3C-compatible ids for instrumentation to adopt. |
| Adapter set | Additive. New `Provider` subclasses; the Anthropic adapter's future is a decision, not a defect. |
| Server package rename to `myvista` | Separate decision. The SDK already imports `myvista`; the gateway remains `llm_fabric`. |
| Vector store behind the semantic intent cache | Contained. `SemanticIntentCache.lookup`/`admit` is the surface an index must satisfy; the isolation suite should survive the swap. |
| Real embedding model | Configuration. `EmbeddingProvider` is the seam, and `classifier_version` already digests the model id so the swap invalidates cached classifications automatically. |
| Routing on intent by default | Configuration, once the cascade's added HTTP latency on the serving path has been measured. In-process cost is in `docs/BENCHMARKS.md` §10. Both halves of the wire exist. |

---

## 10. Engineering rules in force

- **The constitution governs.** [`docs/constitution.md`](docs/constitution.md) is
  the authoritative specification. Where this document disagrees with it, this
  document is wrong and §9 says so explicitly.
- **Authorship.** All production code is authored under Anubhav Bhatia. No AI or
  Cursor authorship, co-author trailers, or generated-by markers. Ownership is
  recorded in [`.github/CODEOWNERS`](.github/CODEOWNERS).
- **No fabrication.** No benchmark, metric, or capability claim appears unless it
  was actually produced by a reproducible run. Three quantitative claims are made
  anywhere in this repository: the test count, reproducible with `pytest`; the
  intent-classifier measurements in
  [`docs/EVALUATIONS.md`](docs/EVALUATIONS.md), reproducible with
  `make bench-intent`; and the gateway throughput and latency figures in
  [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md), reproducible with
  `make bench-load`. Each of those documents states its limits before its
  numbers, and every load figure is against a provider that performs no
  inference.
- **No superiority claims.** Nothing in this repository states or implies that
  the intent classifier, the router, or the gateway is better than any
  alternative, because no comparison has been run. The synthetic fleet's
  attributes are test fixtures, and no number derived from them is a benchmark.
- **A performance change is not claimed until it is measured as one.** The one
  optimisation made for throughput did not improve throughput; that is recorded
  in `docs/BENCHMARKS.md` §6 rather than quietly dropped.
- **Unbuilt means labelled unbuilt**, as in §3 through §8 above. Security
  properties and their gaps are enumerated in [`SECURITY.md`](SECURITY.md).
- **Scope.** Work stays in this repository. No other project is modified.
