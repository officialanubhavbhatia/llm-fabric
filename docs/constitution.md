# MyVista LLM Fabric — Engineering Constitution

**Status:** Authoritative. Received from the project owner on 2026-08-23 and
committed verbatim.

This document is the permanent architecture and engineering specification. It
takes precedence over inference, convention, `AGENTS.md`, and `ARCHITECTURE.md`.
Where `ARCHITECTURE.md` disagrees with this document, `ARCHITECTURE.md` is wrong
and must be corrected.

Amendments are made by editing this file with an accompanying ADR, never by
implementing something different and documenting it after the fact.

---

## Authorship

Project owner and repository author:

**Anubhav Bhatia**

Use Anubhav Bhatia as the project author in package metadata, documentation
attribution, copyright headers where the project uses them, and CODEOWNERS
configuration.

Do not invent cryptographically signed commits or claim a signature that does not
exist.

Do not add AI-generated authors, `Co-authored-by` metadata, fictitious people,
Cursor attribution, or assistant attribution.

---

## Role

You are the principal distributed-systems architect, inference engineer, ML
platform engineer, AI evaluation engineer, SRE, security engineer and SDK
architect for MyVista.

You are building a production-grade platform named **MyVista LLM Fabric**.

The system must provide a developer-friendly SDK and OpenAI-compatible API above
LiteLLM, Ollama and high-throughput inference servers.

This is not a demo.

Do not implement fake functionality, placeholder metrics, fake performance claims,
hard-coded benchmark results, mock production paths or TODO-driven architecture.

When a capability cannot actually be measured or implemented with a backend,
explicitly represent it as unsupported rather than synthesizing a value.

---

## Primary product goal

A developer should eventually be able to write approximately:

```python
from myvista import MyVista

client = MyVista()

response = client.responses.create(
    input="Debug this Python program",
    quality="high",
    latency_slo_ms=2000,
)
```

The developer should not need to understand:

- which LLM provider is used
- which model is loaded
- routing rules
- retries
- fallbacks
- KV-cache configuration
- batching
- context budgeting
- prompt adaptation
- token accounting
- observability
- intent classification
- guardrails
- eval infrastructure
- multi-tenancy
- cost optimization

MyVista handles these through policy and configuration.

---

## Non-negotiable architecture

Separate the system into a **control plane** and a **data plane**.

The synchronous request path must remain extremely small.

Request path:

```
API
-> Authentication
-> Tenant policy
-> Input guardrails
-> Intent engine
-> Context compiler
-> Route planner
-> Inference adapter
-> Output validation
-> Response
```

Telemetry, dataset generation, most evaluations, drift processing and
learning-loop processing must operate asynchronously unless an individual policy
explicitly requires a synchronous quality or safety check.

---

## Provider architecture

Implement provider adapters.

Initial adapters:

- LiteLLM
- Ollama
- OpenAI-compatible inference
- vLLM-compatible inference

Design the interface so additional providers can be registered without modifying
routing logic.

Never spread provider-specific conditionals throughout business logic.

Create a strongly typed capability registry.

---

## Model grading

Implement configurable model grades: `Grade00` ... `Grade29`.

Grades are logical capability classes, not fixed model names. A deployment can
move between grades as benchmark data changes.

Store attributes such as:

**Identity**

- `model_id`
- `deployment_id`
- `provider`
- `grade`

**Quality scores**

- `reasoning_score`
- `coding_score`
- `agent_score`
- `math_score`
- `rag_score`
- `tool_use_score`
- `structured_output_score`
- `safety_score`

**Context**

- `max_context_tokens`
- `recommended_context_tokens`

**Capabilities**

- `supports_tools`
- `supports_json_schema`
- `supports_vision`
- `supports_embeddings`
- `supports_prefix_cache`
- `supports_speculative_decode`

**Performance**

- `p50_ttft_ms`
- `p95_ttft_ms`
- `p99_ttft_ms`
- `p50_tpot_ms`
- `prefill_tokens_per_second`
- `decode_tokens_per_second`

**Cost**

- `input_cost`
- `output_cost`
- `estimated_compute_cost`

**Operational**

- `health_score`
- `error_rate`
- `queue_depth`

**Placement**

- `region`
- `hardware`
- `locality`

Routing must use this registry rather than model-name assumptions.

---

## IntentOS

Intent classification is a primary subsystem.

Create a hierarchical intent schema.

Each classification must support:

- `domain`
- `task`
- `subtask`
- `complexity`
- `reasoning_level`
- `required_capabilities`
- `modality`
- `agent_required`
- `tools_required`
- `structured_output`
- `context_class`
- `risk_class`
- `latency_class`
- `quality_class`
- `cost_class`
- `confidence`
- `alternatives`
- `abstain`
- `classifier_version`
- `taxonomy_version`

Implement the classification cascade:

- **L0** exact cache
- **L1** semantic cache
- **L2** deterministic/rules classifier
- **L3** embedding classifier
- **L4** small-model structured classifier
- **L5** escalation classifier
- **UNKNOWN / ABSTAIN**

A lower layer may return only if its calibrated confidence threshold is satisfied.

Do not force every prompt through an LLM.

Unknown intent is a valid and important result.

### Intent storage

Create a versioned intent taxonomy.

The data model must support:

- `intent_id`
- `parent_intent_id`
- `name`
- `description`
- `examples`
- `counterexamples`
- `hard_negatives`
- `required_capabilities`
- `default_route_policy`
- `created_at`
- `updated_at`
- `taxonomy_version`
- `status`

Do not mutate historical taxonomy versions.

Every production classification records the taxonomy version used.

### Intent cache

Support exact and semantic caching.

Cache keys must include enough state to prevent unsafe reuse:

- `tenant_id`
- `taxonomy_version`
- `classifier_version`
- `policy_version`
- `language`
- `conversation_state_signature`

Exact cache entries and semantic cache entries require separate TTL policies.

Semantic hits must have configurable similarity and confidence thresholds.

Collect false-hit statistics.

### Intent learning loop

The production system may collect candidate hard examples from:

- low-confidence predictions
- classifier disagreement
- route failures
- fallback activity
- user feedback
- evaluation failures
- large route regret
- new semantic clusters

**Do NOT automatically train and promote a new classifier directly from
production inputs.**

Workflow:

```
candidate examples
-> sanitization
-> deduplication
-> dataset
-> train/candidate
-> offline evaluation
-> shadow traffic
-> canary
-> statistical gate
-> promotion
```

Every promotion must have a rollback artifact.

---

## Routing engine

Create a route planner that takes:

- intent
- tenant policy
- quality requirement
- latency SLO
- budget
- context requirement
- model capabilities
- provider health
- queue pressure
- historical quality
- historical latency
- historical error rate
- cache probability

and produces:

- selected deployment
- fallback graph
- routing explanation
- routing score
- expected quality
- expected latency
- expected cost

No hidden routing decision. Each decision must be explainable from stored
features.

Support routing policies:

- quality-first
- latency-first
- cost-first
- balanced
- local-only
- private-only
- custom

### Fallback graph

Do not implement one flat fallback list.

Implement a directed fallback graph supporting reasons:

- `timeout`
- `overloaded`
- `rate_limited`
- `provider_down`
- `context_too_large`
- `model_unavailable`
- `safety_requirement`
- `structured_output_failure`
- `quality_failure`

Prevent fallback loops.

Store fallback depth.

Define maximum fallback cost and maximum fallback latency.

Trace every fallback.

---

## Reliability

Implement:

- circuit breakers
- health checks
- readiness checks
- EWMA latency tracking
- EWMA error tracking
- bounded retries
- exponential backoff with jitter
- concurrency limits
- admission control
- load shedding
- request deadlines
- cancellation propagation
- graceful shutdown
- idempotency where appropriate

Never retry a request indefinitely.

### Self healing

The platform may automatically:

- remove unhealthy deployments
- reroute traffic
- activate fallbacks
- restart disposable workers
- invalidate unhealthy cache entries
- change admission limits
- reduce context ceilings during resource exhaustion
- change traffic allocations
- rollback a bad configuration
- rollback a bad model
- rollback a bad prompt
- rollback a bad classifier
- autoscale when deployment infrastructure supports it

It must **never** automatically disable security, authentication, authorization,
tenant isolation or mandatory guardrails to restore availability.

---

## Context engine

Build a context compiler.

Represent context blocks as typed objects containing:

- `content`
- `type`
- `source`
- `provenance`
- `tenant_id`
- `trust_level`
- `priority`
- `freshness`
- `token_count`
- `cacheability`
- `visibility`

Supported context types:

- `system`
- `developer_policy`
- `tenant_policy`
- `conversation`
- `long_term_memory`
- `retrieval`
- `tool_result`
- `agent_state`
- `user`
- `output_contract`

Implement:

- context token accounting
- reserved output budget
- deduplication
- semantic redundancy removal
- relevance ranking
- trust-aware placement
- compression
- summarization hooks
- truncation
- stable-prefix ordering
- context provenance

**Never include context belonging to another tenant.**

---

## Prompt engineering

Create a versioned Prompt Registry.

Each prompt must have:

- `prompt_id`
- `version`
- `owner`
- `purpose`
- `supported_intents`
- `supported_model_families`
- `template`
- `variables`
- `output_contract`
- `evaluation_suite`
- `created_at`
- `status`

Never modify a published prompt version.

Prompt promotion uses:

```
draft -> candidate -> evaluated -> canary -> production -> retired
```

Model-specific prompt adapters must be isolated.

---

## Structured outputs

Prefer explicit schemas for classification, routing and machine-consumed
operations.

Use Pydantic in Python.

Validate outputs. Do not trust LLM-generated JSON merely because it parses.

Validate:

- schema
- types
- enum constraints
- lengths
- policy constraints
- semantic invariants

Repair attempts must be bounded and traced.

---

## Guardrails

Design a pluggable guardrail engine.

Stages:

- `INPUT`
- `RETRIEVAL`
- `CONTEXT`
- `EXECUTION`
- `OUTPUT`

Support adapters for external guardrail systems such as NeMo Guardrails while
also supporting native deterministic rails.

**Input controls** must support: request limits, prompt injection detection,
jailbreak detection, PII policy, secret detection, tenant policy, modality
restrictions.

**Retrieval controls** must support: provenance validation, tenant checks,
document trust, malicious instruction detection, PII policy.

**Execution controls** must support: tool allowlists, permission checks,
JSON-schema argument validation, network policies, filesystem policies, timeout,
resource limits, maximum agent steps.

**Output controls** must support: schema checking, safety classification,
sensitive data checks, citation/grounding requirements, policy enforcement.

**No model may grant itself authorization.**

---

## Authentication

Support OAuth2/OIDC.

Identity comes from validated authentication claims.

Required internal identity:

- `tenant_id`
- `user_id`
- `subject`
- `roles`
- `scopes`
- `project_id`

Never trust a client-supplied tenant ID unless the authenticated policy
explicitly permits delegation.

Support:

- `Authorization: Bearer ...`
- `traceparent`
- `Idempotency-Key`
- `X-Request-ID`

---

## Multi-tenancy

Tenant isolation is mandatory.

Apply tenant boundaries to: Postgres, Redis, caches, semantic caches, vector
storage, conversations, prompts, traces, evaluations, datasets, intent examples,
API keys, usage, billing.

Add automated cross-tenant penetration tests.

A test must intentionally attempt to retrieve another tenant's cached response,
embeddings, intent information, traces, prompts and conversations.

**Every such attempt must fail.**

---

## Privacy

Do not log secrets.

Raw prompts and outputs must have configurable retention and redaction policies.

Telemetry should use hashes or stable identifiers where raw values are
unnecessary.

Authorization credentials must never appear in logs.

Do not use production conversations as training data unless an explicit tenant
policy allows it.

---

## Ollama

Ollama is the preferred initial local inference engine.

Implement configuration support for:

- context length
- parallel requests
- model keep alive
- maximum loaded models
- queue capacity
- Flash Attention
- KV-cache type

Capture metrics that Ollama actually exposes.

**Do not fake unavailable KV-cache statistics.**

---

## vLLM

Support a high-throughput production inference adapter.

Expose available metrics including: KV-cache usage, prefix-cache queries,
prefix-cache hits, cached prompt tokens, running requests, waiting requests,
prefill tokens, generated tokens, preemptions.

Support feature flags for: prefix caching, continuous batching, chunked prefill,
speculative decoding.

**A performance optimization must be benchmarked before becoming a production
default.**

---

## Caching

Keep these caches logically distinct:

- exact response cache
- semantic response cache
- intent cache
- embedding cache
- retrieval cache
- prompt cache
- context artifact cache
- provider prefix/KV cache

Never call all of these simply "the cache."

Each type needs independent: TTL, namespace, invalidation, metrics, safety rules.

---

## Observability

Use OpenTelemetry conventions.

Every request receives a trace ID.

Trace: HTTP request, authentication, guardrails, intent classification, intent
cache, context compilation, retrieval, route planning, provider call, fallbacks,
tools, output validation, evaluations.

Record model calls as child spans.

**Do not put unbounded-cardinality values into Prometheus labels.**

### Request metrics

Capture:

- RPS, active requests, queue depth, queue latency
- TTFT, TPOT, total latency
- prompt tokens, completion tokens, cached tokens, reasoning tokens when provided
- prefill TPS, decode TPS
- context tokens before optimization, context tokens after optimization, tokens
  compressed, tokens dropped
- fallback count, retry count
- intent confidence, intent classifier layer, intent cache hit
- request cost, compute estimate, cache savings

### Inference metrics

Capture when available: model load state, GPU memory, GPU utilization, KV-cache
utilization, prefix-cache hit rate, batch size, batch utilization, active
sequences, waiting sequences, preemption, prefill TPS, decode TPS, speculative
decoding acceptance.

**Unsupported metrics must be represented as unavailable.**

---

## Economics

Create an economics subsystem.

Measure:

- cost/request, cost/tenant, cost/user, cost/intent, cost/model, cost/provider
- cost/successful request, cost/evaluation point, cost/1K input tokens,
  cost/1K output tokens
- GPU-hours, estimated electricity/compute cost where configured
- cache savings, routing savings, context compression savings

**Never pretend self-hosted inference is free.**

---

## Observability UI

Create a **MyVista Command Center**.

Views:

- **Overview** — request volume, reliability, latency, TPS, token volume, cost,
  quality, safety
- **Models** — loaded models, health, grades, latency, TPS, error rates, KV
  cache, batching, costs, quality
- **Intents** — distribution, confidence, cache hits, classifier layers,
  misclassifications, drift, newly discovered clusters
- **Users** — tenant, user, requests, tokens, cost, quality, errors
- **Traces** — full tree:

```
request
  auth
  input_guardrails
  intent
  context
  route
  llm
  tool
  llm
  output_guardrails
  eval
```

- **Threads** — group traces into complete conversations
- **Economics** — cost and efficiency
- **Evaluations** — current and historical evaluation results
- **Drift** — changes in intent, quality, model behavior, cost, latency and
  routing

---

## Evaluation engine

Evals are first-class objects.

Create: `EvalDataset`, `EvalExample`, `EvalSuite`, `EvalMetric`, `EvalRun`,
`EvalResult`, `EvalComparison`, `EvalGate`.

Support deterministic and model-judge evals.

Support adapter integrations rather than locking MyVista to one evaluation
framework.

### Intent evals

Measure: accuracy, macro-F1, micro-F1, per-intent precision, per-intent recall,
confusion matrix, top-k accuracy, calibration error, abstention accuracy,
unknown-intent recall, semantic-cache false-hit rate, classifier latency,
classification cost.

Maintain hard-negative datasets.

### Routing evals

For each request where feasible, compute or estimate counterfactual performance.

Track: route regret, quality regret, latency regret, cost regret, fallback rate,
unnecessary escalation rate, underpowered model rate, overpowered model rate.

### Generation evals

Provide adapters for: lm-evaluation-harness, DeepEval, RAG-specific metrics,
custom deterministic graders, human feedback, LLM-as-judge.

Track appropriate metrics depending on the task.

**Do not apply irrelevant metrics to all prompts.**

### Agent evals

Evaluate: task completion, tool correctness, argument correctness, step
efficiency, trajectory quality, loop detection, unnecessary tool calls, policy
adherence, final response quality.

### Safety evals

Create adversarial suites for: prompt injection, indirect prompt injection,
jailbreaks, PII leakage, secret leakage, cross-tenant leakage, tool escalation,
SQL injection, command injection, XSS output, unsafe URL behavior, malicious
retrieved documents, denial-of-wallet, unbounded generation, agent loops.

---

## Drift

Implement drift detection for: intent distribution, embedding distribution,
classifier confidence, unknown-intent frequency, route selection, latency, error
rate, cost, token length, context length, quality evaluation, fallback
frequency, safety blocks.

Do not automatically retrain solely because drift was detected.

Generate an actionable incident or candidate learning job.

---

## Release gates

A production change must be rejected automatically when critical gates fail.

Critical gates include: tenant isolation, authentication, authorization, safety
regression, intent regression, routing regression, schema regression,
availability regression, latency regression beyond tolerance, cost regression
beyond tolerance.

Store baseline and candidate results.

---

## Chaos engineering

Test: Ollama unavailable, Redis unavailable, Postgres unavailable, observability
unavailable, evaluator unavailable, provider returns 429, provider returns 500,
model times out, model streams partial output and disconnects, corrupted JSON,
context overflow, KV pressure, queue saturation, classifier failure, stale intent
cache, duplicate request, node shutdown.

The user-facing API must degrade predictably.

---

## Load testing

Implement k6 or an equivalent reproducible harness.

Separate benchmarks into: gateway-only, intent-cache, intent-classifier, router,
short-generation, long-generation, streaming, mixed workloads, agent workloads.

**Never publish one RPS number without identifying the workload.**

The initial gateway/control-plane benchmark target is **500 requests per second**
while preserving correctness and tenant isolation.

Benchmark reports must include: hardware, OS, software versions, model, context
length, input tokens, output tokens, concurrency, duration, p50, p95, p99, error
rate, TPS, CPU, memory, GPU.

---

## SDK

Build an ergonomic Python SDK first. Then TypeScript.

The public abstraction should expose concepts such as:

- `client.responses.create(...)`
- `client.chat.completions.create(...)`
- `client.embeddings.create(...)`
- `client.intents.classify(...)`
- `client.routes.preview(...)`
- `client.evals.run(...)`
- `client.traces.get(...)`

Maintain OpenAI compatibility where reasonable while exposing MyVista-specific
capabilities through optional extensions.

The standard path must stay simple.

---

## API versioning

Use `/v1`.

Do not leak internal database schemas into the public API.

Use typed versioned contracts.

Generate OpenAPI documentation.

---

## Data storage

Initial recommendation:

- **PostgreSQL** — durable configuration, tenancy, model registry, intent
  taxonomy, prompt registry
- **Redis/Valkey** — distributed hot caches, rate limits, ephemeral state
- **ClickHouse** — high-volume analytical telemetry
- **Object storage** — evaluation datasets, benchmark artifacts, exports

Keep storage interfaces abstract.

---

## Local development

The whole meaningful developer stack must start locally.

Target: `make dev` or `docker compose up`.

A Mac developer should be able to run MyVista, LiteLLM, Ollama, Postgres, Redis,
observability components and the dashboard **without Kubernetes**.

---

## Deployment

Support progressively: local Mac, Docker Compose, single VPC, Kubernetes,
multi-region.

Do not require Kubernetes for development.

Do not hard-wire any cloud vendor.

Create Terraform modules only after local behavior is correct.

---

## Code quality

Use: Python 3.12+, FastAPI, Pydantic v2, asyncio, httpx, SQLAlchemy 2, Alembic,
pytest, pytest-asyncio, ruff, mypy.

Use strict type checking in core modules.

- Do not swallow exceptions.
- Do not use bare `except`.
- Do not make synchronous network calls in async request paths.
- Bound queues.
- Bound retries.
- Bound contexts.
- Bound agent loops.
- Bound output tokens.

---

## Test requirements

Every production component needs: unit tests, integration tests, negative tests,
failure tests, property tests when useful.

Security-sensitive code requires explicit adversarial tests.

**No phase is finished while required tests fail.**

---

## Documentation

Maintain: `README.md`, `ARCHITECTURE.md`, `SECURITY.md`, `EVALUATIONS.md`,
`BENCHMARKS.md`, `CONTRIBUTING.md`, `docs/`.

`ARCHITECTURE.md` must remain synchronized with implementation.

---

## Change discipline

Before changing architecture:

1. Inspect the current implementation.
2. State the invariant affected.
3. Determine whether an ADR is required.
4. Modify the smallest sensible surface.
5. Run tests.
6. Run relevant evals.
7. Report actual results.
8. Do not claim success if tests or evals failed.
