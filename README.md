# LLM Fabric

[![CI](https://github.com/officialanubhavbhatia/llm-fabric/actions/workflows/ci.yml/badge.svg)](https://github.com/officialanubhavbhatia/llm-fabric/actions/workflows/ci.yml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](pyproject.toml)

**OpenAI-compatible inference gateway** with identity, tenant quotas, policy-based
model routing, provider failover, metering, and traces.

LLM Fabric authenticates callers, applies tenant quotas and deterministic
guardrails, routes `POST /v1/chat/completions` across heterogeneous providers
with a directed fallback graph, and records cost, provenance, and traces. It is
not an agent runtime, MCP registry, A2A fabric, or RAG system.

| | |
| --- | --- |
| **Try it** | [Quick Start](#quick-start) (~60 seconds, mock provider, no credentials) |
| **Architecture** | [Architecture](#architecture) · [`ARCHITECTURE.md`](ARCHITECTURE.md) · [`docs/constitution.md`](docs/constitution.md) |
| **Production** | [Production Deployment](#production-deployment) · [`PRODUCTION_READINESS.md`](PRODUCTION_READINESS.md) |
| **Contract** | [`docs/CONTRACT.md`](docs/CONTRACT.md) · OpenAPI at `/docs` when running |
| **IntentOS** | Experimental classifier. **Serving-path routing is OFF.** [IntentOS](#intentos) |

> [!IMPORTANT]
> **IntentOS serving-path routing is OFF.** LLM Fabric core is separately GO for a controlled internal single-VPC Kubernetes deployment. Do not conflate the two.

**Maturity (do not conflate these):**

- **LLM Fabric core** — **GO** for a *controlled internal single-VPC Kubernetes*
  deployment with managed PostgreSQL and Redis, API-key or OIDC callers, and
  Helm autoscaling **off**. Not authorized for public SaaS, billing, multi-region
  HA, or an inference-capacity SLA. Details: [`PRODUCTION_READINESS.md`](PRODUCTION_READINESS.md).
- **IntentOS serving-path routing** — **OFF**. Frozen evaluation did not pass the
  hard-negative gate. HashingEmbedder remains the default. MiniLM is opt-in.

License: none selected. Treat the tree as all-rights-reserved until the owner
publishes a `LICENSE`. See [`LICENSE_DECISION.md`](LICENSE_DECISION.md).

---

## Current state at a glance

IntentOS is **one component**, not the product.

| Capability | Status |
| --- | --- |
| Core LLM gateway (`/v1/chat/completions`, SSE) | Built. Internal-VPC production path exists. |
| OpenAI-compatible contract | Built. Honoured vs inert fields in [`docs/CONTRACT.md`](docs/CONTRACT.md). |
| Provider adapters | `mock`, `openai`, `anthropic`. Ollama via OpenAI adapter + `OPENAI_BASE_URL`. |
| Model registry and aliases | Built (`config/models.yaml`). |
| Policy-based model routing | Built (30 grades, seven policies, health, circuit breakers, fallback graph). |
| Route preview | Built (`POST /v1/routes/preview`). No inference. |
| Identity (OIDC, API keys, dev issuer) | Built. Production refuses to start without auth. |
| Multi-tenancy and quotas | Built. Postgres RLS when Postgres is configured. Production quotas are finite. |
| Deterministic input/output guardrails | Built on the chat path (size, secret/PII-shaped redact, injection markers). Retrieval / context / execution stages are types without RAG or tools to bind to. |
| Response / KV inference cache | **Not built.** Namespaces exist; they are not on the serving path. |
| IntentOS (L0–L3 classify) | Experimental. Default embedder: Hashing. |
| IntentOS serving-path routing | **OFF** |
| IntentOS shadow mode | Available (`LLM_FABRIC_INTENT_SHADOW`) |
| IntentOS MiniLM | Opt-in (`--extra embed`) |
| IntentOS L4 local rerank | Experimental, off by default |
| IntentOS paid L4 / L5 | **OFF** |
| IntentOS Phase B routing (30-grade planner) | **Not started** |
| Observability (Prometheus, OTLP HTTP, Command Center, optional Langfuse) | Built |
| Usage ledger | Built (Postgres or in-memory) |
| Evaluation platform | Built (`llm-fabric-eval`). Agent/safety suites are not. |
| Self-healing CLI | Built (`llm-fabric-heal`). Does not mutate a live process from the CLI. |
| Python SDK (`myvista`) / TypeScript SDK | Built. Chat, responses, classify, route preview, eval, traces. Embeddings and agents raise `UnsupportedError`. |
| Docker / Compose / Helm | Built. HPA off by default. |
| Agent orchestration | **Not built** |
| Tool execution / MCP / A2A | **Not built** |
| `/v1/embeddings`, RAG, vector DB, knowledge graph | **Not built** |
| Native vLLM adapter | **Not built** |

---

## Table of contents

1. [Overview](#overview)
2. [Why LLM Fabric?](#why-llm-fabric)
3. [Features](#features)
4. [Architecture](#architecture)
5. [Request lifecycle](#request-lifecycle)
6. [Quick Start](#quick-start)
7. [Local development](#local-development)
8. [Local vs production](#local-vs-production)
9. [Using LLM Fabric](#using-llm-fabric)
10. [Providers and models](#providers-and-models)
11. [Model and provider routing](#model-and-provider-routing)
12. [IntentOS](#intentos)
13. [Caching](#caching)
14. [Security](#security)
15. [Configuration](#configuration)
16. [Observability](#observability)
17. [Production deployment](#production-deployment)
18. [Scaling and reliability](#scaling-and-reliability)
19. [Performance](#performance)
20. [Evaluation](#evaluation)
21. [Operations runbook](#operations-runbook)
22. [Rollback](#rollback)
23. [Troubleshooting](#troubleshooting)
24. [Repository structure](#repository-structure)
25. [Development guide](#development-guide)
26. [Adding a provider](#adding-a-provider)
27. [Contributing](#contributing)
28. [Production checklist](#production-checklist)
29. [FAQ](#faq)
30. [Deeper documentation](#deeper-documentation)

---

## Overview

LLM Fabric is the common HTTP runtime in front of chat-completion backends.

What runs through it today:

- OpenAI-shaped **chat completions** (buffered and SSE)
- **Model discovery** (`GET /v1/models`)
- **Route preview** (explain a decision without calling a provider)
- **Usage** (`GET /v1/usage`)
- **Intent classification as an explicit API** (`POST /v1/intents/classify`) — independent of serving-path routing
- **Named eval runs** (`POST /v1/evals/run`, suite `ci` only)
- **Health, metrics, Command Center**

What does not run through it:

- Agent loops, tool calls, MCP servers, A2A
- Embeddings HTTP, retrieval, knowledge graphs
- Serving-path intent routing (the classifier exists; the serving switch is off)

A caller that already has an OpenAI SDK talks to Fabric by changing `base_url`.
A caller that wants provenance uses the `myvista` SDK or reads `x-fabric-*`
headers.

---

## Why LLM Fabric?

Applications that talk to providers directly each re-implement routing, failover,
auth, quotas, and metering — and they do it inconsistently.

```text
Application
    |
Provider-specific SDK
    |
One vendor, one contract, one failure mode
```

```text
Application
    |
LLM Fabric   (auth, tenant, quota, guardrails, planner, failover, meter, trace)
    |
+--------------------------+
| mock (local, no creds)   |
| OpenAI-compatible API    |
| Anthropic Messages API   |
+--------------------------+
```

What the fabric actually abstracts:

| Problem | What the code does |
| --- | --- |
| Inconsistent APIs | One OpenAI chat-completions dialect. Anthropic shape differences stay in the adapter. |
| Model switching | Registry ids and aliases (`auto`). Response `model` is the **served** id. |
| Provider failure | Directed fallback graph keyed by failure reason; circuit breakers; EWMA health. |
| Cost / latency / quality intent | Named policies over declared features. Missing scores are dropped, not invented. |
| Who paid / which model | Usage ledger + provenance headers on every buffered response. |
| Tenant isolation | Identity before routing; Postgres RLS; tenant-scoped stores and caches. |
| Observability | Bounded Prometheus labels, OTLP HTTP traces, Command Center. |
| Evaluation | Committed suites and gates (`llm-fabric-eval`). |
| Intent of the prompt | IntentOS **may** classify. It does **not** authorize. Serving-path routing stays off. |

It does **not** currently abstract agent orchestration, tool permissioning, or
retrieval. Those are specified in the constitution and are not on this serving
surface.

---

## Features

**Gateway.** FastAPI app in `src/llm_fabric/gateway/`. Default bind
`127.0.0.1:47317`.

**Routing.** `src/llm_fabric/router/`: Grade00–Grade29, policies
`quality_first`, `latency_first`, `cost_first`, `balanced`, `local_only`,
`private_only`, `custom`, plus `declared` for pins.

**Serving.** `src/llm_fabric/serving/adapters/`: mock, OpenAI, Anthropic.

**Identity.** OIDC JWKS, API keys bound to tenants, development issuer
(`POST /v1/dev/token`). Production fail-closed in `Settings.validate_startup`.

**Guardrails (chat path).** `InputGuardrail` / `OutputGuardrail` in
`src/llm_fabric/guardrails/`. Block or redact; not a model-backed safety stack.

**Metering.** Every completion writes usage. Fallback attempts are not folded
into the OpenAI `usage` object; they live on the ledger and `x-fabric-invocations`.

**Heal.** `llm-fabric-heal` analyzes a usage dump and prints proposals. In-process
`HealController` can open a breaker, shift traffic, roll back a remembered spec,
invalidate caches, raise an incident, or queue a learning job. Learning jobs are
not auto-promoted. Authorization policy cannot be mutated from this path.

---

## Architecture

```mermaid
flowchart TB
    Client[Application / OpenAI SDK / myvista]

    Client --> API[LLM Fabric HTTP API]

    API --> MW[Auth · tenant · quota · admission]
    MW --> Chat["POST /v1/chat/completions"]

    Chat --> InRail[Input guardrails]
    InRail --> Intent{"IntentOS<br/>off / shadow / on"}
    Intent --> Planner[Route planner]
    Planner --> Exec[Router execution]
    Exec --> Adapters[Provider adapters]
    Adapters --> OutRail[Output guardrails]
    OutRail --> Meter[Usage + traces + metrics]
    Meter --> Client

    Planner --> Preview["POST /v1/routes/preview"]
    Intent --> Classify["POST /v1/intents/classify"]

    Adapters --> Mock[mock]
    Adapters --> OAI[OpenAI-compatible]
    Adapters --> Anth[Anthropic]

    Meter --> Prom["GET /metrics"]
    Meter --> OTel[OTLP HTTP]
    Meter --> CC["/command-center"]
    Meter --> PG[(PostgreSQL)]
    MW --> Redis[(Redis / Valkey)]
```

Admission refuses new inference when a **required** dependency (Postgres/Redis
when this process uses them) is cached unhealthy. `/healthz` stays up so
Kubernetes does not restart-storm on a database incident. `/readyz` is the
readiness signal.

### Core components

| Component | Responsibility | Production status | Code / docs |
| --- | --- | --- | --- |
| API / gateway | HTTP contract, SSE, errors | Built; internal-VPC GO | `src/llm_fabric/gateway/` |
| Identity | OIDC, API keys, dev issuer, revocation | Built; production fail-closed | `src/llm_fabric/identity/` · [`SECURITY.md`](SECURITY.md) |
| Tenancy / quotas | Tenant scope, RPM/concurrency | Built; Redis-shared in production | `src/llm_fabric/tenancy/` |
| Guardrails | Deterministic INPUT/OUTPUT on chat | Built (narrow) | `src/llm_fabric/guardrails/` |
| Router | Plan + execute + fallback + breakers | Built | `src/llm_fabric/router/` |
| Provider adapters | mock / openai / anthropic | Built | `src/llm_fabric/serving/` |
| IntentOS | Classify prompts; optional routing signal | Experimental; routing **OFF** | `src/llm_fabric/intent/` |
| Tenant cache | Process or Redis; intent L0/L1 use it | Built for those namespaces | `src/llm_fabric/tenancy/cache.py` |
| Observability | Metrics, traces, Command Center | Built | `src/llm_fabric/observability/` |
| Storage | Postgres + Alembic; in-memory fallback | Built; production requires Postgres | `src/llm_fabric/storage/` |
| Eval | Suites, runs, gates | Built | `src/llm_fabric/eval/` |
| Heal | Drift analysis and controls | Built (CLI + in-process) | `src/llm_fabric/heal/` |
| Context budget | Token headroom helper | Library; not a full compiler on every request | `src/llm_fabric/context/` |
| Agent runtime / MCP / RAG | — | **Not built** | — |

---

## Request lifecycle

One `POST /v1/chat/completions` (implementation:
`src/llm_fabric/gateway/middleware.py`, `routes/chat.py`, `router/engine.py`).

```text
 1. Client sends JSON (OpenAI chat shape)
 2. Body size checked against LLM_FABRIC_MAX_REQUEST_BYTES
 3. Inference admission: required deps must be cached healthy
 4. Authentication; tenant from the token, never from the body
 5. Quota admit + concurrency slot (Redis when configured)
 6. Request validated (messages required, n must be 1, …)
 7. Input guardrails (size, secret/PII-shaped patterns, injection markers)
 8. Optional IntentOS classify (newest user turn)
     - classification_enabled: planner may infer policy / capabilities
     - shadow: headers only; route unchanged
     - both off: cascade is not built; classify API still works offline
 9. Planner selects policy, candidates, fallback graph
10. Provider call(s); retryable failures follow the graph
11. Streaming: no failover after the first generated byte (ADR 0003)
12. Output guardrails (block / redact)
13. Usage record, Prometheus, OTel span export
14. Response: OpenAI body + x-fabric-* provenance
```

| Step | Config / failure |
| --- | --- |
| 3 Admission | Postgres/Redis down → `503` on inference, not on `/healthz`. |
| 4 Auth | Missing/invalid credential → `401`. Production never starts anonymous. |
| 5 Quota | Exceeded → `429 quota_exceeded`. Rejected calls do not invoke the provider. |
| 7 Input rails | `GuardrailBlockedError`. |
| 8 IntentOS | Classifier exceptions are logged; the request still routes. Classification never grants access. |
| 9 No candidate | `NoCandidateError` / unknown model **before** streaming starts. |
| 10 Fallback | `LLM_FABRIC_MAX_ATTEMPTS` (default 3). Optional cost/latency ceilings. |
| 11 Mid-stream failure | SSE `error` frame + `[DONE]`; HTTP status already 200. |
| 12 Output rails | Block or redact secret-shaped spans in the completion. |

Pinned `model` uses policy `declared`. Alias `auto` uses the alias policy
(default registry: `cost_first`) unless serving-path IntentOS is on and infers
another policy.

---

## Quick Start

**Recommended for first-time users.** Mock provider only. No API keys.

Requires [Python 3.12+](https://www.python.org/) and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/officialanubhavbhatia/llm-fabric.git
cd llm-fabric
cp .env.example .env   # already sets LLM_FABRIC_ENVIRONMENT=development
make install
make dev
```

`make dev` runs uvicorn with reload and `LLM_FABRIC_ENVIRONMENT=development`.
The gateway listens on **`http://127.0.0.1:47317`**.

This path assumes `.env` matches `.env.example`: no `DATABASE_URL` / `REDIS_URL`,
no API keys, `ALLOW_ANONYMOUS=true`, `REGISTRY_PATH=config/models.yaml`. Leftover
`LLM_FABRIC_*` variables from a previous production-like run will change auth,
admission, and which model `auto` selects.

```bash
curl -s http://127.0.0.1:47317/healthz
# {"status":"ok", ...}

curl -si http://127.0.0.1:47317/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"auto","messages":[{"role":"user","content":"Hello"}]}'
```

Look for `x-fabric-served-model` and `x-fabric-policy`. The mock provider
returns text assembled from the request; it does not call a network LLM.

OpenAPI: [http://127.0.0.1:47317/docs](http://127.0.0.1:47317/docs).

Stop: `Ctrl-C` in the `make dev` terminal.

IntentOS is not required for this path.

### Docker Compose (production-like local stack)

**Recommended when you need Postgres + Redis + auth**, not for the first
hello-world. Compose is a production-*like* test stack, not the internal-VPC
tier.

```bash
docker compose -f deployments/docker/docker-compose.yml up --build
```

Gateway: `http://127.0.0.1:47317` with `LLM_FABRIC_AUTH_MODE=api_key` and
`LLM_FABRIC_API_KEYS=production-test-key-16`.

```bash
curl -s http://127.0.0.1:47317/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer production-test-key-16' \
  -d '{"model":"auto","messages":[{"role":"user","content":"Hello"}]}'
```

Optional Ollama sidecar: `--profile inference` (image `ollama/ollama:0.3.14`).
Point `LLM_FABRIC_OPENAI_BASE_URL` at that server and enable an OpenAI-shaped
model in the registry. There is no native Ollama adapter.

### Native production-shaped process (no reload)

```bash
make serve    # python -m llm_fabric
```

Same port. Use this for local load tests. `make bench-load` currently assumes
**port 8000** (`Makefile` `LOAD_PORT`); start with `LLM_FABRIC_PORT=8000` or
override `LOAD_PORT=47317`.

---

## Local development

**Recommended for contributors.**

### Prerequisites

| Dependency | Notes |
| --- | --- |
| Python ≥ 3.12 | `requires-python` in `pyproject.toml` |
| uv | `make install` → `uv sync --extra dev` |
| Docker | Optional: Compose stack, image builds |
| PostgreSQL 16 / Redis 7 | Optional locally; **required** for `ENVIRONMENT=production` |
| GPU | Not required. MiniLM uses FastEmbed on CPU when `--extra embed` is installed |
| Model weights | Only if you opt into MiniLM / a local OpenAI-compatible server |

### Install

```bash
make install
# optional IntentOS MiniLM:
uv sync --extra embed
```

### Environment

```bash
cp .env.example .env
# LLM_FABRIC_ENVIRONMENT is required (development | test | production)
```

Full variable list: [`docs/CONFIGURATION.md`](docs/CONFIGURATION.md).

### Start

```bash
make dev                          # reload, development
# or
make serve                        # production-shaped uvicorn
# or Compose (Postgres, Redis, migrate Job, gateway):
docker compose -f deployments/docker/docker-compose.yml up --build
```

### Verify

```bash
curl -s http://127.0.0.1:47317/healthz
curl -s http://127.0.0.1:47317/readyz
make doctor                       # PASS/WARN/FAIL without binding extra servers
```

### Tests, lint, types

```bash
make lint
make typecheck                    # mypy src
make test                         # pytest --strict-markers
make test-isolation               # pytest -m isolation
make check                        # lint + types + tests
make eval-gate                    # llm-fabric-eval gate vs datasets/eval/baseline.json
```

`LLM_FABRIC_ENVIRONMENT=test` is what CI sets. Do not use `SKIP_EVALS` /
`LLM_FABRIC_SKIP_EVALS`; the gate refuses them.

### Stop / reset

- `make dev` / `make serve`: `Ctrl-C`.
- Compose: `docker compose -f deployments/docker/docker-compose.yml down`.
- `make clean` removes caches and **`artifacts/`**. It does not drop Docker volumes.

Do not run `alembic downgrade` or `DROP DATABASE` as a routine reset.

---

## Local vs production

| Concern | Local (`development`) | Production (`production`) |
| --- | --- | --- |
| Server | `make dev` (reload) or `make serve` | `python -m llm_fabric` / container `CMD` / Helm |
| Bind | `127.0.0.1:47317` | `0.0.0.0:47317` behind a private LB |
| Configuration | `.env` | Kubernetes Secret + ConfigMap |
| Auth | Anonymous allowed if no identity source (`ALLOW_ANONYMOUS=true`) | Mandatory OIDC or API keys; anonymous/`dev` refused |
| Provider credentials | Optional (mock enabled) | Secret; enable models in registry |
| Postgres | Unset → in-memory | Required; DML role `fabric_app` |
| Redis | Unset → per-process quotas/breakers | Required; shared across replicas |
| Migrations | Optional | Helm pre-upgrade Job as table owner |
| Logging | stdout | Centralized scrape of stdout |
| Metrics | `GET /metrics` | Prometheus scrape |
| Tracing | In-process Command Center unless OTLP set | OTLP HTTP to collector |
| IntentOS routing | OFF | OFF |
| IntentOS shadow | Optional | Observation only |
| MiniLM | Opt-in extra | Not default |
| TLS | None in-process | Terminate at the edge (not in this image) |
| Scaling | One process | `replicaCount` ≥ 2, HPA **off** until proven |
| Quotas | Unlimited if unset | Finite defaults if unset |
| CORS `*` | Possible | Refused at startup |

---

## Using LLM Fabric

Base URL for OpenAI clients: `http://127.0.0.1:47317/v1`.
Python/TS SDK default: `http://127.0.0.1:47317` (they append `/v1`).

Development with anonymous access needs no `Authorization` header. Production
and Compose need a bearer token or `x-api-key`.

Honoured vs ignored request fields: [`docs/CONTRACT.md`](docs/CONTRACT.md).
`tools`, `n>1`, and `response_format` are **not** executed/validated by this
fabric. `response_format` is accepted and ignored; ask for JSON in the prompt.

### Basic chat

**cURL**

```bash
curl -s http://127.0.0.1:47317/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"auto","messages":[{"role":"user","content":"Hello"}]}'
```

**Python** (`myvista`, installed with this repo)

```python
from myvista import MyVista

client = MyVista()  # MYVISTA_BASE_URL / MYVISTA_API_KEY optional
response = client.chat.completions.create(
    model="auto",
    messages=[{"role": "user", "content": "Hello"}],
)
print(response.text)
print(response.request_id, response.fabric.served_model)
```

**OpenAI SDK**

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:47317/v1", api_key="unused")
response = client.chat.completions.create(
    model="auto",
    messages=[{"role": "user", "content": "Hello"}],
)
print(response.model)  # served model, not the alias
```

**TypeScript** (in-repo `sdk/typescript`)

```typescript
import { MyVista } from "myvista";

const client = new MyVista();
const response = await client.chat.completions.create({
  model: "auto",
  messages: [{ role: "user", content: "Hello" }],
});
```

Also: `client.responses.create(input="Hello")` — same completions route.

### Streaming

```bash
curl -N http://127.0.0.1:47317/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"auto","messages":[{"role":"user","content":"Hello"}],"stream":true}'
```

SSE, terminated by `data: [DONE]`. See [`docs/CONTRACT.md`](docs/CONTRACT.md)
and `examples/python/async_stream.py`.

### Model selection

Send a registry **id** (`mock-small`) or an **alias** (`auto`, `auto-reasoning`).
Unknown or disabled ids fail before a provider is called.

### Routing preview (no inference)

```bash
curl -s http://127.0.0.1:47317/v1/routes/preview \
  -H 'Content-Type: application/json' \
  -d '{"model":"auto","messages":[{"role":"user","content":"Hello"}]}'
```

Python: `examples/python/route_preview.py`. Tenant always comes from the token.

### Structured output

Not validated. `examples/python/structured_output.py` shows `response_format`
accepted and ignored.

### Embeddings / agents / tools / batch

- `client.embeddings` and `client.agents` raise `UnsupportedError`
  (`examples/python/embeddings.py`, `examples/python/agents.py`).
- Tool calls are ignored on the wire, not executed.
- There is no batch completions API.

### Classify (explicit IntentOS API)

Does not enable serving-path routing.

```bash
curl -s http://127.0.0.1:47317/v1/intents/classify \
  -H 'Content-Type: application/json' \
  -d '{"input":"debug this python traceback"}'
```

### Usage and models

```bash
curl -s http://127.0.0.1:47317/v1/models
curl -s http://127.0.0.1:47317/v1/usage
curl -s http://127.0.0.1:47317/v1/routes/health
```

More examples: [`examples/python`](examples/python), [`examples/typescript`](examples/typescript).

---

## Providers and models

Adapters in `src/llm_fabric/serving/factory.py`: **`mock`**, **`openai`**,
**`anthropic`**.

| Provider | Streaming | Tools executed | Structured output validated | Embeddings HTTP |
| --- | --- | --- | --- | --- |
| mock | yes | no | no | no |
| openai (incl. OpenAI-compatible: Ollama, proxies) | yes | no (`tools` ignored) | no | no |
| anthropic | yes | no | no | no |

Registry: [`config/models.yaml`](config/models.yaml). Default tree enables only
`mock-small` and `mock-large`. OpenAI/Anthropic entries are `enabled: false`
until you set keys, fill **operator-supplied** USD/MTok prices, and enable them.
A price of `0.0` means “not filled in”, not free — cost ranking is dropped for
the whole decision if any candidate is unpriced.

**Credentials**

- `LLM_FABRIC_OPENAI_API_KEY` or `OPENAI_API_KEY`
- `LLM_FABRIC_OPENAI_BASE_URL` (default `https://api.openai.com/v1`)
- `LLM_FABRIC_ANTHROPIC_API_KEY` or `ANTHROPIC_API_KEY`

**Timeouts / retries.** Per-attempt `LLM_FABRIC_REQUEST_TIMEOUT_S` (60).
`LLM_FABRIC_MAX_ATTEMPTS` (3) across the fallback graph. The **client** SDK
retries HTTP 408/429/5xx separately (`myvista._http`).

**Rate limits.** Tenant/user quotas (Redis). Provider HTTP 429 is a fallback
reason (`rate_limited`). There is no separate token-bucket against OpenAI’s
dashboard limits beyond that.

There is no extra “internal LLM gateway” process. This repository *is* the
gateway.

---

## Model and provider routing

Three different mechanisms. Do not conflate them.

```text
Authorization     →  identity + scopes + tenant (middleware)
Intent classification →  optional label + capabilities (IntentOS)
Model/provider routing →  planner + registry + health + fallback graph
```

**Planner** (`src/llm_fabric/router/plan.py`):

1. Resolve alias or pin.
2. Choose policy: request/alias → (only if serving-path classification is on)
   intent-inferred policy → `LLM_FABRIC_DEFAULT_POLICY`.
3. Filter by capabilities, locality (`local_only` / `private_only`), grade,
   tenant allow-lists, open breakers.
4. Score remaining candidates. Missing quality/latency/cost features are
   **dropped for everyone**, not imputed.
5. Attach a directed fallback graph (`src/llm_fabric/router/fallback.py`).

**Execution** (`src/llm_fabric/router/engine.py`): try the winner; on a retryable
error, follow edges matching that **reason** (timeout vs context-too-large are
different). Loops cannot re-visit a deployment. Depth and optional cost/latency
budgets apply.

**Intent-based routing** is the planner consuming an `IntentClassification`.
That happens only when `LLM_FABRIC_INTENT_CLASSIFICATION_ENABLED=true`. Default
**false**. Shadow mode classifies and **does not** pass the label into the
planner.

Route preview returns the same decision object without calling a provider.

---

## IntentOS

IntentOS is LLM Fabric’s **experimental intent-classification layer**. It
produces routing **information**. It is not authority.

```text
User request
     |
LLM Fabric (auth, quota, guardrails)
     |
IntentOS  →  classification (optional)
     |
Router    →  model / provider  (today: ignores IntentOS on the serving path)
```

> [!WARNING]
> `classification != authorization`. A label must never grant tool access, admin
> rights, provider credentials, tenant access, secrets, filesystem, or network
> privileges. Tool/agent surfaces are not built; the invariant still applies to
> any future wiring.

### Architecture

```text
L0  exact intent cache
L1  semantic intent cache   (production similarity threshold 0.80)
L2  deterministic rules
L3  embedding nearest-centroid   default HashingEmbedder; MiniLM opt-in
L2∩L3 agreement
L4  local description rerank     experimental, off
L5  escalation model             off
ABSTAIN / unknown
```

Classifier failure never fails the chat request.

### Current production status

```text
Routing rollout:     OFF
Default embedder:    HashingEmbedder
MiniLM:              opt-in
Local L4:            experimental
Paid L4:             OFF
L5:                  OFF
Phase B:             not started
Shadow mode:         available
```

### Frozen evaluation (authoritative v1 Hashing)

Frozen 98-case set: `datasets/eval/intentos/final-2026.08.24.json`.
Do not tune on it. Do not edit it to make a candidate pass.

| Metric | Value |
| --- | ---: |
| Accuracy | 0.9082 |
| Macro F1 | 0.9269 |
| Unknown recall | 0.8571 |
| Abstention precision | 0.7059 |
| ECE | 0.1767 |
| High-conf precision @ 0.90 | 0.964 |
| Coverage @ 0.90 | 0.286 |
| Semantic false hit @ 0.60 | 0.00 |
| Hard-negative accuracy | 0.50 |
| Ordinary slice | 0.986 |

Gate result: **ALL GATES NOT PASSED**

```text
hard_negative_accuracy  required >= 0.58
                        current  = 0.50
```

Therefore **intent routing remains OFF.**

Criteria: [`docs/INTENTOS_SUCCESS_CRITERIA.md`](docs/INTENTOS_SUCCESS_CRITERIA.md).
Narrative: [`INTENTOS_EVALUATION.md`](INTENTOS_EVALUATION.md),
[`docs/INTENTOS.md`](docs/INTENTOS.md), [`docs/INTENTOS_V1.1.md`](docs/INTENTOS_V1.1.md).

### Shadow mode (recommended observation)

```bash
export LLM_FABRIC_INTENT_SHADOW=1
```

Classification runs on the serving path. Output is on
`x-fabric-intent-shadow-*` headers. **The route is unchanged.** Ignored if
`LLM_FABRIC_INTENT_CLASSIFICATION_ENABLED` is true.

MiniLM comparison (more memory; does not fix HN accuracy; currently reduces
unknown recall on the frozen set):

```bash
export LLM_FABRIC_INTENT_SHADOW=1
export LLM_FABRIC_INTENT_EMBEDDER=minilm
# requires: uv sync --extra embed
```

MiniLM remains opt-in. It is not the default router embedder.

### Adding or changing an intent

1. Taxonomy snapshot under `datasets/intent/taxonomy/` (published snapshots are immutable).
2. Examples in `datasets/intent/` (`bootstrap.jsonl` is frozen for v1 hashes).
3. Prototypes / hard negatives as documented in [`docs/INTENTOS.md`](docs/INTENTOS.md).
4. Validate on **validation** data, not the frozen 98.
5. Abstention is the honest floor.
6. Never edit frozen examples just to make a candidate pass.
7. Re-run security/isolation tests (`make test-isolation`).

---

## Caching

| Layer | What it is | On chat serving path? |
| --- | --- | --- |
| Tenant-scoped process/Redis | Quotas, breakers, revocation, IntentOS L0/L1 | Supporting state, not a response cache |
| IntentOS L0 exact | Identical normalized prompt + tenant + versions | Only when IntentOS cascade is built |
| IntentOS L1 semantic | Cosine vs stored embeddings | Same |
| Exact/semantic **response** namespaces | Declared in `CacheNamespace` | **Not wired** to completions |
| Provider prompt cache | — | **Not implemented** |
| Embeddings HTTP cache | — | No embeddings API |

Tenant isolation: cache keys include tenant. Cross-tenant hits are treated as
isolation bugs (`tests/security/`).

### IntentOS semantic cache (measured)

Threshold used in the reported production setting: **0.80**.
17 paraphrases. Artifact: `datasets/eval/intentos/cache-1.1.json`.

| Embedder | Hit rate @ 0.80 | False hit |
| --- | ---: | ---: |
| Hashing | 0.12 | 0.00 |
| MiniLM | 0.71 | 0.00 |

Cache quality does **not** justify promoting MiniLM as the intent router.
Unknown recall and hard-negative accuracy still gate serving-path routing.

---

## Security

Trust boundaries:

| Boundary | What happens |
| --- | --- |
| Client → API | TLS is an edge concern. Process speaks HTTP. |
| API | Auth middleware before routes. Unauthenticated callers do not enumerate `/v1/*`. |
| Tenant | Scope from token. `x-tenant-id` only with delegation scope. Postgres RLS. |
| Providers | Server-side keys. Never taken from the client body. |
| Caches / DB / Redis | Tenant discriminators. Redis loss: quotas/revocation fail-closed policy — see [`docs/BACKUP_RECOVERY.md`](docs/BACKUP_RECOVERY.md). |
| Model output | Deterministic output redact/block. Not a full safety model. |
| IntentOS | Signal only. Shadow cannot change the route. |
| Agents / MCP | Not present; no tool permission surface to confuse with classification. |

**Secrets.** Environment / Kubernetes Secret. Image does not bake keys
(`deployments/docker/Dockerfile`). OTLP headers belong in a Secret.

**Prompt injection.** Guardrails look for injection **markers** on input. This is
not a certified prompt-injection defense.

**Automated tests, not a certification.** `tests/unit/test_intentos_v1.py` has
been run at 66 passed in the v1.1 notes. The cross-tenant suite is
`make test-isolation` (`pytest -m isolation`). Passing tests are not a
penetration-test sign-off.

Report vulnerabilities privately: [`SECURITY.md`](SECURITY.md).

---

## Configuration

All gateway settings: [`docs/CONFIGURATION.md`](docs/CONFIGURATION.md) and
[`.env.example`](.env.example).

Minimum local:

```bash
LLM_FABRIC_ENVIRONMENT=development
LLM_FABRIC_HOST=127.0.0.1
LLM_FABRIC_PORT=47317
```

Minimum production (also required: reachable Postgres + Redis, complete auth,
finite quotas via defaults or explicit caps):

```bash
LLM_FABRIC_ENVIRONMENT=production
LLM_FABRIC_ALLOW_ANONYMOUS=false
LLM_FABRIC_DATABASE_URL=...    # fabric_app
LLM_FABRIC_REDIS_URL=...
# plus OIDC or API credentials
```

`make doctor` checks these without serving traffic.

---

## Observability

| Signal | Where |
| --- | --- |
| Logs | Structured stdout (`src/llm_fabric/observability/logging.py`) |
| Metrics | `GET /metrics` (Prometheus). Labels are a **closed set** — no tenant, user, or request id |
| Traces | OTLP HTTP if `LLM_FABRIC_OTEL_EXPORTER_OTLP_ENDPOINT` is set; otherwise in-process |
| Command Center | `GET /command-center` — local-pod diagnostic, **not** fleet history. Dashboard JSON is tenant-scoped unless the caller has `fabric:observe` or an operator role. `/metrics` is public. |
| Usage | `GET /v1/usage` · [`docs/USAGE_METERING.md`](docs/USAGE_METERING.md) |
| Langfuse | Optional; missing config is a no-op; never fails a request |

### Platform health

`fabric_requests_total`, `fabric_request_duration_seconds`,
`fabric_active_requests`, admission rejections, dependency gauges
(`src/llm_fabric/observability/prom.py`).

### Provider health

Per-deployment EWMA + breakers: `GET /v1/routes/health`. Token and cost
counters with bounded model/provider/policy labels.

### Routing

Provenance headers; usage `invocations`; route preview explanations.

### Cache / IntentOS safety (when the cascade runs)

Prefer **unknown recall, hard-negative accuracy, abstention precision, semantic
false hits, shadow disagreement** over aggregate accuracy. Command Center views
that have no backend stay empty rather than inventing series.

---

## Production deployment

Canonical verdict: [`PRODUCTION_READINESS.md`](PRODUCTION_READINESS.md)
(audit 2026-08-24). **GO** for internal single-VPC Kubernetes. **NO-GO** for
public SaaS, billing-grade metering, multi-region failover, unattended HPA.

```mermaid
flowchart TB
    LB[Internal load balancer / TLS edge]
    subgraph Fabric["LLM Fabric replicas (Helm replicaCount 2, HPA off)"]
      API1[Gateway]
      API2[Gateway]
    end
    Redis[(Redis: quotas, breakers, revocation)]
    PG[(PostgreSQL: ledger, tenants, RLS)]
    Prom[Prometheus]
    OTel[OTLP collector]
    Models[OpenAI / Anthropic / OpenAI-compatible]

    LB --> API1
    LB --> API2
    API1 --> Redis
    API2 --> Redis
    API1 --> PG
    API2 --> PG
    API1 --> Models
    API2 --> Models
    API1 --> Prom
    API2 --> Prom
    API1 --> OTel
    API2 --> OTel
```

### Docker image

```bash
docker build -f deployments/docker/Dockerfile -t llm-fabric:0.1.0 .
```

- User `10001`, `EXPOSE 47317`, `CMD python -m llm_fabric`
- Healthcheck hits `/healthz`
- `STOPSIGNAL SIGTERM`; graceful shutdown default 25s

### Docker Compose

[`deployments/docker/docker-compose.yml`](deployments/docker/docker-compose.yml):
Postgres (roles from `postgres-init/`), Redis, OTEL collector, Prometheus,
Grafana, migrate Job, gateway. **Not** the production tier.

### Kubernetes / Helm

Chart: [`deployments/helm/llm-fabric/`](deployments/helm/llm-fabric/).

| Item | Chart default |
| --- | --- |
| `replicaCount` | 2 |
| Service | ClusterIP `47317` |
| Probes | `/healthz` live/startup, `/readyz` ready |
| Resources | 250m/512Mi request, 2 CPU / 2Gi limit |
| HPA | `autoscaling.enabled: false` |
| PDB | `minAvailable: 1` |
| `terminationGracePeriodSeconds` | 30 |
| Migrations | pre-install/pre-upgrade Job |

Secret must contain `LLM_FABRIC_MIGRATION_DATABASE_URL` (DDL owner) and
`LLM_FABRIC_DATABASE_URL` (DML `fabric_app`). Workers never migrate.

There is no cloud-provider module in this repository. Use the Helm chart on
your cluster.

### TLS, secrets, scaling

Terminate TLS at the ingress. Put provider keys and OTLP headers in Secrets.
Scale with **pods**, each one worker, sharing Redis. Do not enable
`ALLOW_UNSAFE_MULTIWORKER`. Do not treat Helm HPA YAML as a verified capacity
control.

---

## Scaling and reliability

| Mechanism | Behavior |
| --- | --- |
| Concurrency | uvicorn asyncio, one worker by default. Production breaker max concurrency 256 per deployment if unset. |
| Quotas | Redis sliding windows; `429` before provider call. |
| Retries | Fallback graph + attempt cap. SDK has its own HTTP retries. |
| Circuit breakers | Consecutive failures and error-rate EWMA; open → half-open. |
| Streaming | No model splice after first byte. SIGTERM drains in-flight SSE within graceful timeout (measured on mock). |
| Idempotency | Not implemented as a client key. Retries may call the provider again. |
| Cache outage | Redis required in production; cached-unhealthy Redis sheds new inference. Intent caches miss and recompute. |
| DB outage | Same admission rule. `/healthz` stays up. |
| IntentOS down | Classify errors degrade routing to default policy; they do not 500 the chat. |

| Failure | Fabric behavior |
| --- | --- |
| Primary provider timeout | Fallback edge `timeout` if attempts remain |
| All candidates fail | `AllCandidatesFailedError` |
| Cache / Redis unavailable (prod) | Admission rejects new inference |
| Postgres unavailable (prod) | Admission rejects new inference |
| IntentOS exception | Log; continue without intent |
| L4 malformed | Cascade abstains / continues; chat still served |
| MCP timeout | N/A — MCP not built |

---

## Performance

Do not quote historical 1,000 or 2,377 req/s as this release. Currently
reproduced **Compose mock gateway** load (2026-08-24): **237 req/s**, 0 errors
(p50 132 ms, p95 168 ms, p99 221 ms). Command and caveats:
[`docs/BENCHMARKS.md`](docs/BENCHMARKS.md).

That is gateway RPS against mock, not token throughput, not GPU inference.

### IntentOS HTTP overhead (separate measurement)

Pinned `mock-small`, anonymous development, artifact
`datasets/eval/intentos/http-1.1.json`. **Not** Compose production-like load.
**Not** a production SLA.

| Mode | RPS | p50 | p95 | p99 | RSS |
| --- | ---: | ---: | ---: | ---: | ---: |
| off | 1575 | 9.98 ms | 11.54 | 13.48 | 109 MB |
| hashing | 1352 | 11.73 | 13.35 | 15.27 | 109 MB |
| MiniLM | 1321 | 11.99 | 13.51 | 15.47 | 342 MB |
| MiniLM+L4 | 1304 | 12.16 | 13.65 | 15.96 | 363 MB |

MiniLM’s primary current cost on this path is **memory**, not hot-path latency
(the measured prompt was L2-accepted, so L3/L4 rarely ran).

A load/HTTP bench that uses `multiprocessing` must be launched from a **real
`.py` file**. Spawned workers cannot reopen a stdin-only script (`python -` /
piped stdin). That is why an earlier stdin-launched HTTP job was aborted.

---

## Evaluation

| Kind | What exists |
| --- | --- |
| CI suite | `datasets/eval/ci-suite.yaml` · `make eval-run` / `make eval-gate` |
| Chat contract | `tests/contract/` |
| Isolation | `make test-isolation` |
| Chaos / degraded deps | `tests/chaos/`, `tests/system/` |
| Routing eval | Fixture match (`route_match`) — **not** routing quality |
| IntentOS | Frozen 98 + gates in [`docs/INTENTOS_SUCCESS_CRITERIA.md`](docs/INTENTOS_SUCCESS_CRITERIA.md) |
| Load | `llm-fabric-load` · [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) |
| DeepEval / lm-eval | Optional extras; adapters report unavailable if missing |
| Agent / safety eval | **Not built** |

IntentOS frozen policy:

```text
validation  →  policy selection  →  LOCK  →  one frozen-98 run  →  gate
```

Do **not** tune on the frozen 98. Do **not** lower gates after seeing results.

HTTP eval: `POST /v1/evals/run` with `"suite": "ci"` only — no client filesystem
paths.

---

## Operations runbook

| Task | Command / note |
| --- | --- |
| Start locally | `make dev` or `make serve` |
| Stop locally | `Ctrl-C` |
| Production process | `python -m llm_fabric` (image `CMD`) |
| Health | `GET /healthz` live · `GET /readyz` ready · `make doctor` |
| Logs | stdout of the process / pod |
| Metrics | `GET /metrics` |
| Trace one request | Send `x-request-id`; Command Center `/v1/observability/traces/{id}` is **local pod** |
| Test provider | Enable in `config/models.yaml`, set key, `POST /v1/chat/completions` with that id |
| Disable provider | `enabled: false` in registry; roll config |
| Shadow IntentOS | `LLM_FABRIC_INTENT_SHADOW=1` |
| Compare MiniLM | shadow + `LLM_FABRIC_INTENT_EMBEDDER=minilm` + `--extra embed` |
| Clear **dev** process cache | Restart the process (in-memory). Redis: flush only a **dev** instance |
| Eval | `make eval-gate` |
| Security isolation | `make test-isolation` |
| Load | `llm-fabric-load --host 127.0.0.1 --port 47317 ...` (see Makefile port mismatch) |
| Usage CLI | `llm-fabric-usage` |
| Heal analyze | `make heal-analyze RECORDS=path.json` (does not mutate live) |
| Migrate | `make migrate` as **table owner**, never as `fabric_app` on empty DB |

---

## Rollback

**Fabric release.** Helm rollback / previous image tag. Workers are stateless
aside from Redis/Postgres. In-flight SSE should drain within
`terminationGracePeriodSeconds`.

**Provider / routing config.** Revert `config/models.yaml` (or ConfigMap).
Pins and allow-lists live there.

**IntentOS (known-safe serving path):**

```text
LLM_FABRIC_INTENT_CLASSIFICATION_ENABLED=false
LLM_FABRIC_INTENT_EMBEDDER=hashing
LLM_FABRIC_INTENT_L4_RERANK=false
# L5 is not a flag; it is off in code
```

Shadow may stay on for observation. Phase B routing has not started; there is
nothing to roll back there.

Heal can roll back a **remembered** model spec / prompt / pinned classifier
inside a process; that is not a substitute for shipping a previous image.

---

## Troubleshooting

| Symptom | What to check |
| --- | --- |
| Chat returns 503 `dependency_unavailable` | `.env` has `DATABASE_URL` or `REDIS_URL` but that store is down. Unset them for mock-only local, or start Compose. |
| Chat hits a real model / 401 instead of mock | Shell or `.env` still has keys, `AUTH_MODE`, or a custom `REGISTRY_PATH`. Compare with `.env.example`. |
| Process will not start | `LLM_FABRIC_ENVIRONMENT` set? Production: auth + Postgres + Redis + Alembic head? Run `make doctor`. |
| `gateway is running without authentication` | Development/test only. Production treats that as `ConfigurationError`. |
| Provider 401/403 | Key env vars; registry `enabled`; `OPENAI_BASE_URL` if using a proxy. |
| Unknown model | Id/alias in `config/models.yaml`; disabled models are unknown to callers. |
| Provider timeout | `LLM_FABRIC_REQUEST_TIMEOUT_S`; fallbacks; breaker open on `/v1/routes/health`. |
| Redis/Postgres | Compose healthchecks; production admission 503 on inference. |
| MiniLM memory | RSS ~340 MB in the HTTP bench vs ~109 MB hashing. Expected. |
| IntentOS unknown-recall drop | MiniLM candidate B measured 0.714 on frozen 98 (fail). Stay on Hashing for default. |
| Structured JSON invalid | Fabric does not validate schemas. |
| `make bench-load` empty | Gateway must already be up; Makefile default port **8000**, app default **47317**. |
| Multiprocessing HTTP bench from stdin | Spawn cannot reopen stdin. Write the driver to a `.py` file and run that. |
| Eval gate skip | `SKIP_EVALS` is refused. Fix the regression or take an audited override. |
| MCP / agent errors | Those APIs are not implemented (`UnsupportedError`). |

---

## Repository structure

```text
.
├── alembic/                  PostgreSQL migrations
├── config/models.yaml        Model registry and aliases
├── datasets/eval/            Eval suites, IntentOS frozen artifacts
├── datasets/intent/          Taxonomy, bootstrap, val, adversarial
├── deployments/docker/       Dockerfile, Compose, OTEL, Prometheus, Postgres init
├── deployments/helm/llm-fabric/
├── docs/                     Contract, IntentOS, metering, ADRs, configuration
├── examples/python/          Copy-paste clients
├── examples/typescript/
├── sdk/typescript/           myvista TS client
├── src/llm_fabric/           Gateway runtime
│   ├── bench/                Load and stage benches
│   ├── context/              Token budget helpers
│   ├── contract/             OpenAI request/response models
│   ├── deps/                 Dependency health
│   ├── eval/                 Evaluation platform
│   ├── gateway/              HTTP API, middleware, Command Center
│   ├── guardrails/           INPUT/OUTPUT engines
│   ├── heal/                 Drift and remediations
│   ├── identity/             OIDC, API keys, revocation
│   ├── intent/               IntentOS
│   ├── observability/        Metrics, traces, metering
│   ├── router/               Plan, policy, fallback, health
│   ├── serving/              Provider adapters
│   ├── storage/              Postgres / in-memory
│   └── tenancy/              Scope, quota, cache
├── src/myvista/              Python SDK
├── tests/                    unit, contract, security, chaos, system
├── ARCHITECTURE.md
├── PRODUCTION_READINESS.md
├── SECURITY.md
├── Makefile
└── pyproject.toml
```

---

## Development guide

- Constitution: [`docs/constitution.md`](docs/constitution.md) overrides inference.
- Authorship: Anubhav Bhatia. No AI co-author trailers ([`CONTRIBUTING.md`](CONTRIBUTING.md)).
- Style: Ruff (`make lint` / `make format`), mypy on `src`.
- Tests: `make check`. Isolation changes need `make test-isolation`.
- Honesty: do not invent metrics. If it was not measured, do not state it.
- Schemas: OpenAI dialect in `src/llm_fabric/contract/openai.py`; inert fields stay inert (ADR 0002).
- Metrics: closed label sets in `prom.py`. Never put tenant or request id on a Prometheus label.
- IntentOS: default Hashing; do not silently enable MiniLM or serving-path routing.

---

## Adding a provider

1. Implement `Provider` in `src/llm_fabric/serving/base.py` (`generate` + `stream`).
2. Add the adapter under `src/llm_fabric/serving/adapters/`.
3. Register construction in `src/llm_fabric/serving/factory.py` (`_KNOWN_PROVIDERS`).
4. Credentials via `Settings` (prefixed env).
5. Map provider errors to `RetryableError` / terminal errors so fallback reasons work.
6. Streaming must end with exactly one `StreamEnd`.
7. Tools/structured output: only claim what you implement. Today the gateway ignores `tools`.
8. Metering uses provider usage when `usage_reported_by_provider` is true.
9. Tests in `tests/unit/test_adapters.py` (and contract tests if the HTTP shape changes).
10. Document the registry row in `config/models.yaml` (disabled until priced).

---

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md).

1. Read `docs/constitution.md`.
2. `make install` && `make check`.
3. Isolation-sensitive changes: `make test-isolation`.
4. Eval-sensitive changes: `make eval-gate`.
5. Do not edit frozen IntentOS 98 to pass a candidate.
6. Update `ARCHITECTURE.md` only to match built vs proposed; do not paper over gaps.
7. Open a PR against `main`. CI: `.github/workflows/ci.yml` (isolation, intent regression, quality, tests).

There is no CLA or DCO in this repository. License is undecided.

---

## Production checklist

### LLM Fabric core

```text
[ ] Provider secrets in a Secret, not the image or ConfigMap
[ ] TLS at the edge
[ ] Auth: OIDC or API keys; ALLOW_ANONYMOUS false
[ ] Tenant isolation verified (make test-isolation on this tree; RLS on your DB)
[ ] Postgres + Redis configured; migrate Job as table owner; workers as fabric_app
[ ] Health probes: /healthz and /readyz
[ ] Prometheus scrape of /metrics
[ ] OTLP endpoint set; Command Center not used as fleet history
[ ] Logging redaction reviewed (guardrails are not a DLP product)
[ ] Timeouts, max_attempts, breakers reviewed
[ ] Production quotas finite (defaults or explicit)
[ ] Resource requests/limits reviewed; HPA still off until proven
[ ] Rollback of image/config tested
[ ] Isolation suite passed on the commit you ship
```

### IntentOS promotion (currently blocked)

```text
[ ] Frozen hashes verified
[ ] HN gate passed                    ← FAIL (0.50 < 0.58)
[ ] Unknown recall gate passed
[ ] Semantic false-hit gate passed
[ ] Security/isolation tests passed
[ ] Shadow route unchanged
[ ] Serving-path classification still OFF until the HN gate passes
```

---

## FAQ

**What is LLM Fabric?**
An OpenAI-compatible chat gateway with auth, tenancy, policy routing, failover,
metering, and traces.

**Is it only a proxy?**
It is a proxy plus planner, fallback graph, identity, quotas, and observability.
It is not an agent platform.

**Which providers are supported?**
`mock`, `openai` (any OpenAI-compatible HTTP API), `anthropic`.

**Does it support local models?**
Yes, if they speak OpenAI chat completions (e.g. Ollama through
`LLM_FABRIC_OPENAI_BASE_URL`). There is no native vLLM adapter.

**How does routing work?**
Registry + alias policy + health + directed fallback. See
[Model and provider routing](#model-and-provider-routing).

**How does provider fallback work?**
Typed reasons, graph edges, attempt cap, no stream splice after first byte.

**Does LLM Fabric support agents / MCP / A2A / RAG?**
No. SDKs raise `UnsupportedError` for agents and embeddings. Do not infer RAG
from IntentOS embeddings.

**Does it include semantic caching?**
IntentOS L1 semantic **intent** cache, when the cascade is built. Completions
are not semantically cached.

**What is IntentOS?**
An experimental classifier that can inform the planner. It is a subsystem.

**Why is IntentOS currently OFF?**
Hard-negative accuracy 0.50 vs required 0.58 on the frozen 98.

**Does intent classification grant tool access?**
No. `classification != authorization`.

**What port?**
`47317`.

**Can I run it locally?**
Yes. [Quick Start](#quick-start).

**Can I deploy it in production?**
Yes, at the **internal single-VPC** tier described in
`PRODUCTION_READINESS.md`, with IntentOS routing left off.

---

## Deeper documentation

| Doc | Topic |
| --- | --- |
| [`docs/constitution.md`](docs/constitution.md) | Authoritative specification |
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | Built vs not built (older Phase 12 paragraph is superseded by `PRODUCTION_READINESS.md` for the 2026-08-24 verdict) |
| [`PRODUCTION_READINESS.md`](PRODUCTION_READINESS.md) | Production GO / NO-GO |
| [`docs/CONFIGURATION.md`](docs/CONFIGURATION.md) | Environment variables |
| [`docs/CONTRACT.md`](docs/CONTRACT.md) | Honoured API fields |
| [`docs/USAGE_METERING.md`](docs/USAGE_METERING.md) | Usage ledger |
| [`docs/AUTH_REVOCATION.md`](docs/AUTH_REVOCATION.md) | Token revocation |
| [`docs/BACKUP_RECOVERY.md`](docs/BACKUP_RECOVERY.md) | Postgres / Redis recovery |
| [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) | Load numbers and limits |
| [`docs/EVALUATIONS.md`](docs/EVALUATIONS.md) | Eval honesty |
| [`SECURITY.md`](SECURITY.md) | Identity and tenancy controls |
| [`docs/INTENTOS.md`](docs/INTENTOS.md) | IntentOS |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | Patch expectations |

`ARCHITECTURE.md` still describes constitution phases 5–6 (full context compiler
and five-stage guardrails) as largely unbuilt. INPUT/OUTPUT guardrails **are**
on the chat path; retrieval/execution stages are not, because those systems are
not built.
