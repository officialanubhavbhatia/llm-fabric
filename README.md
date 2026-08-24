# MyVista LLM Fabric

A control plane for intelligent LLM serving, routing, evaluation,
observability, and guardrails.

[![CI](https://github.com/officialanubhavbhatia/llm-fabric/actions/workflows/ci.yml/badge.svg)](https://github.com/officialanubhavbhatia/llm-fabric/actions/workflows/ci.yml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-3776AB.svg)](pyproject.toml)
[![OpenTelemetry](https://img.shields.io/badge/OpenTelemetry-OTLP-425CC7.svg)](docs/assets/observability-pipeline.svg)
[![Ollama](https://img.shields.io/badge/Ollama-adapter-000000.svg)](docs/PROVIDERS.md)

> **Early access.** IntentOS serving-path routing is **OFF**. A license has not
> been selected — treat the tree as all-rights-reserved until the owner publishes
> [`LICENSE`](LICENSE_DECISION.md). Authorship: **Anubhav Bhatia**.

![MyVista architecture](docs/assets/myvista-overview.svg)

[Quick start](#60-second-quick-start) ·
[Architecture](#architecture) ·
[Command Center](#command-center) ·
[IntentOS](#intentos) ·
[Status](#project-status) ·
[`ARCHITECTURE.md`](ARCHITECTURE.md) ·
[`docs/constitution.md`](docs/constitution.md)

## Contents

1. [What MyVista is](#what-myvista-is)
2. [Command Center](#command-center)
3. [Why MyVista](#why-myvista)
4. [Architecture](#architecture)
5. [60-second quick start](#60-second-quick-start)
6. [IntentOS](#intentos)
7. [Routing](#routing)
8. [Context and cache](#context-and-cache)
9. [Observability](#observability)
10. [Evals](#evals)
11. [Guardrails](#guardrails)
12. [Economics](#economics)
13. [Deployment](#deployment)
14. [Benchmarks](#benchmarks)
15. [SDK](#sdk)
16. [Project status](#project-status)
17. [Roadmap](#roadmap)
18. [Contributing, security, license](#contributing)

---

## What MyVista is

MyVista is an OpenAI-compatible **chat gateway** plus a control plane:

| Built | Not built |
| --- | --- |
| `POST /v1/chat/completions` (buffered + SSE) | Agent runtime, MCP, A2A |
| Identity, tenants, quotas | Public SaaS |
| Policy routing + fallback graph | Intent-aware serving-path routing |
| Durable usage events | Billing-grade exactly-once accounting |
| Command Center + OTEL | KV-cache / batching scrape |
| Eval gates | Context compiler on the serving path |
| INPUT + OUTPUT guardrails on chat | RAG / tool execution |

It is not an agent platform, a RAG system, or a model trainer.

---

## Command Center

The dashboard is served by the gateway at `/command-center`.

![MyVista Command Center overview — local development capture](docs/assets/command-center.png)

This screenshot is a **real capture** of the running UI (tenant `public`,
anonymous development principal, PostgreSQL usage ledger). Numbers are that
process's meter, not a production SLA. Unavailable fields are listed in the UI
instead of being synthesized.

![Command Center views after demo traffic](docs/assets/myvista-demo.gif)

The GIF is a walk through **real** views after API traffic, not a mocked UI.
How to refresh it: [`docs/README_DEMO.md`](docs/README_DEMO.md).

**Live views:** overview, users, tenants, requests, traces, intents, models,
promotion, tiers, routing, fallbacks, tokens, economics, evals, drift,
reliability.

**Present but not backed:** threads, kv_cache, batching, context — the UI says
so. Quality, safety, TPS, and fleet queue depth have no source in this build.

![Command Center observability model](docs/assets/command-center-overview.svg)

---

## Why MyVista

A comparison of **concepts**, not a competitor benchmark.

| Typical LLM proxy | MyVista |
| --- | --- |
| Provider forwarding | Policy-based routing + health-aware fallback |
| Request logs | Traces, usage ledger, Command Center |
| Static fallback list | Directed fallback graph + circuit breakers |
| Prompt-only heuristics | Hierarchical IntentOS (**classify API**; serving path OFF) |
| Token counts in the response | Durable invocation ledger (not billing-grade) |
| Process liveness | Dependency-aware admission (`/readyz`) |
| Unit tests | Eval gates on the change itself |

LiteLLM is **not** a rival in this table. MyVista can call any OpenAI-compatible
URL, including one you run through LiteLLM. There is no native LiteLLM adapter.

---

## Architecture

![MyVista architecture](docs/assets/myvista-overview.svg)

Request path today: **SDK/curl → gateway (auth, quotas, INPUT guardrails) →
route planner → provider → OUTPUT guardrails → usage event → OTEL**.

IntentOS classification is a **separate API**. The context compiler is **not on
the serving path**. Details: [`ARCHITECTURE.md`](ARCHITECTURE.md).

---

## 60-second quick start

Mock provider. No API keys. Docker Compose profile `mock`.

```bash
git clone https://github.com/officialanubhavbhatia/llm-fabric.git
cd llm-fabric

make docker-up
make docker-test
```

Gateway: [http://127.0.0.1:47317](http://127.0.0.1:47317) · Command Center:
[http://127.0.0.1:47317/command-center](http://127.0.0.1:47317/command-center)

```bash
curl -sS http://127.0.0.1:47317/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"auto","messages":[{"role":"user","content":"Say hello in one sentence."}]}'
```

`make docker-up` is the Compose mock stack. It does **not** require a local
`.env`. A local (non-Compose) process does:

```bash
cp .env.example .env
export LLM_FABRIC_ENVIRONMENT=development   # required; unset refuses to start
make dev
```

Ollama on Docker Desktop: `make docker-desktop-ollama`, then
`make ollama-pull` or `make ollama-pull-grades`. Grade00–Grade29 are a **declared
size/family ladder**, not a quality ranking.

---

## IntentOS

Hierarchical classification: exact cache → semantic cache → rules → embedding,
then optional L4, then abstain. **Serving-path routing is OFF** because the
hard-negative gate has not cleared.

![IntentOS cascade](docs/assets/intentos-cascade.svg)

Classify without changing chat routing:

```python
from myvista import MyVista

client = MyVista()  # http://127.0.0.1:47317
result = client.intents.classify("Write a Python function that reverses a list.")
print(result["classification"]["intent_id"], result["classification"]["confidence"])
```

### IntentOS v1 evaluation

Source: [`datasets/eval/intentos/final-2026.08.24.json`](datasets/eval/intentos/final-2026.08.24.json).
Self-authored bootstrap set, **n=98**. A regression tripwire, not production
quality. No comparison against another classifier has been run.

| Metric | Value |
| --- | --- |
| Accuracy | 0.9082 |
| Macro F1 | 0.9269 |
| High-confidence precision (threshold 0.9, n=28, coverage ~0.29) | 0.964 |
| Unknown-intent recall | 0.857 |
| Hard-negative accuracy | **0.50** / target **≥ 0.58** |
| Hard-negative gate | **not cleared** |

Default embedder is HashingEmbedder. MiniLM is opt-in (`--extra embed`). L4 is
experimental and off by default. L5 is not attached by default. Phase B 30-grade
IntentOS routing has **not started**.

Write-up: [`docs/EVALUATIONS.md`](docs/EVALUATIONS.md).

---

## Routing

![Routing and fallback graph](docs/assets/routing.svg)

The planner takes capabilities, tenant policy, optional latency/cost policy, and
provider health, then selects a primary model and may walk a fallback graph.
Each attempt is a usage invocation.

`POST /v1/routes/preview` explains a decision **without** calling a provider.

Grade00–Grade29 exist as **declared classes** in the registry. That is not 30
production-ranked models and not IntentOS-driven grade selection.

---

## Context and cache

| Mechanism | Status |
| --- | --- |
| Context compiler | **Not on the serving path** |
| Response / KV inference cache | **Not built** (Command Center view is unavailable) |
| Continuous batching scrape | **Not built** (vLLM `/metrics` is not scraped) |
| Intent L0/L1 caches | Built on the **classify** path; off on serving-path routing |

Do not read "KV cache" or "context" in the nav as live telemetry.

---

## Observability

![Observability pipeline](docs/assets/observability-pipeline.svg)

Built spans: `request`, `auth`, `input_guardrails`, `intent`, `route`, `llm`,
`output_guardrails`. Unbuilt as serving spans: `context`, `retrieval`, `tool`,
`eval`.

The Command Center trace journal is **per process** and lost on restart. Fleet
history belongs in an OTLP backend (`LLM_FABRIC_OTEL_EXPORTER_OTLP_ENDPOINT`).
Prometheus cardinality is bounded. Secrets and raw private prompts are not
stored by default.

---

## Evals

Evals are first-class. They do not mean every AI behaviour is solved.

![Eval loop](docs/assets/eval-loop.svg)

`llm-fabric-eval` runs suites, comparisons, and gates. The named SDK suite is
`ci`. DeepEval and lm-evaluation-harness are optional adapters and report
unavailable when the extra is missing. Agent and safety suites are **not**
built. IntentOS numbers remain the bootstrap tripwire above.

---

## Guardrails

![Five-stage guardrails](docs/assets/guardrails.svg)

Stages exist as types: INPUT → RETRIEVAL → CONTEXT → EXECUTION → OUTPUT.
**INPUT and OUTPUT are wired on chat completions** (size, secret/PII-shaped
redact, injection markers). RETRIEVAL, CONTEXT, and EXECUTION are **pluggable**
— there is no RAG or tool runtime on the serving path to bind them to.

---

## Economics

![Usage metering](docs/assets/usage-metering.svg)

Provider invocation → `UsageEvent` → PostgreSQL `usage_events` → Redis
best-effort counters → Command Center / Economics.

Counted: request, each invocation (including fallbacks), prompt/completion
tokens, token provenance, estimated vs measured cost (registry price × tokens).

This is **durable usage accounting**, not billing-grade exactly-once accounting.
If the process dies after the provider returns and before INSERT, that
invocation is lost. See [`docs/USAGE_METERING.md`](docs/USAGE_METERING.md).

![Dependency-aware recovery](docs/assets/dependency-health.svg)

When PostgreSQL or Redis is a required serving dependency and is down:
`/healthz` stays 200, `/readyz` is 503, **new** inference is 503. In-flight
provider calls are not cancelled. This is **dependency-aware recovery**, not
fully autonomous self-healing.

---

## Deployment

| Deployment | Status |
| --- | --- |
| Local Mac + Ollama | Supported (`make dev-ollama` / `make docker-desktop-ollama`) |
| Docker Compose mock | Supported (`make docker-up`) |
| Kubernetes + Helm | Verified internal tier (kind + chart; HPA off) |
| HPA autoscaling | Disabled / not yet verified |
| Multi-region | Roadmap |
| AKS / EKS / GKE examples | Helm-rendered, **not** live-tested |

Production path that **has** been reviewed: controlled internal single-VPC
Kubernetes, managed PostgreSQL + Redis, API-key or OIDC, autoscaling **off**.
Verdict and limits: [`PRODUCTION_READINESS.md`](PRODUCTION_READINESS.md).

---

## Benchmarks

**Gateway benchmark — NOT inference throughput.** Mock provider, Compose
production-like stack, 2026-08-24.

| | |
| --- | --- |
| Result | **237 req/s**, 1920 requests, 0 errors |
| Latency | p50 132.34 ms · p95 168.15 ms · p99 221.26 ms |
| Artifact | [`artifacts/audit-2026-08-24/load-chat-short.json`](artifacts/audit-2026-08-24/load-chat-short.json) |
| Command | `uv run llm-fabric-load --host 127.0.0.1 --port 47317 --workload chat-short --duration 8 --warmup 2 --connections 32 --processes 2` |

Historical 1,000 req/s and 2,377 req/s figures are **not** this release's claim.
Ollama token throughput is not a currently reproduced product SLA in
[`docs/BENCHMARKS.md`](docs/BENCHMARKS.md).

---

## SDK

In-tree package `myvista` (installed by `uv sync`). Default base URL is `http://127.0.0.1:47317`.

```python
from myvista import MyVista

client = MyVista()

chat = client.chat.completions.create(
    model="auto",
    messages=[{"role": "user", "content": "Say hello in one sentence."}],
)
print(chat.text, chat.fabric.served_model, chat.usage)

classified = client.intents.classify("Write a Python function that reverses a list.")
print(classified["classification"]["intent_id"])

traces = client.traces.list()
print(traces.keys())
```

Embeddings and agents raise `UnsupportedError`. TypeScript client:
`sdk/typescript`. Contract: [`docs/CONTRACT.md`](docs/CONTRACT.md).

---

## Project status

### Early access

### Verified

- Controlled internal single-VPC Kubernetes (see production-readiness doc)
- PostgreSQL usage ledger and Redis quotas when configured
- OAuth2/OIDC and API keys (production refuses to start without auth)
- Tenant isolation tests
- Durable usage events
- Dependency-aware readiness

### Not claimed

- Public SaaS readiness
- Multi-region HA
- Billing-grade accounting
- Unattended autoscaling
- 1,000 RPS real-model SLA
- Solved prompt injection on every flow
- IntentOS as a production router

---

## Roadmap

Planned work, not a schedule:

- IntentOS hard-negative gate, then serving-path classification
- Phase B 30-grade IntentOS planner
- Context compiler on the serving path
- Verified HPA
- License selection by the owner

---

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md). `make check` is the local gate.

### Security

Report privately via GitHub security advisories. See [`SECURITY.md`](SECURITY.md).

### License

None selected. All rights reserved until the owner publishes `LICENSE`.
[`LICENSE_DECISION.md`](LICENSE_DECISION.md).

### Deeper docs

[`ARCHITECTURE.md`](ARCHITECTURE.md) ·
[`docs/constitution.md`](docs/constitution.md) ·
[`docs/CONFIGURATION.md`](docs/CONFIGURATION.md) ·
[`docs/PROVIDERS.md`](docs/PROVIDERS.md) ·
[`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) ·
[`docs/EVALUATIONS.md`](docs/EVALUATIONS.md) ·
[`docs/deployment/README.md`](docs/deployment/README.md) ·
[`docs/README_DEMO.md`](docs/README_DEMO.md)
