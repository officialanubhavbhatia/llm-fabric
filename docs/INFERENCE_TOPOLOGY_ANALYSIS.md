# Inference topology analysis

**Status:** Phase 1 inventory, then implementation notes.

## Constitution compatibility

This phase implements the existing constitution. It does not change it.

| Field | Value |
| --- | --- |
| Path | A — implement constitution § Provider architecture (LiteLLM, Ollama, OpenAI-compatible, vLLM-compatible) |
| CONSTITUTION AMENDMENT REQUIRED | **NO** |
| ADR REQUIRED | **YES** — clarification only: [`docs/adr/0005-litellm-is-transport.md`](adr/0005-litellm-is-transport.md) |
| Public SDK | Unchanged (`from myvista import MyVista`) |
| Serving-path IntentOS | Remains **off** |
| Context compiler | Not started |
| Command Center redesign | Not started |

**Constitution:** [`docs/constitution.md`](constitution.md) is authoritative.
This document does not amend it.

---

## Constitution sections

| Topic | Section |
| --- | --- |
| Synchronous path (API → auth → tenant → input guardrails → intent → context → route → adapter → output → response) | Non-negotiable architecture |
| Provider adapters: LiteLLM, Ollama, OpenAI-compatible, vLLM-compatible | Provider architecture |
| Grades are logical classes, not model names | Model grading |
| MyVista route planner owns selection and fallback graph | Routing engine |
| Ollama is preferred local engine; do not fake KV metrics | Ollama |
| vLLM-compatible adapter is required; optimizations need benchmarks | vLLM |
| Local stack must run without Kubernetes; LiteLLM + Ollama named | Local development |
| Telemetry/evals asynchronous; request path stays small | Non-negotiable architecture |

Phase 1 does **not** turn on serving-path IntentOS or the context compiler.
Those remain later implementation of the same path. This phase only makes the
**inference adapter** hop topology-complete.

---

## CURRENT

### Provider interface

`src/llm_fabric/serving/base.py` defines `Provider` (`generate`, `stream`,
`aclose`). The constitution's "ProviderAdapter" is this type. There is no
parallel framework.

Concrete adapters:

| Registry `provider:` | Class | How it is built |
| --- | --- | --- |
| `mock` | `MockProvider` | in-process |
| `openai` | `OpenAIProvider` | OpenAI chat completions |
| `anthropic` | `AnthropicProvider` | Anthropic Messages |
| `ollama`, `ollama-*` | `OpenAIProvider(name=…)` | OpenAI-compatible HTTP |
| `vllm`, `vllm-*` | `OpenAIProvider(name=…)` | OpenAI-compatible HTTP |
| `openai-compatible` | `OpenAIProvider` | OpenAI-compatible HTTP |
| **LiteLLM** | **not a named adapter** | missing |

Direct Ollama and direct vLLM are first-class as **URL + OpenAI-compatible
contract**, not embedded engines. Factory comment: weights and `/metrics` scrape
are out of process.

### Model registry / grades

`ModelSpec` already has `id`, `deployment_id`, `provider`, `provider_model`,
`grade` (Grade00–Grade29), `context_window`, capability vector, placement
(`region`, `hardware`, `locality`), quality/performance slots.

Missing as first-class fields: `provider_adapter`, `transport`, `runtime`,
`api_base`, `health_endpoint`, `metrics_endpoint`. Runtime is not derived today
(good); it is also not declared.

### Route planner / fallback

`router.plan` + `router.engine` own semantic fallback. `FallbackBudget` /
`max_attempts` (default 3) bound hops. httpx clients do **not** retry.
LiteLLM's own `num_retries` is not configured because LiteLLM is not in the
tree.

### Usage / OTEL

Usage events store provider, model, deployment_id, tokens, fallback depth.
They do not store transport/runtime/actual served model. LLM spans use
`gen_ai_system=spec.provider`.

### Compose / Helm / Prometheus / Command Center

| Surface | Current |
| --- | --- |
| Compose profiles | `mock`, `local` (Ollama), `observability`, `platform`, `tools` |
| Helm | ClusterIP gateway; optional in-cluster Ollama; ingress off by default; NetworkPolicy optional |
| Prometheus | scrapes gateway |
| Command Center | live views; KV/batch/engine metrics unavailable (not synthesized) |

Ollama's Compose port `11434` is published for local operator pulls. That is a
**development** exposure, not a production ingress.

---

## DESIRED (Phase 1)

Topologies, public SDK unchanged (`from myvista import MyVista`):

| Id | Path | LiteLLM required? |
| --- | --- | --- |
| A | MyVista → Ollama | no |
| B | MyVista → LiteLLM → Ollama | yes (transport) |
| C | MyVista → vLLM-compatible endpoint | no |
| D | MyVista → LiteLLM → vLLM | yes (transport); preferred GPU when LiteLLM is enabled |
| E | MyVista → LiteLLM → approved external providers | yes (transport) |

MyVista Route Planner selects `deployment_id` / `provider_model`. LiteLLM must
not pick a different model. Persist planner identity plus `actual_served_model`
when the backend reports one.

Retry ownership:

- MyVista: semantic / grade / deployment fallback graph
- LiteLLM/httpx: bounded transport retry, default **0** extra retries
- Reject configs whose product `(1 + fabric_attempts) * (1 + transport_retries)`
  exceeds a hard cap (9)

---

## GAPS

1. No `LiteLLMProvider` / factory name `litellm` (`litellm-*` pools).
2. No deployment topology metadata (`transport`, `runtime`, `api_base`, health/metrics URLs).
3. Errors collapse 429/5xx/connect into `provider_unavailable` / `upstream_error`.
   Missing distinct types: `litellm_unavailable`, `ollama_unavailable`,
   `vllm_unavailable`, `rate_limited`, `runtime_timeout`, `model_unavailable`
   (partially `model_not_found`), `route_exhausted` (today `all_candidates_failed`).
4. No Compose/Helm profiles for LiteLLM→Ollama or LiteLLM→vLLM.
5. No measured A vs B (or C vs D) latency comparison in this repository.
6. Serving-path IntentOS and context compiler remain **off / not on path** —
   out of Phase 1 scope; not a topology gap.

---

## ADR REQUIRED

**YES** — `0005-litellm-is-transport`: LiteLLM is an OpenAI-compatible
**transport** adapter. MyVista remains the only semantic router. Direct Ollama
and direct vLLM stay valid. Does not amend the constitution.

---

## What Phase 1 will not do

- IntentOS serving-path enablement
- Context compiler on the serving path
- Command Center redesign
- Scraping Ollama/vLLM `/metrics` into fabricated KV/batch series
- One Kubernetes Deployment per Grade00–Grade29
- Claiming live GPU numbers without a GPU

## Implementation (after this inventory)

The LiteLLM adapter, topology metadata, Compose/Helm profiles, and typed
errors listed under GAPS are now in the tree. Live GPU/vLLM verification is
still **PENDING** unless a real GPU endpoint is supplied. Performance
comparison A vs B is **NOT VERIFIED** until measured.

