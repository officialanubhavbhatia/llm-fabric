# Providers

LLM Fabric talks to inference backends through **adapters**. Adding a model does
not require a new adapter when the backend already speaks a supported contract.

## Compatibility matrix

Statuses mean:

| Word | Meaning |
| --- | --- |
| **yes** | Exercised by the default unit/contract tests against a mock HTTP transport or the in-process mock provider. |
| **same contract** | Uses the OpenAI-compatible adapter. Live Ollama/vLLM processes are **not** started in CI. |
| **not executed** | Accepted on the wire or declared on a registry entry; the fabric does not run tools, validate JSON schema, or serve embeddings. |
| **not built** | No serving path. |

| Feature | mock | Ollama | vLLM | LiteLLM transport | Generic OpenAI-compatible | Anthropic |
| --- | --- | --- | --- | --- | --- | --- |
| Chat | yes | same contract | same contract | same contract | same contract | yes |
| Streaming | yes | same contract | same contract | same contract | same contract | yes |
| JSON / structured output | not executed | not executed | not executed | not executed | not executed | not executed |
| Tool calling | not executed | not executed | not executed | not executed | not executed | not executed |
| Embeddings HTTP | not built | not built | not built | not built | not built | not built |
| Vision | model-dependent, not executed | model-dependent | model-dependent | runtime-dependent | provider-dependent | model-dependent, not executed |
| Reasoning | declared capability only | model-dependent | model-dependent | runtime-dependent | model-dependent | declared capability only |
| Long context | registry `context_window` | model-dependent | model-dependent | runtime-dependent | provider-dependent | provider-dependent |
| Engine `/metrics` (KV, batching) | n/a | `/api/ps` loaded models only; KV UNAVAILABLE | scraped when `metrics_endpoint` is set | transport `/metrics` only; not vLLM KV | n/a | n/a |

LLM Fabric is designed to support a broad range of open-source models through
Ollama, vLLM, Hugging Face **identities in the registry**, and OpenAI-compatible
inference APIs. That is a compatibility strategy, not a claim that every
open-source model has been tested.

A model that the configured Ollama or vLLM runtime can serve can be registered
with capability metadata and become a routing candidate after compatibility and
evaluation checks. There is no fixed model whitelist in application logic.

## How each backend is reached

| Name in registry `provider:` | Adapter | Credentials |
| --- | --- | --- |
| `mock` | In-process. Returns text assembled from the request. | none |
| `openai` | OpenAI chat completions | `OPENAI_API_KEY` / `LLM_FABRIC_OPENAI_API_KEY` |
| `anthropic` | Anthropic Messages | `ANTHROPIC_API_KEY` / `LLM_FABRIC_ANTHROPIC_API_KEY` |
| `ollama`, `ollama-*` | OpenAI-compatible HTTP | none required; optional `LLM_FABRIC_OLLAMA_API_KEY` |
| `vllm`, `vllm-*` | OpenAI-compatible HTTP | none required; optional `LLM_FABRIC_VLLM_API_KEY` |
| `litellm`, `litellm-*` | OpenAI-compatible HTTP to LiteLLM | none required; optional `LLM_FABRIC_LITELLM_API_KEY` |
| `openai-compatible` | OpenAI-compatible HTTP | uses `LLM_FABRIC_OPENAI_BASE_URL`; key optional |

LiteLLM is a **transport**, not a second route planner. Registry fields
`transport: litellm` and `runtime: ollama|vllm|external` are declared; they are
never inferred from the model id. The Route Planner selects `provider_model`
(the LiteLLM `model_name`). LiteLLM `num_retries` defaults to 0; MyVista owns
semantic fallback. See [`adr/0005-litellm-is-transport.md`](adr/0005-litellm-is-transport.md).

Do not mix `provider: ollama` with `transport: litellm`. Use `provider: litellm`
and `runtime: ollama`.

Multiple vLLM **pools** are different provider names (`vllm-coding`,
`vllm-reasoning`, …) with URLs in `LLM_FABRIC_PROVIDER_BASE_URLS`:

```json
{"vllm-coding":"http://vllm-coding:8000/v1","vllm-reasoning":"http://vllm-reasoning:8000/v1"}
```

The fabric does not embed the vLLM Python engine and does not scrape vLLM or
Ollama process metrics. Chat completions can still be routed to those servers.

### Fabric and vLLM responsibilities

Fabric owns registry identity, tenant/policy enforcement, L0–L30 eligibility,
promotion state, bounded typed fallback, externally observed health/latency and
telemetry. vLLM owns weight loading, GPU inference, generation, batching and
its internal scheduler. They are separate processes/services joined by the
OpenAI-compatible HTTP contract.

Fabric observes availability, request latency, errors, timeouts, fallback and
breaker state. TTFT and output tokens/second are recorded only when the API
response/stream makes them measurable. Fabric does not parse private vLLM
metrics or treat API token price `0.0` as zero compute cost.

After starting a real LiteLLM proxy (Ollama behind it):

```bash
export LLM_FABRIC_LITELLM_BASE_URL=http://127.0.0.1:4000/v1
export LLM_FABRIC_LIVE_LITELLM_MODEL=smollm2-135m
make test-litellm-live
```

Skipped tests are not passing benchmarks.

### Optional live profile

After starting a real vLLM endpoint:

```bash
export LLM_FABRIC_VLLM_BASE_URL=http://127.0.0.1:8000/v1
export LLM_FABRIC_LIVE_VLLM_MODEL=<served-model-id>
make test-vllm-live
```

The profile checks discovery, chat normalization, streaming, Fabric routing
headers, selected model/provider/tier, typed timeout fallback, context
rejection, disabled deployment, tenant ceiling and breaker filtering. It
rejects a Fabric mock endpoint on port 8000 as “not vLLM”.

## Hugging Face identities

`huggingface_id`, `revision`, `digest`, and `license` on a registry entry are
**metadata** for operators and for vLLM serve commands. The gateway does not
download weights from Hugging Face.

`trust_remote_code: true` is refused at registry load.

## Adding a provider

1. Prefer an OpenAI-compatible URL and a new `provider:` name (`vllm-coding`).
2. Add a dedicated adapter only when the contract cannot express a capability
   safely.
3. Point the factory at that name; do not scatter model ids through application
   code.

## Adding a model

1. Add an entry to `config/models.yaml`, `config/models.local.yaml`, or the
   local Ollama grade ladder `config/models.ollama-grades.yaml`.
2. Declare `provider`, `provider_model` (the tag the backend expects), `grade`
   or `tiers`, and `capabilities`.
3. Pin `revision` / `digest` in production when the backend supports it.
4. Run `llm-fabric model probe`, `model evaluate`, shadow simulation, then
   promotion. `enabled: true` and a high declared tier do not constitute
   production approval.
