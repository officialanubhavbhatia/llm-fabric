# Production topology (reference)

LLM Fabric is a routing gateway. vLLM remains a separate inference process.
This document describes a topology the repository's configuration can express.
It is not a claim that a cluster was stood up in CI.

```text
Ingress / load balancer
        |
        v
LLM Fabric replicas  (Helm chart deployments/helm/llm-fabric)
        |
        +--> vLLM general pool     (OpenAI-compatible :8000/v1)
        +--> vLLM coding pool
        +--> vLLM reasoning pool
        +--> vLLM long-context pool
        +--> OpenAI / Anthropic    (optional, credentials in a Secret)
```

Shared pieces that **are** implemented:

- ConfigMap + Secret for `LLM_FABRIC_*` (see `examples/helm/vllm-pools-values.yaml`)
- Prometheus scrape of Fabric `/metrics` (Fabric-observed latency, HTTP status, fallback)
- Optional OTLP traces to a collector (`LLM_FABRIC_OTEL_EXPORTER_OTLP_ENDPOINT`)
- PostgreSQL usage ledger and Redis quotas when those URLs are set (production requires them)

Not implied by this topology:

- A Fabric-owned GPU operator
- Scraping vLLM `/metrics` (KV cache, batching) — **not built**
- A response cache, Agents, MCP, or embeddings HTTP — **not built**
- IntentOS serving-path classification — **OFF** by default

vLLM live inference is only measured when a vLLM process is actually running.
Skipped integrations are not passing benchmarks.

Example Fabric configuration:

```bash
export LLM_FABRIC_VLLM_BASE_URL=http://127.0.0.1:8000/v1
export LLM_FABRIC_PROVIDER_BASE_URLS='{"vllm-coding":"http://vllm-coding:8000/v1"}'
```

Pool names in the registry (`provider: vllm-coding`) must match keys in
`LLM_FABRIC_PROVIDER_BASE_URLS`.
