# Topology (reference)

LLM Fabric is a routing gateway. Inference engines stay separate processes
and separate images. This document describes topologies the repository's
configuration can express. AKS, EKS, and GKE layouts are Helm-rendered
examples, not live cloud tests.

```text
Ingress / load balancer / NodePort
        |
        v
LLM Fabric replicas  (one OCI image, Helm chart deployments/helm/llm-fabric)
        |
        +--> Ollama                (local / kind optional sidecar)
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

Local Ollama is an optional separate workload (`make docker-ollama` or Helm
`ollama.enabled`). Model weights live in that workload's volume, never in the
Fabric image. A reachable Ollama or vLLM Service does not promote a model;
lifecycle stays evidence-bound.

Not implied by this topology:

- A Fabric-owned GPU operator
- Scraping vLLM `/metrics` (KV cache, batching) — **not built**
- A response cache, Agents, MCP, or embeddings HTTP — **not built**
- IntentOS serving-path classification — **OFF** by default
- Live validation of AKS, EKS, or GKE

vLLM live inference is only measured when a vLLM process is actually running.
Skipped integrations are not passing benchmarks.

Example Fabric configuration:

```bash
export LLM_FABRIC_VLLM_BASE_URL=http://127.0.0.1:8000/v1
export LLM_FABRIC_PROVIDER_BASE_URLS='{"vllm-coding":"http://vllm-coding:8000/v1"}'
```

Pool names in the registry (`provider: vllm-coding`) must match keys in
`LLM_FABRIC_PROVIDER_BASE_URLS`.
