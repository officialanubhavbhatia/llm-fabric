# ADR 0007 — Context compiler on the serving path; honest inference metrics

**Status:** Accepted  
**Date:** 2026-08-25  
**Constitution:** does not amend [`docs/constitution.md`](../constitution.md)

## Context

The constitution already requires a context compiler, typed blocks, token
accounting, and honest engine metrics. Chat compiled nothing and Command Center
views for `context` / `kv_cache` returned `available: false`. Ollama KV and
vLLM `/metrics` were easy to invent. LiteLLM is transport, not an engine.

## Decision

1. Chat always compiles before `Router.complete` / `stream`. Every
   `USER_RESPONSE` usage event stores `context_record_id`.
2. Every observation carries provenance and scope. Unknown is `UNAVAILABLE`
   with no value. A configured source that goes silent is
   `METRIC_PIPELINE_BROKEN` with no value. Zero is never used for unknown.
3. vLLM `/metrics` is scraped off the request path when
   `metrics_endpoint` is set. Parser names are the documented V1 series and
   legacy V0 aliases. Histogram means are DEPLOYMENT-scoped.
4. Ollama contributes native eval counts/durations when present, else OpenAI
   usage tokens. KV, prefix cache, batch utilisation and queue depth stay
   `UNAVAILABLE — OLLAMA DOES NOT EXPOSE THIS METRIC`.
5. LiteLLM spans and transport histograms are transport-only. They do not
   relabel vLLM KV or prefix cache.
6. GPU telemetry is DCGM exporter via Prometheus, not a custom Fabric monitor.
7. `stable_prefix_tokens` is a fabric prompt-shape label, not a KV hit.

## Consequences

- Public SDK chat/completions shape is unchanged. Provenance headers gain
  `x-fabric-context-record-id`.
- Command Center `context` and `kv_cache` views are available with honest
  UNAVAILABLE / DEPLOYMENT data.
- Live vLLM KV/TTFT/queue numbers appear only after a successful scrape of a
  real engine. Compose does not start vLLM.
