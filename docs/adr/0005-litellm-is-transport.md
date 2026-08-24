# ADR 0005 — LiteLLM is a transport adapter

**Status:** Accepted  
**Date:** 2026-08-25  
**Constitution:** does not amend [`docs/constitution.md`](../constitution.md)

## Context

The constitution lists LiteLLM, Ollama, OpenAI-compatible inference, and
vLLM-compatible inference as initial provider adapters. The request path's
route planner is a MyVista component. LiteLLM can also route. Those two
authorities must not both choose a model on one request.

The existing `Provider` interface already speaks OpenAI-compatible HTTP. A
second provider framework is unnecessary.

## Decision

1. Add `LiteLLMProvider` as a `Provider` implementation over the OpenAI-compatible
   chat-completions contract (the same contract Ollama and vLLM already use).
2. Declare topology on `ModelSpec`: `provider_adapter`, `transport` (`direct` |
   `litellm`), `runtime` (`ollama` | `vllm` | `external` | `mock`). Runtime is
   **not** inferred from the model id string.
3. The Route Planner selects `deployment_id` and `provider_model`. The LiteLLM
   adapter sends that `provider_model` as the LiteLLM model name. LiteLLM must
   not load-balance onto a different logical model unless the operator documents
   an equivalent deployment group.
4. Direct Ollama (`transport: direct`, `runtime: ollama`) and direct vLLM
   (`transport: direct`, `runtime: vllm`) remain first-class. LiteLLM→vLLM is
   the preferred **production GPU** topology when LiteLLM is enabled; it is not
   the only valid GPU topology.
5. MyVista owns semantic fallback. LiteLLM/httpx transport retries default to
   zero. Configurations that multiply retries past a hard cap are refused.

## Consequences

- Public SDK (`myvista.MyVista`) is unchanged.
- Serving-path IntentOS and the context compiler are not enabled by this ADR.
- Engine `/metrics` remain unsynthesized.
