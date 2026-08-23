# Public contract

The fabric speaks the OpenAI chat-completions dialect. This page records exactly
which fields are **honoured** and which are **accepted but inert**, so behaviour
is never inferred from the fact that a request succeeded.

Inert fields are accepted rather than rejected because SDKs send them and refusing
a request the fabric could serve is worse than ignoring a field. See
[ADR 0002](adr/0002-openai-compatible-contract.md).

## `POST /v1/chat/completions`

### Request

| Field | Status | Notes |
| --- | --- | --- |
| `model` | Honoured | A model id or an alias such as `auto`. |
| `messages` | Honoured | At least one required. `system`, `user`, `assistant`, `tool` roles. |
| `stream` | Honoured | SSE when true. |
| `temperature` | Honoured | 0–2. Passed to the backend. |
| `top_p` | Honoured | 0–1 exclusive of 0. Passed to the backend. |
| `max_tokens` | Honoured | Passed to the backend. Anthropic requires it; 1024 is sent when omitted. |
| `stop` | Honoured | String or list. Sent as `stop_sequences` to Anthropic. |
| `n` | **Inert** | Accepted only as `1`. Multiple choices are not implemented. |
| `user` | **Inert** | Accepted and ignored. Not used for routing or attribution. |

Fields not listed are rejected by schema validation.

### Response

Standard OpenAI shape. `model` is the **model that actually served the request**,
not the id that was sent — so a request for `auto` comes back naming a concrete
model. `usage` always contains self-consistent totals.

### Provenance headers

Present on every buffered response:

| Header | Meaning |
| --- | --- |
| `x-fabric-request-id` | Correlation id. Echoes a caller-supplied `x-request-id`. |
| `x-fabric-requested-model` | What the caller asked for. |
| `x-fabric-served-model` | What actually served it. |
| `x-fabric-provider` | The backend behind that model. |
| `x-fabric-policy` | The policy that chose it. |
| `x-fabric-failovers` | How many candidates failed first. |

### Streaming

SSE frames, terminated by `data: [DONE]`.

- The first content chunk carries `delta.role = "assistant"`; later chunks carry
  only `delta.content`.
- All chunks in one response share an `id`.
- The final chunk carries `finish_reason`, a `usage` block, and an `x_fabric`
  block with the same provenance as the headers above.
- A mid-stream failure arrives as a frame containing an `error` object, followed
  by `[DONE]`. The HTTP status is already committed by then, so it stays 200 — see
  [ADR 0003](adr/0003-no-failover-after-first-streamed-byte.md).
- An unknown or disabled model is rejected **before** streaming starts, as a
  normal HTTP error.

## Errors

One envelope everywhere, including schema validation failures:

```json
{ "error": { "message": "unknown model 'x'", "type": "model_not_found", "request_id": null } }
```

| `type` | Status |
| --- | --- |
| `invalid_request_error` | 400 or 422 |
| `model_not_found` | 400 |
| `authentication_error` | 401 |
| `no_candidate` | 503 |
| `all_candidates_failed` | 502 |
| `provider_timeout`, `provider_unavailable`, `upstream_error` | 502 |
| `configuration_error` | 500 |

Validation failures add a `details` array of `{location, message, type}`.

## Authentication

`Authorization: Bearer <key>` or `x-api-key: <key>`, compared in constant time.
When `LLM_FABRIC_API_KEYS` is empty the gateway runs open and logs a warning at
startup. `/healthz` and `/readyz` never require a key.

Keys are never logged. Metering records a truncated SHA-256 fingerprint instead.

## Other endpoints

| Endpoint | Notes |
| --- | --- |
| `GET /v1/models` | Enabled models plus aliases. Aliases are `owned_by: llm-fabric`. Disabled models are not advertised. |
| `GET /v1/models/{id}` | 400 for an unknown id. |
| `GET /v1/usage` | Totals and recent routing decisions. **In-memory, this process only, lost on restart** — stated in the `scope` field of every response. `cost_is_estimated` marks records whose token counts came from the fabric's heuristic rather than the backend. |
| `GET /healthz` | Liveness. |
| `GET /readyz` | Readiness. 503 when no model is enabled. |
| `GET /docs` | Generated OpenAPI reference. |
