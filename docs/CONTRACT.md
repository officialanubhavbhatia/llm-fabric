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
| `model` | Honoured | A model id, an alias such as `auto`, or a public tier such as `L12`. |
| `messages` | Honoured | At least one required. `system`, `user`, `assistant`, `tool` roles. |
| `stream` | Honoured | SSE when true. |
| `temperature` | Honoured | 0–2. Passed to the backend. |
| `top_p` | Honoured | 0–1 exclusive of 0. Passed to the backend. |
| `max_tokens` | Honoured | Passed to the backend. Anthropic requires it; 1024 is sent when omitted. |
| `stop` | Honoured | String or list. Sent as `stop_sequences` to Anthropic. |
| `n` | **Inert** | Accepted only as `1`. Multiple choices are not implemented. |
| `user` | **Inert** | Accepted and ignored. Not used for routing or attribution. |

Fields not listed (including `stream_options`, `frequency_penalty`, `tools`, and
other SDK keys) are **ignored**, not rejected. Refusing them would break clients
the dialect exists to serve. See [ADR 0002](adr/0002-openai-compatible-contract.md).

### Response

Standard OpenAI shape. `model` is the **model that actually served the request**,
not the id that was sent — so a request for `auto` comes back naming a concrete
model. `usage` always contains self-consistent totals for the **final visible
model call**. Fallback and other provider invocations are not folded into that
object; they live on the usage ledger (`/v1/usage` `invocations`,
`x-fabric-invocations`). See [USAGE_METERING.md](USAGE_METERING.md).

### Provenance headers

Present on every buffered response:

| Header | Meaning |
| --- | --- |
| `x-fabric-request-id` | Correlation id. Echoes a caller-supplied `x-request-id`. |
| `x-fabric-requested-model` | What the caller asked for. |
| `x-fabric-served-model` | What actually served it. |
| `x-fabric-selected-tier` | Public service tier of the served deployment (`L12`), when declared. |
| `x-fabric-policy` | The policy that chose it. |
| `x-fabric-failovers` | How many candidates failed first. |
| `x-fabric-invocations` | Provider attempts recorded for this request (includes fallbacks). |
| `x-fabric-fallback-depth` | How far down the fallback graph the served model sat. Omitted when it is zero. |
| `x-fabric-intent` | The classified intent id. Only present when intent classification is enabled. |
| `x-fabric-intent-confidence` | Confidence in that classification. **Uncalibrated** — see [`docs/EVALUATIONS.md`](EVALUATIONS.md). |

For the full decision behind a response, including the candidates that were
excluded and why, use the route preview endpoint below.

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
| `permission_error` | 403 |
| `not_found` | 404 |
| `quota_exceeded` | 429 |
| `no_candidate` | 503 |
| `all_candidates_failed` | 502 |
| `provider_timeout`, `provider_unavailable`, `upstream_error` | 502 |
| `configuration_error` | 500 |

Validation failures add a `details` array of `{location, message, type}`.

A 401 carries `WWW-Authenticate`. A 429 carries `Retry-After` in seconds.

**403 and 404 are not interchangeable.** A resource belonging to another tenant
is reported as **404**, because a 403 would confirm the identifier exists
somewhere and turn the endpoint into an enumeration oracle. 403 means the caller
is known and the resource is theirs, but their token lacks the required scope or
role.

## Authentication

Send `Authorization: Bearer <token>`, or `x-api-key: <key>` for static
credentials. Authentication runs before routing, so an unauthenticated request
receives 401 for any path — including one that does not exist. `/healthz`,
`/readyz`, `/docs` and `/openapi.json` never require a credential.

Four modes, selected by `LLM_FABRIC_AUTH_MODE` or inferred from whichever
credentials are configured: `oidc`, `api_key`, `dev`, `disabled`. With
`disabled`, the gateway runs open under an anonymous principal holding no scopes
or roles, and warns at startup.

See [`SECURITY.md`](../SECURITY.md) for what is enforced on a token.

### Tenancy

Every request is bound to the tenant in its validated credential. `GET /v1/usage`
returns only that tenant's records, and reports its `tenant_id`.

`X-Tenant-Id` requests action on behalf of another tenant. It is honoured **only**
when the token carries the `fabric:delegate_tenant` scope; any other caller
sending it is refused with 401. A client-supplied tenant id is never trusted on
its own.

### Quotas

Per-tenant and per-user ceilings apply to requests per minute, requests per day,
tokens per day and spend per day. Exhausting one returns 429 with `Retry-After`.
Current consumption and configured limits are reported under `quota` in
`GET /v1/usage`.

Limits are enforced per process, so a multi-replica deployment multiplies them.

### Correlation

Every response carries `x-fabric-request-id` and `traceparent`. Send
`x-request-id` to supply your own correlation id, or `traceparent` (W3C) to
continue an existing trace; a malformed `traceparent` is ignored and a new trace
begins. Tenant identifiers are attached to internal telemetry but never to the
outbound `traceparent`.

Credentials are never logged. Static keys appear only as a truncated SHA-256
fingerprint, and never in a metering record.

## `POST /v1/routes/preview`

Asks where a request **would** go. The same planner that serves traffic answers,
and returns the same decision object, so an explanation does not require paying
for inference first. Nothing is sent to a provider, nothing is metered, and
nothing is billed.

### Request

`model` is required and may be an id or an alias. Everything else is optional:

| Field | Meaning |
| --- | --- |
| `messages` | Measured for a token count only. Never sent anywhere. |
| `prompt_tokens` | Use a known count instead of measuring `messages`. |
| `max_tokens` | Reserved output, which counts against a deployment's context window. |
| `policy` | Override the policy for this preview. A tenant's pinned policy still wins. |
| `required_capabilities` | Narrow to deployments declaring all of them. |
| `minimum_grade` | For example `Grade12`. Accepts `Grade07`, `grade07`, `7` or `07`. |
| `latency_slo_ms`, `budget_usd` | Exclude deployments whose declared figures breach them. |

### Response

The `RoutePlan`. The fields that matter most:

| Field | Meaning |
| --- | --- |
| `selected` | The full deployment record that would serve, or `null` if nothing can. |
| `routing_score` | The winner's score in [0, 1] under the resolved policy. |
| `policy`, `requested_policy` | What was used, and what was asked for. They differ when a tenant pins a policy or a legacy name is normalised. |
| `chain` | The ranked candidates, in the order the engine would try them. |
| `scoring` | Per candidate, every feature's raw value, `source`, weight and contribution — the arithmetic behind the score. |
| `excluded` | Every removed candidate with a typed reason and detail. |
| `fallback` | The reason-labelled fallback graph and the budget bounding it. |
| `inputs` | Each planner input, whether it was available, and if not, why. |
| `tenant_policy` | The **calling** tenant's routing policy, or `null`. |
| `explanation` | The same decision in prose. |
| `expected` | Estimated quality and latency, explicitly labelled estimates. |

`source` on a feature is one of:

- `declared` — from the registry: what the operator asserted, not a measurement;
- `observed` — measured by this process from attempts it actually made;
- `absent` — no value, which is **not** zero.

**A feature missing for any eligible candidate is dropped for the whole
decision**, and `scoring.dropped_features` says why. Scoring an unpriced model as
free would let missing data win a route. When every feature drops,
`fell_back_to_registry_order` is `true`.

Two inputs are permanently unavailable in this build and report as such:
`historical_quality` needs a routing eval suite, and `cache_probability` needs a
response cache. Neither exists.

### Isolation

The tenant is taken from the token, never from the body: there is no field in
which to ask to preview as someone else. The response shows only the calling
tenant's policy, and no other tenant's traffic, identity or configuration
appears. A tenant policy narrows and can never be widened — an override in the
body cannot escape a pinned policy, a deny list or a locality restriction.

`400` for an unknown model, an unparseable policy or an out-of-range grade. A
request nothing can serve is **not** an error: it returns `200` with
`selected: null` and the exclusions explaining why.

## `GET /v1/routes/health`

Circuit state and observed rates per deployment: EWMA latency and error rate,
success and failure counts, queue depth, and whether the breaker is closed, open
or half-open.

Measured by the answering process from attempts **it** made, so figures differ
between replicas and are lost on restart. A deployment with no samples is
absent from the list rather than reported as healthy.

Deliberately contains no per-tenant figures, because reporting health per tenant
would leak one tenant's traffic volume to another.

## Other endpoints

| Endpoint | Notes |
| --- | --- |
| `GET /v1/models` | Enabled models plus aliases. Aliases are `owned_by: llm-fabric`. Disabled models are not advertised. |
| `GET /v1/models/{id}` | 400 for an unknown id. |
| `GET /v1/usage` | Totals, quota consumption and recent routing decisions **for the calling tenant**. The `scope` field names the ledger: PostgreSQL `usage_events` when configured, otherwise in-memory. `cost_is_estimated` marks records whose token counts came from the fabric's heuristic rather than the backend. |
| `GET /healthz` | Liveness. The process is alive. Dependency outages do not fail this endpoint. |
| `GET /readyz` | Readiness. 503 when a mandatory serving dependency (PostgreSQL, Redis when wired) is unhealthy, or when no enabled model has a constructible provider. OTEL is optional. Diagnostics are bounded: no DSNs or secrets. |
| `GET /docs` | Generated OpenAPI reference. |
| `POST /v1/intents/classify` | Offline cascade. Does not enable classification on the serving path. |
| `POST /v1/evals/run` | Named suite only (`ci`). Does not accept a filesystem path. |
| `GET /v1/observability/traces` | Recent traces for the calling tenant. **Local-pod diagnostic only — not authoritative fleet trace history.** Per process, lost on restart. Fleet traces belong in the OTLP backend. |
| `GET /v1/observability/traces/{trace_id}` | One trace. **404** if it is missing *or belongs to another tenant*. |
| `POST /v1/dev/token` | **Development mode only.** Mints a token for any identity requested. Not mounted unless `auth_mode` is `dev`. |
