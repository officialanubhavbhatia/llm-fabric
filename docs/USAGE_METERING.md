# Usage metering

Authoritative usage is an **idempotent durable event** per provider
invocation, stored in PostgreSQL `usage_events`. Worker identity does not
affect totals. Redis holds optional fast counters and is not the source of
truth.

```
provider invocation
      → UsageEvent (event_id = invocation_id)
      → PostgreSQL usage_events
      → optional Redis fabric:usage:v1:* hashes
      → /v1/usage, Command Center, llm-fabric-usage reconcile
```

## Three grains

| Grain | What it counts | Where |
| --- | --- | --- |
| Invocation | One actual model/provider call | `usage_events` row |
| Request | OpenAI-compatible `usage` on the visible response (final model) | derived / `UsageRecord` |
| Rollup | tenant/day tokens, requests, invocations | Redis cache; recompute from ledger |

A fallback that times out on model A and succeeds on model B is **two**
invocation events. The JSON `usage` object still reports model B only.
`x-fabric-invocations` and `/v1/usage` `invocations` cover both.

## Token source

Every invocation stores `token_source`:

| Value | Meaning |
| --- | --- |
| `PROVIDER_MEASURED` | Backend reported token fields |
| `LOCAL_TOKENIZER_ESTIMATE` | Fabric heuristic (`approximate_token_count`) |
| `DERIVED` | Computed from other measured fields |
| `UNAVAILABLE` | No reliable count (failed before tokens, missing metadata) |

Estimates are never labelled measured.

## Internal model calls

The schema tags `operation` as `USER_RESPONSE`, `INTENT_CLASSIFIER`,
`GUARDRAIL`, `EVALUATOR`, `ROUTER`, `REPAIR`, `AGENT`, or `OTHER_INTERNAL`.

The serving path currently records **user-response provider attempts only**.
IntentOS on the chat path is lexical/embedding, not an LLM invocation, so it
does not create a usage event. An LLM judge or L4/L5 classifier that later
calls `Provider.generate` must emit an event with the matching operation. Do
not invent calls to populate the ledger.

## Streaming

| Case | What is stored |
| --- | --- |
| Successful stream with a final usage frame | Those counts; `PROVIDER_MEASURED` if the backend reported them, otherwise `LOCAL_TOKENIZER_ESTIMATE` |
| Client disconnect | Tokens from bytes already emitted, estimated; `LOCAL_TOKENIZER_ESTIMATE`. A disconnect does not mean the provider incurred zero cost. |
| Provider disconnect / timeout after first byte | Same: known emitted text estimated; no failover |
| Output guardrail stops the stream | Estimated from emitted text |
| Provider never reports final usage | Estimate or `UNAVAILABLE`, never silently `PROVIDER_MEASURED` |

## Persistence sequence

There is no distributed transaction across provider, Postgres, Redis, and HTTP.

1. Provider result is in hand.
2. Durable insert (`ON CONFLICT` / unique `event_id`).
3. If the insert was **new**, Redis `HINCRBY` (atomic pipeline).
4. HTTP response.

Redis is best-effort. Losing Redis does not erase the ledger. Replaying the
same `event_id` does not increment Redis again.

## Crash windows (not exactly-once)

This is **idempotent durable event recording**, not exactly-once execution.

| Sequence | Result |
| --- | --- |
| Provider returns → process dies before INSERT | Invocation is lost. Classified remaining loss window. |
| INSERT succeeds → process dies before HTTP | Client may retry. A new HTTP request creates a **new** `invocation_id` (new `Attempt`). If the provider ran again, both real calls are counted. If the client retries without a new provider call, that is a new attempt in this gateway. |
| INSERT succeeds → Redis INCR fails | Ledger is ahead. `llm-fabric-usage reconcile` reports `DRIFT`. `--repair` overwrites Redis from the ledger, never the reverse. |
| Duplicate delivery of the same event | One row. |

## Postgres failure during a request

Happy path is a **synchronous insert** (backpressure: a slow database slows
the request). On insert failure the event is placed in a **bounded** retry
buffer (256). The next persist attempt flushes it. If the buffer is full the
event is dropped, a counter is incremented, and an error is logged. The HTTP
response is still returned because generation already happened; failing it
would invite a client retry and a second real provider call.

Silently dropping is not acceptable: drops are counted and logged. Unmetered
production inference under a sustained outage is closed by **P0-FIX-4**:
once PostgreSQL is known unhealthy, `/readyz` is 503 and new chat requests
are refused **before** a provider call. The retry buffer remains capped at
256. It is not filled by post-outage generations.

## In-flight request when a dependency fails mid-call

If PostgreSQL or Redis disappears **after** a provider invocation has already
started, MyVista does not undo provider consumption.

| Already started | Behaviour |
| --- | --- |
| Provider generation | Allowed to finish (bounded by existing provider timeouts). Not cancelled by P0-FIX-4. |
| Usage persist | Synchronous insert; on failure the event enters the bounded retry buffer. HTTP still returns the generation. |
| Subsequent new requests | Admission 503. No new provider call. |

New work stops. In-flight work follows this bounded path.

## Client disconnect

The ASGI generator is closed. Metering records whatever was already emitted
(estimated) plus any attempt the router stored. The gateway does not wait for
a provider-final usage frame that will never arrive.

## Multi-worker

`uvicorn --workers N` without `LLM_FABRIC_DATABASE_URL` and
`LLM_FABRIC_REDIS_URL` is refused (unless the unsafe acknowledgement is set,
and that acknowledgement is forbidden in production).
