# Configuration reference

Every process setting is loaded by `src/llm_fabric/config.py` (`Settings`).
Names are prefixed `LLM_FABRIC_`. Provider credentials also accept the unprefixed
names the providers themselves document (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`);
the prefixed form wins when both are set.

Copy [`.env.example`](../.env.example) to `.env` for local work. Unset, empty, or
unknown `LLM_FABRIC_ENVIRONMENT` is a startup failure — the process does not
quietly become development.

This page lists the fields that exist in code. It is not a recommendation to set
all of them.

## Server

| Variable | Type | Default | Required | Production recommendation | Purpose |
| --- | --- | --- | --- | --- | --- |
| `LLM_FABRIC_ENVIRONMENT` | `development` \| `test` \| `production` | none | **yes** | `production` | Fail-closed deployment mode. |
| `LLM_FABRIC_HOST` | str | `127.0.0.1` | no | `0.0.0.0` behind a private load balancer | Bind address. |
| `LLM_FABRIC_PORT` | int | `47317` | no | `47317` | Listen port. |
| `LLM_FABRIC_LOG_LEVEL` | str | `INFO` | no | `INFO` or `WARNING` | Process log level. |
| `LLM_FABRIC_REGISTRY_PATH` | path | `config/models.yaml` | no | versioned registry | Model and alias registry. |
| `LLM_FABRIC_ROUTING_CONFIG_PATH` | path | `config/routing.yaml` | no | versioned policy | Intent → tier policy. Missing file loads empty. |
| `LLM_FABRIC_PROMOTION_CONFIG_PATH` | path | `config/promotion.yaml` | no | versioned policy | Evidence requirements and production lifecycle gates. |
| `LLM_FABRIC_PROMOTION_STATE_PATH` | path | `datasets/eval/models/promotion-state.json` | no | controlled artifact | Evidence-bound lifecycle overlay and audit history. Public requests cannot write it. |
| `LLM_FABRIC_WORKERS` | int \| unset | unset (one worker) | no | one worker, or replicas of one-worker pods | OS processes. See [multi-worker](#multi-worker). |
| `LLM_FABRIC_ALLOW_UNSAFE_MULTIWORKER` | bool | `false` | no | `false` | Required to start with `workers > 1`. Production refuses this flag. |
| `LLM_FABRIC_BACKLOG` | int | `2048` | no | leave default unless measured | Kernel accept queue. |
| `LLM_FABRIC_GRACEFUL_SHUTDOWN_TIMEOUT_S` | int | `25` | no | below Kubernetes `terminationGracePeriodSeconds` (chart default 30) | uvicorn wait for in-flight work including SSE. |
| `LLM_FABRIC_MAX_REQUESTS_PER_WORKER` | int \| unset | unset | no | set if you recycle for leak containment | Requests before a worker is replaced. |
| `LLM_FABRIC_MAX_REQUEST_BYTES` | int | `1048576` | no | keep a finite ceiling | Reject oversized bodies before routing. |
| `LLM_FABRIC_CORS_ORIGINS` | list | empty (CORS off) | no | explicit origins; `*` refused in production | Browser CORS allow-list. |
| `LLM_FABRIC_TRUSTED_PROXIES` | list | empty | no | the load-balancer CIDRs | Peers allowed to set `X-Forwarded-For` / `X-Forwarded-Proto`. Empty ignores those headers. |

## Identity and tenancy

| Variable | Type | Default | Required | Production recommendation | Purpose |
| --- | --- | --- | --- | --- | --- |
| `LLM_FABRIC_AUTH_MODE` | `disabled` \| `api_key` \| `dev` \| `oidc` \| unset | inferred | production: complete source | `oidc` or `api_key` | Identity source. Inference is a development convenience. |
| `LLM_FABRIC_ALLOW_ANONYMOUS` | bool | `true` | no | `false` (production refuses `true`) | Anonymous access in development/test when no identity source is configured. |
| `LLM_FABRIC_REQUIRED_SCOPES` | list | empty | no | the scopes every caller must carry | Global scope floor. Routes still enforce their own checks. |
| `LLM_FABRIC_API_CREDENTIALS` | JSON list | empty | api-key mode | per-tenant keys ≥ 16 chars | `{"key","tenant_id","user_id","roles","scopes"}`. |
| `LLM_FABRIC_API_KEYS` | list | empty | no | prefer `API_CREDENTIALS` | Legacy flat keys, all in tenant `default`. |
| `LLM_FABRIC_OIDC_ISSUER` | URL | unset | oidc | pinned issuer | OIDC issuer. |
| `LLM_FABRIC_OIDC_AUDIENCE` | str | unset | oidc | pinned audience | Tokens without this audience are rejected. |
| `LLM_FABRIC_OIDC_JWKS_URI` | URL | discovered | no | explicit if discovery is blocked | JWKS endpoint. |
| `LLM_FABRIC_OIDC_JWKS_CACHE_SECONDS` | float | `300` | no | leave default | JWKS cache TTL. |
| `LLM_FABRIC_DEV_AUTH_SECRET` | str ≥ 32 | unset | `dev` mode | **must not be set** | Local HS256 issuer. Production refuses it. |
| `LLM_FABRIC_CLAIM_TENANT` | str | `tenant_id` | no | match the issuer | Tenant claim name. |
| `LLM_FABRIC_CLAIM_USER` | str | `user_id` | no | match the issuer | User claim name. |
| `LLM_FABRIC_CLAIM_PROJECT` | str | `project_id` | no | match the issuer | Project claim name. |
| `LLM_FABRIC_CLAIM_ROLES` | str | `roles` | no | match the issuer | Roles claim name. |
| `LLM_FABRIC_CLAIM_SCOPES` | str | `scope` | no | match the issuer | Scopes claim name. |
| `LLM_FABRIC_ANONYMOUS_TENANT` | str | `public` | no | unused in production | Tenant when authentication is disabled. |

Delegation header `x-tenant-id` is honoured only when the **validated** token
carries the delegation scope. See [`SECURITY.md`](../SECURITY.md) and
[`docs/AUTH_REVOCATION.md`](AUTH_REVOCATION.md).

## Persistence

| Variable | Type | Default | Required | Production recommendation | Purpose |
| --- | --- | --- | --- | --- | --- |
| `LLM_FABRIC_DATABASE_URL` | DSN | unset (in-memory) | **production yes** | managed PostgreSQL, DML role `fabric_app` | Tenants, usage, traces, eval metadata. |
| `LLM_FABRIC_REDIS_URL` | DSN | unset (in-memory) | **production yes** | managed Redis/Valkey | Shared quotas, breakers, revocation, hot caches. |
| `LLM_FABRIC_MIGRATION_DATABASE_URL` | DSN | falls back to `DATABASE_URL` | migrate Job | table-owner role, never the app role | Alembic only (`alembic/env.py`). Not a `Settings` field. |
| `LLM_FABRIC_ANALYTICS_URL` | DSN | unset (discard) | no | optional ClickHouse | Off the request path. Unset discards analytics events. |

Workers refuse to start in production unless Alembic revision `0004_usage_topology`
is applied. Do not run `alembic upgrade` as `fabric_app`.

## Quotas

Empty means unlimited in development/test. Production fills finite defaults when
a field is unset (`src/llm_fabric/config.py`: 3,000 tenant RPM, 64 tenant
concurrency, and the rest of the production ceilings). Redis-backed when
`LLM_FABRIC_REDIS_URL` is set, so every replica shares one ceiling.

| Variable | Default (dev/test) | Production if unset |
| --- | --- | --- |
| `LLM_FABRIC_QUOTA_TENANT_REQUESTS_PER_MINUTE` | unlimited | 3,000 |
| `LLM_FABRIC_QUOTA_TENANT_REQUESTS_PER_DAY` | unlimited | 500,000 |
| `LLM_FABRIC_QUOTA_TENANT_REQUESTS_PER_MONTH` | unlimited | 10,000,000 |
| `LLM_FABRIC_QUOTA_TENANT_TOKENS_PER_DAY` | unlimited | 50,000,000 |
| `LLM_FABRIC_QUOTA_TENANT_COST_PER_DAY_USD` | unlimited | remains unset (no USD cap unless set) |
| `LLM_FABRIC_QUOTA_TENANT_MAX_CONCURRENCY` | unlimited | 64 |
| `LLM_FABRIC_QUOTA_TENANT_PROJECT_REQUESTS_PER_MINUTE` | unlimited | 3,000 |
| `LLM_FABRIC_QUOTA_TENANT_PROVIDER_REQUESTS_PER_MINUTE` | unlimited | 6,000 |
| `LLM_FABRIC_QUOTA_TENANT_MODEL_REQUESTS_PER_MINUTE` | unlimited | 3,000 |
| `LLM_FABRIC_QUOTA_USER_REQUESTS_PER_MINUTE` | unlimited | 1,200 |
| `LLM_FABRIC_QUOTA_USER_REQUESTS_PER_DAY` | unlimited | 100,000 |
| `LLM_FABRIC_QUOTA_USER_TOKENS_PER_DAY` | unlimited | 10,000,000 |
| `LLM_FABRIC_QUOTA_USER_COST_PER_DAY_USD` | unlimited | remains unset |
| `LLM_FABRIC_QUOTA_USER_MAX_CONCURRENCY` | unlimited | 16 |

These are operator ceilings, not billing-grade metering. See
[`docs/USAGE_METERING.md`](USAGE_METERING.md).

## Providers and routing

| Variable | Type | Default | Required | Production recommendation | Purpose |
| --- | --- | --- | --- | --- | --- |
| `LLM_FABRIC_DEFAULT_POLICY` | policy name | `cost_first` | no | the policy aliases should use | Policy for aliases that do not name one. `cheapest` still parses as `cost_first`. |
| `LLM_FABRIC_REQUEST_TIMEOUT_S` | float | `60` | no | per-attempt SLO | Per-attempt ceiling, not a total request budget. |
| `LLM_FABRIC_MAX_ATTEMPTS` | int | `3` | no | measured | Total attempts including the first, across the fallback graph. |
| `LLM_FABRIC_FALLBACK_MAX_COST_USD` | float \| unset | unset | no | set if spend during failover must be bounded | Extra bound beyond attempt count. |
| `LLM_FABRIC_FALLBACK_MAX_LATENCY_MS` | float \| unset | unset | no | set if failover latency must be bounded | Extra bound beyond attempt count. |
| `LLM_FABRIC_OPENAI_API_KEY` / `OPENAI_API_KEY` | secret | unset | when openai models enabled | secret store | OpenAI adapter only. |
| `LLM_FABRIC_OPENAI_BASE_URL` | URL | `https://api.openai.com/v1` | no | OpenAI or a generic compatible proxy | Not required for Ollama/vLLM named providers. |
| `LLM_FABRIC_OLLAMA_BASE_URL` | URL | `http://127.0.0.1:11434/v1` | no | local daemon or Compose service | Ollama OpenAI-compatible root. No OpenAI key required. |
| `LLM_FABRIC_OLLAMA_API_KEY` | secret | unset | no | only if the daemon is locked | Optional bearer for Ollama. |
| `LLM_FABRIC_VLLM_BASE_URL` | URL | `http://127.0.0.1:8000/v1` | no | the vLLM pool URL | vLLM OpenAI-compatible root. No OpenAI key required. |
| `LLM_FABRIC_VLLM_API_KEY` | secret | unset | no | if vLLM `--api-key` is set | Optional bearer for vLLM. |
| `LLM_FABRIC_LITELLM_BASE_URL` | URL | `http://127.0.0.1:4000/v1` | no | ClusterIP LiteLLM `/v1` | LiteLLM OpenAI-compatible root. Transport only. |
| `LLM_FABRIC_LITELLM_API_KEY` | secret | unset | no | if the proxy requires a key | Optional bearer; defaults to `litellm` when unset. |
| `LLM_FABRIC_LITELLM_NUM_RETRIES` | int | `0` | no | `0` | Expected LiteLLM `num_retries`. Values above 1, or a retry product above 9, refuse startup. |
| `LLM_FABRIC_PROVIDER_BASE_URLS` | JSON object | `{}` | no | per-pool URLs | e.g. `{"vllm-coding":"http://vllm-coding:8000/v1"}`. |
| `LLM_FABRIC_ANTHROPIC_API_KEY` / `ANTHROPIC_API_KEY` | secret | unset | when anthropic models enabled | secret store | Anthropic adapter. |
| `LLM_FABRIC_ANTHROPIC_BASE_URL` | URL | `https://api.anthropic.com/v1` | no | leave unless proxying | Anthropic API root. |
| `LLM_FABRIC_MAX_INPUT_TOKENS` | int \| unset | unset (prod 32,000) | no | finite | Prompt token ceiling. |
| `LLM_FABRIC_MAX_OUTPUT_TOKENS` | int \| unset | unset (prod 8,192) | no | finite | Completion token ceiling. |
| `LLM_FABRIC_MOCK_DELAY_S` | float | `0` | no | `0` | Test-only mock provider delay. |

Circuit breakers (per deployment, EWMA + consecutive failures):

| Variable | Default |
| --- | --- |
| `LLM_FABRIC_BREAKER_CONSECUTIVE_FAILURES` | `5` |
| `LLM_FABRIC_BREAKER_ERROR_RATE` | `0.5` |
| `LLM_FABRIC_BREAKER_MINIMUM_SAMPLES` | `10` |
| `LLM_FABRIC_BREAKER_OPEN_DURATION_S` | `30` |
| `LLM_FABRIC_BREAKER_HALF_OPEN_SUCCESSES` | `2` |
| `LLM_FABRIC_BREAKER_MAX_CONCURRENCY` | unset (prod 256) |

## IntentOS

Serving-path classification is **mandatory in every environment**. There is no
development or test bypass. A catastrophic cascade failure degrades to a typed
`SAFE_FALLBACK`, as required by the selected availability policy. See
[`docs/INTENTOS.md`](INTENTOS.md) and [ADR 0006](adr/0006-serving-path-intentos.md).

| Variable | Type | Default | Production recommendation | Purpose |
| --- | --- | --- | --- | --- |
| `LLM_FABRIC_INTENT_CLASSIFICATION_ENABLED` | bool | `true` | leave `true` | Deprecated compatibility setting. Classification is mandatory regardless. |
| `LLM_FABRIC_INTENT_ROUTING_ENABLED` | bool | `false` | `false` until gates clear | Apply the classification to route planning. Classification still runs when false. |
| `LLM_FABRIC_INTENT_SHADOW` | bool | `false` | optional observation | Duplicate classification in `x-fabric-intent-shadow-*` headers. |
| `LLM_FABRIC_INTENT_EMBEDDER` | `hashing` \| `minilm` \| `local` / `bge-small` | `hashing` | `local` or `minilm`, or hashing with the allow flag | L3 embedder. MiniLM needs `uv sync --extra embed`. |
| `LLM_FABRIC_INTENT_ALLOW_HASHING_EMBEDDER` | bool | `false` | required if embedder is hashing | Explicit acceptance that L3 is lexical hashing, not a semantic model. |
| `LLM_FABRIC_INTENT_L4_RERANK` | bool | `false` | `false` | Local description reranker as L4. L5 stays off in code. |
| `LLM_FABRIC_ROUTING_QUALITY_SHADOW` | bool | `false` | `false` until measured | Rank the same eligible set under `quality_first` and record the comparison. Does **not** change the served route. |

## Context compiler

The compiler always runs on `/v1/chat/completions`. Registry YAML may set
`metrics_endpoint` (vLLM `/metrics`) and `api_base` (Ollama `/api/ps`) for
off-path engine scrapes. See [`docs/CONTEXT.md`](CONTEXT.md) and
[ADR 0007](adr/0007-context-compiler-observability.md).

## Observability

| Variable | Type | Default | Production recommendation | Purpose |
| --- | --- | --- | --- | --- |
| `LLM_FABRIC_OTEL_EXPORTER_OTLP_ENDPOINT` | URL | unset (in-process only) | collector `/v1/traces` (host:port is accepted and `/v1/traces` is appended) | OTLP HTTP traces. Collector outage is fail-soft. |
| `LLM_FABRIC_OTEL_EXPORTER_OTLP_HEADERS` | `k=v,...` | unset | Secret, never ConfigMap | Auth headers for the collector. |
| `LLM_FABRIC_OTEL_EXPORTER_OTLP_CERTIFICATE` | path | unset | if TLS to collector | CA file. |
| `LLM_FABRIC_LANGFUSE_HOST` | URL | unset | optional | Langfuse. All three Langfuse vars must be set or the adapter is a no-op. |
| `LLM_FABRIC_LANGFUSE_PUBLIC_KEY` | secret | unset | Secret | Langfuse public key. |
| `LLM_FABRIC_LANGFUSE_SECRET_KEY` | secret | unset | Secret | Langfuse secret key. |

Dependency health probes (Postgres/Redis when configured):

| Variable | Default |
| --- | --- |
| `LLM_FABRIC_HEALTH_PROBE_INTERVAL_S` | `2` |
| `LLM_FABRIC_HEALTH_PROBE_TIMEOUT_S` | `1` |
| `LLM_FABRIC_HEALTH_FAIL_THRESHOLD` | `2` |
| `LLM_FABRIC_HEALTH_RECOVERY_THRESHOLD` | `2` |

## SDK (client, not the gateway)

| Variable | Purpose |
| --- | --- |
| `MYVISTA_BASE_URL` | Client base URL. Default `http://127.0.0.1:47317`. |
| `MYVISTA_API_KEY` | Client credential. |

## Test-only (not Settings)

| Variable | Purpose |
| --- | --- |
| `LLM_FABRIC_TEST_DATABASE_URL` | Isolation/CI Postgres. |
| `LLM_FABRIC_TEST_REDIS_URL` | Live Redis tests. |
| `LLM_FABRIC_KIND_TEST` | Enable kind/Helm system tests when `1`. |
| `SKIP_EVALS` / `LLM_FABRIC_SKIP_EVALS` | **Refused** by `llm-fabric-eval gate`. |

## Multi-worker

Quotas, breakers, the usage meter, and in-process caches live in one process
unless Redis (and Postgres) back them. `workers > 1` without Redis multiplies
every per-process limit. The process refuses that configuration unless
`LLM_FABRIC_ALLOW_UNSAFE_MULTIWORKER=true`, and production refuses the flag
entirely. Scale with replicas that share Redis, not with in-process workers.
