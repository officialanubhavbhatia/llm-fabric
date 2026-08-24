# Production-readiness review

Verdict: **GO** for the intended internal tier.

This review authorizes a **controlled internal single-VPC Kubernetes**
deployment with managed PostgreSQL and Redis, API-key or OIDC callers, and
Helm `autoscaling.enabled=false`. It does not authorize public internet
serving, multi-region HA, financial billing, unattended autoscaling, or any
inference-capacity SLA.

This audit was run from scratch on 2026-08-24 against the working tree on top
of `92ae8cc` (uncommitted P1 work). Closed P0-FIX-1..4 were not reopened.
A Helm template, a skipped test, or a mock provider was not counted as
production verification unless a live path was also exercised.

---

## Executive summary

The P1 items that previously kept this tree at CONDITIONAL GO were closed on
live PostgreSQL, Redis, Docker Compose, and kind:

- Production workers do not DDL. They refuse to start unless `alembic_version`
  is `0003_revoke_app_ddl`. The Helm pre-upgrade Job runs
  `python -m alembic upgrade head` as the table-owner role.
- `fabric_app` has no `CREATE` on `public`. Kind and Compose serve as
  `fabric_app`. Kind chat as that role returned 200.
- CI `quality` is green on this tree: `ruff check`, `ruff format --check`,
  `mypy src`.
- `llm-fabric-eval gate` with `LLM_FABRIC_ENVIRONMENT=test` exited 0;
  `failed=[]`. The report names `environment`, `dataset`, and `baseline`.
- Production quotas are finite. Four local workers sharing Redis admitted 5
  chat requests and returned 7×429 `quota_exceeded` (no 500/503). PostgreSQL
  `usage_events` for that tenant had 5 rows — rejected calls did not invoke
  the provider.
- Helm HPA is off by default. After upgrade, kind has no HPA object.
  Autoscaling was not observed and is not claimed.
- `pg_dump`/`pg_restore` of a current-head database preserved Alembic
  revision, RLS isolation, and usage rows (dump 17 115 bytes, restore
  0.324 s on Compose Postgres). That is not multi-region DR.
- An in-cluster SSE stream during `kubectl rollout restart` completed with
  `[DONE]`. Local SIGTERM of a delayed mock stream also drained to `[DONE]`.
- Two gateway processes exported OTLP HTTP traces to one sink; protobuf
  payloads contained distinct `service.instance.id` values. A host:port
  collector URL is normalized to `/v1/traces` (Compose was 404ing without
  that path).
- Command Center traces are labelled **local-pod diagnostic only — not
  authoritative fleet trace history**.
- `docs/BENCHMARKS.md` labels 2,377 req/s and 1,000 req/s as historical /
  not reproduced on Compose. Currently reproduced Compose mock gateway load
  remains **237 req/s**, 0 errors.

It is still **NO-GO** for public SaaS, billing-grade metering, multi-region
failover, and “HPA will save us.” There is no `LICENSE` file; treat the tree
as all-rights-reserved internal use until the owner publishes one.

---

## Intended deployment tier

Covered:

```text
controlled internal single-VPC Kubernetes
managed PostgreSQL and Redis
API-key or OIDC callers
OpenAI-compatible providers (including local Ollama via the OpenAI adapter)
replicaCount >= 2
autoscaling.enabled=false
workers = fabric_app (DML, no CREATE)
migrate Job = table-owner role (Compose/kind: fabric)
```

Not covered:

- multi-region or multi-AZ automatic failover of the fabric itself
- public internet exposure without a separate TLS-terminating edge
- billing-grade or exactly-once economics
- unattended autoscaling
- agents / tool execution (not on the serving surface)
- “500 inference RPS”, “1,000 gateway RPS”, or any capacity SLA

Compose on a laptop is a production-*like* stack for tests, not this tier.

### Operational conditions (required)

1. Leave Helm `autoscaling.enabled` false until metrics-server + CPU metrics
   are proven on *that* cluster. YAML is not verified autoscaling.
2. Point gateway pods at `fabric_app` (or equivalent). Point the migrate Job
   at `LLM_FABRIC_MIGRATION_DATABASE_URL` (table owner). Do not run Alembic
   as the app role on an empty database.
3. Set `LLM_FABRIC_OTEL_EXPORTER_OTLP_ENDPOINT` (full `/v1/traces` URL or
   host:port; the process appends `/v1/traces`). Headers stay in the Secret.
   `GET /v1/observability/traces` is not fleet history.
4. Prove `pg_dump`/`pg_restore` (or the vendor’s PITR) on the **managed**
   database you will actually use. The 17 KiB Compose dump is not that proof.
5. Internal use only until a `LICENSE` exists.
6. Do not quote 1,000 req/s or 2,377 req/s as this release’s Compose/kind
   capacity. The currently reproduced mock Compose figure is 237 req/s.

---

## P0 table

Unchanged from the prior audit on this machine. Not re-litigated.

| issue | status |
| --- | --- |
| P0-FIX-1 unset / invalid environment starts as development | **closed** |
| P0-FIX-2 serve while Postgres/Redis down at start | **closed** |
| P0-FIX-3 unmetered / per-worker usage ledger | **closed** |
| P0-FIX-4 serve new inference while mandatory deps down | **closed** |
| Authentication bypass on `/v1/*` | **closed** |
| Cross-tenant durable leak (Postgres RLS / app role) | **closed** |
| Fail-open production anonymous / `auth_mode=dev` | **closed** |

No unresolved P0 was found. That is why the verdict is not NO-GO.

---

## P1 table

| issue | status | evidence |
| --- | --- | --- |
| Schema lifecycle is `create_all` on worker start | **closed** | Production `build_engine` skips `init_schema`. Startup asserts `alembic_version=0003_revoke_app_ddl`. Helm migrate Job: `python -m alembic upgrade head`. kind after upgrade: `0003_revoke_app_ddl`. |
| App role can DDL (`CREATE` on `public`) | **closed** | `0003_revoke_app_ddl` revokes CREATE. kind: `has_schema_privilege(fabric_app, public, CREATE)=f`. Compose live tests serve as `fabric_app`. |
| Default quotas unlimited; no 429 overload shed | **closed** | Production fills finite RPM/RPD/concurrency/token ceilings. Live: 4 workers, Redis db 14, tenant RPM=5 → 5×200, 7×429 `quota_exceeded`, 5 usage rows, no 500/503. |
| HPA enabled without metrics-server | **closed** | Chart default `autoscaling.enabled: false`. kind after helm upgrade: `No resources found` for HPA. Autoscaling **not** claimed. |
| CI `quality` fails | **closed** | `uv run ruff check .` pass; `ruff format --check .` pass; `mypy src` “Success: no issues found in 140 source files”. |
| CI `eval-gates` missing `LLM_FABRIC_ENVIRONMENT` | **closed** | Workflow `env.LLM_FABRIC_ENVIRONMENT=test` plus job env. CLI refuses unset env (exit 2). Live gate exit 0, `failed=[]`, report includes environment/dataset/baseline. |
| Postgres backup/restore not proven | **closed for current schema, not DR** | `test_postgres_backup_live`: Alembic head, pg_dump -Fc, pg_restore, RLS (tenant B empty), usage tokens 4 vs 0. Size 17115 bytes, restore 0.324 s. Not multi-region. |
| Rolling-update stream drain not proven in-cluster | **closed** | `test_kind_rolling_stream` passed twice (26.20 s then 14.48 s). In-cluster client saw `[DONE]` during `rollout restart`. Local SIGTERM mock stream also drained. Uvicorn `timeout_graceful_shutdown=25`. |
| Kind/Helm has no OTEL; traces API is in-process | **closed as labelled** | Helm ConfigMap may set endpoint; headers stay in Secret. Two local workers posted OTLP to one HTTP sink (`worker-a` / `worker-b` in payloads). Command Center + `/v1/observability/traces` say local-pod diagnostic only. kind default endpoint remains empty until operators set it. |
| BENCHMARKS.md presents 2,377 / 1,000 req/s as current | **closed** | Document now labels CURRENTLY REPRODUCED (237 req/s Compose) vs HISTORICAL / NOT REPRODUCED (saturation 2,377 and open-loop 1,000 on a single local process). |

These P1s are closed for the intended tier. They are not a license to enable HPA or to sell metering.

---

## Verified this pass

- Alembic-only production schema; workers refuse wrong/missing revision
- `fabric_app` without schema CREATE; DML serving on Compose and kind
- Finite production quotas shared across four processes via Redis; 429 not 500
- Helm HPA absent unless opted in
- `python -m alembic` migrate Job (venv `alembic` script shebang is `/src/.venv/bin/python` and does not exist in the runtime image)
- OTLP URL normalization; two-process export to one collector
- Trace UI/API labelled non-authoritative
- kind rolling restart drained an in-cluster SSE stream
- Eval gate with explicit `test` environment
- ruff / mypy green

---

## Still out of scope / not claimed

- **LICENSE.** None. Internal / all rights reserved.
- **HPA.** Disabled. `cpu: <unknown>` was the previous kind state; it is gone
  because the object is gone, not because metrics-server works.
- **Managed-database RPO/RTO.** Not measured. Compose dump is a procedure
  proof on this laptop’s Postgres container.
- **kind Postgres durability.** The kind Postgres used here lost `fabric_app`
  and all tables when `deploy/postgres` scaled to 0 (emptyDir). A leaked
  `LLM_FABRIC_KIND_TEST=1` caused `test_kind_readiness` to do that during a
  full suite run. Recovery required recreating the role and `alembic upgrade
  head`. Managed Postgres with a real volume would not behave like that.
- **Fleet OTEL on kind.** Not configured. Two local processes proved the
  exporter path. Operators must set the endpoint.
- **1,000 req/s / 2,377 req/s.** Historical single-process figures. Not this
  Compose stack.
- **Inference RPS / token throughput.** Not measured. Mock provider only for
  load.
- **Billing-grade usage.** Durable and idempotent is not exactly-once
  economics.

---

## Evidence appendix

**Hardware:** Apple M2 Pro, 32 GiB, Darwin 25.6.0 (`arm64`), CPython 3.12.8.

**Git:** working tree on top of `92ae8cc76f4e7ae6fdaff149ec2ec9e58d50b267`.
These P1 changes were not committed at audit time.

### Quality

| command | result |
| --- | --- |
| `uv run ruff check .` | pass |
| `uv run ruff format --check .` | 256 files already formatted |
| `uv run mypy src` | Success: no issues found in 140 source files |
| `cd sdk/typescript && npm test` | 3 passed |

### Tests

| command | result |
| --- | --- |
| `uv run pytest --strict-markers -q` with `LLM_FABRIC_KIND_TEST=1` still set in the shell | **1105 passed**, **2 failed** (kind readiness recovery after ephemeral Postgres recreate; rolling-stream blocked while those pods were NotReady). 456 s. |
| `LLM_FABRIC_KIND_TEST=1 uv run pytest tests/system/test_kind_rolling_stream.py -s` after remigrate | **1 passed**, 14.48 s |
| `uv run pytest tests/unit/test_quota_production_live.py -v` | **1 passed** (5×200, 7×429, 5 usage rows) |
| `uv run pytest tests/unit/test_postgres_backup_live.py -s` | **1 passed** (`backup_size_bytes=17115 restore_duration_s=0.324`) |
| `uv run pytest tests/unit/test_otel_two_workers_live.py` | **1 passed** |
| `uv run pytest tests/unit/test_stream_shutdown_live.py` | **1 passed** |

Default collection without kind env: **1107** tests. Kind tests are skipped
unless `LLM_FABRIC_KIND_TEST=1`. Do not export that variable in a developer
shell if kind Postgres is ephemeral.

### Eval gate

```text
command: LLM_FABRIC_ENVIRONMENT=test \
  LLM_FABRIC_REGISTRY_PATH=config/models.yaml \
  LLM_FABRIC_ALLOW_ANONYMOUS=true \
  uv run llm-fabric-eval gate --suite datasets/eval/ci-suite.yaml \
  --baseline datasets/eval/baseline.json
exit: 0
failed: []
environment: test
accuracy: 0.8776  macro_f1: 0.9012  exact_match: 1.0  route_match: 1.0
deepeval / lm_eval tasks named no metrics (“None were applied”)
```

### kind (after helm upgrade of this chart)

```text
alembic_version=0003_revoke_app_ddl
fabric_app CREATE on public=false
runtime DSN user=fabric_app
HPA: none
readyz 200, chat 200 (mock-small)
rolling restart during in-cluster SSE: [DONE]
```

### Load (mock, Compose gateway) — CURRENTLY REPRODUCED

Unchanged from the prior audit on this machine; not re-run in this P1 pass.

```text
237 req/s, 1920 requests, errors 0.0000%, p50 132.34 ms,
p95 168.15 ms, p99 221.26 ms
Compose production gateway, mock provider, 1 worker, Postgres+Redis
artifact: artifacts/audit-2026-08-24/load-chat-short.json
```

---

The verdict reflects the system that was actually exercised on 2026-08-24,
not a future SaaS shape.
