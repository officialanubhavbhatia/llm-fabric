# Load benchmarks

Every number here was produced by `llm-fabric-load` against a running gateway
and can be reproduced with the command printed beside it. Nothing is estimated,
extrapolated, or carried over from another machine.

**Status labels.** Figures in this file are marked:

- **CURRENTLY REPRODUCED** — re-run on this tree against the production-like
  Compose gateway (Postgres + Redis + mock provider) on 2026-08-24.
- **HISTORICAL** — measured on an earlier tree or a different serving shape
  (typically a single local process without the production dependency stack).
  Kept for comparison. Not this release's capacity claim.
- **NOT REPRODUCED** — not re-run on the production-like Compose/kind stack
  used for the 2026-08-24 audit.

**Read the limits before the numbers.** They are in §1, deliberately first.

---

## Currently reproduced (2026-08-24)

| | |
| --- | --- |
| Status | CURRENTLY REPRODUCED |
| What | Gateway RPS against the mock provider. Not inference RPS. Not token throughput. |
| Stack | Docker Compose production gateway, 1 worker, PostgreSQL usage ledger, Redis quotas, API-key auth |
| Hardware | Apple M2 Pro, 32 GiB, Darwin 25.6.0, `arm64` |
| Command | `uv run llm-fabric-load --host 127.0.0.1 --port 47317 --workload chat-short --duration 8 --warmup 2 --connections 32 --processes 2` |
| Result | **237 req/s**, 1920 requests, 0 errors, p50 132.34 ms, p95 168.15 ms, p99 221.26 ms |
| Memory | `docker-gateway-1` 94.4 MiB → 97.3 MiB |
| Artifact | `artifacts/audit-2026-08-24/load-chat-short.json` |

Open-loop 500 rps offer on the same gateway: achieved 226 req/s, 0% HTTP errors,
p99 319.67 ms. The generator did not sustain 500 rps. This is **not** the
historical 1,000 req/s open-loop figure in §5.

---

## 1. What these measurements are not

**They are not a measure of inference.** Every workload runs against the mock
provider, which returns text assembled from the request and performs no
inference. What is measured is the fabric's own cost — authentication, tenancy,
quota, routing, adapter dispatch, metering, serialisation. A real provider adds
its own latency, which no amount of gateway tuning removes.

**Three different throughputs.** Never collapse these into one RPS number:

- **Gateway RPS** — HTTP requests the control plane can accept (this document).
- **Inference request throughput** — completed generations against a real model. **Not measured here.**
- **Token throughput** — prefill/decode tokens per second from a real engine. **Not measured here.**

The numbers below are Gateway RPS against the mock provider.

**They were measured on a laptop.** An Apple M2 Pro under macOS, with the load
generator running on the same machine and competing for the same CPUs. A server
under Linux will behave differently in both directions. These figures are
trustworthy as a before/after comparison on one machine and as an order of
magnitude; they are not a capacity plan.

**They are not a comparison against anything.** No other gateway was measured.
No claim is made or implied that this is fast relative to any alternative.

**One process, one machine.** No replica count is assumed, no load balancer is
involved, and nothing here says what a fleet would do.

---

## 2. Environment

Captured automatically into every result file, so a report can never drift from
the machine that produced it.

| | |
| --- | --- |
| Hardware | Apple `arm64`, 12 logical CPUs |
| OS | Darwin 25.6.0 (macOS), `xnu-12377.161.13~4` |
| Python | 3.12.8 |
| uvicorn / FastAPI / pydantic | 0.52.4 / 0.141.1 / 2.13.4 |
| Event loop / HTTP parser | uvloop 0.22.1 / httptools 0.8.0 |
| Server | one worker, access log off, `LLM_FABRIC_LOG_LEVEL=WARNING` |
| Generator | 4 processes, on the same machine as the server |
| Registry | `config/models.yaml`, mock provider only |

Model, context length and token counts are given per workload in §3. GPU is not
reported because no workload touches one.

---

## 3. Workloads

The constitution forbids publishing an RPS number without saying what was being
served. Each workload is a fixed request shape defined in
`src/llm_fabric/bench/load.py`.

| Workload | What it exercises | Prompt tokens | Output tokens |
| --- | --- | --- | --- |
| `liveness` | `GET /healthz`. ASGI and event loop only. | 0 | 0 |
| `models` | `GET /v1/models`. Auth, tenancy, registry serialisation. | 0 | 0 |
| `route-preview` | `POST /v1/routes/preview`. The full planner, no inference. | ~14 | 0 |
| `chat-short` | `POST /v1/chat/completions`, alias `auto`. Whole serving path. | ~14 | ~20 |
| `chat-long` | As `chat-short`, ~40x longer prompt. | ~330 | ~20 |
| `chat-stream` | Streaming SSE, read to `[DONE]`. | ~14 | ~20 |
| `chat-pinned` | A pinned model, so no ranking runs. | ~14 | ~20 |

Token counts are the fabric's own heuristic estimates, not provider-reported
figures — the mock provider reports none.

---

## 4. Capacity (HISTORICAL / NOT REPRODUCED on Compose)

Closed-loop saturation on a **single local process**, mock provider, measured
on an Apple M2 Pro laptop. This is **not** the currently reproduced Compose
figure (237 req/s). It has not been re-run against the production-like
Postgres+Redis gateway in the 2026-08-24 audit.

Closed loop: 64 connections, load offered as fast as the server accepts it, so
these are saturation figures. 8 seconds measured, 3 seconds of warmup discarded.

```bash
llm-fabric-load --workload chat-short --duration 8 --warmup 2 \
  --connections 64 --processes 4
```

| Workload | Throughput | p50 | p95 | p99 | Errors |
| --- | ---: | ---: | ---: | ---: | ---: |
| `liveness` | 8,845 req/s | 7.18 ms | 7.93 ms | 19.44 ms | 0 |
| `models` | 5,968 req/s | 10.62 ms | 11.68 ms | 23.32 ms | 0 |
| `route-preview` | 3,341 req/s | 18.79 ms | 23.57 ms | 32.46 ms | 0 |
| `chat-pinned` | 2,531 req/s | 24.71 ms | 28.32 ms | 38.84 ms | 0 |
| `chat-short` | 2,377 req/s | 26.48 ms | 29.86 ms | 40.18 ms | 0 |
| `chat-long` | 2,314 req/s | 27.20 ms | 30.30 ms | 42.01 ms | 0 |
| `chat-stream` | 1,538 req/s | 39.76 ms | 54.67 ms | 56.24 ms | 0 |

`liveness` is a floor for the machine and the generator together, not a gateway
capability. The workloads below 3,000 req/s are server-bound, and the
differences between them are real.

**`chat-stream` is the binding constraint** at 1,538 req/s. A deployment whose
traffic is mostly streaming should plan against that number, not `chat-short`.

`tokens_per_s` is `null` on every HTTP result: the mock provider does not report
tokens. GPU is `null`: `nvidia-smi` is absent and this process does not use the
Apple GPU. Queue depth is `null` on HTTP results: the harness does not invent
one from connection count. CPU and RSS in those artifacts are the generator,
not the server.

### Saturation

Throughput is flat from 16 to 256 connections while latency grows linearly —
the signature of a fully saturated single bottleneck. The server reaches its
ceiling at 16 connections; everything above that is queueing.

| Connections | Throughput | p50 | p99 | max |
| ---: | ---: | ---: | ---: | ---: |
| 16 | 2,422 req/s | 6.57 ms | 8.27 ms | 10.28 ms |
| 32 | 2,324 req/s | 13.43 ms | 26.74 ms | 41.04 ms |
| 64 | 2,377 req/s | 26.48 ms | 40.18 ms | 46.29 ms |
| 128 | 2,261 req/s | 55.56 ms | 86.95 ms | 96.19 ms |
| 256 | 2,278 req/s | 113.31 ms | 166.54 ms | 2,902.57 ms |

Adding connections past 16 buys no throughput and costs latency proportionally.
The 2.9 s maximum at 256 connections is a real observation, not discarded.

---

## 5. The 1,000 req/s target (HISTORICAL / NOT REPRODUCED on Compose)

The constitution sets an initial gateway target of **500 req/s** "while
preserving correctness and tenant isolation". The figure below is 1,000, which
is the number that was asked for on a single local process. It is **not**
currently reproduced on the Compose production-like stack (see the 237 req/s
run above). Meeting 1,000 on that earlier shape also met the constitution's
500; that does not transfer to Compose or Kubernetes without a new measurement.

Open loop: the generator offers a fixed arrival rate whether or not the server
keeps up, so a struggling server is still offered the full rate. 20 seconds
measured, 3 discarded, 128 connections.

```bash
llm-fabric-load --workload chat-short --rate 1000 --duration 20 --warmup 3 \
  --connections 128 --processes 4 --target-rps 990 --max-error-rate 0
```

| Workload | Offered | Achieved | p50 | p95 | p99 | max | Errors |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `chat-short` | 1,000 req/s | 1,000 req/s | 1.16 ms | 4.89 ms | 44.22 ms | 120.80 ms | 0 |
| `chat-stream` | 1,000 req/s | 1,000 req/s | 1.84 ms | 3.35 ms | 10.65 ms | 30.59 ms | 0 |

**On that historical single-process run, a single worker sustained 1,000 req/s
on both, with no errors.** That sentence does not describe Compose or
Kubernetes. `chat-short` ran at roughly 42% of the saturation throughput
measured in §4, which is why its median stayed low. The 44 ms p99 and 121 ms
maximum on that same run are queueing on a laptop that is also generating the
load; they are not dropped to make the tail look smaller.

The tail moves between runs, because the generator shares a laptop with the
server. The medians and the achieved rate were stable. Treat the tail figures
as an order of magnitude and nothing finer.

No horizontal scaling, no worker pool and no configuration change was needed to
reach this on that historical single-process run. It is not a Compose or
Kubernetes result. See §7.

---

## 6. Where the time goes

`llm-fabric-profile` drives the ASGI application in-process, with no sockets and
no HTTP parsing, to separate the fabric's own cost from the transport's.

```bash
llm-fabric-profile --workload chat-short --iterations 3000 --time-only
```

One `chat-short` request costs **346.6 µs** of in-process work (best of 5 runs of
3,000 requests). That is the fabric's own cost with no sockets. Historical
saturated HTTP `chat-short` on a single local process was 2,377 req/s, which
is one completion every 421 µs of wall time — those two numbers are different
experiments and are not subtracted into a transport percentage. Currently
reproduced Compose gateway RPS is 237 req/s.

### The one change made for performance

Every gateway dependency was `def`. FastAPI runs a *sync* dependency in a worker
thread, on the assumption it might block; none of these block, so the chat path
was dispatching **24 requests to a thread pool per request** to perform
attribute lookups. They are now `async def`.

Measured, without a profiler attached, on the tree that introduced the change
(not this session's 346.6 µs figure above):

| | In-process cost per request |
| --- | ---: |
| Sync dependencies | 708.9 µs |
| Async dependencies | 200.8 µs |

**This did not change saturated throughput**: `chat-short` measured 2,027 req/s
before and 2,029 req/s after, which is noise. Under concurrency the thread hops
overlap with other requests' work, so they cost latency rather than capacity.
The change is kept because it removes 24 needless thread dispatches per request
and with them a thread-pool exhaustion mode at high concurrency — not because it
made anything faster. No throughput improvement is claimed for it.

`cProfile` reported a far larger win (4.81 s → 1.69 s per 3,000 requests) than
the profiler-free timing shows, because it charges per function call and the
thread-dispatch path is call-heavy. That is why the table above is measured
without it.

---

## 7. Horizontal scaling and shared state

Uvicorn `workers > 1` in a single process supervisor is still refused in
production. Kubernetes `replicaCount` is a different shape: separate gateway
processes that share PostgreSQL (usage, tenants) and Redis (quotas, breakers,
revocation, cache).

A second **uvicorn worker** without Redis and PostgreSQL still multiplies
in-process limits. That is why `workers > 1` requires both URLs (or the
development `ALLOW_UNSAFE_MULTIWORKER` hatch, which production rejects).

Replica count greater than 1 is the supported horizontal path. It is not a
measured fleet capacity figure. Nothing in this document is a Kubernetes RPS
number.

The constitution's target is throughput *"while preserving correctness and
tenant isolation"*. Per-process quotas without Redis are neither. Shared Redis
quotas across four local processes were verified with a 5 request/minute
ceiling (5×200, 7×429, five usage rows, no 500/503). That is an admission-
control proof, not a throughput proof.

---

## 8. Not measured

Stated so nothing here is read as broader than it is.

- **Real providers.** Every figure uses the mock provider. OpenAI and Anthropic
  adapters have never been load tested.
- **Long generations.** `chat-long` varies the *prompt*; no workload produces a
  long completion, because the mock provider does not generate one.
- **Sustained load.** The longest run here is 20 seconds. Nothing is known about
  memory growth, leak behaviour, or performance over hours.
- **Intent classification on the HTTP serving path.** All HTTP runs have it
  disabled, which is the default. The offline cascade itself is timed in §10
  (0.58 ms p50 in-process). The extra HTTP cost of turning it on for every
  request has not been measured.
- **Concurrent tenants.** Every run uses one identity. Quota-ledger contention
  across many tenants is unmeasured.
- **OIDC authentication.** HTTP runs use the anonymous path. The in-process
  API-key verifier is timed in §10. JWKS signature verification per request is
  unmeasured.
- **Failure behaviour under load.** Circuit breakers, failover and fallback
  budgets are correctness-tested but never load tested.
- **Linux, containers, and real networks.** All of it.

---

## 9. Reproducing

```bash
# One terminal: a single-worker gateway with its own logs quiet.
LLM_FABRIC_LOG_LEVEL=WARNING LLM_FABRIC_PORT=8000 python -m llm_fabric

# Another: capacity, then the target.
make bench-load          # chat-short saturation
make bench-load-target   # 1,000 req/s open loop, fails if unmet
make bench-load-all      # every workload
make bench-stages        # isolated in-process slices → artifacts/bench
```

`--calibrate` measures the generator's own ceiling in the same run and warns
when a result comes within 20% of it, which is the point at which the harness is
measuring itself.

Every `llm-fabric-load` and `llm-fabric-perf` run also writes a versioned JSON
file under `artifacts/bench/<utc>/<kind>-<commit>.json` (gitignored) plus a
`<kind>-latest.json` pointer. The numbers that matter are copied here after a
run, not invented in the artifact.

These benchmarks are **not** run in CI. A shared runner's throughput varies by
more than the regressions worth catching, so a gate there would either be too
loose to matter or too flaky to trust. `--target-rps`, `--max-p99-ms` and
`--max-error-rate` exist for a dedicated machine where that is not true.

---

## 10. Isolated stage benches

`llm-fabric-perf stages` times each named slice in-process, with no sockets.
Errors are counted; they are never dropped to inflate throughput. A stage whose
backend is not built returns `available=false` and no timings.

```bash
llm-fabric-perf stages --iterations 2000 --warmup 100
```

Measured on this machine, 2,000 iterations after 100 warmup, 0 errors on every
available stage. GPU is `null`. RSS is the bench process, about 69–74 MB.

| Stage | What it is | p50 | p95 | p99 | Throughput | CPU | Queue |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `api_gateway` | ASGI `GET /healthz` | 0.065 ms | 0.082 ms | 0.119 ms | 14,707 /s | 0.14 s | — |
| `auth` | one API-key verify | 0.002 ms | 0.003 ms | 0.006 ms | 406,151 /s | 0.01 s | — |
| `intent_exact_cache` | L0 hit | 0.005 ms | 0.005 ms | 0.009 ms | 194,493 /s | 0.01 s | — |
| `semantic_intent_cache` | L1 hit, `HashingEmbedder` | 0.026 ms | 0.032 ms | 0.038 ms | 36,640 /s | 0.06 s | — |
| `classifier` | offline cascade L0–L3 | 0.583 ms | 0.681 ms | 0.723 ms | 1,675 /s | 1.26 s | — |
| `router` | `RoutePlanner.plan` `auto` | 0.031 ms | 0.039 ms | 0.064 ms | 30,878 /s | 0.07 s | 0 |
| `ollama_inference` | adapter not built | — | — | — | unavailable | — | — |
| `vllm_inference` | adapter not built | — | — | — | unavailable | — | — |
| `streaming` | `MockProvider.stream` drain | 0.002 ms | 0.006 ms | 0.008 ms | 347,022 /s | 0.01 s | — |
| `full_system` | ASGI `chat-short` | 0.349 ms | 0.577 ms | 0.816 ms | 2,593 /s | 0.80 s | — |

These throughputs are **not** HTTP req/s. Do not compare `full_system` 2,593 /s
to historical §4 2,377 req/s, or to currently reproduced Compose 237 req/s, as
if they were the same experiment.

**Bottleneck from this run.** The offline classifier is the slowest built slice
(0.58 ms p50, 1,675 /s). That is why intent classification stays off on the
serving path: putting it in front of every request would cap that path at the
cascade, and the HTTP cost of doing so has not been measured.

`cProfile` on 3,000 in-process `chat-short` requests (2.975 s with the profiler
attached — not a capacity figure) spent the most fabric-owned cumulative time in
OpenTelemetry span setup (`otel.span` 0.784 s, 8 spans per request) and FastAPI
dependency solving (0.516 s). `RoutePlanner.plan` was 0.340 s; `score_candidates`
was 0.155 s. The profiler charges per call; `--time-only` (346.6 µs) is the
number to believe for cost.

---

## 11. Optimizations not enabled

The constitution requires a measured gain before a technique becomes a
production default. `llm-fabric-perf optimizations` lists every named technique
as `enabled: false`.

| Technique | Why it is off |
| --- | --- |
| connection pooling | httpx keep-alive already exists on the OpenAI/Anthropic adapters as client plumbing. It has not been A/B tested against a real provider. |
| async batching | No batching engine. |
| semantic caching | The L1 intent cache exists and is off on the serving path. A response semantic cache is not built. |
| prefix caching | A vLLM/Ollama property. Those adapters are not built. |
| quantized KV cache | A vLLM property. The adapter is not built. |
| continuous batching | A vLLM property. The adapter is not built. |
| chunked prefill | A vLLM property. The adapter is not built. |
| speculative decoding | A vLLM property. The adapter is not built. |
| model residency | No model loader. |
| request coalescing | Not implemented. |

The one change that *was* measured for performance — making FastAPI dependencies
`async def` — did not change saturated throughput. That result stays in §6.
Nothing was enabled on the strength of a profile alone.
