# MyVista LLM Fabric — Architecture

> **Document status: DRAFT — awaiting the engineering constitution.**
>
> The first message to this repository referenced an attached "project engineering
> constitution" to be treated as the permanent architecture and engineering
> specification. **That attachment did not reach this session** — only the cover
> note describing it arrived. See
> [Blocked on the constitution](#blocked-on-the-constitution).
>
> Everything in this document that is not under
> [Repository inspection](#1-repository-inspection) is a **proposal inferred from
> the repository description and standard practice for this class of system**, not
> a restatement of the specification. It must be reconciled against the
> constitution before any of it is treated as settled. Nothing here has been
> implemented.

---

## 1. Repository inspection

Inspection is complete. Findings are factual and current as of this commit.

### 1.1 Remote state

| Property | Value |
| --- | --- |
| Canonical remote | `github.com/officialanubhavbhatia/llm-fabric` |
| Description | "LLM Gateway, Serving and Inferencing Layers" |
| Visibility | Public |
| Default branch | `main` |
| Created | 2026-08-23 |
| Contents on GitHub | Empty — no commits, no tree |

### 1.2 Working tree state

The repository contains no application code, no build system, no dependency
manifest, no tests, no CI configuration, and no infrastructure definitions.

Tracked files, in full:

| Path | Origin | Purpose |
| --- | --- | --- |
| `README.md` | This session | Placeholder describing the repo |
| `AGENTS.md` | This session | Scope note: work stays in this repo |
| `.cursor/rules/llm-fabric.mdc` | This session | Same scope note, as an editor rule |

There is no prior engineering history to preserve, no legacy interface to keep
compatible, and no migration to plan. This is a greenfield repository.

### 1.3 What this means for sequencing

Because the tree is empty, repository inspection cannot surface the design
constraints that would normally shape Phase 1 — there is no existing module
layout, dependency choice, or interface contract to conform to. The constitution
is therefore the **only** source of those constraints, which is why Phase 1 has
not been started.

---

## 2. Architecture as currently understood

Derived from the repository description "LLM Gateway, Serving and Inferencing
Layers", which names three concerns. Presented for confirmation or correction.

The system is a **fabric**: a single control point that accepts inference
traffic, decides where each request should run, and executes it against either
external provider APIs or self-hosted model servers, while remaining accountable
for cost, latency, and reliability.

### 2.1 Gateway layer (ingress and contract)

The stable public surface. Owns everything that must happen before a routing
decision is possible:

- Request admission: authentication, tenant and project identity, API-key
  lifecycle.
- Enforcement: rate limits, quotas, spend ceilings, per-tenant policy.
- Contract: a versioned request/response schema, with streaming support.
  An OpenAI-compatible surface is the usual choice, because it makes the fabric
  adoptable without client rewrites.
- Normalization: turning many client dialects into one internal request object.

The value of this layer is that it is the *only* thing callers depend on.
Providers, models, and serving topology behind it stay replaceable.

### 2.2 Routing and policy layer (the decision)

Given a normalized request, decide which model on which backend should serve it:

- Model selection against a **model registry** — capability, context window,
  modality, cost, and availability per model.
- Policy-driven routing: cheapest capable model, lowest latency, pinned model,
  or tenant-specific preference.
- Resilience: fallback chains and provider failover, timeouts, retries with
  budget, circuit breaking on a degraded backend.
- Caching: exact-match first; semantic caching only if the constitution calls
  for it, since it trades correctness risk for cost.

This layer is where the fabric earns its name. It is also the layer most likely
to be specified in detail by the constitution, so it is the one I am least
willing to guess at.

### 2.3 Serving and inference layer (execution)

Executes the decision against a concrete backend:

- **Provider adapters** for external APIs, behind one internal interface, so a
  new provider is an adapter and not a change to routing.
- **Self-hosted serving** for in-house models: engine integration, replica pool
  selection, and cache-aware placement so requests sharing a prefix land on a
  replica that already holds that KV cache.
- Uniform translation of backend responses, errors, and token accounting back
  into the internal contract.

### 2.4 Observability and control plane (cross-cutting)

Not a layer so much as a requirement on every layer:

- Per-request metering: tokens in and out, cost, latency broken down by stage,
  chosen route, and why it was chosen.
- Tracing across gateway, routing decision, and backend call.
- Operational surfaces: health, readiness, and configuration state.

Routing decisions are only defensible if they are attributable after the fact,
so decision provenance is treated as a first-class output rather than a log line.

### 2.5 Request path

```
client
  -> gateway        auth, tenancy, limits, normalize
  -> router         model registry + policy -> route decision
  -> adapter        provider API  |  self-hosted pool (cache-aware)
  -> response       normalize, meter, trace
  -> client
```

---

## 3. Conflicts between the specification and existing code

**No code-level conflicts exist.** The repository has no application code, so
there is nothing that can contradict the specification.

Two items still need reconciliation, both introduced during this session rather
than inherited:

| Item | Nature | Resolution |
| --- | --- | --- |
| `README.md`, `AGENTS.md`, `.cursor/rules/llm-fabric.mdc` | Placeholders written before the constitution was requested. They assert scope, not architecture. | Rewrite or delete to match the constitution's documentation conventions. |
| Commit authorship | Earlier commits in this session were recorded under the default agent identity. | Fixed going forward: git identity is now `Anubhav Bhatia`, and signing with the agent key is disabled. |

The genuine risk is not conflict but **divergence**: if Phase 1 is implemented
against the inferred architecture in section 2 and the constitution specifies a
different language, module boundary, or routing model, the work is discarded.
That is the reason for holding.

---

## 4. Proposed repository structure

Proposal only. The concrete layout depends on the language the constitution
mandates, so this is expressed as a **module decomposition** — the boundaries
matter more than the file names, and the boundaries should survive translation
into whichever stack is specified.

```
llm-fabric/
├── ARCHITECTURE.md          # this document
├── README.md                # what it is, how to run it
├── docs/
│   ├── constitution.md      # the specification, committed verbatim
│   └── adr/                 # architecture decision records
│
├── gateway/                 # §2.1 ingress
│   ├── http/                #   server, routes, streaming
│   ├── auth/                #   keys, tenancy
│   ├── limits/              #   rate limits, quotas, spend
│   └── contract/            #   versioned public schema
│
├── router/                  # §2.2 the decision
│   ├── registry/            #   model catalog + capabilities
│   ├── policy/              #   selection strategies
│   ├── resilience/          #   fallback, retry, circuit breaking
│   └── cache/               #   response caching
│
├── serving/                 # §2.3 execution
│   ├── adapters/            #   one module per external provider
│   ├── selfhosted/          #   engine integration, pool selection
│   └── normalize/           #   responses, errors, token accounting
│
├── observability/           # §2.4 metering, tracing, health
├── config/                  # typed configuration and secret loading
├── tests/
│   ├── unit/
│   ├── integration/         #   against recorded provider fixtures
│   └── contract/            #   public surface stability
└── deploy/                  # containers, manifests, CI
```

Three properties this layout is chosen to guarantee:

1. **Adding a provider touches one directory.** `serving/adapters/` only.
2. **Routing is testable without network access.** `router/` depends on the
   registry and policy interfaces, not on live providers.
3. **The public contract is versioned separately from its implementation**, so
   the gateway can be refactored without breaking callers.

### 4.1 Language and runtime — undecided

Deliberately left open, because it is the single decision most likely to be
mandated by the constitution and the most expensive one to reverse.

For the record, the trade-off as I see it: Python maximizes ecosystem fit for
provider SDKs and model tooling, while Go or Rust give a materially better
proxy hot path under concurrency. A split — hot path in one, control plane in
the other — is defensible but doubles the build and operational surface. **I
have not benchmarked any of these and make no performance claims.**

---

## 5. Blocked on the constitution

Phase 1 is not started. To begin it without guessing, the following must come
from the specification rather than from inference:

1. **The constitution document itself**, committed to `docs/constitution.md` as
   the durable source of truth.
2. **Phase definitions** — what Phase 1 delivers, and its acceptance criteria.
3. **Language, runtime, and framework.**
4. **The public API contract** — OpenAI-compatible or bespoke; which endpoints.
5. **Deployment target** — single service, or Kubernetes-native with a control
   plane.
6. **Scope of self-hosted serving** in Phase 1, versus external providers only.
7. **State dependencies** — datastore and cache, if any, in Phase 1.
8. **Non-functional requirements** — the latency, throughput, and availability
   targets the design must actually meet.

Send the constitution and Phase 1 begins against it directly.

---

## 6. Engineering rules in force

Recorded here because they are permanent, and they apply to every commit in
this repository.

- **Authorship.** All production code is authored under Anubhav Bhatia. No AI or
  Cursor authorship, co-author trailers, or generated-by markers appear in
  commits, code, or documentation.
- **No fabrication.** Benchmarks, test results, metrics, and capability claims
  are only stated when they have actually been produced by a run that can be
  reproduced. Unmeasured means unstated. Unimplemented means described as
  unimplemented — as in section 2 of this document.
- **Scope.** Work stays within this repository. No other project is modified.
- **Documentation tracks reality.** This file is updated when the architecture
  changes, and it distinguishes what is built from what is planned.
