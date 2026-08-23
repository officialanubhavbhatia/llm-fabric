# ADR 0001 — Language and runtime

**Status:** Accepted, provisionally. Superseded by the engineering constitution if
it mandates otherwise.

## Context

Phase 1 needed a language. The engineering constitution, which would normally
settle this, did not reach the session — it was referenced twice and attached
neither time. The repository was empty, so there was no existing code to conform
to and no default to inherit.

Waiting was considered and rejected: the instruction to proceed was given twice,
and an empty repository is not a useful state to leave behind.

## Decision

Python 3.12 with FastAPI, pydantic for the schema, httpx for provider calls, and
uvicorn as the server.

## Rationale

The deciding factor was **provider ecosystem fit**. The serving layer's job is
absorbing the dialect differences between vendor APIs, and those APIs, their
reference clients, and their documented examples are Python-first. Async httpx
covers the concurrency this layer actually needs, which is I/O-bound waiting on
upstream inference, not CPU work.

pydantic earns its place separately: the public contract is a schema, and having
validation, serialisation, and the generated OpenAPI reference derive from one
type definition removes an entire class of drift between documentation and
behaviour.

## Alternatives

**Go or Rust.** A better fit for a proxy hot path under high concurrency, and the
plausible long-term answer if the fabric's own overhead ever becomes significant
relative to inference latency. Rejected for Phase 1 because it trades ecosystem
fit for a performance property that has not been measured and may not matter — a
gateway's overhead is usually small next to model generation time.

**A split: hot path in Go, control plane in Python.** Defensible, and rejected as
premature. It doubles the build and operational surface to solve a problem not yet
demonstrated to exist.

## Honest limits

**No benchmarks were run.** No latency or throughput measurement of any candidate
informed this decision, and none is claimed. The reasoning is about ecosystem fit
and the shape of the workload, not about measured performance.

## Consequences

If the constitution mandates a different language, the code is rewritten. What
survives is the design: the layer boundaries, the provider interface, the registry
schema, and the public contract are all expressible in any of the alternatives.
That is the cost of having proceeded, and it is stated up front rather than
discovered later.
