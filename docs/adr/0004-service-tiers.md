# ADR 0004 — Public service tiers L0–L30

**Status:** Accepted
**Date:** 2026-08-24

## Context

The constitution names thirty capability bands, `Grade00` through `Grade29`.
Operators and the README need a shorter public spelling. The platform target
also asks for labels `L0` through `L30`.

Thirty-one public labels cannot become thirty-one constitutional grades without
amending the constitution.

## Decision

- `L0` … `L29` are the public names of `Grade00` … `Grade29`.
- `L30` is an exceptional-escalation label that maps onto `Grade29`. It is not
  a thirty-first grade.
- Routing still selects **deployments** from the model registry. A tier is never
  a model name.
- Preferred-tier lists live in `config/routing.yaml` and may only narrow an
  already-eligible set.
- IntentOS classification is a separate signal. Serving-path classification
  remains off until its own gates pass.

## Consequences

`Grade.parse("L7")` and `Grade.parse("L30")` work. Requesting `model: "L12"`
selects enabled deployments that serve that tier. Callers cannot invent
`Grade30`.
