# Routing

Intent, capability, service tier, model, provider, and authorization are
different questions. Mixing them produces a toy router.

```text
Request
  → authentication / tenant / policy
  → optional IntentOS classification (serving path OFF by default)
  → capability requirements
  → service tier L0–L30 (public spelling of Grade00–Grade29; L30 → Grade29)
  → eligible registry deployments
  → score (declared quality / cost / latency; observed health)
  → provider adapter
  → execution with bounded fallback
```

IntentOS does **not** pick a provider. A classification, when present, may
contribute capabilities and a preferred-tier list from `config/routing.yaml`.
Serving-path classification stays off (`LLM_FABRIC_INTENT_CLASSIFICATION_ENABLED`
defaults to false) until IntentOS gates pass.

## L0–L30

Tiers are configurable service levels, not parameter counts and not model names.
See [ADR 0004](adr/0004-service-tiers.md).

A coding-specialist deployment at L14 can beat a general L20 model for coding
when the intent policy prefers coding tiers. Capability match happens before
blind escalation.

`default_tier` in `config/routing.yaml` is operator documentation. It is **not**
applied as a global minimum (that would exclude `mock-small` at L2).

## Planner

Implemented in `src/llm_fabric/router/plan.py`:

1. Resolve an alias, a public tier (`L12`), or a pinned model id.
2. Filter: enabled, tenant allow/deny, locality, grade floor/ceiling, capabilities,
   context window, open circuit, budget, latency SLO.
3. If an intent policy names preferred tiers or models, **narrow only when at
   least one eligible deployment matches**. Otherwise keep the filtered set.
4. Score remaining candidates. Quality and latency missing for anyone are
   dropped for everyone. **Cost is ranked only among known API prices**; an
   omitted price is unknown, `0.0` is known-zero. See [`docs/COST-MODEL.md`](COST-MODEL.md).
5. Apply model promotion. Development/test preserve declared mock fixtures.
   Production auto-selection and pinned model requests require
   artifact-bound `approved`; a YAML lifecycle declaration alone is rejected.
   `approved_tiers` can narrow declared `tiers`, and explicit workload approval
   can narrow them again.
6. Attach a directed fallback graph (`src/llm_fabric/router/fallback.py`).

Tenant policy and a request `maximum_grade` / `max_tier` can only **narrow**.
Requesting `L30` cannot raise a tenant ceiling.

## Explainability

`POST /v1/routes/preview` and `llm-fabric route explain` return the same decision
object the serving path uses, including `reason_codes`, `selected_tier`,
exclusions, `routing_policy_version`, and `routing_policy_hash`. They do not
call a provider. Human output is the default; `--json` prints the object.
Promotion exclusions are explicit (`not_probed`, `not_evaluated`,
`not_approved`, `not_approved_for_tier`, `not_approved_for_workload`,
`artifact_revision_mismatch`, `disabled`).
`llm-fabric route replay` / `simulate` re-plan from saved request metadata
without inference.

Chat responses add `x-fabric-selected-tier` when the chosen deployment has a
tier. They do not dump internal tenant policy to untrusted clients.

## Fallback and escalation

Fallback follows **typed reasons** (timeout vs context overflow vs provider
down), not a flat list. Depth is bounded by `LLM_FABRIC_MAX_ATTEMPTS` and, when
`config/routing.yaml` is versioned, `routing.escalation.max_steps`. There is no
automatic jump to L30 on ordinary failure.

## Overrides

Authorized callers may request a model id, a tier (`L12`), a minimum grade, or a
maximum grade (preview). Overrides still obey tenant deny lists, locality,
capabilities, promotion state, and availability. A pinned unapproved model is
not production-usable merely because the caller knows its id.

## What is not built

- Serving-path IntentOS routing (switch stays off).
- Response / semantic completion cache on the serving path.
- GPU utilisation or vLLM queue-depth scraping.
- Config hot-reload (restart to pick up YAML).
- Automatic capability probing against live GPUs in CI.
