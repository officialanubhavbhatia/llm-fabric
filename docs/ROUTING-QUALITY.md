# Routing quality

These metrics ask whether L0–L30 routing picked a *defensible* deployment, not
whether a higher public tier is “smarter”. A specialist at L14 can beat a
general L20 model for coding. Higher tier is not always better.

## Overrouting

The selected deployment has a **known** API cost strictly greater than another
eligible deployment that already met the same capability and context
requirements.

- Known-zero (`0.0`) is a valid price.
- Unknown prices are not compared.
- Grade / tier ordinal is **not** used as a quality proxy.

```text
overrouting = selected_known_cost > min(known_cost of other eligible candidates)
```

The boolean is undefined (`None`) when fewer than two known prices exist.

## Underrouting

The selected deployment failed a required capability that another ranked
eligible deployment would have satisfied.

```text
underrouting = selected lacks required capability
               AND another eligible candidate has it
```

Undefined when nothing was selected.

## Regret (only with known values)

```text
cost_regret = selected_cost - min(cost of comparably priced alternatives)
```

Computed only when **both** terms are known. Latency and quality regret use the
same rule on declared TTFT / declared quality means. Missing values stay
`null`.

## Other rates

| Metric | Definition |
| --- | --- |
| `route_success_rate` | Fraction of plans that selected a deployment |
| `fallback_rate` | Fraction with recorded failover_count > 0 (needs execution records) |
| `escalation_rate` | Fraction whose exclusions include preferred-tier narrowing |
| `capability_mismatch_rate` | Fraction with a missing-capability exclusion |
| `policy_rejection_rate` | Deny list, provider allow-list, or tenant ceiling exclusion |
| `context_rejection_rate` | Context window exclusion |
| `provider_unavailable_rate` | Open-circuit exclusion |

Shadow comparison (`LLM_FABRIC_ROUTING_QUALITY_SHADOW=true`, default **false**)
records quality-first vs the live policy **without changing the served route**.

`llm-fabric route simulate --candidate-policy other.yaml` compares two YAML
policies offline (no provider call).
