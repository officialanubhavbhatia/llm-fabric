# Cost model

API token prices on a registry entry have three states. They are not mixed.

| YAML | Meaning | Ranking |
| --- | --- | --- |
| field omitted or `null` | **unknown** | Excluded from cost ranking. Not treated as free. |
| `0.0` | **known-zero** | A declared $0 API price (typical for self-hosted Ollama/vLLM). Ranked as cheapest API cost. |
| `> 0` | **known-nonzero** | Ranked against other known API prices. |

Both `input_cost_per_mtok` and `output_cost_per_mtok` must be present before a
deployment participates in cost ranking. A missing side is not imputed as zero.

**One unknown model does not drop cost for the whole fleet.** Cost is scored
only among candidates with known API prices. Unknown-cost candidates receive no
cost contribution and their remaining weights are not renormalised to fill that
slot, so they cannot win `cost_first` by looking free.

Optional `cost_class` records resource semantics the operator actually knows:

- `marginal_api_price`
- `resource_cost_known`
- `resource_cost_unknown`
- `estimated_gpu_cost`
- `unknown`

`estimated_compute_cost_per_hour_usd` is used only when the operator supplies
it. The fabric does not invent GPU-hour dollars from a provider name.

Missing features stay missing. See `src/llm_fabric/router/policy.py`.
