# Context compiler

The context compiler runs on every `/v1/chat/completions` request after
IntentOS and before the Route Planner. There is no bypass.

```
Auth → Tenant Policy → Input Guardrails → IntentOS → Context Compiler
  → Route Planner → Provider Adapter → Runtime → Output Validation
```

## ContextRecord

Every eligible provider invocation (`USER_RESPONSE`) stores
`context_record_id` on the usage event. Coverage invariant:

`provider_invocations_without_context_record = 0`

Absent block types are counted as **0**. A model window that is not yet known
is **UNAVAILABLE**, never guessed as zero.

## Token provenance

Every count identifies one of:

| Provenance | Meaning |
| --- | --- |
| `PROVIDER_MEASURED` | Backend reported the value |
| `TOKENIZER_MEASURED` | An exact tokenizer counted it |
| `DERIVED` | Computed from other trustworthy fields |
| `ESTIMATED` | Approximate local counter (`approximate-chars-v1`) |
| `UNAVAILABLE` | Unknown. Carries no numeric value |

Zero is a measurement. Unknown is `UNAVAILABLE`. A configured scrape that goes
silent is `METRIC_PIPELINE_BROKEN` and also carries no value.

## Stable prefix

`stable_prefix_tokens` / `volatile_prompt_tokens` describe **prompt shape**
(authoritative blocks first). They are not evidence that a runtime reused KV
blocks. Runtime prefix-cache hits come from vLLM `/metrics` when scraped.

## TPS

There is no single TPS number. Named formulas in
`src/llm_fabric/observability/tps.py`:

| Name | Equation | Scope |
| --- | --- | --- |
| `prefill_tokens_per_second` | `prompt_tokens / prefill_duration_s` | REQUEST |
| `decode_tokens_per_second` | `completion_tokens / decode_duration_s` | REQUEST |
| `aggregate_generation_tokens_per_second` | `completion_tokens / generation_duration_s` | REQUEST |
| `request_effective_tokens_per_second` | `(prompt + completion) / e2e_duration_s` | REQUEST |

Gateway end-to-end time is **not** decode duration. Histogram means from vLLM
`/metrics` are **DEPLOYMENT**, not this request's TTFT/TPOT.

Request TTFT is gateway stream first-byte. Request TPOT is
`(now - first_byte) / max(1, completion_tokens - 1)` on streaming responses.
No per-token OTEL spans.

## Scope

| Scope | Example |
| --- | --- |
| REQUEST | compiled tokens, stream TTFT |
| MODEL | unused here |
| POD | DCGM GPU series when scraped |
| DEPLOYMENT | vLLM KV utilisation, running/waiting |
| FLEET | not synthesized from one pod |

Never display pod KV utilisation as "this request used X% KV cache".

## Compression and summarization

Hooks exist. They do not run unless a compressor is supplied. Related counters
are **0** with a note, not UNAVAILABLE.
