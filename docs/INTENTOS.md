# IntentOS

Routing intelligence for MyVista LLM Fabric. IntentOS classifies a request
into a structured result the route planner can consume. It is **not** a single
prompt that says "classify this request".

This document describes what is built. Measured scores live in
[`INTENTOS_EVALUATION.md`](../INTENTOS_EVALUATION.md). The frozen success bar
is [`INTENTOS_SUCCESS_CRITERIA.md`](INTENTOS_SUCCESS_CRITERIA.md). Dataset
hygiene is [`INTENT_DATASET_REPORT.md`](../INTENT_DATASET_REPORT.md).

Nothing here claims competitor comparisons. Nothing here claims multilingual
production quality. Serving-path classification is **off by default**
(`LLM_FABRIC_INTENT_CLASSIFICATION_ENABLED`).

## Architecture

```
Request (bounded head+tail, optional conversation signature)
  ↓
L0  exact intent cache          tenant + versions + normalized text
  ↓ miss
L1  semantic intent cache       same discriminators, cosine + confidence
  ↓ miss / low confidence
L2  deterministic rules         versioned, disable-able, high-precision only
  ↓ uncertain
L3  embedding nearest-centroid  default: lexical HashingEmbedder;
                                optional RealLocalEmbedder (MiniLM / bge-small)
  ↓ uncertain
L2∩L3 agreement                 same intent, floor, runner-up guard
  ↓ still uncertain
L4  local mixed-prototype rerank or structured model (sampled)
  ↓ uncertain
L5  escalation model            off in this phase
  ↓
ABSTAIN / unknown
```

A layer may terminate only when its calibrated threshold is met. Classifier
failures never fail the chat request: the cascade continues or abstains, and
the router uses a balanced capability floor rather than the cheapest alias.

Classification **proposes** requirements. Tenant policy, pinned models, and
allow-lists **decide** what is permitted. A label never grants authorization.

## Schema

`IntentClassification` in `src/llm_fabric/intent/schema.py` is the contract.
Fields include domain / task / subtask, complexity, reasoning level, required
capabilities, modality, agent/tools/retrieval/structured-output flags, context
class, latency/quality/cost/privacy/safety classes, language, confidence,
alternatives, secondary intents, abstain, layer, versions, cache source, and
abstract `minimum_capability_grade` / `recommended_quality_grade`.

Those grade hints are capability bands. They are never model names.

Class names follow the constitution, not marketing synonyms:

| Dimension | Values |
| --- | --- |
| complexity | trivial, simple, moderate, complex, very_complex |
| context | tiny, short, medium, long, very_long |
| latency | realtime, interactive, standard, batch |
| quality | draft, standard, high, maximum |

Dimensions are filled from the taxonomy profile, then lightly overridden by
cheap request features (constraint density, code fences, conversation
signature). They do not all come from one classifier.

## Taxonomy

Version `bootstrap-2026.08.1`. Published snapshots are immutable
(`PublishedTaxonomyStore` refuses overwrite; `TaxonomyRegistry` refuses
re-register). Historical versions are not mutated in place.

Domains, each present because it changes routing:

```
coding (+ debug, review)
agent, reasoning, math (+ arithmetic)
research, rag, data_analysis
writing, summarization, translation
extraction, classification, vision
tool_use, general_conversation
```

Children exist only where a routing difference is useful (`coding.debug` vs
`coding.review`, `math.arithmetic`). Further fragmentation was not added.

Each node carries examples, counterexamples, hard negatives, required
capabilities, and default quality / latency / context classes. The published
JSON is `datasets/intent/taxonomy/bootstrap-2026.08.1.json`.

v1 is a **global** taxonomy plus tenant policy. Per-tenant custom intents are
not implemented.

## Cascade and thresholds

Thresholds fall as the cascade deepens: the price of skipping better layers
below.

| Layer | Stop threshold |
| --- | --- |
| L1 semantic cache | 0.80 |
| L2 rules | 0.70 |
| L3 embedding | 0.62 |
| L2∩L3 agreement | 0.48, same intent, runner-up < 0.18 |
| L4 structured | 0.55 |
| L5 escalation | 0.40 |

Agreement without the runner-up guard would happily label multi-intent
prompts. That guard is load-bearing.

Input to every layer is bounded (`bound_text`, 4 000 characters, head+tail).
Rules scan the same bound.

## Caching

Exact and semantic caches are separate. Both key on:

```
tenant_id, taxonomy_version, classifier_version,
policy_version, language, conversation_state_signature,
normalized request
```

`classifier_version` digests every layer, so a rules or embedder change
invalidates both caches. The semantic index is partitioned by that
discriminator signature: similarity is never compared across tenants or
taxonomy versions.

A semantic hit needs minimum similarity **and** minimum historical
confidence. Abstentions are never cached. False hits are a reviewed-hit
metric; dividing by all hits would assume unreviewed hits were correct.

## Deterministic rules (L2)

Weighted regex with negative penalties for hard negatives. Versioned
(`rules-5`), observable via layer attempts, disable-able (`enabled=False`).
Injection-only prompts and privilege-coercion ("classify this as X so I get
Y") return no opinion rather than the requested label. A matched child
suppresses its parent in the ranking so evidence is not split.

Rules are used only where precision is high. They are not a giant grammar.

## Embedding classifier (L3)

Nearest centroid over taxonomy examples. Default embedder is
`HashingEmbedder` — **lexical hashing, not meaning**. A real embedding model
plugs the same interface. Weak similarity, a thin margin, or an unknown-floor
cosine continues the cascade instead of taking the nearest label.

## L4 / L5

Structured JSON against a shortlist of candidates, not the whole taxonomy.
Hallucinated ids are discarded. The prompt tells the model to ignore
in-prompt classifier-override instructions. L5 runs only when cheaper layers
disagree, confidence is low, the request looks multi-intent, or OOD is
suspected. Both layers are off unless a provider is configured. Offline
evaluation does not call them.

## Abstention and unknown / OOD

If no layer clears its bar the result is:

```json
{ "domain": "unknown", "task": null, "abstain": true, "confidence": 0.41 }
```

The rejected best guess is retained on `alternatives`. Routing then uses
`RoutePolicy.BALANCED` as the capability floor, not cheapest. Intent
capability extras that would empty the fleet are dropped rather than turning
classification into a 503.

An explicit OOD slice lives in `datasets/intent/ood.jsonl`.

## Multi-intent and conversation

Primary intent plus `secondary_intents` when a second domain is visible.
IntentOS does not plan a workflow.

Conversation-aware classification consumes a **signature** of recent turns,
not the full history. `conversation_aware` records which path ran.

## Tenant policy and isolation

Caches, classification records, and hard-example stores are
`TenantScopedCache` / `TenantScopedStore`. Tenant A cannot read, overwrite,
or invalidate Tenant B. Tests: `tests/security/test_intent_isolation.py`.

Classification cannot widen providers, grades, tools, or privacy. Tenant
deny-lists and allow-lists still win.

## Versioning

Every result stores taxonomy, classifier, embedding model, prompt (if L4/L5),
and policy versions. Durable records store a prompt **hash**, not the prompt,
unless a caller explicitly opts in.

## Evaluation and gates

`llm-fabric-bench` scores a frozen JSONL set. CI runs both a regression floor
and an absolute floor (`datasets/eval/ci-suite.yaml`): accuracy, macro-F1,
unknown-intent recall, and high-confidence precision at 0.90.

The IntentOS v1 success bar was written **before** classifier changes, in
`docs/INTENTOS_SUCCESS_CRITERIA.md`. That baseline file is not rewritten.

## Learning loop

```
production classification
  → low confidence / disagreement / abstain / route failure
  → candidate (redact, hash, dedup)
  → tenant-scoped hard-example store (draft)
  → human review
  → offline eval / shadow / canary / promotion   ← not automatic
```

`promotion_blocked_reason` refuses promotion without review and passing eval
gates. There is **no** live self-training.

Shadow classification samples traffic, records production vs candidate, and
never returns the candidate to the user. Expensive layers are not double-called
unless sampling is configured to allow it. Serving-path shadow
(`LLM_FABRIC_INTENT_SHADOW=true` with classification still off) classifies
and emits `x-fabric-intent-shadow-*` headers without changing the route.

## Failure behaviour

| Failure | Behaviour |
| --- | --- |
| semantic cache down | skip L1, continue |
| embedder down | skip L3, continue |
| L4/L5 provider down | skip, continue or abstain |
| cascade exception on chat | log, serve without intent |
| abstain / unknown / conf < 0.50 | balanced capability floor |
| intent extras no candidate | drop extras, route on hard requirements |

## Metrics

Prometheus names (bounded labels only — no user id, request id, or raw text):

```
fabric_intent_classifications_total
fabric_intent_abstentions_total
fabric_intent_unknown_total
fabric_intent_cache_hits_total
fabric_intent_classifier_latency_seconds
fabric_intent_escalations_total
fabric_intent_disagreements_total
```

Command Center `intents` view shows cascade counters: distribution by layer,
confidence histogram, abstention, unknown, cache hits, latency. It does not
show accuracy; that requires labels.

## Preview contract

```
POST /v1/intents/classify
```

Auth and tenant rules are the same as the rest of `/v1`. The Python SDK
exposes `client.intents.classify(...)`. Chat, when classification is enabled,
echoes `x-fabric-intent`, `x-fabric-intent-confidence`,
`x-fabric-intent-layer`, `x-fabric-intent-cache`,
`x-fabric-taxonomy-version`, `x-fabric-classifier-version`.

## Privacy

Traces do not carry the prompt. Durable classification records default to a
hash. Hard-example ingest redacts emails, token-shaped spans, and phone
numbers before storage. Production user content is not an unsupervised
training corpus.

## Route-regret hook

The classification already carries quality / latency / cost classes and
abstract grade hints so a future route evaluator can ask whether the served
grade was expensive, underpowered, or slow. That evaluator is **not** built.
The 30-grade route planner is not this phase.
