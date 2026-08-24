# IntentOS serving-path audit

**Status:** Phase 2 inventory, written before the serving-path changes in this
phase. Implementation closed the gaps below. This file remains the audit of
what the tree looked like *before* those changes.

**Current implementation:** IntentOS classification now runs on every chat in
every environment. `LLM_FABRIC_INTENT_ROUTING_ENABLED` separately controls
whether the planner consumes the classification. Catastrophic cascade failure
uses `SAFE_FALLBACK`; no provider invocation proceeds without an IntentResult.

**Constitution:** [`docs/constitution.md`](constitution.md) § IntentOS and
§ Non-negotiable architecture (Intent engine before Route planner).

**CONSTITUTION AMENDMENT REQUIRED:** NO  
**ADR REQUIRED:** YES — clarification only (`docs/adr/0006-serving-path-intentos.md`).
The constitution already places IntentOS on the synchronous path. The tree had
turned it off. This phase implements the constitution; it does not change it.

**Coverage vs accuracy:** `provider_invocations_without_intent = 0` is a
**coverage** invariant. UNKNOWN / ABSTAIN / SAFE_FALLBACK are valid IntentResults.
This is not a claim of 100% classification accuracy.

---

## Constitution sections

| Topic | Section |
| --- | --- |
| Request path includes Intent engine before Route planner | Non-negotiable architecture |
| Typed classification fields | IntentOS |
| Cascade L0–L5 then UNKNOWN/ABSTAIN | IntentOS |
| Unknown intent is valid | IntentOS |
| Cache key discriminators | Intent cache |
| Intent confidence as a metric | Observability |

Phase 2 does **not** start the context compiler, Command Center UI redesign,
or automatic classifier promotion.

---

## CURRENT (before this phase)

Serving-path classification is **off by default**
(`LLM_FABRIC_INTENT_CLASSIFICATION_ENABLED=false`). Chat may invoke a provider
with `RouteRequest.intent is None`. Cascade exceptions on chat are logged and
the request continues **without** an IntentResult.

Helm ConfigMap hard-codes `LLM_FABRIC_INTENT_CLASSIFICATION_ENABLED: "false"`.

---

## Paths that can invoke inference without IntentOS

| # | Path | Mechanism | Eligible production serving? |
| --- | --- | --- | --- |
| 1 | `POST /v1/chat/completions` buffered | `intent_classification_enabled=false` → `_classify` unused for routing | **yes** |
| 2 | Same, streaming | Identical flag; SSE does not classify | **yes** |
| 3 | Chat with cascade built for shadow only | `intent_shadow=true` and classification off: classifies for headers, **does not** attach intent to `RouteRequest` | **yes** |
| 4 | Chat `_classify` exception | `except Exception: return None` | **yes** |
| 5 | Chat empty latest user turn | `_classify` returns `None` even when enabled | **yes** |
| 6 | `create_app` with both flags false | `app.state.intent = None`; `get_intent_cascade` is optional | **yes** |
| 7 | Helm / Compose | ConfigMap and local values force the flag off | **yes** (if production used those values) |
| 8 | `Router.complete` / `Router.stream` | `RouteRequest.intent` optional; tests and CLI call this directly | tests / internal |
| 9 | `Router.resolve` for SSE header planning | Plans without intent | planning only |
| 10 | Fallback / retry hops | Same `RouteRequest`; if intent was None at start, every hop lacks it | **yes** if 1–5 |
| 11 | Direct Ollama / vLLM / LiteLLM adapters | No IntentOS of their own; they inherit the route | **yes** if chat skipped IntentOS |
| 12 | `models.probe` / `models.eval` / eval judge | Call `provider.generate` offline | **no** (not serving-path USER_RESPONSE) |
| 13 | L4/L5 structured classifier | Calls `provider.generate` **during** classification | classifier-internal, before route |
| 14 | Load bench / `llm-fabric-load` | Hits chat; inherits gateway flag | **yes** if flag off |
| 15 | Intent `/v1/intents/classify` | Classifies only; no chat provider call | n/a |

There is **no** provider-specific or transport-specific IntentOS bypass inside
adapters. Bypass is always “the router ran without an IntentResult”.

---

## DESIRED

1. Production refuses to start with classification disabled.
2. Every USER_RESPONSE provider invocation carries a typed IntentResult
   (`KNOWN` \| `UNKNOWN` \| `ABSTAIN` \| `SAFE_FALLBACK`).
3. Cascade or layer failure never skips classification: degrade to another
   layer or SAFE_FALLBACK / UNKNOWN / ABSTAIN.
4. Route aggressiveness follows confidence; uncertain results use a safe
   capability floor, not cheapest.
5. `provider_invocations_without_intent = 0` on the usage ledger.

---

## GAPS (closed by this phase unless noted)

1. Production can start with IntentOS off.
2. Chat may route with `intent is None`.
3. Cascade exceptions skip IntentOS.
4. Serving classification state is not distinct (UNKNOWN conflated with ABSTAIN).
5. Usage events lack `intent_result_id` / taxonomy / classifier versions.
6. Confidence policy infers QUALITY_FIRST below a high-confidence bar.
7. Semantic cache is process-local (L0 can already use Redis via `TenantScopedCache`).
8. Coverage metrics named in the phase brief are incomplete.
9. Command Center `intents` view still reports routing OFF.

**Not in this phase:** context compiler, Command Center UI redesign, fake GPU
embedding benchmarks, lowering the hard-negative gate (0.58).
