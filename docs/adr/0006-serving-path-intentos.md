# ADR 0006 — Serving-path IntentOS is mandatory

**Status:** Accepted  
**Date:** 2026-08-25  
**Constitution:** does not amend [`docs/constitution.md`](../constitution.md)

## Context

The constitution places the Intent engine on the synchronous request path
before the Route planner. The implementation defaulted
`LLM_FABRIC_INTENT_CLASSIFICATION_ENABLED` to false so chat could run before
HTTP classification cost was measured. That left production able to invoke
providers with no IntentResult.

Unknown / abstain are constitutionally valid. Coverage is not accuracy.

## Decision

1. Every chat request in every environment runs the IntentOS cascade. The
   compatibility setting `LLM_FABRIC_INTENT_CLASSIFICATION_ENABLED` cannot
   disable it.
2. `IntentClassification` carries `serving_state`: `known`, `unknown`,
   `abstain`, `safe_fallback`, plus `intent_result_id`.
3. Chat always produces an IntentResult before `Router.complete` / `stream`.
   Cascade or dependency failure degrades to another layer or SAFE_FALLBACK;
   it never continues with `intent is None`.
4. Classification and route influence are separate. The cascade always runs;
   `LLM_FABRIC_INTENT_ROUTING_ENABLED` determines whether the planner consumes
   the result. High-confidence known intents may use optimized route policy. Medium,
   low, unknown, abstain, and safe-fallback use a balanced capability floor,
   never cheapest-by-uncertainty.
5. HashingEmbedder remains the deterministic test embedder. Production may
   select a real local embedder (`local` / `minilm`). Lexical hashing in
   production requires an explicit allow flag.

## Consequences

- Public SDK chat/completions shape is unchanged; provenance headers gain
  serving-state / intent result id and explicit route-influence state.
- Hard-negative accuracy gate stays 0.58 until a measured run clears it.
- L4/L5 classifier provider calls remain classification-internal and are not
  USER_RESPONSE ledger rows.
