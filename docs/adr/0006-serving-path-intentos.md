# ADR 0006 — Serving-path IntentOS is mandatory in production

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

1. Production refuses to start unless serving-path IntentOS is enabled.
   Development and test may disable the cascade explicitly; they still attach
   a `SAFE_FALLBACK` IntentResult so `provider_invocations_without_intent`
   stays 0.
2. `IntentClassification` carries `serving_state`: `known`, `unknown`,
   `abstain`, `safe_fallback`, plus `intent_result_id`.
3. Chat always produces an IntentResult before `Router.complete` / `stream`.
   Cascade or dependency failure degrades to another layer or SAFE_FALLBACK;
   it never continues with `intent is None`.
4. High-confidence known intents may use optimized route policy. Medium,
   low, unknown, abstain, and safe-fallback use a balanced capability floor,
   never cheapest-by-uncertainty.
5. HashingEmbedder remains the deterministic test embedder. Production may
   select a real local embedder (`local` / `minilm`). Lexical hashing in
   production requires an explicit allow flag.

## Consequences

- Public SDK chat/completions shape is unchanged; provenance headers gain
  serving-state / intent result id.
- Hard-negative accuracy gate stays 0.58 until a measured run clears it.
- L4/L5 classifier provider calls remain classification-internal and are not
  USER_RESPONSE ledger rows.
