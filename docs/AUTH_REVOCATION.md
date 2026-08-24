# Token revocation

A signed JWT is valid until `exp`. The issuer cannot take it back, and the
gateway cannot know it has been compromised unless it consults a denylist on
every request. That is a property of stateless bearer tokens, not a missing
flag.

## What this build does

Every accepted credential is checked against a revocation store after (and,
for fingerprints, before) cryptographic verification.

| Key | Meaning |
| --- | --- |
| `jti` | Issuer-assigned token id, stored on `Principal.token_id` |
| fingerprint | SHA-256 of the presented bearer string (JWT or API key) |

An operator revokes by `jti` when the issuer assigned one, or by fingerprint
when they hold the stolen credential. Entries may carry `expires_at` so the
denylist cannot grow past the natural lifetime of the tokens it tracks.

API keys have no `exp`. Revoking one records its fingerprint until an operator
removes the entry or the process restarts. The in-memory store caps at 100,000
entries and evicts the oldest if that cap is hit.

## What this build does not do

The store is **process-local** (`InMemoryRevocationStore`). A revoke on worker
A is invisible to worker B. A process restart forgets the denylist. Tokens
that were revoked only in memory become valid again.

A Redis- or Valkey-backed denylist is planned with distributed state. Until
then, immediate fleet-wide revocation is **not implemented**. Compensating
controls that *are* implemented:

- short token lifetimes (`exp`)
- JWKS rotation: an unknown `kid` is refused; a rotated key is accepted after
  a bounded cache refresh
- production refuses to start without a complete identity source

## Stateless JWT limitation

Revoking a JWT without a shared denylist (or issuer introspection) is
impossible. Short `exp` plus key rotation limits the window. Immediate
revocation requires distributed storage and is not claimed here.

Do not treat an in-memory denylist as a session system. It is a per-process
tripwire for tests and single-worker development. Production refuses to start
with more than one worker until a shared store exists.
