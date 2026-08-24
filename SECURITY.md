# Security

This document describes the security properties the fabric **currently
enforces**, and — just as importantly — the ones it does not. Anything not
listed as built is not built. Do not infer a control from its absence here.

Scope: Phase 2 (identity and tenancy). Guardrails, prompt-injection defence,
retrieval-time authorization and output filtering are later phases and are
**not implemented**.

---

## 1. Reporting a vulnerability

Report privately to the repository owner (Anubhav Bhatia) via a GitHub security
advisory on [`officialanubhavbhatia/llm-fabric`](https://github.com/officialanubhavbhatia/llm-fabric/security/advisories).
Please do not open a public issue for a suspected isolation failure.

Anything that allows one tenant to observe another tenant's data, spend or
traffic is treated as the highest severity, regardless of how contrived the
path is.

---

## 2. Identity

A caller is identified before any route runs. `AuthenticationMiddleware` sits in
front of routing, so an unauthenticated request cannot reach an endpoint, and
cannot map which endpoints exist.

### Environments

| `LLM_FABRIC_ENVIRONMENT` | Authentication |
| --- | --- |
| `development` (default) | Anonymous access only if `LLM_FABRIC_ALLOW_ANONYMOUS=true` (the default) and no identity source is configured |
| `test` | Same rules as development |
| `production` | Authentication is mandatory. Incomplete, invalid or development-only identity configuration refuses to start. The process does not bind a port. |

Production rejects `LLM_FABRIC_ALLOW_ANONYMOUS`, `LLM_FABRIC_DEV_AUTH_SECRET`,
`auth_mode=disabled`, `auth_mode=dev`, incomplete OIDC, empty API-key mode, and
the multi-worker escape hatch. The log line `gateway is running without
authentication` is emitted only in development or test; in production that
state is a `ConfigurationError` and startup terminates.

`python -m llm_fabric doctor` (or `make doctor`) reports PASS / WARN / FAIL for
these prerequisites without starting the server.

### Supported identity sources

| Mode | Source | Intended use |
| --- | --- | --- |
| `oidc` | OAuth2/OIDC bearer token, validated against the issuer's JWKS | Production |
| `api_key` | Static key bound to a tenant by the operator | Machine callers that cannot hold an OIDC flow |
| `dev` | Local development issuer, HS256 | Local development only — refused in production |
| `disabled` | No authentication | Local experimentation only — refused in production |

The mode is inferred from whichever credentials are configured. Inference is a
development convenience. Production fail-closed validation is
`validate_startup` in `src/llm_fabric/config.py`.

### What is enforced on a token

- **Signature**, against a key published by the pinned issuer.
- **Algorithm allowlist.** Only `RS256`, `RS384`, `RS512`, `ES256` and `ES384`
  are accepted. Symmetric algorithms are refused outright, and the algorithm is
  checked *before* any key is fetched. A JWKS is public, so accepting `HS256`
  would let anyone sign a token using the published key material as an HMAC
  secret and mint any identity they liked.
- **`alg: none`** is refused by the same allowlist.
- **Issuer** and **audience** are both pinned. An unpinned audience makes any
  token the issuer ever minted replayable against this gateway.
- **Expiry** and **not-before**, with bounded clock leeway.
- **Required claims**: `sub`, `iss`, `aud`, `exp`, and a tenant claim.
- **Scopes**, when `LLM_FABRIC_REQUIRED_SCOPES` is set: a token missing any of
  those scopes is refused.
- **Revocation**: `jti` and a SHA-256 fingerprint of the presented credential
  are checked against a denylist. See [`docs/AUTH_REVOCATION.md`](docs/AUTH_REVOCATION.md).

A token that authenticates a person but carries no tenant is **refused**.
Defaulting the tenant would silently merge customers.

### Key handling

- JWKS responses are cached, and refresh is rate limited. An unknown `kid` is
  attacker-controlled input, so unbounded refresh would turn a stream of junk
  tokens into a traffic amplifier aimed at the identity provider. A rotated
  key is accepted after the cache expires. Production OIDC **warms JWKS at
  startup**; an unreachable issuer is a start failure, not a first-request 401.
- Static API keys must be at least 16 characters. A static key is a bearer
  credential with no expiry, so it has to carry its entropy on its own.
- Key comparison is constant-time and cannot raise on malformed input. All
  configured credentials are compared even after a match, so response time does
  not reveal which key matched or where it sat in the list.

---

## 3. Tenant isolation

Isolation is enforced **twice**, by two independent mechanisms.

1. **Namespacing.** Records live in a per-tenant partition and cache keys are
   fingerprinted with the tenant mixed in. One tenant's key does not exist in
   another tenant's namespace, so there is no cross-tenant query to get wrong.
2. **Ownership re-check.** Every record carries its owning `tenant_id`, and
   every read asserts it matches the requesting scope. A mismatch raises
   `TenantIsolationError`, an internal error that is recorded in an audit log.

The second check is redundant while the first is correct. That is precisely why
it is kept: the first is the one a future refactor can quietly break. Both are
verified by mutation testing — breaking either one alone is caught by the suite.

### The mechanism

No repository accepts a bare tenant id. Every read and write takes a
`TenantScope`, which can only be constructed from an authenticated `Principal`.
Tenant isolation fails in practice when the tenant is an optional argument
somebody forgets, so it is not an argument that can be forgotten.

### Cross-tenant reads report absence, not denial

A request for another tenant's record returns **404, never 403**. A 403 would
confirm the identifier exists somewhere, turning the endpoint into an oracle for
enumerating another tenant's record ids.

### Tenant delegation

`X-Tenant-Id` lets a caller act for another tenant. It is honoured **only** when
the validated token carries the `fabric:delegate_tenant` scope; for every other
caller it is ignored and the request is refused. When delegation is exercised,
the original tenant is preserved in `delegated_from` so the audit trail
distinguishes real identity from assumed identity.

A client-supplied tenant id is never trusted on its own.

### What is covered

Tenant-scoped storage exists for conversations, traces, intent examples, prompt
definitions, evaluation datasets, usage records, and all seven cache namespaces.
Each is covered by the adversarial suite.

---

## 4. Quotas

Per-tenant and per-user ceilings are enforced on requests per minute, requests
per day, tokens per day, and spend per day.

Both levels are enforced, because a tenant-wide limit does not protect one user
from another inside the same tenant, and a per-user limit does not bound the
tenant in aggregate.

Quotas are also the fabric's denial-of-wallet defence: an inference gateway with
no ceiling turns a leaked client key directly into an unbounded bill.

Token and spend ceilings apply to the *next* admission, not retroactively — a
request that has already run cannot be un-run.

**Limitation:** the ledger is in-process. Limits therefore apply **per replica**,
not per fleet. A distributed ledger is not built.

---

## 5. Privacy in telemetry

- Credentials are never logged. Only a truncated, non-reversible SHA-256
  fingerprint of an API key leaves the identity module.
- `Principal.audit_fields()` returns a closed set of identifiers and no claim
  the fabric does not need. Scopes and roles are excluded.
- Tenant identity is attached to trace context for operator-visible telemetry,
  but **not** to the outbound `traceparent` header. Tenant ids do not belong in
  a header that proxies and downstream systems will log.
- An inbound `traceparent` is untrusted input: malformed values are dropped and
  a fresh trace is started rather than propagating a value that corrupts the
  trace graph.

**Not built:** configurable retention, payload redaction policies, and
prompt/response storage controls.

---

## 6. Verification

The adversarial suite lives in `tests/security/` and is marked `isolation`.

```bash
make test-isolation
```

It runs as its own CI job, `Tenant isolation (cross-tenant leak gate)`, so a
cross-tenant failure is legible on the checks list rather than buried in a
general test log. Because `pytest -m isolation` exits non-zero when it collects
nothing, deleting or renaming the marker fails CI rather than silently
disabling the gate.

Each attack is paired with a control assertion proving the victim can still
reach their own data. Without the control, a fixture that silently stored
nothing would make the suite pass while proving nothing.

### Attacks covered

Against conversations, traces, intent examples, prompt definitions, evaluation
datasets and every cache namespace:

- direct read by another tenant's identifier
- enumeration through listing, key iteration and counting
- deletion and namespace-wide invalidation of another tenant's records
- writing a record owned by another tenant (poisoning, and mislabelled exfiltration)
- collision when two tenants share a user id
- cache collision when two tenants send byte-identical inputs
- noisy-neighbour quota exhaustion

Against identity:

- forged signature, unsigned (`alg: none`), and tampered payload
- algorithm confusion (symmetric token against an asymmetric verifier)
- expired token, not-before, wrong audience, wrong issuer
- missing or empty tenant claim, tampered tenant in the payload
- revoked `jti` / fingerprint
- insufficient required scope
- unknown signing key
- tenant spoofing via `X-Tenant-Id` without the delegation scope
- malformed input that must produce 401 rather than a 500

Against routing, which reads tenant configuration and reports internal state:

- previewing a route as another tenant, by supplying a tenant id in the body
- reading another tenant's routing policy, deny list or locality restriction
- escaping a pinned tenant policy through a policy override, a pinned model, a
  grade floor or a budget
- reaching another tenant's prompts or traffic through a route explanation
- attributing a request to a tenant through the fleet-health endpoint
- billing a tenant by driving the preview endpoint, which performs no inference

---

## 7. Known gaps

These are **not implemented**. They are listed so nobody mistakes silence for a
guarantee.

| Gap | Consequence |
| --- | --- |
| Storage is in-memory | Nothing survives restart; isolation guarantees are not yet proven against a real database |
| No Postgres row-level security | The second line of defence a database can provide is absent |
| Quota ledger is per-process | Limits apply per replica, not per fleet |
| Revocation denylist is per-process | A revoke on one worker is invisible to the others; restart forgets the list. Stateless JWTs cannot be revoked without a shared store. See `docs/AUTH_REVOCATION.md`. |
| No rate limiting on authentication failures | Credential stuffing is not slowed down |
| No audit log persistence | Isolation violations are counted in memory only |
| No guardrails engine | No prompt-injection, PII or output-safety controls |
| No retrieval-time authorization | Not yet applicable; no retrieval exists |
| No secrets manager integration | Credentials come from environment variables |
| No mTLS or network policy | Transport security is deployment-provided |
| Route preview exposes fleet configuration | Any authenticated caller can enumerate model ids, capabilities, localities and registry prices. No other tenant's policy or traffic is reachable, but the fleet's shape is not a secret from a tenant. |
| Circuit-breaker state is per-process | A deployment tripped out on one replica stays in rotation on the others |

---

## 8. Operational guidance

- Set `LLM_FABRIC_ENVIRONMENT` to `development`, `test`, or `production`.
  Unset, empty, and unknown values refuse to start. There is no implicit
  development default. Run `python -m llm_fabric doctor` before serving.
- Never set `LLM_FABRIC_ALLOW_ANONYMOUS` or `LLM_FABRIC_DEV_AUTH_SECRET` in
  production. Either one refuses startup.
- Production `LLM_FABRIC_DATABASE_URL` must be the DML role `fabric_app`
  (USAGE on `public`, table SELECT/INSERT/UPDATE/DELETE, no `CREATE` on
  schema, no `CREATEDB` / `CREATEROLE` / `BYPASSRLS`). Migrations use
  `LLM_FABRIC_MIGRATION_DATABASE_URL` as the table-owner role. Do not point
  gateway workers at the migration role.
- Production startup probes PostgreSQL and Redis from `initialize_runtime`,
  which `create_app` always runs. `uvicorn --factory` cannot skip it. A missing
  or unreachable DSN refuses serving. Probe errors do not include credentials.
- Never set `LLM_FABRIC_DEV_AUTH_SECRET` in a production environment. The
  development issuer mints any identity it is asked for. It refuses a secret
  shorter than 32 characters and the gateway warns on every startup while it is
  enabled.
- Prefer `LLM_FABRIC_API_CREDENTIALS` over `LLM_FABRIC_API_KEYS`. The legacy
  flat list has no tenant of its own and places every key in a single `default`
  tenant.
- Grant `fabric:delegate_tenant` to operator identities only.
- Set per-tenant and per-user quotas before exposing the gateway to untrusted
  callers.
- Declare `locality` on every deployment in `config/models.yaml`. An entry that
  omits it is treated as `external`, which is the safe default but will exclude
  it from tenants restricted to local or private serving.
- Use a tenant routing policy, not a client-side convention, to keep a regulated
  tenant off external deployments. A policy pinned on the tenant cannot be
  overridden by anything the caller sends.
