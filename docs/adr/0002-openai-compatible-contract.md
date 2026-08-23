# ADR 0002 — An OpenAI-compatible public contract

**Status:** Accepted, provisionally. Superseded by the engineering constitution if
it specifies a different surface.

## Context

The gateway needs a request and response schema. The fabric's value proposition is
that callers depend on the fabric rather than on any provider, which makes the
choice of public dialect the most consequential interface decision in the system:
it is the one thing that cannot be changed later without breaking every client.

## Decision

Speak the OpenAI chat-completions dialect, at `/v1/chat/completions` with
`/v1/models` for discovery, including SSE streaming in the same wire format.

Accept fields the fabric does not act on rather than rejecting them.

## Rationale

**Adoption cost is the whole point.** Every major LLM client library, framework,
and tool already speaks this dialect. Pointing an existing client at the fabric
becomes a base-URL change instead of a rewrite, which is the difference between a
gateway that gets adopted and one that gets bypassed.

**Rejecting unknown fields would break clients for no benefit.** SDKs send fields
the fabric has no opinion on. Refusing them would surface as a client error for a
request the fabric could serve perfectly well, so unhandled fields are accepted
and documented as inert in `docs/CONTRACT.md` rather than silently pretended to
work.

**Aliases extend the dialect without violating it.** A fabric-specific concept —
`auto`, resolved by policy — is expressed as a model id, which every client can
already send. No client change is needed to use the fabric's routing, and
`x-fabric-served-model` tells the caller what they actually got.

## Alternatives

**A bespoke contract.** Cleaner room to express fabric-native concepts like
routing policy and cost ceilings as first-class request fields. Rejected: it
imposes a client rewrite on every caller, which defeats the purpose. Fabric-native
concepts are carried in `x-fabric-*` headers and the `x_fabric` block of the final
streamed chunk, where they extend the dialect additively.

## Consequences

The fabric is constrained by someone else's schema evolution, and features without
a natural expression in the dialect need a header or an extension block. That is
accepted in exchange for zero-cost client adoption.

`contract/` is kept structurally separate from the layers behind it, so replacing
the dialect touches one package rather than the whole gateway.
