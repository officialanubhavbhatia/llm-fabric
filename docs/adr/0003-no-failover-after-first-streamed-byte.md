# ADR 0003 — No failover after the first streamed byte

**Status:** Accepted.

## Context

The router fails over between candidate models when a backend fails. Streaming
breaks the assumption that makes this safe.

In a buffered request, a mid-flight failure is invisible to the caller: the fabric
retries elsewhere and returns whichever attempt succeeded. In a streamed request,
generated text has already been handed to the client. The client may have rendered
it. A retry cannot recall it.

Two failure windows therefore exist, and they are not the same problem:

1. the backend fails **before** producing any text, and
2. the backend fails **after** producing some.

## Decision

Fail over in case 1. Never in case 2 — propagate the error instead.

The rule is enforced in the routing engine, which tracks whether any delta has
been emitted for the current attempt, not in the individual adapters.

## Rationale

**Failing over mid-stream corrupts the response.** The client would receive the
first model's partial output concatenated with the second model's output, which is
not a valid response from either. A truncated response with an explicit error is
worse for the user and better for correctness — it is honest about what happened,
and it is detectable, whereas a spliced response looks fine and is wrong.

**Enforcement belongs in the engine.** Every adapter could implement this rule,
and each would be a place to get it wrong. The engine already owns candidate
iteration, so it is the natural single point of enforcement, and a new adapter
inherits the guarantee rather than re-earning it.

**Errors after commitment need an in-band channel.** Once streaming begins the
HTTP status is already on the wire and cannot be changed. So the gateway resolves
candidates *before* opening the stream — an unknown or disabled model is a proper
HTTP 4xx, not a 200 containing an error — and a genuine mid-stream failure is
reported as an error frame inside the stream, followed by `[DONE]`. Clients see an
explicit failure rather than a silent truncation.

## Consequences

A streamed request is less resilient than a buffered one, by design. Callers who
want maximum resilience should not stream, and that trade-off is theirs to make
per request.

The engine must track emission state per attempt, which is the small amount of
complexity this guarantee costs.
