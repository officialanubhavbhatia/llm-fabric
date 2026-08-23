"""Client authentication.

Keys are compared with a constant-time comparison so that a timing signal does
not leak how much of a candidate key was correct.

When no keys are configured the gateway is open. That is a deliberate local
development affordance, and it is logged as a warning at startup so an open
deployment is never silent.
"""

from __future__ import annotations

import hashlib
import secrets

from fastapi import Request

from llm_fabric.errors import AuthenticationError


def _presented_key(request: Request) -> str | None:
    header = request.headers.get("authorization")
    if header:
        scheme, _, credential = header.partition(" ")
        if scheme.lower() == "bearer" and credential.strip():
            return credential.strip()
    api_key = request.headers.get("x-api-key")
    return api_key.strip() if api_key else None


def client_fingerprint(key: str) -> str:
    """A short, stable, non-reversible label for a key, safe to put in logs."""
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]


def authenticate(request: Request, accepted_keys: list[str]) -> str | None:
    """Return a client fingerprint, or None when the gateway is running open."""
    if not accepted_keys:
        return None

    presented = _presented_key(request)
    if not presented:
        raise AuthenticationError(
            "missing API key: supply 'Authorization: Bearer <key>' or 'x-api-key'"
        )

    for candidate in accepted_keys:
        if secrets.compare_digest(presented, candidate):
            return client_fingerprint(presented)

    raise AuthenticationError("invalid API key")
