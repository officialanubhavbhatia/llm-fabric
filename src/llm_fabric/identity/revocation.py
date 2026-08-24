"""Token revocation: a denylist over otherwise-stateless credentials.

A signed JWT is valid until `exp`. There is no protocol-level way to take it
back. Immediate revocation therefore requires a store the verifier consults on
every request, keyed by `jti` and/or a fingerprint of the presented credential.

This module is the process-local store used until a distributed backend exists.
A revoke on one worker is invisible to the others. See `docs/AUTH_REVOCATION.md`.
"""

from __future__ import annotations

import hashlib
import threading
import time
from collections.abc import Sequence
from typing import Protocol

from llm_fabric.errors import AuthenticationError
from llm_fabric.identity.claims import Principal
from llm_fabric.identity.verifier import TokenVerifier


def token_fingerprint(token: str) -> str:
    """Stable, non-reversible label for a presented credential."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


class TokenRevocationStore(Protocol):
    """Records and answers revocation. Must not store the raw token."""

    def is_revoked(self, *, token_id: str | None = None, fingerprint: str | None = None) -> bool:
        """True when this credential has been revoked and the entry has not expired."""
        ...

    def revoke(
        self,
        *,
        token_id: str | None = None,
        fingerprint: str | None = None,
        expires_at: float | None = None,
    ) -> None:
        """Denylist `token_id` (`jti`) and/or the credential fingerprint."""
        ...


#: Hard cap so a revoke loop cannot grow this process without bound.
_MAX_ENTRIES = 100_000


class InMemoryRevocationStore:
    """A process-local denylist.

    Entries expire at `expires_at` (unix seconds) so a denylist cannot grow
    without bound past the natural lifetime of the tokens it tracks. A hard
    cap evicts the oldest entries if an operator revokes without expiry.
    The store is safe for concurrent use inside one process and is *not*
    shared across workers.
    """

    def __init__(self, *, max_entries: int = _MAX_ENTRIES) -> None:
        self._by_id: dict[str, float | None] = {}
        self._by_fingerprint: dict[str, float | None] = {}
        self._max_entries = max_entries
        self._lock = threading.Lock()

    def is_revoked(self, *, token_id: str | None = None, fingerprint: str | None = None) -> bool:
        with self._lock:
            if token_id and self._active(self._by_id, token_id):
                return True
            if fingerprint and self._active(self._by_fingerprint, fingerprint):
                return True
        return False

    def revoke(
        self,
        *,
        token_id: str | None = None,
        fingerprint: str | None = None,
        expires_at: float | None = None,
    ) -> None:
        if token_id is None and fingerprint is None:
            raise ValueError("revoke requires a token_id or a fingerprint")
        with self._lock:
            self._purge_expired_unlocked()
            if token_id:
                self._by_id[token_id] = expires_at
            if fingerprint:
                self._by_fingerprint[fingerprint] = expires_at
            self._evict_overflow_unlocked()

    def _purge_expired_unlocked(self) -> None:
        now = time.time()
        for table in (self._by_id, self._by_fingerprint):
            stale = [
                key
                for key, expires_at in table.items()
                if expires_at is not None and expires_at <= now
            ]
            for key in stale:
                del table[key]

    def _evict_overflow_unlocked(self) -> None:
        while len(self._by_id) + len(self._by_fingerprint) > self._max_entries:
            if self._by_id:
                self._by_id.pop(next(iter(self._by_id)))
                continue
            if self._by_fingerprint:
                self._by_fingerprint.pop(next(iter(self._by_fingerprint)))

    def _active(self, table: dict[str, float | None], key: str) -> bool:
        if key not in table:
            return False
        expires_at = table[key]
        if expires_at is not None and expires_at <= time.time():
            del table[key]
            return False
        return True


class RevokingVerifier:
    """Runs an inner verifier, then refuses credentials on the denylist.

    Global required scopes are applied here so every identity source — OIDC,
    API keys, the development issuer — is subject to the same floor.
    """

    def __init__(
        self,
        inner: TokenVerifier,
        store: TokenRevocationStore,
        *,
        required_scopes: Sequence[str] = (),
    ) -> None:
        self._inner = inner
        self._store = store
        self._required_scopes = tuple(scope for scope in required_scopes if scope)

    @property
    def inner(self) -> TokenVerifier:
        return self._inner

    @property
    def store(self) -> TokenRevocationStore:
        return self._store

    async def aclose(self) -> None:
        await self._inner.aclose()

    async def verify(self, token: str) -> Principal:
        fingerprint = token_fingerprint(token)
        if self._store.is_revoked(fingerprint=fingerprint):
            raise AuthenticationError("token has been revoked")

        principal = await self._inner.verify(token)

        if self._store.is_revoked(token_id=principal.token_id, fingerprint=fingerprint):
            raise AuthenticationError("token has been revoked")

        if self._required_scopes and any(
            not principal.has_scope(scope) for scope in self._required_scopes
        ):
            raise AuthenticationError("token is missing a required scope")

        return principal
