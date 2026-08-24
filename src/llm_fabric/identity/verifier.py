"""The verification interface every identity source implements."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from llm_fabric.identity.claims import Principal


@runtime_checkable
class TokenVerifier(Protocol):
    """Turns a bearer credential into a `Principal`, or refuses.

    Implementations must raise `AuthenticationError` on any failure and must
    never return a principal built from unverified input.
    """

    async def verify(self, token: str) -> Principal:
        """Validate `token` and return the caller it identifies."""
        ...

    async def aclose(self) -> None:
        """Release any held resources."""
        ...
