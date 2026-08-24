"""Token issuance for the local development identity provider.

Mounted only when `auth_mode` resolves to `dev`. It exists so a developer can
obtain a real token and exercise the real authorization path offline, instead of
running with authentication disabled and discovering the difference in staging.

This endpoint mints any identity it is asked for. That is acceptable precisely
because enabling it requires deliberately setting a long development secret, and
the gateway warns about it on every startup.
"""

from __future__ import annotations

from fastapi import APIRouter, Request
from pydantic import BaseModel, ConfigDict, Field

from llm_fabric.errors import ConfigurationError
from llm_fabric.identity.dev import DevIdentityProvider
from llm_fabric.identity.revocation import RevokingVerifier

router = APIRouter(prefix="/v1/dev", tags=["development"])

DEV_TOKEN_PATH = "/v1/dev/token"


class DevTokenRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tenant_id: str = Field(min_length=1)
    user_id: str = Field(default="developer", min_length=1)
    project_id: str | None = None
    roles: list[str] = Field(default_factory=list)
    scopes: list[str] = Field(default_factory=list)
    ttl_seconds: int = Field(default=3600, ge=60, le=86_400)


class DevTokenResponse(BaseModel):
    access_token: str
    token_type: str = "Bearer"
    expires_in: int


@router.post("/token", response_model=DevTokenResponse, summary="Issue a development token")
async def issue_token(body: DevTokenRequest, request: Request) -> DevTokenResponse:
    provider = request.app.state.verifier
    if isinstance(provider, RevokingVerifier):
        provider = provider.inner
    if not isinstance(provider, DevIdentityProvider):
        raise ConfigurationError("the development identity provider is not enabled")

    token = provider.issue_token(
        tenant_id=body.tenant_id,
        user_id=body.user_id,
        project_id=body.project_id,
        roles=body.roles,
        scopes=body.scopes,
        ttl_seconds=body.ttl_seconds,
    )
    return DevTokenResponse(access_token=token, expires_in=body.ttl_seconds)
