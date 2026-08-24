"""Scope and role enforcement, and prompt version immutability."""

from __future__ import annotations

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from llm_fabric.errors import AuthorizationError, FabricError, InvalidRequestError
from llm_fabric.gateway.dependencies import require_roles, require_scopes
from llm_fabric.gateway.middleware import AuthenticationMiddleware, anonymous_principal
from llm_fabric.identity.claims import Principal
from llm_fabric.identity.dev import DevIdentityProvider
from llm_fabric.storage.records import PromptDefinition, PromptStatus
from llm_fabric.storage.repositories import PromptRepository
from llm_fabric.tenancy.scope import TenantScope

SECRET = "development-secret-that-is-long-enough"


@pytest.fixture
def issuer() -> DevIdentityProvider:
    return DevIdentityProvider(secret=SECRET)


NEEDS_PROMPT_WRITE = Depends(require_scopes("prompts:write"))
NEEDS_ADMIN_OR_OPERATOR = Depends(require_roles("admin", "operator"))


@pytest.fixture
def app(issuer: DevIdentityProvider) -> FastAPI:
    application = FastAPI()

    @application.get("/needs-scope")
    async def needs_scope(_: Principal = NEEDS_PROMPT_WRITE) -> dict:
        return {"ok": True}

    @application.get("/needs-role")
    async def needs_role(_: Principal = NEEDS_ADMIN_OR_OPERATOR) -> dict:
        return {"ok": True}

    @application.exception_handler(FabricError)
    async def handle(request, exc: FabricError):  # type: ignore[no-untyped-def]
        from fastapi.responses import JSONResponse

        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"message": exc.message, "type": exc.error_type}},
        )

    application.add_middleware(
        AuthenticationMiddleware, verifier=issuer, auth_required=True, public_paths=()
    )
    return application


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def test_a_caller_with_the_scope_is_allowed(app: FastAPI, issuer: DevIdentityProvider) -> None:
    token = issuer.issue_token(tenant_id="acme", user_id="alice", scopes=["prompts:write"])

    with TestClient(app) as client:
        assert client.get("/needs-scope", headers=_auth(token)).status_code == 200


def test_a_caller_without_the_scope_is_refused(app: FastAPI, issuer: DevIdentityProvider) -> None:
    token = issuer.issue_token(tenant_id="acme", user_id="alice", scopes=["prompts:read"])

    with TestClient(app) as client:
        response = client.get("/needs-scope", headers=_auth(token))

    assert response.status_code == 403
    assert response.json()["error"]["type"] == "permission_error"


def test_authentication_failure_is_401_not_403(app: FastAPI) -> None:
    """The two must not be conflated: one means "who are you", the other "no"."""
    with TestClient(app) as client:
        assert client.get("/needs-scope").status_code == 401


def test_any_one_of_the_accepted_roles_suffices(app: FastAPI, issuer: DevIdentityProvider) -> None:
    token = issuer.issue_token(tenant_id="acme", user_id="alice", roles=["operator"])

    with TestClient(app) as client:
        assert client.get("/needs-role", headers=_auth(token)).status_code == 200


def test_an_unrelated_role_is_refused(app: FastAPI, issuer: DevIdentityProvider) -> None:
    token = issuer.issue_token(tenant_id="acme", user_id="alice", roles=["viewer"])

    with TestClient(app) as client:
        assert client.get("/needs-role", headers=_auth(token)).status_code == 403


async def test_scope_checks_are_exact_not_prefix_matched() -> None:
    """`prompts:read` must not satisfy `prompts:write`.

    The dependency is awaited rather than called: gateway dependencies are
    `async def` so FastAPI runs them on the event loop instead of dispatching
    each to a worker thread. Calling one without awaiting would build a
    coroutine that never runs, and this assertion would pass while checking
    nothing.
    """
    principal = Principal(
        tenant_id="acme",
        user_id="alice",
        subject="s",
        issuer="i",
        scopes=frozenset({"prompts:read", "prompts:write:draft"}),
    )
    dependency = require_scopes("prompts:write")

    class _Request:
        class state:  # noqa: N801
            principal = None

    _Request.state.principal = principal  # type: ignore[attr-defined]

    with pytest.raises(AuthorizationError):
        await dependency(_Request)  # type: ignore[arg-type]


def test_the_anonymous_principal_holds_no_scopes_or_roles() -> None:
    """Disabling authentication must not accidentally grant authority."""
    principal = anonymous_principal()

    assert principal.scopes == frozenset()
    assert principal.roles == frozenset()
    assert principal.may_delegate_tenant is False


# -- prompt immutability -----------------------------------------------------


def test_a_published_prompt_version_cannot_be_rewritten() -> None:
    repository = PromptRepository()
    scope = TenantScope(tenant_id="acme", user_id="alice")
    published = PromptDefinition(
        tenant_id="acme",
        prompt_id="triage",
        version=1,
        owner="alice",
        purpose="triage",
        template="original",
        status=PromptStatus.PRODUCTION,
    )
    repository.publish(scope, published)

    with pytest.raises(InvalidRequestError, match="cannot be modified"):
        repository.publish(
            scope,
            PromptDefinition(
                tenant_id="acme",
                prompt_id="triage",
                version=1,
                owner="mallory",
                purpose="triage",
                template="rewritten",
                status=PromptStatus.PRODUCTION,
            ),
        )

    assert repository.require(scope, "triage", 1).template == "original"


def test_a_draft_can_still_be_edited() -> None:
    repository = PromptRepository()
    scope = TenantScope(tenant_id="acme", user_id="alice")
    draft = PromptDefinition(
        tenant_id="acme",
        prompt_id="triage",
        version=1,
        owner="alice",
        purpose="triage",
        template="first attempt",
    )
    repository.publish(scope, draft)

    from dataclasses import replace

    repository.publish(scope, replace(draft, template="second attempt"))

    assert repository.require(scope, "triage", 1).template == "second attempt"


def test_promotion_advances_status_without_changing_content() -> None:
    repository = PromptRepository()
    scope = TenantScope(tenant_id="acme", user_id="alice")
    repository.publish(
        scope,
        PromptDefinition(
            tenant_id="acme",
            prompt_id="triage",
            version=2,
            owner="alice",
            purpose="triage",
            template="content",
        ),
    )

    promoted = repository.promote(scope, "triage", 2, PromptStatus.PRODUCTION)

    assert promoted.status is PromptStatus.PRODUCTION
    assert promoted.template == "content"
