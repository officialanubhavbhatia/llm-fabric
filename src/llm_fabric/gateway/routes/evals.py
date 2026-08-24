"""Run a named evaluation suite through the gateway.

Only committed, named suites are accepted. A client-supplied filesystem path
would let a caller read the server's disk.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from llm_fabric.errors import InvalidRequestError
from llm_fabric.eval.runner import load_suite, run_suite
from llm_fabric.gateway.dependencies import (
    get_classify_cascade,
    get_router,
    get_stores,
    get_tenant_scope,
)
from llm_fabric.intent.cascade import IntentCascade
from llm_fabric.router.engine import Router
from llm_fabric.storage.repositories import TenantStores
from llm_fabric.tenancy.scope import TenantScope

router = APIRouter(prefix="/v1/evals", tags=["evals"])

NAMED_SUITES: dict[str, Path] = {
    "ci": Path("datasets/eval/ci-suite.yaml"),
}


class EvalRunBody(BaseModel):
    suite: str = Field(description="A named suite. Currently only `ci`.")


@router.post("/run", summary="Run a named evaluation suite")
async def run_eval(
    body: EvalRunBody,
    scope: TenantScope = Depends(get_tenant_scope),
    fabric: Router = Depends(get_router),
    cascade: IntentCascade = Depends(get_classify_cascade),
    stores: TenantStores = Depends(get_stores),
) -> dict[str, Any]:
    path = NAMED_SUITES.get(body.suite.strip().lower())
    if path is None:
        available = ", ".join(sorted(NAMED_SUITES))
        raise InvalidRequestError(f"unknown eval suite '{body.suite}' (available: {available})")
    if not path.is_file():
        raise InvalidRequestError(
            f"eval suite '{body.suite}' is not present on this process (expected {path})"
        )
    run = await run_suite(
        load_suite(path),
        scope=scope,
        router=fabric,
        cascade=cascade,
        store=stores.eval_runs,
    )
    return run.as_dict()
