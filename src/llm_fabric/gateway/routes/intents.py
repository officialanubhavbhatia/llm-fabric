"""Intent classification as an explicit request, not a serving-path side effect."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from llm_fabric.gateway.dependencies import get_classify_cascade, get_tenant_scope
from llm_fabric.intent.cascade import IntentCascade
from llm_fabric.intent.features import bound_text
from llm_fabric.intent.schema import ClassificationRequest
from llm_fabric.tenancy.scope import TenantScope

router = APIRouter(prefix="/v1/intents", tags=["intents"])


class ClassifyBody(BaseModel):
    input: str = Field(min_length=1)
    language: str = "en"
    conversation_context: str = ""


@router.post("/classify", summary="Classify a prompt")
async def classify(
    body: ClassifyBody,
    cascade: IntentCascade = Depends(get_classify_cascade),
    scope: TenantScope = Depends(get_tenant_scope),
) -> dict[str, Any]:
    decision = await cascade.classify(
        scope,
        ClassificationRequest(
            text=bound_text(body.input),
            language=body.language,
            conversation_state_signature=body.conversation_context,
        ),
    )
    return decision.as_dict()
