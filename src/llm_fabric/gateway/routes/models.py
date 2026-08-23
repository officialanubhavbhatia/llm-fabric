"""Model discovery, in the OpenAI `/v1/models` shape.

Aliases are listed alongside concrete models, owned by `llm-fabric`, so a client
can discover `auto` and route through the fabric's own policy without knowing
which providers sit behind it.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from llm_fabric.contract.openai import ModelCard, ModelList
from llm_fabric.errors import ModelNotFoundError
from llm_fabric.gateway.dependencies import get_client_id, get_registry
from llm_fabric.router.registry import ModelRegistry

router = APIRouter(prefix="/v1", tags=["models"])


@router.get("/models", response_model=ModelList, summary="List servable models")
async def list_models(
    registry: ModelRegistry = Depends(get_registry),
    _client: str | None = Depends(get_client_id),
) -> ModelList:
    return ModelList(data=registry.cards())


@router.get("/models/{model_id}", response_model=ModelCard, summary="Describe one model")
async def get_model(
    model_id: str,
    registry: ModelRegistry = Depends(get_registry),
    _client: str | None = Depends(get_client_id),
) -> ModelCard:
    if alias := registry.alias(model_id):
        return ModelCard(id=alias.id, owned_by="llm-fabric")
    if not registry.known(model_id):
        raise ModelNotFoundError(f"unknown model '{model_id}'")
    return registry.get(model_id).to_card()
