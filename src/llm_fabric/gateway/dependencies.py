"""Request-scoped dependencies.

Components live on `app.state` because they are built once at startup and shared;
these helpers are the only sanctioned way for a route to reach them, so routes
never touch application state directly.
"""

from __future__ import annotations

import uuid

from fastapi import Depends, Request

from llm_fabric.config import Settings
from llm_fabric.gateway.auth import authenticate
from llm_fabric.observability.metering import InMemoryMeter
from llm_fabric.router.engine import Router
from llm_fabric.router.registry import ModelRegistry


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_registry(request: Request) -> ModelRegistry:
    return request.app.state.registry


def get_router(request: Request) -> Router:
    return request.app.state.router


def get_meter(request: Request) -> InMemoryMeter:
    return request.app.state.meter


def get_request_id(request: Request) -> str:
    """Honour a caller-supplied correlation id so traces can be joined up."""
    return request.headers.get("x-request-id") or uuid.uuid4().hex


def get_client_id(
    request: Request,
    settings: Settings = Depends(get_settings),
) -> str | None:
    return authenticate(request, settings.api_keys)
