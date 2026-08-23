"""The gateway layer: the stable public surface callers depend on."""

from llm_fabric.gateway.app import create_app

__all__ = ["create_app"]
