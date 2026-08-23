"""Backend adapters. One module per provider; nothing else imports them directly."""

from llm_fabric.serving.adapters.anthropic import AnthropicProvider
from llm_fabric.serving.adapters.mock import MockProvider
from llm_fabric.serving.adapters.openai import OpenAIProvider

__all__ = ["AnthropicProvider", "MockProvider", "OpenAIProvider"]
