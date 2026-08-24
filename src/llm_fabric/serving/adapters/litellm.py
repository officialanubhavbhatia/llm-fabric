"""LiteLLM proxy adapter.

LiteLLM is an OpenAI-compatible *transport*. The MyVista Route Planner still
selects the deployment and `provider_model` (the LiteLLM model name). This
module does not choose a different model.
"""

from __future__ import annotations

from llm_fabric.serving.adapters.openai import OpenAIProvider


class LiteLLMProvider(OpenAIProvider):
    name = "litellm"
