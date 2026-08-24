"""Model probe, evaluation, and registration helpers.

Declared registry metadata is not a measurement. Probe and eval produce
versioned evidence. They do not enable Agents, MCP, or serving-path IntentOS.
"""

from llm_fabric.models.probe import PROBE_VERSION, probe_deployment, probe_provider_model

__all__ = ["PROBE_VERSION", "probe_deployment", "probe_provider_model"]
