"""The model registry: what the fabric can serve, and what it costs.

The registry is declarative configuration rather than code so that adding a
model, repricing one, or disabling one is a config change and not a deploy of new
logic. Prices are expressed per million tokens because that is how providers
publish them; they are operator-supplied inputs, not measurements the fabric
makes.

Two kinds of entry exist:

* a **model**, which maps a fabric-facing id onto one provider and that
  provider's own model name, and
* an **alias**, a virtual id that resolves to several models under a policy.
  `auto` is an alias.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from llm_fabric.contract.openai import ModelCard
from llm_fabric.errors import ConfigurationError, ModelNotFoundError


@dataclass(frozen=True, slots=True)
class ModelSpec:
    id: str
    provider: str
    provider_model: str
    context_window: int | None = None
    input_cost_per_mtok: float = 0.0
    output_cost_per_mtok: float = 0.0
    capabilities: frozenset[str] = frozenset()
    enabled: bool = True
    fallbacks: tuple[str, ...] = ()

    def cost_usd(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Cost of a call at registry prices. Zero when the model is unpriced."""
        return (
            prompt_tokens * self.input_cost_per_mtok + completion_tokens * self.output_cost_per_mtok
        ) / 1_000_000

    @property
    def blended_cost_per_mtok(self) -> float:
        """A single comparable price, weighting output more heavily than input.

        Used only to order candidates for the `cheapest` policy. The 1:3 weighting
        reflects that generation is the priced-heavier side for most providers; it
        is a heuristic for ranking, not a prediction of spend.
        """
        return (self.input_cost_per_mtok + 3 * self.output_cost_per_mtok) / 4

    def to_card(self) -> ModelCard:
        return ModelCard(id=self.id, owned_by=self.provider, context_window=self.context_window)


@dataclass(frozen=True, slots=True)
class Alias:
    id: str
    candidates: tuple[str, ...]
    policy: str | None = None
    requires: frozenset[str] = frozenset()


def _as_capabilities(value: Any) -> frozenset[str]:
    if not value:
        return frozenset()
    if isinstance(value, str):
        return frozenset({value})
    return frozenset(str(item) for item in value)


def _as_tuple(value: Any) -> tuple[str, ...]:
    if not value:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(str(item) for item in value)


class ModelRegistry:
    def __init__(self, models: list[ModelSpec], aliases: list[Alias] | None = None) -> None:
        self._models: dict[str, ModelSpec] = {}
        for spec in models:
            if spec.id in self._models:
                raise ConfigurationError(f"duplicate model id in registry: {spec.id}")
            self._models[spec.id] = spec

        self._aliases: dict[str, Alias] = {}
        for alias in aliases or []:
            if alias.id in self._models:
                raise ConfigurationError(f"alias '{alias.id}' collides with a model id")
            self._aliases[alias.id] = alias

        self._validate_references()

    def _validate_references(self) -> None:
        for spec in self._models.values():
            for fallback in spec.fallbacks:
                if fallback not in self._models:
                    raise ConfigurationError(
                        f"model '{spec.id}' declares unknown fallback '{fallback}'"
                    )
        for alias in self._aliases.values():
            if not alias.candidates:
                raise ConfigurationError(f"alias '{alias.id}' has no candidates")
            for candidate in alias.candidates:
                if candidate not in self._models:
                    raise ConfigurationError(
                        f"alias '{alias.id}' references unknown model '{candidate}'"
                    )

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> ModelRegistry:
        raw_models = data.get("models") or []
        if not isinstance(raw_models, list):
            raise ConfigurationError("registry 'models' must be a list")

        models: list[ModelSpec] = []
        for entry in raw_models:
            if not isinstance(entry, dict):
                raise ConfigurationError("each registry model must be a mapping")
            missing = {"id", "provider"} - entry.keys()
            if missing:
                raise ConfigurationError(
                    f"registry model missing required field(s): {sorted(missing)}"
                )
            models.append(
                ModelSpec(
                    id=str(entry["id"]),
                    provider=str(entry["provider"]),
                    provider_model=str(entry.get("provider_model", entry["id"])),
                    context_window=(
                        int(entry["context_window"]) if entry.get("context_window") else None
                    ),
                    input_cost_per_mtok=float(entry.get("input_cost_per_mtok", 0.0)),
                    output_cost_per_mtok=float(entry.get("output_cost_per_mtok", 0.0)),
                    capabilities=_as_capabilities(entry.get("capabilities")),
                    enabled=bool(entry.get("enabled", True)),
                    fallbacks=_as_tuple(entry.get("fallbacks")),
                )
            )

        aliases = [
            Alias(
                id=str(entry["id"]),
                candidates=_as_tuple(entry.get("candidates")),
                policy=str(entry["policy"]) if entry.get("policy") else None,
                requires=_as_capabilities(entry.get("requires")),
            )
            for entry in data.get("aliases") or []
            if isinstance(entry, dict) and entry.get("id")
        ]

        return cls(models, aliases)

    @classmethod
    def from_yaml(cls, path: Path) -> ModelRegistry:
        if not path.exists():
            raise ConfigurationError(f"model registry not found at {path}")
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        if not isinstance(data, dict):
            raise ConfigurationError(f"model registry at {path} must be a mapping")
        return cls.from_mapping(data)

    def get(self, model_id: str) -> ModelSpec:
        try:
            return self._models[model_id]
        except KeyError:
            raise ModelNotFoundError(f"unknown model '{model_id}'") from None

    def alias(self, alias_id: str) -> Alias | None:
        return self._aliases.get(alias_id)

    def is_alias(self, model_id: str) -> bool:
        return model_id in self._aliases

    def known(self, model_id: str) -> bool:
        return model_id in self._models or model_id in self._aliases

    def enabled_models(self) -> list[ModelSpec]:
        return [spec for spec in self._models.values() if spec.enabled]

    def providers_in_use(self) -> set[str]:
        return {spec.provider for spec in self.enabled_models()}

    def cards(self) -> list[ModelCard]:
        cards = [spec.to_card() for spec in self.enabled_models()]
        cards.extend(
            ModelCard(id=alias.id, owned_by="llm-fabric") for alias in self._aliases.values()
        )
        return cards
