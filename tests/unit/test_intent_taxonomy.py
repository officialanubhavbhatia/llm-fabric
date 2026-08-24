"""Taxonomy invariants: immutability, structure, and versioning."""

from __future__ import annotations

import pytest

from llm_fabric.errors import ConfigurationError, ResourceNotFoundError
from llm_fabric.intent.bootstrap import BOOTSTRAP_TAXONOMY_VERSION, bootstrap_taxonomy
from llm_fabric.intent.taxonomy import (
    IntentNode,
    IntentStatus,
    IntentTaxonomy,
    TaxonomyRegistry,
)


def node(intent_id: str, **kwargs: object) -> IntentNode:
    return IntentNode(
        intent_id=intent_id,
        name=kwargs.pop("name", intent_id),  # type: ignore[arg-type]
        description=kwargs.pop("description", f"the {intent_id} intent"),  # type: ignore[arg-type]
        **kwargs,  # type: ignore[arg-type]
    )


def test_hierarchy_is_derived_from_the_dotted_id() -> None:
    taxonomy = IntentTaxonomy(
        "v1", [node("coding"), node("coding.debug.stacktrace"), node("coding.debug")]
    )

    leaf = taxonomy.require("coding.debug.stacktrace")
    assert leaf.domain == "coding"
    assert leaf.task == "debug"
    assert leaf.subtask == "stacktrace"
    assert leaf.parent_intent_id == "coding.debug"
    assert taxonomy.ancestors("coding.debug.stacktrace") == ("coding.debug", "coding")


def test_a_node_cannot_claim_a_version_it_is_not_in() -> None:
    stray = node("coding", taxonomy_version="v99")
    taxonomy = IntentTaxonomy("v1", [stray])

    assert taxonomy.require("coding").taxonomy_version == "v1"


def test_a_missing_parent_is_refused() -> None:
    with pytest.raises(ConfigurationError, match="not in the taxonomy"):
        IntentTaxonomy("v1", [node("coding.debug")])


def test_a_cycle_is_refused() -> None:
    left = IntentNode(intent_id="a", name="a", description="a", parent_intent_id="b")
    right = IntentNode(intent_id="b", name="b", description="b", parent_intent_id="a")

    with pytest.raises(ConfigurationError, match="cycle"):
        IntentTaxonomy("v1", [left, right])


def test_duplicate_ids_are_refused() -> None:
    with pytest.raises(ConfigurationError, match="duplicate"):
        IntentTaxonomy("v1", [node("coding"), node("coding")])


def test_the_unknown_id_is_reserved() -> None:
    with pytest.raises(ConfigurationError, match="reserved"):
        node("unknown")


def test_an_unknown_intent_raises_rather_than_returning_none_on_require() -> None:
    taxonomy = IntentTaxonomy("v1", [node("coding")])

    assert taxonomy.get("nope") is None
    with pytest.raises(ResourceNotFoundError):
        taxonomy.require("nope")


def test_evolving_leaves_the_previous_version_untouched() -> None:
    first = IntentTaxonomy("v1", [node("coding"), node("writing")])
    second = first.evolve("v2", add=[node("vision")], retire=["writing"])

    assert "vision" in second
    assert "vision" not in first
    assert second.require("writing").status is IntentStatus.RETIRED
    assert first.require("writing").status is IntentStatus.ACTIVE
    assert first.version == "v1" and second.version == "v2"


def test_evolving_into_the_same_version_is_refused() -> None:
    taxonomy = IntentTaxonomy("v1", [node("coding")])

    with pytest.raises(ConfigurationError, match="immutable"):
        taxonomy.evolve("v1", add=[node("writing")])


def test_a_retired_intent_stays_readable_but_is_not_classifiable() -> None:
    taxonomy = IntentTaxonomy("v1", [node("coding"), node("writing")]).evolve(
        "v2", retire=["writing"]
    )

    assert taxonomy.require("writing").name == "writing"
    assert "writing" not in {node.intent_id for node in taxonomy.classifiable()}


def test_the_registry_refuses_to_overwrite_a_version() -> None:
    registry = TaxonomyRegistry([IntentTaxonomy("v1", [node("coding")])])

    with pytest.raises(ConfigurationError, match="already registered"):
        registry.register(IntentTaxonomy("v1", [node("writing")]))


def test_the_registry_returns_the_most_recently_registered_version() -> None:
    registry = TaxonomyRegistry()
    registry.register(IntentTaxonomy("v1", [node("coding")]))
    latest = registry.register(IntentTaxonomy("v2", [node("coding"), node("writing")]))

    assert registry.latest() is latest
    assert registry.versions == ("v1", "v2")
    assert registry.require("v1").version == "v1"


def test_an_empty_registry_reports_that_rather_than_returning_nothing() -> None:
    with pytest.raises(ConfigurationError, match="no taxonomy"):
        TaxonomyRegistry().latest()


class TestBootstrapTaxonomy:
    def test_it_carries_the_requested_domains(self) -> None:
        taxonomy = bootstrap_taxonomy()

        expected = {
            "coding",
            "agent",
            "reasoning",
            "math",
            "research",
            "rag",
            "data_analysis",
            "writing",
            "summarization",
            "translation",
            "extraction",
            "classification",
            "vision",
            "tool_use",
            "general_conversation",
        }
        assert expected <= set(taxonomy.domains())

    def test_every_domain_carries_examples_and_hard_negatives(self) -> None:
        taxonomy = bootstrap_taxonomy()

        for root in taxonomy.roots():
            assert root.examples, f"{root.intent_id} has no examples"
            assert root.hard_negatives, f"{root.intent_id} has no hard negatives"

    def test_each_call_returns_an_independent_taxonomy(self) -> None:
        assert bootstrap_taxonomy() is not bootstrap_taxonomy()
        assert bootstrap_taxonomy().version == BOOTSTRAP_TAXONOMY_VERSION

    def test_training_texts_exclude_negatives(self) -> None:
        node = bootstrap_taxonomy().require("translation")

        texts = node.training_texts()
        assert set(texts) == set(node.examples)
        for negative in node.hard_negatives:
            assert negative not in texts
