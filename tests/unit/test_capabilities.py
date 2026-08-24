"""Capability matching decides eligibility, so its edges are pinned here."""

from __future__ import annotations

from llm_fabric.router.capabilities import Capability, CapabilityVector, normalise


def test_a_vector_satisfies_what_it_declares() -> None:
    vector = CapabilityVector.of(Capability.CHAT, Capability.VISION)
    assert vector.satisfies({Capability.CHAT})
    assert vector.satisfies({Capability.CHAT, Capability.VISION})
    assert not vector.satisfies({Capability.TOOLS})


def test_missing_names_exactly_what_is_lacking() -> None:
    vector = CapabilityVector.of(Capability.CHAT)
    assert vector.missing({Capability.CHAT}) == frozenset()
    assert vector.missing({Capability.CHAT, Capability.TOOLS, Capability.VISION}) == {
        Capability.TOOLS,
        Capability.VISION,
    }


def test_an_empty_requirement_is_satisfied_by_anything() -> None:
    assert CapabilityVector().satisfies(frozenset())


def test_json_schema_implies_structured_output() -> None:
    vector = CapabilityVector.of(Capability.JSON_SCHEMA)
    assert vector.has(Capability.STRUCTURED_OUTPUT)
    assert vector.satisfies({Capability.STRUCTURED_OUTPUT})
    assert Capability.STRUCTURED_OUTPUT in vector.effective
    # The declared set stays as the operator wrote it.
    assert Capability.STRUCTURED_OUTPUT not in vector.declared


def test_the_implication_does_not_run_backwards() -> None:
    # Being able to produce structured output does not mean a strict schema can
    # be enforced, so this must not resolve the other way.
    vector = CapabilityVector.of(Capability.STRUCTURED_OUTPUT)
    assert not vector.has(Capability.JSON_SCHEMA)


def test_unknown_capabilities_are_preserved_not_dropped() -> None:
    vector = CapabilityVector.from_config(["chat", "some_bespoke_thing"])
    assert vector.has("some_bespoke_thing")
    assert vector.satisfies({"some_bespoke_thing"})


def test_config_values_are_normalised() -> None:
    assert normalise("  Chat ") == {"chat"}
    assert normalise(["Chat", "VISION"]) == {"chat", "vision"}
    assert normalise(None) == frozenset()
    assert normalise([]) == frozenset()
    assert normalise("") == frozenset()


def test_supports_flags_derive_from_the_vector() -> None:
    vector = CapabilityVector.of(
        Capability.TOOLS,
        Capability.JSON_SCHEMA,
        Capability.VISION,
        Capability.EMBEDDINGS,
        Capability.PREFIX_CACHE,
        Capability.SPECULATIVE_DECODE,
    )
    assert vector.supports_tools
    assert vector.supports_json_schema
    assert vector.supports_vision
    assert vector.supports_embeddings
    assert vector.supports_prefix_cache
    assert vector.supports_speculative_decode

    empty = CapabilityVector()
    assert not empty.supports_tools
    assert not empty.supports_vision


def test_union_merges_declared_sets() -> None:
    merged = CapabilityVector.of(Capability.CHAT).union(CapabilityVector.of(Capability.VISION))
    assert merged.satisfies({Capability.CHAT, Capability.VISION})


def test_vectors_compare_by_declared_capabilities() -> None:
    assert CapabilityVector.of("chat", "vision") == CapabilityVector.of("vision", "chat")
    assert CapabilityVector.of("chat") != CapabilityVector.of("vision")
