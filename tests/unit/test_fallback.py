"""The fallback graph: reasons, loop prevention, depth and budgets."""

from __future__ import annotations

import pytest

from llm_fabric.errors import (
    ConfigurationError,
    InvalidRequestError,
    ModelNotFoundError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    QuotaExceededError,
    RetryableError,
)
from llm_fabric.router.fallback import (
    ANY_REASON,
    ContextTooLargeError,
    FallbackBudget,
    FallbackEdge,
    FallbackGraph,
    FallbackHop,
    FallbackLedger,
    FallbackReason,
    FallbackTrace,
    reason_for_error,
)

TIMEOUT = FallbackReason.TIMEOUT
OVERLOADED = FallbackReason.OVERLOADED
CONTEXT = FallbackReason.CONTEXT_TOO_LARGE


# -- reasons are the constitution's list --------------------------------------


def test_every_mandated_reason_exists() -> None:
    assert {reason.value for reason in FallbackReason} == {
        "timeout",
        "overloaded",
        "rate_limited",
        "provider_down",
        "context_too_large",
        "model_unavailable",
        "safety_requirement",
        "structured_output_failure",
        "quality_failure",
    }


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (ProviderTimeoutError("slow"), FallbackReason.TIMEOUT),
        (QuotaExceededError("too many"), FallbackReason.RATE_LIMITED),
        (ContextTooLargeError("too big"), FallbackReason.CONTEXT_TOO_LARGE),
        (ModelNotFoundError("gone"), FallbackReason.MODEL_UNAVAILABLE),
        (ProviderUnavailableError("down"), FallbackReason.PROVIDER_DOWN),
        (InvalidRequestError("bad"), FallbackReason.MODEL_UNAVAILABLE),
        (RetryableError("upstream"), FallbackReason.PROVIDER_DOWN),
    ],
)
def test_errors_classify_into_reasons(error: Exception, expected: FallbackReason) -> None:
    assert reason_for_error(error) is expected


def test_an_unrecognised_error_gets_the_least_specific_reason() -> None:
    # Guessing a specific category would send the request down edges chosen for
    # a problem it does not have.
    assert reason_for_error(ValueError("who knows")) is FallbackReason.PROVIDER_DOWN


def test_context_too_large_is_a_caller_error_that_is_still_routable() -> None:
    error = ContextTooLargeError("too big")
    assert error.status_code == 400
    assert reason_for_error(error) is FallbackReason.CONTEXT_TOO_LARGE


def test_capacity_reasons_are_distinguished_from_model_reasons() -> None:
    assert FallbackReason.TIMEOUT.is_capacity
    assert FallbackReason.OVERLOADED.is_capacity
    assert not FallbackReason.CONTEXT_TOO_LARGE.is_capacity
    assert not FallbackReason.QUALITY_FAILURE.is_capacity


# -- edges --------------------------------------------------------------------


def test_an_edge_answers_only_its_reasons() -> None:
    edge = FallbackEdge("a", "b", reasons=frozenset({TIMEOUT}))
    assert edge.answers(TIMEOUT)
    assert not edge.answers(CONTEXT)


def test_a_self_edge_is_refused() -> None:
    with pytest.raises(ConfigurationError, match="itself"):
        FallbackEdge("a", "a")


def test_an_edge_with_no_reasons_is_refused() -> None:
    with pytest.raises(ConfigurationError, match="no reasons"):
        FallbackEdge("a", "b", reasons=frozenset())


# -- routing by reason --------------------------------------------------------


def _graph() -> FallbackGraph:
    return FallbackGraph(
        [
            FallbackEdge("small", "big", reasons=frozenset({CONTEXT})),
            FallbackEdge("small", "other-provider", reasons=frozenset({OVERLOADED, TIMEOUT})),
        ]
    )


def test_different_reasons_lead_to_different_places() -> None:
    graph = _graph()
    assert graph.next_hop("small", CONTEXT) == "big"
    assert graph.next_hop("small", OVERLOADED) == "other-provider"
    assert graph.next_hop("small", TIMEOUT) == "other-provider"


def test_a_reason_with_no_edge_ends_the_chain() -> None:
    assert _graph().next_hop("small", FallbackReason.SAFETY_REQUIREMENT) is None


def test_a_visited_deployment_is_never_returned_to() -> None:
    graph = _graph()
    assert graph.next_hop("small", CONTEXT, visited=["big"]) is None


def test_hops_are_confined_to_eligible_candidates() -> None:
    graph = _graph()
    assert graph.next_hop("small", CONTEXT, eligible=["small", "other-provider"]) is None
    assert graph.next_hop("small", CONTEXT, eligible=["small", "big"]) == "big"


def test_successors_keep_declaration_order() -> None:
    graph = FallbackGraph(
        [
            FallbackEdge("a", "first", reasons=frozenset({TIMEOUT})),
            FallbackEdge("a", "second", reasons=frozenset({TIMEOUT})),
        ]
    )
    assert graph.successors("a", TIMEOUT) == ("first", "second")
    assert graph.next_hop("a", TIMEOUT, visited=["first"]) == "second"


# -- loops --------------------------------------------------------------------


def test_a_mutual_pair_is_allowed_to_exist() -> None:
    graph = FallbackGraph([FallbackEdge("a", "b"), FallbackEdge("b", "a")])
    cycles = graph.detect_cycles()
    assert len(cycles) == 1
    assert set(cycles[0]) == {"a", "b"}


def test_traversal_cannot_loop_even_when_the_graph_does() -> None:
    graph = FallbackGraph([FallbackEdge("a", "b"), FallbackEdge("b", "a")])
    visited: set[str] = set()
    node: str | None = "a"
    seen: list[str] = []
    while node is not None and len(seen) < 10:
        seen.append(node)
        visited.add(node)
        node = graph.next_hop(node, TIMEOUT, visited=visited)
    assert seen == ["a", "b"]


def test_an_acyclic_graph_reports_no_cycles() -> None:
    graph = FallbackGraph([FallbackEdge("a", "b"), FallbackEdge("b", "c")])
    assert graph.detect_cycles() == ()


def test_a_longer_cycle_is_detected() -> None:
    graph = FallbackGraph([FallbackEdge("a", "b"), FallbackEdge("b", "c"), FallbackEdge("c", "a")])
    cycles = graph.detect_cycles()
    assert len(cycles) == 1
    assert set(cycles[0]) == {"a", "b", "c"}


# -- construction -------------------------------------------------------------


def test_a_flat_list_becomes_a_linear_graph() -> None:
    graph = FallbackGraph.from_chain(["a", "b", "c"])
    assert len(graph) == 2
    # The point of the constitution's rule: every reason gets the same answer.
    assert graph.next_hop("a", TIMEOUT) == "b"
    assert graph.next_hop("a", CONTEXT) == "b"


def test_a_single_element_chain_has_no_edges() -> None:
    assert len(FallbackGraph.from_chain(["only"])) == 0
    assert not FallbackGraph.from_chain([])


def test_config_edges_parse_their_reasons() -> None:
    graph = FallbackGraph.from_config(
        [
            {"from": "a", "to": "b", "reasons": ["timeout", "overloaded"]},
            {"from": "a", "to": "c", "on": "context_too_large"},
            {"from": "b", "to": "c"},
        ]
    )
    assert graph.next_hop("a", TIMEOUT) == "b"
    assert graph.next_hop("a", CONTEXT) == "c"
    # No reasons given means every reason.
    assert graph.next_hop("b", FallbackReason.QUALITY_FAILURE) == "c"


def test_config_rejects_unknown_reasons_and_missing_ends() -> None:
    with pytest.raises(ConfigurationError, match="unknown fallback reason"):
        FallbackGraph.from_config([{"from": "a", "to": "b", "reasons": ["catastrophe"]}])
    with pytest.raises(ConfigurationError, match="'from' and 'to'"):
        FallbackGraph.from_config([{"from": "a"}])


def test_restricting_drops_edges_that_leave_the_set() -> None:
    graph = FallbackGraph([FallbackEdge("a", "b"), FallbackEdge("b", "c")])
    restricted = graph.restricted_to(["a", "b"])
    assert len(restricted) == 1
    assert restricted.next_hop("a", TIMEOUT) == "b"


# -- budgets ------------------------------------------------------------------


def test_depth_is_bounded() -> None:
    ledger = FallbackLedger(budget=FallbackBudget(max_depth=2))
    assert ledger.refuse_reason() is None
    ledger.advance()
    assert ledger.refuse_reason() is None
    ledger.advance()
    assert "depth" in (ledger.refuse_reason() or "")


def test_a_zero_depth_budget_forbids_any_fallback() -> None:
    ledger = FallbackLedger(budget=FallbackBudget(max_depth=0))
    assert "depth" in (ledger.refuse_reason() or "")


def test_cost_is_checked_against_the_hop_being_contemplated() -> None:
    # Noticing the overspend afterwards is too late, so the projected cost of
    # the next hop counts against the ceiling.
    ledger = FallbackLedger(budget=FallbackBudget(max_depth=9, max_cost_usd=0.01))
    ledger.charge(cost_usd=0.009)
    assert ledger.refuse_reason(next_cost_usd=0.0005) is None
    assert "cost" in (ledger.refuse_reason(next_cost_usd=0.005) or "")


def test_latency_is_bounded() -> None:
    ledger = FallbackLedger(budget=FallbackBudget(max_depth=9, max_latency_ms=100.0))
    ledger.charge(latency_ms=99.0)
    assert ledger.refuse_reason() is None
    ledger.charge(latency_ms=2.0)
    assert "latency" in (ledger.refuse_reason() or "")


def test_an_unset_ceiling_never_refuses() -> None:
    ledger = FallbackLedger(budget=FallbackBudget(max_depth=99))
    ledger.charge(cost_usd=1000.0, latency_ms=1_000_000.0)
    assert ledger.refuse_reason(next_cost_usd=1000.0) is None


def test_negative_budgets_are_refused() -> None:
    for kwargs in ({"max_depth": -1}, {"max_cost_usd": -1.0}, {"max_latency_ms": -1.0}):
        with pytest.raises(ConfigurationError):
            FallbackBudget(**kwargs)  # type: ignore[arg-type]


def test_charges_ignore_negative_amounts() -> None:
    ledger = FallbackLedger(budget=FallbackBudget())
    ledger.charge(cost_usd=-5.0, latency_ms=-5.0)
    assert ledger.spent_usd == 0.0
    assert ledger.elapsed_ms == 0.0


# -- tracing ------------------------------------------------------------------


def test_a_trace_records_hops_and_refusals() -> None:
    trace = FallbackTrace()
    assert trace.depth == 0

    trace.record(FallbackHop("a", "b", TIMEOUT, depth=1, error="slow"))
    trace.refuse("budget exhausted")

    assert trace.depth == 1
    payload = trace.as_dict()
    assert payload["hops"][0] == {
        "from": "a",
        "to": "b",
        "reason": "timeout",
        "depth": 1,
        "error": "slow",
    }
    assert payload["refused"] == ["budget exhausted"]


def test_any_reason_covers_everything() -> None:
    assert frozenset(FallbackReason) == ANY_REASON
