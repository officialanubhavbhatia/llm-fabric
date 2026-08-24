"""Production sampling and dataset creation from traces.

Sampled examples are unlabelled unless the source already carried a label.
Nothing here invents an `expected` field from a prompt.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

from llm_fabric.observability.metering import UsageRecord
from llm_fabric.observability.otel import RecordedSpan
from llm_fabric.storage.records import EvalDataset, EvalExample
from llm_fabric.tenancy.scope import TenantScope


def sample_usage(
    records: Sequence[UsageRecord],
    *,
    scope: TenantScope,
    rate: float,
    limit: int,
    name: str = "production-sample",
) -> EvalDataset:
    """Keep every 1/rate-th record, deterministically, without inventing labels."""
    if not 0.0 < rate <= 1.0:
        raise ValueError("sample rate must lie in (0, 1]")
    period = max(1, round(1.0 / rate))
    chosen = list(records)[::period][:limit]
    examples = tuple(
        EvalExample(
            input=f"request:{record.request_id}",
            expected=record.intent_id,
            metadata={
                "source": "usage",
                "id": record.request_id,
                "request_id": record.request_id,
                "trace_id": record.trace_id,
                "served_model": record.served_model,
                "policy": record.policy,
                "failover_count": record.failover_count,
                "labelled": record.intent_id is not None,
            },
        )
        for record in chosen
    )
    return EvalDataset(
        tenant_id=scope.tenant_id,
        name=name,
        description=(
            "Sampled from this process's usage buffer. expected is the intent "
            "id when classification ran, otherwise absent."
        ),
        examples=examples,
    )


def dataset_from_traces(
    traces: Sequence[dict[str, object]] | Sequence[object],
    *,
    scope: TenantScope,
    name: str = "trace-dataset",
) -> EvalDataset:
    """One example per trace. No expected label unless the trace already has one."""
    examples: list[EvalExample] = []
    for raw in traces:
        if isinstance(raw, dict):
            trace_id = str(raw.get("trace_id") or "")
            spans = raw.get("spans") or []
            names = [str(span.get("name")) for span in spans if isinstance(span, dict)]
        else:
            continue
        if not trace_id:
            continue
        examples.append(
            EvalExample(
                input=trace_id,
                expected=None,
                metadata={
                    "source": "trace",
                    "id": trace_id,
                    "trace_id": trace_id,
                    "span_names": names,
                    "labelled": False,
                },
            )
        )
    return EvalDataset(
        tenant_id=scope.tenant_id,
        name=name,
        description="Created from stored traces. Unlabelled; expected is absent.",
        examples=tuple(examples),
    )


def dataset_from_spans(
    spans: Sequence[RecordedSpan],
    *,
    scope: TenantScope,
    name: str = "span-dataset",
) -> EvalDataset:
    by_trace: dict[str, list[RecordedSpan]] = {}
    for span in spans:
        by_trace.setdefault(span.trace_id, []).append(span)
    examples = tuple(
        EvalExample(
            input=trace_id,
            expected=None,
            metadata={
                "source": "span_journal",
                "id": trace_id,
                "span_names": [span.name for span in group],
                "labelled": False,
            },
        )
        for trace_id, group in by_trace.items()
    )
    return EvalDataset(tenant_id=scope.tenant_id, name=name, examples=examples)


def content_fingerprint(dataset: EvalDataset) -> str:
    payload = "|".join(example.input for example in dataset.examples)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]
