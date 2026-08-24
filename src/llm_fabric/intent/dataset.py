"""Evaluation dataset hygiene for IntentOS.

A classifier that is tuned against a broken benchmark will look improved and
be worse at routing. This module audits labelled JSONL before anyone treats a
score as evidence.

Splits are first-class. `train` examples must never be scored as test.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from llm_fabric.intent.benchmark import BenchmarkCase, load_dataset
from llm_fabric.intent.embeddings import HashingEmbedder, cosine_similarity

Split = Literal["train", "validation", "test", "hard_negative", "ood", "adversarial"]

NEAR_DUPLICATE_SIMILARITY = 0.92
SHORT_PROMPT_CHARS = 24
LONG_PROMPT_CHARS = 1_200


@dataclass(frozen=True, slots=True)
class DatasetAudit:
    path: str
    cases: int
    labels: int
    by_label: dict[str, int]
    class_balance_ratio: float | None
    duplicate_rate: float
    near_duplicate_rate: float
    short_prompts: int
    long_prompts: int
    languages: dict[str, int]
    hard_negatives: int
    ood_or_unknown: int
    ambiguous: int
    with_conversation: int
    overlap_with_other: int
    notes: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "cases": self.cases,
            "labels": self.labels,
            "by_label": self.by_label,
            "class_balance_ratio": self.class_balance_ratio,
            "duplicate_rate": round(self.duplicate_rate, 4),
            "near_duplicate_rate": round(self.near_duplicate_rate, 4),
            "short_prompts": self.short_prompts,
            "long_prompts": self.long_prompts,
            "languages": self.languages,
            "hard_negatives": self.hard_negatives,
            "ood_or_unknown": self.ood_or_unknown,
            "ambiguous": self.ambiguous,
            "with_conversation": self.with_conversation,
            "overlap_with_other": self.overlap_with_other,
            "notes": list(self.notes),
        }


def load_split(path: Path | str, *, split: Split | None = None) -> list[BenchmarkCase]:
    """Load a JSONL dataset, optionally keeping one split.

    Cases without a `split` field are treated as `test` so the historical
    bootstrap file remains a frozen test set.
    """
    cases = load_dataset(path)
    if split is None:
        return cases
    return [case for case in cases if _split_of(case) == split]


def audit_dataset(
    path: Path | str,
    *,
    other_texts: Iterable[str] = (),
) -> DatasetAudit:
    cases = load_dataset(path)
    labels: Counter[str] = Counter(case.expected_intent_id for case in cases)
    languages: Counter[str] = Counter(case.language for case in cases)
    texts = [case.text.strip().lower() for case in cases]
    exact_dupes = sum(count - 1 for count in Counter(texts).values() if count > 1)

    near = _near_duplicate_pairs(texts)
    other = {normalise_for_overlap(text) for text in other_texts if text.strip()}
    overlap = sum(1 for text in texts if normalise_for_overlap(text) in other)

    counts: list[int] = list(labels.values())
    minimum = min(counts) if counts else 0
    balance = (max(counts) / minimum) if minimum else None

    notes: list[str] = []
    if balance is not None and balance >= 4:
        notes.append(f"class imbalance: most frequent label is {balance:.1f}× the least frequent")
    if exact_dupes:
        notes.append(f"{exact_dupes} exact duplicate prompt(s)")
    if near:
        notes.append(f"{len(near)} near-duplicate pair(s) at cosine ≥ {NEAR_DUPLICATE_SIMILARITY}")
    if overlap:
        notes.append(
            f"{overlap} prompt(s) overlap another corpus (taxonomy examples or train split)"
        )
    unknown = sum(1 for case in cases if case.expects_abstention)
    if unknown / len(cases) < 0.05:
        notes.append("OOD/unknown coverage is thin; unknown-intent recall will be noisy")

    return DatasetAudit(
        path=str(path),
        cases=len(cases),
        labels=len(labels),
        by_label=dict(sorted(labels.items())),
        class_balance_ratio=round(balance, 3) if balance is not None else None,
        duplicate_rate=exact_dupes / len(cases) if cases else 0.0,
        near_duplicate_rate=(2 * len(near)) / len(cases) if cases else 0.0,
        short_prompts=sum(1 for case in cases if len(case.text) < SHORT_PROMPT_CHARS),
        long_prompts=sum(1 for case in cases if len(case.text) > LONG_PROMPT_CHARS),
        languages=dict(sorted(languages.items())),
        hard_negatives=sum(1 for case in cases if case.hard_negative),
        ood_or_unknown=unknown,
        ambiguous=sum(1 for case in cases if case.acceptable_intent_ids),
        with_conversation=sum(1 for case in cases if case.conversation_context),
        overlap_with_other=overlap,
        notes=tuple(notes),
    )


def taxonomy_example_texts(nodes: Sequence[object]) -> list[str]:
    texts: list[str] = []
    for node in nodes:
        texts.extend(getattr(node, "examples", ()) or ())
        texts.extend(getattr(node, "counterexamples", ()) or ())
        texts.extend(getattr(node, "hard_negatives", ()) or ())
    return texts


def normalise_for_overlap(text: str) -> str:
    collapsed = re.sub(r"\s+", " ", text.strip().lower())
    return collapsed


def _split_of(case: BenchmarkCase) -> str:
    return case.split or "test"


def _near_duplicate_pairs(texts: Sequence[str]) -> list[tuple[int, int]]:
    if len(texts) < 2:
        return []
    embedder = HashingEmbedder(dimensions=256)
    vectors = [embedder.embed_one(text) for text in texts]
    pairs: list[tuple[int, int]] = []
    for i, left in enumerate(vectors):
        for j in range(i + 1, len(vectors)):
            if cosine_similarity(left, vectors[j]) >= NEAR_DUPLICATE_SIMILARITY:
                pairs.append((i, j))
    return pairs


def prompt_hash(text: str) -> str:
    return hashlib.sha256(normalise_for_overlap(text).encode("utf-8")).hexdigest()[:24]
