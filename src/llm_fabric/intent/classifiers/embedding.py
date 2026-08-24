"""L3: the embedding classifier.

Each classifiable intent becomes a centroid of its example embeddings; a prompt
is assigned to the nearest centroid. This catches phrasings the rules never
anticipated, which is the entire reason it sits above L2.

It is an *adapter*: the quality of this layer is the quality of the
`EmbeddingProvider` behind it, and nothing here compensates for a weak one. With
the default offline `HashingEmbedder` this layer measures lexical overlap, so it
will disagree with a real embedding model on exactly the paraphrases that matter
most. Swapping in a trained model is the intended deployment.

Confidence combines two things that are easy to conflate. A softmax over
centroid similarities says *how clearly this intent beat the others*; a
saturating function of the raw similarity says *whether the prompt looked like
anything at all*. A prompt equidistant from every centroid scores low on the
first; a prompt far from all of them scores low on the second. Both are needed:
the softmax alone would report high confidence for a nonsense prompt that
happened to be marginally nearer one centroid.

As with the rules layer, these numbers are **uncalibrated** until the benchmark
measures calibration error against a dataset that is not the taxonomy's own
examples.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

from llm_fabric.intent.classifiers.base import MAX_ALTERNATIVES, ClassifierVerdict
from llm_fabric.intent.embeddings import (
    EmbeddingProvider,
    Vector,
    centroid,
    cosine_similarity,
)
from llm_fabric.intent.schema import (
    ClassificationRequest,
    ClassifierLayer,
    IntentAlternative,
)
from llm_fabric.intent.taxonomy import IntentNode, IntentTaxonomy

#: Softmax temperature. Lower sharpens the distribution. Cosine similarities sit
#: in a narrow band, so the temperature has to be small to separate them at all.
DEFAULT_TEMPERATURE = 0.05

#: Similarity at which the absolute-evidence term reaches one half.
SIMILARITY_MIDPOINT = 0.20

#: Weight of the hard-negative centroid repulsion. Measured on validation, not
#: the frozen test set; 0.35 is the v1 default and remains until val says otherwise.
DEFAULT_HN_LAMBDA = 0.35

#: Weight of counterexample repulsion. Zero keeps v1 hashing behaviour; a
#: non-zero value is selected only after a validation sweep.
DEFAULT_CX_LAMBDA = 0.0

#: Longest prompt embedded. Beyond this, intent is diluted rather than clarified,
#: and embedding cost grows without bound.
MAX_EMBED_CHARS = 4_000


class PrototypeKind(StrEnum):
    """How an intent is represented in vector space."""

    EXAMPLES = "examples"
    DESCRIPTION = "description"
    MIXED = "mixed"
    NEAREST = "nearest"


@dataclass(frozen=True, slots=True)
class _Centroids:
    taxonomy_version: str
    model_id: str
    intent_ids: tuple[str, ...]
    vectors: tuple[Vector, ...]
    hard_negative_vectors: tuple[Vector | None, ...]
    counterexample_vectors: tuple[Vector | None, ...]
    example_vectors: tuple[tuple[Vector, ...], ...]
    prototype: PrototypeKind
    #: For each intent, the centroid indices whose hard-negatives apply to it
    #: (itself plus ancestors). A parent HN meaning "looks like coding, is not"
    #: must also repel coding.debug and coding.review, or residual mass slides
    #: onto the child.
    ancestor_hn_indices: tuple[tuple[int, ...], ...]


class EmbeddingClassifier:
    """Nearest-centroid (or nearest-example) classification over intent texts."""

    def __init__(
        self,
        embedder: EmbeddingProvider,
        *,
        temperature: float = DEFAULT_TEMPERATURE,
        similarity_midpoint: float = SIMILARITY_MIDPOINT,
        max_embed_chars: int = MAX_EMBED_CHARS,
        prototype: PrototypeKind = PrototypeKind.EXAMPLES,
        hn_lambda: float = DEFAULT_HN_LAMBDA,
        cx_lambda: float = DEFAULT_CX_LAMBDA,
        propagate_ancestor_hn: bool = True,
    ) -> None:
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")
        if similarity_midpoint <= 0.0:
            raise ValueError("similarity_midpoint must be positive")
        if not 0.0 <= hn_lambda <= 1.0:
            raise ValueError("hn_lambda must lie in [0, 1]")
        if not 0.0 <= cx_lambda <= 1.0:
            raise ValueError("cx_lambda must lie in [0, 1]")

        self._embedder = embedder
        self._temperature = temperature
        self._midpoint = similarity_midpoint
        self._max_embed_chars = max_embed_chars
        self._prototype = prototype
        self._hn_lambda = hn_lambda
        self._cx_lambda = cx_lambda
        self._propagate_ancestor_hn = propagate_ancestor_hn
        self._centroids: _Centroids | None = None

    @property
    def layer(self) -> ClassifierLayer:
        return ClassifierLayer.L3_EMBEDDING

    @property
    def version(self) -> str:
        flags = "anc" if self._propagate_ancestor_hn else "noanc"
        return (
            f"embedding-1.1:{self._embedder.model_id}:"
            f"{self._prototype.value}:hn{self._hn_lambda:.2f}:"
            f"cx{self._cx_lambda:.2f}:{flags}"
        )

    @property
    def embedder(self) -> EmbeddingProvider:
        return self._embedder

    def prototype_pairs(self) -> tuple[tuple[str, Vector], ...]:
        """Prepared (intent_id, prototype) pairs, or empty before `prepare`."""
        resolved = self._centroids
        if resolved is None:
            return ()
        return tuple(zip(resolved.intent_ids, resolved.vectors, strict=True))

    async def prepare(self, taxonomy: IntentTaxonomy) -> None:
        """Build prototypes. Idempotent for a given taxonomy, model and mode."""
        current = self._centroids
        if (
            current is not None
            and current.taxonomy_version == taxonomy.version
            and current.model_id == self._embedder.model_id
            and current.prototype == self._prototype
        ):
            return

        intent_ids: list[str] = []
        texts: list[str] = []
        spans: list[tuple[int, int]] = []
        hn_texts: list[str] = []
        hn_spans: list[tuple[int, int] | None] = []
        cx_texts: list[str] = []
        cx_spans: list[tuple[int, int] | None] = []

        for node in taxonomy.classifiable():
            examples = _prototype_texts(node, self._prototype, self._max_embed_chars)
            if not examples:
                continue
            start = len(texts)
            texts.extend(examples)
            spans.append((start, len(texts)))
            intent_ids.append(node.intent_id)
            negatives = tuple(node.hard_negatives)[:8]
            if negatives:
                hn_start = len(hn_texts)
                hn_texts.extend(item[: self._max_embed_chars] for item in negatives)
                hn_spans.append((hn_start, len(hn_texts)))
            else:
                hn_spans.append(None)
            counters = tuple(node.counterexamples)[:8]
            if counters:
                cx_start = len(cx_texts)
                cx_texts.extend(item[: self._max_embed_chars] for item in counters)
                cx_spans.append((cx_start, len(cx_texts)))
            else:
                cx_spans.append(None)

        if not texts:
            self._centroids = _empty_centroids(
                taxonomy.version, self._embedder.model_id, self._prototype
            )
            return

        embedded = await self._embedder.embed(texts)
        vectors = tuple(centroid(embedded[start:end]) for start, end in spans)
        per_intent = tuple(tuple(embedded[start:end]) for start, end in spans)
        hn_vectors = await _embed_optional_groups(
            self._embedder, hn_texts, hn_spans, len(intent_ids)
        )
        cx_vectors = await _embed_optional_groups(
            self._embedder, cx_texts, cx_spans, len(intent_ids)
        )
        id_to_index = {intent_id: index for index, intent_id in enumerate(intent_ids)}
        ancestor_hn_indices = tuple(
            _ancestor_indices(intent_id, id_to_index, taxonomy) for intent_id in intent_ids
        )

        self._centroids = _Centroids(
            taxonomy_version=taxonomy.version,
            model_id=self._embedder.model_id,
            intent_ids=tuple(intent_ids),
            vectors=vectors,
            hard_negative_vectors=tuple(hn_vectors),
            counterexample_vectors=tuple(cx_vectors),
            example_vectors=per_intent,
            prototype=self._prototype,
            ancestor_hn_indices=ancestor_hn_indices,
        )

    async def classify(
        self, request: ClassificationRequest, taxonomy: IntentTaxonomy
    ) -> ClassifierVerdict:
        if not request.text.strip():
            return ClassifierVerdict.no_opinion("empty prompt")

        await self.prepare(taxonomy)
        centroids = self._centroids
        if centroids is None or not centroids.intent_ids:
            return ClassifierVerdict.no_opinion("no intent has examples to embed")

        embedded = await self._embedder.embed([request.text[: self._max_embed_chars]])
        if not embedded:
            return ClassifierVerdict.no_opinion("embedder returned nothing")

        return self.score(embedded[0], centroids)

    def adjusted_similarities(
        self, vector: Vector, centroids: _Centroids | None = None
    ) -> list[tuple[str, float]]:
        """HN-adjusted cosine per intent, unsorted. Empty if unprepared."""
        resolved = centroids or self._centroids
        if resolved is None or not resolved.intent_ids:
            return []
        if resolved.prototype is PrototypeKind.NEAREST:
            similarities = [
                max((cosine_similarity(vector, example) for example in examples), default=0.0)
                for examples in resolved.example_vectors
            ]
        else:
            similarities = [cosine_similarity(vector, candidate) for candidate in resolved.vectors]
        hn_raw = [
            cosine_similarity(vector, repulsive) if repulsive is not None else 0.0
            for repulsive in resolved.hard_negative_vectors
        ]
        cx_raw = [
            cosine_similarity(vector, repulsive) if repulsive is not None else 0.0
            for repulsive in resolved.counterexample_vectors
        ]
        adjusted: list[tuple[str, float]] = []
        for index, similarity in enumerate(similarities):
            penalty = hn_raw[index] if index < len(hn_raw) else 0.0
            if self._propagate_ancestor_hn and resolved.ancestor_hn_indices:
                penalty = max(
                    (hn_raw[source] for source in resolved.ancestor_hn_indices[index]),
                    default=penalty,
                )
            contrast = self._hn_lambda * penalty
            if index < len(cx_raw):
                contrast += self._cx_lambda * cx_raw[index]
            adjusted.append((resolved.intent_ids[index], max(0.0, similarity - contrast)))
        return adjusted

    async def embed_prompt(self, text: str) -> Vector:
        embedded = await self._embedder.embed([text[: self._max_embed_chars]])
        return embedded[0] if embedded else ()

    def score(self, vector: Vector, centroids: _Centroids | None = None) -> ClassifierVerdict:
        resolved = centroids or self._centroids
        if resolved is None or not resolved.intent_ids:
            return ClassifierVerdict.no_opinion("classifier not prepared")

        pairs = self.adjusted_similarities(vector, resolved)
        if not pairs or not any(similarity for _, similarity in pairs):
            return ClassifierVerdict.no_opinion("prompt is orthogonal to every intent")

        similarities = [similarity for _, similarity in pairs]
        top_similarity = max(similarities)
        if top_similarity < 0.08:
            return ClassifierVerdict.no_opinion(
                f"nearest centroid cosine {top_similarity:.3f} is below the unknown floor"
            )

        weights = _softmax(similarities, self._temperature)
        ranked = sorted(
            zip((intent_id for intent_id, _ in pairs), weights, similarities, strict=True),
            key=lambda item: (-item[1], item[0]),
        )

        top_id, top_weight, top_similarity = ranked[0]
        second_similarity = ranked[1][2] if len(ranked) > 1 else 0.0
        margin = max(0.0, top_similarity - second_similarity)
        absolute = top_similarity / (top_similarity + self._midpoint)
        # Softmax share: the winner beat the field. Absolute similarity: the
        # prompt looked like *something*. Margin is recorded, not multiplied in
        # — that would suppress clear winners whose softmax is already peaked.
        # Cosine is never used as confidence.
        confidence = top_weight * absolute

        alternatives = tuple(
            IntentAlternative(
                intent_id=intent_id,
                confidence=weight * (similarity / (similarity + self._midpoint)),
            )
            for intent_id, weight, similarity in ranked[1 : 1 + MAX_ALTERNATIVES]
        )

        return ClassifierVerdict(
            intent_id=top_id,
            confidence=confidence,
            alternatives=alternatives,
            rationale=(f"cosine {top_similarity:.3f}, share {top_weight:.3f}, margin {margin:.3f}"),
        )


def _prototype_texts(node: IntentNode, kind: PrototypeKind, max_chars: int) -> tuple[str, ...]:
    """The strings that represent one intent for a given prototype design."""
    examples = tuple(example[:max_chars] for example in node.examples)
    heading = f"{node.name}. {node.description}".strip()[:max_chars]
    if kind is PrototypeKind.EXAMPLES:
        return examples
    if kind is PrototypeKind.DESCRIPTION:
        return (heading,) if heading else examples
    if kind is PrototypeKind.MIXED:
        if heading and examples:
            return (heading, *examples)
        return examples or ((heading,) if heading else ())
    return examples


def _softmax(values: Sequence[float], temperature: float) -> list[float]:
    scaled = [value / temperature for value in values]
    ceiling = max(scaled)
    exponentiated = [math.exp(value - ceiling) for value in scaled]
    total = sum(exponentiated)
    if total == 0.0:
        return [1.0 / len(values)] * len(values)
    return [value / total for value in exponentiated]


def _empty_centroids(taxonomy_version: str, model_id: str, prototype: PrototypeKind) -> _Centroids:
    return _Centroids(
        taxonomy_version=taxonomy_version,
        model_id=model_id,
        intent_ids=(),
        vectors=(),
        hard_negative_vectors=(),
        counterexample_vectors=(),
        example_vectors=(),
        prototype=prototype,
        ancestor_hn_indices=(),
    )


async def _embed_optional_groups(
    embedder: EmbeddingProvider,
    texts: list[str],
    spans: list[tuple[int, int] | None],
    count: int,
) -> list[Vector | None]:
    vectors: list[Vector | None] = [None] * count
    if not texts:
        return vectors
    embedded = await embedder.embed(texts)
    for index, span in enumerate(spans):
        if span is None:
            continue
        vectors[index] = centroid(embedded[span[0] : span[1]])
    return vectors


def _ancestor_indices(
    intent_id: str, id_to_index: dict[str, int], taxonomy: IntentTaxonomy
) -> tuple[int, ...]:
    own = id_to_index[intent_id]
    inherited = [
        id_to_index[ancestor]
        for ancestor in taxonomy.ancestors(intent_id)
        if ancestor in id_to_index
    ]
    return (own, *inherited)
