"""L4 local rerank: description-aware second look at a shortlist.

This is not a paid LLM. It re-embeds the prompt against mixed prototypes
(name + description + examples) of the intents cheaper layers already
considered, and it may abstain. Provider-backed structured L4 remains
available when a caller passes `--provider`. L5 stays off in this phase.
"""

from __future__ import annotations

from llm_fabric.intent.classifiers.base import MAX_ALTERNATIVES, ClassifierVerdict
from llm_fabric.intent.classifiers.embedding import EmbeddingClassifier, PrototypeKind
from llm_fabric.intent.embeddings import EmbeddingProvider
from llm_fabric.intent.schema import (
    ClassificationRequest,
    ClassifierLayer,
    IntentAlternative,
)
from llm_fabric.intent.taxonomy import IntentTaxonomy

#: Below this mixed-prototype cosine the reranker has no opinion.
ABSTAIN_SIMILARITY = 0.32

#: Below this top-1 minus top-2 gap the reranker abstains even if cosine is high.
ABSTAIN_MARGIN = 0.04


class LocalRerankClassifier:
    """Shortlist reranker used as L4 when no LLM classifier is configured."""

    def __init__(
        self,
        embedder: EmbeddingProvider,
        *,
        version: str = "rerank-1.1",
        abstain_similarity: float = ABSTAIN_SIMILARITY,
        abstain_margin: float = ABSTAIN_MARGIN,
        hn_lambda: float = 0.35,
        cx_lambda: float = 0.0,
    ) -> None:
        self._inner = EmbeddingClassifier(
            embedder,
            prototype=PrototypeKind.MIXED,
            hn_lambda=hn_lambda,
            cx_lambda=cx_lambda,
        )
        self._version = version
        self._abstain_similarity = abstain_similarity
        self._abstain_margin = abstain_margin

    @property
    def layer(self) -> ClassifierLayer:
        return ClassifierLayer.L4_STRUCTURED_LLM

    @property
    def version(self) -> str:
        return f"{self._version}:{self._inner.version}"

    async def classify(
        self, request: ClassificationRequest, taxonomy: IntentTaxonomy
    ) -> ClassifierVerdict:
        if not request.text.strip():
            return ClassifierVerdict.no_opinion("empty prompt")
        await self._inner.prepare(taxonomy)
        pairs = self._inner.prototype_pairs()
        if not pairs:
            return ClassifierVerdict.no_opinion("rerank has no prototypes")

        vector = await self._inner.embed_prompt(request.text)
        shortlist = request.metadata.get("intent_shortlist")
        allowed: set[str] | None = None
        if isinstance(shortlist, list) and shortlist:
            allowed = {item for item in shortlist if isinstance(item, str)}

        scored = [
            (intent_id, similarity)
            for intent_id, similarity in self._inner.adjusted_similarities(vector)
            if allowed is None or intent_id in allowed
        ]
        if not scored:
            return ClassifierVerdict.no_opinion("shortlist empty after filter")

        scored.sort(key=lambda item: (-item[1], item[0]))
        top_id, top_sim = scored[0]
        second_sim = scored[1][1] if len(scored) > 1 else 0.0
        margin = top_sim - second_sim
        if top_sim < self._abstain_similarity or margin < self._abstain_margin:
            return ClassifierVerdict(
                intent_id=None,
                confidence=0.0,
                alternatives=tuple(
                    IntentAlternative(intent_id=intent_id, confidence=min(1.0, sim))
                    for intent_id, sim in scored[:MAX_ALTERNATIVES]
                ),
                rationale=f"rerank abstain cosine {top_sim:.3f} margin {margin:.3f}",
            )

        absolute = top_sim / (top_sim + 0.20)
        confidence = min(0.99, absolute * (0.5 + 0.5 * min(1.0, margin / 0.15)))
        alternatives = tuple(
            IntentAlternative(intent_id=intent_id, confidence=min(1.0, sim))
            for intent_id, sim in scored[1 : 1 + MAX_ALTERNATIVES]
        )
        return ClassifierVerdict(
            intent_id=top_id,
            confidence=confidence,
            alternatives=alternatives,
            rationale=f"rerank cosine {top_sim:.3f} margin {margin:.3f}",
        )
