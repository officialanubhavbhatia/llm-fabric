"""Local embedder resolution. Hashing stays the default; trained models are opt-in."""

from __future__ import annotations

import pytest

from llm_fabric.intent.classifiers.embedding import EmbeddingClassifier, PrototypeKind
from llm_fabric.intent.classifiers.rerank import LocalRerankClassifier
from llm_fabric.intent.embeddings import HashingEmbedder, RealLocalEmbedder, resolve_embedder
from llm_fabric.intent.schema import ClassificationRequest
from llm_fabric.intent.taxonomy import IntentNode, IntentTaxonomy


def test_hashing_is_the_default_embedder() -> None:
    embedder = resolve_embedder(None)
    assert isinstance(embedder, HashingEmbedder)
    assert resolve_embedder("hashing").model_id == HashingEmbedder().model_id


def test_unknown_embedder_names_fail_closed() -> None:
    with pytest.raises(ValueError, match="unknown intent embedder"):
        resolve_embedder("openai-text-embedding-3")


def test_local_embedder_is_optional() -> None:
    if RealLocalEmbedder.available():
        embedder = resolve_embedder("minilm")
        assert embedder.dimensions > 0
        assert "local:" in embedder.model_id
    else:
        with pytest.raises(RuntimeError, match="no local embedding backend"):
            resolve_embedder("local")


@pytest.mark.asyncio
async def test_mixed_prototypes_can_use_a_description_without_examples() -> None:
    taxonomy = IntentTaxonomy(
        "v1",
        [
            IntentNode(
                intent_id="described_only",
                name="Described only",
                description="summarising long documents into a short gist",
            ),
            IntentNode(
                intent_id="exemplified",
                name="Exemplified",
                description="has examples",
                examples=("write a python parser",),
            ),
        ],
    )
    classifier = EmbeddingClassifier(HashingEmbedder(dimensions=256), prototype=PrototypeKind.MIXED)
    verdict = await classifier.classify(
        ClassificationRequest(text="summarising long documents into a short gist"), taxonomy
    )
    assert verdict.intent_id == "described_only"


@pytest.mark.asyncio
async def test_parent_hard_negative_repels_children() -> None:
    """A parent HN meaning 'looks like coding, is not' must also repel children.

    Residual lexical mass otherwise slides onto coding.review. This is a class
    of prompts (prose overview of a repository), not a frozen test id.
    """
    taxonomy = IntentTaxonomy(
        "v1",
        [
            IntentNode(
                intent_id="coding",
                name="Coding",
                description="Writing software",
                examples=("implement a parser in rust",),
                hard_negatives=("give me a prose overview of this repository",),
            ),
            IntentNode(
                intent_id="coding.review",
                name="Review",
                description="Assessing a diff",
                examples=("review this pull request for races",),
            ),
            IntentNode(
                intent_id="summarization",
                name="Summarisation",
                description="Condensing supplied content",
                examples=(
                    "summarise this article in three bullets",
                    "give me the gist of the meeting notes",
                ),
            ),
        ],
    )
    prompt = "give me a prose overview of this repository"
    propagating = EmbeddingClassifier(HashingEmbedder(), hn_lambda=0.35)
    isolated = EmbeddingClassifier(HashingEmbedder(), hn_lambda=0.35, propagate_ancestor_hn=False)
    await propagating.prepare(taxonomy)
    await isolated.prepare(taxonomy)
    vector = await propagating.embed_prompt(prompt)
    with_prop = dict(propagating.adjusted_similarities(vector))
    without = dict(isolated.adjusted_similarities(vector))
    assert with_prop["coding.review"] <= without["coding.review"]
    assert with_prop["summarization"] >= with_prop["coding.review"]


@pytest.mark.asyncio
async def test_local_rerank_abstains_on_nonsense() -> None:
    from llm_fabric.intent.bootstrap import bootstrap_taxonomy

    tax = bootstrap_taxonomy()
    rerank = LocalRerankClassifier(HashingEmbedder())
    verdict = await rerank.classify(ClassificationRequest(text="asdf qwer zxcv 1234"), tax)
    assert verdict.has_opinion is False
    assert "abstain" in verdict.rationale


@pytest.mark.asyncio
async def test_embedding_classifier_survives_a_missing_centroid_prepare() -> None:
    classifier = EmbeddingClassifier(HashingEmbedder(dimensions=32))
    assert classifier.score((0.0,) * 32).has_opinion is False
