"""Embedding interface, plus an offline implementation for tests and local runs.

The interface is the point. A real deployment plugs in a hosted or self-hosted
embedding model; the classifier and the semantic cache only ever see
`EmbeddingProvider`.

`HashingEmbedder` exists so the suite runs deterministically with no network and
no model weights. **It is not a semantic model.** It is a hashed character- and
word-n-gram vectoriser: it measures lexical overlap and nothing else. It will
score "translate this to French" and "translate this to German" as near
identical, and it will score "car" and "automobile" as unrelated. Where it is
used as the default, that is a decision about determinism, not a claim about
quality.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence
from typing import Any, Protocol, cast, runtime_checkable

from llm_fabric.tenancy.cache import CacheNamespace, TenantScopedCache
from llm_fabric.tenancy.scope import TenantScope

Vector = tuple[float, ...]

_TOKEN = re.compile(r"[a-z0-9]+")


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Turns text into vectors."""

    @property
    def model_id(self) -> str:
        """Stable identifier, recorded in classifier versions and cache keys."""
        ...

    @property
    def dimensions(self) -> int: ...

    async def embed(self, texts: Sequence[str]) -> list[Vector]:
        """Embed a batch. Order of the result matches order of the input."""
        ...


class HashingEmbedder:
    """A deterministic, offline, purely lexical vectoriser.

    Uses hashed word unigrams and bigrams plus character 4-grams, with sublinear
    term weighting and L2 normalisation. Good enough for the pipeline to run and
    for tests to be repeatable. Not a substitute for a trained embedding model.
    """

    def __init__(self, dimensions: int = 512, *, seed: str = "llm-fabric") -> None:
        if dimensions <= 0:
            raise ValueError("dimensions must be positive")
        self._dimensions = dimensions
        self._seed = seed

    @property
    def model_id(self) -> str:
        return f"hashing-{self._dimensions}d"

    @property
    def dimensions(self) -> int:
        return self._dimensions

    async def embed(self, texts: Sequence[str]) -> list[Vector]:
        return [self.embed_one(text) for text in texts]

    def embed_one(self, text: str) -> Vector:
        counts: dict[int, float] = {}
        for feature in self._features(text):
            index = self._bucket(feature)
            counts[index] = counts.get(index, 0.0) + 1.0

        if not counts:
            return tuple([0.0] * self._dimensions)

        vector = [0.0] * self._dimensions
        for index, count in counts.items():
            # Sublinear scaling, so a word repeated twenty times does not swamp
            # every other signal in a short prompt.
            vector[index] = 1.0 + math.log(count)

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0.0:
            return tuple(vector)
        return tuple(value / norm for value in vector)

    def _features(self, text: str) -> list[str]:
        lowered = text.lower()
        words = _TOKEN.findall(lowered)

        features: list[str] = [f"w:{word}" for word in words]
        features.extend(
            f"b:{first}_{second}" for first, second in zip(words, words[1:], strict=False)
        )

        # Character n-grams give partial credit for morphology and typos, which
        # word features alone cannot do.
        squashed = " ".join(words)
        features.extend(f"c:{squashed[i : i + 4]}" for i in range(max(0, len(squashed) - 3)))
        return features

    def _bucket(self, feature: str) -> int:
        digest = hashlib.blake2b(
            feature.encode("utf-8"), digest_size=8, key=self._seed.encode("utf-8")
        ).digest()
        return int.from_bytes(digest, "big") % self._dimensions


#: Local ONNX models FastEmbed can load without a paid API. MiniLM is the
#: smaller of the two; bge-small is the quality default when the extra is
#: installed. Neither is downloaded by tests — HashingEmbedder stays the default.
LOCAL_MINILM_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LOCAL_BGE_SMALL_MODEL = "BAAI/bge-small-en-v1.5"


class RealLocalEmbedder:
    """A trained local embedding model behind `EmbeddingProvider`.

    Prefers FastEmbed (ONNX, no PyTorch). Falls back to sentence-transformers
    if that extra is what is installed. Missing weights are an error for this
    class; callers that must run offline should use `HashingEmbedder`.
    """

    def __init__(self, model_id: str = LOCAL_BGE_SMALL_MODEL) -> None:
        backend, model, dimensions = _load_local_backend(model_id)
        self._backend = backend
        self._model = model
        self._model_id = model_id
        self._dimensions = dimensions

    @classmethod
    def available(cls) -> bool:
        return _fastembed_available() or _sentence_transformers_available()

    @property
    def model_id(self) -> str:
        return f"local:{self._backend}:{self._model_id}"

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def backend(self) -> str:
        return self._backend

    async def embed(self, texts: Sequence[str]) -> list[Vector]:
        if not texts:
            return []
        model = cast(Any, self._model)
        if self._backend == "fastembed":
            raw = list(model.embed(list(texts)))
            return [tuple(float(value) for value in row) for row in raw]
        encoded = model.encode(list(texts), normalize_embeddings=True)
        return [tuple(float(value) for value in row) for row in encoded]


def resolve_embedder(name: str | None = None) -> EmbeddingProvider:
    """Build an embedder from a short name. Unknown names fail closed."""
    key = (name or "hashing").strip().lower()
    if key in {"hashing", "hash", "lexical"}:
        return HashingEmbedder()
    if key in {"local", "bge-small", "bge_small", LOCAL_BGE_SMALL_MODEL.lower()}:
        return RealLocalEmbedder(LOCAL_BGE_SMALL_MODEL)
    if key in {"minilm", "mini-lm", LOCAL_MINILM_MODEL.lower()}:
        return RealLocalEmbedder(LOCAL_MINILM_MODEL)
    raise ValueError(f"unknown intent embedder {name!r}; use hashing, local/bge-small, or minilm")


def _fastembed_available() -> bool:
    try:
        import fastembed  # noqa: F401
    except ImportError:
        return False
    return True


def _sentence_transformers_available() -> bool:
    try:
        import sentence_transformers  # type: ignore[import-not-found]  # noqa: F401
    except ImportError:
        return False
    return True


def _load_local_backend(model_id: str) -> tuple[str, object, int]:
    if _fastembed_available():
        from fastembed import TextEmbedding

        model = TextEmbedding(model_name=model_id)
        probe = next(iter(model.embed(["dimension probe"])))
        return "fastembed", model, int(len(probe))
    if _sentence_transformers_available():
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(model_id)
        return "sentence-transformers", model, int(model.get_sentence_embedding_dimension())
    raise RuntimeError(
        "no local embedding backend is installed. "
        "Install the optional extra `embed` (fastembed) or use HashingEmbedder."
    )


class CachingEmbedder:
    """Wraps an embedding provider with the tenant-scoped embedding cache.

    Embeddings are stable for a given (model, text) pair, so recomputing them is
    pure waste. The cache is tenant-scoped like every other: two tenants
    embedding the same string do not share an entry, because the embedding cache
    is a side channel like any other cache.
    """

    def __init__(
        self,
        inner: EmbeddingProvider,
        cache: TenantScopedCache,
        scope: TenantScope,
    ) -> None:
        self._inner = inner
        self._cache = cache
        self._scope = scope

    @property
    def model_id(self) -> str:
        return self._inner.model_id

    @property
    def dimensions(self) -> int:
        return self._inner.dimensions

    async def embed(self, texts: Sequence[str]) -> list[Vector]:
        results: list[Vector | None] = []
        misses: list[tuple[int, str]] = []

        for index, text in enumerate(texts):
            cached = self._cache.get(
                self._scope,
                CacheNamespace.EMBEDDING,
                {"model": self._inner.model_id, "text": text},
            )
            if isinstance(cached, tuple):
                results.append(cached)
            else:
                results.append(None)
                misses.append((index, text))

        if misses:
            computed = await self._inner.embed([text for _, text in misses])
            # strict: an embedder returning a different number of vectors than
            # it was given texts has misaligned them, and silently zipping to
            # the shorter list would cache each vector against the wrong text.
            for (index, text), vector in zip(misses, computed, strict=True):
                results[index] = vector
                self._cache.put(
                    self._scope,
                    CacheNamespace.EMBEDDING,
                    {"model": self._inner.model_id, "text": text},
                    vector,
                )

        return [vector for vector in results if vector is not None]


def cosine_similarity(left: Vector, right: Vector) -> float:
    """Cosine similarity, clamped to [0, 1].

    Clamped because these values feed confidence thresholds, and a small
    negative from floating-point drift would be meaningless there. Inputs of
    differing length are a programming error, not a degenerate case to smooth
    over.
    """
    if len(left) != len(right):
        raise ValueError(
            f"cannot compare vectors of different dimensions: {len(left)} and {len(right)}"
        )

    dot = 0.0
    left_norm = 0.0
    right_norm = 0.0
    for a, b in zip(left, right, strict=True):
        dot += a * b
        left_norm += a * a
        right_norm += b * b

    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0

    similarity = dot / (math.sqrt(left_norm) * math.sqrt(right_norm))
    return max(0.0, min(1.0, similarity))


def centroid(vectors: Sequence[Vector]) -> Vector:
    """Mean vector, re-normalised. Empty input yields the zero vector."""
    if not vectors:
        return ()

    dimensions = len(vectors[0])
    total = [0.0] * dimensions
    for vector in vectors:
        if len(vector) != dimensions:
            raise ValueError("cannot average vectors of differing dimensions")
        for index, value in enumerate(vector):
            total[index] += value

    norm = math.sqrt(sum(value * value for value in total))
    if norm == 0.0:
        return tuple(total)
    return tuple(value / norm for value in total)
