"""IntentOS: deciding what a prompt is actually asking for.

The subsystem is a cascade of increasingly expensive classifiers behind two
caches, over a versioned taxonomy. Most prompts should never reach a model, and
a prompt nobody can classify confidently is answered with `unknown` rather than
a guess.

Nothing in this package claims to classify well. Whether it does is a question
for `llm_fabric.intent.benchmark` and a labelled dataset.
"""

from llm_fabric.intent.bootstrap import BOOTSTRAP_TAXONOMY_VERSION, bootstrap_taxonomy
from llm_fabric.intent.cache import (
    ExactIntentCache,
    IntentCacheDiscriminators,
    IntentCacheStats,
    SemanticCachePolicy,
    SemanticIntentCache,
)
from llm_fabric.intent.cascade import (
    CandidateBuffer,
    CandidateExample,
    CascadeThresholds,
    IntentCascade,
    IntentDecision,
    LayerAttempt,
)
from llm_fabric.intent.embeddings import EmbeddingProvider, HashingEmbedder, RealLocalEmbedder
from llm_fabric.intent.factory import build_full_cascade, build_offline_cascade
from llm_fabric.intent.metrics import IntentMetrics
from llm_fabric.intent.schema import (
    UNKNOWN_INTENT_ID,
    ClassificationRequest,
    ClassifierLayer,
    Complexity,
    ContextClass,
    CostClass,
    IntentAlternative,
    IntentClassification,
    IntentProfile,
    LatencyClass,
    Modality,
    PrivacyClass,
    QualityClass,
    ReasoningLevel,
    RiskClass,
    SafetyClass,
)
from llm_fabric.intent.taxonomy import (
    IntentNode,
    IntentStatus,
    IntentTaxonomy,
    TaxonomyRegistry,
)

__all__ = [
    "BOOTSTRAP_TAXONOMY_VERSION",
    "UNKNOWN_INTENT_ID",
    "CandidateBuffer",
    "CandidateExample",
    "CascadeThresholds",
    "ClassificationRequest",
    "ClassifierLayer",
    "Complexity",
    "ContextClass",
    "CostClass",
    "EmbeddingProvider",
    "ExactIntentCache",
    "HashingEmbedder",
    "RealLocalEmbedder",
    "IntentAlternative",
    "IntentCacheDiscriminators",
    "IntentCacheStats",
    "IntentCascade",
    "IntentClassification",
    "IntentDecision",
    "IntentMetrics",
    "IntentNode",
    "IntentProfile",
    "IntentStatus",
    "IntentTaxonomy",
    "LatencyClass",
    "LayerAttempt",
    "Modality",
    "PrivacyClass",
    "QualityClass",
    "ReasoningLevel",
    "RiskClass",
    "SafetyClass",
    "SemanticCachePolicy",
    "SemanticIntentCache",
    "TaxonomyRegistry",
    "bootstrap_taxonomy",
    "build_full_cascade",
    "build_offline_cascade",
]
