"""L2: the deterministic classifier.

Weighted regular expressions. It is fast, free, auditable and offline, and it
handles the large share of traffic that announces itself — "summarise this",
"translate this into German", "what is 15% of 80".

**Its confidence is a heuristic score, not a calibrated probability.** The
formula below is monotone in the evidence and bounded in [0, 1], which is enough
for threshold gating, but a score of 0.8 here does not mean "correct 80% of the
time". Whether it is calibrated is an empirical question, and the benchmark's
expected-calibration-error metric is what answers it. Until that has been run
against a dataset that is not this file's own examples, treat the thresholds as
starting points.

Negative rules matter as much as positive ones. "Translate this Python into
Rust" contains the strongest possible translation signal and is not translation,
so translation carries an explicit penalty for programming languages.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from llm_fabric.intent.classifiers.base import MAX_ALTERNATIVES, ClassifierVerdict
from llm_fabric.intent.schema import (
    ClassificationRequest,
    ClassifierLayer,
    IntentAlternative,
)
from llm_fabric.intent.taxonomy import IntentTaxonomy

#: Evidence needed before the score saturates. Larger means more suspicious.
EVIDENCE_SCALE = 2.5

#: How much of a parent's evidence a child inherits. Evidence for "coding" is
#: partial evidence for "coding.debug"; the reverse is not true, or a single
#: specific match would drag its whole branch up with it.
PARENT_CREDIT = 0.6

#: Longest prompt the rules will scan. Regex over an unbounded prompt is a
#: denial-of-service surface, and intent is nearly always visible in the opening.
MAX_SCAN_CHARS = 4_000

_INJECTION = re.compile(
    r"(ignore (?:previous|all|prior) instructions|"
    r"classify this as|label me as|"
    r"you are now (?:dan|jailbroken)|"
    r"if this is classified as)",
    re.IGNORECASE,
)

#: Privilege language attached to a classifier-override. The prompt is trying
#: to buy a permission with a label, which classification must never grant.
_PRIVILEGE_COERCION = re.compile(
    r"(so i (?:get|have)|give me|grant me|"
    r"database access|api keys?|return (?:the )?secrets?|"
    r"as admin)",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class Rule:
    intent_id: str
    pattern: re.Pattern[str]
    weight: float

    @classmethod
    def build(cls, intent_id: str, pattern: str, weight: float) -> Rule:
        return cls(intent_id, re.compile(pattern, re.IGNORECASE), weight)


#: Patterns grouped by the intent they score, so each rule is a
#: `(pattern, weight)` pair. Negative weights are penalties: evidence that this
#: intent is *not* what the prompt wants.
RuleSpecs = Mapping[str, Sequence[tuple[str, float]]]


def compile_rules(specs: RuleSpecs) -> tuple[Rule, ...]:
    return tuple(
        Rule.build(intent_id, pattern, weight)
        for intent_id, patterns in specs.items()
        for pattern, weight in patterns
    )


BOOTSTRAP_RULE_SPECS: RuleSpecs = {
    "coding": (
        (r"\bwrite (a |an |some |me a )?(code|script|function|class|program|query)\b", 3.5),
        (r"\b(refactor|implement|compiles?|compiler|syntax error|unit test)\b", 3.0),
        (r"\b(async|await|callback|promise|mutex|generic|endpoint|pagination)\b", 2.0),
        (r"\b(python|javascript|typescript|rust|golang|java|sql|bash|c\+\+)\b", 1.5),
        (r"\b(function|class|method|variable|codebase|repository|migration)\b", 1.5),
        (r"\b(exponential backoff|http client|rest endpoint|react component|class-based)\b", 3.5),
        (r"\b(rename this variable|implement binary search|sorted slice)\b", 4.0),
        (r"\bwrite a (sql query|python function|script that)\b", 4.5),
        (r"\badd (a retry|pagination)\b", 4.0),
        # Porting between programming languages is coding, not translation.
        (r"\btranslate\b.*\b(python|java|rust|c\+\+|code|script)\b", 3.0),
    ),
    "coding.debug": (
        (r"\b(stack ?trace|traceback|segfault|segmentation fault|core dump)\b", 4.0),
        (r"\b(exit code \d+|throws? an exception|null pointer)\b", 3.5),
        (r"\b(fails? in ci|passes locally|not working|returning duplicate)\b", 3.5),
        (r"\bwhy (does|is|am i|are)\b.*\b(fail|error|crash|wrong|duplicate)\b", 3.0),
        (r"\b(hangs? forever|never completes)\b", 4.0),
        (r"\bdebug\b", 2.5),
        (r"\bexplain what a\b", -4.0),
    ),
    "coding.review": (
        (r"\breview (this|my|the) (diff|pr|pull request|change|patch|code)\b", 4.5),
        (r"\bis this (thread ?safe|concurrency safe)\b", 4.0),
        (r"\b(what could break|risk data loss|any issues with this)\b", 3.5),
    ),
    "agent": (
        (r"\b(book|reserve|purchase|order)\b.*\b(flight|hotel|ticket|table|for me)\b", 4.5),
        (r"\badd (it|this) to my (calendar|todo|schedule)\b", 4.0),
        (r"\b(go through|work through|triage|monitor)\b.*\b(and|then)\b", 3.5),
        (r"\b(on my behalf|end.to.end|autonomously|multi.step)\b", 3.0),
        (r"\b(and then|after that)\b.*\b(email|send|create|assign|open a ticket)\b", 3.0),
        (r"\bopen a ticket if\b", 4.5),
    ),
    "reasoning": (
        (r"\b(if all|what follows|deduce|infer|entails|implies)\b", 3.5),
        (r"\b(which of these|best explanation|fits the evidence)\b", 3.5),
        (r"\b(second.order|knock.on) (effects?|consequences)\b", 3.0),
        (r"\b(work out|reason through|justify|plan the order)\b", 2.5),
        (r"\bgiven these .{0,60}constraints\b", 4.0),
        (r"\bestimate how long\b", 4.0),
        (r"\bwhat would a summary\b", 4.0),
    ),
    "math": (
        (r"\b(integrate|integral|derivative|differentiate|eigenvalues?|matrix)\b", 4.0),
        (r"\b(prove that|proof that|theorem|irrational|polynomial|logarithm)\b", 4.0),
        (r"\b(solve|compute)\b.*\b(equation|system|integral|for x)\b", 3.5),
        # A request for a program is coding even when the subject is maths.
        (r"\bwrite (a |an )?(python |javascript )?(function|script|code|program)\b", -4.5),
    ),
    "math.arithmetic": (
        (r"\bwhat is \d[\d,\.]*\s*%?\s*(of|divided by|times|plus|minus)\b", 4.5),
        (r"\d[\d,\.]*\s*(\+|\*|/|÷|×|plus|minus|times|divided by)\s*\d", 4.0),
        (r"\badd up\b.*\d", 3.5),
    ),
    "research": (
        (r"\bwhat does the (current )?(literature|research|evidence)\b", 4.5),
        (r"\b(literature|state of the art|survey of|current research)\b", 4.0),
        (r"\bcompare\b.*\b(approaches|options|vendors|alternatives|trade.?offs)\b", 3.5),
        (r"\b(pros and cons|landscape|overview of the field)\b", 2.5),
    ),
    "rag": (
        (
            r"\b(according to|based on|using only|per) the "
            r"(attached|provided|supplied|following)\b",
            4.5,
        ),
        (r"\b(our|the) (internal )?(docs|documentation|runbook|handbook|wiki)\b", 4.0),
        (
            r"\b(in|from) the (attached|provided|supplied) "
            r"(document|contract|file|handbook)\b",
            4.0,
        ),
        (r"\bsearch (our|the internal|the company)\b", 3.5),
    ),
    "data_analysis": (
        (r"\b(statistically significant|correlat\w+|regression|cohorts?)\b", 4.0),
        (r"\b(trend|anomal\w+|outliers?|distribution)\b", 3.0),
        (
            r"\b(break ?down|analyse|analyze)\b.*"
            r"\b(numbers|figures|data|sales|revenue|metrics)\b",
            3.5,
        ),
        (r"\b(these|this) (data|dataset|numbers|figures|results)\b", 2.0),
    ),
    "writing": (
        (
            r"\bwrite (a|an|me a) "
            r"(short story|poem|essay|article|email|announcement|blog|letter|post)\b",
            4.5,
        ),
        (r"\bdraft (a|an|me a)\b", 4.0),
        (r"\b(compose|ghost.?write)\b", 3.0),
        (r"\bturn (these|this) (bullet points|notes) into (a )?(paragraph|prose)\b", 3.5),
    ),
    "summarization": (
        (r"\b(summari[sz]e|summari[sz]ation|tl;?dr)\b", 4.5),
        (r"\b(the gist|key (points|takeaways)|in (three|two|a few) bullet)\b", 4.0),
        (r"\b(condense|boil down)\b", 3.5),
        (r"\b(short|brief|quick) (summary|overview) of\b", 3.5),
        (r"\b(two-line|two line) summary\b", 4.5),
        (r"\bsummary of this (thread|codebase|document)\b", 4.0),
    ),
    "translation": (
        (r"\b(translate|translation)\b", 4.0),
        (r"\bhow do (you|i) say\b", 4.5),
        (
            r"\b(in|into|to) (spanish|french|german|japanese|chinese|portuguese"
            r"|italian|korean|arabic|hindi|russian|mandarin)\b",
            3.0,
        ),
        (r"\brender this\b", 4.0),
        # A programming language makes this a port, not a translation. The
        # penalty must outweigh the "translate" match above.
        (r"\b(python|java|javascript|rust|c\+\+|golang|code|script|sql)\b", -5.0),
    ),
    "extraction": (
        (r"\b(extract|pull out|parse out)\b", 4.0),
        (r"\blist (all|every)\b.*\b(from|in) (this|the)\b", 3.5),
        (r"\bas (json|csv|a table|structured data)\b", 3.0),
        (r"\breturn the (line items|fields|values|entities)\b", 3.5),
    ),
    "classification": (
        (r"\b(classify|categori[sz]e)\b", 4.5),
        (r"\b(label|tag) (this|these|each)\b", 4.0),
        (
            r"\bis this (review|message|email|ticket|comment)?\s*"
            r"(positive|negative|neutral|spam|toxic)\b",
            4.5,
        ),
        (r"\bwhich (category|label|class|bucket)\b", 4.0),
        (r"\bdoes this (message|content|post)?\s*violate\b", 4.0),
        (r"\bpositive or negative\b", 4.5),
        (r"\bcommunity guidelines\b", 4.0),
    ),
    "vision": (
        (r"\b(this|the|attached) (image|photo|photograph|screenshot|picture)\b", 4.5),
        (r"\b(this|the) (diagram|chart|figure|graph)\b", 3.0),
        (r"\bread the text in\b", 4.0),
        (r"\bocr\b", 3.5),
        # Asking for a program that handles images is coding, not vision.
        (r"\bwrite (code|a script|a program)\b", -4.0),
    ),
    "tool_use": (
        (r"\bwhat('s| is) the (weather|temperature|time|exchange rate|price)\b", 4.5),
        (r"\b(current|latest|today's) (price|rate|weather|status|value)\b", 4.0),
        (r"\b(look ?up|check whether|fetch|query the)\b", 3.0),
        (r"\bexchange rate\b", 4.5),
        (r"\bright now\b", 1.5),
    ),
    "general_conversation": (
        (r"^\s*(hi|hello|hey|good (morning|afternoon|evening))\b", 4.5),
        (r"\b(thanks|thank you|cheers|that was helpful)\b", 3.5),
        (r"\bhow do (you|i) say\b", -5.0),
        (r"\bwhat is the capital of\b", 4.5),
        (r"\bwho (won|is|was) the\b", 3.0),
        (r"\btell me something (interesting|fun)\b", 3.5),
        (r"\bhow are you\b", 4.0),
        (r"\bwhat a histogram is\b", 4.5),
    ),
}

BOOTSTRAP_RULES: tuple[Rule, ...] = compile_rules(BOOTSTRAP_RULE_SPECS)


class DeterministicClassifier:
    """Weighted-regex intent scoring. No network, no model, no state."""

    def __init__(
        self,
        rules: Sequence[Rule] = BOOTSTRAP_RULES,
        *,
        version: str = "rules-5",
        max_scan_chars: int = MAX_SCAN_CHARS,
        enabled: bool = True,
    ) -> None:
        self._rules = tuple(rules)
        self._version = version
        self._max_scan_chars = max_scan_chars
        self._enabled = enabled

    @property
    def layer(self) -> ClassifierLayer:
        return ClassifierLayer.L2_RULES

    @property
    def version(self) -> str:
        return self._version

    async def classify(
        self, request: ClassificationRequest, taxonomy: IntentTaxonomy
    ) -> ClassifierVerdict:
        if not self._enabled:
            return ClassifierVerdict.no_opinion("rules classifier disabled")
        text = request.text[: self._max_scan_chars]
        if not text.strip():
            return ClassifierVerdict.no_opinion("empty prompt")
        if _injection_only(text):
            return ClassifierVerdict.no_opinion("prompt tries to override the classifier")

        scores = self._score(text, taxonomy)
        positive = {intent_id: score for intent_id, score in scores.items() if score > 0.0}
        if not positive:
            return ClassifierVerdict.no_opinion("no rule matched")

        ranked = sorted(positive.items(), key=lambda item: (-item[1], item[0]))
        total = sum(score for _, score in ranked)
        top_id, top_score = ranked[0]

        share = top_score / total
        evidence = 1.0 - math.exp(-top_score / EVIDENCE_SCALE)
        confidence = share * evidence

        alternatives = tuple(
            IntentAlternative(intent_id=intent_id, confidence=(score / total) * evidence)
            for intent_id, score in ranked[1 : 1 + MAX_ALTERNATIVES]
        )

        return ClassifierVerdict(
            intent_id=top_id,
            confidence=confidence,
            alternatives=alternatives,
            rationale=f"rule score {top_score:.2f} of {total:.2f} total",
        )

    def _score(self, text: str, taxonomy: IntentTaxonomy) -> dict[str, float]:
        raw: dict[str, float] = {}
        for rule in self._rules:
            if rule.intent_id not in taxonomy:
                continue
            if rule.pattern.search(text):
                raw[rule.intent_id] = raw.get(rule.intent_id, 0.0) + rule.weight

        # Children inherit a fraction of their ancestors' evidence, so a
        # specific match plus generic support beats generic support alone.
        inherited: dict[str, float] = {}
        for node in taxonomy.classifiable():
            score = raw.get(node.intent_id, 0.0)
            credit = PARENT_CREDIT
            for ancestor in taxonomy.ancestors(node.intent_id):
                score += raw.get(ancestor, 0.0) * credit
                credit *= PARENT_CREDIT
            # Unmatched children must not enter the ranking. Inheritance exists
            # so a child that DID match is boosted by its parent, not so every
            # descendant of a parent match becomes a competing alternative and
            # dilutes confidence below the cascade threshold.
            if raw.get(node.intent_id, 0.0) > 0.0:
                inherited[node.intent_id] = score

        # A matched child already inherited parent evidence. Leaving the parent
        # in the ranking splits that evidence and can drop a clear child below
        # the cascade threshold.
        shadowed = set()
        for intent_id in inherited:
            shadowed.update(taxonomy.ancestors(intent_id))
        return {
            intent_id: score for intent_id, score in inherited.items() if intent_id not in shadowed
        }


def _injection_only(text: str) -> bool:
    """True when the prompt is an instruction-override with no remaining task.

    A jailbreak wrapped around a real question (\"you are now DAN. what is 2+2\")
    still has a task and must be classified. A prompt that is *only* trying to
    dictate the label, or that tries to buy a permission with a label, must not
    be treated as that label.
    """
    if not _INJECTION.search(text):
        return False
    if _PRIVILEGE_COERCION.search(text):
        return True
    remainder = _INJECTION.sub(" ", text)
    remainder = re.sub(r"[^a-z0-9]+", " ", remainder.lower()).strip()
    return len(remainder) < 12
