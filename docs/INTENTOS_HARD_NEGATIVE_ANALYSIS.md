# IntentOS hard-negative analysis (v1.1, before classifier changes)

Frozen v1 artifacts were **not** modified. SHA-256 at the start of this phase:

```
5191409fb5595ee8c3b167942f1aa937ff10b272f4e22481296648ff08479b5d  datasets/eval/intentos/final-2026.08.24.json
a32a76117b1972cda56699782e5dca4fbc965253913aee819f73d8637d290865  datasets/eval/intentos/baseline-2026.08.24.json
106964c63d0879e13569e049417de77d471a7cb03055a441d203395f9137109f  datasets/intent/bootstrap.jsonl
cd74a507cd7fcc7f1afb2f5bf187a78a17becdfd4a999d4e7e01724d1c9dda9f  INTENTOS_EVALUATION.md
b7b4f570476fd4a81f313fafd74bffb90c5e6d5ca3c577594871f66fdb0c2f31  docs/INTENTOS_SUCCESS_CRITERIA.md
```

Classifier under analysis: `cascade-1:fc401d447e15`, L0–L3, `HashingEmbedder`.
L2 stop 0.70, L3 stop 0.62, L2∩L3 agreement floor 0.48 with runner-up < 0.18.

This file is diagnosis. It does not change the frozen test set.

## Shared mechanism

All five misses **abstain**. None is a confident wrong label. L3 never
clears 0.62. HashingEmbedder centroids measure lexical overlap with taxonomy
*examples*, then subtract `0.35 × cosine(prompt, hard-negative centroid)`.
That repulsion is doing real work — and with a lexical embedder it also
punishes prompts that merely share words with a taxonomy hard negative.

---

## hn-003

```
text:     Summarise what this codebase does for a new joiner
expected: summarization
acceptable: coding
predicted: unknown (abstain)
cascade confidence: 0.626
```

| Layer | Intent | Confidence | Runner-up | Margin |
| --- | --- | --- | --- | --- |
| L2 | summarization | 0.626 | coding 0.209 | 0.417 |
| L3 | coding.review | 0.137 | data_analysis 0.050 | 0.087 |
| Cascade | unknown | 0.626 | summarization | — |

**Why L2 fired and still abstained.** `summari[sz]e` scores summarization 4.5.
`codebase` scores coding 1.5. Share × evidence = 0.626, under the 0.70 stop.
This is a **class**: summarise-a-codebase-as-prose. The deliverable is a
summary; the subject happens to be code.

**Why L3 picked coding.review.** Raw cosine is highest for `coding` (0.341)
and `coding.review` (0.331). Taxonomy `coding.hard_negatives` includes
"Summarise what this repository does for a non-technical reader", which is
the same class. HN cosine to coding is 0.453, so coding is *repelled* and
review wins a weak lexical contest. Summarization examples are
article/thread/report — little overlap with "codebase" / "joiner".

**Taxonomy.** Not ambiguous about the *deliverable* (summary). Ambiguous
about whether "codebase" should pull coding. Annotation is fair:
`acceptable_intent_ids` already includes coding. Strict expected is
summarization. Not under-specified.

**Generalization, not a test-id patch.** A semantic embedder should treat
"summarise what X does" as summarization even when X is a repository.
Taxonomy already states that class as a coding hard negative.

---

## hn-005

```
text:     Explain what a segmentation fault actually is
expected: coding
acceptable: general_conversation
predicted: unknown (abstain)
cascade confidence: 0.170
```

| Layer | Intent | Confidence | Runner-up | Margin |
| --- | --- | --- | --- | --- |
| L2 | (no opinion) | 0.000 | — | — |
| L3 | reasoning | 0.170 | research 0.083 | 0.087 |
| Cascade | unknown | 0.170 | reasoning | — |

**Why L2 abstained.** `segmentation fault` would score `coding.debug` +4.0,
but `explain what a` is a −4.0 penalty on that same node, and parent
suppression then drops unmatched `coding`. Net: no rule matched. The penalty
is the right *class* (definitional CS vs a live bug); it currently zeros the
whole coding branch instead of leaving general `coding`.

**Why L3 picked reasoning.** Raw cosine actually prefers `coding.debug`
(0.335). Taxonomy `coding.debug.hard_negatives` is "Explain what a
segmentation fault is" — nearly this prompt — HN cosine 0.602. After
repulsion, debug drops and a weak `reasoning` centroid wins. `coding`
examples are all implementation tasks, so the definitional prompt has
nowhere to land.

**Taxonomy.** Debug vs "what is a segfault" is well specified. Strict
`coding` vs acceptable `general_conversation` is a mild annotation stretch:
explaining a CS term is not really "write/change software". Not
under-specified enough to drop.

**Generalization.** Semantic L3 should place definitional programming
questions nearer `coding` (or conversation) than `reasoning`. Do not add an
`if "segmentation fault"` rule.

---

## hn-008

```
text:     What would a summary of this document leave out?
expected: reasoning
acceptable: summarization
predicted: unknown (abstain)
cascade confidence: 0.399
```

| Layer | Intent | Confidence | Runner-up | Margin |
| --- | --- | --- | --- | --- |
| L2 | reasoning | 0.399 | summarization 0.399 | 0.000 |
| L3 | data_analysis | 0.147 | math 0.041 | 0.106 |
| Cascade | unknown | 0.399 | reasoning | — |

**Why L2 split.** `what would a summary` scores reasoning 4.0;
`summary`/`document` scores summarization 4.0. Equal evidence, confidence
0.399. The reasoning pattern names a **class** (meta-questions about a
summary, not a request to summarise). It is not an id-specific string of the
frozen row, but it is a thin pattern. Hashing L3 cannot break the tie.

**Why L3 picked data_analysis.** Lexical noise. Raw cosine: data_analysis
0.315, coding 0.309, reasoning 0.271. No semantic understanding of "leave
out".

**Taxonomy.** reasoning.hard_negatives includes "Summarise the argument in
this essay" (the inverse class). The prompt is a critique of a hypothetical
summary. Acceptable summarization is defensible. Not under-specified.

**Generalization.** Needs meaning: "what would a summary omit" ≠ "summarise
this". L3 semantics, not another regex.

---

## hn-010

```
text:     Explain how autonomous agents plan multi-step work
expected: general_conversation
acceptable: research, reasoning
predicted: unknown (abstain)
cascade confidence: 0.699
```

| Layer | Intent | Confidence | Runner-up | Margin |
| --- | --- | --- | --- | --- |
| L2 | agent | 0.699 | — | — |
| L3 | reasoning | 0.077 | rag 0.073 | 0.003 |
| Cascade | unknown | 0.699 | agent | — |

**Why L2 almost stopped.** `multi.step` scores agent 3.0 and nothing else
matches, so confidence is 0.6988 — **0.0012 under 0.70**. The class is
explanatory agent talk vs executable agent work. Taxonomy already has the
counterexample "Explain how autonomous agents work".

**Why L3 did not recover.** Weak, scattered cosines (rag 0.277, reasoning
0.239). `agent` is not even in the raw top-6. Hashing cannot see "explain
how X works" as conversation.

**Taxonomy.** Clear: nothing is to be executed. Label `general_conversation`
is broader than "explain this concept"; `research`/`reasoning` are listed as
acceptable. Not under-specified.

**Generalization.** Semantic L3 plus the existing agent counterexample should
dominate `multi-step` lexical fire. Do not special-case this sentence.

---

## data-002

```
text:     Which of these variables correlates most strongly with retention?
expected: data_analysis
acceptable: (none)
predicted: unknown (abstain)
cascade confidence: 0.487
hard_negative: false
```

| Layer | Intent | Confidence | Runner-up | Margin |
| --- | --- | --- | --- | --- |
| L2 | data_analysis | 0.426 | reasoning 0.372 | 0.054 |
| L3 | data_analysis | 0.487 | reasoning 0.022 | 0.465 |
| Cascade | unknown | 0.487 | data_analysis | — |

**Why L2 split.** `correlat\w+` scores data_analysis 4.0. `which of these`
scores reasoning 3.5. Combined confidence 0.426.

**Why L3 is actually right, and still unused.** Raw cosine 0.457 to
data_analysis (taxonomy example: "Which of these features correlates most
with churn?"). Share 0.740. Confidence 0.487 is under the 0.62 stop.

**Why agreement did not save it.** L2 and L3 **name the same intent**. Max
conf 0.487 ≥ agreement floor 0.48. L3 runner-up is tiny. **L2 runner-up
reasoning 0.372 ≥ 0.18**, so `_agreed_verdict` refuses. That guard exists
for multi-intent prompts; here it blocks a clear in-taxonomy match because
a reasoning keyword co-occurred.

**Taxonomy.** Not ambiguous. This is the ordinary data_analysis class. The
prompt is close to a taxonomy example; HashingEmbedder is just low-confidence
and the agreement guard is L2-split-sensitive.

**Generalization.** A real embedder should clear 0.62 on this class without
touching the agreement rule. Changing the runner-up guard to "L2-only" to
fit this row would be overfitting a cascade knob to one example.

---

## Weak-class recalls (from v1 final, not extra failures)

| Class | Recall | Reading |
| --- | --- | --- |
| summarization | 0.67 | hn-003 abstain + hn-009 extraction (lenient OK) |
| data_analysis | 0.75 | data-002 abstain (1 of 4) |
| general_conversation | 0.75 | hn-010 abstain + hn-012 tool_use (lenient OK) |

Ordinary-slice accuracy is 0.986. The remaining misses are concentrated in
hard-negative / sibling-vocabulary rows. Regex patches on those rows would
move the slice without teaching the cascade the class.

---

## HashingEmbedder (inspected)

`src/llm_fabric/intent/embeddings.py`. Hashed word unigrams/bigrams +
character 4-grams, L2-normalised. Deterministic, no weights, no meaning.
`car` vs `automobile` is unrelated. It is the test fallback, not L3 quality.

`EmbeddingProvider` already exists (`embed(texts) -> list[Vector]`,
`model_id`, `dimensions`). A real local adapter should implement that
protocol. Do not remove HashingEmbedder.

## Realistic local adapters (architecture-compatible)

| Candidate | Why it is in scope | Cost |
| --- | --- | --- |
| `HashingEmbedder` | Deterministic CI / tests | none |
| ONNX `fastembed` MiniLM / bge-small | Local, no PyTorch, provider-neutral wrapper | disk + CPU/MPS |
| `sentence-transformers` MiniLM / mpnet | Local, well known, heavier | disk + torch |
| HTTP OpenAI-compatible embeddings | Replaceable hosted/self-hosted | network |

Prefer the **smallest local model that moves hard-negative generalization**.
Do not default CI to a downloaded model.

Paid LLM L4 is optional and must stay an escalation. L5 stays disabled.

## Smallest experiment (proposed, before implementation)

1. Keep HashingEmbedder as default in tests.
2. Add a local `EmbeddingProvider` (ONNX MiniLM first, bge-small as a
   comparison if the extra is present).
3. On **validation only**, compare prototype designs: examples-centroid
   (current), name+description+examples, nearest-example. Keep the winner
   if it helps val hard-negatives without dropping unknown recall.
4. Do not fit temperature scaling on 22 val rows.
5. Expand validation with *new* class examples, not paraphrases of the
   frozen five.
6. L4: run only when L2/L3 do not accept (already true). Local description
   rerank on the shortlist, with abstain. Provider-backed JSON L4 remains
   available but is not required for the candidate if no model is present.
7. Freeze combination A as today's hashing cascade. Score B (real L3) and
   C (real L3 + sampled L4) on the **same** frozen 98. Do not tune on it.

If hard-negative accuracy still stays below 0.58 after that single
experiment: **stop**. Classify remaining misses. Do not iterate on the
frozen file.

## After cascade-1.1 (measured)

The experiment was run. Combination B (MiniLM examples, `hn_lambda=0`) and C
(B + local L4) are in `datasets/eval/intentos/candidate-1.1-*.json`.
Hard-negative accuracy stayed **0.50**. See `docs/INTENTOS_V1.1.md`.

Post-B status of the original five:

- **hn-003** L3 now ranks summarization (was coding.review under hashing) but
  margin 0.001, confidence 0.148. Still abstain. Classifier limitation.
- **hn-005** L3 ranks coding.debug. Still abstain. Insufficient prototypes
  for definitional CS.
- **hn-008** L3 ranks summarization. C’s local L4 accepts summarization
  (lenient OK, strict miss). Taxonomy ambiguity.
- **hn-010** L3 ranks reasoning; L2 still agent 0.6988. Still abstain.
  Taxonomy ambiguity.
- **data-002** L3 accepts data_analysis at 0.690. Fixed.

Remaining strict HN misses also include hn-009 and hn-012 (lenient-acceptable
L2 answers). Do not add test-id rules. Next experiment is structured L4 on an
expanded val set, not another frozen-98 loop.
