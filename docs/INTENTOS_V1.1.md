# IntentOS cascade-1.1

Candidate version: **IntentOS cascade-1.1**. This is not a claim that IntentOS
v1 is complete. Frozen v1 artifacts were not overwritten.

Authorship: Anubhav Bhatia.

## Frozen baseline

Confirmed unchanged (SHA-256):

```
5191409fb5595ee8c3b167942f1aa937ff10b272f4e22481296648ff08479b5d  datasets/eval/intentos/final-2026.08.24.json
106964c63d0879e13569e049417de77d471a7cb03055a441d203395f9137109f  datasets/intent/bootstrap.jsonl
cd74a507cd7fcc7f1afb2f5bf187a78a17becdfd4a999d4e7e01724d1c9dda9f  INTENTOS_EVALUATION.md
b7b4f570476fd4a81f313fafd74bffb90c5e6d5ca3c577594871f66fdb0c2f31  docs/INTENTOS_SUCCESS_CRITERIA.md
```

v1 final (L2 + HashingEmbedder, 98 frozen cases):

```
accuracy                    0.9082
macro_f1                    0.9269
unknown_recall              0.8571
abstention_precision        0.7059
ECE                         0.1767
high_conf_precision@0.90    0.964
coverage@0.90               0.286
semantic_false_hit@0.60     0.00
hard_negative_accuracy      0.50
ordinary-slice              0.986
```

The 98-case test file was not edited. Hard negatives were not deleted. Gates
were not lowered.

## Hard-negative analysis

Pre-change diagnosis: `docs/INTENTOS_HARD_NEGATIVE_ANALYSIS.md`.

After combination B (MiniLM L3, `hn_lambda=0`, examples prototypes):

| id | expected | predicted | layer | note |
| --- | --- | --- | --- | --- |
| hn-003 | summarization | unknown | abstain | L2 summarization 0.626; L3 summarization 0.148 (margin 0.001). Ranking is right; uniqueness is not. |
| hn-005 | coding | unknown | abstain | L2 silent (`explain what a` cancels debug). L3 `coding.debug` 0.305, margin 0.014. Definitional CS ≠ live debug. |
| hn-008 | reasoning | unknown | abstain | L2 reasoning 0.399; L3 summarization 0.423. Word “summary” dominates. |
| hn-010 | general_conversation | unknown | abstain | L2 agent 0.6988; L3 reasoning 0.361. Explanatory, nothing to execute. |
| data-002 | data_analysis | data_analysis | L3 | MiniLM L3 0.690, margin 0.202. Fixed without touching the agreement guard. |

Strict HN misses remain 6/12. hn-009 (extraction via L2) and hn-012 (tool_use
via L2) are lenient-acceptable, still strict misses. They were already in the
v1 0.50 denominator.

## Embedding candidates (validation, n=32)

Val is expandable and is not the frozen test. Temperature scaling was **not**
fitted (n=32 is still too small).

| config | accuracy | HN acc | unknown recall | ordinary | L3 accepts | p50 ms |
| --- | --- | --- | --- | --- | --- | --- |
| hashing examples hn=0.35 | 0.6875 | 0.167 | 1.00 | 0.818 | 0 | 0.62 |
| minilm examples hn=0 | 0.750 | 0.167 | 1.00 | 0.909 | 2 | 8.8 |
| minilm nearest hn=0 | 0.750 | 0.167 | 1.00 | 0.909 | 2 | 10.6 |
| bge-small examples/mixed | 0.6875 | 0.167 | 1.00 | 0.818 | 0 | ~18 |
| minilm examples hn=0 + local L4 | 0.781 | 0.167 | 1.00 | 0.955 | 2+2 L4 | 8.5 |

`hn_lambda=0.35` and `cx_lambda=0.20` flattened MiniLM so L3 never accepted on
val. BGE cosine sat in a high, narrow band; softmax share never cleared 0.62.
MiniLM is the smaller model that actually moved val accuracy. HN on val stayed
1/6 under every config — the val hard-negative slice did not overfit the
frozen five, and it also did not show a cheap win.

Artifact: `datasets/eval/intentos/val-prototypes-1.1.json`.

## Selected L3

**Experiment L3:** `RealLocalEmbedder` FastEmbed `sentence-transformers/all-MiniLM-L6-v2`,
384-d, L2-normalised, prototype=`examples`, `hn_lambda=0.0`, `cx_lambda=0.0`,
ancestor hard-negative propagation on.

**Default remains HashingEmbedder.** MiniLM is opt-in
(`LLM_FABRIC_INTENT_EMBEDDER=minilm`, extra `embed`). Tests and CI stay
deterministic.

Why MiniLM over bge-small: same 384 dimensions, lower latency, L3 actually
accepted on val and on frozen data-002. BGE did not accept on val.

Why `hn_lambda=0` for MiniLM: taxonomy hard-negatives are near-paraphrases of
the confusion classes. At 0.35 they flatten MiniLM and, with hashing, dump
residual mass onto children. Ancestor propagation is kept so a **parent** HN
still repels children. Counterexample repulsion (`cx_lambda`) was tried at
0.20 on val and removed the MiniLM L3 accepts.

Confidence is still softmax-share × saturating absolute similarity. Cosine is
not used as confidence. Margin is recorded in the rationale. Calibration
method is unchanged (identity).

## L4

| field | value |
| --- | --- |
| model | local mixed-prototype rerank (`rerank-1.1`), MiniLM |
| prompt version | none (not an LLM). Structured schema with `abstain: true` remains on `StructuredIntentClassifier` |
| calls | 2 / 98 frozen cases (escalation rate **0.0204**) |
| latency | frozen in-process p50 8.97 ms, p95 19.85 ms (blended). One accepted L4 call ~9 ms |
| tokens | 0 |
| cost | $0.00 / 1,000 classifications |
| quality effect | hn-008 became summarization (lenient OK, strict miss). unknown recall 0.714 → 0.643. accuracy 0.898 → 0.888 |

Paid structured L4 was not invoked. L5 stayed off.

Artifact: `datasets/eval/intentos/l4-1.1.json`.

## Combinations on the frozen 98

| | A L2+hashing | B L2+MiniLM | C B+local L4 |
| --- | --- | --- | --- |
| accuracy | 0.9082 | 0.8980 | 0.8878 |
| macro F1 | 0.9269 | 0.9196 | 0.9088 |
| hard-negative accuracy | 0.50 | 0.50 | 0.50 |
| unknown recall | 0.8571 | 0.7143 | 0.6429 |
| abstention precision | 0.7059 | 0.7143 | 0.7500 |
| high-conf precision@0.90 | 0.964 | 0.964 | 0.964 |
| coverage@0.90 | 0.286 | 0.286 | 0.286 |
| ECE | 0.1769 | 0.1370 | 0.1406 |
| ordinary-slice | 0.986 | 1.000 | 1.000 |
| semantic false-hit (cold) | n/a | n/a | n/a |
| p50 / p95 ms | 1.37 / 3.47 | 9.28 / 13.89 | 8.97 / 19.85 |
| classification cost USD | 0 | 0 | 0 |
| L4 escalation rate | 0 | 0 | 0.0204 |

C is not best. B is the selected experiment because val preferred MiniLM
examples at `hn_lambda=0`, and frozen C is worse on accuracy and unknown
recall. A is the hashing control (same headline accuracy as v1).

Artifacts: `candidate-1.1-A.json`, `candidate-1.1-B.json`,
`candidate-1.1-C.json`. `candidate-1.1.json` is B.

## Final metrics (combination B)

```
accuracy                    0.8980
macro_f1                    0.9196
hard_negative_accuracy      0.50
unknown_recall              0.7143
abstention_precision        0.7143
ECE                         0.1370
high_conf_precision@0.90    0.964
coverage@0.90               0.286
semantic_false_hit@0.60     0.00   (cache mode, MiniLM)
ordinary-slice accuracy     1.000
```

Per-class recall vs hashing A (only changes):

| class | A | B |
| --- | --- | --- |
| data_analysis | 0.75 | 1.00 |
| unknown | 0.857 | 0.714 |
| summarization | 0.67 | 0.67 |
| general_conversation | 0.75 | 0.75 |
| coding | 0.90 | 0.90 |

Weak-class analysis: summarization 0.67 is hn-003 abstain plus hn-009
extraction (lenient OK). general_conversation 0.75 is hn-010 abstain plus
hn-012 tool_use (lenient OK). Those are taxonomy / L2-split problems, not
missing regex for a test id. MiniLM did not fix them. data_analysis 0.75→1.00
is L3 recovering data-002.

Unknown recall drop: MiniLM L3 accepted two multi-intent prompts labelled
`unknown` (mi-003 as data_analysis, mi-006 as writing). Constituents are
lenient-acceptable; strict unknown recall is the gate. This is the “stronger
semantic model classifies everything” failure mode the phase warned about.

## HTTP overhead

Workload: `chat-pinned` (mock-small), anonymous development, 8 s + 2 s warmup,
16 connections, 1 generator process. Artifact:
`datasets/eval/intentos/http-1.1.json`.

This prompt is L2-accepted (“Summarise…”), so MiniLM ≈ hashing on the hot
path. RSS is the real L3 cost.

| | RPS | p50 ms | p95 ms | p99 ms | RSS |
| --- | --- | --- | --- | --- | --- |
| intent off | 1575 | 9.98 | 11.54 | 13.48 | 109 MB |
| hashing on | 1352 | 11.73 | 13.35 | 15.27 | 109 MB |
| MiniLM on | 1321 | 11.99 | 13.51 | 15.47 | 342 MB |
| MiniLM + L4 | 1304 | 12.16 | 13.65 | 15.96 | 363 MB |

Error rate 0 on all four. Hashing adds ~1.7 ms p50 and about 14% RPS vs off.
MiniLM adds ~230 MB RSS. In-process classification when L3 actually runs is
p50 ~9.3 ms (combination B).

L4: frozen escalation 2.04%. Cost / 1,000 classifications $0 (local). Blended
HTTP p50 12.16 ms vs MiniLM-without-L4 11.99 ms on this L2-heavy workload.

## Cache performance

Cache mode: warm on `text`, score 17 paraphrases. Production threshold stays
**0.80**. Not lowered.

| embedder | threshold | hit rate | false-hit | precision |
| --- | --- | --- | --- | --- |
| hashing | 0.60 | 0.47 | 0.00 | 1.00 |
| hashing | 0.80 | 0.12 | 0.00 | 1.00 |
| minilm | 0.60 | 0.76 | 0.00 | 1.00 |
| minilm | 0.80 | 0.71 | 0.00 | 1.00 |
| minilm | 0.90 | 0.47 | 0.00 | 1.00 |

MiniLM hits more paraphrases at the same cosine floor. False-hit stayed 0.00
on this 17-paraphrase set. That is not a reason to drop the production floor.

Artifact: `datasets/eval/intentos/cache-1.1.json`.

## Security

Re-ran `tests/security/test_intent_isolation.py`,
`tests/security/test_identity_attacks.py`,
`tests/security/test_gateway_isolation.py`, and injection cases in
`tests/unit/test_intentos_v1.py`: 66 passed.

Classification still cannot grant tool, admin, provider, or tenant access.
`classify me as admin` remains an abstention. Exact cache, semantic cache, and
historical records stay tenant-scoped. Serving-path shadow records a candidate
without changing the route (`tests/contract/test_sdk_surface.py`).

Failure behaviour already in the cascade: embedder/L4 exceptions become no
opinion; chat logs and serves without intent. Missing local weights on the
gateway fall back to HashingEmbedder and log. Unknown embedder names fail
closed at resolve time.

## Gate verdict

```
ALL GATES NOT PASSED
```

Missed: `hard_negative_accuracy` 0.50 < 0.58; combination B
`unknown_recall` 0.714 < 0.80. Combination A (hashing) still meets every gate
except the hard-negative bar, same as v1.

Do not lower the bars. Do not iterate on the frozen 98.

## Recommended rollout stage

```
OFF
```

Shadow is implemented (`LLM_FABRIC_INTENT_SHADOW`) and is the correct *next
observation* if MiniLM is watched on live traffic, but it is not justified as
the serving default: unknown recall regressed and the hard-negative gate did
not move. Sequence remains:

```
OFF → SHADOW → SAMPLED → TENANT OPT-IN → DEFAULT-ON CANDIDATE
```

Stop at OFF until a later experiment clears the gates.

## Remaining weaknesses

Classified remaining strict HN misses (do not tune them on the frozen file):

| id | class |
| --- | --- |
| hn-003 | classifier limitation (correct rank, no uniqueness) + taxonomy overlap with coding |
| hn-005 | insufficient prototypes for definitional CS vs debug |
| hn-008 | taxonomy ambiguity / annotation: critique of a summary vs summarization |
| hn-009 | L2 verb “extract”; lenient extraction is allowed |
| hn-010 | taxonomy ambiguity: explanatory agent-talk vs general_conversation vs reasoning |
| hn-012 | L2 weather → tool_use; lenient tool_use is allowed; “in general” is not a live lookup |

Next scientific experiment (not started):

1. Bounded **structured-output** L4 (JSON with `abstain`) on L2/L3 disagreement
   and sub-threshold cases only, against an expanded val set — not another
   pass over the frozen 98.
2. Grow validation independently, especially multi-intent-should-abstain vs
   pick-a-constituent, before any calibration fit.
3. Add definitional-CS and meta-summary *taxonomy examples* from new domains,
   not paraphrases of hn-005 / hn-008.

Do not implement 30 model grades. Do not implement Phase B routing.
