# Evaluations

What has actually been measured, by which run, with which caveats. If a number
is not in this file, it has not been measured, and no part of this repository is
entitled to claim it.

---

## 1. What the numbers below are not

Read this before the tables.

**The dataset is self-authored.** `datasets/intent/bootstrap.jsonl` was written
by the same author as the classifier rules, in the same sitting. It shares that
author's idea of what each intent means and what a typical prompt looks like.
That makes it a smoke test and a regression tripwire. It is **not** evidence
about production traffic, and it is not an unbiased evaluation of anything.

**It is small.** 98 cases across 16 labels. Several per-intent figures rest on
four or five examples, where a single case moves recall by twenty points. Treat
per-intent numbers as directional at best.

**The labels are judgements.** Whether "Estimate how long it would take to
rewrite this service" is `reasoning` or `coding` is arguable. Cases like that
carry `acceptable_intent_ids` and `notes`, and both strict and lenient accuracy
are reported so it is visible how much of the score depends on those calls.

**No comparison has been run.** No other classifier, service or model has been
evaluated on this dataset. There is therefore no basis for any statement that
this classifier is better, faster or cheaper than an alternative, and no such
statement appears anywhere in this repository.

**The thresholds were not tuned against this dataset**, deliberately. Fitting
the cascade's confidence thresholds to the same 98 cases that are then reported
would make the reported numbers meaningless. The defaults in
`CascadeThresholds` are documented judgements; `llm-fabric-bench` exposes
`--rules-threshold` and friends so they can be tuned against *your* data, on a
split you do not report.

---

## 2. Intent classifier

Current IntentOS v1 measurements, including the frozen baseline from
2026-08-24, are in [`INTENTOS_EVALUATION.md`](../INTENTOS_EVALUATION.md). The
tables in this section are a **historical** cascade run and are not rewritten.

**Run:** `make bench-intent`
**Configuration:** offline cascade — L0 exact cache, L1 semantic cache, L2 rules,
L3 embedding. **L4 and L5 were not enabled**, so no model was consulted.
**Embedder:** `HashingEmbedder`, which is a lexical hashing vectoriser and not a
semantic model.
**Taxonomy:** `bootstrap-2026.08.1`. **Dataset:** 98 cases, cold cache.

### Headline

| Metric | Value |
| --- | --- |
| Accuracy (strict) | 0.663 |
| Accuracy (lenient) | 0.704 |
| Top-2 accuracy | 0.949 |
| Top-3 accuracy | 0.949 |
| Macro F1 | 0.729 |
| Micro F1 | 0.663 |
| Expected calibration error | 0.175 |

### Abstention

| Metric | Value |
| --- | --- |
| Abstention rate | 0.398 |
| Abstention precision | 0.308 |
| Unknown-intent recall | 0.857 |
| Abstention accuracy | 0.704 |

### Slices

| Slice | n | Accuracy |
| --- | --- | --- |
| Ordinary | 72 | 0.722 |
| Expects abstention | 14 | 0.857 |
| Multi-intent | 6 | 0.667 |
| Hard negatives | 12 | 0.083 |

### Cost and latency

| Metric | Value |
| --- | --- |
| Answered by L2 rules | 53 |
| Answered by L3 embedding | 6 |
| Abstained | 39 |
| Latency p50 | 1.01 ms |
| Latency p95 | 2.01 ms |
| Classification cost | $0.00 |

### What these numbers say

**Top-2 accuracy is 0.949 while top-1 is 0.704.** The correct label is almost
always in the classifier's shortlist and often is not the label it picks. The
ranking works considerably better than the confidence gating does — which,
combined with a calibration error of 0.175, says the confidence scale rather
than the ranking is the weak part.

**39 of 98 cases abstained, and only 14 should have.** Abstention precision of
0.308 means roughly two in three abstentions were unnecessary. In the offline
configuration those are dead ends. In a full cascade they are what L4 exists to
catch — but **whether a model layer would classify them correctly is unmeasured
here**, because no model was run.

**Hard negatives score 0.083.** Twelve prompts written specifically to look like
one intent while being another, and eleven were not classified correctly. This
is the honest headline: the deterministic rules are pattern matches, and prompts
built to defeat pattern matches defeat them. This is the slice to watch when
anything changes.

**Multi-intent prompts scored 0.667 strict**, where "correct" means abstaining.
Splitting the evidence between two intents does suppress confidence enough to
reach abstention in most cases. It does not when both candidate intents sit in
the same domain — "Review this diff and fix any bugs" is answered confidently as
`coding.review`, which `tests/unit/test_intent_ambiguity.py` records as a known
limitation rather than hiding.

---

## 3. Intent caches

**Run:** `make bench-intent-cache`, which pins `--semantic-similarity 0.60` for
the reason given below the table.
Cache mode warms on each case's `text`, then scores its `paraphrases`. Because
the benchmark holds ground truth, the false-hit rate below is **measured**, not
estimated from runtime counters.

| Metric | Value |
| --- | --- |
| Scored (paraphrases) | 17 |
| Semantic cache hits | 5 |
| Hit rate | 0.294 |
| **Semantic false-hit rate** | **0.000** |
| Exact cache hits | 0 |

At the shipped default similarity threshold of 0.92, **zero** paraphrases hit.
That is the lexical embedder, not the cache: a reworded prompt does not reach
0.92 cosine under a bag-of-words vectoriser. Sweeping the threshold:

| Similarity threshold | Semantic hits | False-hit rate |
| --- | --- | --- |
| 0.90 | 0 | unmeasured |
| 0.75 | 1 | 0.000 |
| 0.60 | 5 | 0.000 |
| 0.45 | 8 | 0.000 |

**Five to eight hits is far too small a sample to conclude the false-hit rate is
low.** A rate of zero over eight observations is consistent with a true rate
well above ten percent. The correct reading of this table is that the
measurement machinery works and has not yet been given enough data to say
anything. Re-run it with a real embedding model and a dataset of hundreds of
paraphrases before drawing any conclusion about where the threshold belongs.

---

## 4. Not measured

Named so that absence is explicit rather than assumed.

- **L4 / L5 model classifier quality.** Never run. Enabling `--provider` costs
  money and needs credentials, so no figure exists for accuracy, latency or
  cost of the model-backed layers.
- **Real embedding models.** Every embedding number here comes from
  `HashingEmbedder`. A trained model will change L1 and L3 substantially, in a
  direction this repository has not measured.
- **Production traffic of any kind.** No live prompts have passed through this
  code.
- **Routing-quality evals.** Labelled planner match (`route_match`,
  `policy_match`) is measured in §6 against three mock-registry fixtures. Route
  regret, quality regret, escalation rates and the rest of the constitution's
  routing metrics stay unavailable unless both sides already carry declared
  numbers. That is not a claim that the routes are good.
- **Generation evals.** Adapters exist for lm-evaluation-harness, DeepEval and
  an LLM judge. Those packages are not installed here, no judge model was
  called, and no generation scores were produced. RAG metrics and human
  feedback are not implemented.
- **Drift detection.** Not implemented.
- **Classifier latency under load.** The p50/p95 figures above are
  single-threaded, in-process, and measure the classifier only. Gateway
  throughput against the mock provider is in
  [`docs/BENCHMARKS.md`](BENCHMARKS.md); that is a different measurement.

---

## 5. Reproducing

```bash
make install
make bench-intent          # classifier, offline layers, no cost
make bench-intent-cache    # cache hit rate and measured false-hit rate
make eval-run              # CI evaluation suite → artifacts/eval-run.json
make eval-gate             # fail if a critical metric drops vs datasets/eval/baseline.json
```

Intent benches write JSON to `artifacts/`. Useful flags:

```bash
llm-fabric-bench --describe-dataset          # dataset shape before trusting it
llm-fabric-bench --show-failures             # every case that was got wrong
llm-fabric-bench --min-accuracy 0.60         # gate; exit 1 when missed
llm-fabric-bench --provider openai --structured-model gpt-4o-mini   # costs money
```

A gate whose metric could not be measured **fails**. "Not measured" is never
allowed to pass as "met the bar".

---

## 6. Evaluation platform (CI suite)

**Run:** `llm-fabric-eval run --suite datasets/eval/ci-suite.yaml`
**Output written to:** `datasets/eval/baseline.json`
**Configuration:** offline cascade (same as §2), default `config/models.yaml`
registry, identity outputs for deterministic cases (`metadata.output`).
**DeepEval / lm-evaluation-harness / LLM judge:** not scored. The suite names
those tasks with empty metric lists so the adapters are exercised and apply
nothing.

This is a CI tripwire. The deterministic and routing cases are three
self-authored fixtures each. The classification numbers are the same bootstrap
dataset as §2, re-measured through the evaluation runner. They are not a second,
independent evaluation of the classifier.

| Metric | Value | Gate |
| --- | --- | --- |
| exact_match | 1.0 | absolute floor 1.0 |
| json_valid | 1.0 | not gated |
| route_match | 1.0 | absolute floor 1.0 |
| policy_match | 1.0 | not gated |
| accuracy | 0.8776 | absolute floor 0.80; regression max degradation 0.05 |
| macro_f1 | 0.9012 | absolute floor 0.85; regression max degradation 0.05 |
| unknown_intent_recall | 0.8571 | absolute floor 0.75; regression max degradation 0.05 |

Measured 2026-08-24 on `datasets/intent/bootstrap.jsonl` (98 labelled cases)
after the unmatched-child score-dilution fix. These numbers are not 0.97 F1
and are not a comparison against any other router.

**Provenance on that run:** `metric_version` `eval-metrics-v1`; taxonomy
`bootstrap-2026.08.1`; dataset hash `23bea5d76a17f633`; commit
`92ae8cc76f4e7ae6fdaff149ec2ec9e58d50b267` (the tree at measurement time).
Model, model version and prompt version were unset because no judge or
generation adapter ran.

`llm-fabric-eval gate` against this baseline reported no critical failures.
CI runs the same command. A material drop on a critical metric, or an
unmeasured critical metric, fails the job.
