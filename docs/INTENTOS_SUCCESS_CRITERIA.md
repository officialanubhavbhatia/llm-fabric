# IntentOS v1 — frozen success criteria

Defined **before** classifier behaviour was changed for this phase. The
measurement that these numbers came from is
`datasets/eval/intentos/baseline-2026.08.24.json`, produced by:

```text
uv run llm-fabric-bench --show-failures --format json
```

against the frozen test set `datasets/intent/bootstrap.jsonl` (98 cases,
taxonomy `bootstrap-2026.08.1`, offline cascade L0–L3, no L4/L5).

`docs/EVALUATIONS.md` previously published older figures (strict accuracy
0.663, macro-F1 0.729). Those are **not** rewritten. They remain a historical
record of an earlier cascade. This file is the IntentOS v1 baseline.

## Frozen test set

Do not edit labels, delete hard cases, or merge intents in
`datasets/intent/bootstrap.jsonl` in order to improve a score.

New examples belong in other files (`ood`, `adversarial`, `val`,
`multilingual`). Training-style examples live on taxonomy nodes. Evaluation
never scores taxonomy example strings as if they were a held-out test.

## Baseline (offline L0–L3)

| Metric | Value |
| --- | --- |
| Strict accuracy | 0.8776 |
| Lenient accuracy | 0.9184 |
| Top-2 accuracy | 0.9796 |
| Macro F1 | 0.9012 |
| Micro F1 | 0.8776 |
| ECE | 0.1785 |
| Abstention rate | 0.2041 |
| Abstention precision | 0.6000 |
| Unknown-intent recall | 0.8571 |
| Hard-negative accuracy | 0.5000 |
| Multi-intent accuracy | 0.6667 |
| Ordinary-slice accuracy | 0.9444 |
| L2 answers / L3 answers / abstain | 74 / 4 / 20 |
| Latency p50 / p95 / p99 (ms) | 1.48 / 2.38 / 16.13 |
| Classification cost | 0.00 USD (offline) |

Calibration bins already show the product-relevant pattern: confidence
0.9–1.0 is right about 0.96 of the time (n=26); 0.6–0.7 is right about 0.43
of the time (n=7). Ranking is strong (top-2 0.98); the remaining failures are
mostly **over-abstention** on known intents, plus hard negatives.

High-confidence routing precision at explicit thresholds is added as a
benchmark metric in this phase and recorded in the same baseline file once
the metric exists, still before classifier changes.

## Why the remaining errors happen

1. L2 confidence is an uncalibrated heuristic. Several known intents land
   just under the 0.70 stop threshold and become abstentions.
2. Default L3 uses `HashingEmbedder` (lexical hashing, not meaning). It
   rarely beats L2's threshold, so the cascade does not recover those misses.
3. Hard negatives share vocabulary with a sibling intent; rules that fire on
   that vocabulary then under-shoot because evidence is split.
4. Serving-path classification is off by default, so live routing often never
   sees an intent at all.
5. The labelled set is self-authored and overlaps the rule author. Scores are
   a regression tripwire, not production evidence.

## Material improvement (defined now)

A candidate cascade is a **material improvement** if, on the **frozen**
bootstrap.jsonl test set, **all** of the following hold:

1. Strict accuracy ≥ 0.8776 (no regression) and ideally ≥ 0.90.
2. Unknown-intent recall ≥ 0.80 (baseline 0.857; CI absolute floor remains 0.75).
3. Abstention precision ≥ 0.70 (baseline 0.60) — fewer incorrect abstentions.
4. Hard-negative accuracy ≥ 0.58 (at least 7/12, baseline 6/12).
5. ECE ≤ 0.20 (no material worsening; baseline 0.1785).
6. High-confidence precision at threshold 0.90 ≥ 0.95, with coverage reported
   and not driven to zero by abstaining everything.
7. Semantic-cache false-hit rate, when measured at cosine 0.60, stays ≤ 0.05.
8. Wins are **not** obtained by deleting hard cases, merging intents, or
   rewriting this baseline.

Unacceptable regressions: unknown recall dropping more than 0.05 from 0.857,
or high-confidence (≥0.90) precision falling below 0.90.

These bars are about **routing risk**: a wrong high-confidence label is worse
than ABSTAIN; a taxonomy that cannot say "unknown" is worse than a lower
macro-F1.

## Out of scope

The 30-grade route planner is not this phase. IntentOS may emit abstract
grade *hints*. It must not bind to model names.
