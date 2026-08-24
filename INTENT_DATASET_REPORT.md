# Intent dataset report

Audited **before** using the bootstrap set as an optimisation target. The
frozen test set remains `datasets/intent/bootstrap.jsonl`. New slices live
beside it and are **not** mixed into that file.

## Frozen test: `datasets/intent/bootstrap.jsonl`

| Fact | Value |
| --- | --- |
| Examples | 98 |
| Labels | 19 |
| Languages | English only |
| Hard negatives | 12 |
| Unknown / OOD labels | 14 |
| Multi-intent flags | 6 |
| Cases with paraphrases | 17 |
| Exact duplicates | 0 |
| Near-duplicates (lexical cosine ≥ 0.92) | 0 |
| Short prompts (< 24 chars) | 6 |
| Long prompts (> 1200 chars) | 0 |
| Conversation context | 0 |
| Overlap with taxonomy examples | 5 |
| Class balance (max/min support) | 7.0× (`unknown` 14 vs `coding.review` 2) |

This set is **self-authored alongside the rules**. It is a regression
tripwire, not production evidence. Five prompts overlap taxonomy example
strings, so embedding-centroid evaluation is slightly optimistic on those
items. We did not delete them: changing the frozen set to look cleaner would
hide the overlap rather than fix it.

`unknown` is the most frequent label. That is intentional: a production
classifier must be measured on abstention, not only on in-taxonomy traffic.

## Other slices (not the frozen baseline)

| File | Role |
| --- | --- |
| `datasets/intent/val.jsonl` | Held-out validation for calibration fits. Not scored as the v1 baseline. |
| `datasets/intent/ood.jsonl` | Explicit out-of-distribution / context-dependent / nonsense. |
| `datasets/intent/adversarial.jsonl` | Prompt-injection and jailbreak-shaped inputs. |
| `datasets/intent/multilingual.jsonl` | Small non-English slice. Not a multilingual claim. |
| `datasets/eval/intentos/extended.jsonl` | Earlier extra tags; still loaded by unit tests. |
| `datasets/intent/taxonomy/bootstrap-2026.08.1.json` | Published immutable snapshot of the bootstrap taxonomy. |

No training split is scored. Taxonomy `examples` are the only texts the
embedding centroids see. `llm-fabric-bench` default remains the frozen
bootstrap file.

## What this means for scores

- Macro-F1 on 19 labels with n=2–14 is noisy. Per-class tables matter more
  than the average.
- English-only test coverage cannot support a multilingual quality claim.
- Over-abstention on known intents and hard-negative confusion are the
  failure modes the baseline already showed. The v1 work targets those,
  without editing this file's labels.
