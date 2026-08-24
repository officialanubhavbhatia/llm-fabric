"""LLM-as-judge adapter.

A judge is a model call with a fixed rubric. If no provider is configured the
adapter reports unavailable rather than inventing a score. Parse failures are
`None`, not zero — a garbled judge is not a failing example.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from llm_fabric.contract.openai import ChatMessage
from llm_fabric.eval.schema import EvalResult, EvaluatorKind, ExampleResult
from llm_fabric.serving.base import InferenceRequest, Provider
from llm_fabric.storage.records import EvalExample

SUPPORTED = frozenset({"judge_score"})

_RUBRIC = """You are grading a model output against an expected answer.
Reply with a JSON object only: {"score": <number 0.0-1.0>, "rationale": "<short>"}.
Do not award a score if you cannot judge the output.
Expected:
{expected}
Output:
{output}
"""


@dataclass(frozen=True, slots=True)
class JudgeVerdict:
    score: float | None
    rationale: str
    raw: str


class LLMJudge:
    def __init__(self, provider: Provider, model: str) -> None:
        self._provider = provider
        self._model = model

    @property
    def model(self) -> str:
        return self._model

    async def score(self, *, expected: str, output: str) -> JudgeVerdict:
        request = InferenceRequest(
            model=self._model,
            messages=[
                ChatMessage(
                    role="user",
                    content=_RUBRIC.format(expected=expected, output=output),
                )
            ],
        )
        result = await self._provider.generate(request)
        parsed = _parse(result.text)
        raw_score = parsed.get("score")
        score = raw_score if isinstance(raw_score, int | float) else None
        return JudgeVerdict(
            score=float(score) if score is not None else None,
            rationale=str(parsed.get("rationale") or ""),
            raw=result.text,
        )


async def evaluate(
    *,
    task: str,
    examples: list[EvalExample],
    outputs: dict[str, str],
    metrics: tuple[str, ...],
    judge: LLMJudge | None,
) -> EvalResult:
    if judge is None:
        return EvalResult(
            task=task,
            evaluator=EvaluatorKind.JUDGE,
            metrics=dict.fromkeys(metrics, None),
            unavailable=metrics,
            note="No judge provider is configured. Scores are not synthesized.",
        )
    requested = tuple(name for name in metrics if name in SUPPORTED)
    unknown = tuple(name for name in metrics if name not in SUPPORTED)
    rows: list[ExampleResult] = []
    for index, example in enumerate(examples):
        example_id = str(example.metadata.get("id") or index)
        output = outputs.get(example_id)
        if output is None or example.expected is None:
            rows.append(
                ExampleResult(
                    example_id=example_id,
                    scores={"judge_score": None},
                    error="missing output or expected label",
                )
            )
            continue
        verdict = await judge.score(expected=example.expected, output=output)
        rows.append(
            ExampleResult(
                example_id=example_id,
                scores={"judge_score": verdict.score},
                output=output,
                notes=verdict.rationale or None,
            )
        )
    values: list[float] = []
    for row in rows:
        value = row.scores.get("judge_score")
        if value is not None:
            values.append(value)
    metrics_out: dict[str, float | None] = {
        "judge_score": sum(values) / len(values) if values else None
    }
    return EvalResult(
        task=task,
        evaluator=EvaluatorKind.JUDGE,
        metrics={name: metrics_out.get(name) for name in requested},
        examples=tuple(rows),
        unavailable=unknown,
        note=f"Judged by {judge.model}. Ungraded examples do not contribute to the mean.",
    )


def _parse(text: str) -> dict[str, Any]:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match is None:
        return {}
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
    if not isinstance(payload, dict):
        return {}
    score = payload.get("score")
    if isinstance(score, int | float) and 0.0 <= float(score) <= 1.0:
        return {"score": float(score), "rationale": payload.get("rationale")}
    return {"rationale": payload.get("rationale")}
