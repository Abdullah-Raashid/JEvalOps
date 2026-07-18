from __future__ import annotations

import json
from typing import Any

from jevalops.data.schema import EvaluationExample
from jevalops.evaluation.efficiency import aggregate_efficiency, efficiency_adjusted_score
from jevalops.evaluation.exact_metrics import exact_match, field_accuracy, json_validity, parse_json_object, schema_compliance
from jevalops.evaluation.rubric_metrics import score_japanese_generation
from jevalops.inference.base import ModelBackend


def build_prompt(example: EvaluationExample) -> str:
    parts = []
    if example.context:
        parts.append(f"文書:\n{example.context}")
    parts.append(f"指示:\n{example.instruction}")
    parts.append(f"入力:\n{example.input}")
    parts.append("回答:")
    return "\n\n".join(parts)


def evaluate_example(example: EvaluationExample, backend: ModelBackend) -> dict[str, Any]:
    generation = backend.generate(build_prompt(example))
    prediction_text = generation.text
    metrics = _score_prediction(example, prediction_text)
    quality_score = metrics.get("quality_score", 0.0)
    return {
        "example_id": example.id,
        "task": example.task,
        "model": generation.model_name,
        "backend": generation.backend_name,
        "prediction": prediction_text,
        "reference_answer": example.reference_answer,
        "metrics": metrics,
        "efficiency": generation.to_dict(),
        "efficiency_adjusted_score": efficiency_adjusted_score(quality_score, generation.latency_seconds),
    }


def evaluate_dataset(examples: list[EvaluationExample], backend: ModelBackend) -> dict[str, Any]:
    records = [evaluate_example(example, backend) for example in examples]
    task_scores: dict[str, list[float]] = {}
    for record in records:
        task_scores.setdefault(record["task"], []).append(record["metrics"]["quality_score"])
    by_task = {task: sum(scores) / len(scores) for task, scores in task_scores.items()}
    quality = sum(by_task.values()) / len(by_task) if by_task else 0.0
    return {
        "model": getattr(backend, "model_name", "unknown"),
        "backend": getattr(backend, "backend_name", "unknown"),
        "quality_score": quality,
        "by_task": by_task,
        "efficiency": aggregate_efficiency([record["efficiency"] for record in records]),
        "records": records,
    }


def _score_prediction(example: EvaluationExample, prediction_text: str) -> dict[str, Any]:
    if isinstance(example.reference_answer, dict):
        parsed = parse_json_object(prediction_text)
        if parsed is None:
            return {"quality_score": 0.0, "json_validity": 0.0, "schema_compliance": 0.0, "field_accuracy": 0.0}
        field_score = field_accuracy(parsed, example.reference_answer)
        schema_score = schema_compliance(parsed, example.expected_schema)
        return {
            "quality_score": (field_score + schema_score) / 2,
            "json_validity": json_validity(prediction_text),
            "schema_compliance": schema_score,
            "field_accuracy": field_score,
        }
    if example.task in {"business_rewriting", "grounded_qa", "robustness"} and isinstance(example.reference_answer, str):
        rubric = score_japanese_generation(prediction_text, example.reference_answer, example.instruction)
        return {"quality_score": rubric.mean / 5, **rubric.to_dict(), "exact_match": exact_match(prediction_text, example.reference_answer)}
    return {"quality_score": exact_match(prediction_text, example.reference_answer), "exact_match": exact_match(prediction_text, example.reference_answer)}


def save_evaluation(path, evaluation: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(evaluation, handle, ensure_ascii=False, indent=2)
