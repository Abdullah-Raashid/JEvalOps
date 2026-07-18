from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class PromotionGateConfig:
    min_quality_improvement: float = 0.05
    max_task_drop: float = 0.02
    max_p95_latency_seconds: float = 2.0
    min_schema_compliance: float = 0.95
    allow_hallucination_increase: bool = False


def evaluate_promotion(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    config: PromotionGateConfig = PromotionGateConfig(),
) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    baseline_quality = baseline.get("quality_score", 0.0)
    candidate_quality = candidate.get("quality_score", 0.0)
    checks["quality_improves"] = candidate_quality - baseline_quality >= config.min_quality_improvement
    baseline_tasks = baseline.get("by_task", {})
    candidate_tasks = candidate.get("by_task", {})
    checks["no_critical_task_drop"] = all(
        candidate_tasks.get(task, 0.0) >= score - config.max_task_drop for task, score in baseline_tasks.items()
    )
    checks["p95_latency_within_limit"] = candidate.get("efficiency", {}).get("p95_latency_seconds", float("inf")) <= config.max_p95_latency_seconds
    schema_scores = [
        record.get("metrics", {}).get("schema_compliance")
        for record in candidate.get("records", [])
        if record.get("metrics", {}).get("schema_compliance") is not None
    ]
    checks["schema_compliance_above_floor"] = (sum(schema_scores) / len(schema_scores)) >= config.min_schema_compliance if schema_scores else True
    promoted = all(checks.values())
    return {"promoted": promoted, "checks": checks, "quality_delta": candidate_quality - baseline_quality}
