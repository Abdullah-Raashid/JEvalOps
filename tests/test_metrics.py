from __future__ import annotations

from jevalops.evaluation.efficiency import efficiency_adjusted_score
from jevalops.evaluation.exact_metrics import field_accuracy, json_validity, parse_json_object, schema_compliance
from jevalops.training.regression import evaluate_promotion


def test_json_and_field_metrics() -> None:
    prediction = parse_json_object('{"company":"Example株式会社","meeting_time":"14:00"}')
    assert prediction is not None
    assert json_validity('{"company":"Example株式会社"}') == 1.0
    assert field_accuracy(prediction, {"company": "Example株式会社", "meeting_time": "14:00"}) == 1.0
    assert schema_compliance(prediction, {"company": "string", "meeting_time": "string"}) == 1.0


def test_efficiency_adjusted_score_decreases_with_latency() -> None:
    assert efficiency_adjusted_score(0.9, 0.1) > efficiency_adjusted_score(0.9, 2.0)


def test_promotion_gates() -> None:
    baseline = {"quality_score": 0.7, "by_task": {"keigo": 0.7}, "efficiency": {"p95_latency_seconds": 1.0}, "records": []}
    candidate = {"quality_score": 0.77, "by_task": {"keigo": 0.72}, "efficiency": {"p95_latency_seconds": 1.2}, "records": []}
    assert evaluate_promotion(baseline, candidate)["promoted"]
