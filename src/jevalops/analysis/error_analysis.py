from __future__ import annotations

from collections import Counter
from typing import Any

from jevalops.evaluation.exact_metrics import parse_json_object


def classify_error(record: dict[str, Any]) -> dict[str, Any] | None:
    metrics = record.get("metrics", {})
    if metrics.get("quality_score", 1.0) >= 0.85:
        return None
    prediction = str(record.get("prediction", ""))
    task = record.get("task")
    if task in {"information_extraction", "summarization"} and parse_json_object(prediction) is None:
        error_type = "invalid_json"
        severity = "high"
        expected = "Return valid JSON matching the expected schema."
        cause = "Structured-output constraint not followed."
    elif task == "grounded_qa" and any(term in str(record.get("reference_answer", "")) for term in ["記載されていません", "不明"]):
        error_type = "failure_to_abstain"
        severity = "high"
        expected = "State that the supplied document does not specify the answer."
        cause = "Weak grounding and overconfident generation."
    elif task == "business_rewriting" and not any(term in prediction for term in ["いただ", "ござい", "幸い", "でしょうか"]):
        error_type = "inappropriate_politeness_level"
        severity = "medium"
        expected = "Use appropriate business Japanese and keigo."
        cause = "Insufficient honorific control."
    elif task == "robustness" and ("2026" in str(record.get("reference_answer")) and "2026" not in prediction):
        error_type = "date_normalization_error"
        severity = "medium"
        expected = "Normalize Japanese era or mixed-script date expressions."
        cause = "Normalization edge case missed."
    else:
        error_type = "instruction_omission"
        severity = "medium"
        expected = "Satisfy every instruction without adding unsupported information."
        cause = "General task-following failure."
    return {
        "example_id": record.get("example_id"),
        "model": record.get("model"),
        "error_type": error_type,
        "severity": severity,
        "expected_behavior": expected,
        "observed_behavior": prediction,
        "possible_cause": cause,
    }


def analyze_errors(evaluation: dict[str, Any]) -> dict[str, Any]:
    errors = [error for record in evaluation.get("records", []) if (error := classify_error(record))]
    counts = Counter(error["error_type"] for error in errors)
    severities = Counter(error["severity"] for error in errors)
    return {"error_count": len(errors), "by_error_type": dict(counts), "by_severity": dict(severities), "errors": errors}
