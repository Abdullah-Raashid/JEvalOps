from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Mapping
from typing import Any


def normalize_for_match(value: Any) -> str:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True) if not isinstance(value, str) else value
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", "", text)
    return text.lower()


def exact_match(prediction: Any, reference: Any) -> float:
    return float(normalize_for_match(prediction) == normalize_for_match(reference))


def parse_json_object(text: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def json_validity(text: str) -> float:
    return float(parse_json_object(text) is not None)


def field_accuracy(prediction: Mapping[str, Any], reference: Mapping[str, Any]) -> float:
    if not reference:
        return 0.0
    correct = sum(1 for key, value in reference.items() if normalize_for_match(prediction.get(key)) == normalize_for_match(value))
    return correct / len(reference)


def precision_recall_f1(prediction: Mapping[str, Any], reference: Mapping[str, Any]) -> dict[str, float]:
    predicted_items = {(key, normalize_for_match(value)) for key, value in prediction.items()}
    reference_items = {(key, normalize_for_match(value)) for key, value in reference.items()}
    true_positive = len(predicted_items & reference_items)
    precision = true_positive / len(predicted_items) if predicted_items else 0.0
    recall = true_positive / len(reference_items) if reference_items else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def schema_compliance(prediction: Mapping[str, Any], expected_schema: Mapping[str, str] | None) -> float:
    if not expected_schema:
        return 1.0
    for key, expected_type in expected_schema.items():
        if key not in prediction:
            return 0.0
        value = prediction[key]
        if expected_type == "string" and not isinstance(value, str):
            return 0.0
        if expected_type == "array" and not isinstance(value, list):
            return 0.0
        if expected_type == "object" and not isinstance(value, dict):
            return 0.0
    return 1.0
