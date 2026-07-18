from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from jevalops.data.deduplicate import find_duplicate_groups, is_near_duplicate
from jevalops.data.normalize import normalize_record
from jevalops.data.pii import contains_pii
from jevalops.data.schema import DatasetValidationReport, EvaluationExample, ValidationIssue, japanese_character_ratio


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                records.append({"id": f"line_{line_number:04d}", "__json_error__": str(exc)})
    return records


def write_jsonl(path: Path, records: list[EvaluationExample | dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            payload = record.model_dump(mode="json") if isinstance(record, EvaluationExample) else record
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def validate_records(records: list[dict[str, Any]], *, min_japanese_ratio: float = 0.2) -> tuple[list[EvaluationExample], DatasetValidationReport]:
    valid: list[EvaluationExample] = []
    issues: list[ValidationIssue] = []

    for raw_record in records:
        if "__json_error__" in raw_record:
            issues.append(
                ValidationIssue(
                    example_id=raw_record.get("id"),
                    severity="error",
                    code="invalid_json",
                    message=raw_record["__json_error__"],
                )
            )
            continue

        record = normalize_record(raw_record)
        example_id = record.get("id")
        text_for_pii = " ".join(
            str(record.get(key, "")) for key in ("input", "instruction", "reference_answer", "context")
        )
        if contains_pii(text_for_pii) or record.get("contains_pii"):
            issues.append(
                ValidationIssue(
                    example_id=example_id,
                    severity="error",
                    code="pii_detected",
                    message="Record contains or declares personal information.",
                )
            )
        if japanese_character_ratio(f"{record.get('input', '')} {record.get('instruction', '')}") < min_japanese_ratio:
            issues.append(
                ValidationIssue(
                    example_id=example_id,
                    severity="error",
                    code="not_predominantly_japanese",
                    message="Input and instruction do not contain enough Japanese text.",
                )
            )
        try:
            valid.append(EvaluationExample.model_validate(record))
        except ValidationError as exc:
            issues.append(
                ValidationIssue(
                    example_id=example_id,
                    severity="error",
                    code="schema_validation_failed",
                    message=str(exc.errors(include_url=False)),
                )
            )

    for left_id, right_id, score in find_duplicate_groups([item.model_dump() for item in valid]):
        issues.append(
            ValidationIssue(
                example_id=left_id,
                severity="warning",
                code="near_duplicate_input",
                message=f"Near duplicate of {right_id} with similarity={score:.3f}.",
            )
        )

    report = DatasetValidationReport(
        total_records=len(records),
        valid_records=len(valid),
        issues=issues,
    )
    return valid, report


def assert_no_split_leakage(train: list[EvaluationExample], test: list[EvaluationExample], threshold: float = 0.9) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    for train_item in train:
        for test_item in test:
            if is_near_duplicate(train_item.input, test_item.input, threshold=threshold):
                issues.append(
                    ValidationIssue(
                        example_id=test_item.id,
                        severity="error",
                        code="train_test_leakage",
                        message=f"Test item is too similar to train item {train_item.id}.",
                    )
                )
    return issues


def validate_jsonl(path: Path) -> DatasetValidationReport:
    _, report = validate_records(load_jsonl(path))
    return report
