from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

TaskName = Literal[
    "business_rewriting",
    "information_extraction",
    "summarization",
    "grounded_qa",
    "robustness",
]
Difficulty = Literal["easy", "medium", "hard"]
SourceType = Literal["human_authored", "synthetic", "public_dataset"]
ReviewStatus = Literal["unreviewed", "reviewed", "approved", "rejected"]


JAPANESE_RE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff]")


def japanese_character_ratio(text: str) -> float:
    visible = [char for char in text if not char.isspace()]
    if not visible:
        return 0.0
    return sum(1 for char in visible if JAPANESE_RE.search(char)) / len(visible)


class EvaluationExample(BaseModel):
    """Governed JSONL record used across dataset, evaluation, and training."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[a-z]+_[0-9]{4}$")
    task: TaskName
    domain: str = Field(min_length=1)
    difficulty: Difficulty = "medium"
    input: str = Field(min_length=1)
    instruction: str = Field(min_length=1)
    reference_answer: str | dict[str, Any] | list[Any]
    context: str | None = None
    expected_schema: dict[str, str] | None = None
    source_type: SourceType
    source_name: str = Field(min_length=1)
    license: str = Field(min_length=1)
    contains_pii: bool
    is_synthetic: bool = False
    creation_method: str = Field(min_length=1)
    review_status: ReviewStatus
    duplicate_group_id: str | None = None
    dataset_version: str = Field(pattern=r"^v[0-9]+\.[0-9]+\.[0-9]+$")
    tags: list[str] = Field(default_factory=list, min_length=1)

    @field_validator("domain", "source_name", "license", "creation_method")
    @classmethod
    def reject_blank_strings(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("field cannot be blank")
        return stripped

    @model_validator(mode="after")
    def check_governance_consistency(self) -> "EvaluationExample":
        if self.source_type == "synthetic" and not self.is_synthetic:
            raise ValueError("synthetic source_type must set is_synthetic=true")
        if self.source_type != "synthetic" and self.is_synthetic:
            raise ValueError("is_synthetic=true requires source_type=synthetic")
        if self.task == "information_extraction":
            if not isinstance(self.reference_answer, dict):
                raise ValueError("information_extraction requires object reference_answer")
            if not self.expected_schema:
                raise ValueError("information_extraction requires expected_schema")
        if self.task == "grounded_qa" and not self.context:
            raise ValueError("grounded_qa requires context")
        if isinstance(self.reference_answer, str) and self.input.strip() == self.reference_answer.strip():
            raise ValueError("input and reference_answer must not be identical")
        return self


class ValidationIssue(BaseModel):
    example_id: str | None
    severity: Literal["error", "warning"]
    code: str
    message: str


class DatasetValidationReport(BaseModel):
    total_records: int
    valid_records: int
    issues: list[ValidationIssue] = Field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        return any(issue.severity == "error" for issue in self.issues)
