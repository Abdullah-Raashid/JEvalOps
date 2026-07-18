from __future__ import annotations

import pytest
from pydantic import ValidationError

from jevalops.data.schema import EvaluationExample, japanese_character_ratio
from jevalops.generation.templates import business_rewriting, information_extraction


def test_valid_business_record() -> None:
    example = EvaluationExample.model_validate(business_rewriting(1, synthetic=False))
    assert example.task == "business_rewriting"
    assert example.review_status == "approved"


def test_information_extraction_requires_schema() -> None:
    record = information_extraction(1, synthetic=False)
    record.pop("expected_schema")
    with pytest.raises(ValidationError):
        EvaluationExample.model_validate(record)


def test_japanese_ratio_detects_japanese_text() -> None:
    assert japanese_character_ratio("資料をご確認ください") > 0.8
    assert japanese_character_ratio("please confirm the document") == 0.0
