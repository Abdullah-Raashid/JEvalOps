from __future__ import annotations

from jevalops.data.schema import EvaluationExample
from jevalops.evaluation.evaluator import build_prompt, evaluate_dataset
from jevalops.generation.templates import business_rewriting, information_extraction
from jevalops.inference.rule_based_backend import RuleBasedBackend


def test_rule_based_backend_generates_keigo() -> None:
    backend = RuleBasedBackend()
    result = backend.generate("指示:\n取引先への丁寧な依頼文に書き換えてください。\n入力:\n資料見たら返事ください\n回答:")
    assert "ご確認" in result.text
    assert result.latency_seconds > 0


def test_evaluate_dataset_smoke() -> None:
    examples = [
        EvaluationExample.model_validate(business_rewriting(1, synthetic=False)),
        EvaluationExample.model_validate(information_extraction(1, synthetic=False)),
    ]
    evaluation = evaluate_dataset(examples, RuleBasedBackend())
    assert evaluation["quality_score"] > 0.5
    assert set(evaluation["by_task"]) == {"business_rewriting", "information_extraction"}


def test_prompt_contains_context_when_present() -> None:
    record = business_rewriting(1, synthetic=False)
    example = EvaluationExample.model_validate(record)
    assert "指示" in build_prompt(example)
