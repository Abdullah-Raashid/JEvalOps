from __future__ import annotations

import json
from pathlib import Path

from jevalops.data.schema import EvaluationExample
from jevalops.evaluation.evaluator import build_prompt


def to_sft_record(example: EvaluationExample) -> dict[str, str]:
    answer = json.dumps(example.reference_answer, ensure_ascii=False) if not isinstance(example.reference_answer, str) else example.reference_answer
    return {"prompt": build_prompt(example), "response": answer}


def write_sft_jsonl(path: Path, examples: list[EvaluationExample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(to_sft_record(example), ensure_ascii=False) + "\n")
