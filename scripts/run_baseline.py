from __future__ import annotations

import argparse
import json
from pathlib import Path

from jevalops.data.validate import load_jsonl, validate_records
from jevalops.evaluation.evaluator import evaluate_dataset, save_evaluation
from jevalops.inference.rule_based_backend import RuleBasedBackend


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an offline baseline evaluation.")
    parser.add_argument("--dataset", type=Path, default=Path("data/test.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("reports/baseline_results.json"))
    args = parser.parse_args()
    examples, report = validate_records(load_jsonl(args.dataset))
    if report.has_errors:
        raise SystemExit(json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2))
    evaluation = evaluate_dataset(examples, RuleBasedBackend())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_evaluation(args.output, evaluation)
    print(json.dumps({"output": str(args.output), "quality_score": evaluation["quality_score"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
