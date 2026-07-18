from __future__ import annotations

import argparse
import json
from pathlib import Path

from jevalops.data.schema import EvaluationExample
from jevalops.data.validate import assert_no_split_leakage, validate_records, write_jsonl
from jevalops.generation.synthetic import generate_dataset
from jevalops.training.dataset import write_sft_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Build governed Japanese enterprise benchmark splits.")
    parser.add_argument("--train-size", type=int, default=300)
    parser.add_argument("--validation-size", type=int, default=100)
    parser.add_argument("--test-size", type=int, default=250)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args()

    splits = generate_dataset(args.train_size, args.validation_size, args.test_size)
    validated: dict[str, list[EvaluationExample]] = {}
    for split, records in splits.items():
        examples, report = validate_records(records)
        if report.has_errors:
            raise SystemExit(json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2))
        validated[split] = examples
        write_jsonl(args.data_dir / f"{split}.jsonl", examples)

    leakage_issues = assert_no_split_leakage(validated["train"], validated["test"], threshold=0.995)
    if leakage_issues:
        raise SystemExit(json.dumps([issue.model_dump() for issue in leakage_issues], ensure_ascii=False, indent=2))

    write_sft_jsonl(args.data_dir / "processed" / "sft_train.jsonl", validated["train"])
    print(
        json.dumps(
            {split: len(records) for split, records in validated.items()} | {"sft_path": str(args.data_dir / "processed" / "sft_train.jsonl")},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
