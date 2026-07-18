from __future__ import annotations

from jevalops.data.validate import validate_records
from jevalops.generation.filters import passes_quality_filters
from jevalops.generation.templates import generate_template_records


def generate_dataset(train_size: int = 300, validation_size: int = 100, test_size: int = 250) -> dict[str, list[dict]]:
    splits = {
        "train": generate_template_records(train_size, synthetic=True, start_index=0),
        "validation": generate_template_records(validation_size, synthetic=False, start_index=3000),
        "test": generate_template_records(test_size, synthetic=False, start_index=6000),
    }
    accepted: dict[str, list[dict]] = {}
    for split, records in splits.items():
        split_records: list[dict] = []
        for record in records:
            ok, _ = passes_quality_filters(record, split_records)
            if ok:
                split_records.append(record)
        valid, report = validate_records(split_records)
        errors = [issue for issue in report.issues if issue.severity == "error"]
        if errors:
            messages = "; ".join(f"{issue.example_id}:{issue.code}" for issue in errors)
            raise ValueError(f"Generated invalid {split} split: {messages}")
        accepted[split] = [item.model_dump(mode="json") for item in valid]
    return accepted
