from __future__ import annotations

from pathlib import Path

from jevalops.data.validate import load_jsonl, validate_records, write_jsonl


def ingest_jsonl(raw_path: Path, output_path: Path) -> int:
    records = load_jsonl(raw_path)
    valid, report = validate_records(records)
    if report.has_errors:
        messages = "\n".join(f"{issue.example_id}: {issue.code} - {issue.message}" for issue in report.issues)
        raise ValueError(f"Cannot ingest invalid dataset:\n{messages}")
    write_jsonl(output_path, valid)
    return len(valid)
