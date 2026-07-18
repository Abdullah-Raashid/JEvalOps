from __future__ import annotations

from jevalops.data.deduplicate import similarity
from jevalops.data.schema import japanese_character_ratio


def passes_quality_filters(record: dict, existing_records: list[dict] | None = None) -> tuple[bool, list[str]]:
    issues: list[str] = []
    if japanese_character_ratio(f"{record.get('input', '')} {record.get('reference_answer', '')}") < 0.2:
        issues.append("low_japanese_ratio")
    if str(record.get("reference_answer", "")).strip() in str(record.get("instruction", "")).strip():
        issues.append("reference_leaks_into_instruction")
    for existing in existing_records or []:
        if similarity(record.get("input", ""), existing.get("input", "")) > 0.995:
            issues.append(f"near_duplicate:{existing.get('id')}")
            break
    return not issues, issues
