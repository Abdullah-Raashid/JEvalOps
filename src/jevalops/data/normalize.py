from __future__ import annotations

import re
import unicodedata
from typing import Any

ERA_START_YEARS = {"令和": 2018, "平成": 1988, "昭和": 1925}
ERA_DATE_RE = re.compile(r"(令和|平成|昭和)(元|[0-9０-９]+)年")
WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    normalized = WHITESPACE_RE.sub(" ", normalized).strip()
    return ERA_DATE_RE.sub(_replace_era_year, normalized)


def _replace_era_year(match: re.Match[str]) -> str:
    era = match.group(1)
    raw_year = match.group(2)
    era_year = 1 if raw_year == "元" else int(unicodedata.normalize("NFKC", raw_year))
    return f"{ERA_START_YEARS[era] + era_year}年"


def normalize_reference(value: str | dict[str, Any] | list[Any]) -> str | dict[str, Any] | list[Any]:
    if isinstance(value, str):
        return normalize_text(value)
    if isinstance(value, dict):
        return {key: normalize_reference(val) for key, val in value.items()}
    if isinstance(value, list):
        return [normalize_reference(item) for item in value]
    return value


def normalize_record(record: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(record)
    for key in ("input", "instruction", "context"):
        if isinstance(normalized.get(key), str):
            normalized[key] = normalize_text(normalized[key])
    if "reference_answer" in normalized:
        normalized["reference_answer"] = normalize_reference(normalized["reference_answer"])
    return normalized
