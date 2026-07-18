from __future__ import annotations

import hashlib
from difflib import SequenceMatcher

from jevalops.data.normalize import normalize_text


def fingerprint(text: str) -> str:
    normalized = normalize_text(text).lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def similarity(left: str, right: str) -> float:
    return SequenceMatcher(None, normalize_text(left), normalize_text(right)).ratio()


def is_near_duplicate(left: str, right: str, threshold: float = 0.92) -> bool:
    return similarity(left, right) >= threshold


def find_duplicate_groups(records: list[dict], threshold: float = 0.92) -> list[tuple[str, str, float]]:
    duplicates: list[tuple[str, str, float]] = []
    for left_index, left in enumerate(records):
        for right in records[left_index + 1 :]:
            score = similarity(left.get("input", ""), right.get("input", ""))
            if score >= threshold:
                duplicates.append((left.get("id", ""), right.get("id", ""), score))
    return duplicates
