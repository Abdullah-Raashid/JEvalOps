from __future__ import annotations

import re

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
PHONE_RE = re.compile(r"\b(?:0\d{1,4}[-ー]\d{1,4}[-ー]\d{3,4}|0\d{9,10})\b")
MY_NUMBER_RE = re.compile(r"\b\d{4}[- ]?\d{4}[- ]?\d{4}\b")
POSTAL_RE = re.compile(r"\b\d{3}-\d{4}\b")

PATTERNS = {
    "email": EMAIL_RE,
    "phone": PHONE_RE,
    "my_number": MY_NUMBER_RE,
    "postal_code": POSTAL_RE,
}


def detect_pii(text: str) -> list[str]:
    return [label for label, pattern in PATTERNS.items() if pattern.search(text)]


def contains_pii(text: str) -> bool:
    return bool(detect_pii(text))


def redact_pii(text: str) -> str:
    redacted = text
    for label, pattern in PATTERNS.items():
        redacted = pattern.sub(f"[REDACTED_{label.upper()}]", redacted)
    return redacted
