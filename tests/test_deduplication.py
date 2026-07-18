from __future__ import annotations

from jevalops.data.deduplicate import fingerprint, is_near_duplicate
from jevalops.data.normalize import normalize_text
from jevalops.data.pii import contains_pii, redact_pii


def test_normalize_full_width_and_era_year() -> None:
    assert normalize_text("令和8年　ＡＰＩ") == "2026年 API"


def test_duplicate_detection() -> None:
    assert fingerprint("資料をご確認ください") == fingerprint("資料をご確認ください")
    assert is_near_duplicate("資料をご確認ください", "資料をご確認ください。")


def test_pii_redaction() -> None:
    text = "連絡先は taro@example.com です"
    assert contains_pii(text)
    assert "[REDACTED_EMAIL]" in redact_pii(text)
