from __future__ import annotations

import json
import re
import time

from jevalops.data.normalize import normalize_text
from jevalops.inference.base import GenerationResult, count_tokens_roughly


class RuleBasedBackend:
    """Deterministic local backend for CI, demos, and pipeline sanity checks."""

    model_name = "rule-based-japanese-enterprise-baseline"
    backend_name = "local_rule_based"

    def generate(self, prompt: str, max_tokens: int = 256, temperature: float = 0.0) -> GenerationResult:
        start = time.perf_counter()
        text = self._answer(prompt)
        latency = max(time.perf_counter() - start, 0.001)
        output_tokens = count_tokens_roughly(text)
        return GenerationResult(
            text=text,
            latency_seconds=latency,
            time_to_first_token=latency,
            output_tokens=output_tokens,
            tokens_per_second=output_tokens / latency,
            peak_memory_mb=None,
            model_name=self.model_name,
            backend_name=self.backend_name,
            metadata={"temperature": temperature, "max_tokens": max_tokens},
        )

    def _answer(self, prompt: str) -> str:
        normalized = normalize_text(prompt)
        if "JSON" in normalized and "会議情報" in normalized:
            return json.dumps(
                {
                    "company": _match(normalized, r"(Example株式会社|東都システムズ|青葉物流|未来バイオ)", "不明"),
                    "meeting_date": _match(normalized, r"(2026-[0-9]{2}-[0-9]{2})", "不明"),
                    "meeting_time": _match(normalized, r"([0-9]{2}:[0-9]{2})", "不明"),
                    "meeting_type": "オンライン" if "オンライン" in normalized else "不明",
                    "required_action": "参加可否を返信" if "参加可否" in normalized else "不明",
                },
                ensure_ascii=False,
            )
        if "要約" in normalized and "アクション" in normalized:
            owner = _match(normalized, r"(佐藤|田中|鈴木|高橋)さん", "担当者")
            deadline = _match(normalized, r"(2026-[0-9]{2}-[0-9]{2})", "未定")
            return json.dumps(
                {
                    "summary": "法人向けFAQの回答品質を揃えるため、評価基準を整理する。",
                    "decisions": ["部署共通のレビュー表を作成する"],
                    "action_items": [{"owner": owner, "task": "評価基準を整理してレビュー表を共有", "deadline": deadline}],
                },
                ensure_ascii=False,
            )
        if "資料見たら" in normalized:
            return "お手数をおかけいたしますが、資料をご確認のうえ、ご返信いただけますと幸いです。"
        if "打ち合わせ" in normalized and "無理" in normalized:
            return "明日の打ち合わせにつきまして、ご都合が難しい場合はお知らせいただけますでしょうか。"
        if "請求書まだ" in normalized:
            return "請求書のご送付状況につきまして、念のため確認させていただけますでしょうか。"
        if "納期遅れ" in normalized:
            return "納期に遅れが生じる見込みとなり、誠に申し訳ございません。状況と今後の対応について速やかにご共有いたします。"
        if "交通費" in normalized and "30日以内" in normalized:
            return "利用日から30日以内に申請する必要があります。"
        if "消耗品" in normalized:
            return "いいえ、消耗品は保証対象外です。"
        if "個人情報" in normalized and "外部" in normalized:
            return "いいえ、入力してはなりません。"
        if "海外から勤務" in normalized:
            return "本文には海外からの勤務可否は記載されていません。"
        if "3,000,000円" in normalized or "3000000円" in normalized:
            return json.dumps({"date": "2026年7月18日", "amount_jpy": "3000000"}, ensure_ascii=False)
        if "ＡＰＩ" in prompt or "API" in normalized:
            return "API レスポンスが 500 になりました"
        if "ミーティングわ" in normalized:
            return "ミーティングは明日で、担当は佐藤です。"
        if "三百万円" in normalized:
            return "差分はありません。どちらも300万円を表しています。"
        return "本文の情報だけでは判断できません。"


def _match(text: str, pattern: str, default: str) -> str:
    match = re.search(pattern, text)
    return match.group(1) if match else default
