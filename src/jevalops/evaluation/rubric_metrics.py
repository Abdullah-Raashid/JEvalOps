from __future__ import annotations

from dataclasses import dataclass, asdict

from jevalops.evaluation.exact_metrics import exact_match, normalize_for_match


@dataclass
class RubricScore:
    correctness: int
    naturalness: int
    politeness: int
    faithfulness: int
    completeness: int
    format_compliance: int

    @property
    def mean(self) -> float:
        return sum(asdict(self).values()) / 6

    def to_dict(self) -> dict[str, float | int]:
        payload = asdict(self)
        payload["mean"] = self.mean
        return payload


def score_japanese_generation(prediction: str, reference: str, instruction: str) -> RubricScore:
    pred_norm = normalize_for_match(prediction)
    ref_norm = normalize_for_match(reference)
    overlap = _character_overlap(pred_norm, ref_norm)
    correctness = 5 if exact_match(prediction, reference) else _bucket(overlap)
    politeness = 5 if any(term in prediction for term in ["いただ", "ござい", "幸い", "でしょうか", "申し訳"]) else 3
    naturalness = 2 if any(term in prediction for term in ["ご確認して", "させていただきますでしょうか"]) else min(5, max(2, correctness))
    faithfulness = 5 if len(prediction) <= max(len(reference) * 2, 20) else 3
    completeness = 5 if len(prediction.strip()) > 0 and overlap > 0.45 else 3
    format_compliance = 5 if ("JSON" not in instruction and "json" not in instruction.lower()) else 2
    return RubricScore(correctness, naturalness, politeness, faithfulness, completeness, format_compliance)


def _character_overlap(left: str, right: str) -> float:
    if not right:
        return 0.0
    return len(set(left) & set(right)) / len(set(right))


def _bucket(score: float) -> int:
    if score >= 0.8:
        return 4
    if score >= 0.6:
        return 3
    if score >= 0.35:
        return 2
    return 1
