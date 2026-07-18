from __future__ import annotations

import argparse
import json
from pathlib import Path

from jevalops.training.regression import PromotionGateConfig, evaluate_promotion


def main() -> None:
    parser = argparse.ArgumentParser(description="Run model promotion regression gates.")
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    args = parser.parse_args()
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    result = evaluate_promotion(baseline, candidate, PromotionGateConfig())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["promoted"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
