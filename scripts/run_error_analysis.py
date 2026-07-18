from __future__ import annotations

import argparse
import json
from pathlib import Path

from jevalops.analysis.error_analysis import analyze_errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify Japanese enterprise LLM failures.")
    parser.add_argument("--evaluation", type=Path, default=Path("reports/baseline_results.json"))
    parser.add_argument("--output", type=Path, default=Path("reports/error_analysis.json"))
    args = parser.parse_args()
    evaluation = json.loads(args.evaluation.read_text(encoding="utf-8"))
    analysis = analyze_errors(evaluation)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "error_count": analysis["error_count"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
