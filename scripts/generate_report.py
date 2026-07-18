from __future__ import annotations

import argparse
import json
from pathlib import Path

from jevalops.analysis.reporting import write_markdown_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a Markdown model evaluation report.")
    parser.add_argument("--evaluation", type=Path, default=Path("reports/baseline_results.json"))
    parser.add_argument("--errors", type=Path, default=Path("reports/error_analysis.json"))
    parser.add_argument("--output", type=Path, default=Path("reports/baseline_report.md"))
    args = parser.parse_args()
    evaluation = json.loads(args.evaluation.read_text(encoding="utf-8"))
    errors = json.loads(args.errors.read_text(encoding="utf-8"))
    write_markdown_report(args.output, evaluation, errors)
    print(json.dumps({"output": str(args.output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
