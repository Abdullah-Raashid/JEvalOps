from __future__ import annotations

import argparse
import json
from pathlib import Path

from jevalops.data.validate import write_jsonl
from jevalops.generation.templates import generate_template_records


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate controlled synthetic Japanese training records.")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--output", type=Path, default=Path("data/interim/synthetic.jsonl"))
    args = parser.parse_args()
    records = generate_template_records(args.count, synthetic=True)
    write_jsonl(args.output, records)
    print(json.dumps({"output": str(args.output), "records": len(records)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
