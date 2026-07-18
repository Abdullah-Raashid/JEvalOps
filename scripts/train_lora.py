from __future__ import annotations

import argparse
from pathlib import Path

from jevalops.training.lora import train_lora


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a LoRA adapter on targeted Japanese error-driven data.")
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--train-jsonl", type=Path, default=Path("data/processed/sft_train.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/lora"))
    args = parser.parse_args()
    train_lora(args.model_name, args.train_jsonl, args.output_dir)


if __name__ == "__main__":
    main()
