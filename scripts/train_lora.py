from __future__ import annotations

import argparse
import json
from pathlib import Path

from jevalops.training.lora import LoRAConfig, TrainingConfig, train_lora


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a LoRA adapter on targeted Japanese error-driven data.")
    parser.add_argument("--model-name", default="rinna/japanese-gpt2-small")
    parser.add_argument("--train-jsonl", type=Path, default=Path("data/processed/sft_train.jsonl"))
    parser.add_argument("--validation-jsonl", type=Path, default=Path("data/validation.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/lora"))
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--target-modules", nargs="*", default=None)
    parser.add_argument("--slow-tokenizer", action="store_true")
    args = parser.parse_args()
    metrics = train_lora(
        args.model_name,
        args.train_jsonl,
        args.output_dir,
        validation_jsonl=args.validation_jsonl,
        lora_config=LoRAConfig(target_modules=args.target_modules),
        training_config=TrainingConfig(
            max_steps=args.max_steps,
            max_length=args.max_length,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            device=args.device,
            use_fast_tokenizer=not args.slow_tokenizer,
        ),
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
