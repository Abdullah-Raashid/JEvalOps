from __future__ import annotations

import argparse
import json
from pathlib import Path

from jevalops.analysis.error_analysis import analyze_errors
from jevalops.data.validate import load_jsonl, validate_records
from jevalops.evaluation.evaluator import evaluate_dataset, save_evaluation
from jevalops.inference.huggingface_backend import HuggingFaceBackend
from jevalops.training.lora import LoRAConfig, TrainingConfig, train_lora
from jevalops.training.regression import evaluate_promotion


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a full baseline -> LoRA -> regression experiment.")
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--train-jsonl", type=Path, default=Path("data/processed/sft_train.jsonl"))
    parser.add_argument("--validation-jsonl", type=Path, default=Path("data/validation.jsonl"))
    parser.add_argument("--test-jsonl", type=Path, default=Path("data/test.jsonl"))
    parser.add_argument("--adapter-dir", type=Path, default=Path("checkpoints/lora"))
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    parser.add_argument("--eval-limit", type=int, default=50)
    parser.add_argument("--eval-max-new-tokens", type=int, default=64)
    parser.add_argument("--eval-device-map", default=None)
    parser.add_argument("--eval-device", default="auto", help="Inference device: auto, cpu, cuda, mps, or metal.")
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--mixed-precision", choices=["no", "fp16", "bf16"], default="no")
    parser.add_argument("--target-modules", nargs="*", default=None)
    parser.add_argument("--slow-tokenizer", action="store_true")
    parser.add_argument("--disable-mps-fallback", action="store_true")
    parser.add_argument("--mlflow-experiment", default=None)
    args = parser.parse_args()

    examples, report = validate_records(load_jsonl(args.test_jsonl))
    if report.has_errors:
        raise SystemExit(json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2))
    examples = examples[: args.eval_limit] if args.eval_limit else examples

    args.reports_dir.mkdir(parents=True, exist_ok=True)
    baseline = evaluate_dataset(
        examples,
        HuggingFaceBackend(
            args.model_name,
            device_map=args.eval_device_map,
            device=args.eval_device,
            default_max_new_tokens=args.eval_max_new_tokens,
            use_fast_tokenizer=not args.slow_tokenizer,
            enable_mps_fallback=not args.disable_mps_fallback,
        ),
    )
    save_evaluation(args.reports_dir / "hf_baseline_results.json", baseline)

    training_metrics = train_lora(
        args.model_name,
        args.train_jsonl,
        args.adapter_dir,
        validation_jsonl=args.validation_jsonl,
        lora_config=LoRAConfig(r=args.lora_r, alpha=args.lora_alpha, target_modules=args.target_modules),
        training_config=TrainingConfig(
            max_steps=args.max_steps,
            max_length=args.max_length,
            device=args.device,
            learning_rate=args.learning_rate,
            mixed_precision=args.mixed_precision,
            enable_mps_fallback=not args.disable_mps_fallback,
            use_fast_tokenizer=not args.slow_tokenizer,
        ),
    )
    candidate = evaluate_dataset(
        examples,
        HuggingFaceBackend(
            args.model_name,
            device_map=args.eval_device_map,
            device=args.eval_device,
            adapter_path=args.adapter_dir,
            default_max_new_tokens=args.eval_max_new_tokens,
            use_fast_tokenizer=not args.slow_tokenizer,
            enable_mps_fallback=not args.disable_mps_fallback,
        ),
    )
    save_evaluation(args.reports_dir / "lora_results.json", candidate)

    baseline_errors = analyze_errors(baseline)
    candidate_errors = analyze_errors(candidate)
    promotion = evaluate_promotion(baseline, candidate)
    summary = {
        "model_name": args.model_name,
        "adapter_dir": str(args.adapter_dir),
        "eval_examples": len(examples),
        "baseline_quality": baseline["quality_score"],
        "lora_quality": candidate["quality_score"],
        "absolute_quality_delta": candidate["quality_score"] - baseline["quality_score"],
        "relative_quality_delta_percent": _relative_delta_percent(baseline["quality_score"], candidate["quality_score"]),
        "baseline_p95_latency_seconds": baseline["efficiency"]["p95_latency_seconds"],
        "lora_p95_latency_seconds": candidate["efficiency"]["p95_latency_seconds"],
        "p95_latency_delta_seconds": candidate["efficiency"]["p95_latency_seconds"] - baseline["efficiency"]["p95_latency_seconds"],
        "baseline_error_count": baseline_errors["error_count"],
        "lora_error_count": candidate_errors["error_count"],
        "promotion": promotion,
        "training": training_metrics,
    }
    (args.reports_dir / "fine_tuning_results.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.reports_dir / "fine_tuning_report.md").write_text(_render_report(summary, baseline, candidate), encoding="utf-8")
    if args.mlflow_experiment:
        from jevalops.tracking.mlflow import log_finetuning_summary

        log_finetuning_summary(summary, experiment_name=args.mlflow_experiment)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def _relative_delta_percent(baseline: float, candidate: float) -> float:
    return ((candidate - baseline) / baseline * 100) if baseline else 0.0


def _render_report(summary: dict, baseline: dict, candidate: dict) -> str:
    return "\n".join(
        [
            "# Fine-Tuning Report",
            "",
            f"Base model: `{summary['model_name']}`",
            f"Adapter: `{summary['adapter_dir']}`",
            f"Evaluation examples: {summary['eval_examples']}",
            "",
            "## Before / After",
            "",
            f"- Baseline quality: {summary['baseline_quality']:.4f}",
            f"- LoRA quality: {summary['lora_quality']:.4f}",
            f"- Absolute quality delta: {summary['absolute_quality_delta']:.4f}",
            f"- Relative quality delta: {summary['relative_quality_delta_percent']:.2f}%",
            f"- Baseline P95 latency: {summary['baseline_p95_latency_seconds']:.4f}s",
            f"- LoRA P95 latency: {summary['lora_p95_latency_seconds']:.4f}s",
            f"- P95 latency delta: {summary['p95_latency_delta_seconds']:.4f}s",
            "",
            "## Task Scores",
            "",
            "| Task | Baseline | LoRA | Delta |",
            "| --- | ---: | ---: | ---: |",
            *[
                f"| `{task}` | {baseline['by_task'].get(task, 0):.4f} | {candidate['by_task'].get(task, 0):.4f} | "
                f"{candidate['by_task'].get(task, 0) - baseline['by_task'].get(task, 0):.4f} |"
                for task in sorted(set(baseline["by_task"]) | set(candidate["by_task"]))
            ],
            "",
            "## Promotion Gates",
            "",
            f"Promoted: **{summary['promotion']['promoted']}**",
            *[f"- `{name}`: {passed}" for name, passed in summary["promotion"]["checks"].items()],
            "",
            "## Training",
            "",
            f"- Final train loss: {summary['training']['final_train_loss']:.4f}",
            f"- Mean train loss: {summary['training']['mean_train_loss']:.4f}",
            f"- Training seconds: {summary['training']['training_seconds']:.2f}",
            f"- Resolved device: `{summary['training']['training']['resolved_device']}`",
            "",
        ]
    )


if __name__ == "__main__":
    main()
