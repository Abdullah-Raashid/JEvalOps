from __future__ import annotations

import argparse
import json
from pathlib import Path

from jevalops.analysis.error_analysis import analyze_errors
from jevalops.data.validate import load_jsonl, validate_records
from jevalops.evaluation.evaluator import evaluate_dataset, save_evaluation
from jevalops.inference.huggingface_backend import HuggingFaceBackend
from jevalops.training.regression import evaluate_promotion


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained LoRA adapter against its base model.")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--test-jsonl", type=Path, default=Path("data/test.jsonl"))
    parser.add_argument("--reports-dir", type=Path, default=Path("reports/lora_eval"))
    parser.add_argument("--eval-limit", type=int, default=50)
    parser.add_argument("--eval-max-new-tokens", type=int, default=48)
    parser.add_argument("--slow-tokenizer", action="store_true")
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
            device_map=None,
            default_max_new_tokens=args.eval_max_new_tokens,
            use_fast_tokenizer=not args.slow_tokenizer,
        ),
    )
    candidate = evaluate_dataset(
        examples,
        HuggingFaceBackend(
            args.model_name,
            device_map=None,
            adapter_path=args.adapter_dir,
            default_max_new_tokens=args.eval_max_new_tokens,
            use_fast_tokenizer=not args.slow_tokenizer,
        ),
    )
    save_evaluation(args.reports_dir / "hf_baseline_results.json", baseline)
    save_evaluation(args.reports_dir / "lora_results.json", candidate)

    summary = _summarize(args.model_name, args.adapter_dir, baseline, candidate, len(examples))
    (args.reports_dir / "fine_tuning_results.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.reports_dir / "fine_tuning_report.md").write_text(_render_report(summary, baseline, candidate), encoding="utf-8")
    if args.mlflow_experiment:
        from jevalops.tracking.mlflow import log_finetuning_summary

        log_finetuning_summary(summary, experiment_name=args.mlflow_experiment)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def _summarize(model_name: str, adapter_dir: Path, baseline: dict, candidate: dict, eval_examples: int) -> dict:
    baseline_errors = analyze_errors(baseline)
    candidate_errors = analyze_errors(candidate)
    return {
        "model_name": model_name,
        "adapter_dir": str(adapter_dir),
        "eval_examples": eval_examples,
        "baseline_quality": baseline["quality_score"],
        "lora_quality": candidate["quality_score"],
        "absolute_quality_delta": candidate["quality_score"] - baseline["quality_score"],
        "relative_quality_delta_percent": _relative_delta_percent(baseline["quality_score"], candidate["quality_score"]),
        "baseline_p95_latency_seconds": baseline["efficiency"]["p95_latency_seconds"],
        "lora_p95_latency_seconds": candidate["efficiency"]["p95_latency_seconds"],
        "p95_latency_delta_seconds": candidate["efficiency"]["p95_latency_seconds"] - baseline["efficiency"]["p95_latency_seconds"],
        "baseline_error_count": baseline_errors["error_count"],
        "lora_error_count": candidate_errors["error_count"],
        "promotion": evaluate_promotion(baseline, candidate),
    }


def _relative_delta_percent(baseline: float, candidate: float) -> float:
    return ((candidate - baseline) / baseline * 100) if baseline else 0.0


def _render_report(summary: dict, baseline: dict, candidate: dict) -> str:
    return "\n".join(
        [
            "# Fine-Tuning Evaluation Report",
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
            f"- Baseline error count: {summary['baseline_error_count']}",
            f"- LoRA error count: {summary['lora_error_count']}",
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
        ]
    )


if __name__ == "__main__":
    main()
