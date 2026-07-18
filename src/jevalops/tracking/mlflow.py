from __future__ import annotations

from typing import Any


def log_finetuning_summary(summary: dict[str, Any], *, experiment_name: str = "jevalops-lora") -> None:
    try:
        import mlflow
    except ImportError as exc:
        raise ImportError("Install MLflow support with `pip install -e '.[tracking]'`.") from exc

    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=f"{summary['model_name']} LoRA"):
        mlflow.log_param("model_name", summary["model_name"])
        mlflow.log_param("adapter_dir", summary["adapter_dir"])
        mlflow.log_param("eval_examples", summary["eval_examples"])
        mlflow.log_metric("baseline_quality", summary["baseline_quality"])
        mlflow.log_metric("lora_quality", summary["lora_quality"])
        mlflow.log_metric("absolute_quality_delta", summary["absolute_quality_delta"])
        mlflow.log_metric("relative_quality_delta_percent", summary["relative_quality_delta_percent"])
        mlflow.log_metric("baseline_p95_latency_seconds", summary["baseline_p95_latency_seconds"])
        mlflow.log_metric("lora_p95_latency_seconds", summary["lora_p95_latency_seconds"])
        mlflow.log_metric("p95_latency_delta_seconds", summary["p95_latency_delta_seconds"])
        mlflow.log_metric("baseline_error_count", summary["baseline_error_count"])
        mlflow.log_metric("lora_error_count", summary["lora_error_count"])
        training = summary.get("training", {})
        accelerator = training.get("accelerator", {})
        if accelerator:
            mlflow.log_param("accelerator", accelerator.get("accelerator"))
            mlflow.log_param("resolved_device", accelerator.get("resolved_device"))
            mlflow.log_param("apple_metal_enabled", accelerator.get("apple_metal_enabled"))
        for name, passed in summary["promotion"]["checks"].items():
            mlflow.log_metric(f"gate_{name}", int(passed))
