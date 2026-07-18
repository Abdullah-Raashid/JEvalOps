from __future__ import annotations

from pathlib import Path
from typing import Any

from jevalops.analysis.taxonomy import ERROR_TAXONOMY


def render_markdown_report(evaluation: dict[str, Any], error_analysis: dict[str, Any]) -> str:
    lines = [
        "# JEvalOps Evaluation Report",
        "",
        f"Model: `{evaluation.get('model')}`",
        f"Backend: `{evaluation.get('backend')}`",
        f"Composite quality score: **{evaluation.get('quality_score', 0):.3f}**",
        "",
        "## Task Scores",
    ]
    for task, score in evaluation.get("by_task", {}).items():
        lines.append(f"- `{task}`: {score:.3f}")
    efficiency = evaluation.get("efficiency", {})
    lines.extend(
        [
            "",
            "## Efficiency",
            f"- Mean latency seconds: {efficiency.get('mean_latency_seconds', 0):.4f}",
            f"- P95 latency seconds: {efficiency.get('p95_latency_seconds', 0):.4f}",
            f"- Mean tokens/sec: {efficiency.get('mean_tokens_per_second', 0):.2f}",
            "",
            "## Error Summary",
        ]
    )
    for error_type, count in error_analysis.get("by_error_type", {}).items():
        lines.append(f"- `{error_type}`: {count}")
    lines.extend(["", "## Japanese Error Taxonomy"])
    for family, labels in ERROR_TAXONOMY.items():
        lines.append(f"- `{family}`: {', '.join(labels)}")
    return "\n".join(lines) + "\n"


def write_markdown_report(path: Path, evaluation: dict[str, Any], error_analysis: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_markdown_report(evaluation, error_analysis), encoding="utf-8")
