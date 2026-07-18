from __future__ import annotations

from pathlib import Path


def train_lora(model_name: str, train_jsonl: Path, output_dir: Path) -> None:
    try:
        import peft  # noqa: F401
        import transformers  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "LoRA training requires optional dependencies. Install with `pip install -e '.[train]'` "
            "and run on a machine with enough memory for the selected model."
        ) from exc
    output_dir.mkdir(parents=True, exist_ok=True)
    raise NotImplementedError(
        f"Training entrypoint is wired for {model_name} with {train_jsonl}, but the concrete "
        "trainer should be configured after choosing hardware and base model."
    )
