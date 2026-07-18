from __future__ import annotations

from pydantic import BaseModel

from jevalops.inference.rule_based_backend import RuleBasedBackend

try:
    from fastapi import FastAPI
except ImportError:  # pragma: no cover - exercised only when optional API deps are absent.
    FastAPI = None


class EvaluationRequest(BaseModel):
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.0


backend = RuleBasedBackend()

if FastAPI is not None:
    app = FastAPI(title="JEvalOps API", version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/generate")
    def generate(request: EvaluationRequest) -> dict:
        return backend.generate(request.prompt, request.max_tokens, request.temperature).to_dict()
else:
    app = None
