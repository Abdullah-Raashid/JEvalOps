FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY scripts ./scripts
COPY configs ./configs
COPY data/dataset_card.md ./data/dataset_card.md
RUN pip install --no-cache-dir -e ".[api]"
EXPOSE 8000
CMD ["uvicorn", "jevalops.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
