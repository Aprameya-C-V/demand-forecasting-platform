# syntax=docker/dockerfile:1
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH"

# libgomp is what LightGBM links against on Linux; curl is used by the healthcheck
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.10.7 /uv /uvx /bin/

WORKDIR /app

# dependencies first, so a source change does not invalidate the dependency layer
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src
COPY configs ./configs
RUN uv sync --frozen --no-dev

# the feature store and the registry are mounted at runtime, never baked into the image
ENV DF_DATA_DIR=/app/data \
    DF_MODEL_DIR=/app/models \
    DF_PANEL_PATH=/app/data/processed/panel.parquet \
    DF_PREDICTION_LOG=/app/data/predictions/predictions.jsonl \
    DF_WORKERS=2

RUN useradd --create-home --uid 1000 app && mkdir -p /app/data /app/models && chown -R app /app
USER 1000

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
    CMD curl -fsS http://127.0.0.1:7860/healthz || exit 1

CMD ["uvicorn", "demand_forecast.serve.app:app", "--host", "0.0.0.0", "--port", "7860"]
