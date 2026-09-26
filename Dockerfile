# ML inference service (Backend → ML Contract v1).
#
# Artifact Bundle v1 is NOT baked into the image: mount it read-only and point
# MOSTRANSPORT_ARTIFACT_DIR at it. Without a valid bundle the service starts but
# /ready and /api/v1/predict answer 503.
#
#   docker build -t mostransport-ml .
#   docker run --rm -p 8000:8000 \
#     -v /tmp/mostransport-integration-artifact:/artifact:ro \
#     mostransport-ml
#
# Dependencies come only from uv.lock (--frozen): no dependency drift.
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.12.17 /uv /bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-install-project --extra serving

COPY src ./src
RUN uv sync --frozen --no-editable --extra serving

RUN useradd --create-home --uid 10001 mlservice
USER mlservice

ENV MOSTRANSPORT_ARTIFACT_DIR=/artifact
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"

CMD ["uvicorn", "mostransport_ml.serving.artifact_app:create_app_from_env", "--factory", "--host", "0.0.0.0", "--port", "8000"]
