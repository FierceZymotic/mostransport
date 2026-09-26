"""Python ML inference service (FastAPI) по Backend → ML Contract v1.

`POST /api/v1/predict`: Contract v1 запрос → общий runtime-адаптер →
CanonicalBatch → Predictor → ответ. Реальный predictor —
`inference.ArtifactPredictor` из Artifact Bundle v1 (`artifact_app.py`);
`MockPredictor` подключается только явно (`mock_app.py`). Схема признаков и
модель здесь не реализуются. Serving stateless: история telemetry приходит от
Backend'а в каждом запросе и не хранится. Контракт —
docs/BACKEND_ML_INTEGRATION.md, карта реализации — docs/ML_SERVING_CONTRACT.md.
"""
