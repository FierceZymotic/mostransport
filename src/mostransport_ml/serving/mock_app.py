"""Explicit development app, подключённое к `MockPredictor` (Contract v1, delay=0).

Запуск:

    uv run uvicorn mostransport_ml.serving.mock_app:app --host 127.0.0.1 --port 8000

Это подключение mock'а всегда явное — `serving/app.py::create_app`
никогда не выбирает `MockPredictor` по умолчанию; этот модуль — то самое
единственное место, которое решает его использовать.
"""

from __future__ import annotations

from mostransport_ml.serving.app import create_app
from mostransport_ml.serving.mock import MockPredictor

app = create_app(MockPredictor())
