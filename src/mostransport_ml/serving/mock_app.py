"""Explicit development app wired to `MockPredictor`.

Run with:

    uv run uvicorn mostransport_ml.serving.mock_app:app --host 127.0.0.1 --port 8000

This mock wiring is always explicit — `serving/app.py::create_app` never
defaults to `MockPredictor` on its own; this module is the one place that
chooses to use it.
"""

from __future__ import annotations

from mostransport_ml.serving.app import create_app
from mostransport_ml.serving.mock import MockPredictor

app = create_app(MockPredictor())
