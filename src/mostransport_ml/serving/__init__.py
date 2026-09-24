"""Python ML inference service (FastAPI) — provisional pre-hackathon shell.

What exists now: an HTTP shell (`app.py`), a replaceable `Predictor`
boundary (`service.py`), a deterministic `MockPredictor` (`mock.py`), and an
explicit mock dev app (`mock_app.py`). No real feature engineering or model
exists yet — see docs/ML_SERVING_CONTRACT.md and docs/ARCHITECTURE.md §7.

Owned by Valeria. Stays stateless with respect to per-vehicle history:
recent telemetry is supplied by the backend on each request, never stored
here.
"""
