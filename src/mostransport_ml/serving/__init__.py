"""Python ML inference service (FastAPI) — provisional pre-hackathon shell.

Что есть сейчас: HTTP-shell (`app.py`), заменяемая граница `Predictor`
(`service.py`), детерминированный `MockPredictor` (`mock.py`) и explicit
mock dev app (`mock_app.py`). Реальной feature-логики или модели ещё нет —
см. docs/ML_SERVING_CONTRACT.md и docs/ARCHITECTURE.md §7.

Владелец — Valeria. Остаётся stateless по отношению к истории конкретного
vehicle: недавняя телеметрия приходит от backend'а в каждом запросе и
никогда здесь не хранится.
"""
