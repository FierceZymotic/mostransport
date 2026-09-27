"""Point-in-time Feature Builder `tabular-v1` (M1) и его проекция `runtime-safe-v1`.

`builder.build_features()` получает prediction points, telemetry и
плановый schedule-контекст явно (без чтения файлов), чтобы та же
feature-логика могла использоваться offline и online. Схема признаков —
`schema.py`, пространственные helpers — `spatial.py`. Канонический
ML-контекст и единственный вход в builder — `context.py`; нормализующие
адаптеры offline-данных и Backend → ML Contract v1 — `adapters.py`.
"""
