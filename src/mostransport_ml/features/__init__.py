"""Point-in-time Feature Builder `tabular-v1` (M1).

`builder.build_features()` получает prediction points, telemetry и
плановый schedule-контекст явно (без чтения файлов), чтобы та же
feature-логика могла использоваться offline и online. Схема признаков —
`schema.py`, пространственные helpers — `spatial.py`.
"""
