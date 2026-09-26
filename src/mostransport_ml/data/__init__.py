"""Инспекция датасета, manifest, canonicalization и official adapter.

`inspection`/`manifest`/`canonical` — generic и не предполагают схему
организаторов. `official` — безопасный adapter опубликованного официального
датасета (allowlist колонок, без factual schedule). `safe_deviation` —
offline/training-only: point-in-time-safe текущее отклонение из factual
schedule (`time_fact_begin <= T`); наружу отдаёт только значения отклонения.
"""
