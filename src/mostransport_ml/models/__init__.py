"""Модельный слой: фиксированные конфигурации моделей и train regimes.

`catboost_v1.py` — зафиксированный конфиг M1 CatBoost (legacy, без HPO);
`hgb_v1.py` — зафиксированный production-кандидат HGB H0 (`runtime-safe-v1`,
DIRECT); `regimes.py` — структурные train-группы, режимы обучения и сетка из
шести заранее определённых экспериментов M1.
"""
