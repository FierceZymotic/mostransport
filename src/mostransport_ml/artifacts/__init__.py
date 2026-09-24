"""ML artifact bundle — ещё не реализован; `metadata.py` — исключение.

`metadata.ArtifactMetadata` — минимальная, стабильная форма метаданных
(версия модели, информация о target, опциональный validation MAE). Это
намеренно не artifact loader: здесь нет загрузки `.cbm`/joblib/
preprocessing, потому что реальный формат модели неизвестен, пока не
появится реальная модель.

Владелец — Valeria. Ожидаемая форма полноценного artifact bundle, когда
появится реальный model pipeline — docs/ARCHITECTURE.md §7.
"""
