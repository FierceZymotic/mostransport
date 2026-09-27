"""Метаданные и целостность ML-артефактов. Загрузки моделей здесь нет.

- `manifest.ArtifactManifest` — Artifact Manifest v1: модельно-независимые,
  строго валидируемые метаданные артефакта с канонической сериализацией,
  SHA-256 fingerprint и проверкой совместимости по `feature_schema_version`
  (`validate_compatibility`: одна закреплённая схема или набор поддерживаемых).
- `bundle` — Artifact Bundle v1: каталог `manifest.json` + бинарник модели +
  `bundle.json` с SHA-256 обоих файлов; `load_bundle` проверяет безопасность
  имён, хеши, manifest, совместимость и поддержку `model_family` до того, как
  кто-либо загрузит модель.
- `metadata.ArtifactMetadata` — LEGACY: прежняя provisional-форма метаданных,
  сохранена без изменений только для обратной совместимости. Новый код
  использует `ArtifactManifest`.

Загрузка/сериализация конкретных model family — `mostransport_ml.inference`.
Пакет зависит только от stdlib.

Архитектурный контекст — docs/ARCHITECTURE.md.
"""
