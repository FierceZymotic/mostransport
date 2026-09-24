# mostransport-ml

ML-часть хакатона Мостранспорта: раннее прогнозирование задержек
наземного городского транспорта. Официальная метрика — MAE фактической
задержки. Проект — offline-инструментарий fz (`data → target →
evaluation correctness`) плюс provisional serving-фундамент Valeria
(`features → model → serving`), собранный заранее, чтобы не блокировать
интеграцию с backend'ом на ожидании реальной модели.

Это не runtime backend — тот живёт в отдельном репозитории
(`/home/fz/projects/backend`, NestJS/TypeScript), доступен отсюда только
READ-ONLY. Подробнее о границе — [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) §5.

## С чего начать чтение

1. **README** (этот файл) — быстрый вход и команды.
2. [`docs/PROJECT_KNOWLEDGE.md`](docs/PROJECT_KNOWLEDGE.md) — главная база
   знаний: назначение, ownership, TBD, модули, зависимости.
3. [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — технические границы,
   диаграммы, инварианты.
4. [`docs/ML_SERVING_CONTRACT.md`](docs/ML_SERVING_CONTRACT.md) — точный
   HTTP-контракт serving-слоя.
5. [`docs/HACKATHON_RUNBOOK.md`](docs/HACKATHON_RUNBOOK.md) — что делать
   после публикации официального ТЗ.

Для AI coding agents — сначала [`AGENTS.md`](AGENTS.md).

## Что уже реализовано

- **Offline (fz)**: schema-agnostic инспекция CSV (`inspect_csv.py`),
  `DatasetManifest` (метаданные воспроизводимости, никогда не сырые
  строки), `CanonicalMapping` (явный rename/required-field механизм),
  `TargetSpec`, строго хронологический `split_by_time_boundaries`,
  `mae()`, `MedianBaselineRegressor`, `ExperimentLogger`.
- **Serving (Valeria, provisional)**: FastAPI-shell (`GET /health`,
  `GET /ready`, `POST /api/v1/predict/batch`), детерминированный
  `MockPredictor`, runtime-проверяемый контракт `Predictor` (readiness,
  защита от мутации запроса), санитизированные ошибки, минимальный
  `ArtifactMetadata`.

## Что ещё не реализовано

Реальный target, реальные признаки, реальная модель, реальный Artifact
Bundle, реальный predictor — всё это ждёт публикации официального ТЗ
организаторов. Полный список non-goals —
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) §16.

## Роли fz и Valeria

fz владеет offline-корректностью данных/таргета/оценки; Valeria — общим
Feature Builder, моделью и serving-слоем. Полная таблица ownership,
включая Andrey и Lisa — [`docs/PROJECT_KNOWLEDGE.md`](docs/PROJECT_KNOWLEDGE.md) §10.4.

## Быстрый старт: offline

```bash
cd /home/fz/projects/mostransport-ml
uv sync --extra dev
uv run python scripts/smoke_offline.py
uv run python scripts/inspect_csv.py --path data/raw/organizer.csv
```

## Быстрый старт: serving

```bash
uv sync --extra dev --extra serving
uv run uvicorn mostransport_ml.serving.mock_app:app --host 127.0.0.1 --port 8000
```

Затем: `GET http://127.0.0.1:8000/health`, `/ready`, `/docs`.

## Тесты

```bash
uv sync --extra dev --extra serving   # обязателен для полного набора тестов
uv run pytest -q
uv run ruff check .
```

## Куда идти за подробностями

Все команды и нюансы (proxy в WSL, гигиена данных, git) —
[`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md). Про данные организаторов —
[`data/README.md`](data/README.md) и
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) §13.
