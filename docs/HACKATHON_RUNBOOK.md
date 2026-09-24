# HACKATHON_RUNBOOK — что делать после публикации полного ТЗ

Это практический план действий, **не** speculative implementation plan.
Ничего из шагов ниже не начинать до Phase 0. Не путать со
статусом "TBD" в [`PROJECT_KNOWLEDGE.md`](PROJECT_KNOWLEDGE.md) §10.3 —
этот документ описывает *порядок действий*, когда TBD начнут разрешаться,
а не угадывает их значения заранее.

Хакатон длится **48 часов** — держите это в голове на каждой фазе:
предпочитайте работающий baseline красивой архитектуре.

## Phase 0 — Freeze

Не начинать писать код до фиксации Task Spec. Первое действие всей
команды — прочитать официальные материалы целиком (task spec, схему CSV,
описание эмулятора, mapping) и только потом планировать работу.

## Phase 1 — ML Task Spec

Вместе, всей командой, зафиксировать письменно:

- точное определение target (что такое "задержка" в терминах данных);
- единицу измерения;
- prediction horizon;
- как именно считается MAE (какие строки участвуют, есть ли фильтрация,
  агрегация по маршрутам/остановкам);
- список CSV-файлов и их полей;
- поля эмулятора;
- официальный mapping CSV ↔ emulator fields.

Обновить [`PROJECT_KNOWLEDGE.md`](PROJECT_KNOWLEDGE.md) §10.2/§10.3: то,
что было TBD, помечается ПОДТВЕРЖДЕНО ОРГАНИЗАТОРАМИ, что осталось
неизвестным — остаётся TBD, не додумывается.

## Phase 2 — Canonical Schema

Sync fz + Valeria + Andrey: согласовать общее доменное представление
(имена сущностей `vehicle`/`route`/`trip`/`stop` и их поля), в которое
приводятся и организаторский CSV (через `CanonicalMapping`, offline), и
online-путь backend'а (через будущий Feature Contract, не обязательно
буквально тем же кодом — см. [`PROJECT_KNOWLEDGE.md`](PROJECT_KNOWLEDGE.md)
§10.10).

## Phase 3 — Параллельная работа

**fz:**
- `uv run python scripts/inspect_csv.py` по реальному CSV;
- построение реального `TargetSpec` и target-колонки;
- temporal evaluation (`split_by_time_boundaries`);
- median baseline (`MedianBaselineRegressor` + `mae()`), первый honest
  baseline MAE в experiment log.

**Valeria:**
- `FeatureBuilder` в `features/`;
- модель;
- Artifact Bundle поверх сегодняшнего `ArtifactMetadata`;
- интеграция в `serving/` (замена `MockPredictor` на artifact-backed
  predictor, без изменения HTTP-слоя).

**Andrey:**
- интеграция с backend/emulator online-путём;
- `PredictionService` + ML Client, вызывающий `POST /api/v1/predict/batch`.

**Lisa:**
- поддержка по БД/reference data.

## Phase 4 — Первая реальная модель

Сначала baseline (Phase 3, fz), потом — кандидат вроде CatBoost, если он
подходит под реальные данные. **Не обещать заранее, что CatBoost будет
финальным выбором** — решение зависит от реальных данных и оставшегося
времени.

## Phase 5 — Паритет offline → online

Для одного и того же исторического момента offline-путь (обучение) и
runtime-запрос (`PredictionBatchRequest`) обязаны давать эквивалентную
feature-семантику и семантику прогноза. Это главная проверка на
training-serving skew (см. [`PROJECT_KNOWLEDGE.md`](PROJECT_KNOWLEDGE.md)
§10.15) — без нее locally-хорошая offline-модель может оказаться
бесполезной в проде.

## Phase 6 — End-to-end MVP

Только после того, как весь путь заработал целиком (Phase 5 пройдена):

- HPO;
- более богатые признаки;
- внешние источники данных;
- explainability;

— и только если осталось время. Приоритет всегда у работающего
end-to-end пути, а не у улучшения отдельного компонента.

---

Везде, где на любой фазе встречается вопрос, ответ на который не был
явно зафиксирован в Phase 1 — это TBD, а не место для предположения.
Смотрите [`PROJECT_KNOWLEDGE.md`](PROJECT_KNOWLEDGE.md) §10.3 и
[`../AGENTS.md`](../AGENTS.md) — при неоднозначности в доменном решении
AI-агент обязан остановиться и спросить, а не угадывать.
