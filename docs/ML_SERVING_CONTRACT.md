# ML serving: реализация Contract v1

**Сам контракт** (запрос, ответ, время, история telemetry, ошибки, quickstart)
описан в одном каноническом документе —
[`BACKEND_ML_INTEGRATION.md`](BACKEND_ML_INTEGRATION.md). Этот файл — только
карта реализации на стороне ML: где какое правило обеспечивается. Provisional
v0 (`POST /api/v1/predict/batch`, непрозрачный `context`) удалён.

## 1. Где что проверяется

| Правило | Где |
|---|---|
| Форма запроса, типы, aware-время, `speed` — конечное число (не `null`), горизонт `(10, 15]`, запрет лишних/запрещённых полей | `serving/schemas.py` (Pydantic, → `422`) |
| aware ISO → naive UTC; naive-время — ошибка; отбрасывание пакетов `> T` (defense-in-depth) | `features/adapters.py::runtime_context` |
| `event_time <= T`, окна `(T−w, T]`, strict GPS, признаки `tabular-v1` | `features/builder.py` (заморожен, M1) |
| Целостность bundle, совместимость схемы признаков, family | `artifacts/bundle.py`, `artifacts/manifest.py` |
| `catboost` = обученная скалярная регрессия (objective и форма выхода самой модели) | `inference/model_families.py` |
| direct/residual → `delay_seconds` | `target/formulation.py::final_prediction` (тот же, что в M1) |
| readiness, контракт predictor'а, `target_time = target_time_begin + delay_seconds`, `generated_at` | `serving/service.py` |
| эндпоинты, санитизация `422`, коды `500`/`503`, логирование | `serving/app.py` |
| загрузка artifact из `MOSTRANSPORT_ARTIFACT_DIR`, `503` при проблеме | `serving/artifact_app.py` |

Внутренний канонический контекст (`features/context.py`) шире wire-контракта:
он допускает отсутствующую скорость исторических данных. Ужесточение
`speed` — только на HTTP-границе.

Pydantic-схема терпимее канонического контракта в одном месте:
`location_valid` принимает `null` (трактуется как невалидный GPS), тогда
как по Contract v1 producer шлёт boolean. Для корректных запросов это
ничего не меняет; ужесточение до strict boolean — отдельное изменение кода
и тестов.

## 2. Точка расширения Predictor

`serving/service.py::Predictor`: `is_ready() -> bool`, `model_version()`,
`feature_schema_version()`, `predict(CanonicalBatch) -> Sequence[float]`
(ровно одно конечное значение на точку). `InferenceService` проверяет это в
рантайме; нарушение → `503` на `/ready` или `500` на `/api/v1/predict`.
Реализации: `inference.ArtifactPredictor` (production), `serving.mock.MockPredictor`
(только явно, `mock_app.py`).

## 3. Безопасность и логирование

- `422` без `input`/`ctx`; неизвестные имена полей в `loc` → `<field>`.
- `500`/`503` — фиксированные сообщения; без текста исключений и traceback.
- В логах только тип события/исключения, число пакетов, версия модели и
  коды причин недоступности artifact — не тела запросов, не telemetry, не
  пути.

## 4. Тесты

- `tests/test_serving_contract_v1.py` — HTTP-контракт, wire-инварианты,
  ошибки, OpenAPI;
- `tests/test_contract_v1_fixture.py` — канонический пример и граничные случаи;
- `tests/test_runtime_timestamps.py` — время;
- `tests/test_catboost_task_safety.py` — тип модели;
- `tests/test_e2e_infrastructure.py`, `tests/test_integration_handoff.py` —
  end-to-end, integration artifact, live smoke.
