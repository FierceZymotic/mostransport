# Backend → ML: интеграция по Contract v1

Канонический документ для владельца Backend (Andrey): как вызывать ML-сервис,
что отправлять и что приходит в ответ. Прочитав только его, можно
интегрироваться, не зная истории ML-разработки. Это единственное каноническое
описание контракта; источник истины по схеме — Pydantic-модели сервиса
(`/openapi.json` работающего сервиса), этот документ им соответствует.

> **Статус контракта.** Contract v1 согласован и **заморожен** — и форма
> запроса/ответа, и семантика полей (включая §5.1 и `prediction.target_time`,
> §8). Изменение формы — только новой версией контракта.

## 0. Быстрый старт

**Финальная модель** — `hgb-h0-runtime-safe-v1-group-a-v1` (HGB H0, DIRECT,
схема признаков `runtime-safe-v1`). Её Artifact Bundle v1 воспроизводимо
строится из официальных данных одной командой ML
(`scripts/train_final_hgb.py`, см. [HACKATHON_RUNBOOK.md](HACKATHON_RUNBOOK.md))
и не коммитится: организаторские данные и производные artifacts живут
локально. Backend'у нужен только путь к этому каталогу
(`MOSTRANSPORT_ARTIFACT_DIR`); форма Contract v1 не меняется.

Без официального датасета для интеграции есть детерминированный
**integration-only artifact** (**INTEGRATION TEST ONLY · NOT FOR SUBMISSION · NOT A QUALITY MODEL**): крошечная синтетическая модель.

> **INTEGRATION TEST ONLY · NOT FOR SUBMISSION · NOT A QUALITY MODEL.**
> `model_version = integration-fixture-v1`. Её прогнозы бессмысленны; она
> нужна только чтобы проверить HTTP-интеграцию.

```bash
# из корня репозитория
uv sync --extra serving
# INTEGRATION TEST ONLY · NOT FOR SUBMISSION · NOT A QUALITY MODEL
uv run python scripts/build_integration_artifact.py \
  --output /tmp/mostransport-integration-artifact --replace
MOSTRANSPORT_ARTIFACT_DIR=/tmp/mostransport-integration-artifact \
  uv run uvicorn mostransport_ml.serving.artifact_app:create_app_from_env \
  --factory --host 0.0.0.0 --port 8000
```

В другом терминале:

```bash
curl -s --noproxy '*' http://127.0.0.1:8000/health
curl -s --noproxy '*' http://127.0.0.1:8000/ready
curl -s --noproxy '*' -X POST http://127.0.0.1:8000/api/v1/predict \
  -H 'Content-Type: application/json' -d @tests/fixtures/contract_v1_request.json
uv run python scripts/smoke_contract_v1.py --base-url http://127.0.0.1:8000
```

Схема работающего сервиса: `http://127.0.0.1:8000/docs` (Swagger UI) и
`http://127.0.0.1:8000/openapi.json`. Датасет организаторов для этого не нужен.

**Замена на финальную модель:** меняется только `MOSTRANSPORT_ARTIFACT_DIR`
(например `artifacts/hgb-h0-runtime-safe-v1-group-a-v1`). Схемы запроса/ответа
и код Backend не меняются; в `/ready` и ответе будут `model_version =
hgb-h0-runtime-safe-v1-group-a-v1` и `feature_schema_version = runtime-safe-v1`
(smoke: `--expected-feature-schema-version runtime-safe-v1`). Integration
artifact той же схемы: `scripts/build_integration_artifact.py --model-family
hist_gradient_boosting` (`integration-fixture-hgb-v1`, тоже INTEGRATION TEST ONLY).

## 1. Состояние системы

**Готово на стороне ML:** Contract v1 (схема + валидация), нормализация
времени в UTC, канонический ML-контекст, общий Feature Builder `tabular-v1`
(и его проекция `runtime-safe-v1`) для offline и runtime, ArtifactManifest / Artifact Bundle v1 с проверкой
целостности и типа модели, artifact-backed predictor, HTTP-сервис,
validate/submission-путь, integration artifact и smoke, воспроизводимое
обучение финального HGB artifact'а (P3).

**Требуется от Backend** (код Backend находится в этом репозитории,
`backend/`; здесь описаны обязанности по контракту, а не аудит его текущего
состояния): приём и
парсинг NDTP (`G6CellNav00`), `VehicleState`, история telemetry по ТС,
schedule/domain matching, заполнение `schedule_context` по §5.1, история
telemetry по §7, решение о пригодности точки к прогнозу, HTTP-клиент ML и
хранение истории прогнозов (§11).

**Модель:** HGB H0 DIRECT на `runtime-safe-v1` (без raw packet counts; group
B не используется), обучена на point-in-time-safe отклонении §5.1; финальный
artifact строится воспроизводимо (P3). **Интеграционное требование:**
присылаемый Backend'ом `current_deviation_seconds` должен иметь ту же
point-in-time семантику §5.1 — это проверяется на стороне Backend, ML не может
его восстановить. Integration artifact — не модель.

## 2. Граница ответственности

| Backend | ML |
|---|---|
| приём и парсинг NDTP | point-in-time валидация (`event_time <= T`) |
| `VehicleState`, `TelemetryHistory` | канонизация запроса |
| schedule/route matching, целевое плановое событие | model-specific признаки (`tabular-v1` / `runtime-safe-v1`) |
| текущее отклонение от расписания, `manual_fill` | проверка и загрузка Artifact Bundle |
| prediction eligibility, момент/частота запросов | прогноз `delay_seconds` |
| ML HTTP client, хранение истории прогнозов | версии модели и схемы признаков |

**Backend присылает доменные факты + сырую telemetry. ML сам строит
model-specific признаки** (rolling-окна, GPS/скоростные статистики).
Backend не считает признаки модели.

## 3. Online-цепочка

```
NDTP emulator → TCP receiver → G6CellNav00 parser → VehicleState       (Backend)
             → история telemetry по ТС                                  (Backend)
             → schedule/domain matching → Contract v1 request           (Backend)
             → POST /api/v1/predict → ML → Contract v1 response         (ML: реализовано)
```

## 4. Эндпоинты

| Method | Path | Ответ |
|---|---|---|
| `GET` | `/health` | `200 {"status": "ok"}` — процесс жив |
| `GET` | `/ready` | `200` если artifact загружен и проверен, иначе `503` |
| `POST` | `/api/v1/predict` | прогноз для одной точки `(tr_id, T)` |

`/ready` (пример с integration artifact — **INTEGRATION TEST ONLY · NOT FOR SUBMISSION · NOT A QUALITY MODEL**):

```json
{"ready": true, "model_version": "integration-fixture-v1", "feature_schema_version": "tabular-v1"}
```

`feature_schema_version` идентифицирует схему признаков загруженного
artifact'а (`tabular-v1` у legacy CatBoost и `integration-fixture-v1`,
`runtime-safe-v1` у выбранного HGB); конкретное значение не предполагайте.

При отсутствующем, повреждённом, несовместимом или неверного типа artifact
(например классификатор вместо регрессии) сервис стартует, но `/ready` →
`503 {"ready": false, "model_version": null, "feature_schema_version": null}`,
и `/api/v1/predict` → `503`. Прогноз при этом не выдаётся никогда.

## 5. Запрос `POST /api/v1/predict`

Пример (это `tests/fixtures/contract_v1_request.json`):

```json
{
  "request_id": "3d6f2c1a-7b8e-4f5a-9c0d-1e2f3a4b5c6d",
  "prediction_time": "2026-01-06T03:35:00Z",
  "vehicle_context": {"unit_id": "unit-demo", "tr_id": "131672", "route_id": "route-demo"},
  "schedule_context": {
    "target_action_id": "53700172828",
    "target_time_begin": "2026-01-06T03:47:00Z",
    "target_lat": 55.7512,
    "target_lon": 37.6184,
    "current_deviation_seconds": 274,
    "manual_fill": false
  },
  "telemetry": [
    {"event_time": "2026-01-06T02:55:00Z", "lat": 55.7401, "lon": 37.6002, "location_valid": true,  "speed": 30.0},
    {"event_time": "2026-01-06T03:25:00Z", "lat": 55.7403, "lon": 37.6004, "location_valid": false, "speed": 10.0},
    {"event_time": "2026-01-06T03:31:00Z", "lat": 55.7404, "lon": 37.6005, "location_valid": false, "speed": 0.0},
    {"event_time": "2026-01-06T03:34:30Z", "lat": 55.7405, "lon": 37.6006, "location_valid": false, "speed": 8.0},
    {"event_time": "2026-01-06T03:35:00Z", "lat": 55.7406, "lon": 37.6007, "location_valid": false, "speed": 11.0},
    {"event_time": "2026-01-06T03:35:00Z", "lat": 55.7407, "lon": 37.6008, "location_valid": false, "speed": 22.0}
  ]
}
```

| Поле | Тип | Правило |
|---|---|---|
| `request_id` | string | непустой; **opaque correlation id**: UUID рекомендуется, не обязателен; возвращается без изменений |
| `prediction_time` | aware ISO-8601 | момент прогноза `T` |
| `vehicle_context.unit_id` / `tr_id` / `route_id` | string | непустые строки; только трассировка, не признаки. Backend заполняет `route_id` значением `tr_id` (сущности маршрута нет) |
| `schedule_context.target_action_id` | string | выбранное целевое плановое событие (§5.1); только трассировка |
| `schedule_context.target_time_begin` | aware ISO-8601 | его плановое время; `target_time_begin − T ∈ (10, 15]` минут |
| `schedule_context.target_lat` / `target_lon` | number | координаты этого события; `[-90, 90]` / `[-180, 180]` |
| `schedule_context.current_deviation_seconds` | number | конечное; point-in-time-safe отклонение (§5.1) |
| `schedule_context.manual_fill` | bool | строго `true`/`false`, как в выбранной строке расписания (§5.1) |
| `telemetry` | array | история (§7); может быть пустой |

Неизвестные поля на верхнем уровне, в `vehicle_context` и `schedule_context`
отклоняются (`422`). Числа — только JSON-числа (не строки и не bool).
Имена `time_fact_begin`, `target_delay_s`, `target_class` запрещены везде.

### 5.1. Как Backend заполняет `schedule_context`

**Целевое событие.** Для `tr_id` и момента `T` взять плановые события
расписания с `T + 10 мин < time_begin <= T + 15 мин` и найти минимальный
`time_begin`. Если на этом времени ровно одно событие:

- `target_action_id` = его `tt_action_item_id`;
- `target_time_begin` = его `time_begin`;
- `target_lat` / `target_lon` = его координаты;
- `manual_fill` = его `manual_fill`.

Правило эмпирически подтверждено на train/test/validate официального
датасета. **Ничья:** если на минимальном `time_begin` несколько событий,
подтверждённого правила выбора у организаторов нет (ни порядок в файле, ни
min/max `tt_action_item_id` его не воспроизводят). Backend разрешает ничью по
своему сопоставлению положения/прогресса ТС на маршруте; если однозначно
разрешить нельзя — не выбирать случайный id и запрос в ML не формировать
(точка непригодна к прогнозу).

**`current_deviation_seconds`** = `actual_pass_time − planned_time` для
**последнего подтверждённо пройденного** события, о прохождении которого
Backend реально знает к моменту `T`; если такого ещё нет — `0`. Только
point-in-time-safe информация: ничего, что стало известно после `T`.
Это runtime-значение **не** является аналогом поля `cur_dev_s`
официального датасета. Финальная модель обучена именно на point-in-time-safe
семантике (последний подтверждённый факт `<= T`, иначе `0`), поэтому
соответствие этому правилу — интеграционное требование к Backend; контракт
оно не меняет.

**`manual_fill`** Backend не вычисляет: передаёт значение из выбранной
строки расписания как есть (`true` — только если в строке уже `true`).
Собственные правила вычисления запрещены; отсутствие значения нельзя молча
считать `false` — для такой точки запрос не формируется. Бизнес-смысл
`manual_fill` в материалах организаторов не определён.

### Пакет telemetry (mapping `G6CellNav00`)

| NDTP `G6CellNav00` | Contract v1 | Правило |
|---|---|---|
| `timestamp` | `event_time` | aware ISO-8601, обязательно |
| `longitude` | `lon` | number или `null` |
| `latitude` | `lat` | number или `null` |
| `locationValid` | `location_valid` | bool, обязательно (producer шлёт `true`/`false`) |
| `speedAvg` | `speed` | **конечное число, обязательно, не `null`** |

`speed` — это `speedAvg` (официальный mapping организаторов: исторический
`traffic.csv speed = G6CellNav00 speedAvg`). `speedMax` — **не** скорость
признаков модели. Все пять полей обязательны; дополнительные сырые поля
допускаются и игнорируются моделью. (Реализация ML дополнительно терпит
`location_valid: null`, трактуя его как невалидный GPS; Backend должен
слать boolean.)

**Strict-valid GPS:** `location_valid == true` **и** `lat != null` **и**
`lon != null`.

## 6. Время

- Backend шлёт **timezone-aware ISO-8601**, предпочтительно UTC `Z`:
  `2026-01-06T03:35:00Z`. Смещения допустимы: `2026-01-06T06:35:00+03:00` —
  тот же момент и тот же прогноз.
- ML переводит время в UTC и хранит внутри как naive UTC (так представлены
  официальные данные, на которых обучаются модели).
- **Naive-время (`2026-01-06T03:35:00`) и unix-числа отклоняются** (`422`).

## 7. История telemetry

`telemetry` для точки `T` обязана содержать:

1. все пакеты с `event_time ∈ (T − 15 мин, T]`;
2. плюс последний пакет с `event_time <= T`;
3. плюс последний strict-valid GPS пакет с `event_time <= T`.

Якорь, уже попавший в окно, не дублируется. При равном `event_time`
сохраняйте порядок источника — он определяет, какой пакет «последний».
Отдельный якорь «последняя non-null скорость» не нужен: `speed` всегда число.

Зачем якоря: часть признаков (`tabular-v1`, а значит и `runtime-safe-v1`)
берёт последнее наблюдение без
нижней границы по времени (`latest_packet_lag_s`, `latest_valid_gps_lag_s`,
координаты последнего валидного GPS → `distance_to_target_m`, `speed_last`).
На официальных данных последний валидный GPS старше 15 минут у ~5.5% точек.
ML не обрезает историю окном.

Границы:

| Пакет | Что происходит |
|---|---|
| `event_time == T` | включён |
| `event_time == T − 15 мин` | не входит в 15-минутное окно (окна полуоткрытые `(T−w, T]`) |
| `event_time > T` | **Backend не должен такое присылать** (producer: `event_time <= prediction_time`); если всё же пришёл, ML его отбрасывает (defense-in-depth) и он не влияет на прогноз |

## 8. Ответ `200`

Реальный ответ integration artifact (**INTEGRATION TEST ONLY · NOT FOR SUBMISSION · NOT A QUALITY MODEL**; значения бессмысленны) на
пример из §5:

```json
{
  "request_id": "3d6f2c1a-7b8e-4f5a-9c0d-1e2f3a4b5c6d",
  "status": "success",
  "prediction": {
    "delay_seconds": 264.28390040781596,
    "target_time": "2026-01-06T03:51:24.283900Z",
    "reason": null
  },
  "generated_at": "2026-01-06T03:35:00.412000Z",
  "model_version": "integration-fixture-v1",
  "feature_schema_version": "tabular-v1"
}
```

- `request_id` — тот же, что в запросе;
- `delay_seconds` — прогноз задержки целевого события
  (`schedule_context.target_action_id`), конечное число секунд: `+`
  опоздание, `−` опережение (`delay = факт − план`);
- `reason` — сейчас всегда `null` (модели причины нет; фиктивные причины не
  генерируются);
- `generated_at` — время ответа ML, UTC;
- `model_version`, `feature_schema_version` — из manifest загруженного artifact
  (`feature_schema_version` — не константа: `tabular-v1` в примере выше,
  `runtime-safe-v1` у выбранного HGB);
- `target_time = schedule_context.target_time_begin + delay_seconds` —
  прогноз фактического момента целевого события (так как
  `delay = факт − план`); всегда UTC; **согласовано**. В примере:
  `03:47:00Z + 264.28 с = 03:51:24.283900Z`.

## 9. Ошибки

| Код | Когда | Тело |
|---|---|---|
| `422` | запрос не проходит схему (неверный тип, `null` speed, naive-время, горизонт вне `(10, 15]`, лишние поля) | `{"detail": [{"loc": [...], "msg": "...", "type": "..."}]}` |
| `503` | artifact не настроен/повреждён/несовместим/неверного типа | `{"detail": "predictor is not ready"}` |
| `500` | внутренняя ошибка инференса | `{"detail": "internal error during inference"}` |

Примеры `422`:

```json
{"detail":[{"loc":["body","telemetry",0,"speed"],"msg":"Input should be a valid number","type":"float_type"}]}
{"detail":[{"loc":["body","prediction_time"],"msg":"Input should have timezone info","type":"timezone_aware"}]}
```

Тела ошибок не содержат значений запроса; неизвестные имена полей в `loc`
заменяются на `<field>`. ML не логирует тела запросов и telemetry.

## 10. Какие ТС отправлять

NDTP-поток может содержать ТС без расписания/контекста (unknown или
context-only). Для них прогноз не нужен: **eligibility решает Backend**.
Вызывайте ML только когда для точки однозначно определён весь
`schedule_context` по §5.1 (целевое событие в горизонте `(10, 15]` мин без
неразрешённой ничьей, его координаты, `current_deviation_seconds`,
`manual_fill` из строки расписания). ML не придумывает контекст за Backend.

## 11. Версионирование

Сохраняйте вместе с каждым прогнозом в истории: `request_id`,
`generated_at`, `model_version`, `feature_schema_version`. Их выдаёт ML,
Backend их не вычисляет. Смена модели видна по `model_version`; смена схемы
признаков — по `feature_schema_version`.

## 12. Частота telemetry и current deviation

> **Не прореживайте telemetry эмулятора на стороне Backend.** Сохраняйте
> историю по §7.

Исторический CSV имеет шаг ~12–15 с между пакетами; частота runtime-потока
не гарантирована и может отличаться. Организаторы пояснили: датасет получен их
внутренним процессингом из telemetry того же типа, что даёт эмулятор; точное
преобразование не раскрыто.

Offline и runtime используют один канонический Feature Builder: при
эквивалентном каноническом входе признаки строятся одним детерминированным
путём. Зависимость от частоты пакетов закрыта на стороне ML: финальная
схема `runtime-safe-v1` не использует счётчики `rows_*`/`valid_gps_count_*`.
Остаётся интеграционное требование: `current_deviation_seconds` по §5.1.
Схема запроса от этого не меняется.

## 13. Docker (опционально)

В репозитории есть минимальный `Dockerfile` ML-сервиса (зависимости строго
из `uv.lock`; artifact монтируется read-only):

```bash
docker build -t mostransport-ml .
docker run --rm -p 8000:8000 \
  -v /tmp/mostransport-integration-artifact:/artifact:ro mostransport-ml
# финальная модель: смонтировать artifacts/hgb-h0-runtime-safe-v1-group-a-v1
```

Сборка образа не проверялась в ML-окружении (Docker там недоступен); шаги
установки и запуск сервиса из такого же изолированного окружения с
read-only artifact проверены. При проблеме со сборкой используйте запуск
через `uv` из §0.
