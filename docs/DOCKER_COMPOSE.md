# Docker Compose: полный стек для демо

Поднимает вместе: `ml` (ML-сервис), `backend` (NestJS), `frontend` (React,
за nginx), `emulator` (NDTP-эмулятор организаторов) и `emulator-init`
(одноразовая настройка эмулятора).

## 0. Разовая подготовка

### ML artifact

ML-сервису нужен Artifact Bundle, примонтированный в `/artifact`
(`docker-compose.yml` монтирует
`${ML_ARTIFACT_DIR:-./artifacts/hgb-h0-runtime-safe-v1-group-a-v1}`).
Healthcheck `ml` — `/ready` (200 только когда модель загружена и совместима);
`/health` — только liveness. Каталог `/artifacts/` в git не хранится.

**Финальная модель** (`hgb-h0-runtime-safe-v1-group-a-v1`) строится из
официального датасета (см. `docs/HACKATHON_RUNBOOK.md`) и подключается через
`ML_ARTIFACT_DIR` в `.env` — compose и backend не меняются:

```bash
uv sync --extra dev --extra serving
MOSTRANSPORT_DATASET=/path/to/official/dataset uv run python scripts/train_final_hgb.py \
  --artifact-dir artifacts/hgb-h0-runtime-safe-v1-group-a-v1
echo "ML_ARTIFACT_DIR=./artifacts/hgb-h0-runtime-safe-v1-group-a-v1" >> .env
```

Без официального датасета — integration-only fixture (INTEGRATION TEST ONLY,
прогнозы бессмысленны; см. `docs/BACKEND_ML_INTEGRATION.md`, §0):

```bash
uv sync --extra serving
uv run python scripts/build_integration_artifact.py \
  --output ./ml-artifact --replace
```

Это создаст `./ml-artifact/`; подключается только явно: `ML_ARTIFACT_DIR=./ml-artifact`.
`/ready` показывает, какой artifact загружен (`model_version`,
`feature_schema_version`); для финала — `hgb-h0-runtime-safe-v1-group-a-v1`.

### NDTP-эмулятор

Образ организаторов не публикуется в registry — грузим из переданного
tar-архива один раз:

```bash
docker load -i ndtp-telemetry-emulator.tar
# должно появиться: ndtp-telemetry-emulator:1.0
```

### Переменные окружения

```bash
cp .env.example .env
# впиши реальный VITE_YANDEX_MAPS_API_KEY
```

## 1. Запуск

```bash
docker compose up --build
```

Порядок автоматический:

1. `ml` стартует; healthcheck — `/ready` (модель загружена).
2. `backend` стартует после старта `ml` (не ждёт готовности модели): приём
   телеметрии и UI работают, а прогноз, пока `ml` не готов, отвечает `503`.
3. `emulator` стартует.
4. `emulator-init` дожидается готовности control API эмулятора (`:18080`) и
   один раз шлёт `POST /api/config` (`docker/emulator-config.json`):
   указывает эмулятору слать NDTP на `backend:9000` и поднимает 4 условных
   ТС с автогенерируемой телеметрией (`G6CellNav00`) раз в 5 секунд (это
   настройка checked-in конфигурации, а не свойство реального потока).
5. `frontend` стартует, отдаёт статику на `:8080`.

## 2. Проверка

```bash
curl http://localhost:8000/ready           # ML: 200 только с загруженной моделью
curl http://localhost:3000/prediction/run/1001   # backend → ML (ручной триггер; live-прогнозы идут сами)
curl http://localhost:18080/api/config     # текущая конфигурация эмулятора
open http://localhost:8080                 # dashboard (пока на mock-данных)
```

Логи backend должны показывать `VehicleState: ...` по мере прихода пакетов
от эмулятора.

## 3. Известные ограничения этой сборки

- **`ready: false` на ML, если каталог `ML_ARTIFACT_DIR` (по умолчанию
  `./artifacts/hgb-h0-runtime-safe-v1-group-a-v1`) не содержит bundle** — см. шаг 0.
- **Frontend всё ещё на mock-данных** — WebSocket-слой backend↔frontend не
  реализован, это следующий шаг разработки, не задача этого compose-файла.
- **Контекст запроса к ML формирует backend**: `PredictionService` строит
  `schedule_context` из расписания в БД (trip matcher, целевое событие,
  `current_deviation_seconds`). Соответствие этих значений правилам §5.1
  контракта (в т.ч. point-in-time семантика отклонения, на которой обучена
  финальная модель) проверяется на стороне backend; из ML-зоны оно не
  подтверждено. С integration-fixture прогнозы бессмысленны в любом случае.
- **Часы демо (`DEMO_CLOCK_MODE`).** Эмулятор ставит в `G6CellNav00` текущее
  время. В этом стеке по умолчанию `day_shift`: вся сессия backend сдвигается на
  целое число суток на `DEMO_CLOCK_TARGET_DAY` (2026-01-06) — один постоянный
  сдвиг, интервалы и порядок пакетов сохраняются. Сдвиг фиксируется при старте
  backend: после полуночи UTC время переходит на 2026-01-07, где расписания нет
  (перезапустите backend). Для данных организатора, уже лежащих на 2026-01-06, —
  `DEMO_CLOCK_MODE=off`. Тот же сдвиг используется для T, окна истории, trip
  matching и выбора целевого события.
- **`current_deviation_seconds` в live-режиме недоступен.** В системе нет
  источника фактических времён прохождения (`schedule_actions.time_fact_begin`
  никто не пишет). При `SCHEDULE_FACT_SOURCE=none` (по умолчанию) backend
  шлёт `0` (Contract v1 не умеет «неизвестно»), в `predictions.reason` пишет
  `degraded:current_deviation_unavailable(no_fact_source)` (виден в
  `/prediction/dashboard/alerts`), а WS-событие `prediction` несёт
  `meta.current_deviation_status`. Факты из GPS не выводятся. Только для
  воспроизведения train-дня: `python scripts/import_db_seed.py --replay-facts
  <train/schedule.csv>` и `SCHEDULE_FACT_SOURCE=replay_import` (используются
  только факты `≤ T`).
- **Идентичность ТС и рейса.** `unit_id` — полный u32 `peerAddress` NPL.
  Таблицу `vehicles (unit_id → current_tr_id)` заполняет только
  `import_db_seed.py --vehicles-from-traffic <traffic.csv>`; для сопоставленных
  ТС используется только их рейс, для остальных (например, юниты эмулятора) —
  ближайший рейс с событием в горизонте `(T+10, T+15]` мин. Если несколько
  разных остановок имеют одно и то же ближайшее плановое время, прогноз не
  строится (`422 TARGET_AMBIGUOUS`, live-сервис пропускает цикл).
  `vehicle_context.route_id` = `tr_id`: сущности маршрута нет, поле — трассировка.
- **Contract v1 и пакеты без навигации.** ~6 % строк телеметрии организатора не
  содержат навигации (нет координат и скорости). Contract v1 требует конечную
  `speed`, поэтому такие пакеты точно не представимы: известное ограничение,
  контракт в этом проходе не меняется.
- **Live-прогноз** — `LivePredictionService` запускает прогноз сам по приходу
  пакета с валидным GPS (не чаще раза в 60 с времени событий на ТС) при
  `T` = время этого пакета; результат уходит WS-событием `prediction`.
  Вручную: `GET /prediction/run/:unitId` (T = текущее время в часах демо) или
  `GET /prediction/run-at/:unitId?time=<ISO-8601>` (явное T).
