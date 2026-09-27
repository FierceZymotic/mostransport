# Docker Compose: полный стек для демо

Поднимает вместе: `ml` (ML-сервис), `backend` (NestJS), `frontend` (React,
за nginx), `emulator` (NDTP-эмулятор организаторов) и `emulator-init`
(одноразовая настройка эмулятора).

## 0. Разовая подготовка

### ML artifact

ML-сервису нужен Artifact Bundle, примонтированный в `/artifact`
(`docker-compose.yml` монтирует `${ML_ARTIFACT_DIR:-./ml-artifact}`).

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

Это создаст `./ml-artifact/` — путь по умолчанию. `/ready` показывает, какой
artifact загружен (`model_version`, `feature_schema_version`).

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

1. `ml` стартует, ждём `/health` (healthcheck).
2. `backend` стартует после здорового `ml`.
3. `emulator` стартует.
4. `emulator-init` дожидается готовности control API эмулятора (`:18080`) и
   один раз шлёт `POST /api/config` (`docker/emulator-config.json`):
   указывает эмулятору слать NDTP на `backend:9000` и поднимает 4 условных
   ТС с автогенерируемой телеметрией (`G6CellNav00`) раз в 5 секунд (это
   настройка checked-in конфигурации, а не свойство реального потока).
5. `frontend` стартует, отдаёт статику на `:8080`.

## 2. Проверка

```bash
curl http://localhost:8000/health          # ML
curl http://localhost:3000/prediction/run/1001   # backend → ML (ручной триггер, пока нет scheduler)
curl http://localhost:18080/api/config     # текущая конфигурация эмулятора
open http://localhost:8080                 # dashboard (пока на mock-данных)
```

Логи backend должны показывать `VehicleState: ...` по мере прихода пакетов
от эмулятора.

## 3. Известные ограничения этой сборки

- **`ready: false` на ML, если каталог `ML_ARTIFACT_DIR` (по умолчанию
  `./ml-artifact`) не создан** — см. шаг 0.
- **Frontend всё ещё на mock-данных** — WebSocket-слой backend↔frontend не
  реализован, это следующий шаг разработки, не задача этого compose-файла.
- **Контекст запроса к ML формирует backend**: `PredictionService` строит
  `schedule_context` из расписания в БД (trip matcher, целевое событие,
  `current_deviation_seconds`). Соответствие этих значений правилам §5.1
  контракта (в т.ч. point-in-time семантика отклонения, на которой обучена
  финальная модель) проверяется на стороне backend; из ML-зоны оно не
  подтверждено. С integration-fixture прогнозы бессмысленны в любом случае.
- **Нет automatic scheduler** — прогноз не запускается сам по приходу
  телеметрии, только вручную через `GET /prediction/run/:unitId`.
