# Docker Compose: полный стек для демо

Поднимает вместе: `ml` (ML-сервис), `backend` (NestJS), `frontend` (React,
за nginx), `emulator` (NDTP-эмулятор организаторов) и `emulator-init`
(одноразовая настройка эмулятора).

## 0. Разовая подготовка

### ML artifact

ML-сервису нужен Artifact Bundle, примонтированный в `/artifact`. Пока нет
финальной модели — используем integration-only fixture (см.
`docs/BACKEND_ML_INTEGRATION.md`, §0):

```bash
uv sync --extra serving
uv run python scripts/build_integration_artifact.py \
  --output ./ml-artifact --replace
```

Это создаст `./ml-artifact/` — путь, который `docker-compose.yml` монтирует
в `ml` по умолчанию (переопределяется через `ML_ARTIFACT_DIR` в `.env`).

> Когда появится реальная модель — просто пересоздать `./ml-artifact` (или
> поменять `ML_ARTIFACT_DIR`) её Artifact Bundle. Ничего в compose/backend
> менять не нужно.

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
   ТС с автогенерируемой телеметрией (`G6CellNav00`) раз в 5 секунд.
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

- **`ready: false` на ML, если `./ml-artifact` не создан** — см. шаг 0.
- **Frontend всё ещё на mock-данных** — WebSocket-слой backend↔frontend не
  реализован, это следующий шаг разработки, не задача этого compose-файла.
- **`PredictionService` — заглушка контракта**: шлёт `unknown`/фиктивные
  `schedule_context`, поэтому прогнозы ML по нему бессмысленны уже на этом
  уровне (не только из-за integration-fixture). Реальный schedule/route
  matching (§5.1 контракта) не реализован — нужны данные расписания.
- **Нет automatic scheduler** — прогноз не запускается сам по приходу
  телеметрии, только вручную через `GET /prediction/run/:unitId`.
