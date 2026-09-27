# DEVELOPMENT — практическое руководство разработчика

Команды и workflow для ML-части репозитория (Python). Backend/Frontend
(`backend/`, `frontend/`) описываются своими документами. Все
команды ниже выполняются из корня клонированного репозитория. Общая картина
проекта — в
[`PROJECT_KNOWLEDGE.md`](PROJECT_KNOWLEDGE.md), технические границы — в
[`ARCHITECTURE.md`](ARCHITECTURE.md).

## Окружение

- **OS**: Windows 11 + WSL2, дистрибутив Ubuntu.
- **Python**: 3.12 (см. `.python-version`).
- **Менеджер пакетов**: [`uv`](https://docs.astral.sh/uv/).
- **Где держать клон**: на Linux-файловой системе WSL (например в `~/...`),
  а не под `/mnt/c/...` — путь в Windows-файловой системе через WSL9p делает
  файловые операции (в частности `uv sync` и pytest) заметно медленнее и
  иногда ломает file-watcher'ы.

## Offline-only workflow

Offline/research-работа (loaders, признаки, обучение, bundle, submission)
без FastAPI/pydantic/uvicorn.

```bash
uv sync --extra dev
uv run ruff check src scripts tests
uv run ruff format --check src scripts tests
uv run python scripts/smoke_offline.py
```

**`uv sync --extra dev` (без `--extra serving`) не даёт запустить полный
`pytest -q`** — serving/E2E/handoff-тесты (`tests/test_serving_contract_v1.py`,
`tests/test_e2e_infrastructure.py`, `tests/test_integration_handoff.py` и др.)
импортируют `fastapi`, которого в этом наборе зависимостей нет, и pytest
падает уже на сборе тестов (см. ниже про full-workflow).

Осмотр реального CSV, как только он появится:

```bash
uv run python scripts/inspect_csv.py --path data/raw/organizer.csv
uv run python scripts/inspect_csv.py --path data/raw/organizer.csv \
    --nrows 5000 --time-column event_time
```

## Full / serving workflow

Нужен для serving-разработки и для полного тестового прогона.

```bash
uv sync --extra dev --extra serving
uv run pytest -q
uv run ruff check src scripts tests
uv run ruff format --check src scripts tests
uv run python scripts/smoke_offline.py
uv run python scripts/inspect_csv.py --help
```

Запуск сервера с реальной моделью из Artifact Bundle v1 (production-путь):

```bash
MOSTRANSPORT_ARTIFACT_DIR=<bundle> \
  uv run uvicorn mostransport_ml.serving.artifact_app:create_app_from_env --factory \
  --host 127.0.0.1 --port 8000
```

Или mock для интеграции (явно, delay = 0):

```bash
uv run uvicorn mostransport_ml.serving.mock_app:app --host 127.0.0.1 --port 8000
```

Проверка вручную (в отдельном терминале, пока сервер запущен):

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/ready
curl -X POST http://127.0.0.1:8000/api/v1/predict \
  -H 'Content-Type: application/json' \
  -d @tests/fixtures/contract_v1_request.json
```

Интерактивная документация — `http://127.0.0.1:8000/docs`, JSON-схема —
`http://127.0.0.1:8000/openapi.json`.

Остановить сервер после проверки (например `Ctrl+C` в терминале, где он
запущен, либо `pkill -f "uvicorn mostransport_ml.serving.mock_app:app"`)
— не оставлять его висеть в фоне.

### Известный нюанс: локальный HTTP-прокси в WSL/sandbox-окружениях

В некоторых WSL/sandbox-окружениях переменные `http_proxy`/`https_proxy`
настроены так, что перехватывают даже запросы на `127.0.0.1`, и curl
получает `503` от самого прокси, а не от приложения — это выглядит как
падение сервиса, но им не является. Если `curl http://127.0.0.1:8000/...`
неожиданно возвращает `503 Service Unavailable` без тела ответа,
проверьте это явным обходом прокси для локальных запросов:

```bash
curl --noproxy '*' http://127.0.0.1:8000/health
```

Не меняйте глобальную конфигурацию прокси пользователя ради этого —
`--noproxy '*'` достаточно для одного запроса.

## Тесты

```bash
uv run pytest -q                          # весь набор (нужен --extra serving)
uv run pytest tests/test_serving_contract_v1.py -q   # один файл (HTTP Contract v1)
uv run pytest -k "speed" -q                          # по подстроке имени теста
```

Тесты детерминированы, не используют внешнюю сеть и реальные
организаторские данные — только синтетические payload'ы (live smoke-тест
поднимает локальный uvicorn на `127.0.0.1`).

Интеграционный запуск сервиса с integration-only artifact
(**INTEGRATION TEST ONLY · NOT FOR SUBMISSION · NOT A QUALITY MODEL**), `curl` и smoke
(`scripts/build_integration_artifact.py`, `scripts/smoke_contract_v1.py`) —
[`BACKEND_ML_INTEGRATION.md`](BACKEND_ML_INTEGRATION.md) §0.

## Ruff

```bash
uv run ruff check src scripts tests            # линт
uv run ruff format --check src scripts tests   # проверка форматирования, без изменений
uv run ruff format src scripts tests           # применить форматирование
```

Конфигурация — в `pyproject.toml` (`[tool.ruff]`). Scope `src scripts tests`: замороженный
notebook Dataset Evidence v1 содержит собственные lint-замечания и не правится.

## Git / гигиена данных

- Организаторские данные никогда не коммитятся: `data/raw/`,
  `data/interim/`, `data/processed/` в `.gitignore` (см.
  [`../data/README.md`](../data/README.md)).
- Перед отправкой организаторских данных в любой внешний
  сервис/инструмент/AI — сначала сверьтесь с правилами хакатона.
- Перед коммитом с широким `git add` проверяйте `git status`/`git diff` —
  не должно случайно попасть ничего из `data/raw|interim|processed/`.
- Не коммитить сгенерированное: Artifact Bundle (`/artifacts/`),
  `submission*.csv`, `.env`, `*:Zone.Identifier` (см. `.gitignore`).
- Ветки/коммиты — по обычным соглашениям команды; отдельного CI сейчас нет.

## Граница с Backend/Frontend

Backend и Frontend лежат в этом репозитории как соседние каталоги верхнего
уровня (`backend/`, `frontend/`). Задача на ML не меняет их код (и наоборот);
взаимодействие — только через Backend → ML Contract v1
([`BACKEND_ML_INTEGRATION.md`](BACKEND_ML_INTEGRATION.md)).
