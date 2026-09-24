# DEVELOPMENT — практическое руководство разработчика

Команды и workflow для работы с этим репозиторием. Не описывает backend
Андрея — только `mostransport-ml`. Общая картина проекта — в
[`PROJECT_KNOWLEDGE.md`](PROJECT_KNOWLEDGE.md), технические границы — в
[`ARCHITECTURE.md`](ARCHITECTURE.md).

## Окружение

- **OS**: Windows 11 + WSL2, дистрибутив Ubuntu.
- **Python**: 3.12 (см. `.python-version`).
- **Менеджер пакетов**: [`uv`](https://docs.astral.sh/uv/).
- **Путь репозитория**: `/home/fz/projects/mostransport-ml`. Репозиторий
  должен жить именно под `/home/fz/projects`, а не под `/mnt/c/...` — путь
  в Windows-файловой системе через WSL9p делает файловые операции (в
  частности `uv sync` и pytest) заметно медленнее и иногда ломает
  file-watcher'ы. Держите репозиторий на Linux-файловой системе WSL.
- Backend Андрея — отдельный репозиторий, `/home/fz/projects/backend`.
  Для работы над `mostransport-ml` он не нужен; см. ниже, если нужно
  свериться с его текущим состоянием.

## Offline-only workflow

Только зона fz: `data/`, `target/`, `evaluation/`, `experiments/`. Лёгкий
набор зависимостей, без FastAPI/pydantic/uvicorn.

```bash
cd /home/fz/projects/mostransport-ml
uv sync --extra dev
uv run ruff check .
uv run ruff format --check .
uv run python scripts/smoke_offline.py
```

**`uv sync --extra dev` (без `--extra serving`) не даёт запустить полный
`pytest -q`** — `tests/test_serving_*.py` импортируют `fastapi`, которого
в этом наборе зависимостей нет, и pytest падает уже на сборе тестов, до
того как выполнится хоть один test (см. ниже про full-workflow).

Осмотр реального CSV, как только он появится:

```bash
uv run python scripts/inspect_csv.py --path data/raw/organizer.csv
uv run python scripts/inspect_csv.py --path data/raw/organizer.csv \
    --nrows 5000 --time-column event_time
```

## Full / serving workflow

Нужен для serving-разработки и для полного тестового прогона.

```bash
cd /home/fz/projects/mostransport-ml
uv sync --extra dev --extra serving
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
uv run python scripts/smoke_offline.py
uv run python scripts/inspect_csv.py --help
```

Запуск mock-сервера (явно, без скрытого default-to-mock):

```bash
uv run uvicorn mostransport_ml.serving.mock_app:app --host 127.0.0.1 --port 8000
```

Проверка вручную (в отдельном терминале, пока сервер запущен):

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/ready
curl -X POST http://127.0.0.1:8000/api/v1/predict/batch \
  -H 'Content-Type: application/json' \
  -d '{"prediction_time":"2026-01-01T00:00:00Z","vehicles":[{"vehicle_id":"synthetic-1","context":{}}]}'
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
uv run pytest tests/test_serving_errors.py -q   # один файл
uv run pytest -k "mutation" -q                  # по подстроке имени теста
```

Тесты детерминированы, не используют сеть и реальные организаторские
данные — только синтетические payload'ы.

## Ruff

```bash
uv run ruff check .            # линт
uv run ruff format --check .   # проверка форматирования, без изменений
uv run ruff format .           # применить форматирование
```

Конфигурация — в `pyproject.toml` (`[tool.ruff]`).

## Git / гигиена данных

- Организаторские данные никогда не коммитятся: `data/raw/`,
  `data/interim/`, `data/processed/` в `.gitignore` (см.
  [`../data/README.md`](../data/README.md)).
- Перед отправкой организаторских данных в любой внешний
  сервис/инструмент/AI — сначала сверьтесь с правилами хакатона.
- Перед коммитом с широким `git add` проверяйте `git status`/`git diff` —
  не должно случайно попасть ничего из `data/raw|interim|processed/`.
- Ветки/коммиты — по обычным соглашениям команды; отдельного CI в этом
  репозитории нет (см. non-goals в [`ARCHITECTURE.md`](ARCHITECTURE.md) §16).

## Если нужно свериться с backend Андрея

Backend — отдельный репозиторий, `/home/fz/projects/backend`. Из
`mostransport-ml` его код **не изменяется**. Если нужно проверить
совместимость (например, тип `vehicle_id`, текущий эндпоинт телеметрии),
читайте его READ-ONLY и сверяйтесь с
[`PROJECT_KNOWLEDGE.md`](PROJECT_KNOWLEDGE.md) §10.16 и
[`ARCHITECTURE.md`](ARCHITECTURE.md) §5, где уже зафиксирован
проверенный срез его текущего состояния.
