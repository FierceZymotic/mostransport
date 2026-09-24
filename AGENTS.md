# AGENTS.md — правила для AI coding agents

Короткая операционная инструкция для Codex и других AI coding agents,
работающих в этом репозитории. Полная база знаний —
[`docs/PROJECT_KNOWLEDGE.md`](docs/PROJECT_KNOWLEDGE.md) — читайте её
первой при любой задаче сложнее опечатки.

## Что это за проект

`mostransport-ml` — offline ML-инструментарий (fz) плюс provisional
serving-фундамент (Valeria) для 48-часового хакатона Мостранспорта:
раннее прогнозирование задержек наземного транспорта. Официальная
метрика — MAE. Полное ТЗ организаторов ещё не опубликовано — многое ниже
явно помечено TBD, и это не ошибка, а честное текущее состояние.

## Порядок чтения перед изменениями

1. [`docs/PROJECT_KNOWLEDGE.md`](docs/PROJECT_KNOWLEDGE.md) — что это за
   проект, кто чем владеет, что CURRENT / PLANNED / TBD.
2. [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — технические границы,
   диаграммы, инварианты.
3. Если задача касается `serving/` — дополнительно
   [`docs/ML_SERVING_CONTRACT.md`](docs/ML_SERVING_CONTRACT.md).
4. Прогнать тесты: `uv run pytest -q` (нужен `uv sync --extra dev
   --extra serving` — только `--extra dev` не даёт собрать
   `test_serving_*.py`, см. [`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md)).
5. Для своей задачи явно определить: что из затрагиваемого — уже
   реализовано, что только запланировано, а что TBD.

## Ownership и замороженные границы

| Зона | Владелец |
|------|----------|
| `data/`, `target/`, `evaluation/`, `experiments/` | fz |
| `features/`, `models/`, `artifacts/`, `serving/` | Valeria |
| `/home/fz/projects/backend` (отдельный репозиторий) | Andrey |

Не менять ownership без явного решения команды. Backend Андрея —
**READ-ONLY**, если задача явно не поставлена как работа над backend'ом:
можно читать его код для сверки границы (см.
[`docs/PROJECT_KNOWLEDGE.md`](docs/PROJECT_KNOWLEDGE.md) §10.16), но не
изменять ни одного файла в нём.

## Текущие команды валидации

```bash
uv sync --extra dev --extra serving
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
uv run python scripts/smoke_offline.py
```

Полный список команд, включая запуск mock-сервера — в
[`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md).

## Ключевые инварианты (не нарушать)

- Никакой canonical-схемы полей CSV/эмулятора нигде не захардкожено —
  и не должно быть, пока организаторы её не опубликуют.
- `data/canonical.py` — offline DataFrame-механизм, а не обязательный
  runtime-конвертер между backend'ом и Python (детали:
  [`docs/PROJECT_KNOWLEDGE.md`](docs/PROJECT_KNOWLEDGE.md) §10.10).
- Одна и та же feature-логика должна в итоге использоваться offline и
  online (будущий Feature Builder, сейчас `features/` пуст — не
  реализовывать его заранее без Feature Contract sync point).
- Python-сервис не хранит операционное состояние между запросами: ни
  `VehicleState`, ни scheduler, ни историю по vehicle — это зона backend'а.
- `serving/app.py::create_app` никогда не должен по умолчанию
  подставлять `MockPredictor` — только явное подключение
  (`serving/mock_app.py`).
- Организаторские данные не коммитятся и не публикуются; перед отправкой
  куда-либо внешнее — сверяться с правилами хакатона.

## Запрещено агенту

- Угадывать схему организаторов (поля CSV/эмулятора, единицы delay,
  число для horizon) — вместо этого фиксировать как TBD.
- Превращать TBD в факт в коде, тестах или документации.
- Дублировать feature-логику вместо использования общего `features/`
  (после того как он появится).
- Создавать в Python операционный `VehicleState` или scheduler.
- Менять ownership зон без явного решения команды.
- Расширять scope "про запас" — если задача не просит новую подсистему,
  не создавать её.
- Менять backend Андрея, если это явно не поставленная задача.
- Утверждать в документации, что запланированный (PLANNED) компонент уже
  реализован.

## Если решение зависит от организаторского ТЗ

Останавливаться и спрашивать, а не предполагать. Смотреть
[`docs/HACKATHON_RUNBOOK.md`](docs/HACKATHON_RUNBOOK.md) — там описан
порядок действий после публикации ТЗ.

## Документация

При изменении архитектуры/контракта — обновлять соответствующий документ
в том же PR/коммите (не оставлять документацию, противоречащую коду).
Не дублировать один и тот же факт в нескольких документах — см. правило
"роль каждого документа" в [`docs/PROJECT_KNOWLEDGE.md`](docs/PROJECT_KNOWLEDGE.md).
