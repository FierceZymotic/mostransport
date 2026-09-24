# experiments/

Append-only лог экспериментов, пишется через
`mostransport_ml.experiments.log`.

Сгенерированные файлы лога (например, `runs.jsonl`) в `.gitignore` — это
локальная рабочая история, а не source-of-truth хранилище артефактов.
Если результат нужно сохранить между машинами или поделиться им с
командой — скопируйте нужные строки, а не коммитьте весь лог.

Каждая строка — JSON-объект как минимум с полями:

- `run_id`, `created_at`
- `data_version`, `target_version`, `feature_version`
- `model_name`, `params`
- `validation_mae`
- `notes`

Про сам writer — `mostransport_ml.experiments.log.ExperimentLogger`; про
то, как это встраивается в workflow — см.
[`../docs/ARCHITECTURE.md`](../docs/ARCHITECTURE.md).
