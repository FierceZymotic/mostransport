# HACKATHON_RUNBOOK — рабочий цикл после закрытия инфраструктуры

Хакатон длится **48 часов**: предпочитайте работающий end-to-end путь
улучшению отдельного компонента.

## Что уже сделано (не переделывать)

| Фаза | Результат | Checkpoint |
|---|---|---|
| Task spec / данные | семантика target, горизонт `(10, 15]`, MAE, свойства и утечки датасета доказаны | `dataset-evidence-v1` (92-ячеечная версия; см. `notebooks/fz/01_official_dataset_evidence.PROVENANCE.md`) |
| Первая модель (ИСТОРИЧЕСКИ) | безопасные loaders, `tabular-v1`, CatBoost, 6 заранее заданных экспериментов | `m1-tabular-baseline-v1` |
| Паритет offline ↔ online | `CanonicalBatch`, один builder для обоих путей | `m2-i1-streaming-context-v1` |
| Инфраструктура | aware-UTC Contract v1, ArtifactManifest, Artifact Bundle v1, `ArtifactPredictor`, `POST /api/v1/predict`, validate submission | M2 infrastructure closure |
| P1 runtime-safe contract | frozen `runtime-safe-v1` (29, проекция 37 → 29), offline safe current deviation, override в `offline_context` | `ml-runtime-safe-v1` |
| P2 HGB artifact capability | family `hist_gradient_boosting` (`skops`), H0 (`models/hgb_v1.py`), `ArtifactPredictor` выбирает схему по manifest | `ml-hgb-artifact-v1` |
| P3 final training path | Group A → safe deviation → builder → `runtime-safe-v1` → OOF gate → финальный H0 → bundle → load-back parity; submission и serving проверены | P3 (`scripts/train_final_hgb.py`) |

Research-решение заморожено: `HistGradientBoostingRegressor` H0, DIRECT,
`runtime-safe-v1`, только группа A. Не переоткрывать: HPO, другие семейства,
группа B, ансамбли, новые признаки.

## Текущий цикл: финальный artifact → submission → serving

```bash
uv sync --extra dev --extra serving
export MOSTRANSPORT_DATASET=/path/to/official/dataset

# 1. обучение: OOF gate → финальный H0 → Artifact Bundle v1 → load-back parity
uv run python scripts/train_final_hgb.py \
  --artifact-dir artifacts/hgb-h0-runtime-safe-v1-group-a-v1 \
  --report artifacts/reports/p3-hgb-h0-group-a-v1.json

# 2. validate submission (только validate/points, traffic, schedule_plan, sample_submission)
uv run python scripts/make_submission.py \
  --artifact-dir artifacts/hgb-h0-runtime-safe-v1-group-a-v1 \
  --output submission-hgb-h0-runtime-safe-v1.csv

# 3. serving + Contract smoke
MOSTRANSPORT_ARTIFACT_DIR=artifacts/hgb-h0-runtime-safe-v1-group-a-v1 \
  uv run uvicorn mostransport_ml.serving.artifact_app:create_app_from_env --factory
uv run python scripts/smoke_contract_v1.py \
  --base-url http://127.0.0.1:8000 --expected-feature-schema-version runtime-safe-v1
```

Шаг 1 (`training/final_hgb.py`):

1. Группа A — train points, чей `tr_id` входит в planned-only real universe
   (`load_shared_real_vehicle_ids`, плановые поля `test/schedule.csv`); ожидается
   1141 точка / 13 `tr_id`, иначе остановка.
2. Target — по `sample_id` из `labels_train`.
3. Safe deviation: `load_train_schedule_facts` (только `train/schedule.csv`,
   только факт-колонки, порядок CSV) → `safe_current_deviation_seconds`.
   Official `cur_dev_s` в обучение не попадает.
4. `offline_context(..., current_deviation_seconds=safe)` →
   `build_features_from_context` → `project_runtime_safe_features` (1141 × 29,
   NaN допустимы, ±inf нет).
5. `GroupKFold(5, groups=tr_id)` OOF H0 → gate: |OOF MAE − 77.34108978455868| ≤ 1e-9
   (research reference). Провал → artifact не создаётся (exit code 3).
6. `fit_h0` на всей группе A → manifest (`model_version =
   hgb-h0-runtime-safe-v1-group-a-v1`, evaluation = OOF, честный
   `code_provenance`: `git_head`, `git_dirty`, `source_tree_sha256`) →
   `export_bundle` → свежая загрузка с диска и parity прогнозов.

Artifact, отчёт и submission — в `artifacts/` и `submission*.csv` (в
`.gitignore`), вне официального датасета, не коммитятся. `labels_test`,
`test/traffic.csv`, `validate/**` и факт test schedule training не читает.
Байты `model.skops` не детерминированы (zip-время), поэтому `model_version`
логический и стабилен, а `bundle_sha256` меняется при каждой пересборке.

Submission использует `cur_dev_s` из `validate/points.csv`: factual validate
schedule нет, safe-proxy для validate не восстановить. Это не доказательство
того, что runtime `current_deviation_seconds` Backend'а имеет ту же семантику.

`integration-fixture-v1` / `integration-fixture-hgb-v1`
(`scripts/build_integration_artifact.py`) — **INTEGRATION TEST ONLY · NOT FOR
SUBMISSION · NOT A QUALITY MODEL**. Для Backend смена модели = смена
`MOSTRANSPORT_ARTIFACT_DIR` ([BACKEND_ML_INTEGRATION.md](BACKEND_ML_INTEGRATION.md)).

## Правила

- Research закрыт: производственная приёмка — только воспроизведение
  GroupKFold OOF на группе A. `labels_test` и validate для выбора модели не
  используются.
- Validate: только `validate/points.csv`, `validate/traffic.csv`,
  `validate/schedule_plan.csv`, `sample_submission.csv`; факт test schedule для
  validate запрещён.
- Новые признаки → новая схема рядом с `tabular-v1`/`runtime-safe-v1`, не
  правка замороженных; bundle обязан указывать схему, под которую обучен.
- Закрыто на стороне ML: raw packet counts (исторический шаг CSV ~12–15 с,
  частота runtime-потока не гарантирована → исключены из `runtime-safe-v1`);
  current deviation (обучение на point-in-time-safe отклонении); группа B;
  выбор модели.
- Открыто на стороне интеграции (не ML-модель): совпадение runtime
  `current_deviation_seconds` Backend'а с той же point-in-time семантикой,
  выбор целевого события при ничьей, реальная частота runtime-telemetry.

## Исторический контекст (M1, CatBoost / `tabular-v1`)

M1 обучал CatBoost на `tabular-v1` (official `cur_dev_s`, оценка на
`labels_test`). Такие artifacts по-прежнему загружаются (legacy), но не
являются production-кандидатом. Путь для legacy-кандидата:

```python
X = build_features_from_context(offline_context(train.points, train.telemetry, train.schedule_plan))
# ... обучить CatBoostRegressor на X (колонки ровно FEATURE_NAMES) ...
manifest = ArtifactManifest(..., model_family="catboost", feature_schema_version="tabular-v1", ...)
export_bundle("artifacts/<version>", model, manifest)   # manifest.json + model.cbm + bundle.json
```

При доменной неоднозначности, не закрытой официальным ТЗ или Dataset
Evidence v1, AI-агент обязан остановиться и спросить, а не угадывать
([`../AGENTS.md`](../AGENTS.md)).
