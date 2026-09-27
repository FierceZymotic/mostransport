# HACKATHON_RUNBOOK — рабочий цикл после закрытия инфраструктуры

Хакатон длится **48 часов**: предпочитайте работающий end-to-end путь
улучшению отдельного компонента.

## Что уже сделано (не переделывать)

| Фаза | Результат | Checkpoint |
|---|---|---|
| Task spec / данные | семантика target, горизонт `(10, 15]`, MAE, свойства и утечки датасета доказаны | `dataset-evidence-v1` (92-ячеечная версия; см. `notebooks/fz/01_official_dataset_evidence.PROVENANCE.md`) |
| Первая модель | безопасные loaders, `tabular-v1`, CatBoost, 6 заранее заданных экспериментов | `m1-tabular-baseline-v1` |
| Паритет offline ↔ online | `CanonicalBatch`, один builder для обоих путей | `m2-i1-streaming-context-v1` |
| Инфраструктура | aware-UTC Contract v1, ArtifactManifest, Artifact Bundle v1, `ArtifactPredictor`, `POST /api/v1/predict`, validate submission | M2 infrastructure closure |
| P1 runtime-safe contract | frozen `runtime-safe-v1` (29, проекция 37 → 29), offline safe current deviation, override в `offline_context` | `ml-runtime-safe-v1` |
| P2 HGB artifact capability | family `hist_gradient_boosting` (`skops`), H0 (`models/hgb_v1.py`), `ArtifactPredictor` выбирает схему по manifest | P2 (синтетические artifacts; official artifact не обучен) |

## Цикл modeling → artifact → submission/serving

Любой кандидат поддерживаемой схемы (`tabular-v1` или `runtime-safe-v1`)
проходит один и тот же путь без изменения serving/адаптеров/loader'ов/submission.
Пример ниже — legacy CatBoost/`tabular-v1`; путь production-кандидата HGB — в
разделе «P2 → P3» ниже.

```python
from datetime import UTC, datetime
from mostransport_ml.artifacts.manifest import ArtifactManifest, EvaluationSummary
from mostransport_ml.data.official import load_official_split, resolve_dataset_root
from mostransport_ml.features.adapters import offline_context
from mostransport_ml.features.context import build_features_from_context
from mostransport_ml.inference.predictor import ArtifactPredictor, export_bundle

root = resolve_dataset_root()                      # $MOSTRANSPORT_DATASET
train = load_official_split(root, "train")
X = build_features_from_context(offline_context(train.points, train.telemetry, train.schedule_plan))
# ... обучить модель на X (колонки ровно FEATURE_NAMES) и target (direct или residual) ...

manifest = ArtifactManifest(
    artifact_schema_version="artifact-manifest-v1",
    model_version="<уникальная версия>", model_family="catboost",
    feature_schema_version="tabular-v1", target_formulation="residual",
    train_regime="<режим>", model_params={...},
    training_data_provenance={"data_version": "<fingerprint>"},
    code_provenance={"git_commit": "<sha>"},
    evaluation=EvaluationSummary(split="labels_test", metric="mae", value=..., n_rows=353),
    created_at=datetime.now(UTC),
)
export_bundle("artifacts/<version>", model, manifest)   # manifest.json + model.cbm + bundle.json
# model — обученный CatBoostRegressor (скалярная регрессия); классификатор/ранкер отклоняются

test = load_official_split(root, "test")
predictor = ArtifactPredictor.load("artifacts/<version>")
pred = predictor.predict(offline_context(test.points, test.telemetry, test.schedule_plan))
```

Затем:

```bash
uv run python scripts/make_submission.py --artifact-dir artifacts/<version> --output submission.csv
MOSTRANSPORT_ARTIFACT_DIR=artifacts/<version> \
  uv run uvicorn mostransport_ml.serving.artifact_app:create_app_from_env --factory
```

Каталоги bundle и submission — вне официального датасета и не коммитятся
вместе с организаторскими данными. `integration-fixture-v1`
(`scripts/build_integration_artifact.py`) — **INTEGRATION TEST ONLY · NOT FOR SUBMISSION · NOT A QUALITY MODEL**: только для
проверки интеграции с Backend, не кандидат. Для Backend
смена модели = смена `MOSTRANSPORT_ARTIFACT_DIR`
([BACKEND_ML_INTEGRATION.md](BACKEND_ML_INTEGRATION.md)).

## Правила modeling-фазы

- `labels_test` — официальный local model-selection split; помнить об
  ограничении temporal nearness (Dataset Evidence v1 §15).
- Validate: только `validate/points.csv`, `validate/traffic.csv`,
  `validate/schedule_plan.csv`; факт test schedule для validate запрещён.
- Новые признаки → новая схема (например `tabular-v2`) рядом с
  `tabular-v1`, не правка `tabular-v1`; bundle обязан указывать схему, под
  которую обучен (совместимость проверяется при загрузке).
- Открытый modeling debt (не инфраструктура; учесть до финальной модели):
  - **`cur_dev_s` vs runtime `current_deviation_seconds`**: в runtime это
    point-in-time-safe отклонение последнего подтверждённо пройденного
    события (иначе `0`); official `cur_dev_s` в real-time один в один не
    воспроизводится (в ~44–45% исследованных real/test случаев
    соответствующее фактическое событие — после `T`). `cur_dev_s` — признак
    `tabular-v1` и основа RESIDUAL, поэтому offline-оценка может быть
    оптимистичнее runtime;
  - **частота telemetry**: CSV ~12–15 с vs эмулятор ~1 Гц → count-признаки
    (`rows_*`, `valid_gps_count_*`) могут быть смещены; telemetry молча не
    прореживать;
  - ~~валидность группы B~~ — решено research: не использовать (клоны группы A);
  - ~~direct vs residual и финальный выбор модели~~ — решено research: HGB H0
    DIRECT на `runtime-safe-v1` (без raw counts, safe deviation), только группа A.

## P2 → P3: production-кандидат HGB

P2 даёт только возможность обслуживать artifact (синтетические fixtures).
P3 (не сделано): official Group A training + OOF reproduction + final bundle.

1. Group A — `load_shared_real_vehicle_ids` (плановые поля test schedule);
   точки, telemetry и план train — `load_official_split(root, "train")`.
2. Safe deviation: факты `train/schedule.csv` в порядке CSV
   (`usecols=data.safe_deviation.SCHEDULE_FACT_COLUMNS`) →
   `safe_current_deviation_seconds(points, facts)`.
3. Признаки: `offline_context(points, telemetry, plan,
   current_deviation_seconds=safe)` → `build_features_from_context` →
   `project_runtime_safe_features` (ровно 29).
4. Воспроизвести research GroupKFold OOF (≈ 77.34 MAE) — acceptance gate.
5. `fit_h0(X, y)` на всей Group A → `ArtifactManifest(model_family=
   "hist_gradient_boosting", feature_schema_version="runtime-safe-v1",
   target_formulation="direct", model_params=dict(HGB_H0_PARAMS), ...)` →
   `export_bundle` → проверка через `ArtifactPredictor`, submission и
   `scripts/smoke_contract_v1.py --expected-feature-schema-version runtime-safe-v1`.

При доменной неоднозначности, не закрытой официальным ТЗ или Dataset
Evidence v1, AI-агент обязан остановиться и спросить, а не угадывать
([`../AGENTS.md`](../AGENTS.md)).
