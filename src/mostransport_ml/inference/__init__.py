"""Artifact-backed inference, общий для runtime serving и offline validate/submission.

- `predictor.ArtifactPredictor` — Artifact Bundle v1 → общий Feature Builder
  `tabular-v1` → признаки схемы из manifest (`tabular-v1` | `runtime-safe-v1`)
  → модель → итоговая задержка (direct/residual по manifest);
- `predictor.export_bundle` — обученная модель + ArtifactManifest → bundle;
- `model_families` — поддерживаемые model family (`catboost`,
  `hist_gradient_boosting`);
- `submission` — validate inference и запись `sample_id;prediction`.

Зависит от `artifacts`, `features`, `target`, `data`; не зависит от `serving`.
"""
