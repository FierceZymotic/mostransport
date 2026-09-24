# experiments/

Append-only experiment log, written by `mostransport_ml.experiments.log`.

Generated log files (e.g. `runs.jsonl`) are gitignored — they are local
scratch history, not a source-of-truth artifact store. If a result needs to
survive across machines or be shared with the team, copy the relevant line(s)
out rather than committing the whole log.

Each line is a JSON object with at least:

- `run_id`, `created_at`
- `data_version`, `target_version`, `feature_version`
- `model_name`, `params`
- `validation_mae`
- `notes`

See `mostransport_ml.experiments.log.ExperimentLogger` for the writer, and
`docs/ARCHITECTURE.md` for how this fits into the workflow.
