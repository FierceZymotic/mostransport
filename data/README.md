# data/

Local, untracked working area for organizer-provided datasets.

- `raw/` — files exactly as received from the organizers. Never edit in place.
- `interim/` — intermediate outputs of inspection/canonicalization, kept only
  for local convenience. Safe to delete and regenerate.
- `processed/` — canonicalized / split datasets ready for evaluation or
  training. Safe to delete and regenerate from `raw/`.

## Rules

- **Nothing under `raw/`, `interim/`, or `processed/` is tracked by Git**
  (see `.gitignore`). Only this README and the `.gitkeep` markers are.
- Organizer datasets must not be committed, published, or otherwise leave
  the boundaries the hackathon rules allow.
- Before sending any organizer data to an external service, tool, or AI
  assistant, check the hackathon's data-usage rules first.
- Treat everything in this tree as confidential by default.

See `docs/ARCHITECTURE.md` for how this fits into the offline pipeline.
