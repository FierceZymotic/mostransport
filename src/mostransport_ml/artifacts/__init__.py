"""ML artifact bundle — not yet implemented; `metadata.py` is the exception.

`metadata.ArtifactMetadata` is a minimal, stable metadata shape (model
version, target info, optional validation MAE). It is deliberately not an
artifact loader: no `.cbm`/joblib/preprocessing loading exists here, because
the real model format is unknown until a real model exists.

Owned by Valeria. See docs/ARCHITECTURE.md §7 for the expected shape of a
full artifact bundle once a real model pipeline exists.
"""
