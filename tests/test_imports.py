"""Каждый модуль должен чисто импортироваться, включая ещё не реализованные
пакеты-заглушки, зарезервированные под будущую работу Valeria."""

import importlib

import pytest

MODULES = [
    "mostransport_ml",
    "mostransport_ml.data",
    "mostransport_ml.data.inspection",
    "mostransport_ml.data.manifest",
    "mostransport_ml.data.canonical",
    "mostransport_ml.target",
    "mostransport_ml.target.spec",
    "mostransport_ml.evaluation",
    "mostransport_ml.evaluation.metrics",
    "mostransport_ml.evaluation.temporal",
    "mostransport_ml.evaluation.baseline",
    "mostransport_ml.experiments",
    "mostransport_ml.experiments.log",
    "mostransport_ml.features",
    "mostransport_ml.models",
    "mostransport_ml.artifacts",
    "mostransport_ml.serving",
    "mostransport_ml.data.official",
    "mostransport_ml.features.builder",
    "mostransport_ml.features.context",
    "mostransport_ml.features.adapters",
    "mostransport_ml.artifacts.manifest",
    "mostransport_ml.artifacts.bundle",
    "mostransport_ml.inference",
    "mostransport_ml.inference.model_families",
    "mostransport_ml.inference.predictor",
    "mostransport_ml.inference.submission",
    "mostransport_ml.serving.schemas",
    "mostransport_ml.serving.service",
    "mostransport_ml.serving.app",
    "mostransport_ml.serving.mock_app",
    "mostransport_ml.serving.artifact_app",
]


@pytest.mark.parametrize("module_name", MODULES)
def test_module_imports(module_name: str) -> None:
    importlib.import_module(module_name)
