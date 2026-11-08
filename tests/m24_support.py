"""Local helpers for M2.4 tests; no optional facade or real catalog I/O."""

import importlib.util
from pathlib import Path
import sys
import types

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def hdbscan_module():
    path = Path(__file__).resolve().parents[1] / "src/dimred/hdbscan_analyzer.py"
    spec = importlib.util.spec_from_file_location("m24_hdbscan", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def hdbscan_backend(monkeypatch):
    """Queue labels/errors at the backend boundary, without validating HDBSCAN."""
    instances, outcomes = [], []

    class Backend:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            instances.append(self)

        def fit_predict(self, Z):
            self.input = np.array(Z, copy=True)
            outcome = outcomes.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            self.labels_ = np.array(outcome, dtype=int)
            return self.labels_

    monkeypatch.setitem(sys.modules, "hdbscan", types.SimpleNamespace(HDBSCAN=Backend))
    return instances, outcomes


@pytest.fixture
def hdbscan_csv(tmp_path, monkeypatch, dimred_loader_module):
    # Bypass only the known optional facade (#21); use real loader constants.
    package = types.ModuleType("dimred")
    package.__path__ = [str(Path(__file__).resolve().parents[1] / "src/dimred")]
    monkeypatch.setitem(sys.modules, "dimred", package)
    monkeypatch.setitem(sys.modules, "dimred.data_loader", dimred_loader_module)
    counter = 0

    def write(data):
        nonlocal counter
        stem = f"m24_{counter}"
        counter += 1
        path = tmp_path / f"{stem}.csv"
        pd.DataFrame(data).to_csv(path, index=False)
        return {"PROCESSED_DIR": tmp_path}, stem, path

    return write
