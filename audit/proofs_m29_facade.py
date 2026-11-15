"""Explicit red contracts, outside default testpaths. No xfail or bug blessing."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from facade_probe import assert_success, run_probe  # noqa: E402


@pytest.mark.parametrize(
    "statement",
    [
        "import dimred",
        "from dimred import PCAAnalyzer",
        "from dimred import SpectralAutoencoder",
        "from dimred.pca_analyzer import PCAAnalyzer",
        "from dimred.embedding import EmbeddingEngine",
        "from dimred.data_loader import DimRedDataLoader",
    ],
)
def test_independent_entry_survives_missing_torch(tmp_path, statement):
    result = run_probe(tmp_path, "entry", blocked=("torch",), operation=statement)
    if statement == "from dimred import SpectralAutoencoder" and result.returncode:
        # Explicit actionable optional-feature refusal is an admissible alternative.
        assert "ImportError:" in result.stderr and "torch" in result.stderr.lower()
        assert "NameError:" not in result.stderr, result.stderr
    else:
        assert_success(result)


def test_plain_import_preserves_callers_matplotlib_style(tmp_path):
    assert_success(run_probe(tmp_path, "style"))
