"""Explicit red contracts, outside default testpaths. No xfail or bug blessing."""

import importlib.util
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from facade_probe import (
    TORCH_STATEMENTS,
    assert_success,
    assert_torch_contract,
    run_probe,
    probe_record,
)  # noqa: E402


@pytest.mark.parametrize(
    "statement",
    TORCH_STATEMENTS,
)
def test_independent_entry_survives_missing_torch(tmp_path, statement):
    result = run_probe(
        tmp_path, "torch_boundary", blocked=("torch",), operation=statement
    )
    assert_torch_contract(result)


def test_plain_import_preserves_callers_matplotlib_style(tmp_path):
    if importlib.util.find_spec("torch") is None:
        pytest.skip(
            "style oracle requires real Torch; missing extension is #21, not style"
        )
    result = run_probe(tmp_path, "style")
    measured = probe_record(
        result, "M29_STYLE"
    )  # Missing measurement cannot count as style evidence.
    assert measured["import_failure"] is None
    assert measured["changed"] == [], measured
    assert_success(result)
