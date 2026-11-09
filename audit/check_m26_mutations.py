"""Run M2.6 semantic mutations in disposable copies, never in the worktree.

Record actual executed source lines, assertion diagnostics, exact patches,
positive baseline, input hashes, HEAD and dirty state. Known defective paths
are not silently turned into green contracts. Run with the project Python.

Scope clarification after R4: a full positive baseline is followed by the
designated test function for each mutant (all tests only when test=None).
SURVIVED means survival of that recorded selection, not necessarily the suite.
The historical JSON files are immutable; the review runner separately records
full-suite71-case campaigns and the precise narrower proof selections.
"""

import argparse
import ast
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET


TARGET = "src/dimred/xgboost_bridge.py"
UNIT = "tests/test_dimred_xgboost_bridge.py"
REAL = "tests/test_dimred_xgboost_bridge_real.py"
NOMINAL = "test_nominal_named_values_decode_confidence_and_integrity"
SCHEMA = "X_pred_df = df_aligned[needed].fillna(0)"

# name, old, new, target test function, interpretation
MUTANTS = [
    ("columns_reversed", SCHEMA, SCHEMA + ".iloc[:, ::-1]", NOMINAL, "schema"),
    (
        "rows_reversed",
        SCHEMA,
        SCHEMA + ".iloc[::-1].reset_index(drop=True)",
        NOMINAL,
        "identity",
    ),
    (
        "y_shifted",
        "        y_aligned = y\n",
        "        y_aligned = np.roll(y, 1)\n",
        NOMINAL,
        "identity",
    ),
    (
        "clusters_shifted",
        "        cl_aligned = cluster_labels\n",
        "        cl_aligned = np.roll(cluster_labels, 1)\n",
        NOMINAL,
        "identity",
    ),
    (
        "embedding_shifted",
        "        Z_umap_aligned = Z_umap\n",
        "        Z_umap_aligned = np.roll(Z_umap, 1, axis=0)\n",
        NOMINAL,
        "identity",
    ),
    (
        "snr_strict",
        'df_raw["snr_r"] >= 10.0',
        'df_raw["snr_r"] > 10.0',
        "test_snr_inclusive_boundary_and_absent_snr",
        "boundary",
    ),
    (
        "snr_lowered",
        'df_raw["snr_r"] >= 10.0',
        'df_raw["snr_r"] >= 9.0',
        "test_snr_inclusive_boundary_and_absent_snr",
        "boundary",
    ),
    (
        "nan_strict",
        "df_f[c].isna().mean() <= 0.10",
        "df_f[c].isna().mean() < 0.10",
        "test_candidate_nan_fraction_boundary",
        "boundary",
    ),
    (
        "constant_admitted",
        "and df_f[c].nunique() > 1",
        "and True",
        "test_candidate_variance_and_dtype_control_row_selection",
        "selection",
    ),
    (
        "nonnumeric_admitted",
        "and pd.api.types.is_numeric_dtype(df_f[c])",
        "and True",
        "test_candidate_variance_and_dtype_control_row_selection",
        "selection",
    ),
    (
        "metadata_admitted",
        "if c not in _EXCL_COLS",
        "if True",
        "test_excluded_metadata_nan_cannot_remove_rows",
        "selection",
    ),
    (
        "missing_cutoff_strict",
        "if pct > 0.30:",
        "if pct >= 0.30:",
        "test_missing_feature_fraction_characterization",
        "current technical policy, not statistical approval",
    ),
    (
        "missing_column_one",
        "df_aligned[f] = 0.0",
        "df_aligned[f] = 1.0",
        "test_missing_feature_fraction_characterization",
        "current technical policy, not statistical approval",
    ),
    (
        "decode_ignored",
        "y_pred = clf.label_encoder.inverse_transform(y_pred_enc)",
        "y_pred = np.array(y_pred_enc, dtype=str)",
        "test_decoding_observed_classes_and_fg_mask",
        "decode",
    ),
    (
        "probability_rows_reversed",
        "proba = clf.model_pipeline.predict_proba(X_pred_df)",
        "proba = clf.model_pipeline.predict_proba(X_pred_df)[::-1]",
        "test_probability_row_maximum_does_not_override_decision_rule",
        "association",
    ),
    (
        "probability_axis_zero",
        "confidence = proba.max(axis=1)",
        "confidence = proba.max(axis=0)",
        "test_probability_row_maximum_does_not_override_decision_rule",
        "reduction",
    ),
    (
        "probability_minimum",
        "confidence = proba.max(axis=1)",
        "confidence = proba.min(axis=1)",
        "test_probability_row_maximum_does_not_override_decision_rule",
        "reduction",
    ),
    (
        "fg_only_f",
        'np.isin(y_pred, ["F", "G"])',
        'np.isin(y_pred, ["F"])',
        NOMINAL,
        "mask",
    ),
    (
        "classes_trained_not_observed",
        "classes_pred = sorted(set(y_pred))",
        "classes_pred = clf.class_labels",
        "test_decoding_observed_classes_and_fg_mask",
        "observed classes",
    ),
    (
        "empty_success",
        '"_xgb_ok": False',
        '"_xgb_ok": True',
        "test_empty_structure_is_fresh",
        "failure status",
    ),
    (
        "model_lexical_not_mtime",
        'latest_file(paths["MODELS_DIR"], "spectral_classifier*.pkl")',
        'max(Path(paths["MODELS_DIR"]).glob("spectral_classifier*.pkl"), default=None)',
        "test_model_selection_uses_actual_utils_mtime",
        "artifact choice",
    ),
    (
        "csv_first_not_last",
        "features_csv = files[-1]",
        "features_csv = files[0]",
        "test_explicit_csv_wins_and_fallback_is_lexical",
        "current fallback characterization",
    ),
    (
        "success_flag_false",
        '"_xgb_ok": True',
        '"_xgb_ok": False',
        NOMINAL,
        "result status",
    ),
    (
        "missing_proba_flag_true",
        "has_proba = False",
        "has_proba = True",
        "test_probability_unavailable_has_explicit_flag",
        "availability",
    ),
    (
        "fillna_seven",
        SCHEMA,
        "X_pred_df = df_aligned[needed].fillna(7)",
        NOMINAL,
        "expected gap: finite nominal input, unresolved imputation #89",
    ),
]

CONTROLS = [
    ("copy_named_input", SCHEMA, SCHEMA + ".copy()", NOMINAL, "equivalent"),
    (
        "numpy_row_max",
        "confidence = proba.max(axis=1)",
        "confidence = np.max(proba, axis=1)",
        "test_probability_row_maximum_does_not_override_decision_rule",
        "equivalent",
    ),
    (
        "unique_classes",
        "classes_pred = sorted(set(y_pred))",
        "classes_pred = list(np.unique(y_pred))",
        "test_decoding_observed_classes_and_fg_mask",
        "equivalent",
    ),
    (
        "preserve_pipeline_nan",
        SCHEMA,
        "X_pred_df = df_aligned[needed].copy()",
        None,
        "admissible partial remedy for #89, not a production fix",
    ),
    (
        "refuse_mismatched_population",
        "    if len(df_aligned) != len(Z_umap):",
        '    if not (len(df_aligned) == len(Z_umap) == len(y) == len(cluster_labels)):\n        raise ValueError("Observation alignment length mismatch")\n    if len(df_aligned) != len(Z_umap):',
        None,
        "admissible #88 refusal, no ID architecture chosen",
    ),
]

TRACE_PLUGIN = """import json, sys
from pathlib import Path
import pytest
hits = set()
target = str((Path.cwd() / "src/dimred/xgboost_bridge.py").resolve())
def trace(frame, event, arg):
    if frame.f_code.co_filename == target:
        if event == "line":
            hits.add(frame.f_lineno)
        return trace
    return None
@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    previous = sys.gettrace()
    sys.settrace(trace)
    try:
        yield
    finally:
        sys.settrace(previous)
def pytest_sessionfinish(session, exitstatus):
    Path("trace.json").write_text(json.dumps(sorted(hits)), encoding="utf-8")
"""


def git(root, *args):
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


def digest(path):
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def run_tests(directory, test=None):
    selected = [UNIT, REAL] if test is None else [f"{UNIT}::{test}"]
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        *selected,
        "-p",
        "m26_trace",
        "--tb=short",
        "--junitxml=result.xml",
        "-W",
        "error",
    ]
    environment = os.environ.copy()
    environment["PYTHONHASHSEED"] = "7"
    environment["PYTHONIOENCODING"] = "utf-8"
    environment["MPLBACKEND"] = "Agg"
    result = subprocess.run(
        command,
        cwd=directory,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    cases = []
    if (directory / "result.xml").exists():
        for case in ET.parse(directory / "result.xml").iter("testcase"):
            failure, error, skipped = (
                case.find("failure"),
                case.find("error"),
                case.find("skipped"),
            )
            cases.append(
                {
                    "name": case.attrib["name"],
                    "status": (
                        "ERROR"
                        if error is not None
                        else (
                            "FAIL"
                            if failure is not None
                            else "SKIP" if skipped is not None else "PASS"
                        )
                    ),
                    "diagnostic": (
                        error.text
                        if error is not None
                        else failure.text if failure is not None else None
                    ),
                }
            )
    return {
        "command": command,
        "returncode": result.returncode,
        "cases": cases,
        "executed_lines": (
            json.loads((directory / "trace.json").read_text())
            if (directory / "trace.json").exists()
            else []
        ),
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("test-results/m26-mutations.json")
    )
    parser.add_argument(
        "--only",
        nargs="+",
        help="Replay named cases without repeating the full campaign.",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    paths = [
        TARGET,
        "src/utils.py",
        "src/pipeline/classifier.py",
        "tests/conftest.py",
        UNIT,
        REAL,
        "pyproject.toml",
        "audit/check_m26_mutations.py",
    ]
    inputs = {
        name: {
            "sha256_lf": digest(root / name),
            "git_blob_at_head": git(root, "rev-parse", f"HEAD:{name}"),
        }
        for name in paths
    }
    head, status = git(root, "rev-parse", "HEAD"), git(root, "status", "--porcelain")
    source = (root / TARGET).read_text(encoding="utf-8")
    records = []
    with tempfile.TemporaryDirectory(prefix="astrospectro-m26-mutations-") as temporary:
        directory = Path(temporary)
        for name in paths:
            if name.startswith("audit/"):
                continue
            destination = directory / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(root / name, destination)
        (directory / "m26_trace.py").write_text(TRACE_PLUGIN, encoding="utf-8")
        baseline = run_tests(directory)
        assert baseline["returncode"] == 0 and baseline["cases"]
        assert all(c["status"] == "PASS" for c in baseline["cases"])
        for group, candidates in [("mutation", MUTANTS), ("control", CONTROLS)]:
            for name, old, new, test, interpretation in candidates:
                if args.only and name not in args.only:
                    continue
                assert source.count(old) == 1, (name, source.count(old))
                mutated = source.replace(old, new, 1)
                ast.parse(mutated)
                changed_start = source[: source.index(old)].count("\n") + 1
                changed_lines = list(
                    range(changed_start, changed_start + new.count("\n") + 1)
                )
                (directory / TARGET).write_text(mutated, encoding="utf-8")
                # A previous pyc must never substitute same-size, same-second edits.
                for cache in (directory / "src/dimred").glob(
                    "__pycache__/xgboost_bridge.*.pyc"
                ):
                    cache.unlink()
                for artifact in ["result.xml", "trace.json"]:
                    (directory / artifact).unlink(missing_ok=True)
                outcome = run_tests(directory, test)
                expected = [
                    c["name"]
                    for c in baseline["cases"]
                    if test is None or c["name"].split("[")[0] == test
                ]
                actual = [c["name"] for c in outcome["cases"]]
                relevant = sorted(set(changed_lines) & set(outcome["executed_lines"]))
                failures = [c for c in outcome["cases"] if c["status"] == "FAIL"]
                if actual != expected or any(
                    c["status"] in {"ERROR", "SKIP"} for c in outcome["cases"]
                ):
                    verdict = "INCONCLUSIVE"
                elif not relevant:
                    verdict = "NOT_EXERCISED"
                elif outcome["returncode"] == 0:
                    verdict = "PASS" if group == "control" else "SURVIVED"
                elif failures and all(
                    re.search(
                        r"(?m)^E\s+(?:AssertionError|assert\s)", c["diagnostic"] or ""
                    )
                    for c in failures
                ):
                    verdict = (
                        "REJECTED_BY_ASSERTION"
                        if group == "control"
                        else "KILLED_BY_ASSERTION"
                    )
                else:
                    verdict = "RUNTIME_FAILURE"
                records.append(
                    {
                        "name": name,
                        "group": group,
                        "old": old,
                        "new": new,
                        "test": test,
                        "interpretation": interpretation,
                        "changed_lines": changed_lines,
                        "changed_lines_executed": relevant,
                        "verdict": verdict,
                        **outcome,
                    }
                )
                print(f"{name}: {verdict}", flush=True)
    after = {name: digest(root / name) for name in paths}
    assert all(after[name] == item["sha256_lf"] for name, item in inputs.items())
    assert git(root, "rev-parse", "HEAD") == head
    payload = {
        "git_head_at_run": head,
        "working_tree_at_run": status,
        "git_head_after": git(root, "rev-parse", "HEAD"),
        "working_tree_after": git(root, "status", "--porcelain"),
        "python": sys.version,
        "versions": {
            k: importlib.metadata.version(k)
            for k in ["numpy", "pandas", "scikit-learn", "xgboost", "pytest"]
        },
        "inputs": inputs,
        "selected_cases": args.only,
        "baseline": baseline,
        "results": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                v: sum(r["verdict"] == v for r in records)
                for v in sorted({r["verdict"] for r in records})
            }
        )
    )
    # A documented survivor is evidence, not a reason to hide the output.
    return (
        1
        if any(
            r["verdict"]
            in {
                "INCONCLUSIVE",
                "RUNTIME_FAILURE",
                "NOT_EXERCISED",
                "REJECTED_BY_ASSERTION",
            }
            for r in records
        )
        else 0
    )


if __name__ == "__main__":
    raise SystemExit(main())
