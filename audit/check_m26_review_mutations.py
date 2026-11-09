"""Replay scoped review candidates in temporary copies; never edit production.

Expected red baselines, incomplete remedies and admissible controls are distinct.
Each record declares its exact pytest selection, assertion diagnostics and scope.
"""

import argparse
import ast
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

from check_m26_mutations import TRACE_PLUGIN


TARGET = "src/dimred/xgboost_bridge.py"
PROOF = "tests/proofs_dimred_xgboost_bridge_review.py"
GREEN = [
    "tests/test_dimred_xgboost_bridge.py",
    "tests/test_dimred_xgboost_bridge_real.py",
]
ALIGN = "    if len(df_aligned) != len(Z_umap):"
LENGTH_GUARD = '    if not (len(df_aligned) == len(Z_umap) == len(y) == len(cluster_labels)):\n        raise ValueError("Observation alignment length mismatch")\n'
IDENTITY_GUARD = '    if "id_echo" in df_aligned and not np.array_equal(df_aligned["id_echo"].to_numpy(), Z_umap[:, 0]):\n        raise ValueError("Observation identity mismatch")\n'


def replace(source, changes):
    for old, new in changes:
        assert source.count(old) == 1, (old, source.count(old))
        source = source.replace(old, new, 1)
    ast.parse(source)
    return source


def candidates(phase):
    if phase == "r1":
        selection = [PROOF, "-k", "loader_population"]
        return [
            (
                "baseline",
                [],
                selection,
                3,
                1,
                "red baseline: true loader, unchanged bridge",
            ),
            (
                "length_refusal_only",
                [(ALIGN, LENGTH_GUARD + ALIGN)],
                selection,
                2,
                2,
                "incomplete remedy must still fail equal-size default/NB01 cases",
            ),
            (
                "synthetic_identity_refusal",
                [(ALIGN, IDENTITY_GUARD + ALIGN)],
                selection,
                0,
                4,
                "admissible control only: fixture embedding column0 explicitly stores IDs",
            ),
            (
                "synthetic_shared_population",
                [
                    (
                        "df_aligned = df_f.dropna(subset=pca_feat_cols).reset_index(drop=True)",
                        'df_aligned = df_raw.set_index("obsid").loc[Z_umap[:, 0].astype(int)].reset_index()',
                    )
                ],
                selection,
                0,
                4,
                "admissible synthetic population transport, not proposed production architecture",
            ),
        ]
    raise ValueError(phase)


def git(root, *args):
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


def fingerprint(path):
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def execute(directory, selection):
    for file in (directory / "src/dimred").glob("__pycache__/xgboost_bridge.*.pyc"):
        file.unlink()
    for name in ("trace.json", "result.xml"):
        (directory / name).unlink(missing_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        *selection,
        "-p",
        "m26_trace",
        "--tb=short",
        "--junitxml=result.xml",
        "-W",
        "error",
    ]
    env = os.environ.copy()
    env.update(PYTHONIOENCODING="utf-8", PYTHONHASHSEED="7", MPLBACKEND="Agg")
    result = subprocess.run(
        cmd,
        cwd=directory,
        env=env,
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
            detail = error if error is not None else failure
            cases.append(
                {
                    "name": case.attrib["name"],
                    "outcome": (
                        "ERROR"
                        if error is not None
                        else (
                            "FAIL"
                            if failure is not None
                            else "SKIP" if skipped is not None else "PASS"
                        )
                    ),
                    "detail": detail.text if detail is not None else None,
                }
            )
    return {
        "command": cmd,
        "returncode": result.returncode,
        "cases": cases,
        "executed_bridge_lines": (
            json.loads((directory / "trace.json").read_text())
            if (directory / "trace.json").exists()
            else []
        ),
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["r1"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    paths = [
        TARGET,
        "src/dimred/data_loader.py",
        "src/utils.py",
        "src/pipeline/classifier.py",
        "tests/conftest.py",
        *GREEN,
        PROOF,
        "pyproject.toml",
        "audit/check_m26_mutations.py",
        "audit/check_m26_review_mutations.py",
    ]
    inputs = {
        name: {
            "sha256_lf": fingerprint(root / name),
            "git_blob_at_head": git(root, "rev-parse", f"HEAD:{name}"),
        }
        for name in paths
    }
    head, dirty = git(root, "rev-parse", "HEAD"), git(root, "status", "--porcelain")
    source = (root / TARGET).read_text(encoding="utf-8")
    records = []
    with tempfile.TemporaryDirectory(prefix="astrospectro-m26-review-") as temporary:
        directory = Path(temporary)
        for name in paths:
            if name.startswith("audit/"):
                continue
            destination = directory / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes((root / name).read_bytes())
        (directory / "m26_trace.py").write_text(TRACE_PLUGIN, encoding="utf-8")
        for name, changes, selection, want_fail, want_pass, meaning in candidates(
            args.phase
        ):
            mutated = replace(source, changes)
            (directory / TARGET).write_text(mutated, encoding="utf-8")
            outcome = execute(directory, selection)
            counts = {
                key: sum(c["outcome"] == key for c in outcome["cases"])
                for key in ("FAIL", "PASS", "ERROR", "SKIP")
            }
            assertion_only = all(
                re.search(r"(?m)^E\s+(?:AssertionError|assert\s)", c["detail"] or "")
                for c in outcome["cases"]
                if c["outcome"] == "FAIL"
            )
            matched = counts == {
                "FAIL": want_fail,
                "PASS": want_pass,
                "ERROR": 0,
                "SKIP": 0,
            }
            matched &= (
                outcome["returncode"] == (1 if want_fail else 0) and assertion_only
            )
            records.append(
                {
                    "name": name,
                    "scope": selection,
                    "patches": changes,
                    "meaning": meaning,
                    "expected_fail": want_fail,
                    "expected_pass": want_pass,
                    "counts": counts,
                    "assertion_failures_only": assertion_only,
                    "expectation_met": matched,
                    **outcome,
                }
            )
            print(name, counts, "EXPECTED" if matched else "UNEXPECTED", flush=True)
    assert all(
        fingerprint(root / name) == item["sha256_lf"] for name, item in inputs.items()
    )
    assert git(root, "rev-parse", "HEAD") == head
    payload = {
        "phase": args.phase,
        "git_head_at_run": head,
        "working_tree_at_run": dirty,
        "python": sys.version,
        "versions": {
            p: importlib.metadata.version(p)
            for p in ["numpy", "pandas", "scikit-learn", "xgboost", "pytest"]
        },
        "inputs": inputs,
        "results": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return 0 if all(r["expectation_met"] for r in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
