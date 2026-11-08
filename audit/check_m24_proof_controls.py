"""Check M2.4 proof oracles against admissible and partial local candidates.

Run at the proof archive SHA. Modifies only source copies inside TemporaryDirectory.
This is validation of assertions, not approved production fixes.
"""

import argparse
import ast
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

from check_m24_mutations import digest, git, mutate


def state_candidate(policy):
    action = {
        "retain": "self.__dict__.clear(); self.__dict__.update(snapshot)",
        "invalidate": "clean = type(self)(self.min_cluster_size, self.min_samples, self.metric, self.cluster_selection_method); self.__dict__.clear(); self.__dict__.update(clean.__dict__)",
        "partial": "self.labels_ = None; self.labels_pres_ = None",
    }[policy]
    return f"""
def _audit_guard(method):
    def wrapped(self, *args, **kwargs):
        snapshot = self.__dict__.copy()
        try:
            return method(self, *args, **kwargs)
        except Exception:
            {action}
            raise
    return wrapped
HDBSCANAnalyzer.fit = _audit_guard(HDBSCANAnalyzer.fit)
HDBSCANAnalyzer.fit_presentation = _audit_guard(HDBSCANAnalyzer.fit_presentation)
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    paths = [
        "src/dimred/hdbscan_analyzer.py",
        "src/dimred/data_loader.py",
        "src/dimred/pca_analyzer.py",
        "tests/conftest.py",
        "tests/m24_support.py",
        "tests/test_dimred_hdbscan_analyzer.py",
        "tests/proofs_dimred_hdbscan_analyzer.py",
        "pyproject.toml",
        "audit/check_m24_mutations.py",
        "audit/check_m24_proof_controls.py",
    ]
    hashes = {p: digest(root / p) for p in paths}
    index = git(root, "ls-files", "--stage", "--", *paths)
    original = (root / paths[0]).read_text(encoding="utf-8")
    candidates = [
        (
            f"state_{policy}",
            original + state_candidate(policy),
            ["test_failed_fit_state_is_coherent"],
            4 if policy == "partial" else 0,
        )
        for policy in ["retain", "invalidate", "partial"]
    ]
    definitions = [
        (
            "alignment_refusal",
            "compute_feature_profiles",
            "n_umap = len(Z_umap)",
            'n_umap = len(Z_umap)\n    if len(df_feat)!=n_umap or len(cluster_labels)!=n_umap:\n        raise ValueError("Observation alignment mismatch")',
            [
                "test_feature_filter_cannot_reassign_labels",
                "test_feature_labels_length_checked_independently",
            ],
        ),
        (
            "missing_physics_refusal",
            "physical_profile_table",
            "col_names = [c for c, _ in available]",
            'col_names = [c for c, _ in available]\n        if not col_names:\n            raise ValueError("No physical columns available")',
            ["test_no_available_physical_columns_has_explicit_outcome"],
        ),
        (
            "missing_class_placeholder",
            "physical_profile_table",
            ".apply(lambda x: x.value_counts().index[0])",
            '.apply(lambda x: x.value_counts().index[0] if len(x.value_counts()) else "?")',
            ["test_all_missing_classes_have_explicit_outcome"],
        ),
    ]
    for name, scope, old, new, tests in definitions:
        source, _ = mutate(original, (name, scope, old, new))
        candidates.append((name, source, tests, 0))
    report = dict(
        base_sha=git(root, "rev-parse", "HEAD"), content_hashes_lf=hashes, cases=[]
    )
    try:
        with tempfile.TemporaryDirectory(prefix="m24-proof-oracles-") as directory:
            sandbox = Path(directory)
            for p in paths:
                dest = sandbox / p
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(root / p, dest)
            for name, source, tests, expected in candidates:
                ast.parse(source)
                (sandbox / paths[0]).write_text(source, encoding="utf-8")
                selections = ["tests/test_dimred_hdbscan_analyzer.py"] + [
                    f"tests/proofs_dimred_hdbscan_analyzer.py::{test}" for test in tests
                ]
                junit = sandbox / "result.xml"
                if junit.exists():
                    junit.unlink()
                command = [
                    sys.executable,
                    "-m",
                    "pytest",
                    "-q",
                    *selections,
                    "-W",
                    "error",
                    "--tb=short",
                    f"--junitxml={junit}",
                ]
                run = subprocess.run(
                    command,
                    cwd=sandbox,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=60,
                    env=dict(os.environ, PYTHONHASHSEED="7", MPLBACKEND="Agg"),
                )
                suite = ET.parse(junit).getroot().find("testsuite")
                failures = [
                    c.get("name")
                    for c in suite.findall("testcase")
                    if c.find("failure") is not None
                ]
                green_failures = [
                    c.get("name")
                    for c in suite.findall("testcase")
                    if c.find("failure") is not None
                    and "test_dimred_hdbscan_analyzer" in c.get("classname", "")
                ]
                valid = (
                    run.returncode in (0, 1)
                    and int(suite.get("errors")) == int(suite.get("skipped")) == 0
                    and len(failures) == expected
                    and not green_failures
                )
                report["cases"].append(
                    dict(
                        name=name,
                        expected_proof_failures=expected,
                        tests=int(suite.get("tests")),
                        failures=failures,
                        status="PASS" if valid else "FAIL",
                    )
                )
                if not valid:
                    report["cases"][-1]["diagnostic"] = run.stdout + run.stderr
                print(name, report["cases"][-1]["status"], flush=True)
    finally:
        report["normalized_input_integrity"] = hashes == {
            p: digest(root / p) for p in paths
        }
        report["tracked_index_unchanged"] = index == git(
            root, "ls-files", "--stage", "--", *paths
        )
        report["temporary_copies_removed"] = True
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    assert report["normalized_input_integrity"] and report["tracked_index_unchanged"]
    assert all(c["status"] == "PASS" for c in report["cases"])


if __name__ == "__main__":
    main()
