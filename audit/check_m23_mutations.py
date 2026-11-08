"""Replay named M2.3 mutations on disposable copies, with JUnit evidence.

Run with the project's Python. No source is changed in the working tree.
Compilation, collection and fixture errors are INCONCLUSIVE, never kills.
"""

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET


MUTATIONS = [
    ("seed_ignored", "random_state=self.random_state,", "random_state=0,"),
    ("pca_off_by_one", "X[:, :n_pca_components]", "X[:, :n_pca_components-1]"),
    (
        "umap_overrides_ignored",
        "{**self._UMAP_DEFAULTS, **self.extra_kwargs}",
        "dict(self._UMAP_DEFAULTS)",
    ),
    (
        "tsne_overrides_ignored",
        "{**self._TSNE_DEFAULTS, **self.extra_kwargs}",
        "dict(self._TSNE_DEFAULTS)",
    ),
    ("umap_default_neighbors", '"n_neighbors": 15,', '"n_neighbors": 14,'),
    ("tsne_default_perplexity", '"perplexity": 30,', '"perplexity": 29,'),
    (
        "fit_time_zero",
        "self.fit_time_ = time.perf_counter() - t0",
        "self.fit_time_ = 0.0",
    ),
    ("cache_wrong_embedding", "self._embedding = Z", "self._embedding = Z[::-1]"),
    (
        "transform_refits",
        "return self._model.transform(X_new)",
        "return self._model.fit_transform(X_new)",
    ),
    ("stability_cap_4999", "min(len(X), 5000)", "min(len(X), 4999)"),
    (
        "stability_head_sampling",
        "idx = rng.choice(len(X), size=n, replace=False)",
        "idx = np.arange(n)",
    ),
    ("stability_seed_ignored", "random_state=seed,", "random_state=self.random_state,"),
    ("stability_reference_last", "Z_ref = embeddings[0]", "Z_ref = embeddings[-1]"),
    ("stability_missing_seed", "range(n_seeds)", "range(n_seeds - 1)"),
    ("stability_false_zero", "_, _, dist = procrustes(Z_ref, Z)", "dist = 0.0"),
    ("alignment_failure_false_zero", "dist = np.nan", "dist = 0.0"),
    ("negative_mutates_input", "X_perm = X.copy()", "X_perm = X"),
    (
        "negative_row_shuffle",
        "rng.shuffle(X_perm[:, j])",
        "rng.shuffle(X_perm, axis=0)",
    ),
    ("negative_permutation_seed_fixed", "self.random_state + 999", "999"),
    (
        "negative_ignores_pca_count",
        "engine_neg.fit_transform(X_perm, n_pca_components=n_pca_components)",
        "engine_neg.fit_transform(X_perm)",
    ),
    (
        "sensitivity_override_reversed",
        "{**self.extra_kwargs, param_name: val}",
        "{param_name: val, **self.extra_kwargs}",
    ),
    ("3d_wrong_axis", "[valid, 2]", "[valid, 1]"),
    ("3d_threshold_excludes_100", "if valid.sum() < 100:", "if valid.sum() <= 100:"),
    (
        "3d_signed_ranking",
        "if abs(r3) > abs(axis3_best_corr):",
        "if r3 > axis3_best_corr:",
    ),
    (
        "3d_missing_parameter_mask",
        "valid = np.isfinite(vals)",
        "valid = np.ones(len(vals), dtype=bool)",
    ),
    (
        "3d_plot_seed_fixed",
        "np.random.default_rng(random_state)",
        "np.random.default_rng(0)",
    ),
    ("3d_wrong_class_rows", '"Classe": y,', '"Classe": np.roll(y, 1),'),
    (
        "3d_wrong_metadata_rows",
        "df_3d[label] = meta[col].values",
        "df_3d[label] = np.roll(meta[col].values, 1)",
    ),
    (
        "3d_wrong_cluster_labels",
        "for c in cluster_labels]",
        "for c in np.roll(cluster_labels, 1)]",
    ),
    ("3d_tsne_perplexity_ignored", "perplexity=perplexity,", "perplexity=5,"),
]

# Neutral/compatibility controls are outside mutation counts, not approved fixes.
CONTROLS = [
    (
        "comment_only",
        "# Référence = premier embedding",
        "# First embedding is the reference",
    ),
    (
        "complete_core_metadata",
        "self._params_used = params",
        "self._params_used = {**params, 'random_state': self.random_state, 'n_components': self.n_components}",
    ),
    (
        "permit_jobs_override",
        "n_jobs=-1,\n            **params,",
        "**{'n_jobs': -1, **params},",
    ),
    (
        "skip_constant_correlations",
        "valid = np.isfinite(vals)",
        "valid = np.isfinite(vals)\n            if np.unique(vals[valid]).size < 2:\n                continue",
    ),
    (
        "reject_zero_plot_size",
        "N_PLOT = min(n_subsample, len(df_3d))",
        "if n_subsample == 0:\n            raise ValueError('Empty plot requested')\n        N_PLOT = min(n_subsample, len(df_3d))",
    ),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--reverse", action="store_true", help="Run baseline in reverse test order only"
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    paths = [
        "src/dimred/embedding.py",
        "src/dimred/pca_analyzer.py",
        "src/dimred/dimred_visualizer.py",
        "src/dimred/run_reporter.py",
        "tests/conftest.py",
        "tests/test_dimred_embedding.py",
        "pyproject.toml",
    ]
    hashes = {p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in paths}
    original = (root / paths[0]).read_text(encoding="utf-8")
    report = {"source_hashes": hashes, "python": sys.version, "cases": []}
    scenarios = [("baseline", None, None, "baseline")]
    if not args.reverse:
        scenarios += [(*m, "mutation") for m in MUTATIONS]
        scenarios += [(*m, "control") for m in CONTROLS]
    try:
        with tempfile.TemporaryDirectory(prefix="m23-mutation-") as directory:
            sandbox = Path(directory)
            for p in paths:
                target = sandbox / p
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(root / p, target)
            if args.reverse:
                with (sandbox / "tests/conftest.py").open(
                    "a", encoding="utf-8"
                ) as stream:
                    stream.write(
                        "\n\ndef pytest_collection_modifyitems(items):\n    items.reverse()\n"
                    )
            for name, old, new, kind in scenarios:
                source = original
                if old is not None:
                    if old not in original or old == new:
                        raise RuntimeError(f"Missing or neutral replacement: {name}")
                    source = original.replace(old, new)
                ast.parse(source)
                (sandbox / paths[0]).write_text(source, encoding="utf-8")
                junit = sandbox / f"{name}.xml"
                env = dict(os.environ, PYTHONHASHSEED="7", MPLBACKEND="Agg")
                command = [
                    sys.executable,
                    "-m",
                    "pytest",
                    "-q",
                    "tests/test_dimred_embedding.py",
                    "-W",
                    "error",
                    "--tb=short",
                    f"--junitxml={junit}",
                ]
                result = subprocess.run(
                    command,
                    cwd=sandbox,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=60,
                    env=env,
                )
                suite = (
                    ET.parse(junit).getroot().find("testsuite")
                    if junit.exists()
                    else None
                )
                failures = (
                    []
                    if suite is None
                    else [
                        c.get("name")
                        for c in suite.findall("testcase")
                        if c.find("failure") is not None
                    ]
                )
                invalid = (
                    result.returncode not in (0, 1)
                    or suite is None
                    or int(suite.get("errors", 0))
                    or int(suite.get("skipped", 0))
                )
                status = (
                    "INCONCLUSIVE"
                    if invalid
                    else (
                        ("PASS" if not failures else "FAIL")
                        if kind != "mutation"
                        else ("KILLED" if failures else "SURVIVED")
                    )
                )
                case = dict(
                    name=name,
                    kind=kind,
                    status=status,
                    tests=None if suite is None else int(suite.get("tests")),
                    failures=failures,
                )
                if invalid:
                    case["diagnostic"] = result.stdout + result.stderr
                report["cases"].append(case)
                print(f"{name}: {status} ({len(failures)} failing cases)", flush=True)
                if kind == "baseline" and status != "PASS":
                    raise RuntimeError("Baseline failed")
                if (
                    suite is not None
                    and int(suite.get("tests")) != report["cases"][0]["tests"]
                ):
                    raise RuntimeError(f"Changed case count: {name}")
    finally:
        assert hashes == {
            p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in paths
        }, "Working-tree input changed"
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if any(c["status"] in ("INCONCLUSIVE", "FAIL") for c in report["cases"]):
        raise RuntimeError("Replay includes inconclusive executions or failed controls")


if __name__ == "__main__":
    main()
