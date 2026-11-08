"""Replay named M2.3 mutations on disposable copies, with JUnit evidence.

Run with the project's Python. No source is changed in the working tree.
Compilation, collection and fixture errors are INCONCLUSIVE, never kills.
"""

import argparse
import ast
import hashlib
import json
import os
import re
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


# Specifications below refer to AST function scopes and zero-based occurrences.
# Legacy definitions above are expanded: each occurrence becomes its own scenario.
def scenario(
    name, old, new, scope=None, occurrence=None, kind="mutation", group="review"
):
    return dict(
        name=name,
        old=old,
        new=new,
        scope=scope,
        occurrence=occurrence,
        kind=kind,
        group=group,
    )


def review_scenarios():
    result = []
    for method, prefix in [("umap", "UMAP"), ("tsne", "tSNE")]:
        scope = f"compute_{method}3d"
        result += [
            scenario(
                f"R1_{method}_kendall",
                "from scipy.stats import spearmanr",
                "from scipy.stats import kendalltau as spearmanr",
                scope,
            ),
            scenario(
                f"R1_{method}_first_finite",
                "if abs(r3) > abs(axis3_best_corr):",
                "if axis3_best_param == '?' and np.isfinite(r3):",
                scope,
            ),
            scenario(
                f"R3_{method}_class_z_second_axis",
                f'z="{prefix}3",',
                f'z="{prefix}2",',
                scope,
                0,
            ),
            scenario(
                f"R3_{method}_temperature_color_class",
                'color="T_eff (K)",',
                'color="Classe",',
                scope,
            ),
            scenario(
                f"R3_{method}_temperature_full_population",
                'df_plot.dropna(subset=["T_eff (K)"])',
                'df_3d.dropna(subset=["T_eff (K)"])',
                scope,
            ),
        ]
    result += [
        scenario(
            "R2_first_column_only",
            "for j in range(X_perm.shape[1]):",
            "for j in range(min(1, X_perm.shape[1])):",
            "negative_control",
        ),
        scenario(
            "R4_stability_child_2d",
            "n_components=self.n_components,",
            "n_components=2,",
            "stability_report",
        ),
        scenario(
            "R5_force_neighbors_key",
            "param_name = list(param_grid.keys())[0]",
            "param_name = 'n_neighbors' if self.method == 'umap' else list(param_grid.keys())[0]",
            "parameter_sensitivity",
        ),
        scenario(
            "R3_umap_class_color_cluster",
            'color="Classe",',
            'color="Cluster",',
            "compute_umap3d",
        ),
        scenario(
            "R3_umap_metallicity_range_temperature",
            'float(df_plot["[Fe/H]"].quantile(0.02)),',
            'float(df_plot["T_eff (K)"].quantile(0.02)),',
            "compute_umap3d",
        ),
        scenario(
            "R3_umap_metallicity_color_temperature",
            'color="[Fe/H]",',
            'color="T_eff (K)",',
            "compute_umap3d",
        ),
    ]
    assert len(result) == 16
    return result


def control_scenarios():
    result = [
        scenario(
            "R6a_reject_oversized_pca",
            "if n_pca_components is not None:",
            "if n_pca_components is not None and n_pca_components > X.shape[1]:\n            raise ValueError('PCA count exceeds input width')\n        if n_pca_components is not None:",
            "fit_transform",
        ),
        scenario("R6b_remove_forced_jobs", "            n_jobs=-1,\n", "", "_fit_tsne"),
        scenario(
            "R6c_raise_alignment_domain_error",
            "dist = np.nan",
            "raise",
            "stability_report",
        ),
        scenario(
            "complete_core_metadata",
            "self._params_used = params",
            "self._params_used = {**params, 'random_state': self.random_state, 'n_components': self.n_components}",
        ),
        scenario(
            "permit_jobs_override",
            "n_jobs=-1,\n            **params,",
            "**{'n_jobs': -1, **params},",
            "_fit_tsne",
        ),
        scenario(
            "skip_constant_correlations",
            "valid = np.isfinite(vals)",
            "valid = np.isfinite(vals)\n            if np.unique(vals[valid]).size < 2:\n                continue",
        ),
        scenario(
            "reject_zero_plot_size",
            "N_PLOT = min(n_subsample, len(df_3d))",
            "if n_subsample == 0:\n            raise ValueError('Empty plot requested')\n        N_PLOT = min(n_subsample, len(df_3d))",
        ),
        scenario(
            "comment_only",
            "# R\u00e9f\u00e9rence = premier embedding",
            "# First embedding is the reference",
            "stability_report",
        ),
    ]
    for item in result:
        item.update(kind="compatibility_control", group="controls")
    result[-1]["kind"] = "neutral_control"
    result += [
        scenario(
            "procrustes_arguments_reversed",
            "procrustes(Z_ref, Z)",
            "procrustes(Z, Z_ref)",
            "stability_report",
            kind="equivalent_control",
            group="controls",
        ),
        scenario(
            "last_column_unshuffled",
            "range(X_perm.shape[1])",
            "range(X_perm.shape[1] - 1)",
            "negative_control",
            kind="equivalent_control",
            group="controls",
        ),
        scenario(
            "stability_sample_sorted",
            "idx = rng.choice(len(X), size=n, replace=False)",
            "idx = np.sort(rng.choice(len(X), size=n, replace=False))",
            "stability_report",
            kind="equivalent_control",
            group="controls",
        ),
        scenario(
            "stability_sample_permutation",
            "idx = rng.choice(len(X), size=n, replace=False)",
            "idx = rng.permutation(len(X))[:n]",
            "stability_report",
            kind="sampling_control",
            group="controls",
        ),
    ]
    for method in ("umap", "tsne"):
        scope = f"compute_{method}3d"
        result += [
            scenario(
                f"{method}_plot_sample_sorted",
                "rng.choice(len(df_3d), N_PLOT, replace=False)",
                "np.sort(rng.choice(len(df_3d), N_PLOT, replace=False))",
                scope,
                kind="sampling_control",
                group="controls",
            ),
            scenario(
                f"{method}_plot_sample_permutation",
                "df_plot = df_3d.iloc[rng.choice(len(df_3d), N_PLOT, replace=False)].copy()",
                "if N_PLOT < 0:\n            raise ValueError('Negative subsample size')\n        df_plot = df_3d.iloc[rng.permutation(len(df_3d))[:N_PLOT]].copy()",
                scope,
                kind="sampling_control",
                group="controls",
            ),
        ]
    return result


def locate_sites(source, item):
    lines = source.splitlines(keepends=True)
    start, end = 0, len(source)
    if item["scope"]:
        functions = [
            n
            for n in ast.walk(ast.parse(source))
            if isinstance(n, ast.FunctionDef) and n.name == item["scope"]
        ]
        if len(functions) != 1:
            raise ValueError(f"Ambiguous scope: {item['scope']}")
        node = functions[0]
        start = sum(map(len, lines[: node.lineno - 1]))
        end = sum(map(len, lines[: node.end_lineno]))
    matches = [
        start + m.start()
        for m in re.finditer(re.escape(item["old"]), source[start:end])
    ]
    if not matches or item["old"] == item["new"]:
        raise ValueError(f"Missing or neutral replacement: {item['name']}")
    occurrence = item["occurrence"]
    if occurrence is not None:
        if occurrence < 0 or occurrence >= len(matches):
            raise ValueError(f"Invalid occurrence: {item['name']}")
        return [(occurrence, matches[occurrence])]
    return list(enumerate(matches))


def normalized_hash(path):
    # Text content integrity across LF/CRLF checkouts; raw bytes are not compared.
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def git(root, *args):
    return subprocess.check_output(
        ["git", *args], cwd=root, text=True, encoding="utf-8"
    ).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reverse", action="store_true")
    parser.add_argument(
        "--tests-ref", help="Optional historical Git ref for the test file only"
    )
    parser.add_argument(
        "--group", choices=["all", "legacy", "review", "controls"], default="all"
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
        "audit/check_m23_mutations.py",
    ]
    hashes = {p: normalized_hash(root / p) for p in paths}
    index_before = git(root, "ls-files", "--stage", "--", *paths)
    original = (root / paths[0]).read_text(encoding="utf-8")
    definitions = (
        [scenario(*m, group="legacy") for m in MUTATIONS]
        + review_scenarios()
        + control_scenarios()
    )
    scenarios = [
        dict(
            name="baseline",
            kind="baseline",
            group="baseline",
            source=original,
            site=None,
        )
    ]
    if not args.reverse:
        for item in definitions:
            if args.group != "all" and item["group"] != args.group:
                continue
            for occurrence, pos in locate_sites(original, item):
                mutated = (
                    original[:pos] + item["new"] + original[pos + len(item["old"]) :]
                )
                assert mutated != original
                # Exact reconstruction proves there is one targeted text edit.
                assert mutated[:pos] == original[:pos]
                assert (
                    mutated[pos + len(item["new"]) :]
                    == original[pos + len(item["old"]) :]
                )
                line = original.count("\n", 0, pos) + 1
                owner = [
                    n.name
                    for n in ast.walk(ast.parse(original))
                    if isinstance(n, ast.FunctionDef)
                    and n.lineno <= line <= n.end_lineno
                ]
                site = dict(
                    scope=item["scope"],
                    function=owner[0] if owner else "module",
                    occurrence=occurrence,
                    line=line,
                    old=item["old"],
                    new=item["new"],
                    replacements=1,
                )
                scenarios.append(
                    dict(
                        name=f"{item['name']}__{site['function']}__occ{occurrence}",
                        kind=item["kind"],
                        group=item["group"],
                        source=mutated,
                        site=site,
                    )
                )
    report = {
        "schema": 2,
        "git_head": git(root, "rev-parse", "HEAD"),
        "hash_policy": "SHA-256 of UTF-8 text bytes with CRLF normalized to LF",
        "content_hashes_lf": hashes,
        "head_git_blobs": {p: git(root, "rev-parse", f"HEAD:{p}") for p in paths},
        "working_tree_before": git(
            root, "status", "--short", "--untracked-files=normal"
        ),
        "python": sys.version,
        "group": args.group,
        "cases": [],
    }
    try:
        with tempfile.TemporaryDirectory(prefix="m23-single-site-") as directory:
            sandbox = Path(directory)
            for p in paths:
                target = sandbox / p
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(root / p, target)
            if args.tests_ref:
                historical = subprocess.check_output(
                    ["git", "show", f"{args.tests_ref}:tests/test_dimred_embedding.py"],
                    cwd=root,
                )
                (sandbox / "tests/test_dimred_embedding.py").write_bytes(historical)
                report["tests_ref"] = args.tests_ref
                report["historical_test_sha256_lf"] = hashlib.sha256(
                    historical.replace(b"\r\n", b"\n")
                ).hexdigest()
            if args.reverse:
                with (sandbox / "tests/conftest.py").open(
                    "a", encoding="utf-8"
                ) as stream:
                    stream.write(
                        "\n\ndef pytest_collection_modifyitems(items):\n    items.reverse()\n"
                    )
            for item in scenarios:
                case = {k: item[k] for k in ("name", "kind", "group", "site")}
                try:
                    ast.parse(item["source"])
                    (sandbox / paths[0]).write_text(item["source"], encoding="utf-8")
                    junit = sandbox / "result.xml"
                    if junit.exists():
                        junit.unlink()
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
                        env=dict(os.environ, PYTHONHASHSEED="7", MPLBACKEND="Agg"),
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
                    if (
                        suite is not None
                        and report["cases"]
                        and int(suite.get("tests")) != report["cases"][0]["tests"]
                    ):
                        invalid = True
                    status = (
                        "INCONCLUSIVE"
                        if invalid
                        else (
                            ("KILLED" if failures else "SURVIVED")
                            if item["kind"] == "mutation"
                            else ("FAIL" if failures else "PASS")
                        )
                    )
                    case.update(
                        status=status,
                        tests=None if suite is None else int(suite.get("tests")),
                        failures=failures,
                    )
                    if invalid:
                        case["diagnostic"] = result.stdout + result.stderr
                except (SyntaxError, subprocess.TimeoutExpired, ET.ParseError) as error:
                    case.update(
                        status="INCONCLUSIVE",
                        tests=None,
                        failures=[],
                        diagnostic=str(error),
                    )
                report["cases"].append(case)
                print(
                    f"{case['name']}: {case['status']} ({len(case['failures'])} failing cases)",
                    flush=True,
                )
                if item["kind"] == "baseline" and case["status"] != "PASS":
                    raise RuntimeError("Baseline failed")
    finally:
        report["normalized_input_integrity"] = hashes == {
            p: normalized_hash(root / p) for p in paths
        }
        report["tracked_index_unchanged"] = index_before == git(
            root, "ls-files", "--stage", "--", *paths
        )
        report["temporary_copies_removed"] = True
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if (
        not report["normalized_input_integrity"]
        or not report["tracked_index_unchanged"]
    ):
        raise RuntimeError("Protected real inputs or tracked index changed")
    if any(c["status"] in ("INCONCLUSIVE", "FAIL") for c in report["cases"]):
        raise RuntimeError("Replay includes inconclusive execution or failed controls")


if __name__ == "__main__":
    main()
