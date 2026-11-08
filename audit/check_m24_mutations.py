"""Replay scoped M2.4 single-site candidates in disposable copies.

Use the project Python. Real production and the tracked index are hash guarded.
Collection, syntax, fixture errors and differing test counts are INCONCLUSIVE.
Known defects are not made green-suite obligations; compatible remedies are
separate controls. --reverse runs only the intact suite in reverse/hashseed7.
"""

import argparse
import ast
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET


# (name, owning function, exact single text replacement, replacement)
MUTATIONS = [
    (
        "default_size",
        "__init__",
        "min_cluster_size: int = 75",
        "min_cluster_size: int = 74",
    ),
    ("default_samples", "__init__", "min_samples: int = 20", "min_samples: int = 19"),
    (
        "fit_size_ignored",
        "fit",
        "min_cluster_size=self.min_cluster_size",
        "min_cluster_size=75",
    ),
    ("fit_samples_ignored", "fit", "min_samples=self.min_samples", "min_samples=20"),
    ("fit_metric_ignored", "fit", "metric=self.metric", 'metric="euclidean"'),
    (
        "fit_selection_ignored",
        "fit",
        "cluster_selection_method=self.cluster_selection_method",
        'cluster_selection_method="eom"',
    ),
    (
        "fit_prediction_data",
        "fit",
        "            prediction_data=True,",
        "            prediction_data=False,",
    ),
    (
        "fit_reverses_labels",
        "fit",
        "self.clusterer_.fit_predict(Z)",
        "self.clusterer_.fit_predict(Z)[::-1]",
    ),
    (
        "presentation_size_ignored",
        "fit_presentation",
        "min_cluster_size=min_cluster_size",
        "min_cluster_size=300",
    ),
    (
        "presentation_samples_ignored",
        "fit_presentation",
        "min_samples=self.min_samples",
        "min_samples=20",
    ),
    (
        "presentation_metric_ignored",
        "fit_presentation",
        "metric=self.metric",
        'metric="euclidean"',
    ),
    (
        "presentation_selection_ignored",
        "fit_presentation",
        "cluster_selection_method=self.cluster_selection_method",
        'cluster_selection_method="eom"',
    ),
    (
        "presentation_reverses_labels",
        "fit_presentation",
        "self.clusterer_pres_.fit_predict(Z)",
        "self.clusterer_pres_.fit_predict(Z)[::-1]",
    ),
    (
        "presentation_includes_noise",
        "fit_presentation",
        "set(self.labels_pres_) - {-1}",
        "set(self.labels_pres_)",
    ),
    (
        "presentation_count_plus_one",
        "fit_presentation",
        "len(self.ids_pres_)",
        "len(self.ids_pres_)+1",
    ),
    (
        "presentation_wrong_palette_ids",
        "fit_presentation",
        "enumerate(self.ids_pres_)",
        "enumerate(self.cluster_ids_)",
    ),
    (
        "stats_includes_noise",
        "_compute_stats",
        "set(self.labels_) - {-1}",
        "set(self.labels_)",
    ),
    (
        "stats_count_plus_one",
        "_compute_stats",
        "len(self.cluster_ids_)",
        "len(self.cluster_ids_)+1",
    ),
    ("stats_noise_wrong", "_compute_stats", "self.labels_ == -1", "self.labels_ == 0"),
    (
        "stats_loses_noncontiguous",
        "_compute_stats",
        "sorted(set(self.labels_) - {-1})",
        "list(range(len(set(self.labels_) - {-1})))",
    ),
    ("palette_wrong_id", "_build_palette", "cid: cmap", "i: cmap"),
    (
        "palette_swapped_ranks",
        "_build_palette",
        "enumerate(self.cluster_ids_)",
        "enumerate(self.cluster_ids_[::-1])",
    ),
    ("centroid_ignores_mask", "centroid", "Z[mask].mean(axis=0)", "Z.mean(axis=0)"),
    ("centroid_wrong_axis", "centroid", "mean(axis=0)", "mean(axis=1)"),
    (
        "profile_shift_labels",
        "physical_profile_table",
        'df_profile["cluster"] = self.labels_',
        'df_profile["cluster"] = np.roll(self.labels_,1)',
    ),
    (
        "profile_shift_classes",
        "physical_profile_table",
        'df_profile["class_lamost"] = y',
        'df_profile["class_lamost"] = np.roll(y,1)',
    ),
    (
        "profile_mean_std_swap",
        "physical_profile_table",
        'agg(["mean", "std"])',
        'agg(["std", "mean"]).rename(columns={"std":"mean","mean":"std"},level=1)',
    ),
    (
        "profile_population_ddof",
        "physical_profile_table",
        'agg(["mean", "std"])',
        'agg(["mean", lambda x: x.std(ddof=0)]).rename(columns={"<lambda_0>":"std"},level=1)',
    ),
    (
        "profile_wrong_count",
        "physical_profile_table",
        '"n spectres": int(counts.get(cid, 0)),',
        '"n spectres": int(counts.get(cid, 0))+1,',
    ),
    (
        "profile_wrong_cluster_mean",
        "physical_profile_table",
        "mu = summary.loc[cid, m_col]",
        "mu = summary.loc[summary.index[0], m_col]",
    ),
    (
        "profile_noise_wrong_mean",
        "physical_profile_table",
        "vals = df_noise[col].dropna()",
        "vals = df_profile[col].dropna()",
    ),
    (
        "sensitivity_default_grid",
        "compute_sensitivity",
        "[50, 100, 150, 300, 500]",
        "[50, 100, 150, 300, 499]",
    ),
    (
        "sensitivity_default_samples",
        "compute_sensitivity",
        "min_samples: int = 20",
        "min_samples: int = 19",
    ),
    (
        "sensitivity_reverse_grid",
        "compute_sensitivity",
        "for mcs in min_sizes:",
        "for mcs in min_sizes[::-1]:",
    ),
    (
        "sensitivity_loses_duplicates",
        "compute_sensitivity",
        "for mcs in min_sizes:",
        "for mcs in dict.fromkeys(min_sizes):",
    ),
    (
        "sensitivity_samples_ignored",
        "compute_sensitivity",
        "min_samples=min_samples",
        "min_samples=20",
    ),
    (
        "sensitivity_includes_noise",
        "compute_sensitivity",
        "len(set(cl)) - (1 if -1 in cl else 0)",
        "len(set(cl))",
    ),
    ("sensitivity_noise_wrong", "compute_sensitivity", "cl == -1", "cl == 0"),
    (
        "sensitivity_percent",
        "compute_sensitivity",
        "100 * nn / len(cl)",
        "nn / len(cl)",
    ),
    (
        "sensitivity_wrong_rounding",
        "compute_sensitivity",
        "round(100 * nn / len(cl), 2)",
        "round(100 * nn / len(cl), 1)",
    ),
    (
        "feature_snr_strict",
        "compute_feature_profiles",
        'df_raw["snr_r"] >= 10.0',
        'df_raw["snr_r"] > 10.0',
    ),
    ("feature_no_exclusions", "compute_feature_profiles", "c not in EXCLUDE", "True"),
    (
        "feature_no_gaia_exclusions",
        "compute_feature_profiles",
        "set(_GAIA_META_COLS) |",
        "set() |",
    ),
    (
        "feature_no_instrument_exclusions",
        "compute_feature_profiles",
        "set(_INSTRUMENTAL_COLS)",
        "set()",
    ),
    (
        "feature_no_prefix_exclusions",
        "compute_feature_profiles",
        "not c.startswith(_NON_SPECTRO_PREFIXES)",
        "True",
    ),
    (
        "feature_accept_constants",
        "compute_feature_profiles",
        "df_filt[c].nunique() > 1",
        "df_filt[c].nunique() > 0",
    ),
    (
        "feature_missing_strict",
        "compute_feature_profiles",
        "isna().mean() <= 0.10",
        "isna().mean() < 0.10",
    ),
    (
        "feature_missing_too_permissive",
        "compute_feature_profiles",
        "isna().mean() <= 0.10",
        "isna().mean() <= 0.20",
    ),
    (
        "feature_no_dropna",
        "compute_feature_profiles",
        "df_filt[feat_cols].dropna()",
        "df_filt[feat_cols]",
    ),
    (
        "feature_no_centering",
        "compute_feature_profiles",
        "scaler = StandardScaler()",
        "scaler = StandardScaler(with_mean=False)",
    ),
    (
        "feature_no_scaling",
        "compute_feature_profiles",
        "scaler = StandardScaler()",
        "scaler = StandardScaler(with_std=False)",
    ),
    (
        "feature_reverse_columns",
        "compute_feature_profiles",
        "columns=feat_cols)",
        "columns=feat_cols[::-1])",
    ),
    (
        "feature_shift_cluster_labels",
        "compute_feature_profiles",
        'X_std_df["cluster"] = cluster_labels_aligned',
        'X_std_df["cluster"] = np.roll(cluster_labels_aligned,1)',
    ),
    (
        "feature_includes_noise",
        "compute_feature_profiles",
        'X_std_df[X_std_df["cluster"] != -1]',
        "X_std_df",
    ),
]


def controls():
    return [
        (
            "equivalent_unique",
            "_compute_stats",
            "sorted(set(self.labels_) - {-1})",
            "sorted(c for c in np.unique(self.labels_) if c != -1)",
        ),
        (
            "neutral_comment",
            "centroid",
            "return Z[mask].mean(axis=0)",
            "# Independent neutral control\n        return Z[mask].mean(axis=0)",
        ),
        (
            "reject_feature_mismatch",
            "compute_feature_profiles",
            "n_umap = len(Z_umap)",
            'n_umap = len(Z_umap)\n    if len(df_feat) != n_umap or len(cluster_labels) != n_umap:\n        raise ValueError("Observation alignment mismatch")',
        ),
        (
            "return_standardized_frame",
            "compute_feature_profiles",
            "return df_feat, cluster_means, cluster_labels_aligned",
            "return X_std_df[feat_cols], cluster_means, cluster_labels_aligned",
        ),
        (
            "remove_physical_overclaims",
            "physical_profile_table",
            '"Interprétation": (',
            '"Interprétation": ("Uncalibrated heuristic: " +',
        ),
        (
            "normal_transaction_retention",
            "fit",
            "self.clusterer_ = hdbscan.HDBSCAN(",
            "candidate = hdbscan.HDBSCAN(",
        ),
    ]


def digest(path):
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def git(root, *args):
    return subprocess.check_output(
        ["git", *args], cwd=root, text=True, encoding="utf-8"
    ).strip()


def mutate(source, definition):
    name, owner, old, new = definition
    nodes = [
        n
        for n in ast.walk(ast.parse(source))
        if isinstance(n, ast.FunctionDef) and n.name == owner
    ]
    assert len(nodes) == 1, (name, "ambiguous function")
    node = nodes[0]
    lines = source.splitlines(keepends=True)
    start = sum(map(len, lines[: node.lineno - 1]))
    end = sum(map(len, lines[: node.end_lineno]))
    body = source[start:end]
    assert body.count(old) == 1, (name, "missing/ambiguous site", body.count(old))
    position = start + body.index(old)
    candidate = source[:position] + new + source[position + len(old) :]
    ast.parse(candidate)
    return candidate, dict(
        function=owner,
        line=source.count("\n", 0, position) + 1,
        old=old,
        new=new,
        replacements=1,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reverse", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    paths = [
        "src/dimred/hdbscan_analyzer.py",
        "src/dimred/data_loader.py",
        "src/dimred/pca_analyzer.py",
        "tests/conftest.py",
        "tests/m24_support.py",
        "tests/test_dimred_hdbscan_analyzer.py",
        "pyproject.toml",
        "audit/check_m24_mutations.py",
    ]
    hashes = {p: digest(root / p) for p in paths}
    index = git(root, "ls-files", "--stage", "--", *paths)
    original = (root / paths[0]).read_text(encoding="utf-8")
    scenarios = [dict(name="baseline", kind="baseline", source=original, site=None)]
    if not args.reverse:
        for kind, definitions in [("mutation", MUTATIONS), ("control", controls())]:
            for definition in definitions:
                # Transaction candidate needs two coordinated edits; controls are
                # not single-site mutations and never enter the mutation count.
                if definition[0] == "normal_transaction_retention":
                    candidate, site = mutate(original, definition)
                    candidate = candidate.replace(
                        "self.labels_ = self.clusterer_.fit_predict(Z)",
                        "labels = candidate.fit_predict(Z)\n        self.clusterer_ = candidate\n        self.labels_ = labels",
                    )
                    site["replacements"] = 2
                else:
                    candidate, site = mutate(original, definition)
                scenarios.append(
                    dict(name=definition[0], kind=kind, source=candidate, site=site)
                )
    report = dict(
        schema=1,
        base_sha=git(root, "rev-parse", "HEAD"),
        source_blob=git(root, "hash-object", paths[0]),
        hash_policy="SHA-256, CRLF normalized to LF",
        content_hashes_lf=hashes,
        python=sys.version,
        versions={
            p: importlib.metadata.version(p)
            for p in ["pytest", "numpy", "pandas", "scipy", "scikit-learn"]
        },
        reverse=args.reverse,
        cases=[],
    )
    try:
        with tempfile.TemporaryDirectory(prefix="m24-mutations-") as directory:
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
            for scenario in scenarios:
                (sandbox / paths[0]).write_text(scenario["source"], encoding="utf-8")
                junit = sandbox / "result.xml"
                if junit.exists():
                    junit.unlink()
                command = [
                    sys.executable,
                    "-m",
                    "pytest",
                    "-q",
                    "tests/test_dimred_hdbscan_analyzer.py",
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
                        if scenario["kind"] == "mutation"
                        else ("FAIL" if failures else "PASS")
                    )
                )
                case = {k: scenario[k] for k in ["name", "kind", "site"]}
                case.update(
                    status=status,
                    tests=None if suite is None else int(suite.get("tests")),
                    failures=failures,
                )
                if invalid or status == "FAIL":
                    case["diagnostic"] = result.stdout + result.stderr
                report["cases"].append(case)
                print(
                    f"{scenario['name']}: {status} ({len(failures)} failures)",
                    flush=True,
                )
                if scenario["kind"] == "baseline" and status != "PASS":
                    raise RuntimeError("Intact control failed")
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
    assert not any(c["status"] in ["INCONCLUSIVE", "FAIL"] for c in report["cases"])


if __name__ == "__main__":
    main()
