"""Record invalid-input policies without converting observations into contracts."""

import ast
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import warnings

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src/dimred/run_reporter.py"


def main():
    spec = importlib.util.spec_from_file_location("m27_probes", SOURCE)
    r = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(r)
    results = []

    def record(name, function):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                value = function()
                result = {"name": name, "value": repr(value), "exception": None}
            except Exception as error:
                result = {
                    "name": name,
                    "value": None,
                    "exception": type(error).__name__,
                    "message": str(error),
                }
            result["warnings"] = [str(w.message) for w in caught]
            results.append(result)

    for name, labels in [
        ("mixed_coerced", [1, "F", 1]),
        ("missing", ["F", None]),
        ("object_noncomparable", np.array([1, "F"], dtype=object)),
    ]:
        record("class_counts_" + name, lambda labels=labels: r.class_counts(labels))
    df = pd.DataFrame({"x": [1.0, 4.0, 9.0]}, index=[9, 2, 71])
    for limit in [-1, "two", 0.5]:
        record(f"df_limit_{limit}", lambda limit=limit: r.safe_df_records(df, limit))
    record(
        "df_infinity", lambda: r.safe_df_records(pd.DataFrame({"x": [np.inf, -np.inf]}))
    )
    for dtype in ["float64", "Float64", "str", "string", "Int64", "boolean"]:
        values = (
            [True, None]
            if dtype == "boolean"
            else ["F", None] if dtype in ["str", "string"] else [1, None]
        )
        record(
            "df_missing_" + dtype,
            lambda dtype=dtype, values=values: r.safe_df_records(
                pd.DataFrame({"x": pd.array(values, dtype=dtype)})
            ),
        )
    for name, Z in [
        ("all_nan_axis", [[np.nan, 1.0], [np.nan, 3.0]]),
        ("infinity", [[1.0, 2.0], [np.inf, 3.0]]),
        ("zero_rows", np.empty((0, 2))),
        ("zero_columns", np.empty((3, 0))),
        ("one_dimension", [1.0, 2.0]),
        ("nonnumeric", [["F", "G"], ["M", "K"]]),
    ]:
        for function in [r.embedding_stats, r.embedding_stats_nd]:
            record(
                function.__name__ + "_" + name,
                lambda Z=Z, function=function: function(Z),
            )
    for axes in [-1, 0, 1, 0.5, "two"]:
        record(
            f"max_axes_{axes}",
            lambda axes=axes: r.embedding_stats_nd([[1.0, 2.0], [4.0, 8.0]], axes),
        )
    record("shape_conversion_failure", lambda: r.shape_or_none([[1], [2, 3]]))
    record(
        "numeric_means_nonfinite",
        lambda: r.numeric_means(
            pd.DataFrame({"nan": [np.nan, np.nan], "inf": [np.inf, 2.0]})
        ),
    )
    record("cluster_noninteger", lambda: r.cluster_top_counts([0.1, 0.9]))
    record("cluster_string", lambda: r.cluster_top_counts(["A", "B"]))
    record("cluster_negative_top_n", lambda: r.cluster_top_counts([1, 2, 2, 3], -1))
    record(
        "prediction_missing",
        lambda: r.top3_pred_by_cluster([1, 1, 1], ["F", None, "F"]),
    )
    with tempfile.TemporaryDirectory(prefix="m27-synthetic-probes-") as folder:
        root = Path(folder)
        figures = root / "figures"
        figures.mkdir()
        stale = figures / "old_run.png"
        stale.write_bytes(b"synthetic previous run")
        r.make_timestamp = lambda: "20261109T170304Z"
        result = r.save_umap_tsne_run(
            Z_umap=np.array([[1.0, 3.0], [4.0, 2.0], [7.0, 9.0]]),
            Z_tsne=None,
            y=["F", "G", "M"],
            meta=None,
            scores_95=None,
            paths={"REPORTS_DIR": root / "reports"},
            FIGURES_DIR=figures,
        )
        report = json.loads(result["json_path"].read_text(encoding="utf-8"))
        results.append(
            {
                "name": "foreign_figure_inventory",
                "old_figure_included": str(stale) in report["artifacts"]["png_files"],
                "artifact_fields": list(report["artifacts"]),
                "top_level_report_fields": list(report),
            }
        )
        blocker = root / "not_a_directory"
        blocker.write_text("synthetic", encoding="utf-8")
        record("run_directory_file_collision", lambda: r.make_run_dir(blocker, "nb"))

        class AE:
            history_ = {"train_loss": [4.0, 2.0, 3.0], "val_loss": [3.0, 1.0, 2.0]}

            def encode(self, X):
                return X[:, :2]

            def reconstruction_mse(self, X):
                return 1 / 9

        X = np.array([[1.0, 6], [2, 5], [4, 3], [3, 4], [5, 2], [6, 1]])
        base = dict(
            ae=AE(),
            X=X,
            y=["F", "G", "M", "F", "G", "F"],
            meta=pd.DataFrame({"teff_gspphot": [10.0, 20, 30, 40, 50, 60]}),
            feature_names=["a", "b"],
            FIGURES_DIR=figures,
            paths={"REPORTS_DIR": root / "ae_reports"},
        )
        cases = [
            ("constant_parameter", {"meta": pd.DataFrame({"teff_gspphot": [1.0] * 6})}),
            ("constant_latent", {"Z_ae": np.ones((6, 2))}),
            (
                "two_valid_parameters",
                {
                    "meta": pd.DataFrame(
                        {"teff_gspphot": [1.0, 2, np.nan, np.nan, np.nan, np.nan]}
                    )
                },
            ),
            (
                "two_finite_pairs",
                {
                    "Z_ae": np.array(
                        [
                            [1.0, 6],
                            [2, 5],
                            [np.nan, 4],
                            [np.nan, 3],
                            [np.nan, 2],
                            [np.nan, 1],
                        ]
                    )
                },
            ),
            (
                "nonfinite_history",
                {"history": {"train_loss": [4.0, np.inf], "val_loss": [np.nan, 1.0]}},
            ),
            (
                "array_history",
                {
                    "history": {
                        "train_loss": np.array([4.0, 2.0, 3.0]),
                        "val_loss": np.array([3.0, 1.0, 2.0]),
                    }
                },
            ),
            ("short_latent", {"Z_ae": X[:3]}),
            ("one_latent_axis", {"Z_ae": X[:, :1]}),
        ]
        for i, (name, overrides) in enumerate(cases):
            r.make_timestamp = lambda i=i: f"20261109T1704{i:02d}Z"

            def summarize(overrides=overrides):
                result = r.save_autoencoder_run(**dict(base, **overrides))
                report = json.loads(result["json_path"].read_text(encoding="utf-8"))
                return {
                    "metrics": report["metrics"],
                    "correlations": report["correlations"],
                }

            record("ae_" + name, summarize)

    source = SOURCE.read_text(encoding="utf-8")
    inventory = [
        {"name": node.name, "line": node.lineno, "signature": ast.unparse(node.args)}
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef)
    ]
    ledger = {
        "schema": 1,
        "python": sys.version,
        "versions": {
            name: importlib.metadata.version(name)
            for name in ["numpy", "pandas", "scipy", "scikit-learn", "joblib", "pytest"]
        },
        "inventory": inventory,
        "probes": results,
        "limit": "Observations of current invalid-input policies, not green acceptance requirements; only synthetic temporary data.",
    }
    (ROOT / "audit/m27_boundary_results.json").write_text(
        json.dumps(ledger, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"{len(inventory)} functions; {len(results)} boundary observations recorded")


if __name__ == "__main__":
    main()
