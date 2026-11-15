"""Observe unsupported/degenerate visualization boundaries without freezing policy."""

import ast
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import traceback
import types
import warnings

os.environ["MPLBACKEND"] = "Agg"
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    path = ROOT / f"src/dimred/{name}.py"
    spec = importlib.util.spec_from_file_location(f"dimred.{name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[f"dimred.{name}"] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="test-results/m28-boundaries.json")
    args = parser.parse_args()
    source = ROOT / "src/dimred/dimred_visualizer.py"
    package = types.ModuleType("dimred")
    package.__path__ = [str(source.parent)]
    sys.modules["dimred"] = package
    module = load("dimred_visualizer")
    viz = module.DimRedVisualizer(dpi=25)
    methods = next(
        n
        for n in ast.parse(source.read_text(encoding="utf-8")).body
        if isinstance(n, ast.ClassDef)
    )
    inventory = [
        {
            "name": n.name,
            "line": n.lineno,
            "signature": ast.unparse(n.args),
            "returns": [
                ast.unparse(r.value) if r.value is not None else "None"
                for r in ast.walk(n)
                if isinstance(r, ast.Return)
            ],
        }
        for n in methods.body
        if isinstance(n, ast.FunctionDef)
    ]
    Z = np.random.default_rng(1729).normal(size=(12, 2))
    y = np.array(["STAR"] * 12)
    meta = pd.DataFrame(
        {
            "teff_gspphot": np.linspace(3000, 8000, 12),
            "logg_gspphot": np.linspace(1, 5, 12),
        }
    )
    jobs = []

    def add(label, function):
        jobs.append((label, function))

    for label, points, labels, frame in [
        ("no_stars", Z, np.array(["QSO"] * 12), meta),
        ("singleton", Z[:1], y[:1], meta.iloc[:1]),
        ("two_points", Z[:2], y[:2], meta.iloc[:2]),
        ("collinear", np.column_stack([np.arange(12.0), np.arange(12.0)]), y, meta),
        (
            "NaN_coordinate",
            np.where(np.arange(24).reshape(12, 2) == 4, np.nan, Z),
            y,
            meta,
        ),
        (
            "Inf_coordinate",
            np.where(np.arange(24).reshape(12, 2) == 4, np.inf, Z),
            y,
            meta,
        ),
        ("no_physical", Z, y, pd.DataFrame(index=range(12))),
        ("all_missing_physical", Z, y, pd.DataFrame({"teff_gspphot": [np.nan] * 12})),
    ]:
        add(
            "KDE_" + label,
            lambda points=points, labels=labels, frame=frame: viz.plot_latent_kde(
                points, labels, frame
            ),
        )
    for name in [
        "plot_embedding",
        "plot_embedding_grid",
        "plot_ae_embedding",
        "plot_latent_zoomed",
        "plot_synthesis_grid",
    ]:

        def call(name=name):
            missing = pd.DataFrame({"teff_gspphot": [np.nan] * 12})
            return {
                "plot_embedding": lambda: viz.plot_embedding(
                    Z, color_by=np.full(12, np.nan)
                ),
                "plot_embedding_grid": lambda: viz.plot_embedding_grid(Z, y, missing),
                "plot_ae_embedding": lambda: viz.plot_ae_embedding(Z, missing, y),
                "plot_latent_zoomed": lambda: viz.plot_latent_zoomed(Z, y, missing),
                "plot_synthesis_grid": lambda: viz.plot_synthesis_grid(
                    Z, y, missing, Z
                ),
            }[name]()

        add(name + "_all_missing", call)
    for label, frame in [
        ("empty", meta.iloc[:0]),
        ("all_missing", meta * float("nan")),
        ("singleton", meta.iloc[:1]),
    ]:
        add("HR_" + label, lambda frame=frame: viz.plot_hr_teff_regions(frame))
    for label, errors in [
        ("all_zero", np.zeros(12)),
        ("negative", np.arange(12.0) - 2),
        ("NaN", np.where(np.arange(12) == 2, np.nan, 0.1 + np.arange(12))),
        ("Inf", np.where(np.arange(12) == 2, np.inf, 0.1 + np.arange(12))),
        ("tiny_positive", np.geomspace(1e-10, 1e-6, 12)),
    ]:
        add(
            "log_MSE_" + label,
            lambda errors=errors: viz.plot_ae_error_distribution_logscale(errors, y),
        )
    for n in [0, -1, 20]:
        add(
            f"reconstruction_n{n}",
            lambda n=n: viz.plot_ae_reconstruction(Z, Z + 1, ["a", "b"], n_samples=n),
        )
    for k in [0, -1, 1, 6, 12]:
        add(f"trust_k{k}", lambda k=k: viz.plot_trustworthiness(Z, Z, Z, k_values=[k]))
    for count in [0, 1, 7]:
        frame = pd.DataFrame(
            {
                "seed": np.arange(count),
                "procrustes_distance": np.linspace(0, 0.6, count),
                "method": ["UMAP"] * count,
            }
        )
        add(f"stability_rows{count}", lambda frame=frame: viz.plot_stability(frame))
    for label, history in [
        ("empty", {"train_loss": [], "val_loss": []}),
        ("mismatched", {"train_loss": [1, 2], "val_loss": [1]}),
        ("NaN", {"train_loss": [1, 2], "val_loss": [np.nan, 1]}),
    ]:
        add(
            "history_" + label,
            lambda history=history: viz.plot_training_history(history),
        )
    add("embedding_length", lambda: viz.plot_embedding(Z, y=y[:5]))
    add(
        "family_zero_weights",
        lambda: viz.plot_loadings_family_bar(
            pd.DataFrame({"family": ["A"], "weight": [0.0]}),
            pd.DataFrame({"family": ["B"], "weight": [0.0]}),
            {},
        ),
    )
    add(
        "family_donut_zero",
        lambda: viz.plot_loadings_family_donut(
            pd.DataFrame({"family": ["A"], "weight": [0.0]}),
            pd.DataFrame({"family": ["B"], "weight": [0.0]}),
            {},
        ),
    )
    pca = types.SimpleNamespace(
        loadings=np.array([[0.1, -0.4, 0.9, -0.2]]),
        explained_variance_ratio=np.array([1.0]),
    )
    add(
        "eigenspectra_default_lines",
        lambda: viz.plot_eigenspectra(
            pca, np.array([3800.0, 4400.0, 5500.0, 6800.0]), n_components=1
        ),
    )
    for label, confidence in [
        ("negative", np.full(12, -0.1)),
        ("above_one", np.full(12, 1.2)),
        ("NaN", np.full(12, np.nan)),
        ("Inf", np.full(12, np.inf)),
        ("wrong_length", np.arange(5.0) / 5),
    ]:
        add(
            "confidence_" + label,
            lambda confidence=confidence: viz.plot_xgboost_umap(
                Z,
                np.array(["F"] * 12),
                confidence,
                np.zeros(12, dtype=int),
                {0: "blue"},
            ),
        )
    observations = []
    for label, function in jobs:
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            try:
                result = function()
                outcome = {
                    "result_type": type(result).__name__,
                    "figures": [
                        {
                            "axes": [
                                {
                                    "title": a.get_title(),
                                    "xlim": list(a.get_xlim()),
                                    "ylim": list(a.get_ylim()),
                                    "lines": len(a.lines),
                                    "collections": len(a.collections),
                                }
                                for a in plt.figure(i).axes
                            ]
                        }
                        for i in plt.get_fignums()
                    ],
                }
            except Exception as exc:
                outcome = {
                    "exception": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                }
            observations.append(
                {
                    "case": label,
                    **outcome,
                    "warnings": [str(w.message) for w in captured],
                }
            )
            plt.close("all")
    consumers = []
    for path in (ROOT / "notebooks/dimred").glob("*.ipynb"):
        notebook = json.loads(path.read_text(encoding="utf-8"))
        for index, cell in enumerate(notebook["cells"]):
            if cell.get("cell_type") != "code":
                continue
            content = "".join(cell["source"])
            try:
                tree = ast.parse(content)
            except SyntaxError:
                consumers.append(
                    {"notebook": path.name, "cell_index": index, "unparsed_cell": True}
                )
                continue
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr.startswith("plot_")
                ):
                    consumers.append(
                        {
                            "notebook": path.name,
                            "cell_index": index,
                            "line": node.lineno,
                            "call": ast.unparse(node),
                        }
                    )
    result = {
        "python": sys.version,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "backend": matplotlib.get_backend(),
        "inventory": inventory,
        "observations": observations,
        "static_notebook_calls": consumers,
    }
    destination = ROOT / args.output
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(result, indent=2, ensure_ascii=True, default=float) + "\n",
        encoding="utf-8",
    )
    print(
        f"{len(inventory)} methods; {len(observations)} boundary observations; {destination}"
    )


if __name__ == "__main__":
    main()
