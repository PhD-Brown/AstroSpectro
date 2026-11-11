"""Record R11 reference choices without freezing an unresolved policy in CI."""

import importlib.util
import json
from pathlib import Path
import tempfile

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def main():
    spec = importlib.util.spec_from_file_location(
        "m27_reference_probe", ROOT / "src/dimred/run_reporter.py"
    )
    r = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(r)
    y = [" star ", "qs0", "galaxie", "star", "STAR", "QSO"]
    prediction = ["STAR", "STAR", "QSO", "STAR", "GALAXY", "QSO"]
    short = ["M", "F", "G", "M", "F"]
    records = []
    with tempfile.TemporaryDirectory(prefix="m27-reference-synthetic-") as folder:
        root = Path(folder)
        for i, (name, aligned, clusters) in enumerate(
            [
                ("absent", None, None),
                ("short_aligned", short, np.array([7] * 5)),
                ("raw_equal_length", y, np.array([7] * 6)),
            ]
        ):
            r.make_timestamp = lambda i=i: f"20261109T1900{i:02d}Z"
            result = r.save_umap_tsne_run(
                Z_umap=np.array([[1.0, 8], [2, 5], [4, 6], [3, 2], [7, 9], [11, 3]]),
                Z_tsne=None,
                y=y,
                y_aligned=aligned,
                y_pred=prediction,
                cl_aligned=clusters,
                meta=None,
                scores_95=None,
                paths={"REPORTS_DIR": root},
                FIGURES_DIR=root / "figures",
            )
            report = json.loads(result["json_path"].read_text(encoding="utf-8"))
            summary = report["xgboost_summary"]
            # Manual outcomes: normalized y has matches0/3/5; raw y only row5.
            expected = 1 / 6 if name == "raw_equal_length" else 0.5
            assert abs(summary["accuracy_vs_reference"] - expected) < 1e-14
            assert ("top3_pred_by_cluster" in summary) == (name == "raw_equal_length")
            records.append(
                {
                    "case": name,
                    "input_original_y": y,
                    "input_aligned_y": aligned,
                    "input_predictions": prediction,
                    "aligned_cluster_length": (
                        None if clusters is None else len(clusters)
                    ),
                    "inferred_reference_by_manual_oracle": (
                        "raw y_aligned"
                        if name == "raw_equal_length"
                        else "normalized original y"
                    ),
                    "expected_accuracy": expected,
                    "actual_summary": summary,
                    "reference_provenance_fields": [
                        key
                        for key in summary
                        if any(
                            token in key.lower()
                            for token in [
                                "source",
                                "provenance",
                                "reference_source",
                                "aligned",
                            ]
                        )
                    ],
                    "limit": "Source choice inferred from values, not recorded by production; equal lengths do not demonstrate identity.",
                }
            )
    (ROOT / "audit/m27_review_reference_results.json").write_text(
        json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        "3 reference observations, manual numeric/cluster assertions PASS; not green policy contracts"
    )


if __name__ == "__main__":
    main()
