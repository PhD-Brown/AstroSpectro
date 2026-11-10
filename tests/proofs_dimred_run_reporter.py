"""Explicit red proofs; archive by SHA, then remove from the green head."""

import json
import builtins
import math

import joblib
import numpy as np
import pandas as pd
import pytest


@pytest.mark.parametrize(
    "dtype,values",
    [
        ("float64", [1.5, np.nan]),
        ("str", ["étoile", None]),
    ],
)
def test_dataframe_missing_values_are_strict_json(reporter_module, dtype, values):
    df = pd.DataFrame({"value": pd.array(values, dtype=dtype)})
    original = df.copy(deep=True)
    result = reporter_module.safe_df_records(df)
    pd.testing.assert_frame_equal(df, original)
    assert result[0]["value"] == values[0]
    assert result[1]["value"] is None, "Promised missing->None conversion was lost"
    assert json.loads(json.dumps(result, allow_nan=False)) == result


@pytest.mark.parametrize(
    "value", [np.nan, np.inf, -np.inf, {"nested": np.array([1.0, np.nan])}]
)
def test_report_file_is_strict_json_or_refused(reporter_module, tmp_path, value):
    try:
        paths = reporter_module.save_json_txt_pair(
            {"metric": value}, tmp_path, "run", "latest", ["test"]
        )
    except (ValueError, TypeError):
        return  # Explicit refusal is admissible; no replacement policy imposed.
    for path in paths[:2]:

        def reject(token):
            raise ValueError(f"Nonstandard JSON: {token}")

        try:
            parsed = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject)
        except ValueError as error:
            pytest.fail(str(error))
        assert "metric" in parsed, "Dropping all metrics is not a fix"


@pytest.mark.parametrize("kind", ["joblib", "json"])
def test_archive_collision_cannot_overwrite_first_run(reporter_module, tmp_path, kind):
    def save(obj):
        if kind == "joblib":
            return reporter_module.save_joblib_pair(
                obj, tmp_path, "same_second", "latest"
            )
        return reporter_module.save_json_txt_pair(
            obj, tmp_path, "same_second", "latest", [obj["run"]]
        )

    first = save({"run": "a"})[0]
    original = first.read_bytes()
    try:
        second = save({"run": "b"})[0]
    except FileExistsError:
        assert first.read_bytes() == original
        return
    assert first.read_bytes() == original, "Timestamped archive silently overwritten"
    assert second != first, "Distinct successful runs need distinct archive paths"


@pytest.mark.parametrize("entry", ["pca", "embedding", "ae"])
def test_notebook_same_second_preserves_first_experiment(
    reporter_module, reporter_case, entry
):
    c = reporter_case
    fn, args, key = {
        "pca": (reporter_module.save_pca_run, c.pca_args, "joblib_path"),
        "embedding": (
            reporter_module.save_umap_tsne_run,
            c.embedding_args,
            "joblib_run",
        ),
        "ae": (reporter_module.save_autoencoder_run, c.ae_args, "joblib_path"),
    }[entry]
    first = fn(**args)
    originals = {
        name: first[name].read_bytes() for name in [key, "json_path", "txt_path"]
    }
    new = dict(args, y=c.y[::-1])
    try:
        second = fn(**new)
    except FileExistsError:
        second = None
    assert all(
        first[name].read_bytes() == data for name, data in originals.items()
    ), "First experiment replaced"
    if second is not None:
        assert first[key] != second[key]


@pytest.mark.parametrize("failure_stage", ["json_latest", "txt_latest"])
def test_failed_run_cannot_publish_mixed_latest(
    reporter_module, reporter_case, monkeypatch, failure_stage
):
    c = reporter_case
    reporter_module.save_umap_tsne_run(**c.embedding_args)
    monkeypatch.setattr(reporter_module, "make_timestamp", lambda: "20261109T170305Z")
    real_open = builtins.open
    suffix = ".json" if failure_stage == "json_latest" else ".txt"

    def fail_selected(path, mode="r", *args, **kwargs):
        if str(path).endswith(f"run_latest{suffix}") and "w" in mode:
            raise OSError("synthetic publication interruption")
        return real_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", fail_selected)
    with pytest.raises(OSError, match="publication interruption"):
        reporter_module.save_umap_tsne_run(**dict(c.embedding_args, y=c.y[::-1]))
    run_dir = c.reports / "runs/phy3500_umap_tsne"
    payload = joblib.load(run_dir / "phy3500_embeddings_latest.joblib")
    report = json.loads(
        (run_dir / "phy3500_umap_tsne_run_latest.json").read_text(encoding="utf-8")
    )
    text = (run_dir / "phy3500_umap_tsne_run_latest.txt").read_text(encoding="utf-8")
    assert (
        payload["timestamp_utc"] == report["run_timestamp_utc"]
    ), "Latest JSON and joblib belong to different runs"
    assert payload["timestamp_utc"] in text, "Latest TXT belongs to an earlier run"


def test_prediction_figure_is_not_a_model_path(reporter_module, reporter_case):
    c = reporter_case
    figure = c.figures / "xgb_predictions.png"
    figure.write_bytes(b"synthetic figure")
    result = reporter_module.save_umap_tsne_run(
        **dict(c.embedding_args, y_pred=np.array(["F"] * 6), path_xgb=figure)
    )
    report = json.loads(result["json_path"].read_text(encoding="utf-8"))
    assert report["paths"]["xgboost_prediction_figure"] == str(figure)
    assert report["xgboost_summary"].get("model_path") != str(
        figure
    ), "Figure incorrectly attributed as trained model"


@pytest.mark.parametrize("metric", ["ae", "pca"])
def test_failed_metric_is_diagnosed(
    reporter_module, reporter_case, monkeypatch, caplog, metric
):
    c = reporter_case

    def failed(*args, **kwargs):
        raise ValueError("synthetic numerical failure")

    args = dict(c.ae_args)
    if metric == "ae":
        monkeypatch.setattr(c.ae, "reconstruction_mse", failed)
    else:
        from types import SimpleNamespace

        args["pca"] = SimpleNamespace(reconstruction_error=failed)
    with caplog.at_level("WARNING"):
        try:
            result = reporter_module.save_autoencoder_run(**args)
        except ValueError as error:
            assert "synthetic numerical failure" in str(error)
            return
    report = json.loads(result["json_path"].read_text(encoding="utf-8"))
    # A persistent status or a diagnostic log is acceptable; no schema prescribed.
    diagnostics = (
        json.dumps(
            [
                report.get("metrics"),
                report.get("diagnostics"),
                report.get("errors"),
                report.get("status"),
            ]
        ).lower()
        + caplog.text.lower()
    )
    assert any(
        word in diagnostics for word in ["error", "failed", "unavailable", "failure"]
    ), "Metric failure silently reduced to null"


def test_nonfinite_latent_not_reported_as_ordinary_spearman(
    reporter_module, reporter_case
):
    c = reporter_case
    Z = np.array([[1.0, 6], [2, 5], [np.inf, 3], [3, 4], [5, 2], [6, 1]])
    args = dict(
        c.ae_args, Z_ae=Z, meta=c.meta.assign(teff_gspphot=[10, 20, 30, 40, 50, 60])
    )
    try:
        with np.errstate(invalid="ignore"):
            result = reporter_module.save_autoencoder_run(**args)
    except ValueError as error:
        assert any(
            word in str(error).lower()
            for word in ["finite", "inf", "latent", "correlation"]
        )
        return
    report = json.loads(result["json_path"].read_text(encoding="utf-8"))
    value = report["correlations"]["ae_latent_vs_physical_spearman"]["teff_gspphot"][
        "latent_axis_1"
    ]
    # Finite pairs' ranks are identical: 1,2,3,4,5. Omit/refuse is also valid.
    assert value is None or math.isclose(
        value, 1.0, abs_tol=1e-14
    ), "Inf was ranked as a physical coordinate"


def test_single_latent_axis_is_reported_or_refused_clearly(
    reporter_module, reporter_case
):
    c = reporter_case
    args = dict(
        c.ae_args,
        Z_ae=np.arange(6.0)[:, None],
        meta=c.meta.assign(teff_gspphot=[1.0, 2, 3, 4, 5, 6]),
    )
    try:
        result = reporter_module.save_autoencoder_run(**args)
    except (ValueError, IndexError) as error:
        assert not isinstance(error, IndexError), "Unchecked second latent coordinate"
        assert any(
            word in str(error).lower()
            for word in ["axis", "latent", "dimension", "shape"]
        )
        return
    report = json.loads(result["json_path"].read_text(encoding="utf-8"))
    assert report["shapes"]["Z_ae"] == [6, 1]


def test_positive_valid_report_is_nonempty(reporter_module, reporter_case):
    c = reporter_case
    result = reporter_module.save_autoencoder_run(**c.ae_args)
    report = json.loads(result["json_path"].read_text(encoding="utf-8"))
    assert report["metrics"]["ae_mse_global"] == pytest.approx(1 / 9)
    assert report["class_counts"] == {"STAR": 3, "QSO": 2, "GALAXY": 1}
    np.testing.assert_array_equal(joblib.load(result["joblib_path"])["y"], c.y)
