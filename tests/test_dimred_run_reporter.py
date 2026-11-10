"""Independent synthetic contracts for reporter arithmetic and artifact boundaries."""

from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import pytest


def strict_load(path):
    def reject(token):
        raise ValueError(f"Nonstandard JSON token: {token}")

    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=reject)


@pytest.mark.parametrize(
    "labels, expected",
    [
        (["M", "F", "M", "G", "M", "F"], {"M": 3, "F": 2, "G": 1}),
        ([], {}),
        ([3, -1, 3, 7], {"-1": 1, "3": 2, "7": 1}),
        ([True, False, True], {"True": 2, "False": 1}),
    ],
)
def test_class_counts_manual(reporter_module, labels, expected):
    assert reporter_module.class_counts(labels) == expected


@pytest.mark.parametrize("limit, rows", [(None, 3), (0, 0), (2, 2)])
def test_dataframe_records_finite_and_nullable(reporter_module, limit, rows):
    df = pd.DataFrame(
        {
            "flux / A": pd.array([1.5, 4.5, None], dtype="Float64"),
            "count": pd.array([4, None, 9], dtype="Int64"),
            "name": pd.Series(
                ["étoile", None, "QSO"], dtype=object, index=[91, 7, 300]
            ),
            "flag": pd.array([True, None, False], dtype="boolean"),
        },
        index=[91, 7, 300],
    )
    before = df.copy(deep=True)
    expected = [
        {"flux / A": 1.5, "count": 4, "name": "étoile", "flag": True},
        {"flux / A": 4.5, "count": None, "name": None, "flag": None},
        {"flux / A": None, "count": 9, "name": "QSO", "flag": False},
    ][:rows]
    result = reporter_module.safe_df_records(df, limit)
    assert result == expected
    assert json.loads(json.dumps(result, allow_nan=False)) == expected
    pd.testing.assert_frame_equal(df, before)


def test_dataframe_records_none_and_empty(reporter_module):
    assert reporter_module.safe_df_records(None) is None
    assert reporter_module.safe_df_records(pd.DataFrame(columns=["a"])) == []


@pytest.mark.parametrize(
    "value, expected",
    [
        (np.int64(107), 107),
        (np.uint32(71), 71),
        (np.float32(1.25), 1.25),
        (np.float64(-7.5), -7.5),
        (np.array([[1, 4], [9, 16]]), [[1, 4], [9, 16]]),
        (Path("input") / "étoile.csv", str(Path("input") / "étoile.csv")),
        (datetime(2026, 11, 9, 12, 3, 4), "2026-11-09T12:03:04"),
        (
            datetime(2026, 11, 9, 12, 3, 4, tzinfo=timezone(timedelta(hours=-5))),
            "2026-11-09T12:03:04-05:00",
        ),
        (pd.Timestamp("2026-11-09T17:03:04Z"), "2026-11-09T17:03:04+00:00"),
    ],
)
def test_json_default_exact(reporter_module, value, expected):
    encoded = json.dumps({"outer": [value]}, default=reporter_module.json_default)
    assert json.loads(encoded) == {"outer": [expected]}


@pytest.mark.parametrize("value", [object(), {1, 2}, 1 + 2j, np.bool_(True)])
def test_json_default_unsupported_types_refused(reporter_module, value):
    with pytest.raises(TypeError):
        json.dumps(value, default=reporter_module.json_default)


def test_embedding_axes_independent_population_oracle(reporter_module):
    Z = np.array([[1.0, 4, 9], [2, 7, 3], [8, 1, 6]])
    # Population variances: 86/9, 6, 6; no NumPy reduction as oracle.
    stats = reporter_module.embedding_stats(Z)
    expected = dict(
        shape=[3, 3],
        x_min=1,
        x_max=8,
        y_min=1,
        y_max=7,
        x_mean=11 / 3,
        y_mean=4,
        x_std=math.sqrt(86) / 3,
        y_std=math.sqrt(6),
    )
    assert stats.keys() == expected.keys()
    for key in expected:
        assert stats[key] == pytest.approx(expected[key])
    nd = reporter_module.embedding_stats_nd(Z, max_axes=2)
    assert nd == pytest.approx(
        {
            "shape": [3, 3],
            "axis_1_min": 1,
            "axis_1_max": 8,
            "axis_1_mean": 11 / 3,
            "axis_1_std": math.sqrt(86) / 3,
            "axis_2_min": 1,
            "axis_2_max": 7,
            "axis_2_mean": 4,
            "axis_2_std": math.sqrt(6),
        }
    )
    all_axes = reporter_module.embedding_stats_nd(Z)
    assert all_axes["axis_3_min"] == 3
    assert all_axes["axis_3_max"] == 9
    assert all_axes["axis_3_mean"] == 6
    assert all_axes["axis_3_std"] == pytest.approx(math.sqrt(6))


@pytest.mark.parametrize("Z", [None, [1, 4, 8], [[1], [4]], np.empty((3, 0))])
def test_embedding_shape_only_cases(reporter_module, Z):
    expected = None if Z is None else {"shape": list(np.asarray(Z).shape)}
    assert reporter_module.embedding_stats(Z) == expected
    nd = reporter_module.embedding_stats_nd(Z, max_axes=0)
    assert nd == expected


def test_embedding_missing_values_use_axis_population(reporter_module):
    result = reporter_module.embedding_stats([[1, 4], [np.nan, 7], [9, np.nan]])
    assert result == {
        "shape": [3, 2],
        "x_min": 1,
        "x_max": 9,
        "x_mean": 5,
        "x_std": 4,
        "y_min": 4,
        "y_max": 7,
        "y_mean": 5.5,
        "y_std": 1.5,
    }
    nd = reporter_module.embedding_stats_nd([[3], [11]])
    assert nd == {
        "shape": [2, 1],
        "axis_1_min": 3,
        "axis_1_max": 11,
        "axis_1_mean": 7,
        "axis_1_std": 4,
    }


@pytest.mark.parametrize("function", ["embedding_stats", "embedding_stats_nd"])
def test_embedding_nonnumeric_and_empty_refused(reporter_module, function):
    fn = getattr(reporter_module, function)
    with pytest.raises((ValueError, TypeError)):
        fn([["x", "y"], ["a", "b"]])
    with pytest.raises(ValueError):
        fn(np.empty((0, 2)))


def test_vector_statistics_manual_linear_percentiles(reporter_module):
    # Sorted finite values: 1,2,4,20. Linear p05=1+.15; p95=4+.85*16.
    result = reporter_module.vector_stats([[20, 1, np.nan], [4, np.inf, 2]])
    assert result == pytest.approx(
        {
            "n": 4,
            "min": 1,
            "max": 20,
            "mean": 27 / 4,
            "std": math.sqrt(955) / 4,
            "p05": 1.15,
            "p50": 3,
            "p95": 17.6,
        }
    )


@pytest.mark.parametrize("values", [[], [np.nan, np.inf, -np.inf], None])
def test_vector_unavailable_population(reporter_module, values):
    assert reporter_module.vector_stats(values) is None


@pytest.mark.parametrize("values", [[7], [7, 7, 7], [[7, 7], [7, 7]]])
def test_vector_constant_and_singleton(reporter_module, values):
    result = reporter_module.vector_stats(values)
    assert result == {
        "n": np.asarray(values).size,
        "min": 7,
        "max": 7,
        "mean": 7,
        "std": 0,
        "p05": 7,
        "p50": 7,
        "p95": 7,
    }


def test_numeric_means_columns_and_immutability(reporter_module):
    df = pd.DataFrame(
        {
            "a": [1.0, np.nan, 11],
            "b": [2, 4, 12],
            "s": ["1", "2", "3"],
            "flag": [True, False, True],
        }
    )
    before = df.copy(deep=True)
    assert reporter_module.numeric_means(df) == {"a": 6, "b": 6}
    assert reporter_module.numeric_means(df.iloc[:0]) is None
    assert reporter_module.numeric_means(None) is None
    pd.testing.assert_frame_equal(df, before)


@pytest.mark.parametrize(
    "value, expected",
    [
        (None, "N/A"),
        ("bad", "N/A"),
        (np.nan, "N/A"),
        (np.inf, "N/A"),
        (0, "0.00s"),
        (1.234, "1.23s"),
        (-4.5, "-4.50s"),
        (1e12, "1000000000000.00s"),
    ],
)
def test_duration_format_is_not_physical_validation(reporter_module, value, expected):
    assert reporter_module.fmt_seconds(value) == expected


def test_shapes_and_sensitivity(reporter_module):
    assert reporter_module.shape_or_none(None) is None
    assert reporter_module.shape_or_none([[1], [2]]) == [2, 1]
    assert reporter_module.sensitivity_shapes(None) is None
    assert reporter_module.sensitivity_shapes({9: np.zeros((3, 2)), "x": [1, 2]}) == {
        "9": [3, 2],
        "x": [2],
    }


def test_cluster_counts_and_classes_exact(reporter_module):
    clusters = [7, -1, 2, 7, 2, 7, 2, 7, -1]
    predictions = ["M", "F", "G", "F", "G", "M", "F", "M", "F"]
    assert reporter_module.cluster_top_counts(clusters) == {"7": 4, "2": 3}
    assert reporter_module.cluster_top_counts(clusters, 1) == {"7": 4}
    assert reporter_module.top3_pred_by_cluster(clusters, predictions) == {
        "noise": {"F": 2},
        "2": {"G": 2, "F": 1},
        "7": {"M": 3, "F": 1},
    }
    # No ordering is asserted for tied clusters or tied classes.
    assert reporter_module.cluster_top_counts([8, 3, 8, 3]) == {"8": 2, "3": 2}


@pytest.mark.parametrize("labels, expected", [([], {}), ([-1, -1], {})])
def test_cluster_empty_or_only_noise(reporter_module, labels, expected):
    assert reporter_module.cluster_top_counts(labels) == expected
    preds = ["F"] * len(labels)
    assert reporter_module.top3_pred_by_cluster(labels, preds) == (
        {} if not labels else {"noise": {"F": 2}}
    )


@pytest.mark.parametrize("predictions", [["F"], ["F", "G", "M"]])
def test_cluster_cardinality_refused(reporter_module, predictions):
    with pytest.raises((IndexError, ValueError)):
        reporter_module.top3_pred_by_cluster([0, 1], predictions)


def test_timestamp_utc_controlled_clock(reporter_module, monkeypatch):
    local = datetime(
        2026, 11, 9, 23, 58, 7, 999999, tzinfo=timezone(timedelta(hours=-5))
    )

    class Clock:
        @staticmethod
        def now(tz):
            assert tz == timezone.utc
            return local.astimezone(tz)

    monkeypatch.setattr(reporter_module, "datetime", Clock)
    assert reporter_module.make_timestamp() == "20261110T045807Z"
    # Seconds precision is an observation, not an assertion of unique run IDs.
    assert reporter_module.make_timestamp() == "20261110T045807Z"


def test_run_directory_preserves_content(reporter_module, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for base in [Path("relative"), tmp_path / "absolute"]:
        result = reporter_module.make_run_dir(base, "nb")
        assert result == base / "runs" / "nb"
        sentinel = result / "existing.txt"
        sentinel.write_text("keep", encoding="utf-8")
        assert reporter_module.make_run_dir(base, "nb") == result
        assert sentinel.read_text(encoding="utf-8") == "keep"


def test_run_directory_permission_error_propagates(
    reporter_module, tmp_path, monkeypatch
):
    def denied(*args, **kwargs):
        raise PermissionError("synthetic denied directory")

    monkeypatch.setattr(Path, "mkdir", denied)
    with pytest.raises(PermissionError, match="synthetic denied"):
        reporter_module.make_run_dir(tmp_path, "nb")


def assert_payload(payload, expected):
    assert payload.keys() == expected.keys()
    for key, value in expected.items():
        if isinstance(value, pd.DataFrame):
            pd.testing.assert_frame_equal(payload[key], value)
        elif isinstance(value, np.ndarray):
            np.testing.assert_array_equal(payload[key], value)
        else:
            assert payload[key] == value


def test_joblib_three_destinations_and_run_isolation(reporter_module, tmp_path):
    legacy = tmp_path / "legacy.joblib"
    first = {
        "a": np.array([[7, 2], [31, 8]]),
        "meta": pd.DataFrame({"obsid": [91, 7]}, index=[5, 2]),
        "run": "a",
    }
    run_a, latest = reporter_module.save_joblib_pair(
        first, tmp_path, "a", "latest", legacy
    )
    for path in [run_a, latest, legacy]:
        assert_payload(joblib.load(path), first)
    original = run_a.read_bytes()
    second = {
        "a": np.array([[1, 9]]),
        "meta": pd.DataFrame({"obsid": [300]}),
        "run": "b",
    }
    run_b, latest_b = reporter_module.save_joblib_pair(
        second, tmp_path, "b", "latest", legacy
    )
    assert latest_b == latest
    assert run_a.read_bytes() == original
    for path in [run_b, latest_b, legacy]:
        assert_payload(joblib.load(path), second)


def test_json_txt_four_destinations_and_unicode(reporter_module, tmp_path):
    report = {
        "run": "a",
        "étoile": [np.int64(91), np.array([2, 7])],
        "source": Path("synth.csv"),
    }
    paths = reporter_module.save_json_txt_pair(
        report, tmp_path, "a", "latest", ["étoile", "MSE: 1.25"]
    )
    assert paths == tuple(
        tmp_path / name for name in ["a.json", "latest.json", "a.txt", "latest.txt"]
    )
    for path in paths[:2]:
        assert strict_load(path) == {
            "run": "a",
            "étoile": [91, [2, 7]],
            "source": "synth.csv",
        }
        assert "étoile" in path.read_text(encoding="utf-8")
    for path in paths[2:]:
        assert [
            line for line in path.read_text(encoding="utf-8").splitlines() if line
        ] == ["étoile", "MSE: 1.25"]
    old_json, old_text = paths[0].read_bytes(), paths[2].read_bytes()
    newer = reporter_module.save_json_txt_pair(
        {"run": "b"}, tmp_path, "b", "latest", ["run b"]
    )
    assert strict_load(newer[0]) == strict_load(newer[1]) == {"run": "b"}
    assert newer[2].read_bytes() == newer[3].read_bytes()
    assert paths[0].read_bytes() == old_json and paths[2].read_bytes() == old_text


def test_joblib_without_legacy_and_missing_parent(reporter_module, tmp_path):
    run, latest = reporter_module.save_joblib_pair({"x": 7}, tmp_path, "run", "latest")
    assert joblib.load(run) == joblib.load(latest) == {"x": 7}
    with pytest.raises(FileNotFoundError):
        reporter_module.save_joblib_pair({}, tmp_path / "absent", "run", "latest")


def check_saved_pair(result, joblib_key):
    report = strict_load(result["json_path"])
    text = result["txt_path"].read_text(encoding="utf-8")
    assert result["timestamp"] == report["run_timestamp_utc"]
    assert result["timestamp"] in text
    assert str(result[joblib_key]) in text
    for key in ["json_path", "txt_path", joblib_key]:
        assert result[key].is_file()
        assert str(result[key]) in result["summary"]
    payload = joblib.load(result[joblib_key])
    if joblib_key == "joblib_run":
        assert payload["timestamp_utc"] == result["timestamp"]
    return report, payload, text


@pytest.mark.integration
def test_pca_notebook_roundtrip_thresholds_and_rows(reporter_module, reporter_case):
    c = reporter_case
    c.pca_args.update(
        corr_df=pd.DataFrame({"teff": [0.7, -0.2]}, index=["PC1", "PC2"]),
        sep_df=pd.DataFrame({"distance": [3.5, 8.25]}),
        recon_df=pd.DataFrame({"mse": [0.25, 0.125]}),
    )
    figure = c.figures / "pca.png"
    figure.write_bytes(b"synthetic inventory only")
    result = reporter_module.save_pca_run(**c.pca_args)
    report, payload, text = check_saved_pair(result, "joblib_path")
    for key in [
        "scores",
        "scores_spec",
        "y",
        "y_spec",
        "meta",
        "meta_spec",
        "feature_names",
    ]:
        assert_payload({key: payload[key]}, {key: c.pca_args[key]})
    np.testing.assert_array_equal(payload["scores_95pct"], c.scores[:, :3])
    np.testing.assert_array_equal(payload["scores_spec_95pct"], c.scores[:, [2, 0]])
    assert payload["n_components_95pct"] == 3
    assert payload["n_keep_spec"] == 2
    assert payload["features_stem"] == "features_synthetic"
    assert report["pca_features"]["n_components_for_variance"] == {
        "80pct": 1,
        "90pct": 2,
        "95pct": 3,
        "99pct": 4,
    }
    assert report["pca_spectra"]["n_components_for_variance"] == {
        "80pct": 1,
        "90pct": 2,
        "95pct": 2,
        "99pct": 2,
    }
    assert report["shapes"]["scores_95pct"] == [6, 3]
    assert report["shapes"]["scores_spec_95pct"] == [6, 2]
    assert report["class_counts"] == {
        "y": {"STAR": 3, "QSO": 2, "GALAXY": 1},
        "y_spec": {"STAR": 3, "QSO": 2, "GALAXY": 1},
    }
    assert report["paths"]["features_path"] == str(c.pca_args["FEATURES_PATH"])
    assert report["paths"]["catalog_path"] == str(c.pca_args["CATALOG_PATH"])
    assert report["figures_saved"] == [str(figure)]
    assert report["pca_features"]["variance_explained_total"] == 1
    assert report["pca_features"]["top_features_pc2"][0] == {
        "feature": "gamma",
        "loading": 2,
    }
    assert report["corrélations_pc_gaia_spearman"] == [
        {"index": "PC1", "teff": 0.7},
        {"index": "PC2", "teff": -0.2},
    ]
    assert report["class_séparation"] == [{"distance": 3.5}, {"distance": 8.25}]
    assert report["reconstruction_error"] == [{"mse": 0.25}, {"mse": 0.125}]
    assert "80pct" in text and "99pct" in text
    for path in [
        c.reports / "phy3500_pca_output.joblib",
        result["run_dir"] / "phy3500_pca_run_latest.joblib",
    ]:
        assert path.is_file(), "Advertised PCA alias was not written"
        assert_payload(joblib.load(path), payload)
    assert strict_load(result["run_dir"] / "phy3500_pca_run_latest.json") == report


@pytest.mark.integration
def test_pca_reporter_real_fitted_sign_invariant(
    reporter_module, reporter_case, dimred_pca_class
):
    c = reporter_case
    pca = dimred_pca_class(n_components=3, svd_solver="full")
    scores = pca.fit_transform(c.X, c.pca_args["feature_names"])
    result = reporter_module.save_pca_run(
        **dict(c.pca_args, pca=pca, pca_spec=pca, scores=scores, scores_spec=scores)
    )
    report, payload, _ = check_saved_pair(result, "joblib_path")
    # Six paired coordinates have total squared energy 28, eigenvalue ratios 9:4:1.
    np.testing.assert_allclose(payload["scores"] ** 2, c.X**2, atol=1e-25)
    np.testing.assert_allclose(payload["scores"], scores, atol=1e-14)
    assert report["pca_features"]["n_components_for_variance"] == {
        "80pct": 2,
        "90pct": 2,
        "95pct": 3,
        "99pct": 3,
    }
    assert report["pca_features"]["variance_explained_total"] == pytest.approx(1)


@pytest.mark.integration
@pytest.mark.parametrize("mode", ["umap", "tsne", "both", "3d"])
def test_embedding_optional_modes_roundtrip(reporter_module, reporter_case, mode):
    c = reporter_case
    args = dict(c.embedding_args, Z_umap=None, Z_tsne=None)
    if mode in ["umap", "both"]:
        args["Z_umap"] = c.scores[:, :2]
    if mode in ["tsne", "both"]:
        args["Z_tsne"] = c.scores[:, [2, 3]]
    if mode == "3d":
        args["Z_umap3"] = c.scores[:, :3]
    result = reporter_module.save_umap_tsne_run(**args)
    report, payload, _ = check_saved_pair(result, "joblib_run")
    for name in ["Z_umap", "Z_tsne", "Z_umap3"]:
        expected = args.get(name)
        assert report["captured_sections"][f"has_{name}"] == (expected is not None)
        if expected is None:
            assert payload[name] is None
        else:
            np.testing.assert_array_equal(payload[name], expected)
    np.testing.assert_array_equal(payload["y"], c.y)
    pd.testing.assert_frame_equal(payload["meta"], c.meta)
    assert report["xgboost_summary"] is None and report["hdbscan_summary"] is None
    assert report["zoom_pair_summary"] is None
    assert report["stability"]["umap"] is None
    assert report["sensitivity"]["umap_shapes"] is None
    assert_payload(joblib.load(result["joblib_latest"]), payload)
    assert_payload(joblib.load(c.reports / "phy3500_embeddings.joblib"), payload)
    assert (
        strict_load(result["run_dir"] / "phy3500_umap_tsne_run_latest.json") == report
    )


def test_embedding_no_input_refused_before_writes(reporter_module, reporter_case):
    c = reporter_case
    with pytest.raises(RuntimeError, match="embedding"):
        reporter_module.save_umap_tsne_run(**dict(c.embedding_args, Z_umap=None))
    assert not c.reports.exists()


@pytest.mark.integration
def test_embedding_all_sections_exact(reporter_module, reporter_case):
    c = reporter_case
    Z = np.array([[0.0, 0], [1, 1], [1, 0], [-1, 0], [2, 0.5], [3, 0]])
    # Inclusive rectangle [0,1]x[0,1] includes rows 0,1,2 only.
    png = c.figures / "prediction.png"
    html = c.figures / "umap.html"
    png.write_bytes(b"synthetic inventory only")
    html.write_text("<p>synthetic</p>", encoding="utf-8")
    stability = pd.DataFrame(
        {"seed": [3, 11], "distance": [0.2, 0.8], "label": ["a", "b"]}
    )
    engine = SimpleNamespace(
        fit_time_=2.375, params_used={"random_state": 17, "min_dist": 0.23}
    )
    args = dict(
        c.embedding_args,
        Z_umap=Z,
        Z_tsne=c.scores[:, [2, 3]],
        Z_umap3=c.scores[:, :3],
        Z_umap_neg=-c.scores[:, :2],
        y=np.array([" star ", "glaxaxy", " qs0", "galaxie", "STAR", "QSO"]),
        umap_engine=engine,
        tsne_engine=SimpleNamespace(fit_time_=7.25, params_used={"perplexity": 2}),
        cluster_labels=np.array([7, -1, 2, 7, 2, 7]),
        labels_pres=np.array([7, 7, 2, 7, 2, 7]),
        clusterer=SimpleNamespace(
            min_cluster_size=2,
            min_samples=1,
            metric="euclidean",
            cluster_selection_method="leaf",
        ),
        y_pred=np.array(["F", "G", "F", "M", "G", "F"]),
        y_aligned=np.array(["F", "G", "M", "F", "G", "M"]),
        cl_aligned=np.array([7, -1, 2, 7, 2, 7]),
        confidence=np.array([0.2, 0.3, 0.4, 0.5, 0.7, 0.9]),
        classes_pred=["M", "F", "G"],
        fg_mask=np.array([True, True, True, False, True, True]),
        path_xgb=png,
        path_fg=c.figures / "fg.png",
        pca_output_path="pca_synthetic.joblib",
        axis3_best_param="teff_gspphot",
        axis3_best_corr=-0.75,
        path_3d_html=html,
        path_3d_teff=c.figures / "teff.html",
        path_3d_feh=c.figures / "feh.html",
        stab_umap=stability,
        stab_tsne=stability.iloc[::-1],
        sensitivity_umap={"n=2": Z},
        sensitivity_tsne={"p=3": c.scores[:, :3]},
        ZOOM_X=(0, 1),
        ZOOM_Y=(0, 1),
        features_stem="features_synthetic",
    )
    result = reporter_module.save_umap_tsne_run(**args)
    report = strict_load(result["json_path"])
    payload = joblib.load(result["joblib_run"])
    assert report["zoom_pair_summary"] == {
        "zoom_x": [0, 1],
        "zoom_y": [0, 1],
        "n_points_total": 3,
        "counts_in_zoom": {"STAR": 1, "GALAXY": 1, "QSO": 1},
    }
    assert report["class_counts"] == {"STAR": 2, "GALAXY": 2, "QSO": 2}
    h = report["hdbscan_summary"]
    assert {
        k: h[k] for k in ["n_points", "n_clusters", "n_noise", "largest_clusters"]
    } == {
        "n_points": 6,
        "n_clusters": 2,
        "n_noise": 1,
        "largest_clusters": {"7": 3, "2": 2},
    }
    assert h["pct_noise"] == pytest.approx(100 / 6)
    assert h["presentation_mode"]["n_noise"] == 0
    assert h["presentation_mode"]["largest_clusters"] == {"7": 4, "2": 2}
    assert h["params"] == {
        "min_cluster_size": 2,
        "min_samples": 1,
        "metric": "euclidean",
        "cluster_selection_method": "leaf",
    }
    xgb = report["xgboost_summary"]
    assert xgb["accuracy_vs_reference"] == 0.5
    assert xgb["confusion_matrix"] == {
        "index": ["F", "G", "M"],
        "columns": ["F", "G", "M"],
        "values": [[1, 0, 1], [0, 2, 0], [2, 0, 0]],
    }
    assert xgb["pred_class_counts"] == {"F": 3, "G": 2, "M": 1}
    assert xgb["top3_pred_by_cluster"] == {
        "noise": {"G": 1},
        "2": {"F": 1, "G": 1},
        "7": {"F": 2, "M": 1},
    }
    assert xgb["n_pred_F_or_G"] == 5 and xgb["classes_pred_list"] == ["M", "F", "G"]
    assert xgb["confidence_stats"]["mean"] == pytest.approx(0.5)
    assert xgb["confidence_stats"]["n"] == 6
    assert report["stability"]["umap_numeric_means"] == {"seed": 7, "distance": 0.5}
    assert report["sensitivity"] == {
        "umap_shapes": {"n=2": [6, 2]},
        "tsne_shapes": {"p=3": [6, 3]},
    }
    assert report["artifacts"] == {
        "png_files": [str(png)],
        "html_files": [str(html)],
        "n_png": 1,
        "n_html": 1,
    }
    assert report["parameters"]["umap"] == engine.params_used
    assert report["timings_seconds"] == {"umap_fit_time": 2.375, "tsne_fit_time": 7.25}
    assert report["umap3d_summary"]["axis3_best_corr"] == -0.75
    assert report["umap3d_summary"]["html_exports"]["umap3d_classes"] == str(html)
    assert report["paths"]["pca_output_path"] == "pca_synthetic.joblib"
    np.testing.assert_array_equal(payload["Z_umap_negative_control"], -c.scores[:, :2])
    np.testing.assert_array_equal(payload["xgb_predictions"], args["y_pred"])
    np.testing.assert_array_equal(payload["xgb_confidence"], args["confidence"])
    pd.testing.assert_frame_equal(payload["stability_umap"], stability)
    assert payload["features_stem"] == "features_synthetic"


@pytest.mark.integration
@pytest.mark.parametrize("aligned", [False, True])
def test_accuracy_is_positional_not_identity_proof(
    reporter_module, reporter_case, aligned
):
    c = reporter_case
    truth = np.array(["STAR", "QSO", "STAR", "GALAXY", "STAR", "QSO"])
    reference = truth[[1, 0, 2, 3, 4, 5]] if aligned else None
    result = reporter_module.save_umap_tsne_run(
        **dict(c.embedding_args, y_pred=truth, y_aligned=reference)
    )
    report = strict_load(result["json_path"])
    assert report["xgboost_summary"]["accuracy_vs_reference"] == pytest.approx(
        4 / 6 if aligned else 1
    )


@pytest.mark.integration
@pytest.mark.parametrize("explicit", [False, True])
def test_autoencoder_history_latent_and_mse(reporter_module, reporter_case, explicit):
    c = reporter_case
    args = dict(c.ae_args)
    if explicit:
        args.update(
            Z_ae=c.scores[:, [2, 0]],
            X_recon=c.X * 0.5,
            history={"train_loss": [12.0, 2.0, 7.0], "val_loss": [5.0, 0.25, 3.0]},
        )
    result = reporter_module.save_autoencoder_run(**args)
    report, payload, text = check_saved_pair(result, "joblib_path")
    expected_Z = c.scores[:, [2, 0]] if explicit else c.X[:, :2] * [2, 3]
    np.testing.assert_array_equal(payload["Z_ae"], expected_Z)
    assert len(c.ae.encoded) == (0 if explicit else 1)
    if explicit:
        np.testing.assert_array_equal(payload["X_recon"], c.X * 0.5)
    else:
        assert payload["X_recon"] is None
    assert payload["history"] == (args["history"] if explicit else c.ae.history_)
    metrics = report["metrics"]
    assert metrics["ae_epochs_done"] == (3 if explicit else 4)
    assert metrics["ae_best_val_loss"] == (0.25 if explicit else 1)
    assert metrics["ae_final_val_loss"] == (3 if explicit else 6)
    assert metrics["ae_final_train_loss"] == (7 if explicit else 2)
    assert metrics["ae_fit_time_s"] == 12.345
    assert metrics["ae_mse_global"] == pytest.approx(1 / 9)
    # Supplied reconstruction is an artifact; metric is evaluated via the AE API.
    assert payload["ae_mse_global"] == pytest.approx(1 / 9)
    np.testing.assert_array_equal(payload["y"], c.y)
    pd.testing.assert_frame_equal(payload["meta"], c.meta)
    assert payload["feature_names"] == ["gamma", "alpha", "beta"]
    assert "12.35s" in text
    for path in [
        c.reports / "phy3500_autoencoder_output.joblib",
        result["run_dir"] / "phy3500_autoencoder_run_latest.joblib",
    ]:
        assert path.is_file(), "Advertised AE alias was not written"
        assert_payload(joblib.load(path), payload)
    assert (
        strict_load(result["run_dir"] / "phy3500_autoencoder_run_latest.json") == report
    )


@pytest.mark.integration
def test_autoencoder_real_pca_correlations_and_interpolation(
    reporter_module, reporter_case, dimred_pca_class
):
    c = reporter_case
    pca = dimred_pca_class(n_components=3, svd_solver="full").fit(
        c.X, c.ae_args["feature_names"]
    )
    # No ties: ranks [1,2,4,3,5,6], sum(d^2)=2 => rho=1-12/210=33/35.
    Z = np.array([[1.0, 6], [2, 5], [4, 3], [3, 4], [5, 2], [6, 1]])
    meta = c.meta.assign(
        teff_gspphot=[10, 20, 30, 40, 50, 60],
        logg_gspphot=[1, np.nan, 2, np.nan, np.nan, np.nan],
    )
    tables = pd.DataFrame({"method": ["AE", "PCA"], "error": [1 / 9, 0.25]})
    figure = c.figures / "ae.png"
    figure.write_bytes(b"synthetic inventory only")
    model = c.root / "model.pt"
    args = dict(
        c.ae_args,
        Z_ae=Z,
        meta=meta,
        pca=pca,
        scores_pca=pca.transform(c.X),
        Z_interp=np.array([[1.0, 2], [4, 6]]),
        X_interp=c.X[:2],
        idx_cold=4,
        idx_hot=1,
        teff_cold=3000,
        teff_hot=7000,
        ae_model_path=model,
        MODELS_DIR=c.root / "models",
        comparison_df=tables,
        summary_df=tables,
        df_summary=tables.set_index("method"),
        corr_pca=pd.DataFrame({"teff": [0.25]}, index=["PC1"]),
        Z_umap=c.scores[:, :2],
        Z_tsne=c.scores[:, [2, 3]],
    )
    result = reporter_module.save_autoencoder_run(**args)
    report, payload, text = check_saved_pair(result, "joblib_path")
    assert report["metrics"]["pca_mse_2_components"] == pytest.approx(1 / 9)
    assert report["metrics"]["pca_mse_n_components_95"] == pytest.approx(0, abs=1e-28)
    assert report["metrics"]["n_pcs_95"] == payload["n_pcs_95"] == 3
    correlations = report["correlations"]["ae_latent_vs_physical_spearman"]
    assert correlations["teff_gspphot"] == pytest.approx(
        {"latent_axis_1": 33 / 35, "latent_axis_2": -33 / 35}
    )
    assert correlations["logg_gspphot"] == {
        "latent_axis_1": None,
        "latent_axis_2": None,
    }
    assert report["interpolation"] == {
        "idx_cold": 4,
        "idx_hot": 1,
        "teff_cold": 3000,
        "teff_hot": 7000,
        "latent_distance": 5,
    }
    assert report["paths"]["ae_model_path"] == str(model)
    assert report["figures_saved"] == [str(figure)]
    assert (
        report["embeddings"]["umap_available"]
        and report["embeddings"]["tsne_available"]
    )
    assert report["tables"]["comparison_df"] == [
        {"method": "AE", "error": 1 / 9},
        {"method": "PCA", "error": 0.25},
    ]
    pd.testing.assert_frame_equal(payload["comparison_df"], tables)
    for key in ["scores_pca", "Z_umap", "Z_tsne", "Z_interp", "X_interp"]:
        np.testing.assert_array_equal(payload[key], args[key])
    assert "latent_distance" in text and "5.000000" in text


@pytest.mark.integration
def test_autoencoder_empty_history_and_explicit_pca_count(
    reporter_module, reporter_case, dimred_pca_class
):
    c = reporter_case
    pca = dimred_pca_class(n_components=3, svd_solver="full").fit(c.X)
    result = reporter_module.save_autoencoder_run(
        **dict(c.ae_args, pca=pca, n_pcs_95=2, history={})
    )
    report = strict_load(result["json_path"])
    m = report["metrics"]
    assert m["ae_epochs_done"] == 0
    assert m["ae_best_val_loss"] is None and m["ae_final_train_loss"] is None
    assert m["pca_mse_n_components_95"] == pytest.approx(1 / 9)
    assert m["n_pcs_95"] == 2


@pytest.mark.parametrize("labels", [[], [-1] * 6])
def test_embedding_degenerate_cluster_summaries(reporter_module, reporter_case, labels):
    result = reporter_module.save_umap_tsne_run(
        **dict(reporter_case.embedding_args, cluster_labels=labels)
    )
    h = strict_load(result["json_path"])["hdbscan_summary"]
    assert h["n_clusters"] == 0 and h["largest_clusters"] == {}
    assert h["n_points"] == h["n_noise"] == len(labels)
    assert h["pct_noise"] == (100 if labels else 0)


def test_embedding_without_labels_metadata_or_scores(reporter_module, reporter_case):
    args = dict(reporter_case.embedding_args, y=None, meta=None, scores_95=None)
    result = reporter_module.save_umap_tsne_run(**args)
    report = strict_load(result["json_path"])
    assert report["class_counts"] is None
    for key in ["has_labels", "has_meta", "has_scores_95"]:
        assert report["captured_sections"][key] is False


def test_cluster_top3_ties_allow_any_equal_frequency_order(reporter_module):
    result = reporter_module.top3_pred_by_cluster([3] * 8, list("FGKMFGKM"))["3"]
    assert len(result) == 3 and set(result) <= set("FGKM")
    assert list(result.values()) == [2, 2, 2]


def test_zoom_nonfinite_is_excluded_or_explicitly_refused(
    reporter_module, reporter_case
):
    c = reporter_case
    Z = np.array([[0.0, 0], [1, 1], [1, 0], [-1, 0], [np.nan, 0.5], [np.inf, 0]])
    try:
        with np.errstate(invalid="ignore"):
            result = reporter_module.save_umap_tsne_run(
                **dict(c.embedding_args, Z_umap=Z, ZOOM_X=(0, 1), ZOOM_Y=(0, 1))
            )
    except ValueError as error:
        assert any(
            word in str(error).lower()
            for word in ["finite", "json", "range", "nan", "inf"]
        )
        return
    report = json.loads(result["json_path"].read_text(encoding="utf-8"))
    assert report["zoom_pair_summary"]["n_points_total"] == 3
    assert report["zoom_pair_summary"]["counts_in_zoom"] == {
        "STAR": 2,
        "QSO": 1,
        "GALAXY": 0,
    }


def test_autoencoder_tied_rank_oracle(reporter_module, reporter_case):
    c = reporter_case
    Z = np.column_stack(([1.0, 3, 2, 4, 5, 6], [6.0, 5, 4, 3, 2, 1]))
    args = dict(c.ae_args, Z_ae=Z, meta=c.meta.assign(bp_rp=[1.0, 2, 2, 4, 5, 6]))
    result = reporter_module.save_autoencoder_run(**args)
    corrs = strict_load(result["json_path"])["correlations"][
        "ae_latent_vs_physical_spearman"
    ]["bp_rp"]
    # Average ranks for ties, then independent centered scalar products.
    physical = [1, 2.5, 2.5, 4, 5, 6]
    denominator_p = math.fsum((v - 3.5) ** 2 for v in physical)
    for name, latent in [
        ("latent_axis_1", [1, 3, 2, 4, 5, 6]),
        ("latent_axis_2", [6, 5, 4, 3, 2, 1]),
    ]:
        covariance = math.fsum((a - 3.5) * (b - 3.5) for a, b in zip(latent, physical))
        variance = math.fsum((v - 3.5) ** 2 for v in latent)
        assert corrs[name] == pytest.approx(
            covariance / math.sqrt(variance * denominator_p)
        )


@pytest.mark.parametrize("failure_number", [1, 2, 3])
def test_joblib_failure_propagates_and_retry_recovers(
    reporter_module, tmp_path, monkeypatch, failure_number
):
    real_dump = joblib.dump
    calls = []

    def interrupted(obj, path, **kwargs):
        calls.append(path)
        if len(calls) == failure_number:
            Path(path).write_bytes(b"synthetic interrupted write")
            raise OSError("synthetic joblib failure")
        return real_dump(obj, path, **kwargs)

    legacy = tmp_path / "legacy.joblib"
    with monkeypatch.context() as patch:
        patch.setattr(joblib, "dump", interrupted)
        with pytest.raises(OSError, match="synthetic joblib failure"):
            reporter_module.save_joblib_pair(
                {"run": "a"}, tmp_path, "a", "latest", legacy
            )
    run, latest = reporter_module.save_joblib_pair(
        {"run": "b"}, tmp_path, "b", "latest", legacy
    )
    assert (
        joblib.load(run) == joblib.load(latest) == joblib.load(legacy) == {"run": "b"}
    )


@pytest.mark.parametrize("failure_number", [1, 2, 3, 4])
def test_json_txt_failure_propagates_and_retry_recovers(
    reporter_module, tmp_path, monkeypatch, failure_number
):
    import builtins

    real_open = builtins.open
    calls = []

    def interrupted(path, mode="r", *args, **kwargs):
        if "w" in mode:
            calls.append(path)
            if len(calls) == failure_number:
                raise OSError("synthetic report failure")
        return real_open(path, mode, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(builtins, "open", interrupted)
        with pytest.raises(OSError, match="synthetic report failure"):
            reporter_module.save_json_txt_pair(
                {"run": "a"}, tmp_path, "a", "latest", ["run a"]
            )
    paths = reporter_module.save_json_txt_pair(
        {"run": "b"}, tmp_path, "b", "latest", ["run b"]
    )
    assert strict_load(paths[0]) == strict_load(paths[1]) == {"run": "b"}
    assert (
        paths[2].read_text(encoding="utf-8")
        == paths[3].read_text(encoding="utf-8")
        == "run b"
    )
