"""M3.1 bounded API, numerical artist and synthetic integration contracts."""

import base64
import importlib.util
import io
import math
from pathlib import Path
import re
from types import SimpleNamespace

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pandas as pd
import pytest

_spec = importlib.util.spec_from_file_location(
    "m31_contracts", Path(__file__).with_name("m31_visualizer_contracts.py")
)
contracts = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(contracts)
tools_case = contracts.tools_case
pytestmark = pytest.mark.integration


def test_scan_refresh_and_labels_preserve_nested_identity(tools_case):
    c = tools_case
    for name in ["B/same.fits.gz", "A/same.fits.gz", "A/other.fits.gz"]:
        c.write(name)
    (Path(c.paths["RAW_DATA_DIR"]) / "ignored.txt").write_text("synthetic")
    c.catalog(
        {"fits_name": ["B\\same.fits.gz", "A/same.fits.gz"], "subclass": ["G2", "F5"]}
    )
    assert c.viz.refresh_available_spectra() == 3
    assert c.viz.available_spectra == [
        "A/other.fits.gz",
        "A/same.fits.gz",
        "B/same.fits.gz",
    ]
    assert c.viz._load_labels_catalog() == {"B/same": "G2", "A/same": "F5"}
    assert c.viz.classifier is None


@pytest.mark.parametrize("table", [{"other": [1]}, {"fits_name": [], "subclass": []}])
def test_labels_invalid_schema_has_no_invented_class(tools_case, table):
    c = tools_case
    c.catalog(table)
    assert c.viz._load_labels_catalog() == {}


def test_header_units_identity_and_explicit_error(tools_case):
    c = tools_case
    name, _, _ = c.write(header={"DESIG": "unique-object", "SUBCLASS": "M3"})
    c.viz._display_formatted_header(name)
    assert "unique-object" in c.displayed[-1].data and "M3" in c.displayed[-1].data
    assert "Å" in c.viz._format_header_line({}, "Wavelength", "ABSENT", "Å")
    assert "N/A" in c.viz._format_header_line({}, "Wavelength", "ABSENT")
    c.viz._display_formatted_header_for_streamlit(c.st, name)
    assert "unique-object" in c.st.markdowns[-1]
    c.viz._display_header_for_streamlit(c.st, {"DESIG": "object-2", "RA": 30})
    assert c.st.jsons[-1]["RA"] == 30 and "object-2" in c.st.markdowns[-1]
    c.viz._display_formatted_header("missing.fits.gz")
    assert "Error" in c.displayed[-1].data


@pytest.mark.parametrize("route", ["matplotlib", "notebook", "plotly"])
@pytest.mark.parametrize("continuum", [1.0, 6.0])
def test_finite_gaussian_peak_identity_all_renderers(tools_case, route, continuum):
    c = tools_case
    name, wave, raw = c.gaussian(continuum=continuum)
    flux = raw / continuum
    red, green, target = contracts.render_spectrum(c, route, name)
    contracts.assert_peak_pairs(red, green, wave, flux)
    if route == "plotly":
        # The geometric progression accumulates rounding over 401 pixels;
        # N*eps bounds that independent construction (< 9e-14 relative).
        assert_allclose(target.data[0].x, wave, rtol=1e-13)
        assert_allclose(target.data[0].y, flux, atol=2e-14)
        assert "Å" in target.layout.xaxis.title.text
        assert "F5" in target.layout.title.text
        assert any(s.name == "Hβ" and s.x0 == 4861 for s in target.layout.shapes)
    else:
        assert_allclose(target.lines[0].get_xdata(), wave, rtol=1e-13)
        assert_allclose(target.lines[0].get_ydata(), flux, atol=2e-14)
        assert_allclose(target.get_xlim(), [4840, 4880])
        assert_allclose(target.get_ylim(), [0, 2])
        assert "Å" in target.get_xlabel() and "F5" in target.get_title()
    if route == "notebook":
        links = " ".join(
            getattr(obj, "data", "")
            for obj in c.displayed
            if isinstance(getattr(obj, "data", ""), str)
        )
        png = base64.b64decode(re.search(r"data:image/png;base64,([^\"]+)", links)[1])
        assert png.startswith(b"\x89PNG\r\n\x1a\n")
        csv = base64.b64decode(
            re.search(r"data:text/csv;base64,([^']+)", links)[1]
        ).decode()
        table = pd.read_csv(io.StringIO(csv), index_col=0)
        assert "Hβ" in table.index and "Equiv. width (Å)" in table.columns
        # Gaussian EW is depth*sigma*sqrt(2*pi); tails beyond +/-8 A
        # are negligible. Allow .002 A for log-grid quadrature + CSV rounding.
        assert (
            abs(
                float(table.loc["Hβ", "Equiv. width (Å)"])
                - 0.5 * 0.8 * np.sqrt(2 * np.pi)
            )
            < 0.002
        )
        assert (
            "tabular"
            in base64.b64decode(
                re.search(r"data:text/plain;base64,([^']+)", links)[1]
            ).decode()
        )


@pytest.mark.parametrize("invalid", [False, True])
def test_no_peaks_and_all_invalid_plotly_are_explicit(tools_case, invalid):
    c = tools_case
    name, _, _ = c.write(flux=np.full(15, np.nan if invalid else 2.0))
    with (
        pytest.warns(UserWarning) if invalid else __import__("contextlib").nullcontext()
    ):
        red, green, fig = contracts.render_spectrum(c, "plotly", name)
    assert red.shape == green.shape == (0, 2)
    assert len(fig.data) == 1


@pytest.mark.parametrize("data", [np.ones(8), np.ones((1, 8))])
def test_invalid_fits_dimensions_and_corruption_report_failure(tools_case, data):
    c = tools_case
    name, _, _ = c.write(data=data)
    assert (
        c.viz._plot_spectrum_analysis_plotly(
            c.st, str(Path(c.paths["RAW_DATA_DIR"]) / name), 0.2, 8
        )
        is None
    )
    assert c.st.errors and not c.st.figures
    assert c.viz._plot_spectra_comparison([name]) is None
    path = Path(c.paths["RAW_DATA_DIR"]) / "corrupt.fits.gz"
    path.write_bytes(b"not gzip")
    assert c.viz._plot_spectra_comparison([path.name]) is None
    assert not c.plt.get_fignums()


def test_comparison_scale_invariance_offsets_and_skip_rank(tools_case, capsys):
    c = tools_case
    a, wa, fa = c.write("A/a.fits.gz", [2, 4, 6, 8, 10])
    b, wb, fb = c.write("B/b.fits.gz", [6, 12, 18, 24, 30], start=5000)
    destination = c.tmp / "exports" / "compare.png"
    _, ax = c.viz._plot_spectra_comparison(
        [a, a, "missing.fits.gz", b], offset=0.5, save_path=str(destination)
    )
    assert len(ax.lines) == 2
    assert_allclose(ax.lines[0].get_xdata(), wa, rtol=2e-14)
    assert_allclose(ax.lines[1].get_xdata(), wb, rtol=2e-14)
    assert_allclose(ax.lines[0].get_ydata(), [1 / 3, 2 / 3, 1, 4 / 3, 5 / 3])
    # Current documented i*offset convention: deduplicated selection index;
    # skipped files leave a gap. This is a display convention, not physics.
    assert_allclose(ax.lines[1].get_ydata(), [4 / 3, 5 / 3, 2, 7 / 3, 8 / 3])
    contracts.assert_curve_identities(
        ax, [a, b], {a: (wa, fa / 6), b: (wb, fb / 18 + 1.0)}
    )
    assert "Skipped" in capsys.readouterr().out
    from PIL import Image

    with Image.open(destination) as im:
        assert im.format == "PNG" and min(im.size) > 100
    _, raw = c.viz._plot_spectra_comparison([a, b], normalize=False)
    assert_array_equal(raw.lines[0].get_ydata(), fa)
    assert_array_equal(raw.lines[1].get_ydata(), fb)
    contracts.assert_curve_identities(raw, [a, b], {a: (wa, fa), b: (wb, fb)})


def test_normalization_comparator_uses_each_grid_and_both_panels(tools_case):
    c = tools_case
    a, wa, fa = c.write("a/a.fits.gz")
    b, wb, fb = c.write("b/b.fits.gz", [6, 12, 18, 24, 30], start=5100)
    c.viz.refresh_available_spectra()
    _, axes = c.viz.plot_normalization_comparison([a, b], save_path="")
    for ax in axes:
        assert len(ax.lines) == 2
        assert_allclose(ax.lines[0].get_xdata(), wa, rtol=2e-14)
        assert_allclose(ax.lines[1].get_xdata(), wb, rtol=2e-14)
    assert_array_equal(axes[0].lines[0].get_ydata(), fa)
    assert_array_equal(axes[0].lines[1].get_ydata(), fb)
    for line in axes[1].lines:
        assert_allclose(line.get_ydata(), [1 / 3, 2 / 3, 1, 4 / 3, 5 / 3])
    assert not list(c.tmp.rglob("*.png"))


def test_position_catalog_validity_counts_cache_and_force(tools_case):
    c = tools_case
    c.write("p/a.fits.gz", header={"RA": 30.0, "DEC": 10.0})
    c.write("p/b.fits.gz", header={"RA": 50.0, "DEC": 30.0})
    c.write("q/a.fits.gz", header={"RA": 390.0, "DEC": -90.0})
    c.write("q/bad.fits.gz", header={"DEC": 91.0})
    c.write("q/nan.fits.gz", header={"RA": "nan"})
    c.viz.refresh_available_spectra()
    df = c.viz._generate_position_catalog()
    assert_array_equal(df.plan_id, ["p", "q"])
    assert_allclose(df[["ra", "dec", "spectra_count"]], [[40, 20, 2], [30, -90, 1]])
    pd.testing.assert_frame_equal(df, c.viz._generate_position_catalog())
    c.write("q/new.fits.gz", header={"RA": 90.0, "DEC": -30.0})
    c.viz.refresh_available_spectra()
    fresh = c.viz._generate_position_catalog(force_regenerate=True)
    assert_allclose(
        fresh.loc[1, ["ra", "dec", "spectra_count"]].to_numpy(float), [60, -60, 2]
    )


def test_empty_positions_and_missing_catalogs_are_absent(tools_case):
    c = tools_case
    assert c.viz._generate_position_catalog().empty
    assert c.viz.plot_sky_coverage(save_path=False) is None
    assert c.viz.plot_subclass_distribution() is None
    assert c.viz._plot_spectra_comparison([]) is None
    assert c.viz.plot_normalization_comparison(save_path="") is None
    assert c.viz.analyze_feature_zeros(export=False).empty
    assert c.viz.plot_shap_summary_bar() is None
    assert c.viz.plot_shap_beeswarm() is None
    assert c.viz._plot_feature_importances_bar() is None
    with pytest.raises(RuntimeError, match="No usable features"):
        c.viz._get_features_sample(SimpleNamespace(), 5, "features_csv")
    with pytest.raises(RuntimeError, match="No spectra"):
        c.viz._get_features_sample(SimpleNamespace(), 5, "raw_fits")


def test_sky_artists_encode_counts_and_declinations(tools_case):
    c = tools_case
    table = pd.DataFrame(
        {
            "plan_id": ["A", "B", "C"],
            "ra": [0, 90, 180],
            "dec": [-30, 0, 60],
            "spectra_count": [1, 2, 4],
        }
    )
    c.monkeypatch.setattr(c.viz, "_generate_position_catalog", lambda: table)
    fig, ax = c.viz.plot_sky_coverage(save_path=False)
    sc = ax.collections[0]
    assert ax.name == "mollweide"
    assert_allclose(sc.get_offsets()[:, 1], [-np.pi / 6, 0, np.pi / 3])
    assert_allclose(sc.get_array(), [1, 2, 4])
    assert_allclose(sc.get_sizes(), [29, 44, 74])
    assert "Spectra per plan" in fig.axes[-1].get_xlabel()


def test_subclass_raw_counts_and_full_population_percentages(tools_case):
    c = tools_case
    c.catalog({"subclass": ["A"] * 5 + ["B"] * 4 + ["C"] * 3 + ["D"] * 2 + ["E"]})
    _, ax = c.viz.plot_subclass_distribution(top_n=3)
    assert_array_equal([p.get_height() for p in ax.patches], [5, 4, 3])
    assert [t.get_text() for t in ax.get_xticklabels()] == ["A", "B", "C"]
    _, ax = c.viz.plot_subclass_distribution(top_n=5, normalize=True)
    assert_allclose(
        [p.get_height() for p in ax.patches], np.array([5, 4, 3, 2, 1]) * 100 / 15
    )


def test_dashboard_single_column_producer_ledger_and_class_identity(tools_case):
    c = tools_case
    for name in ["A/same.fits.gz", "B/same.fits.gz", "B/new.fits.gz"]:
        c.write(name)
    c.viz.refresh_available_spectra()
    c.viz.labels_catalog = {"A/same": "F5", "B/same": "G2", "B/new": "G2"}
    pd.DataFrame(
        {"file_path": ["A\\same.fits.gz", "absent.fits.gz", "A/same.fits.gz"]}
    ).to_csv(Path(c.paths["CATALOG_DIR"]) / "trained_spectra.csv", index=False)
    df = c.viz.display_dataset_dashboard(show_charts=True).set_index("metric")[
        "# value"
    ]
    assert df.to_dict() == {
        "total_available": 3,
        "total_trained": 1,
        "total_new_candidates": 2,
        "distinct_subclasses": 2,
        "distinct_plans": 2,
    }
    ax = c.plt.gcf().axes[0]
    assert sorted(p.get_width() for p in ax.patches) == [1, 2]
    assert c.viz._resolve_trained_log_path().endswith("trained_spectra.csv")


def test_zero_statistics_use_processed_rows_and_keep_nan_separate(tools_case):
    c = tools_case
    for i in range(3):
        c.write(f"p/{i}.fits.gz", [i + 1] * 5)
    c.viz.refresh_available_spectra()
    c.viz.available_spectra.append("missing.fits.gz")

    class Features:
        feature_names = ["zero", "constant", "mixed", "missing"]

        def extract_features(self, matched, wave, flux, invvar):
            return [0, 7, 0 if not hasattr(self, "seen") else 3, np.nan]

    # Deterministic sequence replaces extraction only; statistics remain real.
    rows = iter([[0, 7, 0, np.nan], [0, 7, 3, np.nan], [0, 7, 6, np.nan]])
    c.monkeypatch.setattr(Features, "extract_features", lambda *args: next(rows))
    c.monkeypatch.setattr(c.module, "FeatureEngineer", Features)
    summary = c.viz.analyze_feature_zeros(n_files=10, export=True)
    assert summary.loc["zero", "zero_count"] == 3
    assert summary.loc["mixed", "zero_rate"] == 1 / 3
    assert summary.loc["mixed", "mean"] == 3
    assert_allclose(summary.loc["mixed", "std"], np.sqrt(6))
    assert summary.loc["constant", "is_constant"]
    assert summary.loc["missing", "zero_count"] == 0
    links = " ".join(str(getattr(x, "data", "")) for x in c.displayed)
    assert "3 spectrum" in links and "1" in links
    csv = base64.b64decode(
        re.search(r"data:text/csv;base64,([^']+)", links)[1]
    ).decode()
    pd.testing.assert_frame_equal(
        pd.read_csv(io.StringIO(csv), index_col=0), summary, check_names=False
    )


def test_prepare_flat_pipeline_alignment_imputation_scaling_selection(tools_case):
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.feature_selection import VarianceThreshold
    from sklearn.pipeline import Pipeline

    c = tools_case
    train = pd.DataFrame(
        {"alpha": [1.0, 3.0, 5.0], "beta": [9.0, 9.0, 9.0], "gamma": [10.0, 20.0, 30.0]}
    )
    imputer = SimpleImputer().fit(train)
    scaler = StandardScaler().fit(imputer.transform(train))
    selector = VarianceThreshold(0.1).fit(scaler.transform(imputer.transform(train)))
    pipe = Pipeline(
        [("imputer", imputer), ("scaler", scaler), ("feature_selector", selector)]
    )
    wrapper = SimpleNamespace(
        model_pipeline=pipe,
        feature_names_used=list(train.columns),
        selected_features_=["alpha", "gamma"],
    )
    query = pd.DataFrame(
        {"gamma": [30.0, np.nan], "extra": [8, 9], "alpha": [5.0, 1.0]}
    )
    before = query.copy(deep=True)
    Xt, names = c.viz._prepare_for_model(query, wrapper)
    assert names == ["alpha", "gamma"]
    assert_allclose(Xt, [[np.sqrt(1.5), np.sqrt(1.5)], [-np.sqrt(1.5), 0]], atol=1e-14)
    pd.testing.assert_frame_equal(query, before)


def test_features_csv_sampling_is_reproducible_and_aligned(tools_case):
    c = tools_case
    data = pd.DataFrame(
        {
            "gamma": np.arange(20) * 7,
            "alpha": np.arange(20),
            "main_class": ["F"] * 20,
            "unused": 99,
        }
    )
    data.to_csv(Path(c.paths["PROCESSED_DIR"]) / "features_synthetic.csv", index=False)
    clf = SimpleNamespace(feature_names_used=["alpha", "gamma"])
    a = c.viz._get_features_sample(clf, 7, "features_csv")
    b = c.viz._get_features_sample(clf, 7, "auto")
    pd.testing.assert_frame_equal(a, b)
    assert len(a) == 7 and list(a.columns) == ["alpha", "gamma"]
    assert_array_equal(a.gamma, a.alpha * 7)


def test_binary_shap_mean_absolute_values_and_bar_names(tools_case):
    c = tools_case
    values = np.array([[1, -3, 9], [-3, 1, -3], [5, -5, 3], [-7, 7, -5.0]])
    contracts.shap_case(c, values)
    result = c.viz._run_shap_analysis("synthetic")
    assert_allclose(
        result.set_index("feature").loc[["alpha", "beta", "gamma"], "mean_abs_shap"],
        [4, 4, 5],
    )
    c.viz._last_shap_importances = pd.DataFrame(
        {"feature": ["gamma", "alpha", "beta"], "mean_abs_shap": [9.0, 4.0, 1.0]}
    )
    _, ax = c.viz.plot_shap_summary_bar(top_n=2)
    assert [t.get_text() for t in ax.get_yticklabels()] == ["alpha", "gamma"]
    assert_allclose([p.get_width() for p in ax.patches], [4, 9])


def test_model_inspector_selected_names_preserve_importance_associations(tools_case):
    c = tools_case
    core = SimpleNamespace(feature_importances_=np.array([0.2, 0.8]))
    pipe = SimpleNamespace(named_steps={"clf": core})
    clf = SimpleNamespace(
        model_pipeline=pipe,
        feature_names_used=["discarded", "calcium", "hydrogen"],
        selected_features_=["calcium", "hydrogen"],
        class_labels=["M", "F", "G"],
        model_type="synthetic",
    )
    c.monkeypatch.setattr(
        c.module.SpectralClassifier, "load_model", staticmethod(lambda path: clf)
    )
    c.viz._analyze_saved_model("trusted-synthetic-boundary")
    ax = c.plt.gcf().axes[0]
    assert [t.get_text() for t in ax.get_yticklabels()] == ["calcium", "hydrogen"]
    assert_allclose([p.get_width() for p in ax.patches], [0.2, 0.8])
    assert any("M, F, G" in str(getattr(obj, "data", "")) for obj in c.displayed)


def test_feature_explorer_eda_real_rf_and_exports(tools_case):
    c = tools_case
    path = Path(c.paths["PROCESSED_DIR"]) / "features_test.csv"
    data = pd.DataFrame(
        {
            "alpha": np.arange(24),
            "beta": np.arange(24) * -2.0,
            "constant": 7,
            "source_id": [f"object-{n}" for n in range(24)],
            "subclass": ["F"] * 12 + ["G"] * 12,
        }
    )
    data.loc[0, "alpha"] = np.nan
    data.to_csv(path, index=False)
    result = c.viz.feature_explorer(
        str(path), save_dir=str(c.tmp / "eda"), rf_estimators=8, do_permutation=True
    )
    pd.testing.assert_frame_equal(result["df"], data)
    rf = result["rf_importances"]
    assert set(rf.feature) == {"alpha", "beta"}
    assert_allclose(rf.importance.sum(), 1.0)
    assert set(result["perm_importances"].feature) == {"alpha", "beta"}
    pd.testing.assert_frame_equal(pd.read_csv(c.tmp / "eda/rf_importances.csv"), rf)
    axes = [ax for num in c.plt.get_fignums() for ax in c.plt.figure(num).axes]
    corr = next(ax for ax in axes if ax.images)
    # Independent Pearson reference, including the documented median fill:
    # the 23 finite alpha values are 1..23, whose median is 12.
    x = [12.0] + list(range(1, 24))
    y = [-2.0 * i for i in range(24)]
    mx, my = math.fsum(x) / 24, math.fsum(y) / 24
    r = math.fsum((a - mx) * (b - my) for a, b in zip(x, y)) / math.sqrt(
        math.fsum((a - mx) ** 2 for a in x) * math.fsum((b - my) ** 2 for b in y)
    )
    assert_allclose(corr.images[0].get_array(), [[1, r], [r, 1]], atol=1e-14)
    assert {
        "feature_hists.png",
        "corr_heatmap.png",
        "rf_importances_top25.png",
    }.issubset({p.name for p in (c.tmp / "eda").iterdir()})


def test_feature_explorer_no_labels_and_missing_files(tools_case):
    c = tools_case
    path = Path(c.paths["PROCESSED_DIR"]) / "features_test.csv"
    pd.DataFrame({"alpha": [1, 2, 3], "beta": [2, 4, 6]}).to_csv(path, index=False)
    result = c.viz.feature_explorer(str(path), max_hists=0, corr_top_n=0)
    assert result["rf_importances"] is None and result["label_col"] is None
    assert c.viz.feature_explorer(str(c.tmp / "absent*.csv")) is None


def test_beeswarm_real_explanation_and_explicit_fallback(tools_case):
    c = tools_case
    c.viz._last_shap_explanation = c.module.shap.Explanation(
        values=np.array([[1.0, 2], [-2.0, 1], [3.0, -1]]),
        data=np.array([[1.0, 2], [2.0, 4], [3.0, 5]]),
        feature_names=["alpha", "beta"],
    )
    fig, ax = c.viz.plot_shap_beeswarm()
    assert fig is ax.figure and ax.collections

    def unavailable(*args, **kwargs):
        raise ValueError("controlled unsupported explanation")

    c.monkeypatch.setattr(c.module.shap.plots, "beeswarm", unavailable)
    c.monkeypatch.setattr(c.module.shap.plots, "bar", unavailable)
    c.viz._last_shap_importances = pd.DataFrame(
        {"feature": ["beta", "alpha"], "mean_abs_shap": [3.0, 1.0]}
    )
    _, ax = c.viz.plot_shap_beeswarm()
    assert_allclose([p.get_width() for p in ax.patches], [1, 3])


@pytest.mark.parametrize("mode", ["tree", "linear", "kernel", "auto"])
def test_explainer_factory_real_small_models(tools_case, mode):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    c = tools_case
    X = pd.DataFrame({"alpha": [0.0, 1.0, 2.0, 3.0], "beta": [0.0, 1.0, 4.0, 9.0]})
    model = (
        LogisticRegression()
        if mode == "linear"
        else RandomForestClassifier(
            n_estimators=3, max_depth=2, random_state=42, n_jobs=1
        )
    ).fit(X.values, ["F", "F", "G", "G"])
    pipe = Pipeline([("clf", model)])
    explainer, background = c.viz._build_shap_explainer(
        pipe=pipe, model=model, X_df=X, mode=mode
    )
    assert_allclose(background, X.values)
    if mode == "kernel":
        values = explainer.shap_values(X.values[:1], nsamples=4, silent=True)
    else:
        values = explainer(X.values[:1]).values
    assert np.asarray(values).shape[:2] == (1, 2)
    assert np.isfinite(values).all()
    with pytest.raises(ValueError, match="Unknown SHAP mode"):
        c.viz._build_shap_explainer(pipe=pipe, model=model, X_df=X, mode="invalid")


def test_real_raw_feature_sample_contains_analytic_absorption_ew(tools_case):
    c = tools_case
    # Raw sampling uses PeakDetector's default prominence .85.
    c.gaussian(amplitude=0.95)
    c.viz.refresh_available_spectra()
    first = c.viz._get_features_sample(SimpleNamespace(), 5, "raw_fits")
    second = c.viz._get_features_sample(SimpleNamespace(), 5, "raw_fits")
    pd.testing.assert_frame_equal(first, second)
    assert first.shape[0] == 1 and first.shape[1] > 6
    assert (
        abs(first.loc[0, "feature_Hβ_eq_width"] - 0.95 * 0.8 * np.sqrt(2 * np.pi))
        < 0.002
    )
    assert (
        abs(first.loc[0, "feature_Hβ_fwhm"] - 2 * np.sqrt(2 * np.log(2)) * 0.8) < 0.23
    )


def test_feature_importance_bar_preserves_arbitrary_named_values(tools_case):
    c = tools_case
    table = pd.DataFrame(
        {"feature": ["beta", "gamma", "alpha"], "importance": [0.7, 0.2, 0.1]}
    )
    before = table.copy(deep=True)
    _, ax = c.viz._plot_feature_importances_bar(
        table, top_n=2, save_path=str(c.tmp / "figures/bar.png")
    )
    assert [t.get_text() for t in ax.get_yticklabels()] == ["gamma", "beta"]
    assert_allclose([p.get_width() for p in ax.patches], [0.2, 0.7])
    pd.testing.assert_frame_equal(table, before)


def test_small_class_split_may_refuse_explicitly_without_false_importances(tools_case):
    c = tools_case
    path = Path(c.paths["PROCESSED_DIR"]) / "features_synthetic.csv"
    pd.DataFrame(
        {
            "alpha": range(6),
            "beta": [1.0, 4.0, 2.0, 7.0, 3.0, 8.0],
            "subclass": ["A", "A", "B", "B", "C", "C"],
        }
    ).to_csv(path, index=False)
    try:
        result = c.viz.feature_explorer(
            str(path), max_hists=0, corr_top_n=0, rf_estimators=3, do_permutation=False
        )
    except ValueError as error:
        assert ("test_size" in str(error) and "classes" in str(error)) or (
            "independent holdout" in str(error).lower()
            and "identit" in str(error).lower()
        )
        assert getattr(c.viz, "_last_rf_importances", None) is None
    else:
        assert result is not None


def test_identical_catalog_duplicates_keep_unrelated_labels(tools_case):
    c = tools_case
    c.catalog(
        {
            "fits_name": ["A/same.fits.gz", "A/same.fits.gz", "B/other.fits.gz"],
            "subclass": ["F5", "F5", "G2"],
        }
    )
    assert c.viz._load_labels_catalog() == {"A/same": "F5", "B/other": "G2"}


@pytest.mark.parametrize("select", [False, True])
def test_real_pipeline_reference_matches_analytic_values_and_inference(
    tools_case, select
):
    c = tools_case
    wrapper, query, expected, names = contracts.preparation_case(c, select)
    actual = wrapper.model_pipeline[:-1].transform(query)
    assert_allclose(actual, expected, rtol=0, atol=2e-14)
    assert len(names) == actual.shape[1]


@pytest.mark.parametrize(
    "text,admitted",
    [
        (
            "Conditional frequency; displayed=12; total=15; excluded=3; denominator=displayed",
            True,
        ),
        ("Conditional frequency: 12 of 15 examples; 3 excluded", True),
        ("Top 3 sub-classes, LAMOST DR5 (2015)", False),
        ("Conditional frequency; displayed=15; total=12; excluded=3", False),
        ("Frequency of total catalogue; total=15; denominator=total", True),
    ],
)
def test_percentage_oracle_requires_semantic_denominator(tools_case, text, admitted):
    c = tools_case
    _, ax = c.plt.subplots()
    total_mode = text.startswith("Frequency of total")
    ax.bar(["A", "B", "C"], np.array([5, 4, 3]) * 100 / (15 if total_mode else 12))
    ax.set_title(text)
    if admitted:
        contracts.assert_population_percentages(ax, [5, 4, 3, 2, 1], 3)
    else:
        with pytest.raises(AssertionError):
            contracts.assert_population_percentages(ax, [5, 4, 3, 2, 1], 3)


@pytest.mark.parametrize("mode", ["mean", "sum", "per_class"])
def test_multiclass_oracle_accepts_declared_alternatives(mode):
    if mode == "per_class":
        table = pd.DataFrame(
            {
                "feature": ["alpha", "beta", "gamma"] * 2,
                "class": ["F"] * 3 + ["G"] * 3,
                "mean_abs_shap": [4.0, 25.0, 250.0, 6.0, 50.0, 500.0],
            }
        )
    else:
        table = pd.DataFrame(
            {
                "feature": ["alpha", "beta", "gamma"],
                (
                    "mean_sample_sum_class_abs_shap"
                    if mode == "sum"
                    else "mean_abs_shap"
                ): np.array([5.0, 37.5, 375.0])
                * (2 if mode == "sum" else 1),
            }
        )
        table.attrs["class_reduction"] = mode
    contracts.assert_multiclass_importances(table)


def test_partition_oracle_accepts_groups_and_explicit_descriptive_limit():
    contracts.assert_eda_partitions([{"A", "B"}], [{"C"}], "")
    contracts.assert_eda_partitions(
        [], [], "No independent holdout: object identities insufficient"
    )
    contracts.assert_eda_partitions(
        [{"A"}],
        [{"A"}],
        "Descriptive only: not an independent holdout because observations repeat",
        {
            "metric": "Permutation importance (descriptive)",
            "CSV": "permutation_importances.csv, descriptive_context=non-independent",
        },
    )
    with pytest.raises(AssertionError, match="contaminate"):
        contracts.assert_eda_partitions([{"A"}], [{"A"}], "Validation importance")


@pytest.mark.parametrize(
    "surface",
    [
        "Permutation importance (split de validation)",
        "Calculate permutation importance (validation split)",
        "permutation_importances.csv,feature,perm_importance",
        "Independent holdout evaluation",
    ],
)
def test_partition_oracle_rejects_cosmetic_disclaimer(surface):
    with pytest.raises(AssertionError, match="Contradictory|Unqualified"):
        contracts.assert_eda_partitions(
            [{"A"}],
            [{"A"}],
            "Note: descriptive EDA, not an independent holdout.",
            {"exposed": surface},
        )


@pytest.mark.parametrize("same_basename", [True, False])
def test_curve_identity_oracle_checks_actual_flux_and_allows_relative_paths(
    tools_case, same_basename
):
    c = tools_case
    a, wa, fa = c.write("A/same.fits.gz", [1, 2, 3])
    b, wb, fb = c.write(
        "B/same.fits.gz" if same_basename else "B/other.fits.gz", [4, 5, 6]
    )
    _, ax = c.plt.subplots()
    ax.plot(wb, fb, label=b)
    ax.plot(wa, fa, label=a)
    expected = {a: (wa, fa), b: (wb, fb)}
    contracts.assert_curve_identities(ax, [a, b], expected)
    ax.lines[0].set_label(a)
    ax.lines[1].set_label(b)
    with pytest.raises(AssertionError, match="Curve data"):
        contracts.assert_curve_identities(ax, [a, b], expected)


@pytest.mark.parametrize("mode", ["mean", "sum", "per_class"])
def test_multiclass_oracle_checks_table_metadata_and_actual_bars(tools_case, mode):
    c = tools_case
    classes = ["F", "G"] if mode == "per_class" else [None]
    column = "mean_sample_sum_class_abs_shap" if mode == "sum" else "mean_abs_shap"
    rows = []
    _, axes = c.plt.subplots(1, len(classes), squeeze=False)
    for cls, ax in zip(classes, axes.ravel()):
        values = {"F": [4, 25, 250], "G": [6, 50, 500], None: [5, 37.5, 375]}[cls]
        if mode == "sum":
            values = np.array(values) * 2
        for feature, value in zip(["alpha", "beta", "gamma"], values):
            rows.append(
                {"feature": feature, column: value, **({"class": cls} if cls else {})}
            )
        ax.barh(["alpha", "beta", "gamma"], values)
        ax.set_xlabel(
            "mean_sample_sum_class_abs_SHAP" if mode == "sum" else "mean(|SHAP value|)"
        )
        ax.set_title(f"SHAP class {cls}" if cls else "SHAP importances")
    table = pd.DataFrame(rows)
    table.attrs["class_reduction"] = mode
    contracts.assert_multiclass_importances(table, axes)
    if mode == "per_class":
        contracts.assert_multiclass_importances(table, axes.ravel()[::-1])
    if mode != "per_class":
        table.attrs["class_reduction"] = "mean" if mode == "sum" else "sum"
        with pytest.raises(AssertionError, match="metadata contradict"):
            contracts.assert_multiclass_importances(table, axes)
        table.attrs["class_reduction"] = mode
    ax = axes.ravel()[0]
    ax.set_yticks(range(3), ["beta", "alpha", "gamma"])
    with pytest.raises(AssertionError, match="bars have permuted"):
        contracts.assert_multiclass_importances(table, axes)
    ax.set_yticks(range(3), ["alpha", "beta", "gamma"])
    ax.set_xlabel("mean(|SHAP value|)" if mode == "sum" else "sum_class_abs_SHAP")
    with pytest.raises(AssertionError, match="statistic"):
        contracts.assert_multiclass_importances(table, axes)


def test_pipeline_peak_analysis_uses_cleaned_coordinates(tools_case):
    c = tools_case
    _, wave, flux = c.gaussian(invalid=True)
    match = c.module.PeakDetector(prominence=0.2, window=8).analyze_spectrum(wave, flux)
    assert_allclose(match["Hβ"][0], 4861.0, atol=2e-9, rtol=0)


def test_notebook_point_metric_csv_is_independent_of_latent_sigma(tools_case):
    c = tools_case
    name, _, _ = c.gaussian(continuum=6.0)
    original = c.module.safe_sigma_from_invvar
    exported = []
    for factor in [1 / 6, 100.0]:
        c.monkeypatch.setattr(
            c.module, "safe_sigma_from_invvar", lambda inv, f=factor: original(inv) * f
        )
        c.displayed.clear()
        contracts.render_spectrum(c, "notebook", name)
        links = " ".join(str(getattr(obj, "data", "")) for obj in c.displayed)
        exported.append(
            base64.b64decode(re.search(r"data:text/csv;base64,([^']+)", links)[1])
        )
        c.plt.close("all")
    assert exported[0] == exported[1]
