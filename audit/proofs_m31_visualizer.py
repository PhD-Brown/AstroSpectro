"""Explicit red contracts, outside normal pytest collection; no xfail.

Run: python -m pytest -q audit/proofs_m31_visualizer.py --tb=short
Failures are manifestations, not a count of independent defects.
"""

import importlib.util
import io
import base64
from pathlib import Path
import random
import re
import warnings
from types import SimpleNamespace

import matplotlib
import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest

_spec = importlib.util.spec_from_file_location(
    "m31_contracts",
    Path(__file__).resolve().parents[1] / "tests/m31_visualizer_contracts.py",
)
contracts = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(contracts)
tools_case = contracts.tools_case


@pytest.mark.parametrize("route", ["matplotlib", "notebook", "plotly"])
def test_nonfinite_prefix_keeps_original_peak_pixels(tools_case, route, capsys):
    c = tools_case
    name, wave, flux = c.gaussian(invalid=True)
    try:
        red, green, target = contracts.render_spectrum(
            c, route, name, allow_rejection=True
        )
    except ValueError as error:
        assert any(word in str(error).lower() for word in ["finite", "nan", "invalid"])
        return
    if target is None:
        diagnostic = capsys.readouterr().out + " ".join(c.st.errors)
        assert any(
            word in diagnostic.lower() for word in ["finite", "nan", "invalid"]
        ), "Rejected data without identifying the invalid input"
        return
    contracts.assert_peak_pairs(red, green, wave, flux)


def test_ra_wrap_centroid_stays_near_neighboring_directions(tools_case):
    c = tools_case
    c.write("p/a.fits.gz", header={"RA": 359.0, "DEC": 0.0})
    c.write("p/b.fits.gz", header={"RA": 1.0, "DEC": 0.0})
    c.viz.refresh_available_spectra()
    row = c.viz._generate_position_catalog().iloc[0]
    # Equal weights, equal declinations: circular and spherical means both
    # point to 0 deg. Do not prescribe either implementation.
    separation = abs((float(row.ra) + 180) % 360 - 180)
    assert (
        separation < 1e-10
    ), f"RA centroid is {row.ra} deg, opposite the two observations"


def test_sky_ra_tick_labels_match_actual_transformed_points(tools_case):
    c = tools_case
    c.monkeypatch.setattr(
        c.viz,
        "_generate_position_catalog",
        lambda: pd.DataFrame(
            {
                "ra": [0.0, 90.0, 180.0],
                "dec": [0.0, 0.0, 0.0],
                "spectra_count": [1, 1, 1],
            }
        ),
    )
    _, ax = c.viz.plot_sky_coverage(save_path=False)
    offsets = np.asarray(ax.collections[0].get_offsets())
    for (x, _), ra in zip(offsets, [0, 90, 180]):
        tick = int(np.argmin(abs(ax.get_xticks() - x)))
        assert_allclose(ax.get_xticks()[tick], x, atol=1e-14)
        label = ax.get_xticklabels()[tick].get_text()
        hour = float(re.search(r"[-+]?\d+(?:\.\d+)?", label)[0])
        assert hour % 24 == ra / 15, f"RA={ra} deg drawn at tick {label}"


@pytest.mark.parametrize("counts,top_n", [([5, 4, 3, 2, 1], 3), ([11, 7, 4, 3, 2], 2)])
def test_truncated_percentages_disclose_conditional_population(
    tools_case, counts, top_n
):
    c = tools_case
    c.catalog(
        {"subclass": [label for label, n in zip("ABCDE", counts) for _ in range(n)]}
    )
    _, ax = c.viz.plot_subclass_distribution(top_n=top_n, normalize=True)
    contracts.assert_population_percentages(ax, counts, top_n)


@pytest.mark.parametrize("flux", [[-3.0, -2.0, -1.0], [-1.0, 0.0, 1.0]])
def test_failed_normalization_has_truthful_curve_units(tools_case, flux):
    c = tools_case
    name, _, _ = c.write(flux=flux)
    try:
        with warnings.catch_warnings(record=True):
            result = c.viz._plot_spectra_comparison([name], normalize=True)
    except ValueError as error:
        assert any(word in str(error).lower() for word in ["median", "flux", "normal"])
        return
    if result is None:
        return
    _, ax = result
    label = ax.get_ylabel().lower()
    assert (
        "raw" in label
        or "mixed" in label
        or ("normalis" not in label and "normaliz" not in label)
    ), "Raw flux is labelled as normalized after normalization refusal"


def test_normalized_flux_uncertainties_scale_with_flux(tools_case):
    c = tools_case
    name, _, flux = c.gaussian()
    c.write(name, flux * 6, start=4861.0 / (10.0**0.00001) ** 200, step=0.00001)
    spectra = []
    real_spectrum = c.module.Spectrum

    def observe(**kwargs):
        spectra.append(kwargs)
        return real_spectrum(**kwargs)

    c.monkeypatch.setattr(c.module, "Spectrum", observe)
    contracts.render_spectrum(c, "notebook", name)
    # Raw invvar=4 -> sigma=.5; dividing flux by median=6 divides sigma too.
    # Omitting latent uncertainties is admissible if none are advertised.
    for spec in spectra:
        if spec.get("uncertainty") is not None:
            assert_allclose(
                spec["uncertainty"].array,
                np.full(401, 1 / 12),
                rtol=1e-12,
                err_msg="Normalized flux retains raw uncertainty units",
            )


@pytest.mark.parametrize("sigma", [0.5, 0.8, 1.6])
@pytest.mark.parametrize("window", [8, 12])
def test_absorption_fwhm_export_measures_the_line_not_continuum(
    tools_case, sigma, window
):
    c = tools_case
    name, wave, _ = c.gaussian(sigma=sigma)
    contracts.render_spectrum(c, "notebook", name, window=window)
    links = " ".join(str(getattr(x, "data", "")) for x in c.displayed)
    csv = base64.b64decode(
        re.search(r"data:text/csv;base64,([^']+)", links)[1]
    ).decode()
    table = pd.read_csv(io.StringIO(csv), index_col=0)
    expected = 2 * np.sqrt(2 * np.log(2)) * sigma
    measured = float(table.loc["Hβ", "Largeur FWHM (Å)"])
    # Smooth-profile moment/interpolated crossings have second-order grid
    # error. Bound it by FWHM*(h/sigma)^2/6, add two-decimal rounding and
    # the analytic second-moment truncation of a Gaussian at +/-window.
    import math

    a = window / sigma
    tail = expected * (
        1
        - np.sqrt(
            1
            - 2 * a * np.exp(-a * a / 2) / np.sqrt(2 * np.pi) / math.erf(a / np.sqrt(2))
        )
    )
    h = np.max(np.diff(wave))
    tolerance = 0.005 + expected * (h / sigma) ** 2 / 6 + 2 * tail
    assert (
        abs(measured - expected) <= tolerance
    ), f"Gaussian sigma={sigma}, window={window}: FWHM expected {expected}, exported {measured} A; tolerance={tolerance}"


def test_selected_feature_names_derive_from_support_not_prefix(tools_case):
    from sklearn.feature_selection import VarianceThreshold
    from sklearn.pipeline import Pipeline

    c = tools_case
    train = pd.DataFrame(
        {
            "constant": [7.0, 7.0, 7.0],
            "calcium": [1.0, 2.0, 3.0],
            "hydrogen": [10.0, 20.0, 30.0],
        }
    )
    selector = VarianceThreshold().fit(train)
    wrapper = SimpleNamespace(
        model_pipeline=Pipeline([("feature_selector", selector)]),
        feature_names_used=list(train.columns),
    )
    Xt, names = c.viz._prepare_for_model(train, wrapper)
    assert Xt.shape == (3, 2)
    assert names == ["calcium", "hydrogen"], f"Selected values mislabelled: {names}"


@pytest.mark.parametrize("select", [False, True])
def test_actual_classifier_prep_step_is_applied_before_shap(tools_case, select):
    c = tools_case
    wrapper, query, expected, expected_names = contracts.preparation_case(c, select)
    before = query.copy(deep=True)
    Xt, names = c.viz._prepare_for_model(query, wrapper)
    assert (
        names == expected_names
    ), f"Transformed feature identities wrong: {names} vs {expected_names}"
    assert_allclose(
        Xt,
        expected,
        atol=2e-14,
        err_msg="Visualizer bypassed classifier prep transformation",
    )
    pd.testing.assert_frame_equal(query, before)


def test_multiclass_shap_reduces_samples_and_classes_only(tools_case):
    c = tools_case
    values = np.array(
        [
            [[1, 3], [10, 20], [100, 200]],
            [[-3, 5], [-20, 40], [-200, 400]],
            [[5, -7], [30, -60], [300, -600]],
            [[-7, 9], [-40, 80], [-400, 800]],
        ],
        float,
    )
    contracts.shap_case(c, values)
    result = c.viz._run_shap_analysis("synthetic")
    contracts.assert_multiclass_importances(result)
    explanation = c.viz._last_shap_explanation
    assert explanation.values.shape == (
        4,
        3,
        2,
    ), "SHAP axes are not samples/features/classes"
    assert list(explanation.feature_names) == ["alpha", "beta", "gamma"]
    assert_allclose(explanation.values, values)
    plotted = c.viz.plot_shap_summary_bar(top_n=20)
    assert plotted is not None, "Computed SHAP statistic has no summary representation"
    contracts.assert_multiclass_importances(result, plotted[1])


def test_shap_explanation_is_computed_once(tools_case):
    c = tools_case
    calls, _ = contracts.shap_case(c, np.ones((4, 3)))
    result = c.viz._run_shap_analysis("synthetic")
    assert result is not None
    assert len(calls) == 1, f"Same explanation evaluated {len(calls)} times"


def test_peak_plot_restores_global_matplotlib_style(tools_case):
    c = tools_case
    name, _, _ = c.gaussian()
    c.plt.style.use("default")
    before = dict(matplotlib.rcParams)
    contracts.render_spectrum(c, "matplotlib", name)
    changed = [
        key
        for key in before
        if key != "backend" and str(before[key]) != str(matplotlib.rcParams[key])
    ]
    assert not changed, f"AstroVisualizer changed caller style keys: {changed}"


@pytest.mark.parametrize("method", ["zeros", "normalization"])
def test_sampling_preserves_global_random_state(tools_case, method):
    c = tools_case
    for i in range(3):
        c.write(f"p/{i}.fits.gz")
    c.viz.refresh_available_spectra()
    random.seed(912)
    before = random.getstate()
    if method == "zeros":
        c.viz.analyze_feature_zeros(n_files=0, export=False)
    else:
        c.viz.plot_normalization_comparison(random_state=42, save_path="")
    assert random.getstate() == before, "Visualizer reseeded caller's global RNG"


def test_added_backdated_fits_invalidates_position_cache(tools_case):
    import os

    c = tools_case
    name, _, _ = c.write("p/a.fits.gz")
    c.viz.refresh_available_spectra()
    c.viz._generate_position_catalog()
    new, _, _ = c.write("p/b.fits.gz")
    original = Path(c.paths["RAW_DATA_DIR"]) / name
    os.utime(Path(c.paths["RAW_DATA_DIR"]) / new, (original.stat().st_mtime - 20,) * 2)
    c.viz.refresh_available_spectra()
    result = c.viz._generate_position_catalog()
    assert (
        result.spectra_count.sum() == 2
    ), "Cache ignored a newly added FITS with preserved timestamp"


def test_duplicate_basename_legend_preserves_spectrum_identity(tools_case):
    c = tools_case
    a, wa, fa = c.write("A/same.fits.gz", [1, 2, 3])
    b, wb, fb = c.write("B/same.fits.gz", [4, 5, 6])
    _, ax = c.viz._plot_spectra_comparison([a, b], normalize=False)
    contracts.assert_curve_identities(ax, [a, b], {a: (wa, fa), b: (wb, fb)})


def test_duplicate_conflicting_catalog_labels_are_not_silently_overwritten(
    tools_case, capsys
):
    c = tools_case
    c.catalog({"fits_name": ["A/same.fits.gz"] * 2, "subclass": ["F5", "G2"]})
    capsys.readouterr()
    labels = c.viz._load_labels_catalog()
    diagnostic = capsys.readouterr().out.lower()
    assert "A/same" not in labels or any(
        word in diagnostic for word in ["duplicate", "conflict", "ambig"]
    ), "Conflicting catalogue rows silently overwrite an object's class"


def test_features_csv_missing_measurement_remains_missing_until_imputation(tools_case):
    c = tools_case
    pd.DataFrame({"alpha": [1.0, 2.0], "label": ["F", "G"]}).to_csv(
        Path(c.paths["PROCESSED_DIR"]) / "features_synthetic.csv", index=False
    )
    df = c.viz._get_features_sample(
        SimpleNamespace(feature_names_used=["alpha", "missing"]), 4, "features_csv"
    )
    assert (
        df["missing"].isna().all()
    ), "An absent measurement becomes a physical zero before trained imputation"


def test_model_inspector_missing_selection_metadata_does_not_guess_prefix(tools_case):
    c = tools_case
    model = SimpleNamespace(feature_importances_=np.array([0.2, 0.8]))
    clf = SimpleNamespace(
        model_pipeline=SimpleNamespace(named_steps={"clf": model}),
        feature_names_used=["discarded", "calcium", "hydrogen"],
    )
    c.monkeypatch.setattr(
        c.module.SpectralClassifier, "load_model", staticmethod(lambda path: clf)
    )
    c.viz._analyze_saved_model("synthetic")
    axes = [ax for num in c.plt.get_fignums() for ax in c.plt.figure(num).axes]
    assert not any(
        ax.patches for ax in axes
    ), "Model importances are assigned to a guessed prefix without selection provenance"


def test_explainer_mode_is_respected_by_analysis(tools_case):
    c = tools_case
    calls, _ = contracts.shap_case(c, np.ones((4, 3)))
    try:
        result = c.viz._run_shap_analysis("synthetic", explainer_mode="invalid-mode")
    except ValueError as error:
        assert "mode" in str(error).lower() or "explainer" in str(error).lower()
        assert not calls
        return
    assert (
        result is None and not calls
    ), "Analysis ignores explainer_mode and unconditionally computes TreeExplainer"


@pytest.mark.parametrize(
    "route",
    ["summary", "subclass", "comparison", "normalization", "feature_bar", "beeswarm"],
)
def test_basename_export_is_written_in_current_directory(tools_case, route):
    c = tools_case
    c.monkeypatch.chdir(c.tmp)
    c.viz._last_shap_importances = pd.DataFrame(
        {"feature": ["alpha"], "mean_abs_shap": [1.0]}
    )
    if route == "summary":
        c.viz.plot_shap_summary_bar(save_path="summary.png")
    elif route == "subclass":
        c.catalog({"subclass": ["F", "F", "G"]})
        c.viz.plot_subclass_distribution(save_path="summary.png")
    elif route in {"comparison", "normalization"}:
        a, _, _ = c.write("a/a.fits.gz")
        b, _, _ = c.write("b/b.fits.gz")
        c.viz.refresh_available_spectra()
        if route == "comparison":
            c.viz._plot_spectra_comparison([a, b], save_path="summary.png")
        else:
            c.viz.plot_normalization_comparison([a, b], save_path="summary.png")
    elif route == "feature_bar":
        c.viz._plot_feature_importances_bar(
            pd.DataFrame({"feature": ["alpha"], "importance": [1.0]}),
            save_path="summary.png",
        )
    else:
        c.viz._last_shap_explanation = c.module.shap.Explanation(
            values=np.array([[1.0, 2.0], [2.0, 1.0], [-1.0, 3.0]]),
            data=np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]]),
            feature_names=["alpha", "beta"],
        )
        c.viz.plot_shap_beeswarm(save_path="summary.png")
    assert (
        c.tmp / "summary.png"
    ).is_file(), "Valid basename export fails at makedirs('')"


def test_feature_explorer_nonfinite_histogram_has_explicit_partial_result(tools_case):
    c = tools_case
    path = Path(c.paths["PROCESSED_DIR"]) / "features_synthetic.csv"
    pd.DataFrame({"alpha": [1.0, np.inf, 3.0], "beta": [2.0, 4.0, 6.0]}).to_csv(
        path, index=False
    )
    try:
        result = c.viz.feature_explorer(str(path), corr_top_n=0)
    except (ValueError, TypeError) as error:
        # An explicit rejection identifying the offending feature is valid.
        assert "alpha" in str(error) and (
            "finite" in str(error) or "inf" in str(error)
        ), f"Histogram failed without identifying its invalid input feature: {error}"
        return
    assert (
        result is not None
    ), "Finite subset EDA should remain available with an explicit exclusion diagnostic"


@pytest.mark.parametrize("with_ids", [True, False])
def test_feature_explorer_does_not_split_repeat_observations_across_holdout(
    tools_case, with_ids, capsys
):
    c = tools_case
    path, fitted, evaluated = contracts.eda_partition_case(c, with_ids)
    diagnostic = ""
    result = None
    try:
        result = c.viz.feature_explorer(
            str(path), max_hists=0, corr_top_n=0, rf_estimators=3, do_permutation=True
        )
    except ValueError as error:
        diagnostic = str(error)
    text = (
        diagnostic
        + capsys.readouterr().out
        + " ".join(str(getattr(x, "data", "")) for x in c.displayed)
    )
    contracts.assert_eda_partitions(
        fitted, evaluated, text, contracts.eda_surfaces(c, result)
    )


def test_deleted_fits_invalidates_position_cache(tools_case):
    c = tools_case
    c.write("p1/a.fits.gz")
    rel, _, _ = c.write("p2/b.fits.gz")
    c.viz.refresh_available_spectra()
    c.viz._generate_position_catalog()
    synthetic = (Path(c.paths["RAW_DATA_DIR"]) / rel).resolve()
    assert synthetic.is_relative_to(c.tmp.resolve())
    synthetic.unlink()
    assert c.viz.refresh_available_spectra() == 1
    current = c.viz._generate_position_catalog()
    assert list(current.plan_id) == [
        "p1"
    ], "Deleted observation left a phantom plan in the position cache"


def test_missing_subclass_has_explicit_population_policy(tools_case, capsys):
    c = tools_case
    c.catalog({"subclass": ["A", "A", "B", None, None, None]})
    try:
        result = c.viz.plot_subclass_distribution(top_n=5)
    except (TypeError, ValueError) as error:
        assert any(
            word in str(error).lower() for word in ["subclass", "label", "missing"]
        ), f"Missing subclass caused an unqualified backend crash: {error}"
        return
    text = capsys.readouterr().out + " ".join(
        str(getattr(x, "data", "")) for x in c.displayed
    )
    if result is None:
        assert "missing" in text.lower() or "manquant" in text.lower()
        return
    _, ax = result
    counts = [p.get_height() for p in ax.patches]
    if sum(counts) == 6:
        assert sorted(counts) == [1, 2, 3]
    else:
        assert sorted(counts) == [1, 2]
        assert re.search(
            r"(?:excluded|exclu\w*)\s*[=:]?\s*3\b|\b3\s+(?:excluded|exclu\w*)",
            text.lower(),
        ), "Missing-label exclusions are not disclosed"


@pytest.mark.parametrize("route", ["sample", "zeros"])
def test_feature_consumers_keep_original_detected_wavelength(tools_case, route):
    c = tools_case
    start, step = 4861.0 / (10.0**0.0001) ** 200, 0.0001
    wave = start * np.power(10.0**step, np.arange(401))
    flux = 1 - 0.95 * np.exp(-0.5 * ((wave - 4861.0) / 3.0) ** 2)
    flux[150:160] = np.nan
    c.write(flux=flux, start=start, step=step)
    c.viz.refresh_available_spectra()
    observed = []
    original = c.module.FeatureEngineer.extract_features

    def extract(instance, matched, *args, **kwargs):
        # FeatureEngineer also initializes its schema using a dummy spectrum.
        # Observe the audited FITS by wavelength provenance, not call ordinal.
        if (
            args
            and np.asarray(args[0]).shape == wave.shape
            and np.allclose(args[0], wave, rtol=1e-13, atol=0)
        ):
            observed.append(matched)
        return original(instance, matched, *args, **kwargs)

    c.monkeypatch.setattr(c.module.FeatureEngineer, "extract_features", extract)
    if route == "sample":
        c.viz._get_features_sample(SimpleNamespace(), 1, "raw_fits")
    else:
        c.viz.analyze_feature_zeros(n_files=1, prominence=0.85, window=28, export=False)
    assert observed, "The audited FITS never reached feature extraction"
    for matched in observed:
        assert matched["Hβ"] is not None
        assert_allclose(
            matched["Hβ"][0],
            4861.0,
            atol=2e-9,
            rtol=0,
            err_msg="Feature consumer used a compacted index as an original pixel",
        )


@pytest.mark.parametrize("route", ["matplotlib", "notebook", "plotly"])
def test_nonfinite_normalization_refusal_has_truthful_spectrum_label(tools_case, route):
    c = tools_case
    name, _, flux = c.gaussian(continuum=6.0)
    flux[0] = np.nan
    c.write(name, flux, start=4861.0 / (10.0**0.00001) ** 200, step=0.00001)
    _, _, target = contracts.render_spectrum(c, route, name, allow_rejection=True)
    if target is None:
        return
    if route == "plotly":
        plotted = np.asarray(target.data[0].y)
        label = target.layout.yaxis.title.text.lower()
    else:
        plotted = target.lines[0].get_ydata()
        label = target.get_ylabel().lower()
    if np.isclose(np.nanmedian(plotted), 1.0, atol=1e-10):
        return
    assert (
        "raw" in label or "mixed" in label or "not normal" in label
    ), "Normalization refused, but the rendered raw spectrum claims normalized flux"


def test_missing_csv_features_follow_trained_preparation_at_explainer(tools_case):
    c = tools_case
    wrapper, _, _, names = contracts.preparation_case(c, True)
    pd.DataFrame({"alpha": [11.0, 1.0], "alpha_copy": [22.0, 2.0]}).to_csv(
        Path(c.paths["PROCESSED_DIR"]) / "features_missing.csv", index=False
    )
    c.monkeypatch.setattr(
        c.module.SpectralClassifier, "load_model", staticmethod(lambda path: wrapper)
    )
    observed = []

    class Explainer:
        def __call__(self, X):
            observed.append(np.asarray(X).copy())
            return c.module.shap.Explanation(values=np.zeros_like(X), data=X)

    c.monkeypatch.setattr(c.module.shap, "TreeExplainer", lambda model: Explainer())
    result = c.viz._run_shap_analysis("trusted-synthetic", data_source="features_csv")
    assert observed and result is not None
    delta = [0.2, 0.9, 0.1, 0.7, 0.4, 0.3]
    mean = sum(delta) / 6
    scale = (sum((v - mean) ** 2 for v in delta) / 6) ** 0.5
    expected = np.tile([0.0, (0.35 - mean) / scale], (2, 1))
    assert_allclose(
        observed[0],
        expected,
        atol=2e-14,
        err_msg="Missing CSV measurements bypassed fitted imputation/selection before explanation",
    )
    assert list(result.feature) == names
