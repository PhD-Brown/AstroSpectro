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
def test_nonfinite_prefix_keeps_original_peak_pixels(tools_case, route):
    c = tools_case
    name, wave, flux = c.gaussian(invalid=True)
    red, green, _ = contracts.render_spectrum(c, route, name)
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


def test_truncated_percentages_disclose_conditional_population(tools_case):
    c = tools_case
    c.catalog({"subclass": ["A"] * 5 + ["B"] * 4 + ["C"] * 3 + ["D"] * 2 + ["E"]})
    _, ax = c.viz.plot_subclass_distribution(top_n=3, normalize=True)
    values = [p.get_height() for p in ax.patches]
    total = np.array([5, 4, 3]) * 100 / 15
    conditional = np.array([5, 4, 3]) * 100 / 12
    text = " ".join(
        [ax.get_title(), ax.get_ylabel()] + [t.get_text() for t in ax.texts]
    ).lower()
    if np.allclose(values, total):
        return  # percentages of total; omitted classes retain their mass
    assert_allclose(values, conditional)
    assert (
        ("top" in text or "display" in text or "condition" in text)
        and "15" in text
        and "3" in text
    ), "Conditional percentages hide that 3/15 examples are outside the displayed top N"


@pytest.mark.parametrize("flux", [[-3.0, -2.0, -1.0], [-1.0, 0.0, 1.0]])
def test_failed_normalization_has_truthful_curve_units(tools_case, flux):
    c = tools_case
    name, _, _ = c.write(flux=flux)
    with pytest.warns(UserWarning, match="un-standardised"):
        result = c.viz._plot_spectra_comparison([name], normalize=True)
    if result is None:
        return
    _, ax = result
    assert (
        "normalis" not in ax.get_ylabel().lower()
        and "normaliz" not in ax.get_ylabel().lower()
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
    assert len(spectra) == 1
    # Raw invvar=4 -> sigma=.5; dividing flux by median=6 divides sigma too.
    assert_allclose(
        spectra[0]["uncertainty"].array,
        np.full(401, 1 / 12),
        rtol=1e-12,
        err_msg="Normalized flux retains raw uncertainty units",
    )


def test_absorption_fwhm_export_measures_the_line_not_continuum(tools_case):
    c = tools_case
    name, _, _ = c.gaussian()
    contracts.render_spectrum(c, "notebook", name)
    links = " ".join(str(getattr(x, "data", "")) for x in c.displayed)
    csv = base64.b64decode(
        re.search(r"data:text/csv;base64,([^']+)", links)[1]
    ).decode()
    table = pd.read_csv(io.StringIO(csv), index_col=0)
    expected = 2 * np.sqrt(2 * np.log(2)) * 0.8
    measured = float(table.loc["Hβ", "Largeur FWHM (Å)"])
    # Pixel spacing .112 A; allow two pixels plus .005 A display rounding.
    assert (
        abs(measured - expected) <= 0.23
    ), f"Gaussian FWHM expected {expected}, exported {measured} A"


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


def test_actual_classifier_prep_step_is_applied_before_shap(tools_case):
    from sklearn.pipeline import Pipeline

    c = tools_case
    classifier = c.module.SpectralClassifier()
    train = pd.DataFrame({"alpha": [1.0, 3.0, 5.0], "gamma": [10.0, 20.0, 30.0]})
    prep = classifier._build_preprocessor(train, "SVM").fit(train)
    wrapper = SimpleNamespace(
        model_pipeline=Pipeline([("prep", prep)]),
        feature_names_used=list(train.columns),
    )
    query = pd.DataFrame({"gamma": [30.0, np.nan], "alpha": [5.0, 1.0]})
    Xt, _ = c.viz._prepare_for_model(query, wrapper)
    assert_allclose(
        Xt,
        [[np.sqrt(1.5), np.sqrt(1.5)], [-np.sqrt(1.5), 0]],
        atol=1e-14,
        err_msg="Visualizer bypassed classifier prep transformation",
    )


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
    assert set(result.feature) == {
        "alpha",
        "beta",
        "gamma",
    }, "One feature is lost after reducing the feature axis"
    # Eight magnitudes per feature: sums 40,300,3000, divided by 8.
    assert_allclose(
        result.set_index("feature").loc[["alpha", "beta", "gamma"], "mean_abs_shap"],
        [5, 37.5, 375],
    )


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
    a, _, _ = c.write("A/same.fits.gz", [1, 2, 3])
    b, _, _ = c.write("B/same.fits.gz", [4, 5, 6])
    _, ax = c.viz._plot_spectra_comparison([a, b], normalize=False)
    labels = [line.get_label() for line in ax.lines]
    assert (
        len(set(labels)) == 2
    ), f"Distinct observations share indistinguishable labels: {labels}"


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
    result = c.viz._run_shap_analysis("synthetic", explainer_mode="invalid-mode")
    assert (
        result is None and not calls
    ), "Analysis ignores explainer_mode and unconditionally computes TreeExplainer"


def test_basename_export_is_written_in_current_directory(tools_case):
    c = tools_case
    c.monkeypatch.chdir(c.tmp)
    c.viz._last_shap_importances = pd.DataFrame(
        {"feature": ["alpha"], "mean_abs_shap": [1.0]}
    )
    c.viz.plot_shap_summary_bar(save_path="summary.png")
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


def test_feature_explorer_does_not_split_repeat_observations_across_holdout(tools_case):
    from sklearn import model_selection

    c = tools_case
    path = Path(c.paths["PROCESSED_DIR"]) / "features_synthetic.csv"
    pd.DataFrame(
        {
            "source_id": np.repeat(np.arange(12), 2),
            "alpha": np.arange(24),
            "beta": np.sin(np.arange(24)),
            "subclass": np.repeat(["F"] * 6 + ["G"] * 6, 2),
        }
    ).to_csv(path, index=False)
    table = pd.read_csv(path)
    real_split = model_selection.train_test_split
    partitions = []

    def observe(X, y, **kwargs):
        result = real_split(X, y, **kwargs)
        partitions.append((result[0].index, result[1].index))
        return result

    c.monkeypatch.setattr(model_selection, "train_test_split", observe)
    c.viz.feature_explorer(
        str(path), max_hists=0, corr_top_n=0, rf_estimators=3, do_permutation=False
    )
    assert len(partitions) == 1
    train, test = partitions[0]
    overlap = set(table.loc[train, "source_id"]) & set(table.loc[test, "source_id"])
    assert (
        not overlap
    ), f"Repeated object observations contaminate the holdout: {sorted(overlap)}"
