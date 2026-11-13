"""Explicit M2.8 counterexamples; archived and removed from the green head.

No xfail and no production edits. Refusal, masking and unavailable outputs are
alternatives on invalid data; valid positive cases are in the green suite.
"""

import re
from types import SimpleNamespace

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pandas as pd
import pytest


def call_or_domain_refusal(call, domain):
    """Permit relevant explicit refusal, never accept incidental backend errors."""
    try:
        return call()
    except (ValueError, RuntimeError) as exc:
        message = str(exc).lower()
        assert any(word in message for word in domain), message
        assert not any(
            word in message
            for word in [
                "concatenate",
                "percentile",
                "0 columns",
                "singular data covariance",
            ]
        ), message
        return None
    except Exception as exc:
        pytest.fail(
            f"Incidental {type(exc).__name__} instead of explicit domain handling: {exc}"
        )


def gaussian_population(n=250):
    rng = np.random.default_rng(1729)
    Z = rng.normal(size=(n, 2)) * [2.0, 3.0]
    meta = pd.DataFrame({"teff_gspphot": np.arange(n, dtype=float) + 3500})
    return Z, np.array(["STAR"] * n), meta


@pytest.mark.parametrize(
    "case", ["gaussian", "analytic_normal", "mixture31", "mixture7", "mixture83"]
)
def test_kde_labeled_regions_enclose_probability_mass(
    visualizer_case, visualizer_contracts, monkeypatch, case
):
    """Each displayed label belongs to an actual STAR iso-density geometry.

    Mass means fitted-estimator probability, not empirical fraction of stars.
    The broad 0.05-grid mixture quadrature captures >.9999 total mass; .006
    allows cell-boundary discretization, independently checked by C13/K4/K5.
    Geometry rtol=.05 covers interpolation on the production 200x200 grid.
    """
    import scipy.stats
    from matplotlib.axes import Axes
    from matplotlib.path import Path

    c = visualizer_case
    if case.startswith("mixture"):
        rng = np.random.default_rng(int(case.removeprefix("mixture")))
        a = rng.normal(size=(210, 2)) * [1.0, 2.0]
        b = rng.normal(size=(90, 2)) * [1.4, 0.7] + [6.0, 3.0]
        nonstar = rng.normal(size=(40, 2)) * 0.3 + [6.0, -4.0]
        Z = np.vstack([a, b, nonstar])
        y = np.array(["STAR"] * 300 + ["QSO"] * 40)
        meta = pd.DataFrame({"teff_gspphot": np.linspace(3500, 9000, len(Z))})
        gx, gy = np.meshgrid(np.linspace(-12, 18, 601), np.linspace(-14, 14, 561))
    else:
        Z, y, meta = gaussian_population()
        g = np.linspace(-8, 8, 401)
        gx, gy = np.meshgrid(2 * g, 3 * g)
    stars = {tuple(row) for row in Z[y == "STAR"]}
    fitted, labelled = [], []
    real = scipy.stats.gaussian_kde
    original_clabel = Axes.clabel

    def fit(data, **kwargs):
        estimator = (
            visualizer_contracts.AnalyticGaussian(data, **kwargs)
            if case == "analytic_normal"
            else real(data, **kwargs)
        )
        fitted.append(estimator)
        return estimator

    def clabel(ax, cs, *args, **kwargs):
        geometry = [[segment.copy() for segment in level] for level in cs.allsegs]
        labelled.append(
            (ax, cs, kwargs.get("fmt", args[0] if args else None), geometry)
        )
        return original_clabel(ax, cs, *args, **kwargs)

    monkeypatch.setattr(scipy.stats, "gaussian_kde", fit)
    monkeypatch.setattr(Axes, "clabel", clabel)
    fig, axes = c.viz.plot_latent_kde(Z, y, meta)
    assert (
        fitted and labelled and fig is not None
    ), "Valid KDE needs estimator and informative labelled contours"
    estimator = fitted[0]
    assert estimator.dataset.shape[1] >= 3
    assert all(
        tuple(point) in stars for point in estimator.dataset.T
    ), "KDE fitted on non-STAR observations"
    broad = estimator(np.vstack([gx.ravel(), gy.ravel()]))
    cell = (gx[0, 1] - gx[0, 0]) * (gy[1, 0] - gy[0, 0])
    assert_allclose(broad.sum() * cell, 1, atol=1e-4)
    points = np.column_stack([gx.ravel(), gy.ravel()])
    for ax, cs, fmt, geometry in labelled:
        assert cs in ax.collections, "Must test the actual plotted contour artist"
        claims = []
        for level, segments in zip(cs.levels, geometry):
            label = fmt[level] if isinstance(fmt, dict) else fmt(level)
            match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*%\s*", label)
            assert match, f"Uninterpretable contour probability label: {label}"
            claimed = float(match.group(1)) / 100
            claims.append(claimed)
            mass = float(broad[broad >= level].sum() * cell)
            if case == "analytic_normal":
                assert_allclose(mass, max(0.0, 1 - 12 * np.pi * level), atol=0.006)
            assert_allclose(
                mass,
                claimed,
                atol=0.006,
                rtol=0,
                err_msg=f"Actual label {label} disagrees with enclosed KDE mass",
            )
            nonempty = [segment for segment in segments if len(segment)]
            assert nonempty, f"No contour geometry for label {label}"
            enclosed = np.zeros(len(points), dtype=bool)
            for segment in nonempty:
                # Even/odd containment retains disjoint modes and excludes holes.
                enclosed ^= Path(segment).contains_points(points)
            polygon_mass = float(broad[enclosed].sum() * cell)
            assert_allclose(
                polygon_mass,
                claimed,
                atol=0.006,
                rtol=0,
                err_msg="Actual contour polygons do not enclose their labelled mass",
            )
            vertices = np.vstack(nonempty)
            assert_allclose(
                estimator(vertices.T),
                level,
                rtol=0.05,
                atol=1e-10,
                err_msg="Contour vertices are not on the fitted STAR iso-density",
            )
        assert sorted(claims) == [0.68, 0.95, 0.99]


def test_hr_region_vertices_have_gravity_units_and_stay_in_domain(visualizer_case):
    c = visualizer_case
    _, ax = c.viz.plot_hr_teff_regions(c.meta)
    # Independent input-domain envelope: never take admissible gravity units
    # from limits chosen by the implementation under test.
    gravity = c.meta.logg_gspphot.to_numpy()
    temperature = c.meta.teff_gspphot.to_numpy()
    low, high = gravity.min() - 0.1 * np.ptp(gravity), gravity.max() + 0.1 * np.ptp(
        gravity
    )
    assert ax.patches, "Valid HR regional figure must contain informative regions"
    vertices = np.vstack([p.get_xy() for p in ax.patches])
    assert np.all((vertices[:, 1] >= low) & (vertices[:, 1] <= high)), vertices
    tlo, thi = temperature.min() - 0.1 * np.ptp(
        temperature
    ), temperature.max() + 0.1 * np.ptp(temperature)
    assert np.all((vertices[:, 0] >= tlo) & (vertices[:, 0] <= thi))


@pytest.mark.parametrize("n", [6, 9])
def test_hr_population_title_is_counted_or_omitted(visualizer_case, n):
    c = visualizer_case
    meta = pd.concat([c.meta, c.meta.iloc[:3]], ignore_index=True).iloc[:n]
    _, ax = c.viz.plot_hr_teff_regions(meta)
    title = re.sub(r"(?:LAMOST|Gaia)\s*DR\s*\d+", "", ax.get_title(), flags=re.I)
    compact = re.sub(r"(?<=\d)[ \u202f\u00a0](?=\d{3}(?:\D|$))", "", title)
    counts = re.findall(r"\b\d+\b", compact)
    assert not counts or all(int(v) == n for v in counts), title


def test_reconstruction_cardinality_cannot_reassign_spectral_classes(visualizer_case):
    c = visualizer_case
    # Original G row 4 was removed upstream; truncating assigns original F row 8
    # to G. Twelve retained F rows satisfy the >=10
    # display policy. Missing IDs cannot be reconstructed by this API.
    types = np.array(["F"] * 4 + ["G"] * 4 + ["F"] * 9)
    X = np.arange(17.0)[:, None] * 10
    keep = np.delete(np.arange(17), 4)
    expected_F = X[keep][types[keep] == "F"].mean()
    result = call_or_domain_refusal(
        lambda: c.viz.plot_ae_recon_by_spectral_type(
            X[keep], X[keep] + 2, types, ["feature_Hα_eq_width"]
        ),
        ["align", "length", "taille", "cardinal", "observ", "row"],
    )
    if result is not None and result[0] is not None:
        _, axes = result
        assert_allclose(
            axes[0].patches[0].get_height(),
            expected_F,
            err_msg="Truncation is not observation identity",
        )


@pytest.mark.parametrize("name", ["plot_hdbscan_clusters", "plot_hdbscan_hr"])
def test_cluster_colorbar_matches_actual_cluster_rgb(
    visualizer_case, hdbscan_module, name, visualizer_contracts
):
    c = visualizer_case
    hdb = hdbscan_module.HDBSCANAnalyzer()
    hdb.labels_ = np.array([2, -1, 7, 2, 7, 2])
    hdb._compute_stats()
    hdb._build_palette()
    fig, axes = getattr(c.viz, name)(
        c.Z, hdb, c.y if name.endswith("clusters") else c.meta
    )
    coordinates = (
        c.Z
        if name.endswith("clusters")
        else c.meta[["teff_gspphot", "logg_gspphot"]].to_numpy()
    )
    visualizer_contracts.assert_cluster_encoding(fig, axes[0], hdb.labels_, coordinates)


def test_actual_unit_confidence_is_not_claimed_unavailable(visualizer_case):
    c = visualizer_case
    _, axes = c.viz.plot_xgboost_umap(
        c.Z, np.array(["F"] * 6), np.ones(6), np.array([0] * 6), {0: "blue"}
    )
    # Supplied ones are compatible with genuine probability=1; no availability
    # flag enters this API. Numeric values alone cannot establish absence.
    assert not any("non disponible" in t.get_text() for t in axes[1].texts)


@pytest.mark.parametrize("name", ["plot_ae_embedding", "plot_latent_zoomed"])
def test_no_physical_columns_still_supports_classes_or_explicit_refusal(
    visualizer_case, name
):
    c = visualizer_case
    meta = c.meta[["obsid"]]

    def call():
        if name == "plot_ae_embedding":
            return c.viz.plot_ae_embedding(c.Z, meta, c.y)
        return c.viz.plot_latent_zoomed(c.Z, c.y, meta)

    result = call_or_domain_refusal(
        call, ["physical", "param", "gaia", "column", "colonn"]
    )
    if result is not None and result[0] is not None:
        _, axes = result
        ax = np.atleast_1d(axes)[0]
        assert sum(len(sc.get_offsets()) for sc in ax.collections) == 6


@pytest.mark.parametrize("n_show", [1, 0, 4])
def test_cluster_profiles_unavailable_slots_not_presented_as_data(
    visualizer_case, n_show
):
    c = visualizer_case
    means = pd.DataFrame(
        [[1.0, 2, 3, 4, 5], [-1, -2, -3, -4, -5]], index=[2, 7], columns=list("abcde")
    )
    result = call_or_domain_refusal(
        lambda: c.viz.plot_hdbscan_feature_profiles(
            means, np.array([2, 7, 7]), n_show=n_show
        ),
        ["cluster", "n_show", "positive", "count", "nombre"],
    )
    if result is not None and result[0] is not None:
        _, axes = result
        for ax in np.asarray(axes).ravel():
            assert (
                ax.patches or not ax.axison or not ax.get_visible()
            ), "Empty slot looks like a measurement panel"


def test_more_reconstructions_than_rows_has_no_empty_measurement_axes(visualizer_case):
    c = visualizer_case
    _, axes = c.viz.plot_ae_reconstruction(
        c.Z[:2], c.Z[:2] + 1, ["a", "b"], n_samples=4
    )
    for ax in axes:
        if not ax.axison or not ax.get_visible():
            continue
        assert (
            len(ax.lines) >= 2
        ), "Visible panel needs original and reconstructed observations"
        pair = [np.asarray(line.get_ydata()) for line in ax.lines[:2]]
        assert all(
            len(values) == 2 for values in pair
        ), "Empty or incomplete reconstruction lines"
        row = next((i for i in range(2) if np.array_equal(pair[0], c.Z[i])), None)
        assert row is not None, "Panel is not an actual input observation"
        assert_array_equal(pair[1], c.Z[row] + 1)
    populated = [ax for ax in axes if ax.axison and ax.get_visible()]
    assert len(populated) == 2


def test_truncated_variance_annotation_does_not_attribute_wrong_mass(visualizer_case):
    c = visualizer_case
    pca = SimpleNamespace(
        explained_variance_ratio=np.array([0.6, 0.25, 0.15]),
        n_components_for_variance=lambda threshold: 3,
    )
    _, axes = c.viz.plot_variance_explained(pca, threshold=0.95, max_pcs=2)
    for annotation in axes[1].texts:
        label = annotation.get_text()
        percentages = re.findall(r"(\d+(?:[.,]\d+)?)\s*%", label)
        counts = re.findall(r"(\d+)\s*PCs?\b|\bPCs?\s*[:=]?\s*(\d+)", label, re.I)
        counts = [int(a or b) for a, b in counts]
        if percentages and not counts:
            assert hasattr(
                annotation, "xy"
            ), "A variance percentage needs a PC count or an annotated coordinate"
            counts = [int(annotation.xy[0])]
        for count in counts:
            assert 1 <= count <= 3
            for percentage in percentages:
                assert_allclose(
                    float(percentage.replace(",", ".")),
                    sum(pca.explained_variance_ratio[:count]) * 100,
                    atol=0.05,
                    err_msg="Annotated PC count and variance disagree",
                )


def test_log_histogram_keeps_small_positive_errors(
    visualizer_case, visualizer_contracts, monkeypatch
):
    c = visualizer_case
    errors = np.array([1e-8, 1e-7, 1e-6, 0.01, 0.1, 1.0])
    calls = visualizer_contracts.capture_histograms(monkeypatch)
    _, axes = c.viz.plot_ae_error_distribution_logscale(errors, np.array(["STAR"] * 6))
    call = next(call for call in calls if call["ax"] is axes[1])
    missing, edges = visualizer_contracts.assert_histogram_population(call, errors)
    assert missing == 0 or visualizer_contracts.exclusion_report(
        axes[1], "STAR", missing, 6, edges
    ), "Positive MSE population silently truncated"


def test_histogram_does_not_silently_omit_fifth_class(
    visualizer_case, visualizer_contracts, monkeypatch
):
    c = visualizer_case
    labels = np.array(["A", "B", "C", "D", "E", "E"])
    calls = visualizer_contracts.capture_histograms(monkeypatch)
    result = call_or_domain_refusal(
        lambda: c.viz.plot_ae_error_distribution(
            np.array([1.0, 2, 3, 4, 5, 9]), labels
        ),
        ["class", "palette", "population"],
    )
    if result is None:
        return
    _, ax = result
    represented = {label.split(" ")[0] for label in ax.get_legend_handles_labels()[1]}
    assert represented == set(labels), "Palette length silently drops classes"
    values = np.array([1.0, 2, 3, 4, 5, 9])
    assert len(calls) == 5, "Legend-only classes are not histogram populations"
    for cls in set(labels):
        call = next(
            (
                call
                for call in calls
                if cls
                in visualizer_contracts.tokens(call["label"]).union(
                    {s.upper() for s in visualizer_contracts.tokens(call["label"])}
                )
            ),
            None,
        )
        assert call is not None, f"Missing actual histogram for class {cls}"
        missing, _ = visualizer_contracts.assert_histogram_population(
            call, values[labels == cls]
        )
        assert missing == 0


@pytest.mark.parametrize("name", ["plot_training_history", "plot_ae_vs_pca"])
def test_default_output_directory_applies_to_direct_exports(
    visualizer_case, tmp_path, name, visualizer_contracts
):
    c = visualizer_case
    viz = c.module.DimRedVisualizer(output_dir=tmp_path, dpi=30)
    if name == "plot_training_history":
        result = viz.plot_training_history(
            {"train_loss": [2.0, 1], "val_loss": [3.0, 2]}
        )
    else:
        result = viz.plot_ae_vs_pca(
            pd.DataFrame(
                {
                    "method": ["PCA", "AE"],
                    "n_components": [1, 2],
                    "mse_mean": [0.5, 0.3],
                }
            )
        )
    assert result[0] is not None
    assert list(tmp_path.glob("*.png")), "Configured output_dir ignored"
    for path in tmp_path.glob("*.png"):
        visualizer_contracts.assert_png_matches_figure(path, result[0], viz.dpi)


def test_trustworthiness_axis_contains_computed_values(
    visualizer_case, visualizer_contracts
):
    c = visualizer_case
    rng = np.random.default_rng(1729)
    X = rng.normal(size=(12, 2))
    bad = X[[7, 0, 9, 4, 1, 10, 2, 11, 3, 6, 5, 8]]
    _, ax = c.viz.plot_trustworthiness(X, X, bad, k_values=[1, 2, 3])
    visualizer_contracts.assert_visible_or_disclosed(
        ax, {"UMAP": ax.lines[0].get_ydata(), "t-SNE": ax.lines[1].get_ydata()}
    )


def test_default_eigenspectrum_line_labels_are_renderable(visualizer_case):
    c = visualizer_case
    pca = SimpleNamespace(
        loadings=np.array([[0.1, -0.4, 0.9, -0.2]]),
        explained_variance_ratio=np.array([1.0]),
    )
    try:
        fig, axes = c.viz.plot_eigenspectra(
            pca, np.array([3800.0, 4400.0, 5500.0, 6800.0]), n_components=1
        )
        fig.canvas.draw()
    except Exception as exc:
        assert False, f"Valid default eigen-spectrum labels must render: {exc}"
    assert_array_equal(axes[0].lines[0].get_ydata(), pca.loadings[0])


def test_linear_histogram_truncation_discloses_population(
    visualizer_case, visualizer_contracts, monkeypatch
):
    """R10 candidate: conditional renormalization is not full-class coverage."""
    c = visualizer_case
    stars = np.linspace(0.01, 0.1, 40)
    qso = np.r_[[0.04, 0.09, 0.1], np.linspace(4, 7, 47)]
    values = np.r_[stars, qso]
    labels = np.array(["STAR"] * 40 + ["QSO"] * 50)
    calls = visualizer_contracts.capture_histograms(monkeypatch)
    _, axes = c.viz.plot_ae_error_distribution_logscale(values, labels)
    call = next(
        call for call in calls if call["ax"] is axes[0] and "QSO" in call["label"]
    )
    missing, edges = visualizer_contracts.assert_histogram_population(call, qso)
    area = sum(p.get_width() * p.get_height() for p in call["result"][2])
    assert_allclose(area, 1)
    assert missing == 0 or visualizer_contracts.exclusion_report(
        axes[0], "QSO", missing, 50, edges
    ), f"Silent conditional histogram: represented={50-missing}/50, area={area}, median={np.median(qso)}"
