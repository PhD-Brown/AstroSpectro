"""Explicit M2.8 counterexamples; archived and removed from the green head.

No xfail and no production edits. Refusal, masking and unavailable outputs are
alternatives on invalid data; valid positive cases are in the green suite.
"""

import re
from types import SimpleNamespace

import numpy as np
from numpy.testing import assert_allclose
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
    Z = rng.normal(size=(n, 2))
    meta = pd.DataFrame({"teff_gspphot": np.arange(n, dtype=float) + 3500})
    return Z, np.array(["STAR"] * n), meta


@pytest.mark.parametrize(
    "analytic", [False, True], ids=["real_scipy", "analytic_normal"]
)
def test_kde_labeled_regions_enclose_probability_mass(
    visualizer_case, monkeypatch, analytic
):
    c = visualizer_case
    Z, y, meta = gaussian_population()
    observed = []
    from matplotlib.axes import Axes
    import scipy.stats

    original = Axes.contour
    real_kde = scipy.stats.gaussian_kde
    fitted = []

    def fit(data, **kwargs):
        estimator = real_kde(data, **kwargs)
        fitted.append(estimator)
        return estimator

    def contour(ax, XX, YY, density, *args, **kwargs):
        result = original(ax, XX, YY, density, *args, **kwargs)
        observed.append((XX.copy(), YY.copy(), density.copy(), result))
        return result

    monkeypatch.setattr(Axes, "contour", contour)
    if analytic:

        class StandardNormal:
            def __init__(self, data, **kwargs):
                pass

            def __call__(self, grid):
                return np.exp(-np.sum(grid**2, axis=0) / 2) / (2 * np.pi)

        monkeypatch.setattr(scipy.stats, "gaussian_kde", StandardNormal)
    else:
        monkeypatch.setattr(scipy.stats, "gaussian_kde", fit)
    c.viz.plot_latent_kde(Z, y, meta)
    assert observed, "Valid nominal KDE must contain informative contours"
    XX, YY, density, cs = observed[0]
    # Quadrature over actual plotted grid, with endpoint half-cell weights.
    weights = np.ones(density.shape)
    weights[[0, -1], :] *= 0.5
    weights[:, [0, -1]] *= 0.5
    cell = (XX[0, 1] - XX[0, 0]) * (YY[1, 0] - YY[0, 0])
    assert 0.98 < np.sum(weights * density) * cell < 1.01
    # The production display grid excludes tails (~1.6% in this fixture).
    # Integrate on an independent larger domain before comparing global mass.
    grid = np.linspace(-8, 8, 401)
    gx, gy = np.meshgrid(grid, grid)
    positions = np.vstack([gx.ravel(), gy.ravel()])
    broad = (
        np.exp(-np.sum(positions**2, axis=0) / 2) / (2 * np.pi)
        if analytic
        else fitted[0](positions)
    ).reshape(gx.shape)
    area = (grid[1] - grid[0]) ** 2
    assert_allclose(broad.sum() * area, 1, atol=1e-7)
    masses = []
    for level in cs.levels:
        mass = broad[broad >= level].sum() * area
        if analytic:
            assert_allclose(mass, 1 - 2 * np.pi * level, atol=0.006)
        masses.append(mass)
    # Level order low->high corresponds to enclosed masses high->low.
    assert_allclose(
        masses,
        [0.99, 0.95, 0.68],
        atol=0.006,
        err_msg="Density-height percentiles are not probability masses",
    )


def test_hr_region_vertices_have_gravity_units_and_stay_in_domain(visualizer_case):
    c = visualizer_case
    _, ax = c.viz.plot_hr_teff_regions(c.meta)
    low, high = sorted(ax.get_ylim())
    assert ax.patches, "Valid HR regional figure must contain informative regions"
    vertices = np.vstack([p.get_xy() for p in ax.patches])
    assert np.all((vertices[:, 1] >= low) & (vertices[:, 1] <= high)), vertices


@pytest.mark.parametrize("n", [6, 9])
def test_hr_population_title_is_counted_or_omitted(visualizer_case, n):
    c = visualizer_case
    meta = pd.concat([c.meta, c.meta.iloc[:3]], ignore_index=True).iloc[:n]
    _, ax = c.viz.plot_hr_teff_regions(meta)
    title = ax.get_title().replace(" ", "").replace("\u202f", "")
    counts = re.findall(r"(\d+)étoiles", title)
    assert not counts or all(int(v) == n for v in counts), title


def test_reconstruction_cardinality_cannot_reassign_spectral_classes(visualizer_case):
    c = visualizer_case
    # Original row 4 was removed upstream; truncating assigns row 5 to G,
    # although its actual class is F. Twelve retained F rows satisfy the >=10
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
def test_cluster_colorbar_matches_actual_cluster_rgb(visualizer_case, name):
    c = visualizer_case
    colors = {-1: (0.8, 0.8, 0.8, 1), 2: (1, 0, 0, 1), 7: (0, 0, 1, 1)}
    hdb = SimpleNamespace(
        labels_=np.array([2, -1, 7, 2, 7, 2]),
        cluster_ids_=[2, 7],
        color_map_=colors,
        n_clusters_=2,
        n_noise_=1,
    )
    fig, axes = getattr(c.viz, name)(
        c.Z, hdb, c.y if name.endswith("clusters") else c.meta
    )
    cb = next(
        a._colorbar for a in fig.axes if getattr(a, "_colorbar", None) is not None
    )
    for collection, cid in zip(axes[0].collections[1:], [2, 7]):
        assert_allclose(
            cb.mappable.to_rgba(cid)[:3],
            collection.get_facecolors()[0, :3],
            err_msg=f"Colorbar does not encode C{cid}",
        )


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
        assert (
            ax.lines or not ax.axison or not ax.get_visible()
        ), "Unpopulated reconstruction panel"


def test_truncated_variance_annotation_does_not_attribute_wrong_mass(visualizer_case):
    c = visualizer_case
    pca = SimpleNamespace(
        explained_variance_ratio=np.array([0.6, 0.25, 0.15]),
        n_components_for_variance=lambda threshold: 3,
    )
    _, axes = c.viz.plot_variance_explained(pca, threshold=0.95, max_pcs=2)
    annotations = " ".join(t.get_text() for t in axes[1].texts)
    assert "3 PCs\n→ 85.0%" not in annotations, annotations


def test_log_histogram_keeps_small_positive_errors(visualizer_case):
    c = visualizer_case
    errors = np.array([1e-8, 1e-7, 1e-6, 0.01, 0.1, 1.0])
    _, axes = c.viz.plot_ae_error_distribution_logscale(errors, np.array(["STAR"] * 6))
    bars = axes[1].containers[0]
    assert (
        min(p.get_x() for p in bars) <= errors.min()
    ), "Positive MSEs dropped below arbitrary 1e-4 floor"


def test_histogram_does_not_silently_omit_fifth_class(visualizer_case):
    c = visualizer_case
    labels = np.array(["A", "B", "C", "D", "E", "E"])
    _, ax = c.viz.plot_ae_error_distribution(np.array([1.0, 2, 3, 4, 5, 9]), labels)
    represented = {label.split(" ")[0] for label in ax.get_legend_handles_labels()[1]}
    assert represented == set(labels), "Palette length silently drops classes"


@pytest.mark.parametrize("name", ["plot_training_history", "plot_ae_vs_pca"])
def test_default_output_directory_applies_to_direct_exports(
    visualizer_case, tmp_path, name
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


def test_trustworthiness_axis_contains_computed_values(visualizer_case):
    c = visualizer_case
    rng = np.random.default_rng(1729)
    X = rng.normal(size=(12, 2))
    bad = X[[7, 0, 9, 4, 1, 10, 2, 11, 3, 6, 5, 8]]
    _, ax = c.viz.plot_trustworthiness(X, X, bad, k_values=[1, 2, 3])
    lo, hi = sorted(ax.get_ylim())
    vals = np.concatenate([line.get_ydata() for line in ax.lines[:2]])
    assert np.all((vals >= lo) & (vals <= hi)), (vals, (lo, hi))
