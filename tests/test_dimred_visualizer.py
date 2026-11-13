"""M2.8: numeric artists, observation pairing and real synthetic exports.

Tolerances are floating point/printed-decimal tolerances, never astronomical
uncertainties. Scientific defects are executed separately in the red archive.
"""

from math import fsum, sqrt
from types import SimpleNamespace

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pandas as pd
import pytest


def offsets(collection):
    return np.asarray(collection.get_offsets())


def heights(ax):
    return np.array([p.get_height() for p in ax.patches])


def widths(ax):
    return np.array([p.get_width() for p in ax.patches])


def text(ax):
    return " ".join(t.get_text() for t in ax.texts)


@pytest.mark.parametrize("mode", ["classes", "continuous", "plain"])
def test_embedding_exact_observation_associations(visualizer_case, mode):
    c = visualizer_case
    before = c.Z.copy()
    kwargs = {}
    if mode == "classes":
        kwargs["y"] = c.y
    if mode == "continuous":
        values = np.array([8.0, np.nan, 3.0, 21.0, np.inf, 1.0])
        kwargs.update(color_by=values, color_label="measured scalar")
    fig, ax = c.viz.plot_embedding(c.Z, method="t-SNE", **kwargs)
    if mode == "classes":
        assert set(ax.get_legend_handles_labels()[1]) == set(c.y)
        for sc in ax.collections:
            cls = sc.get_label()
            assert_array_equal(offsets(sc), c.Z[c.y == cls])
    elif mode == "continuous":
        assert_array_equal(offsets(ax.collections[0]), c.Z[[1, 4]])
        assert_array_equal(offsets(ax.collections[1]), c.Z[[0, 2, 3, 5]])
        assert_array_equal(ax.collections[1].get_array(), [8, 3, 21, 1])
        # A finite nondegenerate color range is required; clipping quantiles
        # remain an open display policy rather than a fixed 2/98 implementation.
        limits = ax.collections[1].get_clim()
        assert np.isfinite(limits).all() and limits[0] < limits[1]
        assert fig.axes[-1].get_ylabel() == "measured scalar"
    else:
        assert_array_equal(offsets(ax.collections[0]), c.Z)
    assert "t-SNE" in ax.get_xlabel() and "t-SNE" in ax.get_ylabel()
    assert_array_equal(c.Z, before)


@pytest.mark.parametrize("order", [np.arange(6), np.array([4, 0, 5, 2, 1, 3])])
def test_grid_and_ae_color_pairing_under_coherent_permutation(visualizer_case, order):
    c = visualizer_case
    Z, y, meta = c.Z[order], c.y[order], c.meta.iloc[order].copy()
    meta.iloc[1, meta.columns.get_loc("teff_gspphot")] = np.nan
    for name in ["plot_embedding_grid", "plot_ae_embedding", "plot_latent_zoomed"]:
        if name == "plot_embedding_grid":
            _, axes = getattr(c.viz, name)(
                Z, y, meta, params=["teff_gspphot", "absent"]
            )
            assert not axes[2].get_visible()
        elif name == "plot_ae_embedding":
            _, axes = getattr(c.viz, name)(Z, meta, y)
        else:
            _, axes = getattr(c.viz, name)(Z, y, meta)
        for sc in axes[0].collections:
            cls = sc.get_label().split(" ")[0]
            assert_array_equal(offsets(sc), Z[y == cls])
        colored = [s for s in axes[1].collections if s.get_array() is not None][0]
        assert_array_equal(offsets(colored), Z[[0, 2, 3, 4, 5]])
        assert_array_equal(colored.get_array(), meta.teff_gspphot.iloc[[0, 2, 3, 4, 5]])
    assert_array_equal(meta.obsid, c.meta.obsid.iloc[order])


@pytest.mark.parametrize("axis", [0, 1])
def test_hr_finite_triplets_and_physical_axes(visualizer_case, axis):
    c = visualizer_case
    meta, Z = c.meta.copy(), c.Z.copy()
    meta.iloc[1, 1] = np.nan
    meta.iloc[3, 2] = np.inf
    Z[4, axis] = np.inf
    fig, ax = c.viz.plot_hr_diagram_embedding(meta, Z, component=axis)
    expected = [[3100, 4.8], [5200, 4.1], [8100, 0.6]]
    assert_array_equal(offsets(ax.collections[0]), expected)
    assert_array_equal(ax.collections[0].get_array(), Z[[0, 2, 5], axis])
    assert ax.xaxis_inverted() and ax.yaxis_inverted()
    assert "K" in ax.get_xlabel() and "dex" in ax.get_ylabel()
    assert str(axis + 1) in fig.axes[-1].get_ylabel()


def test_hr_missing_columns_refused(visualizer_case):
    c = visualizer_case
    with pytest.raises(ValueError, match="teff|logg|Gaia"):
        c.viz.plot_hr_diagram_embedding(pd.DataFrame(), c.Z)
    assert c.viz.plot_hr_teff_regions(pd.DataFrame()) == (None, None)


def test_real_pca_variance_and_reconstruction_energy(
    visualizer_case, dimred_pca_class, visualizer_contracts
):
    c = visualizer_case
    X = np.array(
        [[3.0, 0, 0], [-3, 0, 0], [0, 2, 0], [0, -2, 0], [0, 0, 1], [0, 0, -1]]
    )
    pca = dimred_pca_class(n_components=3, svd_solver="full").fit(X, ["a", "b", "c"])
    _, axes = c.viz.plot_variance_explained(pca, threshold=0.9, max_pcs=3)
    assert_allclose(heights(axes[0]), np.array([9, 4, 1]) / 14 * 100)
    assert_allclose(axes[1].lines[0].get_ydata(), np.array([9, 13, 14]) / 14 * 100)
    assert_allclose(axes[1].lines[1].get_ydata(), [90, 90])
    assert_allclose(axes[1].lines[2].get_xdata(), [2, 2])
    visualizer_contracts.assert_variance_annotations(
        axes[1], pca.explained_variance_ratio, expected_count=2
    )
    df = pca.reconstruction_error_vs_n(X, [1, 2, 3])
    _, ax = c.viz.plot_reconstruction_error(df)
    assert_allclose(ax.lines[0].get_xdata(), [1, 2, 3])
    assert_allclose(ax.lines[0].get_ydata(), [5 / 9, 1 / 9, 0], atol=1e-28)
    # Per-row squared residuals: [0,0,4/3,4/3,1/3,1/3].
    expected_std = sqrt(
        fsum((x - 5 / 9) ** 2 for x in [0, 0, 4 / 3, 4 / 3, 1 / 3, 1 / 3]) / 6
    )
    vertices = ax.collections[0].get_paths()[0].vertices
    assert_allclose(vertices[vertices[:, 0] == 1, 1].min(), 5 / 9 - expected_std)
    assert ax.get_yscale() == "log"


def test_signed_loadings_selection_and_sign_equivariance(visualizer_case):
    c = visualizer_case
    L = np.array([[0.2, -0.8], [-0.9, 0.1], [0.4, -0.3], [-0.1, 0.7]])
    names = ["a", "b", "c", "d"]
    for sign in [1, -1]:
        values = L * sign
        pca = SimpleNamespace(
            loadings_dataframe=lambda: pd.DataFrame(values, index=names)
        )
        _, ax = c.viz.plot_loadings_heatmap(pca, n_pcs=2, n_features=3)
        assert [t.get_text() for t in ax.get_yticklabels()] == ["b", "a", "d"]
        assert_array_equal(ax.images[0].get_array(), values[[1, 0, 3]])
        assert_allclose(ax.images[0].get_clim(), [-0.9, 0.9])
        pca.explained_variance_ratio = [0.64]
        pca.top_features_per_pc = lambda **kw: pd.DataFrame(
            {"feature": ["b", "a", "c"], "loading": values[[1, 0, 2], 0]}
        )
        _, ax = c.viz.plot_loadings_bar(pca, n_top=3)
        expected = sorted(zip(values[[1, 0, 2], 0], ["b", "a", "c"]))
        assert_allclose(widths(ax), [v for v, _ in expected])
        assert [t.get_text() for t in ax.get_yticklabels()] == [n for _, n in expected]
        assert "64.0%" in text(ax)


def test_family_weights_bar_and_donut_angles(visualizer_case):
    c = visualizer_case
    first = pd.DataFrame({"family": ["A", "B"], "weight": [0.8, 0.2]})
    second = pd.DataFrame({"family": ["B", "C"], "weight": [0.3, 0.7]})
    colors = {"A": "red", "B": "blue", "C": "green"}
    _, ax = c.viz.plot_loadings_family_bar(first, second, colors)
    assert [t.get_text() for t in ax.get_xticklabels()] == ["A", "B", "C"]
    assert_allclose(heights(ax), [80, 20, 0, 0, 30, 70])
    handles, labels = ax.get_legend_handles_labels()
    series = {
        label.replace(" ", "").upper(): handle for handle, label in zip(handles, labels)
    }
    assert set(series) == {"PC1", "PC2"}
    assert_allclose([p.get_height() for p in series["PC1"]], [80, 20, 0])
    assert_allclose([p.get_height() for p in series["PC2"]], [0, 30, 70])
    _, axes = c.viz.plot_loadings_family_donut(first, second, colors)
    for ax, weights in zip(axes, [[0.8, 0.2], [0.3, 0.7]]):
        assert_allclose(
            [(p.theta2 - p.theta1) / 360 for p in ax.patches], weights, atol=1e-7
        )
        assert len(ax.get_legend().get_texts()) == 2


def test_correlation_matrix_axes_values_and_missing_annotations(visualizer_case):
    c = visualizer_case
    df = pd.DataFrame(
        [[0.25, -0.85, np.nan], [-0.4, 0.6, 0]],
        index=["PC1", "PC3"],
        columns=["teff_gspphot", "logg_gspphot", "custom"],
    )
    original = df.copy(deep=True)
    _, ax = c.viz.plot_correlation_heatmap(df)
    assert_allclose(ax.images[0].get_array(), df, equal_nan=True)
    assert_allclose(ax.images[0].get_clim(), [-1, 1])
    assert [t.get_text() for t in ax.get_xticklabels()] == ["T_eff", "log g", "custom"]
    assert {(t.get_position(), t.get_text()) for t in ax.texts} == {
        ((0, 0), "0.25"),
        ((1, 0), "-0.85"),
        ((0, 1), "-0.40"),
        ((1, 1), "0.60"),
        ((2, 1), "0.00"),
    }
    pd.testing.assert_frame_equal(df, original)


@pytest.mark.parametrize("components", [1, 3, 10])
def test_eigenspectra_coordinates_variance_and_line_positions(
    visualizer_case, components
):
    c = visualizer_case
    L = np.array(
        [[0.1, -0.4, 0.9, -0.2], [-0.7, 0.3, -0.1, 0.5], [0.2, 0.6, -0.3, -0.8]]
    )
    pca = SimpleNamespace(
        loadings=L, explained_variance_ratio=np.array([0.7, 0.2, 0.1])
    )
    wl = np.array([3800.0, 4400.0, 5500.0, 6800.0])
    _, axes = c.viz.plot_eigenspectra(
        pca, wl, {"inside": 4861.0, "outside": 7200.0}, components
    )
    for i, ax in enumerate(axes):
        assert_array_equal(ax.lines[0].get_xdata(), wl)
        assert_array_equal(ax.lines[0].get_ydata(), L[i])
        assert_allclose(ax.lines[2].get_xdata(), [4861, 4861])
        assert f"{[70,20,10][i]:.1f}%" in ax.get_ylabel()
    assert len(axes) == min(components, 3)
    assert "Å" in axes[-1].get_xlabel()


@pytest.mark.parametrize("lr", [False, True])
def test_training_history_minimum_is_not_last_and_distinct_series(visualizer_case, lr):
    c = visualizer_case
    h = {"train_loss": [9.0, 3, 5, 2], "val_loss": [8.0, 1, 4, 6]}
    if lr:
        h["lr_history"] = [0.01, 0.01, 0.001, 0.0001]
    _, axes = c.viz.plot_training_history(h)
    assert_array_equal(axes[0].lines[0].get_ydata(), h["train_loss"])
    assert_array_equal(axes[0].lines[1].get_ydata(), h["val_loss"])
    assert_array_equal(axes[0].lines[0].get_xdata(), [1, 2, 3, 4])
    assert_allclose(axes[0].lines[2].get_xdata(), [2, 2])
    assert "1.00000" in text(axes[0])
    if lr:
        assert_array_equal(axes[1].lines[0].get_ydata(), h["lr_history"])


@pytest.mark.parametrize("n_samples", [1, 3, 6])
def test_reconstruction_drawn_rows_and_manual_mse(visualizer_case, n_samples):
    c = visualizer_case
    X = c.Z.copy()
    recon = X + np.column_stack([np.arange(1, 7), np.arange(1, 7) * 2])
    _, axes = c.viz.plot_ae_reconstruction(X, recon, ["a", "b"], n_samples, c.y)
    observed = set()
    for ax in axes:
        row = next(i for i in range(6) if np.array_equal(ax.lines[0].get_ydata(), X[i]))
        observed.add(row)
        assert_array_equal(ax.lines[1].get_ydata(), recon[row])
        assert f"{2.5*(row+1)**2:.5f}" in ax.get_title()
        assert c.y[row] in ax.get_title()
    assert len(observed) == n_samples
    assert_array_equal(X, c.Z)


def test_ae_pca_comparison_does_not_substitute_metrics(visualizer_case):
    c = visualizer_case
    df = pd.DataFrame(
        {
            "method": ["PCA", "AE", "PCA", "AEwide"],
            "n_components": [4, 2, 1, 3],
            "mse_mean": [0.03, 0.17, 0.9, 0.11],
        }
    )
    _, ax = c.viz.plot_ae_vs_pca(df)
    assert_array_equal(ax.lines[0].get_xdata(), [1, 4])
    assert_allclose(ax.lines[0].get_ydata(), [0.9, 0.03])
    assert_allclose(ax.lines[1].get_ydata(), [0.17, 0.17])
    assert_allclose(ax.lines[2].get_ydata(), [0.11, 0.11])
    assert ax.get_legend_handles_labels()[1] == ["PCA", "AE", "AEwide"]


def test_interpolation_actual_path_feature_order_and_endpoints(visualizer_case):
    c = visualizer_case
    Z = c.Z[:3]
    X = np.array([[0.0, 0, 0], [1, 8, 3], [2, 16, 6]])
    _, axes = c.viz.plot_latent_interpolation(
        Z, X, ["slow", "fast", "middle"], "A", "B"
    )
    assert_array_equal(offsets(axes[0].collections[0]), Z)
    assert_allclose(axes[0].collections[0].get_array(), [0, 0.5, 1])
    assert_array_equal(offsets(axes[0].collections[1]), Z[[0]])
    assert_array_equal(offsets(axes[0].collections[2]), Z[[-1]])
    assert axes[0].collections[1].get_label() == "A"
    assert axes[0].collections[2].get_label() == "B"
    assert [t.get_text() for t in axes[1].get_xticklabels()] == [
        "fast",
        "middle",
        "slow",
    ]
    for i, line in enumerate(axes[1].lines):
        assert_array_equal(line.get_ydata(), X[i, [1, 2, 0]])


def test_error_family_mean_is_feature_weighted_and_std_is_between_features(
    visualizer_case,
):
    c = visualizer_case
    X = np.zeros((3, 5))
    # Constant residual columns give feature MSEs 1,4,9,16,25.
    recon = np.tile([1.0, 2, 3, 4, 5], (3, 1))
    pca = SimpleNamespace(
        _FAMILIES={"A": [r"^a"], "B": [r"^b"]}, _FAMILY_COLORS=["red", "blue"]
    )
    _, axes = c.viz.plot_ae_error_by_family(
        X, recon, ["a1", "a2", "b1", "b2", "b3"], pca
    )
    assert_allclose(widths(axes[0]), [2.5, 50 / 3])
    assert_allclose(axes[0].lines[-1].get_xdata(), [11, 11])
    # Sample SD over feature means, not dispersion among the three identical rows.
    segments = axes[0].collections[0].get_segments()
    assert_allclose(segments[0][:, 0], [2.5 - 3 / sqrt(2), 2.5 + 3 / sqrt(2)])
    assert_allclose(segments[1][:, 0], [50 / 3 - sqrt(193 / 3), 50 / 3 + sqrt(193 / 3)])
    assert_allclose(offsets(axes[1].collections[0]), [[2, 2.5]])
    assert_allclose(offsets(axes[1].collections[1]), [[3, 50 / 3]])


def test_spectral_type_means_for_aligned_unequal_classes(visualizer_case):
    c = visualizer_case
    X = np.column_stack([np.arange(22.0), np.arange(22.0) ** 2])
    R = X + [2.0, -3.0]
    types = np.array(["F"] * 10 + ["G"] * 12)
    try:
        result = c.viz.plot_ae_recon_by_spectral_type(
            X, R, types, ["feature_Hα_eq_width", "feature_Na_D_eq_width"]
        )
    except Exception as exc:
        assert False, f"Valid aligned reconstruction must remain usable: {exc}"
    assert (
        isinstance(result, tuple) and result[0] is not None
    ), "Valid aligned data need an informative figure"
    _, axes = result
    assert_allclose(heights(axes[0]), [4.5, 28.5, 6.5, 25.5])
    assert_allclose(heights(axes[1]), [15.5, 1513 / 6, 17.5, 1513 / 6 - 3])
    assert "n=10" in axes[0].get_title() and "n=12" in axes[1].get_title()
    assert c.viz.plot_ae_recon_by_spectral_type(X[:4], R[:4], types[:4], []) == (
        None,
        None,
    )


def test_histogram_densities_integrate_and_class_mean_median_are_distinct(
    visualizer_case,
):
    c = visualizer_case
    values = np.array([1.0, 2, 9, 4, 6, 12])
    _, ax = c.viz.plot_ae_error_distribution(values, c.y)
    # Each class has its own normalized density, even with unequal counts.
    for container in ax.containers:
        assert_allclose(sum(p.get_width() * p.get_height() for p in container), 1)
    import re

    star_label = next(
        label
        for label in ax.get_legend_handles_labels()[1]
        if re.search(r"\bSTAR\b", label)
    )
    assert re.search(r"\b3\b", star_label)
    assert any(
        abs(float(v) - 16 / 3) < 5e-5 for v in re.findall(r"\d+\.\d+", star_label)
    )
    assert_allclose([line.get_xdata()[0] for line in ax.lines], [4, 7, 6])
    _, axes = c.viz.plot_ae_error_distribution_logscale(values, c.y)
    assert axes[1].get_xscale() == "log"
    for ax in axes:
        for container in ax.containers:
            assert_allclose(sum(p.get_width() * p.get_height() for p in container), 1)
        assert_allclose([line.get_xdata()[0] for line in ax.lines], [4, 7, 6])


def hdb_case(module, labels):
    hdb = module.HDBSCANAnalyzer()
    hdb.labels_ = np.asarray(labels)
    hdb._compute_stats()
    hdb._build_palette()
    hdb.labels_pres_ = hdb.labels_.copy()
    hdb.ids_pres_ = hdb.cluster_ids_.copy()
    hdb.n_clusters_pres_ = hdb.n_clusters_
    hdb.color_map_pres_ = hdb.color_map_.copy()
    return hdb


@pytest.mark.parametrize("labels", [[2, -1, 7, 2, 7, 2], [-1] * 6, [7] * 6])
def test_hdbscan_real_analyzer_state_and_point_populations(
    visualizer_case, hdbscan_module, labels
):
    c = visualizer_case
    hdb = hdb_case(hdbscan_module, labels)
    _, axes = c.viz.plot_hdbscan_clusters(c.Z, hdb, c.y)
    for sc, cid in zip(axes[0].collections, [-1] + hdb.cluster_ids_):
        assert_array_equal(offsets(sc), c.Z[hdb.labels_ == cid])
        assert_allclose(sc.get_facecolors()[0, :3], hdb.color_map_[cid][:3])
    for sc, cls in zip(axes[1].collections, sorted(set(c.y))):
        assert_array_equal(offsets(sc), c.Z[c.y == cls])
        import re

        assert re.search(
            rf"\b{cls}\b", sc.get_label()
        ), "Class label differs from plotted population"
        assert re.search(
            rf"\bn\s*=\s*{sum(c.y==cls)}\b|\(\s*{sum(c.y==cls)}\s*\)", sc.get_label()
        )
    assert f"n={labels.count(-1)}" in text(axes[0])


@pytest.mark.parametrize("has_hr", [False, True])
def test_hdbscan_presentation_and_split_consistent_populations(
    visualizer_case, hdbscan_module, has_hr
):
    c = visualizer_case
    hdb = hdb_case(hdbscan_module, [2, -1, 7, 2, 7, 2])
    meta = c.meta if has_hr else c.meta[["obsid"]]
    _, axes = c.viz.plot_hdbscan_presentation(c.Z, hdb, meta)
    for sc, cid in zip(axes[0].collections, [-1, 2, 7]):
        assert_array_equal(offsets(sc), c.Z[hdb.labels_ == cid])
    first, second = c.viz.plot_hdbscan_present_split(c.Z, hdb, meta)
    assert_array_equal(offsets(first.axes[0].collections[1]), c.Z[[0, 3, 5]])
    if has_hr:
        assert_array_equal(
            offsets(second.axes[0].collections[1]),
            c.meta.iloc[[0, 3, 5]][["teff_gspphot", "logg_gspphot"]],
        )
        assert second.axes[0].xaxis_inverted() and second.axes[0].yaxis_inverted()
    else:
        assert second is None
        assert c.viz.plot_hdbscan_hr(c.Z, hdb, meta) == (None, None)


def test_hdbscan_hr_values_and_temperature_overlay(visualizer_case, hdbscan_module):
    c = visualizer_case
    hdb = hdb_case(hdbscan_module, [2, -1, 7, 2, 7, 2])
    meta = c.meta.copy()
    meta.iloc[1, 2] = np.nan
    _, axes = c.viz.plot_hdbscan_hr(c.Z, hdb, meta)
    assert_array_equal(
        offsets(axes[0].collections[1]),
        meta.iloc[[0, 3, 5]][["teff_gspphot", "logg_gspphot"]],
    )
    assert_array_equal(offsets(axes[1].collections[0]), c.Z)
    assert_array_equal(axes[1].collections[0].get_array(), meta.teff_gspphot)
    assert axes[0].xaxis_inverted() and axes[0].yaxis_inverted()


def test_hdbscan_sensitivity_exact_columns_and_marked_parameters(visualizer_case):
    c = visualizer_case
    df = pd.DataFrame(
        {
            "min_cluster_size": [5, 12, 30],
            "n_clusters": [7, 4, 2],
            "pct_noise": [8.0, 21, 43],
        }
    )
    _, axes = c.viz.plot_hdbscan_sensitivity(df, 12, 30)
    for ax, expected in zip(axes, [[7, 4, 2], [8, 21, 43]]):
        assert_array_equal(ax.lines[0].get_xdata(), [5, 12, 30])
        assert_array_equal(ax.lines[0].get_ydata(), expected)
        assert_array_equal(ax.lines[1].get_xdata(), [12, 12])
        assert_array_equal(ax.lines[2].get_xdata(), [30, 30])


def cluster_means():
    return pd.DataFrame(
        [[0.1, -2.0, 1.0, 0.5, -0.3, 0], [2.0, 0.4, -0.1, -0.6, 0.9, 0]],
        index=[2, 7],
        columns=list("abcdef"),
    )


def test_cluster_heatmap_discriminating_columns_counts_and_values(visualizer_case):
    c = visualizer_case
    df = cluster_means()
    _, ax = c.viz.plot_hdbscan_feature_heatmap(df, np.array([7, 2, 7, -1, 7]), n_top=3)
    assert [t.get_text() for t in ax.get_xticklabels()] == ["b", "a", "e"]
    assert_array_equal(ax.images[0].get_array(), df[["b", "a", "e"]])
    assert [t.get_text() for t in ax.get_yticklabels()] == ["C2  (n=1)", "C7  (n=3)"]


def test_cluster_profiles_signed_top_five_not_first_five(visualizer_case):
    c = visualizer_case
    _, axes = c.viz.plot_hdbscan_feature_profiles(
        cluster_means(), np.array([7, 2, 7, -1, 7]), n_show=2
    )
    assert "C7" in axes[0].get_title() and "n=3" in axes[0].get_title()
    assert_allclose(widths(axes[0]), [2, 0.9, -0.6, 0.4, -0.1])
    assert_allclose(widths(axes[1]), [-2, 1, 0.5, -0.3, 0.1])


@pytest.mark.parametrize(
    "confidence", [None, np.array([0.45, 0.7, 0.95, 0.6, 0.8, 0.5])]
)
def test_xgboost_classes_confidence_and_cluster_associations(
    visualizer_case, confidence
):
    c = visualizer_case
    pred = np.array(["F", "G", "F", "M", "F", "G"])
    cl = np.array([2, -1, 7, 2, 7, 2])
    _, axes = c.viz.plot_xgboost_umap(c.Z, pred, confidence, cl, {2: "red", 7: "blue"})
    for sc, cls in zip(axes[0].collections, ["F", "G", "M"]):
        assert_array_equal(offsets(sc), c.Z[pred == cls])
        import re

        assert re.search(
            rf"\b{cls}\b", sc.get_label()
        ), "Prediction label differs from its actual points"
        assert f"n={(pred == cls).sum()}" in sc.get_label()
    if confidence is None:
        assert not axes[1].collections
        assert text(
            axes[1]
        ).strip(), "Unavailable confidence needs an explicit diagnostic"
    else:
        assert_array_equal(offsets(axes[1].collections[0]), c.Z)
        assert_array_equal(axes[1].collections[0].get_array(), confidence)
    assert_array_equal(offsets(axes[2].collections[0]), c.Z[[1]])
    assert_array_equal(offsets(axes[2].collections[1]), c.Z[[0, 3, 5]])
    assert_array_equal(offsets(axes[2].collections[2]), c.Z[[2, 4]])


def test_fg_population_threshold_and_overlays(visualizer_case):
    c = visualizer_case
    assert c.viz.plot_fg_confusion(c.Z, np.array(["F"] * 6)) == (None, None)
    Z = np.arange(26.0).reshape(13, 2)
    pred = np.array(["F"] * 3 + ["M"] * 2 + ["G"] * 8)
    _, ax = c.viz.plot_fg_confusion(Z, pred)
    assert_array_equal(offsets(ax.collections[0]), Z)
    assert_array_equal(offsets(ax.collections[1]), Z[:3])
    assert_array_equal(offsets(ax.collections[2]), Z[5:])
    import re

    for cls, count, label in zip(["F", "G"], [3, 8], ax.get_legend_handles_labels()[1]):
        assert re.search(rf"\b{cls}\b", label)
        assert re.search(rf"\b{count}\b", label)


def rank_trustworthiness(X, Z, k):
    """Independent scalar rank-penalty definition, no sklearn internals.

    Inputs in tests have no distance ties. Self is removed by identity.
    """
    n = len(X)
    penalty = 0
    for i in range(n):
        ref = sorted(
            (j for j in range(n) if j != i), key=lambda j: fsum((X[i] - X[j]) ** 2)
        )
        emb = sorted(
            (j for j in range(n) if j != i), key=lambda j: fsum((Z[i] - Z[j]) ** 2)
        )
        for j in emb[:k]:
            if j not in ref[:k]:
                penalty += ref.index(j) + 1 - k
    return 1 - 2 * penalty / (n * k * (2 * n - 3 * k - 1))


@pytest.mark.integration
@pytest.mark.parametrize("subsample", [12, 9])
def test_real_sklearn_trustworthiness_independent_rank_oracle(
    visualizer_case, subsample, monkeypatch
):
    c = visualizer_case
    rng = np.random.default_rng(1729)
    X = rng.normal(size=(12, 2))
    bad = X[np.array([7, 0, 9, 4, 1, 10, 2, 11, 3, 6, 5, 8])]
    import sklearn.manifold

    real = sklearn.manifold.trustworthiness
    calls = []

    def measured(ref, embedding, **kwargs):
        calls.append((ref.copy(), embedding.copy(), kwargs["n_neighbors"]))
        return real(ref, embedding, **kwargs)

    monkeypatch.setattr(sklearn.manifold, "trustworthiness", measured)
    for repeat in range(2):
        _, ax = c.viz.plot_trustworthiness(
            X, X.copy(), bad, k_values=[1, 2, 3], n_subsample=subsample
        )
    assert len(calls) == 12
    expected = []
    for i, (ref, embedding, k) in enumerate(calls[:6]):
        indices = [
            next(j for j, row in enumerate(X) if np.array_equal(row, r)) for r in ref
        ]
        assert len(set(indices)) == subsample
        assert_array_equal(embedding, (X if i % 2 == 0 else bad)[indices])
        assert_array_equal(calls[i + 6][0], ref)
        assert_array_equal(calls[i + 6][1], embedding)
        assert_array_equal(ref, calls[0][0])
        if i % 2:
            expected.append(rank_trustworthiness(ref, embedding, k))
    assert_allclose(ax.lines[0].get_ydata(), [1, 1, 1], atol=1e-14)
    assert_allclose(ax.lines[1].get_ydata(), expected, atol=1e-14)
    assert_array_equal(ax.lines[1].get_xdata(), [1, 2, 3])
    assert f"{sum(expected)/3:.4f}" in ax.lines[1].get_label()
    assert any(v < 1 for v in expected)


@pytest.mark.parametrize("k", [4, 5, 8])
def test_trustworthiness_impossible_neighbors_explicit_refusal(visualizer_case, k):
    c = visualizer_case
    X = np.arange(16.0).reshape(8, 2)
    with pytest.raises(ValueError, match="n_neighbors"):
        c.viz.plot_trustworthiness(X, X, X, k_values=[k])


def test_stability_producer_order_and_mean_without_reference(
    visualizer_case, embedding_module, monkeypatch
):
    c = visualizer_case

    # Execute the actual report producer; only expensive embedding fitting is
    # replaced with seed-dependent geometry. This is not a real UMAP fit.
    def geometry(engine, X):
        out = X[:, :2].copy()
        out[0, 0] += engine.random_state**2
        return out

    monkeypatch.setattr(embedding_module.EmbeddingEngine, "fit_transform", geometry)
    report = embedding_module.EmbeddingEngine(method="tsne").stability_report(
        c.Z, n_seeds=4
    )
    assert_array_equal(report.seed, [0, 1, 2, 3])
    assert report.procrustes_distance.iloc[0] == 0
    _, ax = c.viz.plot_stability(report)
    assert_allclose(heights(ax), report.procrustes_distance)
    expected = sum(report.procrustes_distance.iloc[1:]) / 3
    assert f"{expected:.4f}" in text(ax)
    # Distinct asymmetric supplied distances discriminate mean from median.
    frame = pd.DataFrame(
        {
            "seed": [0, 1, 2, 3],
            "method": ["TSNE"] * 4,
            "procrustes_distance": [0, 0.01, 0.02, 0.9],
        }
    )
    _, ax = c.viz.plot_stability(frame)
    assert "0.3100" in text(ax)


@pytest.mark.parametrize("umap", [False, True])
def test_synthesis_preserves_distinct_geometries_and_physical_colors(
    visualizer_case, umap
):
    c = visualizer_case
    pca, ae, embedding = c.Z * [2, 3], c.Z + 100, c.Z * [-1, 4]
    fig = c.viz.plot_synthesis_grid(ae, c.y, c.meta, pca, embedding if umap else None)
    data_axes = [a for a in fig.axes if a.get_xlabel()]
    geometries = [pca] + ([embedding] if umap else []) + [ae]
    import re

    for row, Z in enumerate(geometries):
        expected_name = (
            [r"\bPCA\b"] + ([r"\bUMAP\b"] if umap else []) + [r"autoencod|\bAE\b"]
        )[row]
        assert re.search(expected_name, data_axes[row * 3].get_ylabel(), re.I)
        for sc in data_axes[row * 3].collections:
            assert_array_equal(offsets(sc), Z[c.y == sc.get_label()])
        for col, param in [(1, "teff_gspphot"), (2, "mh_gspphot")]:
            sc = data_axes[row * 3 + col].collections[0]
            assert_array_equal(offsets(sc), Z)
            assert_array_equal(sc.get_array(), c.meta[param])


@pytest.mark.parametrize("suffix", ["png", "pdf"])
def test_real_export_readable_and_same_semantic_figure(
    visualizer_case, tmp_path, suffix, visualizer_contracts
):
    from PIL import Image

    c = visualizer_case
    path = tmp_path / "new" / f"figure.{suffix}"
    fig, ax = c.viz.plot_embedding(c.Z, save_path=path)
    assert_array_equal(offsets(ax.collections[0]), c.Z)
    if suffix == "png":
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            pixels = np.asarray(im.convert("RGB"))
            assert pixels.shape[0] > 100 and pixels.std() > 1
        visualizer_contracts.assert_png_matches_figure(path, fig, c.viz.dpi)
    else:
        assert path.read_bytes().startswith(b"%PDF-")
        assert b"%%EOF" in path.read_bytes()[-50:]
    assert fig.canvas.get_renderer() is not None


def test_save_defaults_override_no_output_and_write_failure(
    visualizer_case, tmp_path, monkeypatch
):
    c = visualizer_case
    out = tmp_path / "default"
    viz = c.module.DimRedVisualizer(output_dir=out, dpi=50)
    assert out.is_dir()
    viz.plot_embedding(c.Z)
    assert (out / "embedding_umap.png").is_file()
    explicit = tmp_path / "explicit.png"
    viz.plot_embedding(c.Z, method="t-SNE", save_path=explicit)
    assert explicit.is_file() and not (out / "embedding_t-sne.png").exists()
    monkeypatch.chdir(tmp_path)
    c.viz.plot_embedding(c.Z)
    assert set(tmp_path.iterdir()) == {out, explicit}
    target = tmp_path / "blocked"
    target.write_text("existing", encoding="utf-8")
    with pytest.raises(OSError):
        c.viz.plot_embedding(c.Z, save_path=target / "figure.png")
    assert target.read_text() == "existing"


def test_real_kde_reproducible_star_selection_and_observation_colors(visualizer_case):
    c = visualizer_case
    rng = np.random.default_rng(1729)
    Z = rng.normal(size=(140, 2))
    labels = np.array(["STAR"] * 120 + ["QSO"] * 20)
    meta = pd.DataFrame({"teff_gspphot": np.arange(140.0) + 4000})
    meta.loc[7, "teff_gspphot"] = np.nan
    results = []
    for _ in range(2):
        _, axes = c.viz.plot_latent_kde(Z, labels, meta)
        scatter = next(
            sc
            for sc in axes[0].collections
            if hasattr(sc, "get_offsets") and len(sc.get_offsets()) > 1
        )
        actual = offsets(scatter)
        values = np.asarray(scatter.get_array())
        assert len(actual) == 119
        expected_pairs = {
            (tuple(Z[i]), meta.teff_gspphot[i]) for i in range(120) if i != 7
        }
        assert {
            (tuple(point), value) for point, value in zip(actual, values)
        } == expected_pairs
        results.append((actual.copy(), values.copy()))
    assert_array_equal(results[0][0], results[1][0])
    assert_array_equal(results[0][1], results[1][1])


@pytest.mark.parametrize("suffix", ["png", "pdf"])
def test_direct_ae_export_real_file_and_history_preserved(
    visualizer_case, tmp_path, suffix
):
    c = visualizer_case
    history = {"train_loss": [5.0, 2.0, 1.0], "val_loss": [6.0, 1.0, 3.0]}
    target = tmp_path / f"history.{suffix}"
    _, axes = c.viz.plot_training_history(history, save_path=target)
    assert target.read_bytes().startswith(b"\x89PNG" if suffix == "png" else b"%PDF-")
    assert_array_equal(axes[0].lines[0].get_ydata(), [5, 2, 1])
    assert history == {"train_loss": [5.0, 2.0, 1.0], "val_loss": [6.0, 1.0, 3.0]}


def test_hr_regions_scatter_preserves_input_triplets_despite_geometry_defect(
    visualizer_case,
):
    c = visualizer_case
    original = c.meta.copy(deep=True)
    _, ax = c.viz.plot_hr_teff_regions(c.meta)
    assert_array_equal(
        offsets(ax.collections[0]), c.meta[["teff_gspphot", "logg_gspphot"]]
    )
    assert_array_equal(ax.collections[0].get_array(), c.meta.teff_gspphot)
    assert ax.xaxis_inverted() and ax.yaxis_inverted()
    pd.testing.assert_frame_equal(c.meta, original)


def test_analytic_normal_mass_oracle_has_known_solution():
    # In two dimensions, integrating the polar area element r dr dtheta gives
    # P(f(X)>=t)=1-2*pi*t. This validates the red proof's independent quadrature.
    grid = np.linspace(-7, 7, 601)
    x, y = np.meshgrid(grid, grid)
    density = np.exp(-(x * x + y * y) / 2) / (2 * np.pi)
    area = (grid[1] - grid[0]) ** 2
    for mass in [0.68, 0.95, 0.99]:
        threshold = (1 - mass) / (2 * np.pi)
        assert_allclose(density[density >= threshold].sum() * area, mass, atol=0.002)
