"""Independent PCA contracts on synthetic data (U/I/P/S; see audit/M22_PCA.md).

Float64 tolerances allow accumulated roundoff on these small, well-conditioned
matrices: 2e-13 for linear algebra, 2e-12 for squared residuals. Neither axis
signs nor individual bases in repeated-eigenvalue subspaces are prescribed.
"""

import importlib.util
from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
import pytest
from scipy.stats import ConstantInputWarning


ATOL = 2e-13


def centered_correlation(x, y):
    """Compute Pearson's coefficient from its defining inner product."""
    a, b = x - np.mean(x), y - np.mean(y)
    return np.dot(a, b) / np.sqrt(np.dot(a, a) * np.dot(b, b))


@pytest.mark.unit
def test_constructor_and_unfitted_guards(dimred_pca_class):
    model = dimred_pca_class(2, random_state=17, svd_solver="full")
    assert (model.n_components, model.random_state, model.svd_solver) == (2, 17, "full")
    assert model.feature_names_ is None and model._X_fit is None
    calls = [
        lambda: model.transform(np.ones((2, 3))),
        lambda: model.explained_variance_ratio,
        lambda: model.cumulative_variance,
        lambda: model.n_components_for_variance(),
        model.variance_summary,
        lambda: model.loadings,
        model.loadings_dataframe,
        model.top_features_per_pc,
        model.loadings_family_breakdown,
        lambda: model.correlations_with_params(pd.DataFrame()),
        lambda: model.reconstruction_error(np.ones((2, 3))),
        lambda: model.class_separation(np.ones((2, 3)), np.array(["A", "B"])),
        lambda: model.sklearn_pca,
        lambda: model.mean_,
        model._check_fitted,
    ]
    for call in calls:
        with pytest.raises(RuntimeError, match="fit"):
            call()


@pytest.mark.unit
@pytest.mark.parametrize("solver", ["full", "auto", "covariance_eigh", "randomized"])
def test_fit_covariance_scores_and_variance(dimred_pca_class, pca_axes, solver):
    X, basis, coords, mean = pca_axes
    original = X.copy()
    model = dimred_pca_class(3, random_state=19, svd_solver=solver)
    assert model.fit(X, ["a", "b", "c"]) is model
    scores = model.transform(X)
    assert scores.shape == (6, 3)
    np.testing.assert_array_equal(X, original)
    np.testing.assert_allclose(model.mean_, mean, atol=ATOL, rtol=0)
    np.testing.assert_allclose(scores.mean(axis=0), 0, atol=ATOL, rtol=0)
    np.testing.assert_allclose(
        model.loadings @ model.loadings.T, np.eye(3), atol=ATOL, rtol=0
    )
    covariance = (X - mean).T @ (X - mean) / 5
    eigenvalues = np.linalg.eigvalsh(covariance)[::-1]
    np.testing.assert_allclose(eigenvalues, [18 / 5, 8 / 5, 2 / 5], atol=ATOL, rtol=0)
    np.testing.assert_allclose(
        model.sklearn_pca.explained_variance_, eigenvalues, atol=ATOL, rtol=0
    )
    np.testing.assert_allclose(
        model.explained_variance_ratio, [9 / 14, 4 / 14, 1 / 14], atol=ATOL, rtol=0
    )
    for i in range(3):
        np.testing.assert_allclose(
            np.outer(model.loadings[i], model.loadings[i]),
            np.outer(basis[i], basis[i]),
            atol=ATOL,
            rtol=0,
        )
        np.testing.assert_allclose(
            scores[:, i] ** 2, coords[:, i] ** 2, atol=ATOL, rtol=0
        )
    np.testing.assert_allclose(scores, (X - mean) @ model.loadings.T, atol=ATOL, rtol=0)
    assert model.sklearn_pca.get_params()["random_state"] == 19
    assert model.sklearn_pca.get_params()["svd_solver"] == solver
    assert np.all(np.diff(model.cumulative_variance) >= 0)
    np.testing.assert_allclose(
        model.cumulative_variance, [9 / 14, 13 / 14, 1], atol=ATOL, rtol=0
    )
    summary = model.variance_summary()
    assert summary.columns.tolist() == [
        "PC",
        "variance_individuelle",
        "variance_cumulee",
        "valeur_propre",
    ]
    assert summary.PC.tolist() == ["PC1", "PC2", "PC3"]
    np.testing.assert_allclose(summary.valeur_propre, eigenvalues, atol=ATOL, rtol=0)
    np.testing.assert_allclose(
        summary.variance_individuelle,
        eigenvalues / eigenvalues.sum(),
        atol=ATOL,
        rtol=0,
    )
    np.testing.assert_allclose(
        summary.variance_cumulee, [9 / 14, 13 / 14, 1], atol=ATOL, rtol=0
    )


@pytest.mark.unit
@pytest.mark.parametrize("n,expected", [(1, 1), (2, 2), (0.7, 2), ("mle", 2)])
def test_component_selection_parameters(dimred_pca_class, pca_axes, n, expected):
    X, _, _, _ = pca_axes
    # MLE is checked separately below against a rank-one noise-free contract;
    # its exact statistical choice for general data is not asserted here.
    if n == "mle":
        X = np.column_stack([np.arange(12.0), np.arange(12.0) * 2, np.zeros(12)])
        expected = 1
    model = dimred_pca_class(n, svd_solver="full").fit(X)
    assert model.transform(X).shape == (len(X), expected)


@pytest.mark.unit
def test_centering_does_not_standardize(dimred_pca_class):
    X = np.array([[100, 0], [-100, 0], [0, 1], [0, -1.0]]) + [12, 8]
    model = dimred_pca_class(2, svd_solver="full").fit(X)
    np.testing.assert_allclose(
        model.explained_variance_ratio, [10000 / 10001, 1 / 10001], atol=ATOL, rtol=0
    )
    np.testing.assert_allclose(model.mean_, [12, 8], atol=ATOL, rtol=0)


@pytest.mark.unit
def test_rank_deficient_and_equal_eigenvalue_subspaces(dimred_pca_class):
    X = np.array([[2, 0, 0], [-2, 0, 0], [0, 2, 0], [0, -2, 0.0]]) + [5, 8, 11]
    model = dimred_pca_class(2, svd_solver="full").fit(X)
    np.testing.assert_allclose(
        model.explained_variance_ratio, [0.5, 0.5], atol=ATOL, rtol=0
    )
    np.testing.assert_allclose(
        model.loadings.T @ model.loadings, np.diag([1, 1, 0]), atol=ATOL, rtol=0
    )
    np.testing.assert_allclose(model.reconstruction_error(X), 0, atol=2e-12, rtol=0)
    # A new observation outside the training plane cannot be reconstructed.
    np.testing.assert_allclose(
        model.reconstruction_error(np.array([[5, 8, 14.0]])), [3.0], atol=2e-12, rtol=0
    )


@pytest.mark.unit
def test_constant_data_rejection_or_mathematically_consistent_fit(
    dimred_pca_class,
):
    X = np.full((5, 3), 7.0)
    model = dimred_pca_class(3, svd_solver="full")
    # Variance-zero policy is OPEN (#59). Only a validation rejection is
    # acceptable; unrelated RuntimeError/AttributeError must still fail.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        try:
            model.fit(X)
        except ValueError:
            return
    assert not np.isfinite(model.explained_variance_ratio).any()
    np.testing.assert_array_equal(model.transform(X), np.zeros((5, 3)))
    np.testing.assert_array_equal(model.reconstruction_error(X), np.zeros(5))
    # This tests undefined fractions; it does not endorse component selection on them.


@pytest.mark.unit
def test_reproducibility_translation_permutation_and_no_test_refit(
    dimred_pca_class, pca_axes
):
    X, _, _, _ = pca_axes
    one = dimred_pca_class(2, random_state=23, svd_solver="randomized")
    scores = one.fit_transform(X)
    other = dimred_pca_class(2, random_state=23, svd_solver="randomized").fit(X)
    np.testing.assert_allclose(other.transform(X), scores, atol=ATOL, rtol=0)
    before = (
        one.mean_.copy(),
        one.loadings.copy(),
        one.explained_variance_ratio.copy(),
    )
    new = X[:2] + [100, -200, 500]
    np.testing.assert_allclose(
        one.transform(new), (new - X.mean(axis=0)) @ one.loadings.T, atol=ATOL, rtol=0
    )
    for actual, expected in zip(
        (one.mean_, one.loadings, one.explained_variance_ratio), before
    ):
        np.testing.assert_array_equal(actual, expected)
    perm = np.array([5, 2, 0, 4, 1, 3])
    np.testing.assert_allclose(one.transform(X[perm]), scores[perm], atol=ATOL, rtol=0)
    shifted = dimred_pca_class(2, svd_solver="full").fit(X[perm] + [9, 11, -8])
    np.testing.assert_allclose(
        shifted.loadings.T @ shifted.loadings,
        one.loadings.T @ one.loadings,
        atol=ATOL,
        rtol=0,
    )
    np.testing.assert_allclose(
        shifted.explained_variance_ratio,
        one.explained_variance_ratio,
        atol=ATOL,
        rtol=0,
    )


@pytest.mark.unit
def test_refit_replaces_mean_dimensions_and_names(dimred_pca_class, pca_axes):
    X = pca_axes[0]
    model = dimred_pca_class(2, svd_solver="full").fit(X, ["a", "b", "c"])
    Y = np.column_stack([np.arange(8.0), np.arange(8.0) ** 2])
    model.fit(Y)
    assert model.feature_names_ == ["f0", "f1"]
    assert model.loadings.shape == (2, 2)
    np.testing.assert_allclose(model.mean_, Y.mean(axis=0), atol=ATOL, rtol=0)
    np.testing.assert_allclose(
        model.fit_transform(Y), model.transform(Y), atol=ATOL, rtol=0
    )


@pytest.mark.unit
@pytest.mark.parametrize("threshold,expected", [(0.5, 1), (0.8, 2), (0.95, 3)])
def test_reachable_variance_thresholds(dimred_pca_class, pca_axes, threshold, expected):
    model = dimred_pca_class(3, svd_solver="full").fit(pca_axes[0])
    assert model.n_components_for_variance(threshold) == expected
    # Exact equality must select the first component that reaches the threshold.
    assert model.n_components_for_variance(model.cumulative_variance[0]) == 1


@pytest.mark.unit
def test_loadings_labels_top_absolute_order_and_sign(dimred_pca_class, pca_axes):
    X, basis, _, _ = pca_axes
    model = dimred_pca_class(2, svd_solver="full").fit(X, ["alpha", "beta", "gamma"])
    frame = model.loadings_dataframe()
    assert frame.index.tolist() == ["alpha", "beta", "gamma"]
    assert frame.columns.tolist() == ["PC1", "PC2"]
    np.testing.assert_allclose(frame.to_numpy(), model.loadings.T, atol=ATOL, rtol=0)
    # Reverse an axis explicitly: rankings unchanged and reported signs reversed.
    model.sklearn_pca.components_[0] *= -np.sign(model.loadings[0] @ basis[0])
    sign = np.sign(model.loadings[0] @ basis[0])
    top = model.top_features_per_pc(0, 2)
    assert top.feature.tolist() == ["beta", "alpha"]
    np.testing.assert_allclose(
        top.loading, sign * np.array([0.8, 0.6]), atol=ATOL, rtol=0
    )
    np.testing.assert_allclose(top.abs_loading, [0.8, 0.6], atol=ATOL, rtol=0)
    assert top.PC.tolist() == ["PC1", "PC1"]
    assert len(model.top_features_per_pc(0, 99)) == 3
    assert model.top_features_per_pc(0, 0).empty


@pytest.mark.unit
@pytest.mark.parametrize(
    "feature,prefix",
    [
        ("feature_Halpha_depth", "Balmer"),
        ("feature_CaH_present", "Calcium"),
        ("feature_Mg_b", "Magnésium"),
        ("feature_Fe_5270", "Fer &"),
        ("Na_D", "Sodium"),
        ("feature_TiO5_depth", "Titane"),
        ("feature_Ba_4554", "Strontium"),
        ("continuum_slope", "Continuum"),
        ("wing_width", "Profils"),
        ("Dn4000", "Indices"),
        ("match_unknown", "Détection"),
        ("unclassified", "Autres"),
    ],
)
def test_family_regex_first_match_taxonomy(dimred_pca_class, feature, prefix):
    # A lexical taxonomy test, not independent proof of astrophysical grouping.
    assert dimred_pca_class()._assign_family(feature).startswith(prefix)


@pytest.mark.unit
def test_family_squared_weights_aliases_overrides_colors(dimred_pca_class, pca_axes):
    X = pca_axes[0]
    model = dimred_pca_class(3, svd_solver="full").fit(X, ["Halpha", "CaII", "unknown"])
    out = model.loadings_family_breakdown(pc_indices=(0, 1, 2))
    assert out["contrib_pc1"] is out["contributions"][0]
    assert out["contrib_pc2"] is out["contributions"][1]
    for i, expected in enumerate(([0.36, 0.64, 0], [0.64, 0.36, 0], [0, 0, 1])):
        weights = out["contributions"][i].set_index("family").weight
        np.testing.assert_allclose(
            weights.reindex(out["families"], fill_value=0), expected, atol=ATOL, rtol=0
        )
        assert weights.sum() == pytest.approx(1, abs=ATOL, rel=0)
        assert np.all(np.diff(weights) <= ATOL)
    names = ["Halpha", "Hbeta", "unknown"]
    override = model.loadings_family_breakdown(names, pc_indices=(0,))
    assert override["contributions"][0].iloc[0].weight == pytest.approx(
        1, abs=ATOL, rel=0
    )
    assert model.feature_names_ == ["Halpha", "CaII", "unknown"]
    assert "contrib_pc2" not in override
    assert model.loadings_family_breakdown(pc_indices=())["contributions"] == {}
    colors = out["color_map"]
    assert list(colors) == out["family_list"] == list(out["families_map"])
    assert list(colors.values()) == [
        "#E8593C",
        "#3B8BD4",
        "#4C9B6F",
        "#B07DB8",
        "#F5A623",
        "#2E86AB",
        "#D4A853",
        "#7F8FA6",
        "#C06C84",
        "#6CAE75",
        "#A3B4C5",
        "#CCCCCC",
    ]
    assert colors == model._family_color_map()
    # Normalization must work for non-unit vectors too, with no input mutation.
    raw = np.array([[3.0, -4.0, 0.0]])
    result = model._family_contributions(0, raw, names, ["A", "B", "C"])
    np.testing.assert_allclose(
        result.set_index("family").weight.reindex(["A", "B"]),
        [9 / 25, 16 / 25],
        atol=ATOL,
        rtol=0,
    )
    np.testing.assert_array_equal(raw, [[3, -4, 0]])


@pytest.mark.unit
@pytest.mark.parametrize("index", [-1, 3])
def test_family_indices_rejected(dimred_pca_class, pca_axes, index):
    model = dimred_pca_class(3).fit(pca_axes[0])
    with pytest.raises(IndexError, match="pc_idx"):
        model.loadings_family_breakdown(pc_indices=(index,))


@pytest.mark.unit
def test_missing_family_names_diagnostic(dimred_pca_class, pca_axes):
    model = dimred_pca_class(2).fit(pca_axes[0])
    model.feature_names_ = None
    with pytest.raises(RuntimeError, match="feature_names"):
        model.loadings_family_breakdown()


@pytest.mark.unit
@pytest.mark.parametrize("method", ["pearson", "spearman"])
def test_correlations_analytic_positive_negative_zero_nonlinear(
    dimred_pca_class, method
):
    x = np.arange(-30.0, 30.0) + 0.5
    model = dimred_pca_class(1).fit(x[:, None])
    meta = pd.DataFrame(
        {
            "positive": 2 * x + 7,
            "negative": -3 * x,
            "zero": x * x,
            "nonlinear": x**3,
            "integer": (2 * x).astype(np.int64),
            "text": ["x"] * 60,
        },
        index=np.arange(60)[::-1],
    )
    result = model.correlations_with_params(
        meta, scores=x[:, None], method=method, n_pcs=99
    )
    expected_nonlinear = 1 if method == "spearman" else centered_correlation(x, x**3)
    assert meta["integer"].dtype == np.dtype(np.int64)
    assert result.columns.tolist() == [
        "positive",
        "negative",
        "zero",
        "nonlinear",
        "integer",
    ]
    assert result.index.tolist() == ["PC1"]
    np.testing.assert_allclose(
        result.iloc[0], [1, -1, 0, expected_nonlinear, 1], atol=ATOL, rtol=0
    )
    # Adding integer metadata must not alter any other correlation.
    without_integer = model.correlations_with_params(
        meta.drop(columns="integer"), scores=x[:, None], method=method, n_pcs=99
    )
    pd.testing.assert_frame_equal(result.drop(columns="integer"), without_integer)
    # The documented alignment is positional; paired permutation keeps it.
    order = np.roll(np.arange(60), 17)
    pd.testing.assert_frame_equal(
        result,
        model.correlations_with_params(meta.iloc[order], x[order, None], method, 1),
    )
    # Explicit scores take precedence over the stored fit population.
    reverse = model.correlations_with_params(meta, -x[:, None], method, 1)
    np.testing.assert_allclose(reverse, -result, atol=ATOL, rtol=0)


@pytest.mark.unit
@pytest.mark.parametrize("n", [29, 30, 31])
@pytest.mark.parametrize("method", ["pearson", "spearman"])
def test_correlation_pair_count_boundary(dimred_pca_class, n, method):
    x = np.arange(float(n))
    model = dimred_pca_class(1).fit(x[:, None])
    result = model.correlations_with_params(pd.DataFrame({"p": x}), x[:, None], method)
    if n < 30:
        assert np.isnan(result.iloc[0, 0])
    else:
        assert result.iloc[0, 0] == pytest.approx(1, abs=ATOL, rel=0)


@pytest.mark.unit
@pytest.mark.parametrize("method", ["pearson", "spearman"])
def test_pairwise_nonfinite_mask_and_undefined_constants(dimred_pca_class, method):
    x = np.arange(35.0)
    scores = np.column_stack([x, -x])
    scores[0, 0] = np.nan
    scores[1, 0] = np.inf
    p = x**2
    p[2] = -np.inf
    p[3] = np.nan
    model = dimred_pca_class(1).fit(x[:, None])
    result = model.correlations_with_params(pd.DataFrame({"p": p}), scores, method, 2)
    for i in range(2):
        valid = np.isfinite(scores[:, i]) & np.isfinite(p)
        # Strictly monotonic values: ranks coincide (or reverse).
        expected = (
            (1 if i == 0 else -1)
            if method == "spearman"
            else centered_correlation(scores[valid, i], p[valid])
        )
        assert result.iloc[i, 0] == pytest.approx(expected, abs=ATOL, rel=0)
    with pytest.warns(ConstantInputWarning):
        constant = model.correlations_with_params(
            pd.DataFrame({"constant": np.ones(35)}), x[:, None], method
        )
    assert np.isnan(constant.iloc[0, 0])
    with pytest.warns(ConstantInputWarning):
        constant_score = model.correlations_with_params(
            pd.DataFrame({"p": x}), np.ones((35, 1)), method
        )
    assert np.isnan(constant_score.iloc[0, 0])


@pytest.mark.unit
def test_correlation_recomputed_scores_empty_columns_and_bad_method(dimred_pca_class):
    x = np.arange(35.0)
    model = dimred_pca_class(1).fit(x[:, None])
    meta = pd.DataFrame({"p": x})
    pd.testing.assert_frame_equal(
        model.correlations_with_params(meta),
        model.correlations_with_params(meta, model.transform(x[:, None])),
    )
    result = model.correlations_with_params(pd.DataFrame({"text": ["x"] * 35}))
    assert result.shape == (1, 0)
    with pytest.raises(ValueError, match="method"):
        model.correlations_with_params(meta, method="kendall")


@pytest.mark.unit
def test_default_correlation_count_on_eleven_component_model(dimred_pca_class):
    axes = np.diag(np.arange(1.0, 12.0))
    X = np.vstack([axes, -axes, axes, -axes])
    model = dimred_pca_class(11, svd_solver="full").fit(X)
    result = model.correlations_with_params(pd.DataFrame({"p": X[:, -1]}))
    assert result.index.tolist() == [f"PC{i}" for i in range(1, 11)]
    expected = np.zeros(10)
    expected[0] = np.sign(model.loadings[0, -1])
    np.testing.assert_allclose(result["p"], expected, atol=ATOL, rtol=0)


@pytest.mark.unit
def test_default_class_count_on_six_component_model(dimred_pca_class):
    axes = np.diag(np.arange(1.0, 7.0))
    X = np.vstack([axes, -axes])
    model = dimred_pca_class(6, svd_solver="full").fit(X)
    result = model.class_separation(X, np.array(["A"] * len(X)))
    assert result.index.tolist() == [("A", f"PC{i}") for i in range(1, 6)]
    for i in range(5):
        values = X[:, -1 - i] * np.sign(model.loadings[i, -1 - i])
        expected = [
            len(X),
            values.mean(),
            values.std(),
            np.median(values),
            np.percentile(values, 25),
            np.percentile(values, 75),
        ]
        np.testing.assert_allclose(result.iloc[i], expected, atol=ATOL, rtol=0)


@pytest.mark.unit
@pytest.mark.parametrize("new", [False, True])
def test_reconstruction_projector_mse_and_curve(dimred_pca_class, pca_axes, new):
    X, basis, _, mean = pca_axes
    model = dimred_pca_class(3, svd_solver="full").fit(X)
    Y = X + [4, -3, 2] if new else X.copy()
    original = Y.copy()
    values = []
    for k in range(4):
        projector = basis[:k].T @ basis[:k]
        residual = (Y - mean) @ (np.eye(3) - projector)
        expected = np.sum(residual**2, axis=1) / 3
        np.testing.assert_allclose(
            model.reconstruction_error(Y, k), expected, atol=2e-12, rtol=0
        )
        values.append(expected)
    np.testing.assert_array_equal(Y, original)
    assert np.all(np.diff(np.array(values), axis=0) <= 2e-12)
    np.testing.assert_allclose(
        model.reconstruction_error(Y), values[-1], atol=2e-12, rtol=0
    )
    order = [2, 0, 2, 3, 1]
    curve = model.reconstruction_error_vs_n(Y, order)
    assert curve.n_components.tolist() == order
    np.testing.assert_allclose(
        curve.mse_mean, [values[k].mean() for k in order], atol=2e-12, rtol=0
    )
    np.testing.assert_allclose(
        curve.mse_std, [values[k].std(ddof=0) for k in order], atol=2e-12, rtol=0
    )
    default = model.reconstruction_error_vs_n(Y)
    assert default.n_components.tolist() == [1, 2, 3]


@pytest.mark.unit
def test_truncated_model_reconstruction_and_tail_variance(dimred_pca_class, pca_axes):
    X = pca_axes[0]
    model = dimred_pca_class(1, svd_solver="full").fit(X)
    np.testing.assert_allclose(
        model.explained_variance_ratio, [9 / 14], atol=ATOL, rtol=0
    )
    expected = np.array([0, 0, 4 / 3, 4 / 3, 1 / 3, 1 / 3])
    np.testing.assert_allclose(
        model.reconstruction_error(X), expected, atol=2e-12, rtol=0
    )
    # Mean squared residual = discarded sample eigenvalues * (N-1)/(N*D).
    assert model.reconstruction_error(X).mean() == pytest.approx(
        (8 / 5 + 2 / 5) * 5 / 18, abs=2e-12, rel=0
    )


@pytest.mark.unit
def test_default_curve_stops_at_fifty_components(dimred_pca_class):
    # Paired orthogonal coordinates, distinct eigenvalues, no sampling.
    axes = np.diag(np.arange(1.0, 53.0))
    X = np.vstack([axes, -axes])
    model = dimred_pca_class(52, svd_solver="full").fit(X)
    result = model.reconstruction_error_vs_n(X)
    assert result.n_components.tolist() == list(range(1, 51))
    # At 50 dimensions only amplitudes 1 and 2 are discarded.
    assert result.mse_mean.iloc[-1] == pytest.approx(10 / (104 * 52), abs=2e-12, rel=0)


@pytest.mark.unit
def test_default_correlation_is_spearman_and_mask_count_is_pairwise(dimred_pca_class):
    x = np.arange(31.0)
    model = dimred_pca_class(1).fit(x[:, None])
    meta = pd.DataFrame({"p": x**3})
    assert model.correlations_with_params(meta, x[:, None]).iloc[0, 0] == pytest.approx(
        1, rel=0, abs=ATOL
    )
    scores = x[:, None].copy()
    scores[0, 0] = np.inf
    assert model.correlations_with_params(meta, scores).iloc[0, 0] == pytest.approx(
        1, rel=0, abs=ATOL
    )
    meta.loc[1, "p"] = np.nan
    assert np.isnan(model.correlations_with_params(meta, scores).iloc[0, 0])


@pytest.mark.unit
def test_class_statistics_match_signed_analytic_coordinates(dimred_pca_class, pca_axes):
    X, basis, coords, _ = pca_axes
    model = dimred_pca_class(3, svd_solver="full").fit(X)
    y = np.array(["A", "B", "A", "A", "B", "C"])
    result = model.class_separation(X, y, n_pcs=3)
    assert result.index.names == ["classe", "PC"]
    assert result.shape == (9, 6)
    for cls in ["A", "B", "C"]:
        for i in range(3):
            values = coords[y == cls, i] * np.sign(model.loadings[i] @ basis[i])
            actual = result.loc[(cls, f"PC{i+1}")]
            expected = [
                len(values),
                np.mean(values),
                np.std(values, ddof=0),
                np.median(values),
                np.percentile(values, 25),
                np.percentile(values, 75),
            ]
            np.testing.assert_allclose(actual, expected, atol=ATOL, rtol=0)
    order = [4, 0, 2, 5, 3, 1]
    pd.testing.assert_frame_equal(
        result, model.class_separation(X[order], y[order], 3), atol=ATOL, rtol=0
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "invalid",
    [
        np.empty((0, 3)),
        np.empty((3, 0)),
        np.ones(3),
        np.ones((2, 2, 2)),
        np.array([[1, np.nan], [2, 3]]),
        np.array([[1, np.inf], [2, 3]]),
        np.array([["a", "b"], ["c", "d"]]),
    ],
)
def test_invalid_fit_inputs_fail_loudly(dimred_pca_class, invalid):
    with pytest.raises(ValueError):
        dimred_pca_class(1).fit(invalid)


@pytest.mark.unit
@pytest.mark.parametrize(
    "n,solver",
    [(-1, "full"), (4, "full"), (1.2, "full"), (2, "invalid"), (0.8, "randomized")],
)
def test_invalid_estimator_parameters_fail(dimred_pca_class, pca_axes, n, solver):
    with pytest.raises(ValueError):
        dimred_pca_class(n, svd_solver=solver).fit(pca_axes[0])


@pytest.mark.unit
@pytest.mark.parametrize("method", ["transform", "reconstruction_error"])
@pytest.mark.parametrize(
    "invalid",
    [
        np.empty((0, 3)),
        np.ones((2, 4)),
        np.ones(3),
        np.full((2, 3), np.nan),
        np.full((2, 3), np.inf),
    ],
)
def test_invalid_projection_inputs_fail(dimred_pca_class, pca_axes, method, invalid):
    model = dimred_pca_class(2).fit(pca_axes[0])
    with pytest.raises(ValueError):
        getattr(model, method)(invalid)


@pytest.mark.integration
def test_serialization_preserves_inference_and_explicit_correlation(
    dimred_pca_class, tmp_path
):
    x = np.arange(35.0)
    X = np.column_stack([x, x * x])
    model = dimred_pca_class(2, random_state=17, svd_solver="full").fit(
        X, ["x", "square"]
    )
    path = tmp_path / "pca.joblib"
    model.save(path)
    restored = dimred_pca_class.load(path)
    assert restored.feature_names_ == ["x", "square"]
    assert restored.n_components == 2 and restored._X_fit is None
    for prop in [
        "loadings",
        "mean_",
        "explained_variance_ratio",
        "cumulative_variance",
    ]:
        np.testing.assert_array_equal(getattr(restored, prop), getattr(model, prop))
    Y = X + [20, -10]
    np.testing.assert_array_equal(restored.transform(Y), model.transform(Y))
    np.testing.assert_array_equal(
        restored.reconstruction_error(Y, 1), model.reconstruction_error(Y, 1)
    )
    pd.testing.assert_frame_equal(
        restored.loadings_dataframe(), model.loadings_dataframe()
    )
    pd.testing.assert_frame_equal(restored.variance_summary(), model.variance_summary())
    meta = pd.DataFrame({"p": x})
    scores = model.transform(X)
    pd.testing.assert_frame_equal(
        restored.correlations_with_params(meta, scores),
        model.correlations_with_params(meta, scores),
    )


@pytest.mark.unit
def test_missing_and_invalid_controlled_payloads(dimred_pca_class, tmp_path):
    with pytest.raises(FileNotFoundError):
        dimred_pca_class.load(tmp_path / "absent.joblib")
    path = tmp_path / "controlled.joblib"
    # Only locally generated, trusted payloads are deserialized.
    joblib.dump({}, path)
    # These trusted malformed payloads must fail loudly. Explicit validation
    # (ValueError) and current raw access errors are distinct allowed outcomes.
    with pytest.raises((KeyError, AttributeError, TypeError, ValueError)):
        dimred_pca_class.load(path)
    joblib.dump({"pca": None, "feature_names": []}, path)
    with pytest.raises((KeyError, AttributeError, TypeError, ValueError)):
        dimred_pca_class.load(path)


@pytest.mark.integration
def test_loader_to_pca_preserves_rows_names_scores_and_correlations(
    dimred_pca_class, dimred_feature_loader
):
    ids = np.arange(40)
    x = ids.astype(float) - 19.5
    features = pd.DataFrame(
        {"obsid": ids, "class": ["STAR"] * 40, "feature_b": x * x, "feature_a": x}
    )
    catalog = pd.DataFrame(
        {"obsid": ids[::-1], "teff_gspphot": 1000 + ids[::-1] * 100.0}
    )
    loader = dimred_feature_loader(features, catalog)
    X, y, meta = loader.load(scale=True, snr_min=0)
    model = dimred_pca_class(2, svd_solver="full")
    scores = model.fit_transform(X, loader.get_feature_names())
    assert model.loadings_dataframe().index.tolist() == ["feature_b", "feature_a"]
    np.testing.assert_allclose(
        model.sklearn_pca.inverse_transform(scores), X, atol=ATOL, rtol=0
    )
    raw = loader.inverse_transform(model.sklearn_pca.inverse_transform(scores))
    np.testing.assert_allclose(
        raw, features[["feature_b", "feature_a"]], atol=2e-12, rtol=0
    )
    np.testing.assert_array_equal(y, meta["class"])
    np.testing.assert_array_equal(meta.teff_gspphot, 1000 + ids * 100)
    result = model.correlations_with_params(meta, scores, method="pearson", n_pcs=2)
    for i in range(2):
        expected = centered_correlation(scores[:, i], x)
        assert result.iloc[i, 0] == pytest.approx(expected, abs=ATOL, rel=0)


@pytest.mark.integration
def test_reconstruction_dataframe_visualizer_contract(dimred_pca_class, pca_axes):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(__file__).resolve().parents[1] / "src/dimred/dimred_visualizer.py"
    with plt.rc_context():
        spec = importlib.util.spec_from_file_location("dimred_visualizer_audit", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        model = dimred_pca_class(2).fit(pca_axes[0])
        curve = model.reconstruction_error_vs_n(pca_axes[0], [0, 1, 2])
        fig, ax = module.DimRedVisualizer().plot_reconstruction_error(curve)
        try:
            np.testing.assert_array_equal(ax.lines[0].get_xdata(), [0, 1, 2])
            np.testing.assert_allclose(
                ax.lines[0].get_ydata(), curve.mse_mean, atol=ATOL, rtol=0
            )
        finally:
            plt.close(fig)
