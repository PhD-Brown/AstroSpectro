"""Explicit M2.2 probes: confirmed defects and OPEN proposed contracts.

Not collected by default. Run explicitly; failures are evidence, not xfails.
The fixed proof branch and per-test classification are in audit/M22_PROOFS.md.
"""

import ast
from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError


def assert_variance_outcome(model, threshold, full_rank=False):
    """Check semantic answers without choosing exception versus missing sentinel."""
    try:
        result = model.n_components_for_variance(threshold)
    except ValueError:
        return  # Explicit validation, never an internal IndexError/AttributeError.
    if result is None or (
        isinstance(result, (float, np.floating)) and np.isnan(result)
    ):
        return
    assert isinstance(result, (int, np.integer)) and not isinstance(result, bool)
    K = model.loadings.shape[0]
    assert 1 <= result <= K, "Reported component does not exist in fitted model"
    if full_rank:
        # Independent full-column-rank oracle: all strictly positive sample
        # eigenvalues are needed at 100%, regardless of a 1-ulp ratio deficit.
        assert result == K
    else:
        assert (
            model.cumulative_variance[result - 1] >= threshold
        ), "Returned count does not attain the target"


def test_unreachable_variance_cannot_return_nonexistent_component(
    dimred_pca_class, pca_axes
):
    model = dimred_pca_class(1, svd_solver="full").fit(pca_axes[0])
    assert model.cumulative_variance[-1] < 0.9
    assert_variance_outcome(model, 0.9)


def test_full_rank_target_one_roundoff(dimred_pca_class):
    # Claude's integer counterexample, independently reproduced on Windows.
    X = np.array(
        [[6, -8, -6], [-5, -6, 6], [7, 2, -9], [-8, -3, -1], [2, 0, -4]], float
    )
    assert np.linalg.matrix_rank(X - X.mean(axis=0)) == 3
    assert np.linalg.eigvalsh(np.cov(X, rowvar=False)).min() > 0
    model = dimred_pca_class(3, svd_solver="full").fit(X)
    # No universal epsilon: the full-rank mathematical target is known exactly.
    assert_variance_outcome(model, 1.0, full_rank=True)


def test_variance_oracle_rejects_capping_but_accepts_explicit_diagnostics(
    dimred_pca_class, pca_axes
):
    model = dimred_pca_class(1, svd_solver="full").fit(pca_axes[0])
    assert_variance_outcome(model, 0.5)  # Reachable positive control.
    original = model.n_components_for_variance
    model.n_components_for_variance = lambda threshold: min(1, original(threshold))
    with pytest.raises(AssertionError, match="does not attain"):
        assert_variance_outcome(model, 0.9)
    for sentinel in (None, np.nan):
        model.n_components_for_variance = lambda threshold: sentinel
        assert_variance_outcome(model, 0.9)

    def reject(threshold):
        raise ValueError("unattainable target")

    model.n_components_for_variance = reject
    assert_variance_outcome(model, 0.9)


def test_observe_nan_variance_threshold(dimred_pca_class, pca_axes):
    # OBSERVATION / OPEN: NaN is not a variance target; domain policy undecided.
    model = dimred_pca_class(3).fit(pca_axes[0])
    assert model.n_components_for_variance(np.nan) == 4


def test_loaded_configuration_is_complete(dimred_pca_class, pca_axes, tmp_path):
    model = dimred_pca_class(2, random_state=17, svd_solver="full").fit(pca_axes[0])
    path = tmp_path / "fitted.joblib"
    model.save(path)
    restored = dimred_pca_class.load(path)
    assert (
        getattr(restored, "random_state", None),
        getattr(restored, "svd_solver", None),
    ) == (17, "full")


def test_loaded_model_can_refit(dimred_pca_class, pca_axes, tmp_path):
    model = dimred_pca_class(2, random_state=17, svd_solver="full").fit(pca_axes[0])
    path = tmp_path / "fitted.joblib"
    model.save(path)
    restored = dimred_pca_class.load(path)
    restored.fit(pca_axes[0] + [4, 5, 6])
    np.testing.assert_allclose(restored.mean_, pca_axes[3] + [4, 5, 6], atol=2e-13)


@pytest.mark.parametrize("dtype", ["bool", "Int64", "Float64"])
def test_mixed_numeric_metadata_correlations(dimred_pca_class, dtype):
    x = np.arange(35.0)
    model = dimred_pca_class(1).fit(x[:, None])
    other = (x % 2 == 0) if dtype == "bool" else pd.array(x, dtype=dtype)
    meta = pd.DataFrame({"float": x, "other": other})
    result = model.correlations_with_params(meta, x[:, None], method="pearson")
    assert result.loc["PC1", "float"] == pytest.approx(1)


@pytest.mark.parametrize("previous_fit", [False, True])
def test_failed_fit_does_not_expose_an_unfitted_estimator_as_fitted(
    dimred_pca_class, pca_axes, previous_fit
):
    model = dimred_pca_class(2)
    if previous_fit:
        model.fit(pca_axes[0], ["a", "b", "c"])
    with pytest.raises(ValueError):
        model.fit(np.full((6, 3), np.nan))
    # Either reject access or retain the previous *fitted* model. Transaction
    # policy is open, but returning a new unfitted PCA is never fitted-state proof.
    try:
        estimator = model.sklearn_pca
    except (RuntimeError, NotFittedError):
        return
    assert hasattr(
        estimator, "components_"
    ), "Fitted accessor returned an unfitted estimator"


def assert_default_class_outcome(model, pca_axes):
    X, basis, coordinates, _ = pca_axes
    y = np.array(["A", "B", "A", "A", "B", "C"])
    try:
        result = model.class_separation(X, y)
    except ValueError:
        return  # Explicit validation is as legitimate as a bounded table.
    assert result.index.names == ["classe", "PC"]
    assert result.columns.tolist() == ["n", "mean", "std", "median", "q25", "q75"]
    assert result.index.tolist() == [
        (cls, f"PC{i+1}") for cls in ["A", "B", "C"] for i in range(2)
    ]
    assert result.shape == (6, 6)
    for cls in ["A", "B", "C"]:
        for i in range(2):
            values = coordinates[y == cls, i] * np.sign(model.loadings[i] @ basis[i])
            expected = [
                len(values),
                values.mean(),
                values.std(),
                np.median(values),
                np.percentile(values, 25),
                np.percentile(values, 75),
            ]
            np.testing.assert_allclose(
                result.loc[(cls, f"PC{i+1}")], expected, atol=2e-13, rtol=0
            )


def test_default_class_separation_on_two_component_model(dimred_pca_class, pca_axes):
    model = dimred_pca_class(2).fit(pca_axes[0])
    assert_default_class_outcome(model, pca_axes)


def test_class_oracle_accepts_both_remedies_and_checks_statistics(
    dimred_pca_class, pca_axes
):
    model = dimred_pca_class(2).fit(pca_axes[0])
    original = model.class_separation
    model.class_separation = lambda X, y: original(X, y, n_pcs=2)
    assert_default_class_outcome(model, pca_axes)
    valid = original(pca_axes[0], np.array(["A", "B", "A", "A", "B", "C"]), 2)
    invalid = valid.copy()
    invalid.loc[("A", "PC1"), "mean"] += 1
    model.class_separation = lambda X, y: invalid
    with pytest.raises(AssertionError):
        assert_default_class_outcome(model, pca_axes)

    def reject(X, y):
        raise ValueError("requested PCs exceed fitted model")

    model.class_separation = reject
    assert_default_class_outcome(model, pca_axes)


def test_observe_missing_class_labels_empty_quantile(dimred_pca_class, pca_axes):
    # OBSERVATION / OPEN: rejection versus announced exclusion remains undecided.
    model = dimred_pca_class(2).fit(pca_axes[0])
    y = np.array([1.0, 1.0, 2.0, 2.0, np.nan, np.nan])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        with pytest.raises(IndexError):
            model.class_separation(pca_axes[0], y, n_pcs=2)


@pytest.mark.parametrize(
    "name",
    [
        "feature_ratio_Fe_Mg",
        "feature_ratio_Ca_Fe",
        "feature_ratio_Ba_Fe",
        "feature_ratio_Sr_Fe",
        "feature_flux_ratio_blue_red",
        "feature_rotation_index",
        "feature_rotation_proxy",
        "ratio_EW_Ha_Hb",
        "ratio_EW_MgB_NaD",
        "ratio_FWHM_Ha_Hb",
    ],
)
def test_non_tio_producer_features_not_assigned_to_titanium(dimred_pca_class, name):
    # Source provenance verified statically: no expensive feature producer run.
    producer = (
        Path(__file__).resolve().parents[1] / "src/pipeline/feature_engineering.py"
    )
    tree = ast.parse(producer.read_text(encoding="utf-8"))
    assigned_keys = {
        node.slice.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Subscript)
        and isinstance(node.ctx, ast.Store)
        and isinstance(node.slice, ast.Constant)
    }
    assert name in assigned_keys
    assert "TiO" not in name
    model = dimred_pca_class()
    titanium = next(f for f in model._FAMILIES if f.startswith("Titane"))
    assert re.search("TiO", name, re.IGNORECASE) is not None
    # Neutral about the destination family; only accidental TiO is excluded.
    assert model._assign_family(name) != titanium


def test_observe_open_lick_priorities_and_constant_input(dimred_pca_class):
    model = dimred_pca_class(3)
    assert model._assign_family("hBETA_index").startswith("Balmer")
    assert model._assign_family("TiO_1_Lick").startswith("Titane")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        model.fit(np.full((5, 3), 7.0))
    assert not np.isfinite(model.explained_variance_ratio).any()


@pytest.mark.parametrize("kind", ["array", "index", "empty"])
def test_observe_feature_name_container_contract(dimred_pca_class, pca_axes, kind):
    # N3 / C3: ndarray and Index are outside List[str]; common usage still needs
    # an explicit decision. Empty lists currently trigger generated names.
    model = dimred_pca_class(2)
    if kind == "empty":
        model.fit(pca_axes[0], [])
        assert model.feature_names_ == ["f0", "f1", "f2"]
    else:
        names = (
            np.array(["a", "b", "c"]) if kind == "array" else pd.Index(["a", "b", "c"])
        )
        with pytest.raises(ValueError, match="ambiguous"):
            model.fit(pca_axes[0], names)


def test_observe_negative_count_labels(dimred_pca_class, pca_axes):
    # N4 / C1: silently mislabeled output, not approval of slice semantics.
    model = dimred_pca_class(3).fit(pca_axes[0])
    curve = model.reconstruction_error_vs_n(pca_axes[0], [-1])
    assert curve.n_components.tolist() == [-1]
    np.testing.assert_allclose(
        curve.mse_mean.iloc[0],
        model.reconstruction_error(pca_axes[0], 2).mean(),
        atol=2e-13,
        rtol=0,
    )
    corr = model.correlations_with_params(pd.DataFrame({"p": np.arange(6.0)}), n_pcs=-1)
    assert corr.index.tolist() == ["PC1", "PC2"]


@pytest.mark.parametrize("n", [-1, 4])
def test_reconstruction_rejects_unavailable_component_counts(
    dimred_pca_class, pca_axes, n
):
    # OPEN: explicit validation rather than Python slice semantics.
    model = dimred_pca_class(3).fit(pca_axes[0])
    with pytest.raises(ValueError):
        model.reconstruction_error(pca_axes[0], n)


def test_empty_reconstruction_curve_retains_schema(dimred_pca_class, pca_axes):
    # OPEN: schema-preserving empty result versus explicit ValueError.
    model = dimred_pca_class(3).fit(pca_axes[0])
    result = model.reconstruction_error_vs_n(pca_axes[0], [])
    assert result.columns.tolist() == ["n_components", "mse_mean", "mse_std"]


def test_unfitted_default_curve_has_domain_guard(dimred_pca_class):
    with pytest.raises(RuntimeError, match="fit"):
        dimred_pca_class(2).reconstruction_error_vs_n(np.ones((3, 2)))


def test_feature_name_count_is_validated_at_fit(dimred_pca_class, pca_axes):
    # OPEN: currently failure is deferred to DataFrame creation.
    with pytest.raises(ValueError):
        dimred_pca_class(2).fit(pca_axes[0], ["one"])


def test_training_cache_not_aliased_to_caller(dimred_pca_class):
    # OPEN: defensive snapshot versus documented immutability requirement.
    x = np.arange(35.0)
    X = np.column_stack([x, x * x])
    model = dimred_pca_class(2, svd_solver="full").fit(X)
    meta = pd.DataFrame({"p": x})
    before = model.correlations_with_params(meta)
    X[:] = X[::-1]
    after = model.correlations_with_params(meta)
    pd.testing.assert_frame_equal(after, before)


def test_names_not_aliased_to_caller(dimred_pca_class, pca_axes):
    # OPEN: defensive snapshot versus documented immutability requirement.
    names = ["a", "b", "c"]
    model = dimred_pca_class(2).fit(pca_axes[0], names)
    names[0] = "wrong"
    assert model.loadings_dataframe().index.tolist() == ["a", "b", "c"]


def test_observe_positional_alignment_is_not_identity_alignment(dimred_pca_class):
    # PASS control: current docstring requires the same row order, not an ID join.
    x = np.arange(35.0)
    model = dimred_pca_class(1).fit(x[:, None])
    meta = pd.DataFrame({"p": x})
    aligned = model.correlations_with_params(meta, x[:, None])
    permuted = model.correlations_with_params(meta.iloc[::-1], x[:, None])
    assert aligned.iloc[0, 0] == pytest.approx(1)
    assert permuted.iloc[0, 0] == pytest.approx(-1)


def test_observe_loaded_implicit_correlations_need_training_data(
    dimred_pca_class, tmp_path
):
    # PASS limitation: documented scores must be supplied after load.
    x = np.arange(35.0)
    model = dimred_pca_class(1).fit(x[:, None])
    path = tmp_path / "fitted.joblib"
    model.save(path)
    restored = dimred_pca_class.load(path)
    assert restored._X_fit is None
    with pytest.raises(ValueError):
        restored.correlations_with_params(pd.DataFrame({"p": x}))


def test_observe_negative_indices_duplicate_names_and_zero_components(
    dimred_pca_class, pca_axes
):
    model = dimred_pca_class(3).fit(pca_axes[0], ["same", "same", "c"])
    assert model.loadings_dataframe().index.tolist() == ["same", "same", "c"]
    top = model.top_features_per_pc(-1)
    assert set(top.PC) == {"PC0"}
    zero = dimred_pca_class(0, svd_solver="full").fit(pca_axes[0])
    assert zero.transform(pca_axes[0]).shape == (6, 0)
    # Initial probe incorrectly assumed sklearn inverse_transform accepts K=0.
    # It rejects empty score matrices; zero-rank reconstruction is OPEN.
    with pytest.raises(ValueError, match="0 feature"):
        zero.reconstruction_error(pca_axes[0])


def test_observe_incompatible_metadata_and_labels_fail(dimred_pca_class, pca_axes):
    model = dimred_pca_class(2).fit(pca_axes[0])
    with pytest.raises(ValueError):
        model.correlations_with_params(
            pd.DataFrame({"p": np.arange(35.0)}), np.ones((34, 1))
        )
    with pytest.raises(IndexError):
        model.class_separation(pca_axes[0], np.array(["A", "B"]), 2)


def test_observe_controlled_unfitted_and_wrong_payloads(dimred_pca_class, tmp_path):
    import joblib
    from sklearn.decomposition import PCA

    path = tmp_path / "controlled.joblib"
    dimred_pca_class(2).save(path)
    with pytest.raises(AttributeError):
        dimred_pca_class.load(path)
    joblib.dump({"pca": PCA(), "feature_names": []}, path)
    with pytest.raises(AttributeError):
        dimred_pca_class.load(path)
    joblib.dump([1, 2], path)
    with pytest.raises(TypeError):
        dimred_pca_class.load(path)
