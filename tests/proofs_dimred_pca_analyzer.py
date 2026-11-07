"""Explicit M2.2 probes: confirmed defects and OPEN proposed contracts.

Not collected by default. Run explicitly; failures are evidence, not xfails.
The fixed proof branch and per-test classification are in audit/M22_PROOFS.md.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError


def test_unreachable_variance_cannot_return_nonexistent_component(
    dimred_pca_class, pca_axes
):
    model = dimred_pca_class(1, svd_solver="full").fit(pca_axes[0])
    assert model.cumulative_variance[-1] < 0.9
    try:
        result = model.n_components_for_variance(0.9)
    except ValueError:
        return  # An explicit unattainable-target diagnostic is also coherent.
    assert (
        result is None or 1 <= result <= model.loadings.shape[0]
    ), "Reported component does not exist in fitted model"


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


def test_default_class_separation_on_two_component_model(dimred_pca_class, pca_axes):
    # OPEN: clamp to K or explicitly reject n_pcs>K? The default currently
    # crashes inside the aggregation. This probe proposes clamping, not approval.
    model = dimred_pca_class(2).fit(pca_axes[0])
    result = model.class_separation(pca_axes[0], np.array(["A"] * 6))
    assert len(result) == 2


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
