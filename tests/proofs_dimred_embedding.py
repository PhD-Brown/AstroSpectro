"""Explicit M2.3 red proofs and observations; not automatic pytest discovery.

Run this file explicitly at the immutable proof ref listed in M23_PROOFS.md.
Confirmed failures, open contracts and mathematical counterexamples differ.
"""

import numpy as np
import pandas as pd
import pytest
from threadpoolctl import threadpool_limits


@pytest.mark.parametrize("method", ["umap", "tsne"])
@pytest.mark.parametrize("key,value", [("random_state", 17), ("n_components", 3)])
def test_params_used_records_seed_and_output_dimension(
    embedding_module, embedding_backends, method, key, value
):
    engine = embedding_module.EmbeddingEngine(method, 3, 17)
    engine.fit_transform(np.arange(36.0).reshape(12, 3))
    assert engine.params_used.get(key) == value


def test_tsne_n_jobs_override_is_forwarded(embedding_module, embedding_backends):
    engine = embedding_module.EmbeddingEngine("tsne", n_jobs=1)
    engine.fit_transform(np.arange(36.0).reshape(12, 3))
    assert embedding_backends[-1].kwargs["n_jobs"] == 1


def test_tsne_solver_kwarg_reaches_estimator(embedding_module, embedding_backends):
    # Public **kwargs promises direct forwarding; sklearn calls its solver method.
    options = {"method": "exact"}
    engine = embedding_module.EmbeddingEngine("tsne", **options)
    engine.fit_transform(np.arange(36.0).reshape(12, 3))
    assert embedding_backends[-1].kwargs["method"] == "exact"


@pytest.mark.parametrize("method", ["umap", "tsne"])
@pytest.mark.parametrize("previous_fit", [False, True])
def test_failed_fit_cannot_mix_model_and_cached_result(
    embedding_module, embedding_backends, monkeypatch, method, previous_fit
):
    engine = embedding_module.EmbeddingEngine(method, perplexity=3)
    X = np.arange(36.0).reshape(12, 3)
    if previous_fit:
        engine.fit_transform(X)
    old = (
        engine._model,
        engine._embedding,
        engine.params_used.copy(),
        engine.fit_time_,
    )

    def fail(self, X):
        raise ValueError("controlled estimator fit failure")

    monkeypatch.setattr(
        (
            type(embedding_backends[0])
            if previous_fit
            else _estimator_class(embedding_backends, embedding_module, X)
        ),
        "fit_transform",
        fail,
    )
    engine.extra_kwargs["perplexity"] = 4
    with pytest.raises(ValueError, match="controlled estimator fit failure"):
        engine.fit_transform(X + 10)
    # Both coherent retention and complete invalidation are admissible policies.
    if engine._embedding is None:
        assert engine._model is None and engine.fit_time_ is None
    else:
        assert engine._model is old[0]
        assert engine._embedding is old[1]
        assert engine.params_used == old[2] and engine.fit_time_ == old[3]


def _estimator_class(instances, module, X):
    module.EmbeddingEngine().fit_transform(X)
    return type(instances[-1])


def test_real_tsne_failed_refit_does_not_expose_old_coordinates(embedding_module):
    X = np.random.default_rng(17).normal(size=(16, 4))
    engine = embedding_module.EmbeddingEngine("tsne", perplexity=3, max_iter=300)
    with threadpool_limits(limits=1):
        engine.fit_transform(X)
    old_model, old_embedding = engine._model, engine.embedding
    X[0, 0] = np.nan
    with pytest.raises(ValueError):
        engine.fit_transform(X)
    if engine._embedding is None:
        assert engine._model is None
    else:
        assert engine._model is old_model and engine.embedding is old_embedding


@pytest.mark.parametrize("method", ["umap", "tsne"])
def test_3d_correlations_use_finite_pairs(
    embedding_module, embedding_backends, embedding_figures, tmp_path, method
):
    X = np.column_stack([np.ones(102), np.arange(102), np.arange(102)]).astype(float)
    X[0, 2] = np.nan
    meta = pd.DataFrame({"parallax": np.arange(102, dtype=float)})
    y = np.repeat("STAR", 102)
    engine = embedding_module.EmbeddingEngine()
    try:
        if method == "umap":
            result = engine.compute_umap3d(X, meta, y, np.zeros(102), tmp_path)
        else:
            result = engine.compute_tsne3d(X, meta, y, tmp_path)
    except ValueError:
        return
    # May reject invalid coordinates explicitly, or correlate the 101 finite pairs.
    assert result["axis3_best_param"] == "Parallaxe"
    assert result["axis3_best_corr"] == pytest.approx(1, abs=2e-15)


def test_stability_rejects_unknown_metric(embedding_module, embedding_backends):
    X = np.arange(36.0).reshape(12, 3)
    with pytest.raises(ValueError):
        embedding_module.EmbeddingEngine().stability_report(
            X, n_seeds=2, metric="not-a-metric"
        )


@pytest.mark.parametrize("n_seeds", [0, -1])
def test_open_nonpositive_seed_count_has_explicit_outcome(
    embedding_module, embedding_backends, n_seeds
):
    X = np.arange(36.0).reshape(12, 3)
    try:
        result = embedding_module.EmbeddingEngine().stability_report(X, n_seeds=n_seeds)
    except ValueError:
        return
    assert result.empty


def test_observe_single_hyperparameter_convention(embedding_module, embedding_backends):
    X = np.arange(36.0).reshape(12, 3)
    engine = embedding_module.EmbeddingEngine()
    result = engine.parameter_sensitivity(
        X, {"n_neighbors": [3, 5], "min_dist": [0.2, 0.7]}
    )
    assert list(result) == ["n_neighbors=3", "n_neighbors=5"]
    assert [e.kwargs["min_dist"] for e in embedding_backends] == [0.1, 0.1]


def test_observe_duplicate_grid_values_collapse(embedding_module, embedding_backends):
    result = embedding_module.EmbeddingEngine().parameter_sensitivity(
        np.ones((12, 3)), {"n_neighbors": [3, 3]}
    )
    assert list(result) == ["n_neighbors=3"] and len(embedding_backends) == 2


def test_observe_empty_grid_mapping(embedding_module, embedding_backends):
    with pytest.raises(IndexError):
        embedding_module.EmbeddingEngine().parameter_sensitivity(np.ones((12, 3)), {})


@pytest.mark.parametrize("count", [0, -1])
def test_observe_pca_slice_boundaries(embedding_module, embedding_backends, count):
    X = np.arange(36.0).reshape(12, 3)
    try:
        embedding_module.EmbeddingEngine().fit_transform(X, n_pca_components=count)
    except ZeroDivisionError:
        assert count == 0  # Capture backend boundary, not a green wrapper requirement.
    else:
        np.testing.assert_array_equal(embedding_backends[-1].input, X[:, :count])


def test_observe_transform_requires_preselected_features(
    embedding_module, embedding_backends
):
    X = np.arange(48.0).reshape(12, 4)
    engine = embedding_module.EmbeddingEngine()
    engine.fit_transform(X, n_pca_components=2)
    with pytest.raises(ValueError):
        engine.transform(X)
    assert engine.transform(X[:, :2]).shape == (12, 2)


def test_counterexample_univariate_permutation_keeps_bimodality(
    embedding_module, embedding_backends
):
    X = np.repeat([-10.0, 10.0], 20).reshape(-1, 1)
    embedding_module.EmbeddingEngine().negative_control(X)
    permuted = embedding_backends[-1].input
    assert np.unique(permuted).tolist() == [-10.0, 10.0]
    # Pairwise distance multiset exactly unchanged, despite the permutation.
    np.testing.assert_array_equal(
        np.sort(abs(X - X.T).ravel()), np.sort(abs(permuted - permuted.T).ravel())
    )


def test_counterexample_pca_scores_are_not_implicitly_whitened(
    dimred_pca_class, pca_axes
):
    X, _, _, _ = pca_axes
    scores = dimred_pca_class(3, svd_solver="full").fit_transform(X)
    np.testing.assert_allclose(
        np.var(scores, axis=0, ddof=1), [3.6, 1.6, 0.4], atol=2e-13, rtol=0
    )
    assert not np.allclose(np.var(scores, axis=0, ddof=1), 1)
