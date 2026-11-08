"""M2.3: wrapper contracts, independent numerical oracles and small integrations.

No test asserts astrophysical validity, uniform negative embeddings, or equality
across dependency versions. Red proofs are retained on a separate Git ref.
"""

import importlib.util
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
import pytest
from threadpoolctl import threadpool_limits


@pytest.mark.parametrize("method", ["umap", "tsne"])
def test_constructor_and_unfitted_guards(embedding_module, method):
    engine = embedding_module.EmbeddingEngine(method)
    assert (engine.method, engine.n_components, engine.random_state) == (method, 2, 42)
    assert engine.extra_kwargs == engine.params_used == {}
    assert engine.fit_time_ is None
    with pytest.raises(RuntimeError):
        _ = engine.embedding
    with pytest.raises(RuntimeError):
        engine.transform(np.ones((3, 4)))


@pytest.mark.parametrize("method", ["UMAP", "t-sne", "pca", "", None])
def test_unknown_method_rejected(embedding_module, method):
    with pytest.raises(ValueError, match="method"):
        embedding_module.EmbeddingEngine(method)


@pytest.mark.parametrize("method", ["umap", "tsne"])
@pytest.mark.parametrize("k", [None, 1, 3, 10])
def test_fit_dispatch_dimensions_parameters_and_time(
    embedding_module, embedding_backends, monkeypatch, method, k
):
    X = np.arange(60.0).reshape(12, 5)
    original = X.copy()
    kwargs = (
        dict(n_neighbors=7, min_dist=0.3, n_epochs=13, metric="manhattan")
        if method == "umap"
        else dict(perplexity=4, learning_rate=17, max_iter=300, init="random")
    )
    engine = embedding_module.EmbeddingEngine(method, 3, 17, **kwargs)
    clock = iter([10.0, 12.5])
    monkeypatch.setattr(embedding_module.time, "perf_counter", lambda: next(clock))
    try:
        Z = engine.fit_transform(X, n_pca_components=k)
    except ValueError:
        if k is not None and k > X.shape[1]:
            return  # Explicit validation and identity truncation are both admissible.
        raise
    estimator = embedding_backends[-1]
    np.testing.assert_array_equal(estimator.input, X if k is None else X[:, :k])
    np.testing.assert_array_equal(X, original)
    assert Z.shape == (12, 3)
    np.testing.assert_array_equal(engine.embedding, Z)
    assert engine.fit_time_ == 2.5
    assert engine._model is estimator
    expected = (
        dict(
            n_neighbors=15,
            min_dist=0.1,
            metric="euclidean",
            n_epochs=200,
            low_memory=False,
        )
        if method == "umap"
        else dict(
            perplexity=30,
            learning_rate="auto",
            max_iter=1000,
            metric="euclidean",
            init="pca",
        )
    )
    expected.update(kwargs)
    assert engine.params_used.items() >= expected.items()
    actual = dict(expected, n_components=3, random_state=17)
    transmitted = estimator.kwargs.copy()
    if method == "tsne":
        transmitted.pop("n_jobs", None)  # Backend parallelism policy is open in #66.
    assert transmitted == actual


@pytest.mark.parametrize("method", ["umap", "tsne"])
def test_backend_defaults_and_successful_refit(
    embedding_module, embedding_backends, method
):
    engine = embedding_module.EmbeddingEngine(method)
    first = engine.fit_transform(np.arange(24.0).reshape(6, 4))
    first_model = engine._model
    second = engine.fit_transform(np.arange(30.0).reshape(10, 3))
    assert engine._model is not first_model
    np.testing.assert_array_equal(engine.embedding, second)
    assert first.shape == (6, 2) and second.shape == (10, 2)
    defaults = (
        dict(
            n_neighbors=15,
            min_dist=0.1,
            metric="euclidean",
            n_epochs=200,
            low_memory=False,
        )
        if method == "umap"
        else dict(
            perplexity=30,
            learning_rate="auto",
            max_iter=1000,
            metric="euclidean",
            init="pca",
        )
    )
    assert engine.params_used.items() >= defaults.items()
    assert (
        embedding_backends[-1].kwargs.items()
        >= dict(random_state=42, n_components=2).items()
    )


def test_transform_delegates_without_refit(embedding_module, embedding_backends):
    engine = embedding_module.EmbeddingEngine("umap", random_state=19)
    engine.fit_transform(np.arange(24.0).reshape(6, 4))
    model, Z, elapsed = engine._model, engine.embedding.copy(), engine.fit_time_
    new = np.arange(12.0).reshape(3, 4) + 100
    np.testing.assert_array_equal(engine.transform(new), new[:, :2])
    assert len(embedding_backends) == 1 and engine._model is model
    np.testing.assert_array_equal(engine.embedding, Z)
    assert engine.fit_time_ == elapsed
    with pytest.raises(ValueError):
        engine.transform(new[:, :3])


def test_tsne_transform_is_explicitly_unsupported(embedding_module, embedding_backends):
    engine = embedding_module.EmbeddingEngine("tsne")
    engine.fit_transform(np.arange(24.0).reshape(6, 4))
    with pytest.raises(NotImplementedError, match="t-SNE"):
        engine.transform(np.ones((3, 4)))


def test_umap_missing_import_is_informative(embedding_module, monkeypatch):
    monkeypatch.setitem(sys.modules, "umap", None)
    with pytest.raises(ImportError, match="umap-learn"):
        embedding_module.EmbeddingEngine().fit_transform(np.ones((4, 3)))


@pytest.mark.parametrize("method", ["umap", "tsne"])
@pytest.mark.parametrize(
    "operation", ["stability_report", "negative_control", "parameter_sensitivity"]
)
def test_analysis_methods_preserve_fitted_parent(
    embedding_module, embedding_backends, method, operation
):
    X = np.arange(120.0).reshape(30, 4)
    engine = embedding_module.EmbeddingEngine(
        method, random_state=13, metric="manhattan"
    )
    engine.fit_transform(X)
    model, Z, params, elapsed = (
        engine._model,
        engine.embedding.copy(),
        engine.params_used.copy(),
        engine.fit_time_,
    )
    kwargs = {"n_seeds": 2} if operation == "stability_report" else {}
    if operation == "parameter_sensitivity":
        kwargs = {
            "param_grid": {"n_neighbors" if method == "umap" else "perplexity": [3, 5]}
        }
    getattr(engine, operation)(X, **kwargs)
    assert engine._model is model and engine.fit_time_ == elapsed
    np.testing.assert_array_equal(engine.embedding, Z)
    assert engine.params_used == params


@pytest.mark.parametrize("method", ["umap", "tsne"])
def test_3d_missing_plotly_is_informative(
    embedding_module, monkeypatch, tmp_path, method
):
    monkeypatch.setitem(sys.modules, "plotly.express", None)
    with pytest.raises(ImportError, match="plotly"):
        call_3d(
            embedding_module,
            method,
            np.ones((8, 3)),
            pd.DataFrame(index=range(8)),
            np.repeat("STAR", 8),
            tmp_path,
        )


@pytest.mark.parametrize("n_rows", [8, 5000, 5013])
@pytest.mark.parametrize("n_seeds", [1, 2, 3])
def test_stability_sampling_seeds_geometry_and_parent_isolation(
    embedding_module, monkeypatch, n_rows, n_seeds
):
    X = np.random.default_rng(20).normal(size=(n_rows, 4))
    original = X.copy()
    calls = []

    def backend(engine, data):
        calls.append((engine, data.copy()))
        base = data[:, :2]
        if engine.random_state == 1:
            return 3 * base @ np.array([[0, 1], [1, 0]]) + [90, -10]
        if engine.random_state == 2:
            return base * [1, 3] + [7, 9]
        return base.copy()

    monkeypatch.setattr(embedding_module.EmbeddingEngine, "_fit_umap", backend)
    engine = embedding_module.EmbeddingEngine("umap", random_state=31, min_dist=0.27)
    report = engine.stability_report(X, n_seeds=n_seeds, n_pca_components=3)
    repeated = engine.stability_report(X, n_seeds=n_seeds, n_pca_components=3)
    pd.testing.assert_frame_equal(report, repeated)
    assert report.columns.tolist() == ["seed", "procrustes_distance", "method"]
    assert report.seed.tolist() == list(range(n_seeds))
    assert report.method.tolist() == ["UMAP"] * n_seeds
    assert report.procrustes_distance.iloc[0] == 0
    row_ids = {value: i for i, value in enumerate(X[:, 0])}
    selected = np.array([row_ids[value] for value in calls[0][1][:, 0]])
    assert len(selected) == min(n_rows, 5000)
    assert len(np.unique(selected)) == len(selected)
    for seed, (child, data) in enumerate(calls[:n_seeds]):
        assert child is not engine and child.random_state == seed
        assert child.extra_kwargs == {"min_dist": 0.27}
        np.testing.assert_array_equal(data, X[selected, :3])
        np.testing.assert_array_equal(calls[n_seeds + seed][1], data)
    assert len({id(c[0]) for c in calls}) == 2 * n_seeds
    if n_seeds > 1:
        # Common row reordering changes float64 centering roundoff, not geometry.
        assert report.procrustes_distance.iloc[1] == pytest.approx(0, abs=2e-25)
    if n_seeds > 2:
        # Independent least-squares similarity solution via singular values.
        A = X[selected, :2] - X[selected, :2].mean(axis=0)
        B = A * [1, 3]
        A, B = A / np.linalg.norm(A), B / np.linalg.norm(B)
        optimum = 1 - np.linalg.svd(A.T @ B, compute_uv=False).sum() ** 2
        assert report.procrustes_distance.iloc[2] == pytest.approx(optimum, abs=3e-14)
        assert optimum > 0.01
    np.testing.assert_array_equal(X, original)
    assert (
        engine._model is None and engine.params_used == {} and engine.fit_time_ is None
    )
    if n_rows > 5000:
        changed = embedding_module.EmbeddingEngine("umap", random_state=32)
        changed.stability_report(X, n_seeds=1, n_pca_components=3)
        assert set(calls[-1][1][:, 0]) != set(calls[0][1][:, 0])


def test_alignment_failure_is_missing_not_false_zero(embedding_module, monkeypatch):
    monkeypatch.setattr(
        embedding_module.EmbeddingEngine,
        "_fit_umap",
        lambda self, X: np.zeros((len(X), 2)),
    )
    try:
        report = embedding_module.EmbeddingEngine().stability_report(
            np.ones((4, 3)), n_seeds=2
        )
    except ValueError:
        return  # A domain exception is a valid alternative to missing disparity.
    assert np.isnan(report.procrustes_distance.iloc[1])


@pytest.mark.parametrize("method", ["umap", "tsne"])
def test_stability_child_dimension_method_and_options(
    embedding_module, embedding_backends, monkeypatch, method
):
    children = []
    helper_name = f"_fit_{method}"
    helper = getattr(embedding_module.EmbeddingEngine, helper_name)

    def capture(child, X):
        children.append(child)
        return helper(child, X)

    monkeypatch.setattr(embedding_module.EmbeddingEngine, helper_name, capture)
    options = {"min_dist": 0.23} if method == "umap" else {"perplexity": 4}
    X = np.random.default_rng(27).normal(size=(32, 5))
    parent = embedding_module.EmbeddingEngine(
        method, n_components=3, random_state=29, metric="manhattan", **options
    )
    parent.stability_report(X, n_seeds=3, n_pca_components=4)
    assert len(children) == len(embedding_backends) == 3
    for seed, (child, estimator) in enumerate(zip(children, embedding_backends)):
        assert (child.method, child.n_components, child.random_state) == (
            method,
            3,
            seed,
        )
        assert child.extra_kwargs == {"metric": "manhattan", **options}
        assert estimator.kwargs["n_components"] == 3
        assert estimator.kwargs["random_state"] == seed
        assert estimator.kwargs["metric"] == "manhattan"
        assert estimator.input.shape == (32, 4)
    assert len({id(child) for child in children}) == 3


def test_stability_fit_failure_propagates(embedding_module, monkeypatch):
    def fail(self, X):
        raise ValueError("estimator failed")

    monkeypatch.setattr(embedding_module.EmbeddingEngine, "_fit_umap", fail)
    with pytest.raises(ValueError, match="estimator failed"):
        embedding_module.EmbeddingEngine().stability_report(np.ones((4, 3)))


@pytest.mark.parametrize("method", ["umap", "tsne"])
@pytest.mark.parametrize(
    "shape,k", [((40, 3), None), ((40, 3), 2), ((9, 1), 1), ((2, 3), None)]
)
def test_negative_control_marginals_permutation_and_parameters(
    embedding_module, embedding_backends, method, shape, k
):
    X = np.repeat(np.arange(shape[0], dtype=float)[:, None], shape[1], axis=1)
    X[:, -1] = 2.0 if shape[1] > 1 else X[:, -1]
    original = X.copy()
    engine = embedding_module.EmbeddingEngine(method, 3, 23, metric="manhattan")
    result = engine.negative_control(X, n_pca_components=k)
    permuted = embedding_backends[-1].input
    engine.negative_control(X, n_pca_components=k)
    np.testing.assert_array_equal(embedding_backends[-1].input, permuted)
    embedding_module.EmbeddingEngine(
        method, 3, 24, metric="manhattan"
    ).negative_control(X, n_pca_components=k)
    if shape[0] == 40:
        assert not np.array_equal(embedding_backends[-1].input, permuted)
    np.testing.assert_array_equal(np.sort(permuted, axis=0), np.sort(X[:, :k], axis=0))
    np.testing.assert_array_equal(X, original)
    assert result.shape == (shape[0], 3)
    actual = embedding_backends[0].kwargs
    assert (actual["random_state"], actual["n_components"], actual["metric"]) == (
        23,
        3,
        "manhattan",
    )
    if shape == (40, 3):
        # The selected deterministic example destroys a perfect row association.
        assert not np.array_equal(permuted[:, 0], permuted[:, 1])
        assert abs(np.corrcoef(permuted[:, :2].T)[0, 1]) < 0.5
    assert engine._model is None and engine.params_used == {}


def ordinal_correlation(x, y):
    """Compute Pearson correlation of independently constructed tie-free ranks."""
    assert len(np.unique(x)) == len(x) and len(np.unique(y)) == len(y)
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean()
    ry -= ry.mean()
    return np.dot(rx, ry) / np.sqrt(np.dot(rx, rx) * np.dot(ry, ry))


@pytest.mark.parametrize("method", ["umap", "tsne"])
def test_negative_control_breaks_all_six_multivariable_associations(
    embedding_module, embedding_backends, method
):
    X = np.repeat(np.arange(40.0)[:, None], 4, axis=1)
    original = X.copy()
    engine = embedding_module.EmbeddingEngine(method, random_state=23)
    engine.negative_control(X)
    permuted = embedding_backends[-1].input.copy()
    engine.negative_control(X)
    np.testing.assert_array_equal(embedding_backends[-1].input, permuted)
    np.testing.assert_array_equal(np.sort(permuted, axis=0), original)
    np.testing.assert_array_equal(X, original)
    # Six initially perfect associations; 0.5 is a generous separation from 1
    # for this fixed N=40 example, not a universal permutation-test threshold.
    # No assertion requires every column's row permutation to be nonidentity.
    for left in range(4):
        for right in range(left + 1, 4):
            assert ordinal_correlation(X[:, left], X[:, right]) == 1
            assert abs(ordinal_correlation(permuted[:, left], permuted[:, right])) < 0.5


def intermediate_rank_example():
    rng = np.random.default_rng(315)
    t = np.linspace(-3, 3, 300)
    X = np.column_stack([np.sin(t), t**2, t])
    meta = pd.DataFrame(
        {
            "teff_gspphot": t + rng.normal(0, 3, len(t)),
            "logg_gspphot": -np.exp(t) + rng.normal(0, 1.4, len(t)),
            "mh_gspphot": rng.normal(size=len(t)),
        }
    )
    # A strictly increasing cube preserves ranks while separating raw covariance.
    meta["logg_gspphot"] = meta["logg_gspphot"] ** 3
    return X, meta


@pytest.mark.parametrize("method", ["umap", "tsne"])
def test_3d_intermediate_spearman_value_has_independent_rank_oracle(
    embedding_module, embedding_backends, embedding_figures, tmp_path, method
):
    X, meta = intermediate_rank_example()
    x, y = X[:, 2], meta.logg_gspphot.to_numpy()
    expected = ordinal_correlation(x, y)
    i, j = np.triu_indices(len(x), k=1)
    kendall = np.sign((x[i] - x[j]) * (y[i] - y[j])).mean()
    xc, yc = x - x.mean(), y - y.mean()
    pearson = np.dot(xc, yc) / np.sqrt(np.dot(xc, xc) * np.dot(yc, yc))
    assert -0.95 < expected < -0.6
    assert abs(expected - kendall) > 0.08
    assert abs(expected - pearson) > 0.04
    result = call_3d(
        embedding_module,
        method,
        X,
        meta[["logg_gspphot"]],
        np.repeat("STAR", len(x)),
        tmp_path,
    )
    assert result["axis3_best_corr"] == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("method", ["umap", "tsne"])
def test_3d_winner_is_later_negative_parameter(
    embedding_module, embedding_backends, embedding_figures, tmp_path, method
):
    X, meta = intermediate_rank_example()
    coefficients = {
        col: ordinal_correlation(X[:, 2], meta[col].to_numpy()) for col in meta
    }
    assert 0.35 < coefficients["teff_gspphot"] < 0.75
    assert coefficients["logg_gspphot"] < 0
    assert (
        abs(coefficients["logg_gspphot"])
        > max(abs(coefficients[c]) for c in ("teff_gspphot", "mh_gspphot")) + 0.15
    )
    result = call_3d(
        embedding_module, method, X, meta, np.repeat("STAR", len(X)), tmp_path
    )
    assert result["axis3_best_param"] == "log g"


@pytest.mark.parametrize("method", ["umap", "tsne"])
def test_3d_spearman_rank_not_pearson_and_population_permutation(
    embedding_module, embedding_backends, embedding_figures, tmp_path, method
):
    x = np.linspace(-3, 3, 120)
    X = np.column_stack([np.sin(x), x**2, x])
    meta = pd.DataFrame({"teff_gspphot": x**3})
    y = np.repeat("STAR", 120)
    result = call_3d(embedding_module, method, X, meta, y, tmp_path)
    assert result["axis3_best_param"] == "T_eff (K)"
    assert result["axis3_best_corr"] == pytest.approx(1, abs=2e-15)
    permutation = np.random.default_rng(29).permutation(120)
    joint = call_3d(
        embedding_module,
        method,
        X[permutation],
        meta.iloc[permutation],
        y[permutation],
        tmp_path,
    )
    assert joint["axis3_best_param"] == result["axis3_best_param"]
    assert joint["axis3_best_corr"] == pytest.approx(1, abs=2e-15)
    # Equal lengths cannot establish object identity: positional API limitation.
    shuffled = call_3d(embedding_module, method, X, meta.iloc[permutation], y, tmp_path)
    assert abs(shuffled["axis3_best_corr"]) < 0.3


@pytest.mark.parametrize(
    "method,key", [("umap", "n_neighbors"), ("tsne", "perplexity")]
)
@pytest.mark.parametrize("values", [None, [], [3, 7]])
def test_sensitivity_grid_override_and_isolation(
    embedding_module, embedding_backends, method, key, values
):
    X = np.arange(150.0).reshape(30, 5)
    original = X.copy()
    engine = embedding_module.EmbeddingEngine(
        method, 3, 19, **{key: 99, "metric": "cosine"}
    )
    grid = None if values is None else {key: values}
    results = engine.parameter_sensitivity(X, grid, n_pca_components=3)
    expected = [5, 15, 30, 50, 100] if values is None else values
    assert list(results) == [f"{key}={v}" for v in expected]
    assert len(embedding_backends) == len(expected)
    for estimator, v in zip(embedding_backends, expected):
        assert estimator.kwargs[key] == v
        assert estimator.kwargs["random_state"] == 19
        assert estimator.kwargs["metric"] == "cosine"
        assert estimator.kwargs["n_components"] == 3
        np.testing.assert_array_equal(estimator.input, X[:, :3])
        np.testing.assert_array_equal(results[f"{key}={v}"], estimator.output)
    if len(expected) > 1:
        assert not np.shares_memory(*list(results.values())[:2])
    np.testing.assert_array_equal(X, original)
    assert engine.extra_kwargs[key] == 99 and engine._model is None


@pytest.mark.parametrize(
    "method,key,values,initial",
    [
        ("umap", "min_dist", [0.2, 0.6], 0.4),
        ("tsne", "early_exaggeration", [8.0, 16.0], 12.0),
    ],
)
def test_sensitivity_nondefault_key_and_trial_isolation(
    embedding_module, embedding_backends, method, key, values, initial
):
    X = np.arange(120.0).reshape(24, 5)
    engine = embedding_module.EmbeddingEngine(
        method, 3, 17, metric="cosine", **{key: initial}
    )
    results = engine.parameter_sensitivity(X, {key: values}, n_pca_components=4)
    assert list(results) == [f"{key}={value}" for value in values]
    assert len(embedding_backends) == 2
    for estimator, value in zip(embedding_backends, values):
        assert estimator.kwargs[key] == value
        assert estimator.kwargs["metric"] == "cosine"
        assert (estimator.kwargs["n_components"], estimator.kwargs["random_state"]) == (
            3,
            17,
        )
        np.testing.assert_array_equal(estimator.input, X[:, :4])
        np.testing.assert_array_equal(results[f"{key}={value}"], estimator.output)
    assert embedding_backends[0] is not embedding_backends[1]
    embedding_backends[0].kwargs[key] = -999
    assert embedding_backends[1].kwargs[key] == values[1]
    assert not np.shares_memory(*results.values())
    assert engine.extra_kwargs == {"metric": "cosine", key: initial}


@pytest.mark.parametrize("methods", [None, ["tsne"], ["tsne", "umap"], []])
def test_compare_order_dimensions_and_parameter_forwarding(
    embedding_module, embedding_backends, methods
):
    X = np.arange(100.0).reshape(20, 5)
    result = embedding_module.compare_embeddings(
        X, methods, n_pca_components=3, random_state=29
    )
    expected = ["umap", "tsne"] if methods is None else methods
    assert list(result) == expected
    assert len(embedding_backends) == len(expected)
    for key, estimator in zip(expected, embedding_backends):
        np.testing.assert_array_equal(estimator.input, X[:, :3])
        np.testing.assert_array_equal(result[key], estimator.output)
        assert estimator.kwargs["random_state"] == 29
    if len(expected) > 1:
        assert not np.shares_memory(*result.values())


def call_3d(module, method, X, meta, y, tmp_path, **kwargs):
    engine = module.EmbeddingEngine()
    if method == "umap":
        return engine.compute_umap3d(
            X, meta, y, np.arange(len(X)) % 4 - 1, tmp_path, **kwargs
        )
    return engine.compute_tsne3d(X, meta, y, tmp_path, **kwargs)


@pytest.mark.parametrize("method", ["umap", "tsne"])
@pytest.mark.parametrize("n_subsample", [0, 7, 200])
def test_3d_tables_alignment_sampling_and_exports(
    embedding_module,
    embedding_backends,
    embedding_figures,
    tmp_path,
    method,
    n_subsample,
):
    n = 120
    rng = np.random.default_rng(10)
    X = rng.normal(size=(n, 5))
    X[:, 2] = np.arange(n)
    y = np.array(["STAR", "QSO", "GALAXY"] * 40)
    meta = pd.DataFrame(
        {"teff_gspphot": 1000 - X[:, 2], "mh_gspphot": rng.normal(size=n)},
        index=np.arange(n) + 700,
    )
    meta.loc[meta.index[2], "teff_gspphot"] = np.nan
    meta.loc[meta.index[3], "mh_gspphot"] = np.nan
    before = meta.copy(deep=True)
    try:
        result = call_3d(
            embedding_module,
            method,
            X,
            meta,
            y,
            tmp_path,
            random_state=17,
            n_subsample=n_subsample,
        )
    except ValueError:
        if n_subsample == 0:
            return  # Explicit rejection of an empty plot is also admissible.
        raise
    np.testing.assert_array_equal(result[f"Z_{method}3"], X[:, :3])
    assert result["axis3_best_param"] == "T_eff (K)"
    assert result["axis3_best_corr"] == pytest.approx(-1, abs=2e-15)
    assert result[f"{method}_3d_fit_time"] >= 0
    estimator = embedding_backends[-1]
    np.testing.assert_array_equal(estimator.input, X)
    assert (
        estimator.kwargs["n_components"] == 3 and estimator.kwargs["random_state"] == 17
    )
    if method == "umap":
        assert (
            estimator.kwargs["n_neighbors"] == 15
            and estimator.kwargs["min_dist"] == 0.1
        )
    else:
        assert estimator.kwargs["perplexity"] == 30
    table = embedding_figures[0].table
    prefix = "UMAP" if method == "umap" else "tSNE"
    selected = table[f"{prefix}3"].to_numpy().astype(int)
    assert len(selected) == min(n, n_subsample)
    assert len(set(selected)) == len(selected)
    assert set(selected) <= set(range(n))
    np.testing.assert_array_equal(
        table[[f"{prefix}{i}" for i in (1, 2, 3)]], X[selected, :3]
    )
    np.testing.assert_array_equal(table.Classe, y[selected])
    np.testing.assert_array_equal(
        table["T_eff (K)"], meta.teff_gspphot.to_numpy()[selected]
    )
    np.testing.assert_array_equal(table["[Fe/H]"], meta.mh_gspphot.to_numpy()[selected])
    if method == "umap":
        labels = np.arange(n) % 4 - 1
        assert table.Cluster.tolist() == [
            "Bruit" if c == -1 else f"C{c}" for c in labels[selected]
        ]
    assert Path(result["path_3d_html"]) == tmp_path / f"{method}3d_classes.html"
    assert Path(result["path_3d_teff"]) == tmp_path / f"{method}3d_teff.html"
    if method == "umap":
        assert Path(result["path_3d_feh"]) == tmp_path / "umap3d_feh.html"
    colors = ["Classe", "T_eff (K)"] + (["[Fe/H]"] if method == "umap" else [])
    first_figures = list(embedding_figures)
    assert len(first_figures) == len(colors)
    for fig, color in zip(first_figures, colors):
        assert {axis: fig.kwargs[axis] for axis in ("x", "y", "z")} == {
            axis: f"{prefix}{i}" for i, axis in enumerate(("x", "y", "z"), 1)
        }
        assert fig.kwargs["color"] == color
        expected = table if color == "Classe" else table.dropna(subset=[color])
        pd.testing.assert_frame_equal(fig.table.sort_index(), expected.sort_index())
        if color != "Classe":
            expected_range = table[color].quantile([0.02, 0.98]).to_numpy()
            np.testing.assert_allclose(
                fig.kwargs["range_color"],
                expected_range,
                rtol=0,
                atol=1e-12,
                equal_nan=True,
            )
    if n_subsample >= n:
        assert 2 not in first_figures[1].table.index
        if method == "umap":
            assert 3 not in first_figures[2].table.index
            assert 2 in first_figures[2].table.index
            assert 3 in first_figures[1].table.index
    call_3d(
        embedding_module,
        method,
        X,
        meta,
        y,
        tmp_path,
        random_state=17,
        n_subsample=n_subsample,
    )
    for first, repeated in zip(first_figures, embedding_figures[len(first_figures) :]):
        pd.testing.assert_frame_equal(
            first.table.sort_index(), repeated.table.sort_index()
        )
    if 0 < n_subsample < n:
        call_3d(
            embedding_module,
            method,
            X,
            meta,
            y,
            tmp_path,
            random_state=19,
            n_subsample=n_subsample,
        )
        assert set(embedding_figures[2 * len(first_figures)].table.index) != set(
            table.index
        )
    assert all(p.parent == tmp_path for fig in embedding_figures for p in fig.paths)
    pd.testing.assert_frame_equal(meta, before)


@pytest.mark.parametrize("method", ["umap", "tsne"])
@pytest.mark.parametrize("valid_count", [99, 100, 101])
def test_3d_correlation_threshold_and_missing_params(
    embedding_module,
    embedding_backends,
    embedding_figures,
    tmp_path,
    method,
    valid_count,
):
    X = np.arange(360.0).reshape(120, 3)
    vals = -X[:, 2].copy()
    vals[valid_count:] = np.inf
    result = call_3d(
        embedding_module,
        method,
        X,
        pd.DataFrame({"parallax": vals}),
        np.repeat("STAR", 120),
        tmp_path,
    )
    assert result["axis3_best_param"] == ("?" if valid_count < 100 else "Parallaxe")
    assert result["axis3_best_corr"] == pytest.approx(
        0 if valid_count < 100 else -1, abs=2e-15
    )
    assert result["path_3d_teff"] is None
    if method == "umap":
        assert result["path_3d_feh"] is None


@pytest.mark.parametrize("method", ["umap", "tsne"])
def test_3d_constants_are_undefined_and_do_not_win(
    embedding_module, embedding_backends, embedding_figures, tmp_path, method
):
    from scipy.stats import ConstantInputWarning

    X = np.arange(360.0).reshape(120, 3)
    meta = pd.DataFrame({"teff_gspphot": np.ones(120), "logg_gspphot": -X[:, 2]})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConstantInputWarning)
        result = call_3d(
            embedding_module, method, X, meta, np.repeat("STAR", 120), tmp_path
        )
    assert result["axis3_best_param"] == "log g" and result[
        "axis3_best_corr"
    ] == pytest.approx(-1)


@pytest.mark.parametrize("method", ["umap", "tsne"])
@pytest.mark.parametrize("field", ["meta", "y", "n_subsample"])
def test_3d_incompatible_inputs_are_not_silently_truncated(
    embedding_module, embedding_backends, embedding_figures, tmp_path, method, field
):
    X = np.arange(360.0).reshape(120, 3)
    meta = pd.DataFrame({"parallax": X[:, 2]})
    y = np.repeat("STAR", 120)
    if field == "meta":
        meta = meta.iloc[:-1]
    if field == "y":
        y = y[:-1]
    kwargs = {"n_subsample": -1} if field == "n_subsample" else {}
    with pytest.raises((ValueError, IndexError)):
        call_3d(embedding_module, method, X, meta, y, tmp_path, **kwargs)


@pytest.mark.parametrize(
    "bad", ["nan", "inf", "empty", "zero_features", "1d", "perplexity", "iterations"]
)
def test_real_tsne_invalid_inputs_rejected(embedding_module, bad):
    X = np.random.default_rng(4).normal(size=(12, 4))
    kwargs = dict(perplexity=3, max_iter=250, init="random")
    if bad in ("nan", "inf"):
        X[0, 0] = np.nan if bad == "nan" else np.inf
    elif bad == "empty":
        X = X[:0]
    elif bad == "zero_features":
        X = X[:, :0]
    elif bad == "1d":
        X = X[:, 0]
    elif bad == "perplexity":
        kwargs["perplexity"] = len(X)
    else:
        kwargs["max_iter"] = 249
    with pytest.raises(ValueError):
        embedding_module.EmbeddingEngine("tsne", **kwargs).fit_transform(X)


@pytest.mark.parametrize(
    "grid",
    [{"perplexity": [3, 12]}, {"perplexity": ["invalid"]}, {"unknown_option": [3]}],
)
def test_real_tsne_grid_incompatibility_propagates(embedding_module, grid):
    X = np.random.default_rng(9).normal(size=(12, 4))
    original = X.copy()
    engine = embedding_module.EmbeddingEngine("tsne", max_iter=250, init="random")
    with threadpool_limits(limits=1), pytest.raises((ValueError, TypeError)):
        engine.parameter_sensitivity(X, grid)
    np.testing.assert_array_equal(X, original)
    assert engine._model is None and engine.params_used == {}


@pytest.mark.integration
@pytest.mark.parametrize("dim", [2, 3])
def test_real_pca_to_tsne_reproducibility(embedding_module, dimred_pca_class, dim):
    X = np.random.default_rng(73).normal(size=(24, 6))
    scores = dimred_pca_class(4, svd_solver="full").fit_transform(X)
    original = scores.copy()
    with threadpool_limits(limits=1):
        engine = embedding_module.EmbeddingEngine(
            "tsne", dim, 11, perplexity=4, max_iter=300
        )
        Z = engine.fit_transform(scores, n_pca_components=3)
        repeat = embedding_module.EmbeddingEngine(
            "tsne", dim, 11, perplexity=4, max_iter=300
        ).fit_transform(scores[:, :3])
    assert Z.shape == (24, dim) and np.isfinite(Z).all()
    # Conditional reproducibility only, within this process/version/thread setup.
    np.testing.assert_allclose(Z, repeat, atol=2e-6, rtol=0)
    np.testing.assert_array_equal(scores, original)
    assert engine._model.n_features_in_ == 3


@pytest.mark.integration
def test_real_tsne3d_plotly_export(embedding_module, tmp_path):
    X = np.random.default_rng(52).normal(size=(16, 4))
    meta = pd.DataFrame({"teff_gspphot": np.linspace(3000, 6000, 16)})
    with threadpool_limits(limits=1):
        result = call_3d(
            embedding_module,
            "tsne",
            X,
            meta,
            np.repeat("STAR", 16),
            tmp_path,
            perplexity=3,
            n_subsample=8,
        )
    assert result["Z_tsne3"].shape == (16, 3) and np.isfinite(result["Z_tsne3"]).all()
    for key in ("path_3d_html", "path_3d_teff"):
        text = Path(result[key]).read_text(encoding="utf-8")
        assert "Plotly.newPlot" in text and "scatter3d" in text


@pytest.mark.integration
def test_embeddings_reporter_round_trip(embedding_module, embedding_backends, tmp_path):
    import json
    import joblib

    path = Path(__file__).resolve().parents[1] / "src/dimred/run_reporter.py"
    spec = importlib.util.spec_from_file_location("m23_reporter", path)
    reporter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reporter)
    X = np.arange(48.0).reshape(12, 4)
    meta = pd.DataFrame({"obsid": np.arange(12)[::-1], "parallax": X[:, 0]})
    y = np.array(["STAR", "QSO", "GALAXY"] * 4)
    engine = embedding_module.EmbeddingEngine("umap", random_state=17, min_dist=0.23)
    Z = engine.fit_transform(X, n_pca_components=3)
    stability = engine.stability_report(X, n_seeds=2)
    sensitivity = engine.parameter_sensitivity(X, {"n_neighbors": [3, 5]})
    result = reporter.save_umap_tsne_run(
        Z_umap=Z,
        Z_tsne=None,
        y=y,
        meta=meta,
        scores_95=X,
        umap_engine=engine,
        paths={"REPORTS_DIR": tmp_path / "reports"},
        FIGURES_DIR=tmp_path,
        stab_umap=stability,
        sensitivity_umap=sensitivity,
    )
    payload = joblib.load(result["joblib_run"])
    np.testing.assert_array_equal(payload["Z_umap"], Z)
    np.testing.assert_array_equal(payload["y"], y)
    pd.testing.assert_frame_equal(payload["meta"], meta)
    pd.testing.assert_frame_equal(payload["stability_umap"], stability)
    assert payload["umap_params"]["min_dist"] == 0.23
    for key in sensitivity:
        np.testing.assert_array_equal(
            payload["sensitivity_umap"][key], sensitivity[key]
        )
    report = json.loads(Path(result["json_path"]).read_text(encoding="utf-8"))
    assert report["shapes"]["Z_umap"] == [12, 2]
    assert report["parameters"]["umap"]["min_dist"] == 0.23


@pytest.mark.integration
@pytest.mark.slow
def test_optional_real_umap_fit_transform_and_reproducibility(embedding_module):
    pytest.importorskip("umap", reason="umap-learn is optional and absent from CI lock")
    X = np.random.default_rng(24).normal(size=(32, 4)).astype(np.float32)
    kwargs = dict(n_neighbors=5, n_epochs=30, n_jobs=1, init="random")
    engine = embedding_module.EmbeddingEngine("umap", random_state=17, **kwargs)
    with threadpool_limits(limits=1):
        Z = engine.fit_transform(X)
        repeated = embedding_module.EmbeddingEngine(
            "umap", random_state=17, **kwargs
        ).fit_transform(X)
        new = engine.transform(X[:3] + 0.01)
    assert Z.shape == (32, 2) and np.isfinite(Z).all()
    assert new.shape == (3, 2) and np.isfinite(new).all()
    np.testing.assert_allclose(Z, repeated, atol=2e-6, rtol=0)


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.parametrize(
    "case", ["nan", "inf", "empty", "neighbors", "min_dist", "metric", "constant"]
)
def test_optional_real_umap_boundary_inputs(embedding_module, case):
    pytest.importorskip("umap", reason="optional umap-learn is absent from CI lock")
    X = np.random.default_rng(9).normal(size=(24, 4)).astype(np.float32)
    kwargs = dict(n_neighbors=5, n_epochs=20, n_jobs=1, init="random")
    if case == "constant":
        X[:] = 1
    elif case in ("nan", "inf"):
        X[0, 0] = np.nan if case == "nan" else np.inf
    elif case == "empty":
        X = X[:0]
    elif case == "neighbors":
        kwargs["n_neighbors"] = 1
    elif case == "min_dist":
        kwargs["min_dist"] = -0.1
    else:
        kwargs["metric"] = "not-a-metric"
    engine = embedding_module.EmbeddingEngine(**kwargs)
    with threadpool_limits(limits=1):
        if case == "constant":
            # Acceptance policy is open; accepted coordinates must be usable.
            try:
                Z = engine.fit_transform(X)
            except ValueError:
                return
            assert Z.shape == (24, 2) and np.isfinite(Z).all()
        else:
            with pytest.raises(ValueError):
                engine.fit_transform(X)


@pytest.mark.integration
def test_stability_table_visualizer_contract(embedding_module, monkeypatch, tmp_path):
    import matplotlib.pyplot as plt

    path = Path(__file__).resolve().parents[1] / "src/dimred/dimred_visualizer.py"
    spec = importlib.util.spec_from_file_location("m23_visualizer", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(
        embedding_module.EmbeddingEngine, "_fit_umap", lambda self, X: X[:, :2].copy()
    )
    X = np.random.default_rng(3).normal(size=(12, 4))
    report = embedding_module.EmbeddingEngine().stability_report(X, n_seeds=3)
    viz = module.DimRedVisualizer()
    fig, ax = viz.plot_stability(report, save_path=str(tmp_path / "stability.png"))
    try:
        np.testing.assert_allclose(
            [p.get_height() for p in ax.patches],
            report.procrustes_distance,
            atol=1e-30,
            rtol=0,
        )
        assert (tmp_path / "stability.png").is_file()
    finally:
        plt.close(fig)
