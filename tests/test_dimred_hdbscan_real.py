"""Optional HDBSCAN integration, separate from mandatory CI wrapper contracts.

Select this file with -m slow after installing hdbscan in an isolated target.
Geometry-specific partitions are examples, never universal determinism promises.
"""

import numpy as np
import pytest

pytestmark = [pytest.mark.slow, pytest.mark.integration]


def real_backend():
    return pytest.importorskip(
        "hdbscan", reason="Optional hdbscan integration requires separate installation"
    )


def geometry():
    rng = np.random.default_rng(17)
    return np.r_[
        rng.normal([-10, 0], 0.1, (12, 2)),
        rng.normal([10, 0], 0.1, (12, 2)),
        [[0, 100]],
    ]


@pytest.mark.parametrize("method", ["eom", "leaf"])
def test_real_fit_geometry_probabilities_and_presentation(hdbscan_module, method):
    backend = real_backend()
    Z = geometry()
    before = Z.copy()
    a = hdbscan_module.HDBSCANAnalyzer(4, 2, cluster_selection_method=method).fit(Z)
    assert a.n_clusters_ >= 2 and a.labels_[-1] == -1
    # Partition oracle disregards arbitrary integer naming.
    if method == "eom":
        assert a.n_clusters_ == 2 and a.n_noise_ == 1
        assert len(set(a.labels_[:12])) == len(set(a.labels_[12:24])) == 1
        assert a.labels_[0] != a.labels_[12]
    for cid in a.cluster_ids_:
        rows = np.flatnonzero(a.labels_ == cid)
        assert (rows < 12).all() or ((12 <= rows) & (rows < 24)).all()
    assert a.cluster_ids_ == sorted(set(a.labels_) - {-1})
    probs = a.clusterer_.probabilities_
    assert probs.shape == (25,) and np.isfinite(probs).all()
    assert ((0 <= probs) & (probs <= 1)).all()
    assert (probs[a.labels_ == -1] == 0).all()
    assert a.clusterer_.prediction_data is True
    assert a.clusterer_.prediction_data_ is not None
    ref = backend.HDBSCAN(
        min_cluster_size=4,
        min_samples=2,
        metric="euclidean",
        cluster_selection_method=method,
        prediction_data=True,
    ).fit_predict(Z)
    np.testing.assert_array_equal(a.labels_[:, None] == a.labels_, ref[:, None] == ref)
    result = a.fit_presentation(Z, min_cluster_size=5)
    assert result.shape == (25,) and a.clusterer_pres_.prediction_data is False
    np.testing.assert_array_equal(Z, before)


@pytest.mark.parametrize("case", ["constant", "duplicates"])
def test_real_degenerate_geometry(hdbscan_module, case):
    real_backend()
    Z = (
        np.ones((8, 2))
        if case == "constant"
        else np.r_[geometry()[:24], geometry()[:3]]
    )
    a = hdbscan_module.HDBSCANAnalyzer(4, 2).fit(Z)
    assert len(a.labels_) == len(Z)
    assert a.n_noise_ == int(np.count_nonzero(a.labels_ == -1))
    assert a.n_clusters_ == len(set(a.labels_) - {-1})
    assert np.isfinite(a.clusterer_.probabilities_).all()
    if case == "constant":
        assert a.n_clusters_ == 0 and a.n_noise_ == 8


@pytest.mark.parametrize(
    "case",
    [
        "empty",
        "one",
        "1d",
        "3d",
        "nan",
        "inf",
        "negative_inf",
        "strings",
        "metric",
        "min_cluster_size",
        "min_samples",
        "selection",
    ],
)
def test_real_invalid_inputs_and_parameters(hdbscan_module, case):
    real_backend()
    Z = geometry()
    kwargs = {"min_cluster_size": 4, "min_samples": 2}
    if case == "empty":
        Z = np.empty((0, 2))
    if case == "one":
        Z = np.ones((1, 2))
    if case == "1d":
        Z = np.arange(8.0)
    if case == "3d":
        Z = np.ones((3, 2, 2))
    if case in ["nan", "inf", "negative_inf"]:
        Z[3, 0] = {"nan": np.nan, "inf": np.inf, "negative_inf": -np.inf}[case]
    if case == "strings":
        Z = np.array([["bad", "text"]] * 8)
    if case == "metric":
        kwargs["metric"] = "no-such-metric"
    if case == "min_cluster_size":
        kwargs["min_cluster_size"] = 1
    if case == "min_samples":
        kwargs["min_samples"] = 0
    if case == "selection":
        kwargs["cluster_selection_method"] = "invalid"
    with pytest.raises((ValueError, TypeError)):
        hdbscan_module.HDBSCANAnalyzer(**kwargs).fit(Z)


def test_real_sensitivity_matches_direct_backend(hdbscan_module):
    backend = real_backend()
    Z = geometry()
    result = hdbscan_module.compute_sensitivity(Z, min_samples=2, min_sizes=[4, 8, 4])
    for row, mcs in zip(result.to_dict("records"), [4, 8, 4]):
        labels = backend.HDBSCAN(
            min_cluster_size=mcs,
            min_samples=2,
            metric="euclidean",
            cluster_selection_method="eom",
        ).fit_predict(Z)
        assert row["n_noise"] == np.count_nonzero(labels == -1)
        assert row["n_clusters"] == len(set(labels) - {-1})
        assert row["pct_noise"] == round(100 * np.count_nonzero(labels == -1) / 25, 2)


def test_real_paired_permutation_partition_on_this_geometry(hdbscan_module):
    real_backend()
    Z = geometry()
    p = np.random.default_rng(23).permutation(len(Z))
    a = hdbscan_module.HDBSCANAnalyzer(4, 2).fit(Z).labels_
    b = hdbscan_module.HDBSCANAnalyzer(4, 2).fit(Z[p]).labels_[np.argsort(p)]
    np.testing.assert_array_equal(a == -1, b == -1)
    np.testing.assert_array_equal(a[:, None] == a, b[:, None] == b)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_real_presentation_nonfinite_rows_are_backend_noise(hdbscan_module, bad):
    backend = real_backend()
    Z = geometry()
    Z[3, 0] = bad
    a = hdbscan_module.HDBSCANAnalyzer(4, 2)
    result = a.fit_presentation(Z, min_cluster_size=4)
    ref = backend.HDBSCAN(min_cluster_size=4, min_samples=2).fit_predict(Z)
    np.testing.assert_array_equal(result[:, None] == result, ref[:, None] == ref)
    assert result[3] == -1 and len(result) == len(Z)
