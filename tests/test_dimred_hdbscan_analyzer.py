"""Independent wrapper/statistical contracts; real HDBSCAN is selected separately.

Numerical tolerances concern float64 arithmetic, never astrophysical accuracy.
Identity is a precondition of positional APIs, not inferred from equal lengths.
"""

import builtins
import math
import sys

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal


def fitted(module, backend, labels, **kwargs):
    backend[1].append(labels)
    return module.HDBSCANAnalyzer(**kwargs).fit(np.zeros((len(labels), 2)))


def test_initial_state_and_instance_isolation(hdbscan_module):
    a, b = hdbscan_module.HDBSCANAnalyzer(), hdbscan_module.HDBSCANAnalyzer()
    assert (
        a.min_cluster_size,
        a.min_samples,
        a.metric,
        a.cluster_selection_method,
    ) == (75, 20, "euclidean", "eom")
    for name in ["clusterer_", "labels_", "clusterer_pres_", "labels_pres_"]:
        assert getattr(a, name) is None
    assert (a.n_clusters_, a.n_noise_, a.n_clusters_pres_) == (0, 0, 0)
    assert a.cluster_ids_ == a.ids_pres_ == []
    assert a.color_map_ == a.color_map_pres_ == {}
    a.cluster_ids_.append(7)
    a.color_map_[7] = "sentinel"
    assert b.cluster_ids_ == [] and b.color_map_ == {}


@pytest.mark.parametrize("custom", [False, True])
def test_fit_forwarding_and_success_state(hdbscan_module, hdbscan_backend, custom):
    kwargs = (
        dict(
            min_cluster_size=9,
            min_samples=3,
            metric="manhattan",
            cluster_selection_method="leaf",
        )
        if custom
        else {}
    )
    a = hdbscan_module.HDBSCANAnalyzer(**kwargs)
    Z = np.array([[1.0, 7.0], [2.0, -9.0], [3.0, 8.0]])
    before = Z.copy()
    hdbscan_backend[1].append([7, -1, 7])
    assert a.fit(Z) is a
    expected = dict(
        min_cluster_size=9 if custom else 75,
        min_samples=3 if custom else 20,
        metric="manhattan" if custom else "euclidean",
        cluster_selection_method="leaf" if custom else "eom",
        prediction_data=True,
    )
    assert a.clusterer_.kwargs == expected
    np.testing.assert_array_equal(a.clusterer_.input, before)
    np.testing.assert_array_equal(a.labels_, [7, -1, 7])
    np.testing.assert_array_equal(Z, before)


@pytest.mark.parametrize(
    "labels,ids,noise",
    [
        ([-1, -1, -1], [], 3),
        ([7, 7, 7], [7], 0),
        ([-1, 2, 2, 7, 7], [2, 7], 1),
        ([7, 2, 7, 2], [2, 7], 0),
        ([9, -1, 2, 9, -1, 5], [2, 5, 9], 2),
    ],
)
def test_counts_and_palette_ids(hdbscan_module, hdbscan_backend, labels, ids, noise):
    a = fitted(hdbscan_module, hdbscan_backend, labels)
    assert a.cluster_ids_ == ids
    assert a.n_clusters_ == len(ids) and a.n_noise_ == noise
    assert set(a.color_map_) == set(ids) | {-1}
    assert a.color_map_[-1] == (0.82, 0.82, 0.82, 0.25)
    colors = [a.color_map_[cid] for cid in ids]
    cmap = hdbscan_module.plt.get_cmap("turbo", max(1, len(ids)))
    for rank, cid in enumerate(ids):
        assert a.color_map_[cid] == cmap(rank / max(1, len(ids) - 1))
    assert len(set(colors)) == len(ids)
    assert all(len(c) == 4 and all(0 <= v <= 1 for v in c) for c in colors)
    # Same population/order-independent labels must retain ID-to-color pairing.
    b = fitted(hdbscan_module, hdbscan_backend, labels[::-1])
    assert b.color_map_ == a.color_map_


def test_helpers_recompute_and_successive_fit_replace_state(
    hdbscan_module, hdbscan_backend
):
    a = fitted(hdbscan_module, hdbscan_backend, [-1, 2, 7, 7])
    old = a.clusterer_
    hdbscan_backend[1].append([9, 9])
    a.fit(np.zeros((2, 2)))
    assert a.clusterer_ is not old
    assert a.cluster_ids_ == [9] and a.n_clusters_ == 1 and a.n_noise_ == 0
    assert set(a.color_map_) == {-1, 9}
    a.labels_ = np.array([-1, 11, 3, 11, -1])
    a._compute_stats()
    a._build_palette()
    assert a.cluster_ids_ == [3, 11] and a.n_noise_ == 2
    assert set(a.color_map_) == {-1, 3, 11}


@pytest.mark.parametrize("labels", [[-1, -1, -1], [7, 2, 7], [7, 7, 7], [-1, 2, 7]])
@pytest.mark.parametrize("size", [None, 13])
def test_presentation_is_separate(hdbscan_module, hdbscan_backend, labels, size):
    a = fitted(
        hdbscan_module,
        hdbscan_backend,
        [-1, 8, 8],
        min_samples=4,
        metric="manhattan",
        cluster_selection_method="leaf",
    )
    original = (a.clusterer_, a.labels_.copy(), a.color_map_.copy())
    hdbscan_backend[1].append(labels)
    Z = np.array([[1.0, 2.0], [5.0, 6.0], [9.0, 10.0]])
    result = a.fit_presentation(
        Z, **({} if size is None else {"min_cluster_size": size})
    )
    assert a.clusterer_pres_.kwargs["min_cluster_size"] == (
        300 if size is None else size
    )
    for key, value in dict(
        min_samples=4, metric="manhattan", cluster_selection_method="leaf"
    ).items():
        assert a.clusterer_pres_.kwargs[key] == value
    np.testing.assert_array_equal(a.clusterer_pres_.input, Z)
    np.testing.assert_array_equal(result, labels)
    assert result is a.labels_pres_
    ids = sorted(set(labels) - {-1})
    assert a.ids_pres_ == ids and a.n_clusters_pres_ == len(ids)
    assert set(a.color_map_pres_) == set(ids) | {-1}
    assert len({a.color_map_pres_[c] for c in ids}) == len(ids)
    cmap = hdbscan_module.plt.get_cmap("tab20", len(ids))
    for rank, cid in enumerate(ids):
        assert a.color_map_pres_[cid] == cmap(rank)
    assert a.color_map_pres_[-1] == (0.82, 0.82, 0.82, 0.25)
    assert a.clusterer_ is original[0] and a.color_map_ == original[2]
    np.testing.assert_array_equal(a.labels_, original[1])
    hdbscan_backend[1].append([4, 4])
    a.fit(np.zeros((2, 2)))
    np.testing.assert_array_equal(a.labels_pres_, labels)
    hdbscan_backend[1].append([12, 12, 12])
    a.fit_presentation(Z)
    assert a.ids_pres_ == [12] and set(a.color_map_pres_) == {-1, 12}


@pytest.mark.parametrize("mode", ["fit", "fit_presentation", "sensitivity"])
def test_optional_backend_error(hdbscan_module, monkeypatch, mode):
    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "hdbscan":
            raise ImportError("isolated optional dependency")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    with pytest.raises(ImportError, match="hdbscan"):
        if mode == "sensitivity":
            hdbscan_module.compute_sensitivity(np.ones((3, 2)))
        else:
            getattr(hdbscan_module.HDBSCANAnalyzer(), mode)(np.ones((3, 2)))


@pytest.mark.parametrize(
    "mask,expected",
    [
        ([True, True, False, False], [0.0, 4.0]),
        ([False, False, True, False], [12.0, -8.0]),
        ([True, False, True, True], [22 / 3, -2 / 3]),
    ],
)
def test_centroid_analytic_subsets_and_translation(hdbscan_module, mask, expected):
    Z = np.array([[-2.0, 2.0], [2.0, 6.0], [12.0, -8.0], [12.0, 4.0]])
    mask = np.array(mask)
    before, mask_before = Z.copy(), mask.copy()
    a = hdbscan_module.HDBSCANAnalyzer()
    np.testing.assert_allclose(a.centroid(Z, mask), expected, rtol=0, atol=2e-14)
    np.testing.assert_allclose(
        a.centroid(Z + [17.0, -9.0], mask),
        np.array(expected) + [17.0, -9.0],
        rtol=0,
        atol=2e-14,
    )
    np.testing.assert_array_equal(Z, before)
    np.testing.assert_array_equal(mask, mask_before)


@pytest.mark.parametrize("mask", [np.ones(3, dtype=bool)])
def test_centroid_invalid_mask_is_not_silently_accepted(hdbscan_module, mask):
    with pytest.raises((IndexError, ValueError)):
        hdbscan_module.HDBSCANAnalyzer().centroid(np.ones((4, 2)), mask)


def test_profiles_require_fit(hdbscan_module):
    with pytest.raises(RuntimeError, match="fit"):
        hdbscan_module.HDBSCANAnalyzer().physical_profile_table(
            pd.DataFrame(), np.array([])
        )


@pytest.mark.parametrize("nullable", [False, True])
def test_physical_profiles_analytic_statistics_and_identity(
    hdbscan_module, hdbscan_backend, nullable
):
    labels = np.array([7, 2, 7, -1, 2, 7])
    a = fitted(hdbscan_module, hdbscan_backend, labels)
    meta = pd.DataFrame(
        {
            "obsid": [101, 102, 103, 104, 105, 106],
            "teff_gspphot": [6000, 1000, 6200, 9900, 1400, 6400],
            "logg_gspphot": [4.0, 1.0, 5.0, 9.0, 3.0, np.nan],
            "mh_gspphot": [0.0, -2.0, 0.0, -8.0, -1.0, 0.0],
        },
        index=[8, 2, 91, 17, 35, 4],
    )
    if nullable:
        for col in ["teff_gspphot", "logg_gspphot", "mh_gspphot"]:
            meta[col] = meta[col].astype("Float64")
    y = np.array(["G", "M", "G", "QSO", "M", "F"])
    old_meta, old_y = meta.copy(deep=True), y.copy()
    phys_cols = [
        ("teff_gspphot", "T_eff (K)"),
        ("logg_gspphot", "log g (dex)"),
        ("mh_gspphot", "metallicity (dex)"),
    ]
    table, interp = a.physical_profile_table(meta, y, phys_cols=phys_cols)
    expected = pd.DataFrame(
        {
            "n spectres": [2, 3, 1],
            "Classe dom.": ["M", "G", "—"],
            "T_eff (K)": ["1200 ± 283", "6200 ± 200", "9900"],
            "log g (dex)": ["2.00 ± 1.41", "4.50 ± 0.71", "9.00"],
            "metallicity (dex)": ["-1.50 ± 0.71", "0.00 ± 0.00", "-8.00"],
        },
        index=pd.Index(["C2", "C7", "Bruit"], name="Cluster"),
    )
    assert_frame_equal(table, expected)
    assert interp.index.tolist() == ["C2", "C7", "Bruit"]
    assert interp["n"].tolist() == [2, 3, 1]
    assert_frame_equal(meta, old_meta)
    np.testing.assert_array_equal(y, old_y)
    np.testing.assert_array_equal(a.labels_, labels)
    # Paired row permutation, then arbitrary renaming of cluster IDs.
    p = [5, 2, 0, 4, 1, 3]
    b = fitted(hdbscan_module, hdbscan_backend, labels[p])
    t2, i2 = b.physical_profile_table(meta.iloc[p], y[p], phys_cols=phys_cols)
    assert_frame_equal(t2, table)
    assert_frame_equal(i2, interp)
    b.labels_ = np.array([19 if x == 2 else 4 if x == 7 else -1 for x in labels])
    t3, _ = b.physical_profile_table(meta, y, phys_cols=phys_cols)
    assert t3.loc["C19"].to_dict() == table.loc["C2"].to_dict()
    assert t3.loc["C4"].to_dict() == table.loc["C7"].to_dict()


@pytest.mark.parametrize("labels", [[-1, -1], [2, 2], [2, 7]])
def test_profiles_noise_singleton_and_missing_physics(
    hdbscan_module, hdbscan_backend, labels
):
    a = fitted(hdbscan_module, hdbscan_backend, labels)
    meta = pd.DataFrame({"custom": [2.0, 6.0], "all_missing": [np.nan, np.nan]})
    table, interp = a.physical_profile_table(
        meta,
        np.array(["G", "G"]),
        phys_cols=[("custom", "X"), ("all_missing", "Y"), ("absent", "Z")],
    )
    assert "Z" not in table.columns
    assert table["n spectres"].sum() == 2
    assert "Bruit" in table.index and len(table) == len(set(labels) - {-1}) + 1
    assert interp["n"].tolist() == table["n spectres"].tolist()
    if labels == [2, 2]:
        assert table.loc["C2", "X"] == "4.00 ± 2.83"
        assert table.loc["C2", "Y"] == "nan ± nan"
        assert table.loc["Bruit", "X"] == "N/A"
    if labels == [2, 7]:
        assert table.loc["C2", "X"] == "2.00 ± nan"
        assert table.loc["C7", "X"] == "6.00 ± nan"
    if labels == [-1, -1]:
        assert table.loc["Bruit", "X"] == "4.00"


def test_profiles_unavailable_columns_and_partial_missing_class(
    hdbscan_module, hdbscan_backend
):
    a = fitted(hdbscan_module, hdbscan_backend, [2, 2, 2])
    table, interp = a.physical_profile_table(
        pd.DataFrame({"obsid": [9, 12, 6], "value": [1.0, 2.0, 3.0]}),
        np.array(["G", None, "G"], dtype=object),
        phys_cols=[("value", "V"), ("absent", "Z")],
    )
    assert table.columns.tolist() == ["n spectres", "Classe dom.", "V"]
    assert table.loc["C2", "Classe dom."] == "G"
    assert table.loc["C2", "n spectres"] == 3
    assert interp.loc["C2", "Interprétation"] == "paramètres Gaia insuffisants"


@pytest.mark.parametrize("which", ["meta", "y"])
def test_profiles_reject_incompatible_lengths(hdbscan_module, hdbscan_backend, which):
    a = fitted(hdbscan_module, hdbscan_backend, [2, 2, 2])
    with pytest.raises(ValueError):
        a.physical_profile_table(
            pd.DataFrame(
                {"teff_gspphot": [1.0, 2.0] if which == "meta" else [1.0, 2.0, 3.0]}
            ),
            np.array(["G"] * (2 if which == "y" else 3)),
        )


def test_profile_print_transmits_tables_without_mutating(
    hdbscan_module, hdbscan_backend, capsys
):
    a = fitted(hdbscan_module, hdbscan_backend, [2, 2, -1])
    tables = a.physical_profile_table(
        pd.DataFrame({"teff_gspphot": [1000.0, 1400.0, 9900.0]}),
        np.array(["M", "M", "QSO"]),
    )
    copies = [t.copy(deep=True) for t in tables]
    assert a.print_physical_profile(*tables) is None
    output = capsys.readouterr().out
    assert "HDBSCAN" in output and "1200 ± 283" in output and "9900" in output
    for index, row in tables[1].iterrows():
        assert index in output and row["Interprétation"] in output
    for t, old in zip(tables, copies):
        assert_frame_equal(t, old)


@pytest.mark.parametrize("grid", [None, [7, 3, 7], []])
def test_sensitivity_grid_forwarding_counts_and_rounding(
    hdbscan_module, hdbscan_backend, grid
):
    sizes = [50, 100, 150, 300, 500] if grid is None else grid
    outcomes = [
        [-1, 2, 7, 7, 2, 2],
        [2, 2, 2, 2, 2, 2],
        [-1] * 6,
        [7, 7, -1, -1, 2, 2],
        [7] * 6,
    ]
    hdbscan_backend[1].extend(outcomes[: len(sizes)])
    Z = np.arange(12.0).reshape(6, 2)
    old = Z.copy()
    table = hdbscan_module.compute_sensitivity(Z, min_samples=4, min_sizes=grid)
    assert len(hdbscan_backend[0]) == len(sizes)
    if sizes:
        assert table.columns.tolist() == [
            "min_cluster_size",
            "n_clusters",
            "n_noise",
            "pct_noise",
        ]
        expected = pd.DataFrame(
            {
                "min_cluster_size": sizes,
                "n_clusters": [2, 1, 0, 2, 1][: len(sizes)],
                "n_noise": [1, 0, 6, 2, 0][: len(sizes)],
                "pct_noise": [16.67, 0.0, 100.0, 33.33, 0.0][: len(sizes)],
            }
        )
        assert_frame_equal(table, expected)
    else:
        assert table.empty  # Empty schema is an OPEN contract.
    for instance, size in zip(hdbscan_backend[0], sizes):
        assert instance.kwargs == dict(
            min_cluster_size=size,
            min_samples=4,
            metric="euclidean",
            cluster_selection_method="eom",
        )
        np.testing.assert_array_equal(instance.input, old)
    np.testing.assert_array_equal(Z, old)


def test_sensitivity_default_min_samples_and_property_independence(
    hdbscan_module, hdbscan_backend
):
    a = hdbscan_module.HDBSCANAnalyzer()
    assert a.sensitivity_df.empty
    hdbscan_backend[1].append([-1, 2, 2])
    result = hdbscan_module.compute_sensitivity(np.zeros((3, 2)), min_sizes=[2])
    assert result.loc[0, "pct_noise"] == 33.33
    assert hdbscan_backend[0][0].kwargs["min_samples"] == 20
    assert a.sensitivity_df.empty
    result.loc[0, "n_noise"] = 99
    assert a.sensitivity_df.empty


def test_backend_failure_is_propagated(hdbscan_module, hdbscan_backend):
    for mode in ["fit", "fit_presentation", "sensitivity"]:
        error = ValueError("controlled invalid geometry")
        hdbscan_backend[1].append(error)
        with pytest.raises(ValueError, match="controlled invalid geometry") as caught:
            if mode == "sensitivity":
                hdbscan_module.compute_sensitivity(np.zeros((3, 2)), min_sizes=[2])
            else:
                getattr(hdbscan_module.HDBSCANAnalyzer(), mode)(np.zeros((3, 2)))
        assert caught.value is error


@pytest.mark.integration
@pytest.mark.parametrize(
    "order", [["feature_a", "feature_b"], ["feature_b", "feature_a"]]
)
def test_feature_profiles_independent_zscores_noise_reference(
    hdbscan_module, hdbscan_csv, order
):
    values = {"feature_a": [0.0, 2.0, 6.0, 8.0], "feature_b": [1.0, 5.0, 1.0, 9.0]}
    paths, stem, file = hdbscan_csv({name: values[name] for name in order})
    old = file.read_bytes()
    Z = np.arange(8.0).reshape(4, 2)
    labels = np.array([7, -1, 2, 2])
    Z0, cl0 = Z.copy(), labels.copy()
    frame, means, aligned = hdbscan_module.compute_feature_profiles(
        paths, stem, Z, labels
    )
    expected = pd.DataFrame(
        {
            "feature_a": [3 / math.sqrt(10), -4 / math.sqrt(10)],
            "feature_b": [1 / math.sqrt(11), -3 / math.sqrt(11)],
        },
        index=pd.Index([2, 7], name="cluster"),
    )[order]
    assert_frame_equal(means, expected, rtol=0, atol=3e-15)
    assert frame.columns.tolist() == order and frame.shape == (4, 2)
    np.testing.assert_array_equal(aligned, cl0)
    np.testing.assert_array_equal(labels, cl0)
    np.testing.assert_array_equal(Z, Z0)
    assert file.read_bytes() == old


@pytest.mark.integration
def test_feature_selection_matches_spectro_loader_on_controlled_population(
    hdbscan_module, hdbscan_csv, dimred_loader_module
):
    data = {
        "obsid": np.arange(10),
        "class": ["STAR"] * 10,
        "snr_r": [10.0, 11.0, 9.0, np.nan, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0],
        "feature_a": np.arange(10.0),
        "feature_b": np.arange(10.0) ** 2,
        "feature_constant": [4.0] * 10,
        "text": ["x"] * 10,
    }
    # Mechanism characterization only: known #23 denylist is not a provenance certificate.
    for col in [
        "teff_gspphot",
        "mh_gspphot",
        "source_id",
        "gaia_ra",
        "snr_g",
        "coeff0",
        "phot_g_mean_flux",
        "pmra",
    ]:
        data[col] = np.arange(100.0, 110.0)
    paths, stem, _ = hdbscan_csv(data)
    labels = np.array([2, 2, 7, 7, -1, 7, 7, 7])
    frame, means, aligned = hdbscan_module.compute_feature_profiles(
        paths, stem, np.zeros((8, 2)), labels
    )
    assert frame.columns.tolist() == ["feature_a", "feature_b"]
    assert frame.shape == (8, 2)
    # SNR=10 is included; row 2 and missing SNR row 3 cannot enter this oracle.
    selected = np.array([0.0, 1.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
    mu = 5.0
    variance = 9.0
    assert abs(sum((selected - mu) ** 2) / 8 - variance) < 1e-14
    np.testing.assert_allclose(
        means.loc[2, "feature_a"], -4.5 / math.sqrt(variance), rtol=0, atol=3e-15
    )
    loader = dimred_loader_module.DimRedDataLoader("unused", "unused")
    filtered = pd.DataFrame(data).iloc[[0, 1, 4, 5, 6, 7, 8, 9]]
    assert (
        loader._select_feature_columns(filtered, spectro_only=True)
        == frame.columns.tolist()
    )
    np.testing.assert_array_equal(aligned, labels)


@pytest.mark.parametrize("missing", [0, 1, 2])
def test_feature_missing_threshold_and_dropna(hdbscan_module, hdbscan_csv, missing):
    x = np.arange(10.0)
    x[:missing] = np.nan
    paths, stem, _ = hdbscan_csv(
        {"feature_control": np.arange(10.0), "feature_optional": x}
    )
    n = 9 if missing == 1 else 10
    labels = np.array([2] * 4 + [7] * (n - 4))
    frame, means, aligned = hdbscan_module.compute_feature_profiles(
        paths, stem, np.zeros((n, 2)), labels
    )
    assert frame.columns.tolist() == (
        ["feature_control", "feature_optional"] if missing <= 1 else ["feature_control"]
    )
    assert len(frame) == n and len(aligned) == n
    expected = (
        np.array([-2.5, 2.0]) / math.sqrt(20 / 3)
        if missing == 1
        else np.array([-3.0, 2.0]) / math.sqrt(8.25)
    )
    for col in means.columns:
        np.testing.assert_allclose(means[col], expected, rtol=0, atol=3e-15)
    np.testing.assert_array_equal(aligned, labels)


@pytest.mark.parametrize("case", ["all_noise", "one_feature", "constant_after_filter"])
def test_feature_edge_populations(hdbscan_module, hdbscan_csv, case):
    if case == "constant_after_filter":
        paths, stem, _ = hdbscan_csv(
            {
                "a": [1.0] * 18 + [2.0, 3.0],
                "b": list(np.arange(18.0)) + [np.nan, np.nan],
            }
        )
        # a varies before dropna, then becomes constant in the 18 retained rows.
    else:
        paths, stem, _ = hdbscan_csv({"a": [1.0, 3.0, 5.0]})
    n = 18 if case == "constant_after_filter" else 3
    labels = np.full(n, -1 if case == "all_noise" else 7)
    frame, means, aligned = hdbscan_module.compute_feature_profiles(
        paths, stem, np.zeros((n, 2)), labels
    )
    assert len(frame) == n
    assert means.index.tolist() == ([] if case == "all_noise" else [7])
    if case != "all_noise":
        np.testing.assert_allclose(means.to_numpy(), 0.0, rtol=0, atol=3e-15)
    np.testing.assert_array_equal(aligned, labels)


@pytest.mark.parametrize("bad", [np.inf, -np.inf])
def test_feature_infinity_cannot_produce_finite_profile(
    hdbscan_module, hdbscan_csv, bad
):
    paths, stem, _ = hdbscan_csv({"a": [1.0, bad, 3.0]})
    with pytest.raises(ValueError):
        hdbscan_module.compute_feature_profiles(
            paths, stem, np.zeros((3, 2)), np.array([2, 2, 2])
        )


@pytest.mark.parametrize("case", ["filtered", "no_features"])
def test_feature_empty_population_refused(hdbscan_module, hdbscan_csv, case):
    paths, stem, _ = hdbscan_csv(
        {"a": [1.0, 2.0], "snr_r": [0.0, 1.0]}
        if case == "filtered"
        else {"constant": [1.0, 1.0]}
    )
    with pytest.raises(ValueError):
        hdbscan_module.compute_feature_profiles(
            paths, stem, np.zeros((2, 2)), np.array([2, 2])
        )


def test_feature_file_missing_and_import_fallback(
    hdbscan_module, hdbscan_csv, dimred_loader_module, monkeypatch
):
    paths, stem, _ = hdbscan_csv({"a": [1.0, 3.0, 5.0]})
    with pytest.raises(FileNotFoundError):
        hdbscan_module.compute_feature_profiles(
            paths, "absent", np.zeros((3, 2)), np.array([2, 2, 2])
        )
    monkeypatch.setitem(sys.modules, "data_loader", dimred_loader_module)
    real = builtins.__import__

    def fallback(name, *args, **kwargs):
        if name == "dimred.data_loader":
            raise ImportError("exercise documented local fallback")
        return real(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fallback)
    _, means, aligned = hdbscan_module.compute_feature_profiles(
        paths, stem, np.zeros((3, 2)), np.array([2, 2, 2])
    )
    assert means.index.tolist() == [2]
    np.testing.assert_array_equal(aligned, [2, 2, 2])


@pytest.mark.integration
def test_loader_pca_cluster_profile_observation_identity(
    hdbscan_module, hdbscan_backend, hdbscan_csv, dimred_loader_module, dimred_pca_class
):
    paths, stem, feature_file = hdbscan_csv(
        {
            "obsid": [13, 11, 17, 19],
            "class": ["M", "M", "G", "G"],
            "snr_r": [20.0] * 4,
            "feature_a": [0.0, 2.0, 6.0, 8.0],
            "feature_b": [1.0, 5.0, 1.0, 9.0],
        }
    )
    catalog = feature_file.parent / "catalog.csv"
    pd.DataFrame(
        {"obsid": [19, 17, 11, 13], "teff_gspphot": [6200.0, 6000.0, 1200.0, 1000.0]}
    ).to_csv(catalog, index=False)
    loader = dimred_loader_module.DimRedDataLoader(feature_file, catalog)
    X, y, meta = loader.load(spectro_only=True, scale=True)
    pca = dimred_pca_class(n_components=2).fit(X, loader.get_feature_names())
    Z = pca.transform(X)
    hdbscan_backend[1].append([2, 2, 7, 7])
    a = hdbscan_module.HDBSCANAnalyzer(2, 1).fit(Z)
    np.testing.assert_array_equal(a.clusterer_.input, Z)
    table, _ = a.physical_profile_table(meta, y)
    assert table.loc["C2", "T_eff (K)"] == "1100 ± 141"
    assert table.loc["C7", "T_eff (K)"] == "6100 ± 141"
    _, means, aligned = hdbscan_module.compute_feature_profiles(
        paths, stem, Z, a.labels_
    )
    assert means.index.tolist() == [2, 7]
    np.testing.assert_array_equal(aligned, [2, 2, 7, 7])


def test_scaler_receives_post_filter_population_and_default_options(
    hdbscan_module, hdbscan_csv, monkeypatch
):
    from sklearn import preprocessing

    real = preprocessing.StandardScaler
    calls = []

    class RecordingScaler(real):
        def fit_transform(self, X, **kwargs):
            calls.append((X.copy(), self.get_params()))
            return super().fit_transform(X, **kwargs)

    monkeypatch.setattr(preprocessing, "StandardScaler", RecordingScaler)
    paths, stem, _ = hdbscan_csv(
        {"a": [0.0, 2.0, 6.0, 8.0], "snr_r": [10.0, 20.0, 9.0, 30.0]}
    )
    hdbscan_module.compute_feature_profiles(
        paths, stem, np.zeros((3, 2)), np.array([7, -1, 2])
    )
    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0][0], [[0.0], [2.0], [8.0]])
    assert calls[0][1]["with_mean"] is True and calls[0][1]["with_std"] is True


def test_feature_paired_permutation_and_affine_unit_change(hdbscan_module, hdbscan_csv):
    x = np.array([0.0, 2.0, 6.0, 8.0])
    labels = np.array([7, -1, 2, 2])
    p = [2, 0, 3, 1]
    paths, stem, _ = hdbscan_csv({"a": x})
    _, base, _ = hdbscan_module.compute_feature_profiles(
        paths, stem, np.zeros((4, 2)), labels
    )
    paths, stem, _ = hdbscan_csv({"a": (3 * x + 17)[p]})
    _, moved, _ = hdbscan_module.compute_feature_profiles(
        paths, stem, np.zeros((4, 2)), labels[p]
    )
    assert_frame_equal(base, moved, rtol=0, atol=3e-15)


def test_physical_group_local_change_and_noise_isolation(
    hdbscan_module, hdbscan_backend
):
    a = fitted(hdbscan_module, hdbscan_backend, [2, 2, 7, 7, -1])
    meta = pd.DataFrame({"teff_gspphot": [1000.0, 1200.0, 6000.0, 6200.0, 9000.0]})
    y = np.array(["M", "M", "G", "G", "A"])
    base, _ = a.physical_profile_table(meta, y)
    changed = meta.copy()
    changed.loc[0, "teff_gspphot"] += 200.0
    next_table, _ = a.physical_profile_table(changed, y)
    assert next_table.loc["C2", "T_eff (K)"] == "1200 ± 0"
    assert next_table.loc["C7"].to_dict() == base.loc["C7"].to_dict()
    assert next_table.loc["Bruit"].to_dict() == base.loc["Bruit"].to_dict()
    changed.loc[4, "teff_gspphot"] = -2000.0
    noise_changed, _ = a.physical_profile_table(changed, y)
    assert_frame_equal(noise_changed.loc[["C2", "C7"]], next_table.loc[["C2", "C7"]])


def test_print_empty_supplied_tables(hdbscan_module, capsys):
    table = pd.DataFrame(columns=["n spectres"])
    interp = pd.DataFrame(columns=["n", "Interprétation"])
    hdbscan_module.HDBSCANAnalyzer().print_physical_profile(table, interp)
    assert "HDBSCAN" in capsys.readouterr().out
    assert table.empty and interp.empty


def test_all_default_physical_columns_have_correct_units_and_moments(
    hdbscan_module, hdbscan_backend
):
    a = fitted(hdbscan_module, hdbscan_backend, [2, 2])
    meta = pd.DataFrame(
        {
            "teff_gspphot": [5000.0, 6000.0],
            "logg_gspphot": [3.0, 5.0],
            "mh_gspphot": [-1.0, 1.0],
            "bp_rp": [1.0, 3.0],
            "phot_g_mean_mag": [10.0, 14.0],
            "distance_gspphot": [100.0, 300.0],
        }
    )
    table, _ = a.physical_profile_table(meta, np.array(["G", "G"]))
    expected = {
        "T_eff (K)": "5500 ± 707",
        "log g (dex)": "4.00 ± 1.41",
        "G_BP-G_RP": "2.00 ± 1.41",
        "G mag": "12.00 ± 2.83",
        "Distance (pc)": "200.00 ± 141.42",
    }
    for col, value in expected.items():
        assert table.loc["C2", col] == value
    # Do not freeze the unresolved [M/H] versus [Fe/H] display terminology.
    extra = set(table.columns) - set(expected) - {"n spectres", "Classe dom."}
    assert len(extra) == 1 and table.loc["C2", extra.pop()] == "0.00 ± 1.41"
