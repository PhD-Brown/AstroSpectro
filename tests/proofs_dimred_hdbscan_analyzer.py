"""M2.4 red proofs and explicit OPEN-contract observations, outside CI discovery.

Run explicitly from the archived SHA. Failures are not separate bug counts.
No xfail, no production modification, no real scientific data.
"""

import warnings

import numpy as np
import pandas as pd
import pytest


@pytest.mark.parametrize("mode", ["fit", "fit_presentation"])
@pytest.mark.parametrize("refit", [False, True])
def test_failed_fit_state_is_coherent(hdbscan_module, hdbscan_backend, mode, refit):
    """NEW state defect: accept full retention OR full invalidation, not mixing."""
    a = hdbscan_module.HDBSCANAnalyzer(3, 2)
    normal = mode == "fit"
    names = (
        [
            "clusterer_",
            "labels_",
            "cluster_ids_",
            "n_clusters_",
            "n_noise_",
            "color_map_",
        ]
        if normal
        else [
            "clusterer_pres_",
            "labels_pres_",
            "ids_pres_",
            "n_clusters_pres_",
            "color_map_pres_",
        ]
    )
    if refit:
        hdbscan_backend[1].append([-1, 2, 2, 7, 7])
        getattr(a, mode)(np.ones((5, 2)))
    old = {name: getattr(a, name) for name in names}
    hdbscan_backend[1].append(ValueError("controlled failure"))
    with pytest.raises(ValueError, match="controlled failure"):
        getattr(a, mode)(np.ones((3, 2)))
    if getattr(a, names[0]) is None:
        assert getattr(a, names[1]) is None
        assert getattr(a, names[2]) == []
        assert getattr(a, names[3]) == 0
        assert getattr(a, names[-1]) == {}
        if normal:
            assert a.n_noise_ == 0
    else:
        assert refit, "A first failed fit exposes an unfitted estimator"
        assert (
            getattr(a, names[0]) is old[names[0]]
        ), "Estimator was replaced while cached labels survived"
        np.testing.assert_array_equal(getattr(a, names[1]), old[names[1]])
        for name in names[2:]:
            assert getattr(a, name) == old[name]


@pytest.mark.parametrize("removed_by", ["snr", "dropna"])
def test_feature_filter_cannot_reassign_labels(hdbscan_module, hdbscan_csv, removed_by):
    """NEW alignment defect: removed middle ID must not shift later labels."""
    values = np.arange(10.0) * 10
    data = {"obsid": np.arange(100, 110), "a": values.copy(), "snr_r": [20.0] * 10}
    if removed_by == "snr":
        data["snr_r"][4] = 9.0
    else:
        data["a"][4] = np.nan  # Exactly 10% missing, retained column then dropna.
    paths, stem, _ = hdbscan_csv(data)
    labels = np.arange(100, 110)
    try:
        frame, means, aligned = hdbscan_module.compute_feature_profiles(
            paths, stem, np.zeros((10, 2)), labels
        )
    except ValueError as error:
        assert any(
            word in str(error).lower()
            for word in ["align", "length", "ident", "shape", "provenance"]
        )
        return  # Explicit refusal is a legitimate alternative to identity reconciliation.
    expected_ids = np.array([100, 101, 102, 103, 105, 106, 107, 108, 109])
    assert len(frame) == 9
    np.testing.assert_array_equal(aligned, expected_ids)
    assert set(means.index) == set(expected_ids)


@pytest.mark.parametrize("n_labels", [8, 12])
def test_feature_labels_length_checked_independently(
    hdbscan_module, hdbscan_csv, n_labels
):
    """NEW same validation mechanism: labels must correspond to all 10 Z rows."""
    paths, stem, _ = hdbscan_csv({"a": np.arange(8.0)})
    with pytest.raises(ValueError, match="(?i)align|length|shape|label|observ|ident"):
        hdbscan_module.compute_feature_profiles(
            paths, stem, np.zeros((10, 2)), np.arange(n_labels)
        )


def test_no_available_physical_columns_has_explicit_outcome(
    hdbscan_module, hdbscan_backend
):
    """NEW robustness gap: counts-only table or relevant domain diagnostic allowed."""
    hdbscan_backend[1].append([2, 2])
    a = hdbscan_module.HDBSCANAnalyzer().fit(np.ones((2, 2)))
    try:
        table, interp = a.physical_profile_table(
            pd.DataFrame({"obsid": [101, 102]}), np.array(["G", "G"])
        )
    except ValueError as error:
        assert any(
            word in str(error).lower() for word in ["phys", "column", "param", "gaia"]
        )
        return
    assert table.loc["C2", "n spectres"] == 2
    assert interp.loc["C2", "n"] == 2


def test_all_missing_classes_have_explicit_outcome(hdbscan_module, hdbscan_backend):
    """NEW robustness gap: undefined dominant class cannot index an empty count."""
    hdbscan_backend[1].append([2, 2])
    a = hdbscan_module.HDBSCANAnalyzer().fit(np.ones((2, 2)))
    try:
        table, _ = a.physical_profile_table(
            pd.DataFrame({"teff_gspphot": [5000.0, 6000.0]}),
            np.array([None, None], dtype=object),
        )
    except ValueError as error:
        assert any(
            word in str(error).lower() for word in ["class", "label", "dominant"]
        )
        return
    assert table.loc["C2", "n spectres"] == 2
    assert pd.isna(table.loc["C2", "Classe dom."]) or table.loc[
        "C2", "Classe dom."
    ] in ["?", "—", "N/A"]


def test_equal_length_permuted_csv_is_not_identity_proof(hdbscan_module, hdbscan_csv):
    """OPEN precondition violation observation: API has no embedding IDs to join."""
    paths, stem, _ = hdbscan_csv(
        {"obsid": [102, 100, 101], "marker": [30.0, 10.0, 20.0]}
    )
    _, means, aligned = hdbscan_module.compute_feature_profiles(
        paths, stem, np.zeros((3, 2)), np.array([2, 7, 9])
    )
    # Actual embedding IDs are 100,101,102. Correct ID100 / cluster2 score is negative.
    assert means.loc[2, "marker"] > 0
    assert means.loc[9, "marker"] == 0
    np.testing.assert_array_equal(aligned, [2, 7, 9])


@pytest.mark.parametrize("permuted", ["meta", "y", "labels"])
def test_physical_equal_lengths_cannot_certify_identity(
    hdbscan_module, hdbscan_backend, permuted
):
    """OPEN observation: violating positional precondition gives a different table."""
    labels = np.array([2, 2, 7, 7])
    meta = pd.DataFrame(
        {
            "obsid": [101, 102, 103, 104],
            "teff_gspphot": [1000.0, 1200.0, 8000.0, 8200.0],
        },
        index=[91, 6, 28, 3],
    )
    y = np.array(["M", "M", "A", "A"])
    hdbscan_backend[1].append(labels)
    a = hdbscan_module.HDBSCANAnalyzer().fit(np.ones((4, 2)))
    good, _ = a.physical_profile_table(meta, y)
    p = [2, 3, 0, 1]
    if permuted == "labels":
        a.labels_ = labels[p]
    bad, _ = a.physical_profile_table(
        meta.iloc[p] if permuted == "meta" else meta, y[p] if permuted == "y" else y
    )
    assert not bad.equals(good)


def test_returned_features_are_raw_despite_standardized_docstring(
    hdbscan_module, hdbscan_csv
):
    """NEW documentation inconsistency observation, no choice of corrected API."""
    paths, stem, _ = hdbscan_csv({"a": [10.0, 20.0, 30.0]})
    frame, means, _ = hdbscan_module.compute_feature_profiles(
        paths, stem, np.zeros((3, 2)), np.array([2, 2, 2])
    )
    np.testing.assert_array_equal(frame["a"], [10.0, 20.0, 30.0])
    assert abs(means.loc[2, "a"]) < 1e-15
    assert "standardisé" in hdbscan_module.compute_feature_profiles.__doc__


@pytest.mark.parametrize("case", ["empty", "matrix_mask", "nonfinite"])
def test_centroid_undefined_domain_observation(hdbscan_module, case):
    """OPEN: empty/nonfinite/matrix-mask policy, no mandatory validation imposed."""
    a = hdbscan_module.HDBSCANAnalyzer()
    Z = np.array([[0.0, 2.0], [4.0, 6.0]])
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        if case == "empty":
            assert np.isnan(a.centroid(Z, np.array([False, False]))).all()
        elif case == "matrix_mask":
            assert np.ndim(a.centroid(Z, np.ones((2, 2), dtype=bool))) == 0
        else:
            Z[0, 0] = np.inf
            assert np.isinf(a.centroid(Z, np.array([True, True]))[0])


@pytest.mark.parametrize(
    "case",
    [
        "tie",
        "empty_phys",
        "duplicate_phys",
        "duplicate_ids",
        "nonnumeric",
        "infinite",
        "nullable_missing",
    ],
)
def test_profile_open_policy_observations(hdbscan_module, hdbscan_backend, case):
    """OPEN observations of tie-breaking, schema, types, provenance and finiteness."""
    hdbscan_backend[1].append([2, 2])
    a = hdbscan_module.HDBSCANAnalyzer().fit(np.ones((2, 2)))
    meta = pd.DataFrame({"obsid": [101, 101], "teff_gspphot": [5000.0, 6000.0]})
    y = np.array(["G", "F"])
    kwargs = {}
    if case == "empty_phys":
        kwargs["phys_cols"] = []
    if case == "duplicate_phys":
        kwargs["phys_cols"] = [("teff_gspphot", "T"), ("teff_gspphot", "T")]
    if case == "nonnumeric":
        meta["teff_gspphot"] = ["bad", "text"]
    if case == "infinite":
        meta["teff_gspphot"] = [np.inf, 6000.0]
    if case == "nullable_missing":
        meta["teff_gspphot"] = pd.array([None, None], dtype="Float64")
    if case == "nonnumeric":
        with pytest.raises(TypeError):
            a.physical_profile_table(meta, y, **kwargs)
        return
    if case == "duplicate_phys":
        with pytest.raises(TypeError, match="format"):
            a.physical_profile_table(meta, y, **kwargs)
        return  # Observation of an OPEN duplicate-column validation policy.
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        table, _ = a.physical_profile_table(meta, y, **kwargs)
    assert table.loc["C2", "n spectres"] == 2
    if case == "empty_phys":
        assert "T_eff (K)" in table.columns
    if case == "tie":
        assert table.loc["C2", "Classe dom."] in ["F", "G"]
    if case == "infinite":
        assert "inf" in table.loc["C2", "T_eff (K)"]


def test_sensitivity_zero_length_observation(hdbscan_module, hdbscan_backend):
    """OPEN wrapper domain: artificial empty backend result, real backend rejects N=0."""
    hdbscan_backend[1].append([])
    with pytest.raises(ZeroDivisionError):
        hdbscan_module.compute_sensitivity(np.empty((0, 2)), min_sizes=[2])


def test_algorithmic_noise_does_not_use_physical_measurements(
    hdbscan_module, hdbscan_backend
):
    """NEW methodological/textual overclaim; output ignores SNR/physical evidence."""
    hdbscan_backend[1].append([-1, -1])
    a = hdbscan_module.HDBSCANAnalyzer().fit(np.ones((2, 2)))
    _, interp = a.physical_profile_table(
        pd.DataFrame({"teff_gspphot": [5700.0, 5800.0]}), np.array(["G", "G"])
    )
    assert (
        interp.loc["Bruit", "Interprétation"]
        == "anomalies spectrales, objets atypiques"
    )
    # The statement is emitted with no spectrum, uncertainty, SNR or anomaly diagnostic.


@pytest.mark.parametrize(
    "teff,logg,mh,parts",
    [
        (7501.0, 2.49, -0.81, ["chaudes", "supergéantes", "halo"]),
        (7500.0, 2.5, -0.8, ["solaire", "géantes", "pop. II"]),
        (5501.0, 3.49, -0.31, ["solaire", "géantes", "pop. II"]),
        (5500.0, 3.5, -0.3, ["froides", "naines", "pop. I"]),
        (4001.0, 4.0, 0.0, ["froides", "naines", "pop. I"]),
        (4000.0, 4.0, 0.0, ["très froides", "naines", "pop. I"]),
    ],
)
def test_current_heuristic_display_observation(
    hdbscan_module, hdbscan_backend, teff, logg, mh, parts
):
    """Display fidelity only, not a scientific taxonomy contract for green CI."""
    hdbscan_backend[1].append([2, 2])
    a = hdbscan_module.HDBSCANAnalyzer().fit(np.ones((2, 2)))
    _, interp = a.physical_profile_table(
        pd.DataFrame(
            {
                "teff_gspphot": [teff] * 2,
                "logg_gspphot": [logg] * 2,
                "mh_gspphot": [mh] * 2,
            }
        ),
        np.array(["G", "G"]),
    )
    for text in parts:
        assert text in interp.loc["C2", "Interprétation"]


@pytest.mark.slow
@pytest.mark.parametrize("mode", ["fit", "fit_presentation"])
def test_real_failed_refit_state_observation(hdbscan_module, mode):
    """Real backend confirms mixed cache; observation PASS, not a second bug."""
    pytest.importorskip("hdbscan")
    rng = np.random.default_rng(17)
    Z = np.r_[rng.normal([-10, 0], 0.1, (12, 2)), rng.normal([10, 0], 0.1, (12, 2))]
    a = hdbscan_module.HDBSCANAnalyzer(4, 2)
    kwargs = {"min_cluster_size": 4} if mode == "fit_presentation" else {}
    getattr(a, mode)(Z, **kwargs)
    estimator = "clusterer_" if mode == "fit" else "clusterer_pres_"
    labels = "labels_" if mode == "fit" else "labels_pres_"
    old_model = getattr(a, estimator)
    old_labels = getattr(a, labels).copy()
    # NaN can be remapped to noise when prediction_data=False. A single row
    # cannot support this min_samples=2 fit in either mode on hdbscan 0.8.44.
    bad = np.ones((1, 2))
    with pytest.raises(ValueError):
        getattr(a, mode)(bad, **kwargs)
    assert getattr(a, estimator) is not old_model
    np.testing.assert_array_equal(getattr(a, labels), old_labels)
