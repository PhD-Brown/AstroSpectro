"""Archived M2.1 evidence: failures, accepted contracts and corrected observations.

Run explicitly: python -m pytest -q tests/proofs_dimred_data_loader.py
The filename intentionally does not match pytest's test_*.py/*_test.py discovery.
No xfail/skip, weakened assertions, or production fixes are used. The normal
suite covers correct contracts; these proofs have per-test classifications and are linked to the
separate corrective issues listed in audit/M21_DATA_LOADING.md.
"""

import numpy as np
import pandas as pd
import pytest


@pytest.mark.unit
@pytest.mark.parametrize("label", ["class", "subclass"])
def test_catalog_labels_are_joined_exactly_once(dimred_feature_loader, label):
    """Refs #49 - classification: demonstrated defect."""
    loader = dimred_feature_loader()
    frame = pd.DataFrame({"obsid": [2, 1], "feature": [2.0, 1.0]})
    cat = pd.DataFrame({"obsid": [1, 2], label: ["A", "B"]})
    merged = loader._merge(frame, cat)
    assert merged.columns.is_unique
    assert merged[label].tolist() == ["B", "A"]


@pytest.mark.integration
def test_catalog_supplies_one_dimensional_labels(dimred_feature_loader):
    """Refs #49 - classification: demonstrated defect."""
    loader = dimred_feature_loader(
        {"obsid": [2, 1], "feature": [2.0, 1.0]},
        {"obsid": [1, 2], "class": ["STAR", "QSO"]},
    )
    X, y, meta = loader.load(scale=False)
    assert y.shape == (len(X),)
    assert meta.columns.is_unique
    np.testing.assert_array_equal(y, ["QSO", "STAR"])


@pytest.mark.unit
def test_conflicting_catalog_keys_are_rejected(dimred_feature_loader):
    """Refs #50 - classification: demonstrated defect; remedy proposed."""
    loader = dimred_feature_loader()
    # One observation cannot be assigned two incompatible temperatures.
    with pytest.raises(ValueError):
        loader._merge(
            pd.DataFrame({"obsid": [1, 2], "feature": [10.0, 20.0]}),
            pd.DataFrame(
                {"obsid": [1, 1, 2], "teff_gspphot": [1000.0, 9000.0, 2000.0]}
            ),
        )


@pytest.mark.unit
def test_missing_identity_cannot_match_another_missing_identity(dimred_feature_loader):
    """Refs #50 - classification: demonstrated defect."""
    merged = dimred_feature_loader()._merge(
        pd.DataFrame({"obsid": [1.0, np.nan], "feature": [10.0, 20.0]}),
        pd.DataFrame({"obsid": [1.0, np.nan], "teff_gspphot": [1000.0, 9000.0]}),
    )
    assert np.isnan(merged.teff_gspphot.iloc[1])


@pytest.mark.integration
def test_large_observation_ids_are_not_rounded_before_join(dimred_feature_loader):
    # Requires BOTH a missing ID and integers above float64 exact range.
    # source_id exercises the configurable Gaia-key path; this is not evidence
    # for prevalence in the much smaller default LAMOST observation IDs.
    """Refs #50 - classification: demonstrated defect."""
    first = 2**53
    loader = dimred_feature_loader(
        {
            "source_id": [str(first + 1), None, str(first + 2)],
            "class": ["STAR", "QSO", "GALAXY"],
            "feature": [1.0, 2.0, 3.0],
        },
        {
            "source_id": [str(first), str(first + 1), str(first + 2)],
            "teff_gspphot": [9000.0, 1000.0, 3000.0],
        },
        merge_key="source_id",
    )
    _, _, meta = loader.load(scale=False)
    # Adjacent integer IDs above float64's exact range must remain distinct.
    assert meta.teff_gspphot.iloc[0] == 1000.0


@pytest.mark.integration
def test_unscaled_reload_does_not_reuse_previous_scaler(dimred_feature_loader):
    """Refs #51 - classification: demonstrated defect."""
    loader = dimred_feature_loader()
    loader.load(scale=True)
    X, _, _ = loader.load(scale=False)
    np.testing.assert_array_equal(loader.inverse_transform(X), X)
    assert loader.scaler_ is None


@pytest.mark.integration
@pytest.mark.parametrize("balance", [False, True])
def test_empty_feature_selection_has_domain_error(dimred_feature_loader, balance):
    """Refs #55 - classification: accepted validation contract (2026-11-07); not a production fix."""
    loader = dimred_feature_loader()
    # Accepted #55: a domain error, not sklearn/concat's incidental ValueError.
    with pytest.raises(
        ValueError,
        match=r"(?i)(no observations|empty (batch|catalog)|no valid spectra)",
    ):
        loader.load(classes=["ABSENT"], class_balance=balance)


@pytest.mark.integration
def test_no_usable_features_has_explicit_validation_error(dimred_feature_loader):
    """Refs #55 - classification: accepted validation contract (2026-11-07); not a production fix."""
    loader = dimred_feature_loader(
        {"obsid": [1, 2], "class": ["STAR", "QSO"], "constant": [2.0, 2.0]},
        {"obsid": [1, 2]},
    )
    # Returning (N, 0) as data ready for PCA contradicts the usable-feature contract.
    with pytest.raises(ValueError, match=r"(?i)(no (usable )?features|zero features)"):
        loader.load(scale=False)


@pytest.mark.integration
@pytest.mark.parametrize("value", [np.inf, -np.inf])
def test_nonfinite_features_are_rejected_without_scaling(dimred_feature_loader, value):
    """Refs #55 - classification: accepted validation contract (2026-11-07); not a production fix."""
    loader = dimred_feature_loader(
        {"obsid": [1, 2, 3], "class": ["STAR"] * 3, "feature": [1.0, value, 3.0]},
        {"obsid": [1, 2, 3]},
    )
    with pytest.raises(ValueError):
        loader.load(scale=False)


@pytest.mark.integration
def test_unknown_mode_is_rejected(dimred_feature_loader):
    """Refs #55 - classification: accepted validation contract (2026-11-07); not a production fix."""
    with pytest.raises(ValueError):
        dimred_feature_loader().load(mode="typo", scale=False)


@pytest.mark.unit
def test_balancing_does_not_silently_drop_missing_labels(dimred_feature_loader):
    """Refs #54 - classification: demonstrated defect; remedy proposed."""
    frame = pd.DataFrame(
        {"obsid": [1, 2, 3], "class": ["STAR", None, "QSO"], "feature": [1.0, 2.0, 3.0]}
    )
    with pytest.raises(ValueError):
        dimred_feature_loader()._balance(frame, 10)


@pytest.mark.unit
def test_balance_log_reports_original_population(dimred_feature_loader, caplog):
    """Refs #54 - classification: demonstrated defect."""
    frame = pd.DataFrame({"class": ["STAR"] * 10, "feature": np.arange(10)})
    with caplog.at_level("INFO"):
        dimred_feature_loader()._balance(frame, 3)
    assert "10 → 3" in caplog.text


@pytest.mark.unit
def test_spectro_only_excludes_known_gaia_proxies(dimred_feature_loader):
    # Provenance is fixed by construction, independent of the current denylist.
    """Refs #23 - classification: already known."""
    frame = pd.DataFrame(
        {
            "feature_EW_Ha": [1.0, 2.0, 3.0],
            "feature_synthetic_BV": [4.0, 5.0, 6.0],
            "lum_gspphot": [10.0, 20.0, 30.0],
            "M_G": [3.0, 4.0, 5.0],
            "feature_halo_index": [0.0, 1.0, 2.0],
        }
    )
    assert dimred_feature_loader()._select_feature_columns(
        frame, spectro_only=True
    ) == [
        "feature_EW_Ha",
        "feature_synthetic_BV",
    ]


@pytest.mark.unit
def test_spectro_only_rejects_unknown_provenance(dimred_feature_loader):
    """Refs #23 - classification: already known."""
    with pytest.raises(ValueError):
        dimred_feature_loader()._select_feature_columns(
            pd.DataFrame({"unknown_origin": [1.0, 2.0]}), spectro_only=True
        )


@pytest.mark.integration
@pytest.mark.parametrize("empty_by", ["header_only", "class_filter", "zero_sample"])
def test_empty_spectral_batch_has_domain_error(
    dimred_fits_writer, dimred_spectral_loader, empty_by
):
    """Refs #55 - classification: accepted validation contract (2026-11-07); not a production fix."""
    path, wl = dimred_fits_writer()
    loader = dimred_spectral_loader([path], wl)
    kwargs = {"normalize": False}
    if empty_by == "header_only":
        pd.read_csv(loader.catalog_path).iloc[:0].to_csv(
            loader.catalog_path, index=False
        )
    elif empty_by == "class_filter":
        kwargs["classes"] = ["ABSENT"]
    else:
        kwargs["n_spectra"] = 0
    with pytest.raises(
        ValueError,
        match=r"(?i)(no observations|empty (batch|catalog)|no valid spectra)",
    ):
        loader.load(**kwargs)


@pytest.mark.integration
def test_exactly_twenty_percent_missing_is_rejected(
    dimred_fits_writer, dimred_spectral_loader
):
    """Refs #55 - classification: accepted validation contract (2026-11-07); not a production fix."""
    path, wl = dimred_fits_writer()
    control, _ = dimred_fits_writer(
        "twenty-control.fits",
        coeff0=np.log10(3000.0),
        coeff1=0.02,
        flux=np.full(10, 7.0),
    )
    grid = np.r_[wl[0] - 2.0, wl[0] - 1.0, wl[:8]]
    X, _, meta = dimred_spectral_loader([path, control], grid).load(normalize=False)
    np.testing.assert_array_equal(X, np.full((1, 10), 7.0))
    assert meta.teff_gspphot.tolist() == [1001]


@pytest.mark.integration
@pytest.mark.parametrize("normalize", [False, True])
def test_infinite_flux_does_not_become_extreme_valid_pixels(
    dimred_fits_writer, dimred_spectral_loader, normalize
):
    """Refs #52 - classification: demonstrated defect."""
    flux = np.ones(10)
    flux[4] = np.inf
    path, wl = dimred_fits_writer(flux=flux)
    X, _, _ = dimred_spectral_loader([path], wl).load(normalize=normalize)
    # Policy may reject the row or mask invalid pixels. Max-float replacement
    # is neither: normalized unit continuum cannot contain a 1e308 pixel.
    assert np.all(np.abs(X) <= 1.0)


@pytest.mark.integration
def test_zero_inverse_variance_pixels_are_not_trusted(
    dimred_fits_writer, dimred_spectral_loader
):
    """Refs #34 - classification: already known."""
    flux = np.ones(10)
    flux[4] = 1e6
    invvar = np.ones(10)
    invvar[4] = 0.0
    path, wl = dimred_fits_writer(data=np.vstack([flux, invvar]))
    X, _, _ = dimred_spectral_loader([path], wl).load(normalize=False)
    assert not np.any(X == 1e6)


@pytest.mark.unit
def test_duplicate_fits_basenames_do_not_silently_choose_one(
    dimred_fits_writer, dimred_spectral_loader
):
    """Refs #53 - classification: demonstrated defect."""
    first, wl = dimred_fits_writer("a/same.fits", flux=np.ones(10))
    dimred_fits_writer("b/same.fits", flux=np.full(10, 9.0))
    with pytest.raises(ValueError):
        dimred_spectral_loader([first], wl)._build_fits_index()


@pytest.mark.unit
def test_index_does_not_treat_partial_download_as_fits(
    tmp_path, dimred_spectral_loader
):
    """Refs none - classification: hygiene observation; no linked defect."""
    partial = tmp_path / "incomplete.fits.part"
    partial.write_bytes(b"incomplete download")
    index = dimred_spectral_loader([], [4000.0])._build_fits_index()
    assert partial.name not in index


@pytest.mark.unit
@pytest.mark.parametrize(
    "grid", [[], [[4000.0, 4010.0], [4020.0, 4030.0]], [4000.0, np.nan]]
)
def test_target_grid_requires_finite_nonempty_vector(dimred_spectral_loader, grid):
    """Refs #55 - classification: accepted validation contract (2026-11-07); not a production fix."""
    with pytest.raises(ValueError):
        dimred_spectral_loader([], grid)


@pytest.mark.unit
def test_identical_catalog_duplicates_cannot_multiply_observations(
    dimred_feature_loader,
):
    """Refs #50 - classification: demonstrated defect."""
    merged = dimred_feature_loader()._merge(
        pd.DataFrame({"obsid": [1, 2], "feature": [10.0, 20.0]}),
        pd.DataFrame({"obsid": [1, 1, 2], "teff_gspphot": [1000.0, 1000.0, 2000.0]}),
    )
    # Unlike the conflicting-key test, the oracle does not prescribe rejection
    # versus explicit deduplication. It forbids silently growing N=2 to N=3.
    assert len(merged) == 2


@pytest.mark.unit
@pytest.mark.parametrize("indices", [[5, 5, 2, 7], [7, 5, 2, 0]])
def test_nonincreasing_target_grid_is_rejected(
    dimred_fits_writer, dimred_spectral_loader, indices
):
    """Refs #55 - classification: accepted validation contract (2026-11-07); not a production fix."""
    path, wl = dimred_fits_writer()
    with pytest.raises(ValueError):
        dimred_spectral_loader([path], wl[indices]).load(normalize=False)


@pytest.mark.integration
def test_all_missing_fits_has_domain_error(dimred_spectral_loader, tmp_path):
    """Refs #55 - classification: accepted validation contract (2026-11-07); not a production fix."""
    with pytest.raises(
        ValueError, match=r"(?i)(no observations|no valid spectra|empty batch)"
    ):
        dimred_spectral_loader([tmp_path / "absent.fits"], [4000.0, 4100.0]).load()


@pytest.mark.integration
def test_failed_reload_does_not_leave_mixed_state(dimred_feature_loader):
    """Refs #51 - classification: demonstrated defect; remedy proposed."""
    loader = dimred_feature_loader()
    loader.load()
    old_names = loader.get_feature_names().copy()
    old_scaler, old_frame = loader.scaler_, loader._df_merged
    with pytest.raises(ValueError):
        loader.load(classes=["ABSENT"])
    # A failed transaction must retain one coherent previous state. Clearing
    # all state is another policy to discuss in #51, not silently assumed here.
    assert loader.get_feature_names() == old_names
    assert loader.scaler_ is old_scaler and loader._df_merged is old_frame


@pytest.mark.integration
@pytest.mark.parametrize("jobs", [1, 2, 4])
def test_worker_read_errors_reach_parent_handlers(
    dimred_fits_writer,
    dimred_spectral_loader,
    dimred_guarded_parallel,
    tmp_path,
    caplog,
    jobs,
):
    """Refs #57 - classification: demonstrated defect; serial control."""
    good, wl = dimred_fits_writer()
    bad = tmp_path / "diagnostic-bad.fits"
    bad.write_bytes(b"bad")
    loader = dimred_spectral_loader([bad, good], wl)
    loader.n_jobs = jobs
    with caplog.at_level("WARNING"):
        X, _, _ = loader.load(normalize=False)
    assert len(X) == 1
    assert any(
        "Erreur lecture diagnostic-bad.fits" in r.message for r in caplog.records
    )


@pytest.mark.integration
def test_zero_snr_threshold_historically_differs_between_loaders(
    dimred_feature_loader, dimred_fits_writer, dimred_spectral_loader
):
    """Refs #55 - classification: historical observation, not a scientific guarantee."""
    f = {
        "obsid": [1, 2, 3],
        "class": ["STAR"] * 3,
        "snr_r": [np.nan, 10.0, 20.0],
        "feature": [1.0, 2.0, 3.0],
    }
    X, _, _ = dimred_feature_loader(f, {"obsid": [1, 2, 3]}).load(
        snr_min=0, scale=False
    )
    assert len(X) == 2
    paths = []
    for i in range(3):
        p, wl = dimred_fits_writer(f"snr-zero/{i}.fits")
        paths.append(p)
    cat = {
        "fits_name": [p.name for p in paths],
        "class": f["class"],
        "snr_r": f["snr_r"],
    }
    X, _, _ = dimred_spectral_loader(paths, wl, cat).load(snr_min=0, normalize=False)
    assert len(X) == 3


@pytest.mark.integration
def test_csv_numeric_strings_are_inferred_as_features(dimred_feature_loader):
    """Refs D1/#23 - classification: historical observation, not a scientific guarantee."""
    loader = dimred_feature_loader(
        {"obsid": [1, 2, 3], "class": ["STAR"] * 3, "text": ["1", "2", "3"]},
        {"obsid": [1, 2, 3]},
    )
    X, _, _ = loader.load(scale=False)
    assert loader.get_feature_names() == ["text"]
    np.testing.assert_array_equal(X[:, 0], [1.0, 2.0, 3.0])


@pytest.mark.unit
def test_default_grid_pixel_count_is_historical(dimred_loader_module, tmp_path):
    """Refs N5/N8 - classification: historical observation, not a scientific guarantee."""
    loader = dimred_loader_module.SpectralMatrixLoader(
        tmp_path, tmp_path / "unused.csv"
    )
    assert len(loader.wl_grid) == 3921


@pytest.mark.unit
def test_uppercase_fits_index_is_platform_dependent(
    dimred_fits_writer, dimred_spectral_loader
):
    """Refs N5/N8 - classification: historical observation, not a scientific guarantee."""
    import os

    path, wl = dimred_fits_writer("uppercase.FITS")
    index = dimred_spectral_loader([path], wl)._build_fits_index()
    assert (path.name in index) == (os.name == "nt")


@pytest.mark.integration
@pytest.mark.parametrize("balance", [False, True])
def test_failed_reload_state_historical_observation(dimred_feature_loader, balance):
    """Refs #51 - classification: historical observation of mixed state (N7)."""
    loader = dimred_feature_loader()
    loader.load()
    previous_scaler, previous_frame = loader.scaler_, loader._df_merged
    with pytest.raises(ValueError):
        loader.load(classes=["ABSENT"], class_balance=balance)
    assert loader.feature_names_ == []
    assert loader._df_merged is previous_frame
    if balance:
        assert loader.scaler_ is previous_scaler
    else:
        assert loader.scaler_ is not previous_scaler
        assert not hasattr(loader.scaler_, "mean_")


@pytest.mark.integration
@pytest.mark.parametrize("kind", ["empty", "nan", "two_dimensional"])
def test_invalid_target_current_behavior(
    dimred_fits_writer, dimred_spectral_loader, kind
):
    """Refs #55 - classification: historical observation, not accepted validation (N4)."""
    path, wl = dimred_fits_writer()
    if kind == "empty":
        with pytest.warns(RuntimeWarning):
            X, _, _ = dimred_spectral_loader([path], []).load(normalize=False)
        assert X.shape == (0, 0)
    elif kind == "two_dimensional":
        with pytest.raises(IndexError):
            dimred_spectral_loader([path], wl[:4].reshape(2, 2)).load(normalize=False)
    else:
        grid = wl.copy()
        grid[4] = np.nan
        X, _, _ = dimred_spectral_loader([path], grid).load(normalize=False)
        assert X.shape == (1, 10)
        assert X[0, 4] == 0.0
