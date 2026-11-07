"""Executable FAILING contracts for M2.1; this is a proof artifact in a Draft PR.

Run explicitly: python -m pytest -q tests/proofs_dimred_data_loader.py
The filename intentionally does not match pytest's test_*.py/*_test.py discovery.
No xfail/skip, weakened assertions, or production fixes are used. The normal
suite covers correct contracts; these proofs await independent review and the
separate corrective issues listed in audit/M21_DATA_LOADING.md.
"""

import numpy as np
import pandas as pd
import pytest


@pytest.mark.unit
@pytest.mark.parametrize("label", ["class", "subclass"])
def test_catalog_labels_are_joined_exactly_once(feature_loader, label):
    loader = feature_loader()
    frame = pd.DataFrame({"obsid": [2, 1], "feature": [2.0, 1.0]})
    cat = pd.DataFrame({"obsid": [1, 2], label: ["A", "B"]})
    merged = loader._merge(frame, cat)
    assert merged.columns.is_unique
    assert merged[label].tolist() == ["B", "A"]


@pytest.mark.integration
def test_catalog_supplies_one_dimensional_labels(feature_loader):
    loader = feature_loader(
        {"obsid": [2, 1], "feature": [2.0, 1.0]},
        {"obsid": [1, 2], "class": ["STAR", "QSO"]},
    )
    X, y, meta = loader.load(scale=False)
    assert y.shape == (len(X),)
    assert meta.columns.is_unique
    np.testing.assert_array_equal(y, ["QSO", "STAR"])


@pytest.mark.unit
def test_conflicting_catalog_keys_are_rejected(feature_loader):
    loader = feature_loader()
    # One observation cannot be assigned two incompatible temperatures.
    with pytest.raises(ValueError):
        loader._merge(
            pd.DataFrame({"obsid": [1, 2], "feature": [10.0, 20.0]}),
            pd.DataFrame(
                {"obsid": [1, 1, 2], "teff_gspphot": [1000.0, 9000.0, 2000.0]}
            ),
        )


@pytest.mark.unit
def test_missing_identity_cannot_match_another_missing_identity(feature_loader):
    merged = feature_loader()._merge(
        pd.DataFrame({"obsid": [1.0, np.nan], "feature": [10.0, 20.0]}),
        pd.DataFrame({"obsid": [1.0, np.nan], "teff_gspphot": [1000.0, 9000.0]}),
    )
    assert np.isnan(merged.teff_gspphot.iloc[1])


@pytest.mark.integration
def test_large_observation_ids_are_not_rounded_before_join(feature_loader):
    first = 2**53
    loader = feature_loader(
        {
            "obsid": [str(first + 1), None, str(first + 2)],
            "class": ["STAR", "QSO", "GALAXY"],
            "feature": [1.0, 2.0, 3.0],
        },
        {
            "obsid": [str(first), str(first + 1), str(first + 2)],
            "teff_gspphot": [9000.0, 1000.0, 3000.0],
        },
    )
    _, _, meta = loader.load(scale=False)
    # Adjacent integer IDs above float64's exact range must remain distinct.
    assert meta.teff_gspphot.iloc[0] == 1000.0


@pytest.mark.integration
def test_unscaled_reload_does_not_reuse_previous_scaler(feature_loader):
    loader = feature_loader()
    loader.load(scale=True)
    X, _, _ = loader.load(scale=False)
    np.testing.assert_array_equal(loader.inverse_transform(X), X)
    assert loader.scaler_ is None


@pytest.mark.integration
@pytest.mark.parametrize("balance", [False, True])
def test_empty_feature_selection_returns_empty_batch(feature_loader, balance):
    loader = feature_loader()
    X, y, meta = loader.load(classes=["ABSENT"], class_balance=balance)
    assert X.ndim == 2 and X.shape[0] == 0
    assert y.shape == (0,)
    assert meta.empty


@pytest.mark.integration
def test_no_usable_features_has_explicit_validation_error(feature_loader):
    loader = feature_loader(
        {"obsid": [1, 2], "class": ["STAR", "QSO"], "constant": [2.0, 2.0]},
        {"obsid": [1, 2]},
    )
    # Returning (N, 0) as data ready for PCA contradicts the usable-feature contract.
    with pytest.raises(ValueError):
        loader.load(scale=False)


@pytest.mark.integration
@pytest.mark.parametrize("value", [np.inf, -np.inf])
def test_nonfinite_features_are_rejected_without_scaling(feature_loader, value):
    loader = feature_loader(
        {"obsid": [1, 2, 3], "class": ["STAR"] * 3, "feature": [1.0, value, 3.0]},
        {"obsid": [1, 2, 3]},
    )
    with pytest.raises(ValueError):
        loader.load(scale=False)


@pytest.mark.integration
def test_unknown_mode_is_rejected(feature_loader):
    with pytest.raises(ValueError):
        feature_loader().load(mode="typo", scale=False)


@pytest.mark.unit
def test_balancing_does_not_silently_drop_missing_labels(feature_loader):
    frame = pd.DataFrame(
        {"obsid": [1, 2, 3], "class": ["STAR", None, "QSO"], "feature": [1.0, 2.0, 3.0]}
    )
    with pytest.raises(ValueError):
        feature_loader()._balance(frame, 10)


@pytest.mark.unit
def test_balance_log_reports_original_population(feature_loader, caplog):
    frame = pd.DataFrame({"class": ["STAR"] * 10, "feature": np.arange(10)})
    with caplog.at_level("INFO"):
        feature_loader()._balance(frame, 3)
    assert "10 → 3" in caplog.text


@pytest.mark.unit
def test_spectro_only_excludes_known_gaia_proxies(feature_loader):
    # Provenance is fixed by construction, independent of the current denylist.
    frame = pd.DataFrame(
        {
            "feature_EW_Ha": [1.0, 2.0, 3.0],
            "feature_synthetic_BV": [4.0, 5.0, 6.0],
            "lum_gspphot": [10.0, 20.0, 30.0],
            "M_G": [3.0, 4.0, 5.0],
            "feature_halo_index": [0.0, 1.0, 2.0],
        }
    )
    assert feature_loader()._select_feature_columns(frame, spectro_only=True) == [
        "feature_EW_Ha",
        "feature_synthetic_BV",
    ]


@pytest.mark.unit
def test_spectro_only_rejects_unknown_provenance(feature_loader):
    with pytest.raises(ValueError):
        feature_loader()._select_feature_columns(
            pd.DataFrame({"unknown_origin": [1.0, 2.0]}), spectro_only=True
        )


@pytest.mark.integration
@pytest.mark.parametrize("empty_by", ["header_only", "class_filter", "zero_sample"])
def test_empty_spectral_batch_has_declared_shapes(
    fits_writer, spectral_loader, empty_by
):
    path, wl = fits_writer()
    loader = spectral_loader([path], wl)
    kwargs = {"normalize": False}
    if empty_by == "header_only":
        pd.read_csv(loader.catalog_path).iloc[:0].to_csv(
            loader.catalog_path, index=False
        )
    elif empty_by == "class_filter":
        kwargs["classes"] = ["ABSENT"]
    else:
        kwargs["n_spectra"] = 0
    X, y, meta = loader.load(**kwargs)
    assert X.shape == (0, len(wl))
    assert y.shape == (0,)
    assert meta.empty


@pytest.mark.integration
def test_exactly_twenty_percent_missing_is_not_more_than_twenty(
    fits_writer, spectral_loader
):
    path, wl = fits_writer()
    # Two out-of-coverage pixels / ten; no interp1d NaN propagation ambiguity.
    grid = np.r_[wl[0] - 2.0, wl[0] - 1.0, wl[:8]]
    X, y, _ = spectral_loader([path], grid).load(normalize=False)
    assert X.shape == (1, 10)
    assert y.tolist() == ["STAR"]


@pytest.mark.integration
@pytest.mark.parametrize("normalize", [False, True])
def test_infinite_flux_does_not_become_extreme_valid_pixels(
    fits_writer, spectral_loader, normalize
):
    flux = np.ones(10)
    flux[4] = np.inf
    path, wl = fits_writer(flux=flux)
    X, _, _ = spectral_loader([path], wl).load(normalize=normalize)
    # Policy may reject the row or mask invalid pixels. Max-float replacement
    # is neither: normalized unit continuum cannot contain a 1e308 pixel.
    assert np.all(np.abs(X) <= 1.0)


@pytest.mark.integration
def test_zero_inverse_variance_pixels_are_not_trusted(fits_writer, spectral_loader):
    flux = np.ones(10)
    flux[4] = 1e6
    invvar = np.ones(10)
    invvar[4] = 0.0
    path, wl = fits_writer(data=np.vstack([flux, invvar]))
    X, _, _ = spectral_loader([path], wl).load(normalize=False)
    assert not np.any(X == 1e6)


@pytest.mark.integration
def test_degenerate_native_wavelength_is_rejected(fits_writer, spectral_loader):
    path, wl = fits_writer(coeff1=0.0)
    X, _, _ = spectral_loader([path], wl).load(normalize=False)
    assert X.shape == (0, len(wl))


@pytest.mark.unit
def test_duplicate_fits_basenames_do_not_silently_choose_one(
    fits_writer, spectral_loader
):
    first, wl = fits_writer("a/same.fits", flux=np.ones(10))
    fits_writer("b/same.fits", flux=np.full(10, 9.0))
    with pytest.raises(ValueError):
        spectral_loader([first], wl)._build_fits_index()


@pytest.mark.unit
def test_index_does_not_treat_partial_download_as_fits(tmp_path, spectral_loader):
    partial = tmp_path / "incomplete.fits.part"
    partial.write_bytes(b"incomplete download")
    index = spectral_loader([], [4000.0])._build_fits_index()
    assert partial.name not in index


@pytest.mark.unit
@pytest.mark.parametrize(
    "grid", [[], [[4000.0, 4010.0], [4020.0, 4030.0]], [4000.0, np.nan]]
)
def test_target_grid_requires_finite_nonempty_vector(spectral_loader, grid):
    with pytest.raises(ValueError):
        spectral_loader([], grid)
