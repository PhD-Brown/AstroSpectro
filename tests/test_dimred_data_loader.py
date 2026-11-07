"""Offline contracts for data_loader.py; archived proofs are linked in the matrix.

Expected values use identity joins, population moments, and analytic linear
spectra. No production data, optional Torch import, or learned models are used.
"""

import logging

from astropy.io import fits
import numpy as np
import pandas as pd
import pytest


@pytest.mark.unit
def test_initial_state_and_public_utilities(dimred_feature_loader):
    loader = dimred_feature_loader(random_state=17)
    assert loader.random_state == 17
    assert loader.scaler_ is None
    assert loader._df_merged is None
    with pytest.raises(RuntimeError, match="load"):
        loader.get_feature_names()
    raw = np.array([[1.0, 2.0]])
    np.testing.assert_array_equal(loader.inverse_transform(raw), raw)


@pytest.mark.integration
@pytest.mark.parametrize("catalog_order", [[0, 1, 2, 3], [3, 1, 0, 2]])
@pytest.mark.parametrize("feature_order", [[0, 1, 2, 3], [2, 0, 3, 1]])
def test_feature_identity_join_and_permutations(
    dimred_feature_loader, catalog_order, feature_order
):
    features = pd.DataFrame(
        {
            "obsid": [30, 10, 20, 40],
            "class": ["STAR", "QSO", "STAR", "GALAXY"],
            "feature_b": [3.0, 1.0, 2.0, 4.0],
            "feature_a": [9.0, 1.0, 4.0, 16.0],
        }
    ).iloc[feature_order]
    catalog = pd.DataFrame(
        {
            "obsid": [10, 40, 30, 20],
            "teff_gspphot": [1000.0, 4000.0, 3000.0, 2000.0],
        }
    ).iloc[catalog_order]
    loader = dimred_feature_loader(features, catalog)
    X, y, meta = loader.load(scale=False, snr_min=0)
    np.testing.assert_array_equal(X, features[["feature_b", "feature_a"]])
    np.testing.assert_array_equal(y, features["class"])
    np.testing.assert_array_equal(meta["teff_gspphot"], features["obsid"] * 100)
    assert meta.index.tolist() == list(range(4))
    assert loader.get_feature_names() == ["feature_b", "feature_a"]
    np.testing.assert_array_equal(loader._df_merged["obsid"], features["obsid"])


@pytest.mark.integration
def test_snr_class_filters_and_missing_metadata_preserve_alignment(
    dimred_feature_loader,
):
    loader = dimred_feature_loader(catalog={"obsid": [10], "teff_gspphot": [1000.0]})
    X, y, meta = loader.load(scale=False, classes=["STAR", "QSO"])
    np.testing.assert_array_equal(X, [[3.0, 9.0], [1.0, 1.0]])
    np.testing.assert_array_equal(y, ["STAR", "QSO"])
    assert np.isnan(meta.teff_gspphot.iloc[0])
    assert meta.teff_gspphot.iloc[1] == 1000
    assert meta.snr_r.tolist() == [30.0, 10.0]


@pytest.mark.integration
def test_custom_merge_key_and_feature_side_precedence(dimred_feature_loader):
    loader = dimred_feature_loader(
        {
            "key": ["b", "a"],
            "class": ["STAR", "QSO"],
            "teff_gspphot": [2000.0, 1000.0],
            "feature": [2.0, 1.0],
        },
        {"key": ["a", "b"], "class": ["WRONG", "WRONG"], "teff_gspphot": [-1.0, -2.0]},
        merge_key="key",
    )
    X, y, meta = loader.load(scale=False)
    np.testing.assert_array_equal(X, [[2.0], [1.0]])
    np.testing.assert_array_equal(y, ["STAR", "QSO"])
    assert meta.teff_gspphot.tolist() == [2000.0, 1000.0]


@pytest.mark.integration
def test_snr_absent_warns_and_alternate_band_is_used(dimred_feature_loader, caplog):
    loader = dimred_feature_loader(
        {
            "obsid": [1, 2, 3, 4],
            "class": ["STAR"] * 4,
            "feature": [1.0, 2.0, 3.0, 4.0],
            "snr_g": [5.0, 10.0, np.nan, 20.0],
        },
        {"obsid": [1, 2, 3, 4]},
    )
    with caplog.at_level(logging.WARNING):
        X, _, _ = loader.load(scale=False)
    assert len(X) == 4
    assert "snr_r" in caplog.text
    X, _, _ = loader.load(scale=False, snr_band="snr_g")
    assert loader._df_merged.obsid.tolist() == [2, 4]
    np.testing.assert_array_equal(X, [[2.0], [4.0]])


@pytest.mark.unit
@pytest.mark.parametrize("spectro_only", [False, True])
def test_column_selection_order_nan_boundary_and_exclusions(
    dimred_feature_loader, spectro_only
):
    frame = pd.DataFrame(
        {
            "obsid": [1, 2, 3, 4],
            "class": ["STAR"] * 4,
            "feature_b": [4.0, 3.0, 2.0, 1.0],
            "feature_a": [1.0, 2.0, 3.0, 4.0],
            "at_threshold": [np.nan, 2.0, 3.0, 4.0],
            "over_threshold": [np.nan, np.nan, 3.0, 4.0],
            "constant": [2.0] * 4,
            "all_nan": [np.nan] * 4,
            "text": ["1", "2", "3", "4"],
            "source_id": [101, 102, 103, 104],
            "snr_r": [10.0, 20.0, 30.0, 40.0],
            "teff_gspphot": [1.0, 2.0, 3.0, 4.0],
            "ra": [1.0, 2.0, 3.0, 4.0],
            "fiber_id": [1, 2, 3, 4],
            "radial_velocity": [1.0, 2.0, 3.0, 4.0],
        }
    )
    selected = dimred_feature_loader()._select_feature_columns(
        frame, nan_threshold=0.25, spectro_only=spectro_only
    )
    expected = ["feature_b", "feature_a", "at_threshold"]
    if not spectro_only:
        # Historical default selection, not a scientific allowlist (D1/#23).
        # `text` above is object dtype in memory; CSV inference differs (N6).
        expected += ["ra", "fiber_id", "radial_velocity"]
    assert selected == expected


@pytest.mark.integration
def test_nan_column_then_row_filtering(dimred_feature_loader):
    loader = dimred_feature_loader(
        {
            "obsid": [1, 2, 3, 4],
            "class": ["STAR", "QSO", "GALAXY", "STAR"],
            "feature": [1.0, np.nan, 3.0, 4.0],
            "mostly_nan": [1.0, np.nan, np.nan, 4.0],
        },
        {"obsid": [4, 3, 2, 1], "teff_gspphot": [4000.0, 3000.0, 2000.0, 1000.0]},
    )
    X, y, meta = loader.load(scale=False, drop_nan_threshold=0.25)
    assert loader.get_feature_names() == ["feature"]
    np.testing.assert_array_equal(X[:, 0], [1.0, 3.0, 4.0])
    np.testing.assert_array_equal(y, ["STAR", "GALAXY", "STAR"])
    assert meta.teff_gspphot.tolist() == [1000.0, 3000.0, 4000.0]


@pytest.mark.integration
def test_population_scaling_and_inverse_preserve_feature_order(dimred_feature_loader):
    loader = dimred_feature_loader()
    X, _, _ = loader.load()
    raw = np.array([[3.0, 9.0], [1.0, 1.0], [4.0, 16.0]])
    expected = (raw - raw.mean(axis=0)) / raw.std(axis=0, ddof=0)
    np.testing.assert_allclose(X, expected, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(loader.inverse_transform(X), raw, rtol=1e-14)
    assert loader.get_feature_names() == ["feature_b", "feature_a"]
    # This validates descriptive whole-population scaling, not train-only use.


@pytest.mark.integration
def test_balance_cap_seed_no_replacement_and_alignment(dimred_feature_loader):
    frame = pd.DataFrame(
        {
            "obsid": np.arange(24),
            "class": ["STAR"] * 20 + ["QSO"] * 4,
            "feature": np.arange(24, dtype=float),
        }
    )
    cat = pd.DataFrame({"obsid": np.arange(24), "teff_gspphot": np.arange(24) * 100.0})
    loader = dimred_feature_loader(frame, cat, random_state=19)
    first = loader.load(scale=False, class_balance=True, n_per_class=5)
    second = loader.load(scale=False, class_balance=True, n_per_class=5)
    for a, b in zip(first, second):
        np.testing.assert_array_equal(a, b)
    X, y, meta = first
    assert dict(zip(*np.unique(y, return_counts=True))) == {"QSO": 4, "STAR": 5}
    assert len(np.unique(X[:, 0])) == len(X)
    np.testing.assert_array_equal(meta.teff_gspphot, X[:, 0] * 100)
    np.testing.assert_array_equal(meta["class"], y)
    loader.random_state = 20
    other, _, _ = loader.load(scale=False, class_balance=True, n_per_class=5)
    assert not np.array_equal(X, other)


@pytest.mark.unit
def test_meta_is_independent_copy_with_reset_index(dimred_feature_loader):
    frame = pd.DataFrame(
        {"class": ["STAR", "QSO"], "ra": [1.0, 2.0], "feature": [10.0, 20.0]},
        index=[8, 5],
    )
    meta = dimred_feature_loader()._build_meta(frame)
    assert meta.columns.tolist() == ["class", "ra"]
    assert meta.index.tolist() == [0, 1]
    meta.loc[0, "ra"] = -999
    assert frame.loc[8, "ra"] == 1


@pytest.mark.unit
@pytest.mark.parametrize("source", ["features", "catalog"])
def test_missing_csv_reports_path(dimred_feature_loader, source):
    loader = dimred_feature_loader()
    path = getattr(loader, source + "_path")
    path.unlink()
    with pytest.raises(FileNotFoundError, match=path.name):
        getattr(loader, "_load_" + source)()


@pytest.mark.unit
@pytest.mark.parametrize("source", ["features", "catalog"])
def test_csv_fallback_retains_good_rows_and_warns(
    dimred_feature_loader, caplog, source
):
    loader = dimred_feature_loader()
    path = getattr(loader, source + "_path")
    path.write_text("obsid,feature\n1,2\n2,3\n3,4,extra\n4,5\n", encoding="utf-8")
    with caplog.at_level(logging.WARNING), pytest.warns(pd.errors.ParserWarning):
        frame = getattr(loader, "_load_" + source)()
    assert frame.obsid.tolist() == [1, 2, 4]
    assert "ParserError" in caplog.text


@pytest.mark.unit
@pytest.mark.parametrize("source", ["features", "catalog"])
def test_truncated_csv_reports_omitted_rows(dimred_feature_loader, source, caplog):
    loader = dimred_feature_loader()
    getattr(loader, source + "_path").write_text(
        'obsid,value\n1,"unterminated', encoding="utf-8"
    )
    # Both parsers deliberately support omission with warnings. An earlier
    # audit assertion expecting ParserError here contradicted that contract.
    with caplog.at_level(logging.WARNING), pytest.warns(pd.errors.ParserWarning):
        frame = getattr(loader, "_load_" + source)()
    assert frame.empty
    assert "ParserError" in caplog.text


@pytest.mark.unit
@pytest.mark.parametrize("source", ["features", "catalog"])
def test_empty_csv_has_explicit_parser_error(dimred_feature_loader, source):
    loader = dimred_feature_loader()
    getattr(loader, source + "_path").write_bytes(b"")
    with pytest.raises(pd.errors.EmptyDataError):
        getattr(loader, "_load_" + source)()


@pytest.mark.unit
def test_spectra_mode_delegation_error_precedes_io(dimred_feature_loader):
    loader = dimred_feature_loader()
    loader.features_path.unlink()
    with pytest.raises(NotImplementedError, match="SpectralMatrixLoader"):
        loader.load(mode="spectra")


@pytest.mark.unit
@pytest.mark.parametrize("side", ["features", "catalog"])
def test_missing_merge_key_fails(dimred_feature_loader, side):
    frame = pd.DataFrame(
        {"obsid": [1, 2], "class": ["STAR", "QSO"], "feature": [1.0, 2.0]}
    )
    cat = pd.DataFrame({"obsid": [1, 2], "teff_gspphot": [1000.0, 2000.0]})
    if side == "features":
        frame = frame.drop(columns="obsid")
    else:
        cat = cat.drop(columns="obsid")
    with pytest.raises(KeyError, match="obsid"):
        dimred_feature_loader(frame, cat).load(scale=False)


@pytest.mark.integration
def test_duplicate_feature_observations_are_not_silently_deduplicated(
    dimred_feature_loader,
):
    loader = dimred_feature_loader(
        {
            "obsid": [1, 1, 2],
            "class": ["STAR", "STAR", "QSO"],
            "feature": [2.0, 3.0, 4.0],
        },
        {"obsid": [2, 1], "teff_gspphot": [2000.0, 1000.0]},
    )
    X, _, meta = loader.load(scale=False)
    np.testing.assert_array_equal(X[:, 0], [2.0, 3.0, 4.0])
    assert meta.teff_gspphot.tolist() == [1000.0, 1000.0, 2000.0]


@pytest.mark.unit
def test_default_spectral_grid_and_recursive_index_cache(
    tmp_path, dimred_loader_module, dimred_fits_writer
):
    path, _ = dimred_fits_writer("nested/one.fits.gz")
    loader = dimred_loader_module.SpectralMatrixLoader(
        tmp_path, tmp_path / "catalog.csv"
    )
    assert np.all(np.diff(loader.wl_grid) > 0)
    assert loader.wl_grid[0] == 3690.0
    assert loader.wl_grid[-1] < 9100.0
    # Historical grid, not a scientific endorsement of linear resampling (N5).
    np.testing.assert_allclose(
        np.diff(loader.wl_grid), 1.38, rtol=0, atol=np.spacing(loader.WL_MAX)
    )
    index = loader._build_fits_index()
    assert index == {path.name: path}
    assert loader._build_fits_index() is index


@pytest.mark.integration
@pytest.mark.parametrize("coeff1", [0.001, -0.001])
@pytest.mark.parametrize("normalize", [False, True])
def test_fits_analytic_linear_interpolation_and_median(
    dimred_fits_writer, dimred_spectral_loader, coeff1, normalize
):
    _, wavelength = dimred_fits_writer(coeff1=coeff1)
    flux = 2 + 0.003 * (wavelength - wavelength.min())
    path, wavelength = dimred_fits_writer(flux=flux, coeff1=coeff1)
    # Midpoints in Angstroms, not log wavelength: a linear function is exact.
    ordered = np.sort(wavelength)
    grid = np.r_[ordered[0], (ordered[:-1] + ordered[1:]) / 2, ordered[-1]]
    expected = 2 + 0.003 * (grid - wavelength.min())
    if normalize:
        expected /= np.median(expected)
    X, y, meta = dimred_spectral_loader([path], grid).load(normalize=normalize)
    np.testing.assert_allclose(X[0], expected, rtol=2e-13, atol=2e-13)
    assert y.tolist() == ["STAR"]
    assert meta["class"].tolist() == ["STAR"]


@pytest.mark.integration
def test_partial_boundary_fill_and_too_short_spectral_coverage(
    dimred_fits_writer, dimred_spectral_loader
):
    path, wl = dimred_fits_writer()
    # One outside pixel among eleven is retained; the current documented fill is 0.
    X, _, _ = dimred_spectral_loader([path], np.r_[wl[0] - 1, wl]).load(normalize=False)
    np.testing.assert_array_equal(X, [np.r_[0.0, np.arange(1.0, 11.0)]])
    control, target = dimred_fits_writer(
        "control.fits", coeff0=np.log10(14000.0), flux=np.full(10, 7.0)
    )
    # A valid control avoids imposing the old empty-return behavior (#55).
    X, y, meta = dimred_spectral_loader([path, control], target).load(normalize=False)
    np.testing.assert_array_equal(X, np.full((1, 10), 7.0))
    assert y.tolist() == ["STAR"]
    assert meta.teff_gspphot.tolist() == [1001]


@pytest.mark.integration
@pytest.mark.parametrize("median", [0.0, -2.0])
def test_nonpositive_median_is_left_unscaled(
    dimred_fits_writer, dimred_spectral_loader, median
):
    path, wl = dimred_fits_writer(flux=np.full(10, median))
    X, _, _ = dimred_spectral_loader([path], wl).load()
    np.testing.assert_array_equal(X, np.full((1, 10), median))


@pytest.mark.integration
@pytest.mark.parametrize(
    "invalid",
    ["corrupt", "no_data", "one_dimensional", "three_dimensional", "missing_header"],
)
def test_invalid_fits_rejected_with_diagnostic_and_alignment(
    dimred_fits_writer, dimred_spectral_loader, invalid, tmp_path, caplog
):
    good, wl = dimred_fits_writer("nested/good.fits", flux=np.full(10, 7.0))
    bad = tmp_path / "bad.fits"
    if invalid == "corrupt":
        bad.write_bytes(b"not a FITS file")
    elif invalid == "no_data":
        fits.PrimaryHDU().writeto(bad)
    elif invalid == "missing_header":
        fits.PrimaryHDU(data=np.ones((2, 10))).writeto(bad)
    else:
        data = np.ones(10) if invalid == "one_dimensional" else np.ones((2, 2, 10))
        bad, _ = dimred_fits_writer("bad.fits", data=data)
    missing = tmp_path / "missing.fits"
    cat = {
        "fits_name": [bad.name, missing.name, good.name],
        "class": ["BAD", "MISSING", "STAR"],
        "snr_r": [30.0, 30.0, 30.0],
        "teff_gspphot": [1.0, 2.0, 7000.0],
    }
    with caplog.at_level(logging.INFO):
        X, y, meta = dimred_spectral_loader([bad, missing, good], wl, cat).load(
            normalize=False
        )
    np.testing.assert_array_equal(X, np.full((1, 10), 7.0))
    assert y.tolist() == ["STAR"]
    assert meta.teff_gspphot.tolist() == [7000.0]
    assert "bad.fits" in caplog.text
    assert "1 / 3 fichiers FITS introuvables" in caplog.text


@pytest.mark.integration
def test_spectral_filters_sampling_and_order(
    dimred_fits_writer, dimred_spectral_loader
):
    paths = []
    for i in range(8):
        path, wl = dimred_fits_writer(f"plan/{i}.fits.gz", flux=np.full(10, i + 1.0))
        paths.append(path)
    cat = pd.DataFrame(
        {
            "fits_name": [p.name for p in paths],
            "class": ["STAR"] * 6 + ["QSO"] * 2,
            "snr_r": [9.0, 10.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0],
            "teff_gspphot": np.arange(1.0, 9.0) * 1000.0,
        }
    )
    loader = dimred_spectral_loader(paths, wl, cat, random_state=17)
    # Filters must work without random sampling; equality checks count AND order.
    X, y, meta = loader.load(n_spectra=None, classes=["STAR"], normalize=False)
    np.testing.assert_array_equal(X[:, 0], [2.0, 3.0, 4.0, 5.0, 6.0])
    assert y.tolist() == ["STAR"] * 5
    np.testing.assert_array_equal(meta.teff_gspphot, X[:, 0] * 1000.0)
    a = loader.load(n_spectra=3, classes=["STAR"], normalize=False)
    b = loader.load(n_spectra=3, classes=["STAR"], normalize=False)
    for left, right in zip(a, b):
        np.testing.assert_array_equal(left, right)
    X, y, meta = a
    assert X.shape == (3, 10)
    assert set(y) == {"STAR"}
    assert set(X[:, 0]).issubset({2.0, 3.0, 4.0, 5.0, 6.0})
    np.testing.assert_array_equal(X[:, 0] * 1000.0, meta.teff_gspphot)
    # Fixed seeds 17 and 23 select different subsets, both unlike head(3).
    assert set(X[:, 0]) != {2.0, 3.0, 4.0}
    loader.random_state = 23
    other, _, other_meta = loader.load(n_spectra=3, classes=["STAR"], normalize=False)
    assert set(X[:, 0]) != set(other[:, 0])
    np.testing.assert_array_equal(other[:, 0] * 1000.0, other_meta.teff_gspphot)
    # Catalog order must be preserved when there is no subsampling.
    loader = dimred_spectral_loader(paths, wl, cat.iloc[::-1])
    X, _, meta = loader.load(normalize=False, snr_min=0)
    np.testing.assert_array_equal(X[:, 0], np.arange(8.0, 0.0, -1.0))
    np.testing.assert_array_equal(X[:, 0] * 1000.0, meta.teff_gspphot)


@pytest.mark.integration
def test_fits_inspection_valid_invalid_and_sample_bounds(
    dimred_fits_writer, dimred_spectral_loader, tmp_path
):
    path, wl = dimred_fits_writer()
    (tmp_path / "broken.fits").write_bytes(b"broken")
    loader = dimred_spectral_loader([path], wl)
    lines = loader.inspect_sample_fits(10)
    text = "\n".join(lines)
    assert lines[0].endswith(": 2")
    assert "Data shape : (2, 10)" in text
    assert "NaN count  : 0/10" in text
    assert "ERREUR" in text
    assert len(loader.inspect_sample_fits(0)) == 1
    assert len(loader.inspect_sample_fits(-1)) == 1


@pytest.mark.integration
def test_worker_count_does_not_change_spectral_results(
    dimred_fits_writer, dimred_spectral_loader, dimred_guarded_parallel, tmp_path
):
    paths = []
    for i in range(3):
        path, wl = dimred_fits_writer(f"worker/{i}.fits", flux=np.arange(10.0) + i + 1)
        paths.append(path)
    # A rejected row in the middle must not shift labels or metadata in workers.
    corrupt = tmp_path / "worker/broken.fits"
    corrupt.write_bytes(b"corrupt")
    paths.insert(1, corrupt)
    loader = dimred_spectral_loader(paths, wl)
    serial = loader.load()
    loader.n_jobs = 2
    parallel = loader.load()
    for left, right in zip(serial, parallel):
        np.testing.assert_array_equal(left, right)
    assert serial[2].teff_gspphot.tolist() == [1000, 1002, 1003]


@pytest.mark.integration
@pytest.mark.parametrize("value", [np.inf, -np.inf])
def test_scaling_rejects_nonfinite_features(dimred_feature_loader, value):
    loader = dimred_feature_loader(
        {"obsid": [1, 2, 3], "class": ["STAR"] * 3, "feature": [1.0, value, 3.0]},
        {"obsid": [1, 2, 3]},
    )
    with pytest.raises(ValueError, match="infinity|too large"):
        loader.load()


@pytest.mark.unit
def test_single_row_and_boolean_constant_selection(dimred_feature_loader):
    loader = dimred_feature_loader()
    assert loader._select_feature_columns(pd.DataFrame({"feature": [2.0]})) == []
    assert loader._select_feature_columns(
        pd.DataFrame(
            {
                "flag": [False, True, False],
                "constant": [True] * 3,
                "empty": [np.nan] * 3,
            }
        )
    ) == ["flag"]


@pytest.mark.integration
@pytest.mark.parametrize("missing_count", [1, 3, 10])
def test_nan_flux_masking_and_rejection(
    dimred_fits_writer, dimred_spectral_loader, missing_count
):
    flux = np.ones(10)
    flux[:missing_count] = np.nan
    path, wl = dimred_fits_writer(flux=flux)
    control, _ = dimred_fits_writer("nan-control.fits", flux=np.full(10, 7.0))
    X, y, meta = dimred_spectral_loader([path, control], wl).load(normalize=False)
    if missing_count == 1:
        assert X.shape == (2, 10)
        assert X[0, 0] == 0
        np.testing.assert_array_equal(X[0, 1:], np.ones(9))
        assert meta.teff_gspphot.tolist() == [1000, 1001]
    else:
        np.testing.assert_array_equal(X, np.full((1, 10), 7.0))
        assert y.tolist() == ["STAR"]
        assert meta.teff_gspphot.tolist() == [1001]


@pytest.mark.integration
def test_spectral_catalog_without_snr_and_missing_catalog_error(
    dimred_fits_writer, dimred_spectral_loader
):
    path, wl = dimred_fits_writer()
    loader = dimred_spectral_loader(
        [path], wl, {"fits_name": [path.name], "class": ["QSO"]}
    )
    X, y, _ = loader.load(normalize=False)
    np.testing.assert_array_equal(X, [np.arange(1.0, 11.0)])
    assert y.tolist() == ["QSO"]
    loader.catalog_path.unlink()
    with pytest.raises(FileNotFoundError):
        loader.load()


@pytest.mark.integration
def test_inspection_accepts_1d_and_empty_primary_hdu(
    dimred_fits_writer, dimred_spectral_loader, tmp_path
):
    path, wl = dimred_fits_writer(data=np.arange(10.0))
    fits.PrimaryHDU().writeto(tmp_path / "empty.fits")
    text = "\n".join(dimred_spectral_loader([path], wl).inspect_sample_fits(10))
    assert "Data shape : (10,)" in text
    assert "Data shape : None" in text
    assert "COEFF0 : ABSENT" in text
    assert "ERREUR" not in text


@pytest.mark.integration
@pytest.mark.parametrize("missing_count", [19, 20, 21])
def test_spectral_missing_fraction_boundary(
    dimred_fits_writer, dimred_spectral_loader, missing_count, caplog
):
    path, wl = dimred_fits_writer("coverage.fits", flux=np.ones(100))
    control, _ = dimred_fits_writer(
        "coverage-control.fits",
        coeff0=np.log10(3000.0),
        coeff1=0.01,
        flux=np.full(100, 7.0),
    )
    grid = np.r_[
        np.linspace(wl[0] - 100.0, wl[0] - 1.0, missing_count),
        np.linspace(wl[0], wl[-1], 100 - missing_count),
    ]
    with caplog.at_level(logging.INFO):
        X, y, meta = dimred_spectral_loader([path, control], grid).load(normalize=False)
    if missing_count == 19:
        np.testing.assert_allclose(X[0], np.r_[np.zeros(19), np.ones(81)])
        np.testing.assert_allclose(X[1], np.full(100, 7.0))
        assert meta.teff_gspphot.tolist() == [1000, 1001]
    else:
        # Accepted #55 contract: >=20% rejected. A valid control keeps N>0.
        np.testing.assert_allclose(X, np.full((1, 100), 7.0))
        assert y.tolist() == ["STAR"]
        assert meta.teff_gspphot.tolist() == [1001]
        assert any(
            r.message.startswith("Spectres rejetés") and r.message.endswith(": 1 / 2")
            for r in caplog.records
        )


@pytest.mark.integration
def test_normalization_uses_interpolated_not_native_median(
    dimred_fits_writer, dimred_spectral_loader
):
    _, wl = dimred_fits_writer()
    flux = 2.0 + 0.003 * (wl - wl[0])
    path, _ = dimred_fits_writer(flux=flux)
    grid = np.linspace(wl[5], wl[-1], 13)
    expected = 2.0 + 0.003 * (grid - wl[0])
    assert np.median(expected) > np.median(flux)
    X, _, _ = dimred_spectral_loader([path], grid).load()
    np.testing.assert_allclose(
        X[0], expected / np.median(expected), rtol=2e-13, atol=2e-13
    )


@pytest.mark.integration
def test_degenerate_native_grid_with_valid_target_does_not_prove_bug(
    dimred_fits_writer, dimred_spectral_loader
):
    # Corrected #52 counterexample: the target is independent and increasing.
    bad, _ = dimred_fits_writer("zero-step.fits", coeff1=0.0)
    control, _ = dimred_fits_writer(
        "zero-step-control.fits", coeff0=np.log10(3990.0), flux=np.full(10, 7.0)
    )
    grid = np.array([3999.0, 4000.0, 4001.0])
    X, _, meta = dimred_spectral_loader([bad, control], grid).load(normalize=False)
    np.testing.assert_array_equal(X, np.full((1, 3), 7.0))
    assert meta.teff_gspphot.tolist() == [1001]


@pytest.mark.integration
def test_active_quality_filter_excludes_nan_snr_in_both_loaders(
    dimred_feature_loader, dimred_fits_writer, dimred_spectral_loader
):
    features = {
        "obsid": [1, 2, 3, 4],
        "class": ["STAR"] * 4,
        "snr_r": [np.nan, 9.0, 10.0, 20.0],
        "feature": [1.0, 2.0, 3.0, 4.0],
    }
    X, _, _ = dimred_feature_loader(features, {"obsid": [1, 2, 3, 4]}).load(scale=False)
    np.testing.assert_array_equal(X[:, 0], [3.0, 4.0])
    paths = []
    for i in range(4):
        path, wl = dimred_fits_writer(f"snr/{i}.fits", flux=np.full(10, i + 1.0))
        paths.append(path)
    cat = {
        "fits_name": [p.name for p in paths],
        "class": features["class"],
        "snr_r": features["snr_r"],
    }
    X, _, _ = dimred_spectral_loader(paths, wl, cat).load(normalize=False)
    np.testing.assert_array_equal(X[:, 0], [3.0, 4.0])


@pytest.mark.integration
def test_fixture_factories_keep_previous_csv_inputs(
    dimred_feature_loader, dimred_spectral_loader, dimred_fits_writer
):
    first = dimred_feature_loader()
    first_bytes = first.features_path.read_bytes()
    second = dimred_feature_loader({"obsid": [1], "feature": [9.0]}, {"obsid": [1]})
    assert first.features_path != second.features_path
    assert first.features_path.read_bytes() == first_bytes
    path, wl = dimred_fits_writer()
    a = dimred_spectral_loader([path], wl)
    original = a.catalog_path.read_bytes()
    b = dimred_spectral_loader([], wl)
    assert a.catalog_path != b.catalog_path
    assert a.catalog_path.read_bytes() == original
