"""Offline contracts for data_loader.py; red counterexamples live in proofs_*.

Expected values use identity joins, population moments, and analytic linear
spectra. No production data, optional Torch import, or learned models are used.
"""

import logging

from astropy.io import fits
import numpy as np
import pandas as pd
import pytest


@pytest.mark.unit
def test_initial_state_and_public_utilities(feature_loader):
    loader = feature_loader(random_state=17)
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
    feature_loader, catalog_order, feature_order
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
    loader = feature_loader(features, catalog)
    X, y, meta = loader.load(scale=False, snr_min=0)
    np.testing.assert_array_equal(X, features[["feature_b", "feature_a"]])
    np.testing.assert_array_equal(y, features["class"])
    np.testing.assert_array_equal(meta["teff_gspphot"], features["obsid"] * 100)
    assert meta.index.tolist() == list(range(4))
    assert loader.get_feature_names() == ["feature_b", "feature_a"]
    np.testing.assert_array_equal(loader._df_merged["obsid"], features["obsid"])


@pytest.mark.integration
def test_snr_class_filters_and_missing_metadata_preserve_alignment(feature_loader):
    loader = feature_loader(catalog={"obsid": [10], "teff_gspphot": [1000.0]})
    X, y, meta = loader.load(scale=False, classes=["STAR", "QSO"])
    np.testing.assert_array_equal(X, [[3.0, 9.0], [1.0, 1.0]])
    np.testing.assert_array_equal(y, ["STAR", "QSO"])
    assert np.isnan(meta.teff_gspphot.iloc[0])
    assert meta.teff_gspphot.iloc[1] == 1000
    assert meta.snr_r.tolist() == [30.0, 10.0]


@pytest.mark.integration
def test_custom_merge_key_and_feature_side_precedence(feature_loader):
    loader = feature_loader(
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
def test_snr_absent_warns_and_alternate_band_is_used(feature_loader, caplog):
    loader = feature_loader(
        {
            "obsid": [1, 2, 3],
            "class": ["STAR"] * 3,
            "feature": [1.0, 2.0, 3.0],
            "snr_g": [5.0, 10.0, np.nan],
        },
        {"obsid": [1, 2, 3]},
    )
    with caplog.at_level(logging.WARNING):
        X, _, _ = loader.load(scale=False)
    assert len(X) == 3
    assert "snr_r" in caplog.text
    # Selection removes a constant column after filtering; inspect IDs directly.
    loader.load(scale=False, snr_band="snr_g")
    assert loader._df_merged.obsid.tolist() == [2]


@pytest.mark.unit
@pytest.mark.parametrize("spectro_only", [False, True])
def test_column_selection_order_nan_boundary_and_exclusions(
    feature_loader, spectro_only
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
    selected = feature_loader()._select_feature_columns(
        frame, nan_threshold=0.25, spectro_only=spectro_only
    )
    expected = ["feature_b", "feature_a", "at_threshold"]
    if not spectro_only:
        expected += ["ra", "fiber_id", "radial_velocity"]
    assert selected == expected


@pytest.mark.integration
def test_nan_column_then_row_filtering(feature_loader):
    loader = feature_loader(
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
def test_population_scaling_and_inverse_preserve_feature_order(feature_loader):
    loader = feature_loader()
    X, _, _ = loader.load()
    raw = np.array([[3.0, 9.0], [1.0, 1.0], [4.0, 16.0]])
    expected = (raw - raw.mean(axis=0)) / raw.std(axis=0, ddof=0)
    np.testing.assert_allclose(X, expected, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(loader.inverse_transform(X), raw, rtol=1e-14)
    assert loader.get_feature_names() == ["feature_b", "feature_a"]
    # This validates descriptive whole-population scaling, not train-only use.


@pytest.mark.integration
def test_balance_cap_seed_no_replacement_and_alignment(feature_loader):
    frame = pd.DataFrame(
        {
            "obsid": np.arange(24),
            "class": ["STAR"] * 20 + ["QSO"] * 4,
            "feature": np.arange(24, dtype=float),
        }
    )
    cat = pd.DataFrame({"obsid": np.arange(24), "teff_gspphot": np.arange(24) * 100.0})
    loader = feature_loader(frame, cat, random_state=19)
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
def test_meta_is_independent_copy_with_reset_index(feature_loader):
    frame = pd.DataFrame(
        {"class": ["STAR", "QSO"], "ra": [1.0, 2.0], "feature": [10.0, 20.0]},
        index=[8, 5],
    )
    meta = feature_loader()._build_meta(frame)
    assert meta.columns.tolist() == ["class", "ra"]
    assert meta.index.tolist() == [0, 1]
    meta.loc[0, "ra"] = -999
    assert frame.loc[8, "ra"] == 1


@pytest.mark.unit
@pytest.mark.parametrize("source", ["features", "catalog"])
def test_missing_csv_reports_path(feature_loader, source):
    loader = feature_loader()
    path = getattr(loader, source + "_path")
    path.unlink()
    with pytest.raises(FileNotFoundError, match=path.name):
        getattr(loader, "_load_" + source)()


@pytest.mark.unit
@pytest.mark.parametrize("source", ["features", "catalog"])
def test_csv_fallback_retains_good_rows_and_warns(feature_loader, caplog, source):
    loader = feature_loader()
    path = getattr(loader, source + "_path")
    path.write_text("obsid,feature\n1,2\n2,3\n3,4,extra\n4,5\n", encoding="utf-8")
    with caplog.at_level(logging.WARNING), pytest.warns(pd.errors.ParserWarning):
        frame = getattr(loader, "_load_" + source)()
    assert frame.obsid.tolist() == [1, 2, 4]
    assert "ParserError" in caplog.text


@pytest.mark.unit
@pytest.mark.parametrize("source", ["features", "catalog"])
def test_truncated_csv_reports_omitted_rows(feature_loader, source, caplog):
    loader = feature_loader()
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
def test_empty_csv_has_explicit_parser_error(feature_loader, source):
    loader = feature_loader()
    getattr(loader, source + "_path").write_bytes(b"")
    with pytest.raises(pd.errors.EmptyDataError):
        getattr(loader, "_load_" + source)()


@pytest.mark.unit
def test_spectra_mode_delegation_error_precedes_io(feature_loader):
    loader = feature_loader()
    loader.features_path.unlink()
    with pytest.raises(NotImplementedError, match="SpectralMatrixLoader"):
        loader.load(mode="spectra")


@pytest.mark.unit
@pytest.mark.parametrize("side", ["features", "catalog"])
def test_missing_merge_key_fails(feature_loader, side):
    frame = pd.DataFrame(
        {"obsid": [1, 2], "class": ["STAR", "QSO"], "feature": [1.0, 2.0]}
    )
    cat = pd.DataFrame({"obsid": [1, 2], "teff_gspphot": [1000.0, 2000.0]})
    if side == "features":
        frame = frame.drop(columns="obsid")
    else:
        cat = cat.drop(columns="obsid")
    with pytest.raises(KeyError, match="obsid"):
        feature_loader(frame, cat).load(scale=False)


@pytest.mark.integration
def test_duplicate_feature_observations_are_not_silently_deduplicated(feature_loader):
    loader = feature_loader(
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
    tmp_path, loader_module, fits_writer
):
    path, _ = fits_writer("nested/one.fits.gz")
    loader = loader_module.SpectralMatrixLoader(tmp_path, tmp_path / "catalog.csv")
    assert np.all(np.diff(loader.wl_grid) > 0)
    assert loader.wl_grid[0] == 3690.0
    assert loader.wl_grid[-1] < 9100.0
    np.testing.assert_allclose(np.diff(loader.wl_grid), 1.38, rtol=0, atol=2e-12)
    index = loader._build_fits_index()
    assert index == {path.name: path}
    assert loader._build_fits_index() is index


@pytest.mark.integration
@pytest.mark.parametrize("coeff1", [0.001, -0.001])
@pytest.mark.parametrize("normalize", [False, True])
def test_fits_analytic_linear_interpolation_and_median(
    fits_writer, spectral_loader, coeff1, normalize
):
    _, wavelength = fits_writer(coeff1=coeff1)
    flux = 2 + 0.003 * (wavelength - wavelength.min())
    path, wavelength = fits_writer(flux=flux, coeff1=coeff1)
    # Midpoints in Angstroms, not log wavelength: a linear function is exact.
    ordered = np.sort(wavelength)
    grid = np.r_[ordered[0], (ordered[:-1] + ordered[1:]) / 2, ordered[-1]]
    expected = 2 + 0.003 * (grid - wavelength.min())
    if normalize:
        expected /= np.median(expected)
    X, y, meta = spectral_loader([path], grid).load(normalize=normalize)
    np.testing.assert_allclose(X[0], expected, rtol=2e-13, atol=2e-13)
    assert y.tolist() == ["STAR"]
    assert meta["class"].tolist() == ["STAR"]


@pytest.mark.integration
def test_partial_boundary_fill_and_too_short_spectral_coverage(
    fits_writer, spectral_loader
):
    path, wl = fits_writer()
    # One outside pixel among eleven is retained; the current documented fill is 0.
    X, _, _ = spectral_loader([path], np.r_[wl[0] - 1, wl]).load(normalize=False)
    np.testing.assert_array_equal(X, [np.r_[0.0, np.arange(1.0, 11.0)]])
    X, y, meta = spectral_loader([path], wl + 10000.0).load(normalize=False)
    assert X.shape == (0, len(wl))
    assert y.shape == (0,)
    assert meta.empty


@pytest.mark.integration
@pytest.mark.parametrize("median", [0.0, -2.0])
def test_nonpositive_median_is_left_unscaled(fits_writer, spectral_loader, median):
    path, wl = fits_writer(flux=np.full(10, median))
    X, _, _ = spectral_loader([path], wl).load()
    np.testing.assert_array_equal(X, np.full((1, 10), median))


@pytest.mark.integration
@pytest.mark.parametrize(
    "invalid",
    ["corrupt", "no_data", "one_dimensional", "three_dimensional", "missing_header"],
)
def test_invalid_fits_rejected_with_diagnostic_and_alignment(
    fits_writer, spectral_loader, invalid, tmp_path, caplog
):
    good, wl = fits_writer("nested/good.fits", flux=np.full(10, 7.0))
    bad = tmp_path / "bad.fits"
    if invalid == "corrupt":
        bad.write_bytes(b"not a FITS file")
    elif invalid == "no_data":
        fits.PrimaryHDU().writeto(bad)
    elif invalid == "missing_header":
        fits.PrimaryHDU(data=np.ones((2, 10))).writeto(bad)
    else:
        data = np.ones(10) if invalid == "one_dimensional" else np.ones((2, 2, 10))
        bad, _ = fits_writer("bad.fits", data=data)
    missing = tmp_path / "missing.fits"
    cat = {
        "fits_name": [bad.name, missing.name, good.name],
        "class": ["BAD", "MISSING", "STAR"],
        "snr_r": [30.0, 30.0, 30.0],
        "teff_gspphot": [1.0, 2.0, 7000.0],
    }
    with caplog.at_level(logging.INFO):
        X, y, meta = spectral_loader([bad, missing, good], wl, cat).load(
            normalize=False
        )
    np.testing.assert_array_equal(X, np.full((1, 10), 7.0))
    assert y.tolist() == ["STAR"]
    assert meta.teff_gspphot.tolist() == [7000.0]
    assert "bad.fits" in caplog.text and "missing" not in y
    assert "introuvables" in caplog.text


@pytest.mark.integration
def test_spectral_filters_sampling_and_order(fits_writer, spectral_loader):
    paths = []
    for i in range(8):
        path, wl = fits_writer(f"plan/{i}.fits.gz", flux=np.full(10, i + 1.0))
        paths.append(path)
    cat = pd.DataFrame(
        {
            "fits_name": [p.name for p in paths],
            "class": ["STAR"] * 6 + ["QSO"] * 2,
            "snr_r": [9.0, 10.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0],
            "teff_gspphot": np.arange(1.0, 9.0) * 1000.0,
        }
    )
    loader = spectral_loader(paths, wl, cat, random_state=17)
    a = loader.load(n_spectra=3, classes=["STAR"], normalize=False)
    b = loader.load(n_spectra=3, classes=["STAR"], normalize=False)
    for left, right in zip(a, b):
        np.testing.assert_array_equal(left, right)
    X, y, meta = a
    assert X.shape == (3, 10)
    assert set(y) == {"STAR"}
    assert set(X[:, 0]).issubset({2.0, 3.0, 4.0, 5.0, 6.0})
    np.testing.assert_array_equal(X[:, 0] * 1000.0, meta.teff_gspphot)
    # Catalog order must be preserved when there is no subsampling.
    loader = spectral_loader(paths, wl, cat.iloc[::-1])
    X, _, meta = loader.load(normalize=False, snr_min=0)
    np.testing.assert_array_equal(X[:, 0], np.arange(8.0, 0.0, -1.0))
    np.testing.assert_array_equal(X[:, 0] * 1000.0, meta.teff_gspphot)


@pytest.mark.integration
def test_fits_inspection_valid_invalid_and_sample_bounds(
    fits_writer, spectral_loader, tmp_path
):
    path, wl = fits_writer()
    (tmp_path / "broken.fits").write_bytes(b"broken")
    loader = spectral_loader([path], wl)
    lines = loader.inspect_sample_fits(10)
    text = "\n".join(lines)
    assert "2" in lines[0]
    assert "Data shape : (2, 10)" in text
    assert "NaN count  : 0/10" in text
    assert "ERREUR" in text
    assert len(loader.inspect_sample_fits(0)) == 1
    assert len(loader.inspect_sample_fits(-1)) == 1


@pytest.mark.integration
def test_worker_count_does_not_change_spectral_results(
    fits_writer, spectral_loader, monkeypatch, tmp_path
):
    import joblib

    real_parallel = joblib.Parallel

    def deny_worker_network():
        import socket

        def blocked(*args, **kwargs):
            raise AssertionError("Network forbidden in spectral audit worker")

        socket.socket.connect = blocked
        socket.socket.connect_ex = blocked
        socket.create_connection = blocked

    def guarded_parallel(*args, **kwargs):
        return real_parallel(*args, initializer=deny_worker_network, **kwargs)

    def worker_has_network_guard():
        import socket

        return socket.create_connection.__name__ == "blocked"

    monkeypatch.setattr(joblib, "Parallel", guarded_parallel)
    assert all(
        guarded_parallel(n_jobs=2, backend="loky")(
            joblib.delayed(worker_has_network_guard)() for _ in range(2)
        )
    )
    paths = []
    for i in range(3):
        path, wl = fits_writer(f"worker/{i}.fits", flux=np.arange(10.0) + i + 1)
        paths.append(path)
    # A rejected row in the middle must not shift labels or metadata in workers.
    corrupt = tmp_path / "worker/broken.fits"
    corrupt.write_bytes(b"corrupt")
    paths.insert(1, corrupt)
    loader = spectral_loader(paths, wl)
    serial = loader.load()
    loader.n_jobs = 2
    parallel = loader.load()
    for left, right in zip(serial, parallel):
        np.testing.assert_array_equal(left, right)
    assert serial[2].teff_gspphot.tolist() == [1000, 1002, 1003]


@pytest.mark.integration
@pytest.mark.parametrize("value", [np.inf, -np.inf])
def test_scaling_rejects_nonfinite_features(feature_loader, value):
    loader = feature_loader(
        {"obsid": [1, 2, 3], "class": ["STAR"] * 3, "feature": [1.0, value, 3.0]},
        {"obsid": [1, 2, 3]},
    )
    with pytest.raises(ValueError, match="infinity|too large"):
        loader.load()


@pytest.mark.unit
def test_single_row_and_boolean_constant_selection(feature_loader):
    loader = feature_loader()
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
def test_nan_flux_masking_and_rejection(fits_writer, spectral_loader, missing_count):
    flux = np.ones(10)
    flux[:missing_count] = np.nan
    path, wl = fits_writer(flux=flux)
    X, y, meta = spectral_loader([path], wl).load(normalize=False)
    if missing_count == 1:
        assert X.shape == (1, 10)
        assert X[0, 0] == 0
        np.testing.assert_array_equal(X[0, 1:], np.ones(9))
    else:
        assert X.shape == (0, 10)
        assert y.shape == (0,)
        assert meta.empty


@pytest.mark.integration
def test_all_missing_fits_returns_empty_aligned_arrays(
    spectral_loader, tmp_path, caplog
):
    with caplog.at_level("ERROR"):
        X, y, meta = spectral_loader(
            [tmp_path / "absent.fits"], [4000.0, 4100.0]
        ).load()
    assert X.shape == (0, 2)
    assert y.shape == (0,)
    assert meta.empty
    assert "Aucun fichier FITS" in caplog.text


@pytest.mark.integration
def test_spectral_catalog_without_snr_and_missing_catalog_error(
    fits_writer, spectral_loader
):
    path, wl = fits_writer()
    loader = spectral_loader([path], wl, {"fits_name": [path.name], "class": ["QSO"]})
    X, y, _ = loader.load(normalize=False)
    np.testing.assert_array_equal(X, [np.arange(1.0, 11.0)])
    assert y.tolist() == ["QSO"]
    loader.catalog_path.unlink()
    with pytest.raises(FileNotFoundError):
        loader.load()


@pytest.mark.integration
def test_inspection_accepts_1d_and_empty_primary_hdu(
    fits_writer, spectral_loader, tmp_path
):
    path, wl = fits_writer(data=np.arange(10.0))
    fits.PrimaryHDU().writeto(tmp_path / "empty.fits")
    text = "\n".join(spectral_loader([path], wl).inspect_sample_fits(10))
    assert "Data shape : (10,)" in text
    assert "Data shape : None" in text
    assert "COEFF0 : ABSENT" in text
    assert "ERREUR" not in text
