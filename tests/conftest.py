"""Synthetic CSV/FITS fixtures for the isolated dimred loader audit."""

import importlib.util
from pathlib import Path

from astropy.io import fits
import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="session")
def loader_module():
    # Load the actual source without executing dimred's optional Torch facade
    # (known issue #21). This does not validate `import dimred`.
    path = Path(__file__).resolve().parents[1] / "src/dimred/data_loader.py"
    spec = importlib.util.spec_from_file_location("dimred_loader_audit", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def feature_loader(tmp_path, loader_module):
    def make(features=None, catalog=None, **kwargs):
        if features is None:
            features = {
                "obsid": [30, 10, 20, 40],
                "class": ["STAR", "QSO", "STAR", "GALAXY"],
                "snr_r": [30.0, 10.0, 9.0, 40.0],
                "feature_b": [3.0, 1.0, 2.0, 4.0],
                "feature_a": [9.0, 1.0, 4.0, 16.0],
            }
        if catalog is None:
            catalog = {
                "obsid": [10, 40, 30, 20],
                "teff_gspphot": [1000.0, 4000.0, 3000.0, 2000.0],
            }
        feature_path = tmp_path / "features.csv"
        catalog_path = tmp_path / "catalog.csv"
        pd.DataFrame(features).to_csv(feature_path, index=False)
        pd.DataFrame(catalog).to_csv(catalog_path, index=False)
        return loader_module.DimRedDataLoader(feature_path, catalog_path, **kwargs)

    return make


@pytest.fixture
def fits_writer(tmp_path):
    def write(
        name="plan/spectrum.fits",
        flux=None,
        coeff0=None,
        coeff1=0.001,
        data=None,
        header=None,
    ):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if flux is None:
            flux = np.arange(1.0, 11.0)
        if coeff0 is None:
            coeff0 = np.log10(4000.0)
        wavelength = 10 ** (coeff0 + np.arange(len(flux)) * coeff1)
        if data is None:
            data = np.vstack([flux, np.ones(len(flux))])
        hdu = fits.PrimaryHDU(data=data)
        hdu.header["COEFF0"] = coeff0
        hdu.header["COEFF1"] = coeff1
        if header:
            hdu.header.update(header)
        hdu.writeto(path, overwrite=True)
        return path, wavelength

    return write


@pytest.fixture
def spectral_loader(tmp_path, loader_module):
    def make(paths, grid, catalog=None, **kwargs):
        if catalog is None:
            catalog = {
                "fits_name": [p.name for p in paths],
                "class": ["STAR"] * len(paths),
                "snr_r": [20.0] * len(paths),
                "teff_gspphot": list(range(1000, 1000 + len(paths))),
            }
        catalog_path = tmp_path / "spectral_catalog.csv"
        pd.DataFrame(catalog).to_csv(catalog_path, index=False)
        return loader_module.SpectralMatrixLoader(
            tmp_path, catalog_path, wl_grid=np.asarray(grid), n_jobs=1, **kwargs
        )

    return make
