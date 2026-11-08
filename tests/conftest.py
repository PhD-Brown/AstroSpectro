"""Synthetic CSV/FITS fixtures for the isolated dimred loader audit."""

import importlib.util
from itertools import count
from pathlib import Path

from astropy.io import fits
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def embedding_module():
    # Isolate the optional facade (#21), but execute the real embedding source.
    path = Path(__file__).resolve().parents[1] / "src/dimred/embedding.py"
    spec = importlib.util.spec_from_file_location("dimred_embedding_audit", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def embedding_backends(monkeypatch):
    """Capture estimator boundaries without claiming algorithm validation."""
    import sys
    import types
    import sklearn.manifold

    instances = []

    class Estimator:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.input = None
            self.output = None
            instances.append(self)

        def fit_transform(self, X):
            self.input = np.array(X, copy=True)
            self.n_features_in_ = X.shape[1]
            self.output = np.column_stack(
                [X[:, j % X.shape[1]] for j in range(self.kwargs["n_components"])]
            )
            return self.output

        def transform(self, X):
            if X.shape[1] != self.n_features_in_:
                raise ValueError("Feature count differs from fitted input")
            return np.column_stack(
                [X[:, j % X.shape[1]] for j in range(self.kwargs["n_components"])]
            )

    monkeypatch.setitem(sys.modules, "umap", types.SimpleNamespace(UMAP=Estimator))
    monkeypatch.setattr(sklearn.manifold, "TSNE", Estimator)
    return instances


@pytest.fixture
def embedding_figures(monkeypatch):
    """Capture exact Plotly tables and export requests, without large HTML."""
    import plotly.express as px

    calls = []

    class Figure:
        def __init__(self, data_frame, **kwargs):
            self.table = data_frame.copy(deep=True)
            self.kwargs = kwargs
            self.paths = []
            calls.append(self)

        def update_traces(self, **kwargs):
            return self

        def update_layout(self, **kwargs):
            return self

        def write_html(self, path):
            self.paths.append(Path(path))

    monkeypatch.setattr(px, "scatter_3d", Figure)
    return calls


@pytest.fixture(scope="session")
def dimred_pca_class():
    # Exercise the real source; the optional-dependency facade is tracked in #21.
    path = Path(__file__).resolve().parents[1] / "src/dimred/pca_analyzer.py"
    spec = importlib.util.spec_from_file_location("dimred_pca_audit", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.PCAAnalyzer


@pytest.fixture
def pca_axes():
    # Six paired observations: covariance eigenvalues 18/5, 8/5, 2/5.
    # Rational rotation mixes features without imposing sklearn's axis signs.
    basis = np.array([[0.6, 0.8, 0.0], [-0.8, 0.6, 0.0], [0.0, 0.0, 1.0]])
    coordinates = np.array(
        [
            [3.0, 0, 0],
            [-3.0, 0, 0],
            [0, 2.0, 0],
            [0, -2.0, 0],
            [0, 0, 1.0],
            [0, 0, -1.0],
        ]
    )
    mean = np.array([10.0, -7.0, 4.0])
    return mean + coordinates @ basis, basis, coordinates, mean


@pytest.fixture(scope="session")
def dimred_loader_module():
    # Load the actual source without executing dimred's optional Torch facade
    # (known issue #21). This does not validate `import dimred`.
    path = Path(__file__).resolve().parents[1] / "src/dimred/data_loader.py"
    spec = importlib.util.spec_from_file_location("dimred_loader_audit", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def dimred_feature_loader(tmp_path, dimred_loader_module):
    instances = count()

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
        case_dir = tmp_path / f"dimred-features-{next(instances)}"
        case_dir.mkdir()
        feature_path = case_dir / "features.csv"
        catalog_path = case_dir / "catalog.csv"
        pd.DataFrame(features).to_csv(feature_path, index=False)
        pd.DataFrame(catalog).to_csv(catalog_path, index=False)
        return dimred_loader_module.DimRedDataLoader(
            feature_path, catalog_path, **kwargs
        )

    return make


@pytest.fixture
def dimred_fits_writer(tmp_path):
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
def dimred_spectral_loader(tmp_path, dimred_loader_module):
    instances = count()

    def make(paths, grid, catalog=None, **kwargs):
        if catalog is None:
            catalog = {
                "fits_name": [p.name for p in paths],
                "class": ["STAR"] * len(paths),
                "snr_r": [20.0] * len(paths),
                "teff_gspphot": list(range(1000, 1000 + len(paths))),
            }
        case_dir = tmp_path / f"dimred-spectra-{next(instances)}"
        case_dir.mkdir()
        catalog_path = case_dir / "spectral_catalog.csv"
        pd.DataFrame(catalog).to_csv(catalog_path, index=False)
        return dimred_loader_module.SpectralMatrixLoader(
            tmp_path, catalog_path, wl_grid=np.asarray(grid), n_jobs=1, **kwargs
        )

    return make


@pytest.fixture
def dimred_guarded_parallel(monkeypatch):
    """Keep real loky workers offline without disabling their IPC machinery."""
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
    return guarded_parallel
