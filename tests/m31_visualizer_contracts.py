"""Shared synthetic cases for M3.1 green tests and explicit red proofs."""

import builtins
import importlib.util
import os
from pathlib import Path
import random
from types import SimpleNamespace

from astropy.io import fits
import matplotlib
import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest


@pytest.fixture
def tools_case(tmp_path, monkeypatch):
    """Load the worktree source; isolate displays, RNG, builtins and figures."""
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    import IPython.display

    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "src"))
    source = Path(
        os.environ.get("M31_VISUALIZER_SOURCE", root / "src/tools/visualizer.py")
    ).resolve()
    spec = importlib.util.spec_from_file_location("m31_visualizer", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert (
        Path(module.AstroVisualizer.__init__.__code__.co_filename).resolve()
        == (source).resolve()
    )
    paths = {"PROJECT_ROOT": str(tmp_path)}
    for key in [
        "RAW_DATA_DIR",
        "CATALOG_DIR",
        "PROCESSED_DIR",
        "MODELS_DIR",
        "LOGS_DIR",
        "REPORTS_DIR",
    ]:
        folder = tmp_path / key.lower()
        folder.mkdir()
        paths[key] = str(folder)
    displayed = []
    monkeypatch.setattr(module, "display", lambda *items: displayed.extend(items))
    monkeypatch.setattr(
        IPython.display, "display", lambda *items, **kw: displayed.extend(items)
    )
    monkeypatch.setattr(module, "tqdm", lambda items, **kw: items)
    monkeypatch.setattr(plt, "show", lambda: None)
    monkeypatch.setattr(builtins, "log_analyses", pd.DataFrame(), raising=False)
    py_rng, np_rng = random.getstate(), np.random.get_state()
    previous_figures = set(plt.get_fignums())

    def write(
        name="plan/spectrum.fits.gz",
        flux=None,
        *,
        start=4800.0,
        step=0.0001,
        invvar=None,
        header=None,
        data=None,
    ):
        flux = np.asarray([2.0, 4.0, 6.0, 8.0, 10.0] if flux is None else flux, float)
        invvar = (
            np.full(len(flux), 4.0) if invvar is None else np.asarray(invvar, float)
        )
        hdu = fits.PrimaryHDU(data=np.vstack([flux, invvar]) if data is None else data)
        hdu.header.update(
            {
                "COEFF0": np.log10(start),
                "COEFF1": step,
                "DESIG": name,
                "SUBCLASS": "F5",
                "PLANID": name.split("/")[0],
                "RA": 30.0,
                "DEC": 20.0,
                "VACUUM": True,
                "HELIO": True,
                "HELIO_RV": 12.0,
            }
        )
        if header:
            hdu.header.update(header)
        path = Path(paths["RAW_DATA_DIR"]) / name
        path.parent.mkdir(parents=True, exist_ok=True)
        hdu.writeto(path, overwrite=True)
        # Geometric progression: the logarithmic dispersion means adjacent
        # wavelengths have fixed ratio 10**step, anchored by the first pixel.
        wave = start * np.power(10.0**step, np.arange(len(flux)))
        return name, wave, flux.copy()

    def gaussian(*, invalid=False, amplitude=0.5, sigma=0.8):
        step = 0.00001
        start = 4861.0 / (10.0**step) ** 200
        wave = start * np.power(10.0**step, np.arange(401))
        flux = 1 - amplitude * np.exp(-0.5 * ((wave - 4861.0) / sigma) ** 2)
        if invalid:
            flux[[0, 40]] = [np.nan, np.inf]
        return write(flux=flux, start=start, step=step)

    def catalog(table):
        path = Path(paths["CATALOG_DIR"]) / "master_catalog_temp.csv"
        pd.DataFrame(table).to_csv(path, sep="|", index=False)
        return path

    st = SimpleNamespace(figures=[], errors=[], markdowns=[], jsons=[])
    st.plotly_chart = lambda fig, **kw: st.figures.append(fig)
    st.error = st.errors.append
    st.markdown = lambda value, **kw: st.markdowns.append(value)
    st.json = st.jsons.append
    with matplotlib.rc_context():
        viz = module.AstroVisualizer(paths)
        try:
            yield SimpleNamespace(
                module=module,
                viz=viz,
                paths=paths,
                write=write,
                gaussian=gaussian,
                catalog=catalog,
                plt=plt,
                displayed=displayed,
                st=st,
                monkeypatch=monkeypatch,
                tmp=tmp_path,
            )
        finally:
            random.setstate(py_rng)
            np.random.set_state(np_rng)
            for number in set(plt.get_fignums()) - previous_figures:
                plt.close(number)


def render_spectrum(c, route, path, *, allow_rejection=False):
    """Render through real FITS/preprocessing/detection and return peak pairs."""
    if route == "plotly":
        header = c.viz._plot_spectrum_analysis_plotly(
            c.st, str(Path(c.paths["RAW_DATA_DIR"]) / path), 0.2, 8
        )
        if header is None and allow_rejection and c.st.errors:
            return None, None, None
        assert header is not None, c.st.errors
        fig = c.st.figures[-1]
        traces = {t.name: t for t in fig.data}
        red = traces.get("Detected Peaks")
        green = traces.get("Matched Peaks")

        def pairs(t):
            return np.column_stack((t.x, t.y)) if t is not None else np.empty((0, 2))

        return pairs(red), pairs(green), fig
    if route == "notebook":
        c.viz._plot_peak_detection_notebook(path, 0.2, 8, (4880, 4840), (2, 0), "None")
    else:
        c.viz._plot_spectrum_analysis(path, 0.2, 8, (4880, 4840), (2, 0))
    if not c.plt.get_fignums() and allow_rejection:
        return None, None, None
    assert c.plt.get_fignums(), "Spectrum analysis produced no figure"
    ax = c.plt.gcf().axes[0]
    collections = {sc.get_label(): sc for sc in ax.collections}

    def pairs(name):
        return (
            np.asarray(collections[name].get_offsets())
            if name in collections
            else np.empty((0, 2))
        )

    return pairs("Detected peaks"), pairs("Matched peaks"), ax


def assert_peak_pairs(red, green, wave, flux):
    """A known Gaussian minimum is pixel 200; both encodings must agree."""
    assert_allclose(
        red,
        [[wave[200], flux[200]]],
        rtol=0,
        atol=2e-9,
        err_msg="Detected peak lost its original pixel identity",
    )
    assert_allclose(
        green,
        [[wave[200], flux[200]]],
        rtol=0,
        atol=2e-9,
        err_msg="Matched peak lost its original pixel identity",
    )


def shap_case(c, values):
    """Control only the SHAP boundary; preserve real visualizer aggregation."""
    from sklearn.pipeline import Pipeline
    from sklearn.dummy import DummyClassifier

    X = pd.DataFrame(
        {
            "alpha": [1.0, 2.0, 3.0, 4.0],
            "beta": [7.0, 9.0, 11.0, 13.0],
            "gamma": [20.0, 30.0, 40.0, 50.0],
        }
    )
    pipe = Pipeline([("clf", DummyClassifier().fit(X, ["F", "G", "F", "G"]))])
    wrapper = SimpleNamespace(model_pipeline=pipe, feature_names_used=list(X.columns))
    calls = []
    explanation = c.module.shap.Explanation(
        values=np.asarray(values), data=X.values, feature_names=list(X.columns)
    )

    class Explainer:
        def __call__(self, data):
            calls.append(np.array(data, copy=True))
            return explanation

    c.monkeypatch.setattr(
        c.module.SpectralClassifier, "load_model", staticmethod(lambda path: wrapper)
    )
    c.monkeypatch.setattr(c.viz, "_get_features_sample", lambda *args, **kw: X.copy())
    c.monkeypatch.setattr(c.module.shap, "TreeExplainer", lambda model: Explainer())
    return calls, X
