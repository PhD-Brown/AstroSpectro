"""Shared synthetic cases for M3.1 green tests and explicit red proofs."""

import builtins
import importlib.util
import os
from pathlib import Path
import random
import sys
import json
import hashlib
import re
import unicodedata
from types import SimpleNamespace

from astropy.io import fits
import matplotlib
import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest


@pytest.fixture
def tools_case(tmp_path, monkeypatch, request):
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

    def gaussian(*, invalid=False, amplitude=0.5, sigma=0.8, continuum=1.0):
        step = 0.00001
        start = 4861.0 / (10.0**step) ** 200
        wave = start * np.power(10.0**step, np.arange(401))
        flux = 1 - amplitude * np.exp(-0.5 * ((wave - 4861.0) / sigma) ** 2)
        if invalid:
            flux[[0, 40]] = [np.nan, np.inf]
        return write(flux=continuum * flux, start=start, step=step)

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
        trace_dir = os.environ.get("M31_TRACE_DIR")
        reached = set()
        previous_trace = sys.gettrace()

        def trace(frame, event, arg):
            if frame.f_code.co_filename == str(source):
                if event == "line":
                    reached.add(frame.f_lineno)
                return trace
            return None

        if trace_dir:
            sys.settrace(trace)
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
            if trace_dir:
                sys.settrace(previous_trace)
                output = Path(trace_dir)
                output.mkdir(parents=True, exist_ok=True)
                node = hashlib.sha256(request.node.nodeid.encode()).hexdigest()[:16]
                (output / f"{node}.json").write_text(
                    json.dumps(
                        {
                            "node": request.node.nodeid,
                            "source": str(source),
                            "lines": sorted(reached),
                        }
                    ),
                    encoding="utf-8",
                )
            random.setstate(py_rng)
            np.random.set_state(np_rng)
            for number in set(plt.get_fignums()) - previous_figures:
                plt.close(number)


def render_spectrum(c, route, path, *, allow_rejection=False, window=8):
    """Render through real FITS/preprocessing/detection and return peak pairs."""
    if route == "plotly":
        header = c.viz._plot_spectrum_analysis_plotly(
            c.st, str(Path(c.paths["RAW_DATA_DIR"]) / path), 0.2, window
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
        c.viz._plot_peak_detection_notebook(
            path, 0.2, window, (4880, 4840), (2, 0), "None"
        )
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


def assert_curve_identities(ax, paths):
    """Each curve's visible label must identify its own observation uniquely."""
    labels = [line.get_label().replace("\\", "/") for line in ax.lines]
    assert len(labels) == len(paths) and len(set(labels)) == len(
        paths
    ), "Ambiguous spectrum identities"
    for label, path in zip(labels, paths):
        matches = [p for p in paths if label == p or label == Path(p).name]
        assert matches == [path], f"Legend {label!r} does not uniquely identify {path}"


def preparation_case(c, select):
    """Real classifier transforms, analytic values, and actual inference input."""
    from pipeline.classifier import CollinearityFilter
    from sklearn.feature_selection import SelectKBest
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import Pipeline

    train = pd.DataFrame(
        {
            "alpha": [1.0, 3.0, 5.0, 7.0, 9.0, 11.0],
            "alpha_copy": [2.0, 6.0, 10.0, 14.0, 18.0, 22.0],
            "gamma": [10.0, 40.0, 20.0, 60.0, 30.0, 50.0],
            "delta": [0.2, 0.9, 0.1, 0.7, 0.4, 0.3],
        }
    )
    cf = CollinearityFilter(0.0, 0.98).fit(train)
    assert cf.keep_columns_ == ["alpha", "gamma", "delta"]
    retained = cf.transform(train)
    prep = (
        c.module.SpectralClassifier()._build_preprocessor(retained, "SVM").fit(retained)
    )
    steps = [("colfilter", cf), ("prep", prep)]
    if select:
        selector = SelectKBest(lambda X, y: np.array([1.0, 4.0, 3.0]), k=2).fit(
            prep.transform(retained), np.arange(6)
        )
        steps.append(("fs", selector))
    names = ["gamma", "delta"] if select else ["alpha", "gamma", "delta"]
    transform = Pipeline(steps)
    model = LinearRegression().fit(transform.transform(train), np.arange(6.0))
    pipe = Pipeline(steps + [("clf", model)])
    query = pd.DataFrame(
        {
            "delta": [0.9, 0.1],
            "alpha_copy": [22.0, 2.0],
            "unused": [900.0, 901.0],
            "gamma": [60.0, np.nan],
            "alpha": [11.0, 1.0],
        }
    )
    expected = []
    for _, row in query.iterrows():
        out = []
        for name in names:
            values = train[name].tolist()
            mean = sum(values) / 6
            scale = (sum((v - mean) ** 2 for v in values) / 6) ** 0.5
            value = mean if pd.isna(row[name]) else row[name]
            out.append((value - mean) / scale)
        expected.append(out)
    observed = []
    real_predict = model.predict

    def predict(X):
        observed.append(np.asarray(X).copy())
        return real_predict(X)

    c.monkeypatch.setattr(model, "predict", predict)
    pipe.predict(query)
    assert len(observed) == 1
    assert_allclose(observed[0], expected, atol=2e-14)
    wrapper = SimpleNamespace(
        model_pipeline=pipe, feature_names_used=list(train.columns)
    )
    return wrapper, query, np.asarray(expected), names


def plain_text(value):
    return "".join(
        ch
        for ch in unicodedata.normalize("NFKD", value.casefold())
        if not unicodedata.combining(ch)
    )


def assert_population_percentages(ax, counts, top_n):
    """Require a numeric denominator tied to its population, not stray digits."""
    text = plain_text(
        " ".join(
            [ax.get_title(), ax.get_ylabel()]
            + [t.get_text() for t in ax.texts]
            + [t.get_text() for t in ax.figure.texts]
        )
    )
    total, shown = sum(counts), sum(counts[:top_n])

    def role(words):
        matches = re.findall(
            rf"\b(?:{words})\b\s*(?:catalogue|catalog|population|examples|observations|count|effectif)?\s*(?:\(n\s*)?[=:]?\s*(\d+)(?!\d)",
            text,
        )
        return [int(n) for n in matches]

    totals = role("total|catalogue entier")
    displayed = role("displayed|shown|affiche[es]*")
    excluded = role("excluded|omitted|exclu[es]*")
    pair = any(
        (int(a), int(b)) == (shown, total)
        for a, b in re.findall(
            r"(?<!\d)(\d+)\s*(?:of|sur|/)\s*(\d+)(?!\d)\s*(?:examples|observations|spectra|exemples)",
            text,
        )
    )
    values = [p.get_height() for p in ax.patches]
    assert len(values) == top_n
    total_values = np.asarray(counts[:top_n]) * 100 / total
    conditional_values = np.asarray(counts[:top_n]) * 100 / shown
    if np.allclose(values, total_values, rtol=0, atol=1e-10):
        assert total in totals and re.search(
            r"(?:denominator\s*[=:]\s*total|of (?:the )?total|du (?:catalogue )?total)",
            text,
        ), "Total-catalogue denominator is not explicitly identified"
    else:
        assert_allclose(values, conditional_values, rtol=0, atol=1e-10)
        assert re.search(
            r"condition|among (?:the )?displayed|parmi .*affich|denominator\s*[=:]\s*displayed",
            text,
        ), "Conditional denominator is not identified"
        assert pair or (
            total in totals and (shown in displayed or total - shown in excluded)
        ), "Displayed/total or excluded population is missing or has the wrong semantic role"


def assert_multiclass_importances(result):
    """Accept declared class mean, class sum, or class-specific sample means."""
    assert isinstance(result, pd.DataFrame) and set(result.feature) == {
        "alpha",
        "beta",
        "gamma",
    }, "One feature is lost after reducing the feature axis"
    names = ["alpha", "beta", "gamma"]
    class_means = np.array([[4.0, 25.0, 250.0], [6.0, 50.0, 500.0]])
    if "class" in result:
        assert set(result["class"]) == {"F", "G"}, "SHAP class identities are lost"
        for cls, expected in zip(["F", "G"], class_means):
            rows = result[result["class"] == cls].set_index("feature")
            assert len(rows) == 3
            assert_allclose(rows.loc[names, "mean_abs_shap"], expected)
        return
    assert len(result) == 3
    columns = [col for col in result if col != "feature"]
    assert len(columns) == 1, "Aggregated SHAP statistic is ambiguous"
    column = columns[0]
    mode = result.attrs.get("class_reduction")
    if mode is None:
        mode = (
            "sum"
            if column == "mean_sample_sum_class_abs_shap"
            else "mean" if column == "mean_abs_shap" else None
        )
    assert mode in {"sum", "mean"}, "Class aggregation is not declared"
    expected = class_means.sum(axis=0) if mode == "sum" else class_means.mean(axis=0)
    assert_allclose(result.set_index("feature").loc[names, column], expected)


def eda_partition_case(c, with_ids):
    """Observe estimator fit/evaluation rows, independently of any splitter API."""
    from sklearn.ensemble import RandomForestClassifier

    rng = np.random.default_rng(71)
    groups = np.repeat(
        [f"object-{n}" for n in [81, 4, 75, 13, 90, 8, 66, 23, 5, 42, 16, 39]], 2
    )
    table = pd.DataFrame(
        {
            "alpha": rng.permutation(24) + 0.125,
            "beta": rng.normal(size=24),
            "subclass": np.repeat(["F", "G"] * 6, 2),
        }
    )
    if with_ids:
        table["source_id"] = groups  # string identity, not a numeric predictor
    path = Path(c.paths["PROCESSED_DIR"]) / "features_repeated.csv"
    table.to_csv(path, index=False)
    identity = dict(zip(table.alpha, groups))
    fitted, evaluated = [], []
    real_fit, real_predict = RandomForestClassifier.fit, RandomForestClassifier.predict

    def row_groups(X):
        values = (
            X["alpha"].to_numpy()
            if isinstance(X, pd.DataFrame)
            else np.asarray(X)[:, 0]
        )
        assert all(
            float(v) in identity for v in values
        ), "Evaluation provenance adapter cannot identify transformed rows"
        return set(identity[float(v)] for v in values)

    def fit(model, X, y, *args, **kwargs):
        fitted.append(row_groups(X))
        return real_fit(model, X, y, *args, **kwargs)

    def predict(model, X, *args, **kwargs):
        evaluated.append(row_groups(X))
        return real_predict(model, X, *args, **kwargs)

    c.monkeypatch.setattr(RandomForestClassifier, "fit", fit)
    c.monkeypatch.setattr(RandomForestClassifier, "predict", predict)
    return path, fitted, evaluated


def assert_eda_partitions(fitted, evaluated, disclosure):
    text = plain_text(disclosure)
    explicit = re.search(
        r"not (?:an? )?independent holdout|no independent holdout|aucun holdout independant|pas (?:un |d.)?holdout independant",
        text,
    )
    if explicit:
        assert (
            "object" in text
            or "identit" in text
            or "repeat" in text
            or "descript" in text
        ), "Holdout refusal lacks its scientific reason"
        return
    assert (
        fitted and evaluated
    ), "No observable evaluation and no explicit holdout refusal"
    for training in fitted:
        for evaluation in evaluated:
            overlap = training & evaluation
            assert (
                not overlap
            ), f"Repeated object observations contaminate the holdout: {sorted(overlap)}"


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
