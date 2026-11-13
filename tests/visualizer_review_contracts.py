"""Bounded numeric/artist oracles for the M2.8 review, not production helpers."""

from io import BytesIO
import re

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal


def tokens(label):
    return set(re.findall(r"[\w]+", label.casefold()))


def numbers(label):
    normalized = re.sub(r"(?<=\d)[ \u202f\u00a0](?=\d{3}(?:\D|$))", "", label)
    return [
        float(n)
        for n in re.findall(r"[-+]?\d+(?:[.,]\d+)?(?:e[-+]?\d+)?", normalized, re.I)
    ]


def assert_class_label(label, cls, count):
    assert cls.casefold() in tokens(label), (label, cls)
    assert float(count) in numbers(label), (label, count)


def assert_png_matches_figure(path, fig, dpi):
    """Compare decoded pixels with this same figure's explicit backend export.

    This is a within-process semantic reference, not a cross-version PNG hash.
    Axes-only clones differ from the reference even though both have ink.
    """
    from PIL import Image

    reference = BytesIO()
    fig.savefig(reference, format="png", bbox_inches="tight", dpi=dpi)
    reference.seek(0)
    with Image.open(reference) as image:
        expected = np.asarray(image.convert("RGBA"))
    with Image.open(path) as image:
        actual = np.asarray(image.convert("RGBA"))
    assert (
        actual.shape == expected.shape
    ), "Export geometry differs from the returned figure"
    assert_allclose(
        actual,
        expected,
        atol=1,
        rtol=0,
        err_msg="Exported scientific artists differ from the returned figure",
    )


class AnalyticGaussian:
    """Estimator-protocol oracle for exactly N(0, diag(4,9)).

    ``dataset`` records supplied observations for population auditing. The
    represented law is one analytic Gaussian, not an empirical kernel mixture.
    ``covariance`` and ``factor=1`` describe that Gaussian; resample and evaluate
    use the same law. This supports grid- and simulation-based level solvers.
    """

    def __init__(self, dataset, bw_method=None, weights=None):
        self.dataset = np.asarray(dataset).copy()
        self.d, self.n = self.dataset.shape
        self.factor = 1.0
        self.covariance = np.diag([4.0, 9.0])
        self.inv_cov = np.diag([0.25, 1 / 9])
        self.weights = np.full(self.n, 1 / self.n)
        self.neff = float(self.n)

    def evaluate(self, points):
        points = np.atleast_2d(points)
        exponent = (points[0] / 2) ** 2 + (points[1] / 3) ** 2
        return np.exp(-exponent / 2) / (12 * np.pi)

    __call__ = evaluate
    pdf = evaluate

    def resample(self, size=None, seed=None):
        size = self.n if size is None else size
        rng = (
            seed
            if isinstance(seed, (np.random.Generator, np.random.RandomState))
            else np.random.default_rng(seed)
        )
        return rng.normal(size=(2, size)) * np.array([[2.0], [3.0]])


def capture_histograms(monkeypatch):
    """Forward real hist calls and retain their effective populations and artists."""
    from matplotlib.axes import Axes

    original = Axes.hist
    calls = []

    def hist(ax, values, *args, **kwargs):
        result = original(ax, values, *args, **kwargs)
        calls.append(
            dict(
                ax=ax,
                values=np.asarray(values).copy(),
                result=result,
                density=kwargs.get("density", False),
                label=kwargs.get("label", ""),
            )
        )
        return result

    monkeypatch.setattr(Axes, "hist", hist)
    return calls


def assert_histogram_population(call, expected):
    """Count samples independently, including the closed final bin endpoint."""
    expected = np.asarray(expected)
    _, edges, patches = call["result"]
    actual_values = call["values"]
    wanted = [
        sum((v >= a) and (v < b or (i == len(edges) - 2 and v == b)) for v in expected)
        for i, (a, b) in enumerate(zip(edges[:-1], edges[1:]))
    ]
    received = [
        sum(
            (v >= a) and (v < b or (i == len(edges) - 2 and v == b))
            for v in actual_values
        )
        for i, (a, b) in enumerate(zip(edges[:-1], edges[1:]))
    ]
    assert_array_equal(
        received,
        wanted,
        "Histogram omitted observations inside its own displayed support",
    )
    assert sum(wanted) > 0, "A nominal distribution must contain observations"
    plotted = np.array([p.get_height() for p in patches])
    reference = np.array(wanted, dtype=float)
    if call["density"]:
        reference /= sum(wanted) * np.diff(edges)
    assert_allclose(plotted, reference, atol=1e-10, rtol=1e-10)
    missing = len(expected) - sum(wanted)
    return missing, edges


def exclusion_report(ax, cls, missing, total, edges):
    """Bounded disclosure: population, excluded/total counts and support bound."""
    for artist in ax.texts:
        label = artist.get_text()
        if not tokens(label).intersection(
            {"outside", "excluded", "hors", "exclus", "exclues", "tronque"}
        ):
            continue
        vals = numbers(label)
        if cls.casefold() in tokens(label) and missing in vals and total in vals:
            if any(
                any(abs(v - bound) <= max(5e-4, abs(bound) * 0.005) for v in vals)
                for bound in [edges[0], edges[-1]]
            ):
                return True
    return False


def assert_visible_or_disclosed(ax, series):
    """Admit a clipped T(k) curve only with exact bound/extreme information.

    A single clipped curve can be identified by context. Several clipped curves
    require separate method identification. Generic warnings are insufficient.
    """
    lo, hi = sorted(ax.get_ylim())
    clipped = [
        (name, np.asarray(vals)[(np.asarray(vals) < lo) | (np.asarray(vals) > hi)])
        for name, vals in series.items()
    ]
    clipped = [(name, vals) for name, vals in clipped if len(vals)]
    for name, vals in clipped:
        admitted = False
        for artist in ax.texts:
            label = artist.get_text()
            ns = numbers(label)
            if not tokens(label).intersection(
                {"outside", "excluded", "hors", "exclus", "clipped"}
            ):
                continue
            if len(clipped) > 1 and name.casefold() not in label.casefold():
                continue
            bounds = (
                [lo] if np.all(vals < lo) else [hi] if np.all(vals > hi) else [lo, hi]
            )
            extremes = (
                [min(vals)]
                if np.all(vals < lo)
                else [max(vals)] if np.all(vals > hi) else [min(vals), max(vals)]
            )
            if all(any(abs(n - v) <= 5e-4 for n in ns) for v in bounds + extremes):
                admitted = True
        assert admitted, f"Undisclosed clipped values for {name}: {vals}, range={lo,hi}"


def assert_cluster_encoding(fig, ax, labels, coordinates):
    """Accept colorbar, legend handles, or colored textual cluster annotations."""
    from matplotlib.colors import to_rgba

    legend = ax.get_legend()
    entries = (
        []
        if legend is None
        else list(zip(legend.legend_handles, [t.get_text() for t in legend.texts]))
    )
    colorbars = [
        a._colorbar for a in fig.axes if getattr(a, "_colorbar", None) is not None
    ]
    for cid in sorted(set(labels) - {-1}):
        expected = coordinates[labels == cid]
        collection = next(
            (
                sc
                for sc in ax.collections
                if np.array_equal(np.asarray(sc.get_offsets()), expected)
            ),
            None,
        )
        assert collection is not None, f"Missing actual cluster {cid}"
        rgb = collection.get_facecolors()[0, :3]
        choices = []
        for cb in colorbars:
            # Temperature colorbars are not categorical-ID encodings.
            if re.search(r"cluster|\bid\b", cb.ax.get_ylabel(), re.I):
                choices.append(cb.mappable.to_rgba(cid)[:3])
        if choices:
            assert all(
                np.allclose(choice, rgb) for choice in choices
            ), f"Misleading colorbar for C{cid}"
            continue
        for handle, label in entries:
            if re.search(rf"\bC\s*{cid}\b|cluster\s*{cid}\b", label, re.I):
                if hasattr(handle, "get_facecolor"):
                    choices.append(
                        np.asarray(handle.get_facecolor()).reshape(-1, 4)[0, :3]
                    )
                elif hasattr(handle, "get_color"):
                    choices.append(np.asarray(to_rgba(handle.get_color()))[:3])
        if choices:
            assert all(
                np.allclose(choice, rgb) for choice in choices
            ), f"Misleading categorical legend for C{cid}"
            continue
        for annotation in ax.texts:
            if re.fullmatch(rf"C\s*{cid}", annotation.get_text().strip(), re.I):
                bbox = annotation.get_bbox_patch()
                choices.append(
                    np.asarray(
                        bbox.get_facecolor()
                        if bbox is not None
                        else to_rgba(annotation.get_color())
                    )[:3]
                )
        assert any(
            np.allclose(choice, rgb) for choice in choices
        ), f"No faithful ID/color encoding for C{cid}"
