"""Bounded numeric/artist oracles for the M2.8 review, not production helpers."""

from io import BytesIO
import re
import unicodedata

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal


def tokens(label):
    folded = "".join(
        c
        for c in unicodedata.normalize("NFKD", label.casefold())
        if not unicodedata.combining(c)
    )
    return set(re.findall(r"[\w]+", folded))


def numbers(label):
    """Read decimal dot/comma, spaced thousands, scientific notation and k/M.

    A single comma is decimal (French convention); unambiguous repeated comma
    thousands are supported. A lone ``1,234`` is inherently locale ambiguous.
    """
    label = re.sub(r"\b\d{1,3}(?:,\d{3}){2,}\b", lambda m: m[0].replace(",", ""), label)
    normalized = re.sub(r"(?<=\d)[ \u202f\u00a0](?=\d{3}(?:\D|$))", "", label)
    return [
        float(n.replace(",", ".")) * ({"k": 1e3, "K": 1e3, "M": 1e6}.get(suffix, 1))
        for n, suffix in re.findall(
            r"([-+]?\d+(?:[.,]\d+)?(?:[eE][-+]?\d+)?)\s*([kKM]?)(?!\w)", normalized
        )
    ]


def assert_variance_annotations(ax, ratios, expected_count=None):
    """Link every variance claim to its component count, independent of prose."""
    seen = []
    component = r"(?:PCs?|CPs?|composantes?|components?)"
    for annotation in ax.texts:
        label = annotation.get_text()
        percentages = re.findall(r"(\d+(?:[.,]\d+)?)\s*%", label)
        counts = re.findall(
            rf"(\d+)\s*{component}\b|\b{component}\s*[:=]?\s*(\d+)", label, re.I
        )
        counts = [int(a or b) for a, b in counts]
        if percentages and not counts:
            assert hasattr(
                annotation, "xy"
            ), "Variance claim needs a count or annotated coordinate"
            counts = [int(annotation.xy[0])]
        for count in counts:
            assert 1 <= count <= len(ratios), "Invalid component count"
            for percentage in percentages:
                assert_allclose(
                    numbers(percentage)[0],
                    sum(ratios[:count]) * 100,
                    atol=0.05,
                    rtol=0,
                    err_msg="Annotated component count and variance disagree",
                )
                seen.append(count)
    if expected_count is not None:
        assert expected_count in seen, "Expected variance/component relation is missing"


def visible_legends(fig):
    """Read displayed Legend artists, including figure and additional legends."""
    from matplotlib.legend import Legend

    return [legend for legend in fig.findobj(match=Legend) if legend.get_visible()]


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


def assert_histogram_population(call, expected, normalization="conditional"):
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
    if normalization == "total":
        reference /= len(expected) * np.diff(edges)
    elif call["density"]:
        reference /= sum(wanted) * np.diff(edges)
    assert_allclose(plotted, reference, atol=1e-10, rtol=1e-10)
    missing = len(expected) - sum(wanted)
    return missing, edges


def assert_histogram_representation(call, expected, cls):
    """Admit conditional density with disclosure or identified total density.

    Proposed R10 test policy, subject to scientific review. Heights come from
    independently counted observations; text cannot change the denominator.
    """
    ax = call["ax"]
    texts = [t.get_text() for t in [*ax.texts, *ax.figure.texts] if t.get_visible()]
    texts += [
        t.get_text()
        for legend in visible_legends(ax.figure)
        for t in legend.texts
        if t.get_visible()
    ]
    relevant = [s for s in texts if cls.casefold() in tokens(s)]
    norm_words = {
        "normalized",
        "normalised",
        "normalisee",
        "normalise",
        "normalisation",
        "normalization",
    }
    total_claims = [
        s
        for s in relevant
        if tokens(s).intersection(norm_words) and "total" in tokens(s)
    ]
    convention = "total" if total_claims else "conditional"
    missing, edges = assert_histogram_population(call, expected, convention)
    if convention == "total":
        for label in total_claims:
            ns = numbers(label)
            declared = re.search(
                r"\btotal\b\D*?(\d+(?:[.,]\d+)?\s*[kKM]?)", label, re.I
            )
            assert declared and numbers(declared[1]) == [
                len(expected)
            ], "Wrong total-normalization reference population"
            assert tokens(label + " " + ax.get_ylabel()).intersection(
                {"density", "densite"}
            ), "Density scale must be identified"
            assert all(
                any(abs(v - b) <= max(5e-7, abs(b) * 0.005) for v in ns)
                for b in [edges[0], edges[-1]]
            ), "Total density needs its actual support"
        assert not any(
            tokens(s).intersection({"conditional", "conditionnelle"}) for s in relevant
        ), "Contradictory histogram normalization claims"
    else:
        assert missing == 0 or exclusion_report(
            ax, cls, missing, len(expected), edges, expected
        ), "Undisclosed conditional-density exclusion"
    return convention, missing, edges


def exclusion_report(ax, cls, missing, total, edges, values=None):
    """Require population/count and the bound actually crossed by excluded data."""
    bounds = [edges[0], edges[-1]]
    if values is not None:
        values = np.asarray(values)
        bounds = ([edges[0]] if np.any(values < edges[0]) else []) + (
            [edges[-1]] if np.any(values > edges[-1]) else []
        )
    for artist in ax.texts:
        label = artist.get_text()
        if not tokens(label).intersection(
            {
                "outside",
                "excluded",
                "hors",
                "exclus",
                "exclues",
                "tronque",
                "tronquee",
                "tronques",
                "tronquees",
            }
        ):
            continue
        vals = numbers(label)
        number = r"\d+(?:[ \u202f\u00a0]\d{3})*(?:[.,]\d+)?\s*[kKM]?"
        reports = []
        for pattern in [
            rf"(?:excluded|exclu\w*|outside)\s*({number})\s*(?:of|sur|out of)\s*({number})",
            rf"({number})\s*(?:observations?\s+)?(?:exclu\w*|tronqu\w*)\s*(?:of|sur)\s*({number})",
        ]:
            reports.extend(re.findall(pattern, label, re.I))
        exact_counts = any(
            numbers(a) == [missing] and numbers(b) == [total] for a, b in reports
        )
        if cls.casefold() in tokens(label) and exact_counts:
            if bounds and all(
                any(abs(v - bound) <= max(5e-7, abs(bound) * 0.005) for v in vals)
                for bound in bounds
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
                {
                    "outside",
                    "excluded",
                    "hors",
                    "exclus",
                    "clipped",
                    "tronque",
                    "tronquee",
                    "tronques",
                    "tronquees",
                }
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

    entries = [
        (handle, t.get_text())
        for legend in visible_legends(fig)
        for handle, t in zip(legend.legend_handles, legend.texts)
        if t.get_visible()
    ]
    colorbars = [
        a._colorbar
        for a in fig.axes
        if a.get_visible() and getattr(a, "_colorbar", None) is not None
    ]
    cluster_collections = {}
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
        cluster_collections[cid] = collection
    # A physical colorbar is bound to an actual data-bearing collection.
    # Standalone ScalarMappables cannot acquire that role through label wording.
    physical = [
        sc
        for a in fig.axes
        for sc in a.collections
        if sc not in cluster_collections.values()
        and sc.get_array() is not None
        and np.asarray(sc.get_array()).size
    ]
    categorical_bars = [cb for cb in colorbars if cb.mappable not in physical]
    for cid, collection in cluster_collections.items():
        rgb = collection.get_facecolors()[0, :3]
        choices = [
            ("colorbar", cb.mappable.to_rgba(cid)[:3]) for cb in categorical_bars
        ]
        for handle, label in entries:
            if re.match(rf"^(?:(?:C|cluster|groupe)\s*)?{cid}\b", label.strip(), re.I):
                if hasattr(handle, "get_facecolor"):
                    choices.append(
                        (
                            "categorical legend",
                            np.asarray(handle.get_facecolor()).reshape(-1, 4)[0, :3],
                        )
                    )
                elif hasattr(handle, "get_color"):
                    choices.append(
                        (
                            "categorical legend",
                            np.asarray(to_rgba(handle.get_color()))[:3],
                        )
                    )
        for annotation in [
            t for a in fig.axes if a.get_visible() for t in a.texts
        ] + fig.texts:
            if not annotation.get_visible():
                continue
            if re.match(
                rf"^(?:(?:C|cluster|groupe)\s*)?{cid}\b",
                annotation.get_text().strip(),
                re.I,
            ):
                bbox = annotation.get_bbox_patch()
                choices.append(
                    (
                        "annotation",
                        np.asarray(
                            bbox.get_facecolor()
                            if bbox is not None
                            else to_rgba(annotation.get_color())
                        )[:3],
                    )
                )
        assert choices, f"No faithful ID/color encoding for C{cid}"
        for kind, value in choices:
            assert np.allclose(value, rgb), f"Misleading {kind} for C{cid}"
