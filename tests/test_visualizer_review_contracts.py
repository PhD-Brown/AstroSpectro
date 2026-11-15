"""Check the new review adapters against correct and misleading alternatives."""

import numpy as np
from numpy.testing import assert_allclose
import pytest


def test_analytic_estimator_protocol_is_probability_consistent(visualizer_contracts):
    oracle = visualizer_contracts.AnalyticGaussian(
        np.array([[1.0, 2, 3], [-1.0, 2, 4]])
    )
    assert (oracle.d, oracle.n, oracle.factor) == (2, 3, 1)
    assert_allclose(oracle.covariance, np.diag([4, 9]))
    points = np.array([[0.0, 2.0, 0.0], [0.0, 0.0, 3.0]])
    assert_allclose(
        oracle(points), np.array([1, np.exp(-0.5), np.exp(-0.5)]) / (12 * np.pi)
    )
    assert_allclose(oracle.evaluate(points), oracle(points))
    draws = oracle.resample(40000, seed=np.random.default_rng(11))
    assert_allclose(draws.mean(axis=1), [0, 0], atol=0.05)
    assert_allclose(np.cov(draws), np.diag([4, 9]), atol=0.12)


@pytest.mark.parametrize(
    "message,admitted",
    [
        ("T(k) < 0.75 hors champ : min = 0.500", True),
        ("t-SNE outside range below 0.75; minimum 0.500", True),
        ("Warning: clipped values", False),
        ("T(k) < 0.75 hors champ : min = 0.700", False),
    ],
)
def test_clipping_disclosure_requires_actual_bound_and_extreme(
    visualizer_case, visualizer_contracts, message, admitted
):
    _, ax = visualizer_case.plt.subplots()
    ax.set_ylim(0.75, 1.01)
    ax.text(0.5, 0.5, message)
    if admitted:
        visualizer_contracts.assert_visible_or_disclosed(
            ax, {"t-SNE": [0.5, 0.6], "UMAP": [1, 1]}
        )
    else:
        with pytest.raises(AssertionError, match="Undisclosed"):
            visualizer_contracts.assert_visible_or_disclosed(
                ax, {"t-SNE": [0.5, 0.6], "UMAP": [1, 1]}
            )


@pytest.mark.parametrize("swap", [False, True])
def test_categorical_legend_maps_ids_to_actual_data_colors(
    visualizer_case, visualizer_contracts, swap
):
    from matplotlib.patches import Patch

    c = visualizer_case
    fig, ax = c.plt.subplots()
    labels = np.array([2, 7, 2, 7, 2, 7])
    for cid, color in [(2, "red"), (7, "blue")]:
        ax.scatter(*c.Z[labels == cid].T, c=color)
    colors = ["blue", "red"] if swap else ["red", "blue"]
    ax.legend(
        handles=[
            Patch(color=col, label=f"Cluster {cid}") for cid, col in zip([2, 7], colors)
        ]
    )
    if swap:
        with pytest.raises(AssertionError, match="Misleading categorical"):
            visualizer_contracts.assert_cluster_encoding(fig, ax, labels, c.Z)
    else:
        visualizer_contracts.assert_cluster_encoding(fig, ax, labels, c.Z)


def test_histogram_disclosure_identifies_the_relevant_exclusion_bound(
    visualizer_case, visualizer_contracts
):
    _, ax = visualizer_case.plt.subplots()
    values = np.array([0.04, 0.09, 0.1, 4.0, 7.0])
    for message, admitted in [
        ("QSO: excluded 2 of 5 above 0.15", True),
        ("QSO: excluded 2 of 5 outside 0", False),
        ("QSO: excluded 2 of 50 above 0.15", False),
        ("Warning: QSO clipped", False),
    ]:
        for artist in list(ax.texts):
            artist.remove()
        ax.text(0.1, 0.1, message)
        assert (
            visualizer_contracts.exclusion_report(ax, "QSO", 2, 5, [0.0, 0.15], values)
            == admitted
        ), message


def test_localized_numbers_and_population_suffixes(visualizer_contracts):
    for text, expected in [
        ("0,000100", [0.0001]),
        ("43k étoiles", [43000]),
        ("≈ 1,5M objets", [1500000]),
        ("43 019", [43019]),
        ("1\u202f234\u202f567,25", [1234567.25]),
        ("1,234,567", [1234567]),
        ("1,5e-3", [0.0015]),
    ]:
        assert_allclose(visualizer_contracts.numbers(text), expected, err_msg=text)


def test_variance_claims_use_numbers_and_component_meaning(
    visualizer_case, visualizer_contracts
):
    _, ax = visualizer_case.plt.subplots()
    ratios = np.array([9.0, 4.0, 1.0]) / 14
    for name in ["PC", "PCs", "CP", "composantes"]:
        for count, percentage, admitted in [(2, "92,9", True), (3, "85", False)]:
            for t in list(ax.texts):
                t.remove()
            ax.text(0.1, 0.1, f"{count} {name} → {percentage} %")
            if admitted:
                visualizer_contracts.assert_variance_annotations(
                    ax, ratios, expected_count=2
                )
            else:
                with pytest.raises(AssertionError, match="variance disagree"):
                    visualizer_contracts.assert_variance_annotations(ax, ratios)


@pytest.mark.parametrize("orientation", ["vertical", "horizontal"])
def test_all_categorical_representations_must_agree(
    visualizer_case, visualizer_contracts, orientation
):
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.cm import ScalarMappable
    from matplotlib.patches import Patch

    c = visualizer_case
    fig, ax = c.plt.subplots()
    labels = np.array([2, 7, 2, 7, 2, 7])
    for cid, color in [(2, "red"), (7, "blue")]:
        ax.scatter(*c.Z[labels == cid].T, c=color)
        ax.text(cid, 0, f"C{cid}", bbox=dict(fc=color))
    ax.legend(handles=[Patch(color="red", label="C2"), Patch(color="blue", label="C7")])
    visualizer_contracts.assert_cluster_encoding(fig, ax, labels, c.Z)
    sm = ScalarMappable(
        norm=BoundaryNorm([1.5, 4.5, 7.5], 2), cmap=ListedColormap(["red", "blue"])
    )
    cb = fig.colorbar(sm, ax=ax, orientation=orientation, label="Groupe HDBSCAN")
    visualizer_contracts.assert_cluster_encoding(fig, ax, labels, c.Z)
    sm.set_cmap(ListedColormap(["blue", "red"]))
    with pytest.raises(AssertionError, match="Misleading colorbar"):
        visualizer_contracts.assert_cluster_encoding(fig, ax, labels, c.Z)
    sm.set_cmap(ListedColormap(["red", "blue"]))
    ax.legend(handles=[Patch(color="blue", label="C2"), Patch(color="red", label="C7")])
    with pytest.raises(AssertionError, match="Misleading categorical legend"):
        visualizer_contracts.assert_cluster_encoding(fig, ax, labels, c.Z)
    ax.legend(handles=[Patch(color="red", label="2"), Patch(color="blue", label="7")])
    global_legend = fig.legend(
        handles=[Patch(color="blue", label="C2"), Patch(color="red", label="C7")]
    )
    with pytest.raises(AssertionError, match="Misleading categorical legend"):
        visualizer_contracts.assert_cluster_encoding(fig, ax, labels, c.Z)
    global_legend.remove()
    cb.remove()


def test_histogram_denominators_and_claims_are_distinguishable(
    visualizer_case, visualizer_contracts, monkeypatch
):
    from matplotlib.patches import Patch

    c = visualizer_case
    qso = np.r_[[0.04, 0.09, 0.1], np.linspace(4, 7, 47)]
    edges = np.linspace(0, 0.15, 60)
    calls = visualizer_contracts.capture_histograms(monkeypatch)
    for mode, claim, admitted in [
        (
            "conditional",
            "QSO : 47 exclues sur 50, observations tronquées hors support 0,000000 à 0,150000",
            True,
        ),
        (
            "total",
            "QSO : densité normalisée par effectif total 50, support 0,000000 à 0,150000",
            True,
        ),
        (
            "conditional",
            "QSO : density normalized by total 50, support 0 to 0.15",
            False,
        ),
        ("total", "QSO : conditional density", False),
        ("conditional", "QSO : excluded 50 of 47 outside support 0 to 0.15", False),
        (
            "total",
            "QSO : density normalized by total 500, visible 50, support 0 to 0.15",
            False,
        ),
    ]:
        _, ax = c.plt.subplots()
        if mode == "total":
            ax.hist(
                qso,
                bins=edges,
                weights=np.full(50, 1 / (50 * np.diff(edges)[0])),
                label="QSO",
            )
        else:
            ax.hist(qso, bins=edges, density=True, label="QSO")
        ax.text(0.1, 0.1, claim)
        if admitted:
            convention, missing, _ = (
                visualizer_contracts.assert_histogram_representation(
                    calls[-1], qso, "QSO"
                )
            )
            assert convention == mode and missing == 47
            assert_allclose(
                sum(p.get_width() * p.get_height() for p in calls[-1]["result"][2]),
                3 / 50 if mode == "total" else 1,
            )
        else:
            with pytest.raises(AssertionError):
                visualizer_contracts.assert_histogram_representation(
                    calls[-1], qso, "QSO"
                )
    fig, ax = c.plt.subplots()
    ax.hist(
        qso,
        bins=edges,
        weights=np.full(50, 1 / (50 * np.diff(edges)[0])),
        label="QSO data",
    )
    correct = "QSO: density normalized by total 50, support 0 to 0.15"
    ax.legend(handles=[Patch(label=correct)])
    assert (
        visualizer_contracts.assert_histogram_representation(calls[-1], qso, "QSO")[0]
        == "total"
    )
    fig.legend(handles=[Patch(label="QSO: conditional density")])
    with pytest.raises(AssertionError, match="Contradictory histogram"):
        visualizer_contracts.assert_histogram_representation(calls[-1], qso, "QSO")
