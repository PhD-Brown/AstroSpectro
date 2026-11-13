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
