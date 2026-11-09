"""Supplemental #90 shape proof; explicit collection only, no production change."""

import numpy as np


def test_three_dimensional_probability_output_requires_refusal(bridge_case):
    """Injected boundary probe: a per-object confidence must be a scalar."""
    c = bridge_case
    c.spy.proba_fn = lambda X: np.tile([[[0.2, 0.3, 0.5], [0.1, 0.2, 0.7]]], (5, 1, 1))
    try:
        result = c.run()
    except ValueError:
        return
    if result["_xgb_ok"] and result["has_proba"]:
        assert result["confidence"].shape == (
            5,
        ), "A probability tensor is not scalar confidence."
