"""R1–R3 strengthened proofs; run explicitly from the new immutable archive.

Historical archives 0da7581/7b32abb remain unchanged. No production corrections.
"""

import numpy as np
from pandas.testing import assert_frame_equal
import pytest


@pytest.mark.parametrize("scenario", ["default", "nb01", "unequal"])
def test_actual_loader_population_is_prediction_identity_or_explicit_refusal(
    bridge_loader_population, bridge_review_contracts, scenario
):
    c = bridge_loader_population(scenario)
    result = bridge_review_contracts.invoke(c, "identity")
    if result is None:
        return
    expected = c.population[c.classifier.feature_names_used].reset_index(drop=True)
    received = c.spy.predict_inputs[-1]
    assert_frame_equal(received, expected, check_dtype=False, check_exact=True)
    ids = c.population.obsid.to_numpy()
    np.testing.assert_array_equal(result["Z_umap_aligned"][:, 0], ids)
    np.testing.assert_array_equal(result["y_aligned"], c.args["y"])
    np.testing.assert_array_equal(result["cl_aligned"], ids + 10000)
    np.testing.assert_array_equal(result["y_pred"], np.array(["G", "M", "F"])[ids % 3])


def test_shared_actual_loader_population_is_positive_control(bridge_loader_population):
    c = bridge_loader_population("shared")
    result = c.run()
    assert result["_xgb_ok"] is True
    assert_frame_equal(
        c.spy.predict_inputs[-1],
        c.population[c.classifier.feature_names_used].reset_index(drop=True),
        check_dtype=False,
        check_exact=True,
    )
    np.testing.assert_array_equal(result["Z_umap_aligned"][:, 0], c.population.obsid)
