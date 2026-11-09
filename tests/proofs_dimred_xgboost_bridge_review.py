"""R1–R3 strengthened proofs; run explicitly from the new immutable archive.

Historical archives 0da7581/7b32abb remain unchanged. No production corrections.
"""

import numpy as np
import pandas as pd
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


@pytest.mark.parametrize("column", ["feature_beta", "feature_gamma"])
def test_fitted_train_median_reaches_booster_without_inference_refit(
    bridge_preprocessing_oracle, bridge_review_contracts, column
):
    observation = bridge_preprocessing_oracle
    c = observation.case
    query = c.query.copy()
    query.loc[2, column] = np.nan
    query.to_csv(c.csv, index=False)
    expected = observation.expected(query)
    position = c.classifier.feature_names_used.index(column)
    assert observation.medians[position] != observation.means[position]
    assert (
        expected[2, position] != 0.0
    ), "This oracle must distinguish median from mean/post-scale zero."
    zero_expected = observation.expected(query.fillna(0))
    inference_median_expected = observation.expected(
        query.fillna(query.median(numeric_only=True))
    )
    assert expected[2, position] != zero_expected[2, position]
    assert expected[2, position] != inference_median_expected[2, position]
    observation.inputs.clear()
    result = bridge_review_contracts.invoke(c, "preprocessing")
    if result is None:
        return
    assert len(observation.inputs) == 1
    observation.compare(observation.inputs[0], expected)
    np.testing.assert_array_equal(result["Z_umap_aligned"][:, 0], query.obsid)


@pytest.mark.parametrize("batch_size", [5, 10])
def test_same_missing_object_keeps_correct_preprocessing_and_row_identity(
    bridge_preprocessing_oracle, bridge_review_contracts, batch_size
):
    observation = bridge_preprocessing_oracle
    c = observation.case
    query = c.query.copy()
    query.loc[2, "feature_alpha"] = np.nan
    if batch_size == 10:
        additional = c.train.iloc[[2, 12, 22, 5, 18]].copy().reset_index(drop=True)
        additional["obsid"] = [6001, 6002, 6003, 6004, 6005]
        query = pd.concat([query, additional], ignore_index=True)
    query.to_csv(c.csv, index=False)
    c.args.update(
        Z_umap=np.column_stack((query.obsid, np.arange(batch_size))),
        y=np.array([f"obj{x}" for x in query.obsid]),
        cluster_labels=query.obsid.to_numpy() + 10000,
    )
    observation.inputs.clear()
    # A preprocessing/population incompatibility may be refused explicitly.
    result = bridge_review_contracts.invoke(
        c, "identity" if batch_size == 10 else "preprocessing"
    )
    if result is None:
        return
    ids = np.asarray(result["Z_umap_aligned"][:, 0], dtype=int)
    assert len(set(ids)) == len(ids)
    assert set(query.loc[query.feature_alpha.notna(), "obsid"]).issubset(set(ids))
    assert set(ids).issubset(set(query.obsid))
    selected = query.set_index("obsid").loc[ids].reset_index()
    observation.compare(observation.inputs[0], observation.expected(selected))
    np.testing.assert_array_equal(result["y_aligned"], [f"obj{x}" for x in ids])
    np.testing.assert_array_equal(result["cl_aligned"], ids + 10000)


@pytest.mark.parametrize("field", ["y", "cluster_labels"])
@pytest.mark.parametrize("size", [3, 7])
def test_incompatible_vectors_require_refusal_without_silent_truncation(
    bridge_case, bridge_review_contracts, field, size
):
    c = bridge_case
    supplied = np.arange(size) + 1200
    result = bridge_review_contracts.invoke(c, "identity", **{field: supplied})
    if result is None:
        return
    for key in ("y_pred", "confidence", "y_aligned", "cl_aligned", "Z_umap_aligned"):
        assert len(result[key]) == 5
    expected_y = supplied if field == "y" else c.args["y"]
    expected_cl = supplied if field == "cluster_labels" else c.args["cluster_labels"]
    np.testing.assert_array_equal(result["y_aligned"], expected_y)
    np.testing.assert_array_equal(result["cl_aligned"], expected_cl)


def test_real_undefined_probabilities_cannot_become_invented_available_confidence(
    bridge_real_bundle, bridge_review_contracts
):
    from sklearn.naive_bayes import GaussianNB

    c = bridge_real_bundle
    names = c.classifier.feature_names_used
    train = pd.DataFrame({name: [1.0, 1.0, 2.0, 2.0, 3.0, 3.0] for name in names})
    model = GaussianNB(var_smoothing=0).fit(train, [0, 0, 1, 1, 2, 2])
    assert np.all(model.var_ == 0)
    c.classifier.model_pipeline = model
    c.classifier.save_model(str(c.model))
    with np.errstate(divide="ignore", invalid="ignore"):
        assert np.isnan(model.predict_proba(c.query[names])).all()
        result = bridge_review_contracts.invoke(c, "probability")
    bridge_review_contracts.unavailable(result, 5)


def test_tensor_probabilities_require_refusal_not_flattening(
    bridge_case, bridge_review_contracts
):
    c = bridge_case
    c.spy.proba_fn = lambda X: np.tile([[[0.2, 0.3, 0.5], [0.1, 0.2, 0.7]]], (5, 1, 1))
    bridge_review_contracts.unavailable(
        bridge_review_contracts.invoke(c, "probability"), 5
    )


def test_valid_vectors_and_probabilities_are_positive_control(bridge_case):
    result = bridge_case.run()
    assert result["_xgb_ok"] is True and result["has_proba"] is True
    assert result["confidence"].shape == (5,)
    for key in ("y_aligned", "cl_aligned", "Z_umap_aligned"):
        np.testing.assert_array_equal(
            result[key],
            bridge_case.args[
                {
                    "y_aligned": "y",
                    "cl_aligned": "cluster_labels",
                    "Z_umap_aligned": "Z_umap",
                }[key]
            ],
        )


def test_complete_finite_preprocessing_is_positive_control(bridge_preprocessing_oracle):
    observation = bridge_preprocessing_oracle
    c = observation.case
    observation.inputs.clear()
    assert c.run()["_xgb_ok"]
    observation.compare(observation.inputs[0], observation.expected(c.query))
