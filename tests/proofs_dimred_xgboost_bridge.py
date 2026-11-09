"""Explicit M2.6 red proofs and observations; excluded from normal collection.

Run this file explicitly at its archived SHA. No xfail and no production edits.
Each proof allows an informative refusal where no unique remedy is justified.
"""

import re

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
import pytest


def run_or_refuse(case, **kwargs):
    """Accept explicit relevant input/output validation or structured failure."""
    try:
        result = case.run(**kwargs)
    except ValueError as exc:
        assert re.search(
            r"align|length|row|sample|observ|feature|schema|prob|finite|predict|shape|column|empty",
            str(exc),
            re.I,
        ), str(exc)
        return None
    return result if result["_xgb_ok"] else None


@pytest.mark.parametrize("filter_kind", ["snr", "nan", "multiple"])
def test_filter_cannot_reassign_predictions_to_other_observations(
    bridge_case, filter_kind
):
    """NEW identity defect: internally deleted rows cannot be repaired by prefix slicing."""
    c = bridge_case
    ids = np.array([104, 19, 807, 33, 250, 608, 47, 903, 1102, 2])
    alpha = np.array([3, 10, 8, 23, 4, 6, 12, 17, 26, 40])
    df = pd.DataFrame(
        {
            "obsid": ids,
            "feature_alpha": alpha,
            "feature_beta": [101, 31, 89, 17, 53, 8, 97, 13, 79, 41],
            "feature_gamma": np.arange(10) + 1.0,
            "snr_r": 20.0,
        }
    )
    removed = [4]
    if filter_kind == "nan":
        df.loc[4, "feature_gamma"] = np.nan  # Exactly1/10: selected then dropped.
    else:
        removed = [1, 4, 7] if filter_kind == "multiple" else [4]
        df.loc[removed, "snr_r"] = 9.0
    df.to_csv(c.csv, index=False)
    result = run_or_refuse(
        c,
        Z_umap=np.column_stack((ids, -ids)),
        y=np.array([f"true{x}" for x in ids]),
        cluster_labels=ids + 100,
    )
    if result is None:
        return
    retained = np.delete(np.arange(10), removed)
    # First prove which actual observations reached the model.
    assert_frame_equal(
        c.spy.predict_inputs[0],
        df.iloc[retained][c.classifier.feature_names_used].reset_index(drop=True),
    )
    np.testing.assert_array_equal(result["Z_umap_aligned"][:, 0], ids[retained])
    np.testing.assert_array_equal(
        result["y_aligned"], [f"true{x}" for x in ids[retained]]
    )
    np.testing.assert_array_equal(result["cl_aligned"], ids[retained] + 100)


@pytest.mark.parametrize("field", ["y", "cluster_labels"])
@pytest.mark.parametrize("size", [3, 7])
def test_each_input_cardinality_checked_independently(bridge_case, field, size):
    """NEW identity defect: features=Z=5, exactly one independent vector differs."""
    c = bridge_case
    result = run_or_refuse(c, **{field: np.arange(size)})
    if result is not None:
        key = "y_aligned" if field == "y" else "cl_aligned"
        assert len(result[key]) == len(result["y_pred"]) == 5


@pytest.mark.parametrize("size", [0, 3, 7])
def test_prediction_cardinality_cannot_report_success(bridge_case, size):
    """Boundary robustness: injected estimator output; real backend case below."""
    c = bridge_case
    c.spy.predict_fn = lambda X: np.zeros(size, dtype=int)
    result = run_or_refuse(c)
    if result is not None:
        assert (
            len(result["y_pred"])
            == len(result["confidence"])
            == len(result["y_aligned"])
            == 5
        )


@pytest.mark.parametrize(
    "kind", ["short", "long", "nan", "inf", "negative", "above_one", "unnormalized"]
)
def test_invalid_probability_output_cannot_be_advertised_as_valid(bridge_case, kind):
    """Boundary robustness; does not reject a legitimate threshold decision rule."""
    c = bridge_case
    probabilities = np.tile([0.2, 0.3, 0.5], (5, 1))
    if kind == "short":
        probabilities = probabilities[:3]
    elif kind == "long":
        probabilities = np.tile([0.2, 0.3, 0.5], (7, 1))
    else:
        probabilities[2] = {
            "nan": [np.nan, 0.3, 0.7],
            "inf": [np.inf, 0.3, 0.7],
            "negative": [-0.2, 0.4, 0.8],
            "above_one": [1.2, -0.4, 0.2],
            "unnormalized": [0.2, 0.2, 0.2],
        }[kind]
    c.spy.proba_fn = lambda X: probabilities
    result = run_or_refuse(c)
    if result is not None:
        # Invalid probabilities may be refused or explicitly marked unavailable.
        assert result["has_proba"] is False


@pytest.mark.parametrize("missing_kind", ["present_nan", "absent_column"])
def test_real_fitted_imputer_is_not_silently_overridden(
    bridge_real_bundle, missing_kind
):
    """NEW preparation defect: trained median and zero give different real predictions."""
    c = bridge_real_bundle
    names = c.classifier.feature_names_used
    query = c.query.copy()
    query.loc[2, "feature_alpha"] = np.nan
    # A fully missing column requires <=30% missing names; add one legitimate
    # trained column by duplicating neither values nor roles in this experiment.
    if missing_kind == "absent_column":
        # Three expected names =>1/3 is refused by the existing policy (control).
        query = query.drop(columns="feature_alpha")
        query.to_csv(c.csv, index=False)
        assert c.run()["_xgb_ok"] is False
        return
    query.to_csv(c.csv, index=False)
    expected_codes = c.classifier.model_pipeline.predict(query[names])
    expected = c.classifier.label_encoder.inverse_transform(expected_codes)
    zero_query = query[names].fillna(0)
    zero_codes = c.classifier.model_pipeline.predict(zero_query)
    assert (
        expected_codes[2] != zero_codes[2]
    ), "The fixture must discriminate median from zero."
    result = run_or_refuse(c)
    if result is not None:
        np.testing.assert_array_equal(result["y_pred"], expected)


def test_real_degenerate_gaussian_probabilities_cannot_claim_success(
    bridge_real_bundle,
):
    """NEW boundary defect exercised with a real supported sklearn classifier."""
    from sklearn.naive_bayes import GaussianNB

    c = bridge_real_bundle
    names = c.classifier.feature_names_used
    train = pd.DataFrame({name: [1.0, 1.0, 2.0, 2.0, 3.0, 3.0] for name in names})
    model = GaussianNB(var_smoothing=0).fit(train, [0, 0, 1, 1, 2, 2])
    c.classifier.model_pipeline = model
    c.classifier.save_model(str(c.model))
    # Zero within-class variance produces undefined likelihoods. Warnings are
    # local to this adversarial experiment, not suppressed for the green suite.
    with np.errstate(divide="ignore", invalid="ignore"):
        result = run_or_refuse(c)
    if result is not None:
        assert result["has_proba"] is False or np.isfinite(result["confidence"]).all()


@pytest.mark.parametrize(
    "kind",
    [
        "csv_permutation",
        "embedding_permutation",
        "labels_permutation",
        "replacement_ids",
        "duplicate_ids",
        "absent_ids",
    ],
)
def test_identity_information_missing_from_api_is_observation(bridge_case, kind):
    """OBSERVATION, not detection requirement: equal size cannot establish identity."""
    c = bridge_case
    df = c.frame.copy()
    overrides = {}
    if kind == "csv_permutation":
        df = df.iloc[[4, 0, 2, 1, 3]]
    elif kind == "embedding_permutation":
        overrides["Z_umap"] = c.args["Z_umap"][[4, 0, 2, 1, 3]]
    elif kind == "labels_permutation":
        overrides["y"] = c.args["y"][[4, 0, 2, 1, 3]]
    elif kind == "replacement_ids":
        df["obsid"] = [4, 6, 8, 10, 12]
    elif kind == "duplicate_ids":
        df["obsid"] = [104] * 5
    else:
        df = df.drop(columns="obsid")
    df.to_csv(c.csv, index=False)
    result = c.run(**overrides)
    assert result["_xgb_ok"] is True
    assert len(result["y_pred"]) == 5
    # Raw values are still preserved; the bridge does not inspect IDs.
    assert_frame_equal(
        c.spy.predict_inputs[0],
        df[c.classifier.feature_names_used].reset_index(drop=True),
    )


@pytest.mark.parametrize("kind", ["nan", "inf", "negative_inf", "constant_nan"])
def test_nonfinite_preparation_is_characterization(bridge_case, kind):
    """OBSERVATION: distinguish fillna zero from unchanged infinities."""
    c = bridge_case
    df = c.frame.copy()
    df["feature_beta"] = df["feature_beta"].astype(float)
    if kind == "constant_nan":
        df["feature_beta"] = [2.0, 2.0, np.nan, 2.0, 2.0]
    else:
        df.loc[2, "feature_beta"] = {
            "nan": np.nan,
            "inf": np.inf,
            "negative_inf": -np.inf,
        }[kind]
    df.to_csv(c.csv, index=False)
    c.run()
    expected = df[c.classifier.feature_names_used].fillna(0)
    assert_frame_equal(c.spy.predict_inputs[0], expected)


@pytest.mark.parametrize(
    "kind",
    [
        "empty_names",
        "duplicate_names",
        "duplicate_csv",
        "incompatible",
        "zero_rows",
        "one_row",
        "nonconsecutive_index",
    ],
)
def test_degenerate_schema_and_population_observations(bridge_case, monkeypatch, kind):
    """OBSERVATIONS of current acceptance/refusal; not a future schema policy."""
    c = bridge_case
    df = c.frame.copy()
    if kind == "empty_names":
        c.classifier.feature_names_used = []
        c.spy.predict_fn = lambda X: np.zeros(len(X), dtype=int)
        c.spy.proba_fn = lambda X: np.tile([0.2, 0.3, 0.5], (len(X), 1))
    elif kind == "duplicate_names":
        c.classifier.feature_names_used = [
            "feature_alpha",
            "feature_beta",
            "feature_beta",
        ]
    elif kind == "duplicate_csv":
        # Actual CSV parser mangles duplicate names; values under beta/beta.1 differ.
        c.csv.write_text(
            "feature_alpha,feature_beta,feature_beta\n3,101,2\n10,31,9\n8,89,4\n23,17,6\n4,53,8\n",
            encoding="utf-8",
        )
        c.classifier.feature_names_used = ["feature_alpha", "feature_beta"]
    elif kind == "incompatible":
        df["feature_beta"] = ["bad", "values", "reach", "the", "model"]
    elif kind == "zero_rows":
        df = df.iloc[:0]
    elif kind == "one_row":
        df = df.iloc[:1]
    else:
        df.index = [104, 19, 807, 33, 250]
        monkeypatch.setattr(c.module.pd, "read_csv", lambda *a, **kw: df)
    if kind != "duplicate_csv":
        df.to_csv(c.csv, index=False)
    result = c.run()
    assert result["_xgb_ok"]
    received = c.spy.predict_inputs[0]
    assert list(received.columns) == c.classifier.feature_names_used
    assert len(received) == (
        0 if kind == "zero_rows" else 1 if kind == "one_row" else 5
    )
    if kind == "nonconsecutive_index":
        assert_frame_equal(received, c.frame[c.classifier.feature_names_used])


def test_unrelated_feature_can_change_population_observation(bridge_case):
    """OBSERVATION: candidate selection and inference schema are different stages."""
    c = bridge_case
    df = pd.concat([c.frame, c.frame], ignore_index=True)
    df["unrelated_instrumental_column"] = np.arange(10, dtype=float)
    df.loc[4, "unrelated_instrumental_column"] = np.nan
    df.to_csv(c.csv, index=False)
    c.run(Z_umap=np.zeros((9, 2)), y=np.arange(9), cluster_labels=np.arange(9))
    assert len(c.spy.predict_inputs[0]) == 9
    assert "unrelated_instrumental_column" not in c.spy.predict_inputs[0]
