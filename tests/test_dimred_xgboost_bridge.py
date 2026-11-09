"""Independent wrapper oracles; no claim of training or astrophysical validation."""

import builtins
import logging
import os
import re
from pathlib import Path
import sys
import types

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal as assert_pandas_frame_equal
import pytest


def assert_frame_equal(actual, expected):
    """Protect names/order/values, while allowing lossless numeric representations."""
    assert_pandas_frame_equal(actual, expected, check_dtype=False, check_exact=True)


def assert_confidence_policy(actual, historical_maximum, predicted_class_probability):
    """Keep both R6 semantics admissible, consistently across the whole result."""
    assert any(
        np.array_equal(actual, choice)
        for choice in (historical_maximum, predicted_class_probability)
    )


def assert_filtered_prediction(case, result, population, arrays):
    """Require exact retained values, labels and associations on successful inference."""
    assert result["_xgb_ok"] is True
    expected = population[case.classifier.feature_names_used].reset_index(drop=True)
    assert_frame_equal(case.spy.predict_inputs[-1], expected)
    assert_frame_equal(case.spy.proba_inputs[-1], expected)
    # Independent synthetic classifier contract: alpha modulo3 maps to G/M/F.
    labels = np.array(["G", "M", "F"])[population.feature_alpha.to_numpy() % 3]
    np.testing.assert_array_equal(result["y_pred"], labels)
    for supplied, returned in (
        ("Z_umap", "Z_umap_aligned"),
        ("y", "y_aligned"),
        ("cluster_labels", "cl_aligned"),
    ):
        np.testing.assert_array_equal(result[returned], arrays[supplied])


@pytest.mark.parametrize("structured", [False, True])
@pytest.mark.parametrize(
    "domain,message",
    [
        ("schema", "Non-numeric feature columns cannot reach the model"),
        ("schema", "Feature columns contain non-numeric data"),
        ("preprocessing", "NaN values in input features"),
        ("probability", "Probabilities are non-finite"),
        ("identity", "Cannot verify observation identity"),
        (
            "identity",
            "Observation population provenance cannot be verified after bridge filtering",
        ),
    ],
)
def test_refusal_helper_accepts_domain_and_cause_in_either_order(
    bridge_review_contracts, structured, domain, message
):
    def run():
        if not structured:
            raise ValueError(message)
        logging.getLogger(__name__).error(message)
        return {
            "_xgb_ok": False,
            "has_proba": False,
            "y_pred": None,
            "confidence": None,
        }

    assert (
        bridge_review_contracts.invoke(types.SimpleNamespace(run=run), domain) is None
    )


@pytest.mark.parametrize("structured", [False, True])
@pytest.mark.parametrize(
    "domain,message",
    [
        ("schema", "column index invalid: -1 (off-by-one in insertion loop)"),
        ("schema", "Feature schema missing: column index invalid (insertion loop)"),
        ("identity", "Observation identity is available"),
        ("preprocessing", "NaN values in the probability distribution"),
        ("probability", "Probabilities are finite"),
        ("schema", "Non-numeric thread identifier"),
    ],
)
def test_refusal_helper_rejects_internal_faults_or_missing_domain_cause(
    bridge_review_contracts, structured, domain, message
):
    def run():
        if not structured:
            raise ValueError(message)
        logging.getLogger(__name__).error(message)
        return {
            "_xgb_ok": False,
            "has_proba": False,
            "y_pred": None,
            "confidence": None,
        }

    with pytest.raises(
        AssertionError, match="Unrelated ValueError|relevant error diagnostic"
    ):
        bridge_review_contracts.invoke(types.SimpleNamespace(run=run), domain)


@pytest.mark.parametrize(
    "representation",
    [
        "float32",
        "float64",
        "float16_loss",
        "float16_exact",
        "rows",
        "columns",
        "perturbation",
    ],
)
def test_preprocessing_precision_is_bounded_independently_of_result_dtype(
    bridge_preprocessing_oracle, representation
):
    oracle = bridge_preprocessing_oracle
    expected = np.array(
        [[0.2383656473, 0.0819231921, -2.3757725695], [1.225, -0.9534625892, 5.0]]
    )
    if representation == "float16_exact":
        expected = np.array([[0.0, 1.0, -2.0], [4.0, 0.5, 8.0]])
    actual = expected.astype(
        representation
        if representation in ("float32", "float64")
        else "float16" if representation.startswith("float16") else "float64"
    )
    if representation == "rows":
        actual = actual[::-1]
    elif representation == "columns":
        actual = actual[:, ::-1]
    elif representation == "perturbation":
        actual[0, 0] += 1e-4  # Exceeds the independent float32 error allowance.
    if representation in ("float32", "float64", "float16_exact"):
        oracle.compare(actual, expected)
    else:
        with pytest.raises(AssertionError, match="Not equal to tolerance"):
            oracle.compare(actual, expected)


def assert_empty(result):
    expected = {
        "y_pred": None,
        "confidence": None,
        "has_proba": False,
        "classes_pred": [],
        "y_aligned": None,
        "cl_aligned": None,
        "Z_umap_aligned": None,
        "fg_mask": None,
        "model_path": None,
        "_xgb_ok": False,
    }
    assert result == expected


def test_empty_structure_is_fresh(bridge_module):
    first = bridge_module._empty_result()
    assert_empty(first)
    first["classes_pred"].append("stale")
    assert_empty(bridge_module._empty_result())


def test_nominal_named_values_decode_confidence_and_integrity(bridge_case):
    """Verify values and characterize the historical row-maximum confidence policy."""
    c = bridge_case
    original = c.csv.read_bytes()
    inputs = {k: v.copy() for k, v in c.args.items() if isinstance(v, np.ndarray)}
    palette = {7: "navy"}
    result = c.run(color_map=palette)
    assert set(result) == set(c.module._empty_result())
    assert result["_xgb_ok"] is True
    expected_frame = c.frame[["feature_gamma", "feature_alpha", "feature_beta"]]
    assert_frame_equal(c.spy.predict_inputs[0], expected_frame)
    assert_frame_equal(c.spy.proba_inputs[0], expected_frame)
    np.testing.assert_array_equal(result["y_pred"], ["G", "M", "F", "F", "M"])
    assert_confidence_policy(
        result["confidence"], [0.7, 0.55, 0.8, 0.8, 0.55], [0.1, 0.25, 0.1, 0.1, 0.25]
    )
    assert result["has_proba"] is True
    assert result["classes_pred"] == ["F", "G", "M"]
    np.testing.assert_array_equal(result["fg_mask"], [True, False, True, True, False])
    assert result["fg_mask"].dtype == bool
    assert result["model_path"] == str(c.model)
    for key, returned in [
        ("Z_umap", "Z_umap_aligned"),
        ("y", "y_aligned"),
        ("cluster_labels", "cl_aligned"),
    ]:
        np.testing.assert_array_equal(result[returned], inputs[key])
        np.testing.assert_array_equal(c.args[key], inputs[key])
    assert c.csv.read_bytes() == original
    assert palette == {7: "navy"}


@pytest.mark.parametrize("order", [[4, 0, 2, 1, 3], [3, 1, 4, 2, 0]])
def test_coherent_permutation_preserves_observation_values(bridge_case, order):
    c = bridge_case
    c.frame.iloc[order].to_csv(c.csv, index=False)
    arrays = {
        key: value[order]
        for key, value in c.args.items()
        if isinstance(value, np.ndarray)
    }
    result = c.run(**arrays)
    assert_frame_equal(
        c.spy.predict_inputs[0],
        c.frame.iloc[order][c.classifier.feature_names_used].reset_index(drop=True),
    )
    np.testing.assert_array_equal(
        result["y_pred"], np.array(["G", "M", "F", "F", "M"])[order]
    )
    for key, returned in [
        ("Z_umap", "Z_umap_aligned"),
        ("y", "y_aligned"),
        ("cluster_labels", "cl_aligned"),
    ]:
        np.testing.assert_array_equal(result[returned], arrays[key])


def test_repeat_inference_and_second_call_failure_have_no_stale_state(bridge_case):
    c = bridge_case
    first, second = c.run(), c.run()
    for key in ["y_pred", "confidence", "fg_mask", "Z_umap_aligned"]:
        np.testing.assert_array_equal(first[key], second[key])
    c.model.unlink()  # Synthetic tmp_path artifact only.
    assert_empty(c.run())


@pytest.mark.parametrize(
    "stage", ["no_model", "load", "metadata", "no_csv", "missing_schema"]
)
def test_early_failures_return_complete_empty_result(bridge_case, stage):
    c = bridge_case
    if stage == "no_model":
        c.model.unlink()
    elif stage == "load":
        c.import_module.SpectralClassifier.load_model = lambda path: (
            _ for _ in ()
        ).throw(OSError("unreadable synthetic model"))
    elif stage == "metadata":
        c.classifier.feature_names_used = None
    elif stage == "no_csv":
        c.csv.unlink()
    else:
        c.classifier.feature_names_used = ["absent"]
    assert_empty(c.run())
    assert not c.spy.predict_inputs


def test_model_selection_uses_actual_utils_mtime(bridge_case):
    c = bridge_case
    newer = c.model.with_name("spectral_classifier_aaa.pkl")
    lexical = c.model.with_name("spectral_classifier_zzz.pkl")
    newer.write_text("synthetic", encoding="utf-8")
    lexical.write_text("synthetic", encoding="utf-8")
    for file, timestamp in [(c.model, 100), (newer, 300), (lexical, 200)]:
        os.utime(file, (timestamp, timestamp))
    assert c.run()["model_path"] == str(newer)
    assert c.loaded == [str(newer)]


def test_utils_import_fallback_is_lexical_characterization(bridge_case, monkeypatch):
    """Historical fallback differs from actual utils; no claim this is preferable."""
    c = bridge_case
    latest = c.model.with_name("spectral_classifier_zzz.pkl")
    latest.write_text("synthetic", encoding="utf-8")
    os.utime(latest, (1, 1))
    original = builtins.__import__

    def block(name, *args, **kwargs):
        if name == "utils":
            raise ImportError("exercise real local fallback")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", block)
    assert c.run()["model_path"] == str(latest)


def test_classifier_script_import_fallback(bridge_case, monkeypatch):
    original = builtins.__import__

    def block(name, *args, **kwargs):
        if name == "pipeline.classifier":
            raise ImportError("exercise script import")
        return original(name, *args, **kwargs)

    monkeypatch.setitem(sys.modules, "classifier", bridge_case.import_module)
    monkeypatch.setattr(builtins, "__import__", block)
    assert bridge_case.run()["_xgb_ok"]


def test_classifier_both_imports_unavailable(bridge_case, monkeypatch):
    original = builtins.__import__

    def block(name, *args, **kwargs):
        if name in {"pipeline.classifier", "classifier"}:
            raise ImportError("missing classifier")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", block)
    assert_empty(bridge_case.run())


def test_explicit_csv_wins_and_fallback_is_lexical(bridge_case):
    c = bridge_case
    other = c.csv.with_name("features_zzz.csv")
    modified = c.frame.copy()
    modified["feature_beta"] += 1000
    modified.to_csv(other, index=False)
    os.utime(other, (1, 1))
    c.run()
    assert_frame_equal(
        c.spy.predict_inputs[-1], c.frame[c.classifier.feature_names_used]
    )
    c.run(features_stem="absent")
    assert_frame_equal(
        c.spy.predict_inputs[-1], modified[c.classifier.feature_names_used]
    )


def test_snr_inclusive_boundary_and_absent_snr(bridge_case, bridge_review_contracts):
    c = bridge_case
    df = c.frame.copy()
    # Decimal probes survive CSV inference without rounding back onto the edge.
    df["snr_r"] = [9.999, 10.0, 10.001, np.nan, 15.0]
    df.to_csv(c.csv, index=False)
    keep = [1, 2, 4]
    arrays = {k: v[keep] for k, v in c.args.items() if isinstance(v, np.ndarray)}
    result = bridge_review_contracts.invoke(c, "identity", **arrays)
    if result is not None:
        assert_filtered_prediction(c, result, c.frame.iloc[keep], arrays)
    c.frame.drop(columns="snr_r").to_csv(c.csv, index=False)
    assert c.run()["_xgb_ok"]
    assert len(c.spy.predict_inputs[-1]) == 5


@pytest.mark.parametrize(
    "missing_count,keep",
    [(0, list(range(10))), (1, [0, 1, 2, 3, 5, 6, 7, 8, 9]), (2, list(range(10)))],
)
def test_candidate_nan_fraction_boundary(
    bridge_case, bridge_review_contracts, missing_count, keep
):
    c = bridge_case
    df = pd.concat(
        [c.frame, c.frame.assign(obsid=lambda x: x.obsid + 1000)], ignore_index=True
    )
    # This column is not used by the model; exactly1/10 is selected for dropna.
    df["auxiliary"] = np.arange(10, dtype=float)
    for row in [4, 7][:missing_count]:
        df.loc[row, "auxiliary"] = np.nan
    df.to_csv(c.csv, index=False)
    arrays = dict(
        Z_umap=np.column_stack((df.iloc[keep].obsid, np.zeros(len(keep)))),
        y=np.array([f"id{x}" for x in df.iloc[keep].obsid]),
        cluster_labels=np.arange(len(keep)),
    )
    result = (
        bridge_review_contracts.invoke(c, "identity", **arrays)
        if missing_count == 1
        else c.run(**arrays)
    )
    if result is not None:
        assert_filtered_prediction(c, result, df.iloc[keep], arrays)


@pytest.mark.parametrize(
    "column", ["obsid", "source_id", "teff_gspphot", "snr_g", "subclass"]
)
def test_excluded_metadata_nan_cannot_remove_rows(bridge_case, column):
    c = bridge_case
    df = pd.concat([c.frame, c.frame], ignore_index=True)
    df[column] = [1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    df.to_csv(c.csv, index=False)
    assert c.run(
        Z_umap=np.zeros((10, 2)), y=np.arange(10), cluster_labels=np.arange(10)
    )["_xgb_ok"]
    assert_frame_equal(c.spy.predict_inputs[0], df[c.classifier.feature_names_used])


@pytest.mark.parametrize("kind", ["constant", "string", "nullable_boolean"])
def test_candidate_variance_and_dtype_control_row_selection(
    bridge_case, bridge_review_contracts, monkeypatch, kind
):
    c = bridge_case
    df = pd.concat([c.frame, c.frame], ignore_index=True)
    if kind == "constant":
        df["auxiliary"] = [7.0] * 9 + [np.nan]
    elif kind == "string":
        df["auxiliary"] = list("abcdefghi") + [None]
    else:
        df["auxiliary"] = pd.Series([True, False] * 4 + [True, pd.NA], dtype="boolean")
        monkeypatch.setattr(c.module.pd, "read_csv", lambda *a, **kw: df)
    df.to_csv(c.csv, index=False)
    keep = list(range(9 if kind == "nullable_boolean" else 10))
    arrays = dict(
        Z_umap=np.column_stack((keep, np.zeros(len(keep)))),
        y=np.array(keep),
        cluster_labels=np.array(keep),
    )
    result = (
        bridge_review_contracts.invoke(c, "identity", **arrays)
        if kind == "nullable_boolean"
        else c.run(**arrays)
    )
    if result is not None:
        assert_filtered_prediction(c, result, df.iloc[keep], arrays)


def test_nonconsecutive_pandas_index_preserves_row_values(bridge_case, monkeypatch):
    c = bridge_case
    df = c.frame.copy()
    df.index = [104, 19, 807, 33, 250]
    monkeypatch.setattr(c.module.pd, "read_csv", lambda *a, **kw: df)
    c.run()
    assert_frame_equal(
        c.spy.predict_inputs[0], c.frame[c.classifier.feature_names_used]
    )


@pytest.mark.parametrize(
    "kind", ["constant", "string", "boolean", "all_nan", "constant_after_snr"]
)
def test_candidate_selection_keeps_named_model_columns(
    bridge_case, bridge_review_contracts, kind
):
    c = bridge_case
    df = c.frame.copy()
    extra = {
        "constant": [1] * 5,
        "string": ["a", "b", None, "c", "d"],
        "boolean": [True, False, True, False, True],
        "all_nan": [np.nan] * 5,
        "constant_after_snr": [1, 1, 1, 1, 9],
    }[kind]
    df["extra"] = extra
    keep = [0, 1, 2, 3] if kind == "constant_after_snr" else list(range(5))
    if kind == "constant_after_snr":
        df.loc[4, "snr_r"] = 9
    df.to_csv(c.csv, index=False)
    arrays = {k: v[keep] for k, v in c.args.items() if isinstance(v, np.ndarray)}
    result = (
        bridge_review_contracts.invoke(c, "identity", **arrays)
        if kind == "constant_after_snr"
        else c.run(**arrays)
    )
    if result is not None:
        assert_filtered_prediction(c, result, c.frame.iloc[keep], arrays)


@pytest.mark.parametrize(
    "dtype",
    ["int32", "int64", "float32", "float64", "Int64", "Float64", "boolean", "string"],
)
def test_representable_dataframe_dtypes_are_passed_by_name(
    bridge_case, bridge_review_contracts, monkeypatch, dtype
):
    """read_csv normally infers dtypes; nullable types exercise the DataFrame boundary."""
    c = bridge_case
    df = c.frame.copy()
    values = (
        [True, False, True, False, True] if dtype == "boolean" else [3, 10, 8, 23, 4]
    )
    df["feature_alpha"] = pd.Series(values, dtype=dtype)
    snapshot = df.copy(deep=True)
    monkeypatch.setattr(c.module.pd, "read_csv", lambda *args, **kwargs: df)
    if dtype in {"boolean", "string"}:
        result = bridge_review_contracts.invoke(c, "schema")
    else:
        result = c.run()
    if result is not None and result["_xgb_ok"]:
        actual = c.spy.predict_inputs[0].apply(pd.to_numeric)
        expected = df[c.classifier.feature_names_used].apply(pd.to_numeric)
        assert_frame_equal(actual, expected)
    assert_pandas_frame_equal(df, snapshot)  # Caller-owned data must remain intact.


@pytest.mark.parametrize(
    "total,missing,accepted",
    [
        (3, 0, True),
        (3, 1, False),
        (4, 1, True),
        (10, 3, True),
        (10, 4, False),
        (7, 2, True),
    ],
)
def test_missing_feature_fraction_characterization(
    bridge_case, bridge_review_contracts, total, missing, accepted
):
    """Above30% currently refuses; below it zero/NaN/refusal stay open policies."""
    c = bridge_case
    present = ["feature_alpha"] + [f"available{i}" for i in range(total - missing - 1)]
    df = c.frame.copy()
    for name in present[1:]:
        df[name] = np.arange(5) + 100
    needed = present + [f"absent{i}" for i in range(missing)]
    c.classifier.feature_names_used = needed
    df.to_csv(c.csv, index=False)
    result = (
        bridge_review_contracts.invoke(c, "schema") if missing and accepted else c.run()
    )
    if not accepted:
        assert_empty(result)
        assert not c.spy.predict_inputs
    elif result is not None:
        assert result["_xgb_ok"]
        assert c.spy.predict_inputs, "An accepted schema must reach the estimator."
        expected = df[present].copy()
        actual = c.spy.predict_inputs[0]
        assert list(actual.columns) == needed
        assert_frame_equal(actual[present], expected)
        for name in needed[len(present) :]:
            values = actual[name].to_numpy()
            assert np.all(values == 0) or pd.isna(values).all(), (
                "This compatibility test allows the historical zero placeholder or "
                "missing values for the fitted pipeline; it endorses neither policy."
            )


@pytest.mark.parametrize(
    "labels,expected,mask",
    [
        ([2, 0, 1, 0, 2], ["F", "G", "M", "G", "F"], [1, 1, 0, 1, 1]),
        ([1, 1, 1, 1, 1], ["M"] * 5, [0] * 5),
    ],
)
def test_decoding_observed_classes_and_fg_mask(bridge_case, labels, expected, mask):
    c = bridge_case
    c.spy.predict_fn = lambda X: np.array(labels)
    result = c.run()
    np.testing.assert_array_equal(result["y_pred"], expected)
    assert result["classes_pred"] == sorted(set(expected))
    np.testing.assert_array_equal(result["fg_mask"], np.array(mask, dtype=bool))


@pytest.mark.parametrize(
    "prediction", [[7, 2, 7, 0, 2], ["G", "quasar", "F", "K", "M"]]
)
def test_no_encoder_preserves_labels_as_strings(bridge_case, prediction):
    c = bridge_case
    c.classifier.label_encoder = None
    c.spy.predict_fn = lambda X: np.array(prediction)
    result = c.run()
    np.testing.assert_array_equal(result["y_pred"], np.array(prediction, dtype=str))
    np.testing.assert_array_equal(
        result["fg_mask"], np.isin(np.array(prediction, dtype=str), ["F", "G"])
    )


@pytest.mark.parametrize("mode", ["absent", "raises"])
def test_probability_unavailable_has_explicit_flag(bridge_case, mode):
    c = bridge_case
    if mode == "absent":
        c.classifier.model_pipeline = types.SimpleNamespace(predict=c.spy.predict)
    else:
        c.spy.proba_fn = lambda X: (_ for _ in ()).throw(ValueError("no probabilities"))
    result = c.run()
    np.testing.assert_array_equal(result["confidence"], np.ones(5))
    assert result["has_proba"] is False
    assert result["_xgb_ok"] is True


@pytest.mark.parametrize("kind", ["vector", "empty_class_axis"])
def test_unusable_probability_shape_is_not_advertised(bridge_case, kind):
    c = bridge_case
    c.spy.proba_fn = lambda X: np.arange(5.0) if kind == "vector" else np.empty((5, 0))
    try:
        result = c.run()
    except ValueError:
        return  # Explicit domain refusal is also admissible.
    assert not result["_xgb_ok"] or result["has_proba"] is False
    if result["_xgb_ok"]:
        assert result["confidence"].shape == (5,)


def test_probability_row_maximum_does_not_override_decision_rule(bridge_case):
    """Characterize historical row-max confidence; semantic choice remains R6 OPEN."""
    c = bridge_case
    c.spy.predict_fn = lambda X: np.array([0, 2, 1, 1, 2])
    c.spy.proba_fn = lambda X: np.array(
        [
            [0.05, 0.15, 0.8],
            [0.6, 0.3, 0.1],
            [0.2, 0.7, 0.1],
            [0.9, 0.06, 0.04],
            [0.2, 0.3, 0.5],
        ]
    )
    result = c.run()
    assert_confidence_policy(
        result["confidence"], [0.8, 0.6, 0.7, 0.9, 0.5], [0.05, 0.1, 0.7, 0.06, 0.5]
    )
    np.testing.assert_array_equal(result["y_pred"], ["G", "F", "M", "M", "F"])
    # A documented threshold wrapper may disagree with argmax; do not reject it.
    assert result["has_proba"]


@pytest.mark.parametrize("stage", ["csv", "predict", "decode"])
def test_postload_exceptions_remain_visible(bridge_case, caplog, stage):
    """Permit informative exceptions or a diagnosed structured refusal (R6 OPEN)."""
    c = bridge_case
    if stage == "csv":
        c.csv.write_text("a,b\n1,2\n3,4,5\n", encoding="utf-8")
        expected = pd.errors.ParserError
    elif stage == "predict":
        c.spy.predict_fn = lambda X: (_ for _ in ()).throw(
            ValueError("unsupported features")
        )
        expected = ValueError
    else:
        c.classifier.label_encoder.inverse_transform = lambda codes: (
            _ for _ in ()
        ).throw(ValueError("invalid class codes"))
        expected = ValueError
    caplog.clear()
    try:
        result = c.run()
    except expected as exc:
        assert str(exc), "A propagated domain failure needs a useful diagnostic."
        return
    else:
        assert_empty(result)
        messages = " ".join(r.getMessage() for r in caplog.records if r.levelno >= 40)
        pattern = {
            "csv": r"csv|parse|tokeniz|read",
            "predict": r"predict|features|incompatib",
            "decode": r"decod|class|label|codes",
        }[stage]
        assert re.search(
            pattern, messages, re.I
        ), "A structured refusal needs a relevant error log."


@pytest.mark.parametrize("fallback", [False, True])
def test_model_discovery_ignores_other_pickle_families(
    bridge_case, monkeypatch, fallback
):
    c = bridge_case
    unrelated = c.model.with_name("zzz_other_experiment.pkl")
    unrelated.write_text(
        "Synthetic competing artifact, never deserialized", encoding="utf-8"
    )
    os.utime(c.model, (100, 100))
    os.utime(unrelated, (1000, 1000))
    if fallback:
        original = builtins.__import__

        def block(name, *args, **kwargs):
            if name == "utils":
                raise ImportError("exercise local artifact discovery")
            return original(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", block)
    assert c.run()["model_path"] == str(c.model)
    assert c.loaded == [str(c.model)]


@pytest.mark.integration
def test_actual_loader_shared_population_preserves_all_values(bridge_loader_population):
    c = bridge_loader_population("shared")
    result = c.run()
    assert result["_xgb_ok"]
    assert_frame_equal(
        c.spy.predict_inputs[0],
        c.population[c.classifier.feature_names_used].reset_index(drop=True),
    )
    np.testing.assert_array_equal(result["Z_umap_aligned"][:, 0], c.population.obsid)


@pytest.mark.parametrize("domain", ["identity", "preprocessing", "probability"])
def test_refusal_helper_rejects_unrelated_value_errors(
    bridge_case, bridge_review_contracts, domain
):
    bridge_case.spy.predict_fn = lambda X: (_ for _ in ()).throw(
        ValueError("row buffer overflow")
    )
    with pytest.raises(AssertionError, match="Unrelated ValueError"):
        bridge_review_contracts.invoke(bridge_case, domain)


def test_path_resolution_side_effect_is_idempotent_observation(bridge_case):
    c = bridge_case
    path = str(Path(c.args["paths"]["SRC_DIR"]).resolve())
    sys.path[:] = [p for p in sys.path if p != path]
    c.run()
    c.run()
    assert sys.path.count(path) == 1
    assert sys.path[0] == path
