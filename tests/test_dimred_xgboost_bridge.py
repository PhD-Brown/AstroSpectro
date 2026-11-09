"""Independent wrapper oracles; no claim of training or astrophysical validation."""

import builtins
import os
from pathlib import Path
import sys
import types

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
import pytest


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
    np.testing.assert_array_equal(result["confidence"], [0.7, 0.55, 0.8, 0.8, 0.55])
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


def test_snr_inclusive_boundary_and_absent_snr(bridge_case):
    c = bridge_case
    df = c.frame.copy()
    df["snr_r"] = [
        np.nextafter(10.0, 0.0),
        10.0,
        np.nextafter(10.0, 20.0),
        np.nan,
        15.0,
    ]
    df.to_csv(c.csv, index=False, float_format="%.17g")
    # CSV parser may round nextafter back to10; use representable decimal probes.
    df["snr_r"] = [9.999, 10.0, 10.001, np.nan, 15.0]
    df.to_csv(c.csv, index=False)
    keep = [1, 2, 4]
    arrays = {k: v[keep] for k, v in c.args.items() if isinstance(v, np.ndarray)}
    result = c.run(**arrays)
    assert_frame_equal(
        c.spy.predict_inputs[-1],
        c.frame.iloc[keep][c.classifier.feature_names_used].reset_index(drop=True),
    )
    np.testing.assert_array_equal(result["Z_umap_aligned"][:, 0], [19, 807, 250])
    c.frame.drop(columns="snr_r").to_csv(c.csv, index=False)
    assert c.run()["_xgb_ok"]
    assert len(c.spy.predict_inputs[-1]) == 5


@pytest.mark.parametrize(
    "missing_count,keep",
    [(0, list(range(10))), (1, [0, 1, 2, 3, 5, 6, 7, 8, 9]), (2, list(range(10)))],
)
def test_candidate_nan_fraction_boundary(bridge_case, missing_count, keep):
    c = bridge_case
    df = pd.concat(
        [c.frame, c.frame.assign(obsid=lambda x: x.obsid + 1000)], ignore_index=True
    )
    # This column is not used by the model; exactly1/10 is selected for dropna.
    df["auxiliary"] = np.arange(10, dtype=float)
    for row in [4, 7][:missing_count]:
        df.loc[row, "auxiliary"] = np.nan
    df.to_csv(c.csv, index=False)
    result = c.run(
        Z_umap=np.column_stack((df.iloc[keep].obsid, np.zeros(len(keep)))),
        y=np.array([f"id{x}" for x in df.iloc[keep].obsid]),
        cluster_labels=np.arange(len(keep)),
    )
    assert_frame_equal(
        c.spy.predict_inputs[-1],
        df.iloc[keep][c.classifier.feature_names_used].reset_index(drop=True),
    )
    np.testing.assert_array_equal(result["Z_umap_aligned"][:, 0], df.iloc[keep].obsid)


@pytest.mark.parametrize(
    "column", ["obsid", "source_id", "teff_gspphot", "snr_g", "subclass"]
)
def test_excluded_metadata_nan_cannot_remove_rows(bridge_case, column):
    c = bridge_case
    df = c.frame.copy()
    df[column] = [1.0, 2.0, np.nan, 4.0, 5.0]
    df.to_csv(c.csv, index=False)
    assert c.run()["_xgb_ok"]
    assert_frame_equal(
        c.spy.predict_inputs[0], c.frame[c.classifier.feature_names_used]
    )


@pytest.mark.parametrize(
    "kind", ["constant", "string", "boolean", "all_nan", "constant_after_snr"]
)
def test_candidate_selection_keeps_named_model_columns(bridge_case, kind):
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
    c.run(**{k: v[keep] for k, v in c.args.items() if isinstance(v, np.ndarray)})
    assert_frame_equal(
        c.spy.predict_inputs[0],
        c.frame.iloc[keep][c.classifier.feature_names_used].reset_index(drop=True),
    )


@pytest.mark.parametrize(
    "dtype",
    ["int32", "int64", "float32", "float64", "Int64", "Float64", "boolean", "string"],
)
def test_representable_dataframe_dtypes_are_passed_by_name(
    bridge_case, monkeypatch, dtype
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
    c.run()
    assert_frame_equal(c.spy.predict_inputs[0], df[c.classifier.feature_names_used])
    assert_frame_equal(df, snapshot)


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
    bridge_case, total, missing, accepted
):
    """Protect technical cutoff only; zero substitution is not scientifically approved."""
    c = bridge_case
    present = ["feature_alpha"] + [f"available{i}" for i in range(total - missing - 1)]
    df = c.frame.copy()
    for name in present[1:]:
        df[name] = np.arange(5) + 100
    needed = present + [f"absent{i}" for i in range(missing)]
    c.classifier.feature_names_used = needed
    df.to_csv(c.csv, index=False)
    result = c.run()
    if not accepted:
        assert_empty(result)
        assert not c.spy.predict_inputs
    else:
        expected = df[present].copy()
        for name in needed[len(present) :]:
            expected[name] = 0.0
        assert_frame_equal(c.spy.predict_inputs[0], expected)
        assert result["_xgb_ok"]


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


def test_probability_row_maximum_does_not_override_decision_rule(bridge_case):
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
    np.testing.assert_array_equal(result["confidence"], [0.8, 0.6, 0.7, 0.9, 0.5])
    np.testing.assert_array_equal(result["y_pred"], ["G", "F", "M", "M", "F"])
    # A documented threshold wrapper may disagree with argmax; do not reject it.
    assert result["has_proba"]


@pytest.mark.parametrize("stage", ["csv", "predict", "decode"])
def test_postload_exceptions_remain_visible(bridge_case, stage):
    """Observed explicit failures are admissible; no blanket exception swallowing."""
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
    with pytest.raises(expected):
        c.run()


def test_path_resolution_side_effect_is_idempotent_observation(bridge_case):
    c = bridge_case
    path = str(Path(c.args["paths"]["SRC_DIR"]).resolve())
    sys.path[:] = [p for p in sys.path if p != path]
    c.run()
    c.run()
    assert sys.path.count(path) == 1
    assert sys.path[0] == path
