"""Independent synthetic contracts for reporter arithmetic and artifact boundaries."""

from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest


def strict_load(path):
    def reject(token):
        raise ValueError(f"Nonstandard JSON token: {token}")

    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=reject)


@pytest.mark.parametrize(
    "labels, expected",
    [
        (["M", "F", "M", "G", "M", "F"], {"M": 3, "F": 2, "G": 1}),
        ([], {}),
        ([3, -1, 3, 7], {"-1": 1, "3": 2, "7": 1}),
        ([True, False, True], {"True": 2, "False": 1}),
    ],
)
def test_class_counts_manual(reporter_module, labels, expected):
    assert reporter_module.class_counts(labels) == expected


@pytest.mark.parametrize("limit, rows", [(None, 3), (0, 0), (2, 2)])
def test_dataframe_records_finite_and_nullable(reporter_module, limit, rows):
    df = pd.DataFrame(
        {
            "flux / A": pd.array([1.5, 4.5, None], dtype="Float64"),
            "count": pd.array([4, None, 9], dtype="Int64"),
            "name": pd.Series(
                ["étoile", None, "QSO"], dtype=object, index=[91, 7, 300]
            ),
            "flag": pd.array([True, None, False], dtype="boolean"),
        },
        index=[91, 7, 300],
    )
    before = df.copy(deep=True)
    expected = [
        {"flux / A": 1.5, "count": 4, "name": "étoile", "flag": True},
        {"flux / A": 4.5, "count": None, "name": None, "flag": None},
        {"flux / A": None, "count": 9, "name": "QSO", "flag": False},
    ][:rows]
    result = reporter_module.safe_df_records(df, limit)
    assert result == expected
    assert json.loads(json.dumps(result, allow_nan=False)) == expected
    pd.testing.assert_frame_equal(df, before)


def test_dataframe_records_none_and_empty(reporter_module):
    assert reporter_module.safe_df_records(None) is None
    assert reporter_module.safe_df_records(pd.DataFrame(columns=["a"])) == []


@pytest.mark.parametrize(
    "value, expected",
    [
        (np.int64(107), 107),
        (np.uint32(71), 71),
        (np.float32(1.25), 1.25),
        (np.float64(-7.5), -7.5),
        (np.array([[1, 4], [9, 16]]), [[1, 4], [9, 16]]),
        (Path("input") / "étoile.csv", str(Path("input") / "étoile.csv")),
        (datetime(2026, 11, 9, 12, 3, 4), "2026-11-09T12:03:04"),
        (
            datetime(2026, 11, 9, 12, 3, 4, tzinfo=timezone(timedelta(hours=-5))),
            "2026-11-09T12:03:04-05:00",
        ),
        (pd.Timestamp("2026-11-09T17:03:04Z"), "2026-11-09T17:03:04+00:00"),
    ],
)
def test_json_default_exact(reporter_module, value, expected):
    encoded = json.dumps({"outer": [value]}, default=reporter_module.json_default)
    assert json.loads(encoded) == {"outer": [expected]}


@pytest.mark.parametrize("value", [object(), {1, 2}, 1 + 2j, np.bool_(True)])
def test_json_default_unsupported_types_refused(reporter_module, value):
    with pytest.raises(TypeError):
        json.dumps(value, default=reporter_module.json_default)


def test_embedding_axes_independent_population_oracle(reporter_module):
    Z = np.array([[1.0, 4, 9], [2, 7, 3], [8, 1, 6]])
    # Population variances: 86/9, 6, 6; no NumPy reduction as oracle.
    stats = reporter_module.embedding_stats(Z)
    expected = dict(
        shape=[3, 3],
        x_min=1,
        x_max=8,
        y_min=1,
        y_max=7,
        x_mean=11 / 3,
        y_mean=4,
        x_std=math.sqrt(86) / 3,
        y_std=math.sqrt(6),
    )
    assert stats.keys() == expected.keys()
    for key in expected:
        assert stats[key] == pytest.approx(expected[key])
    nd = reporter_module.embedding_stats_nd(Z, max_axes=2)
    assert nd == pytest.approx(
        {
            "shape": [3, 3],
            "axis_1_min": 1,
            "axis_1_max": 8,
            "axis_1_mean": 11 / 3,
            "axis_1_std": math.sqrt(86) / 3,
            "axis_2_min": 1,
            "axis_2_max": 7,
            "axis_2_mean": 4,
            "axis_2_std": math.sqrt(6),
        }
    )
    all_axes = reporter_module.embedding_stats_nd(Z)
    assert all_axes["axis_3_min"] == 3
    assert all_axes["axis_3_max"] == 9
    assert all_axes["axis_3_mean"] == 6
    assert all_axes["axis_3_std"] == pytest.approx(math.sqrt(6))


@pytest.mark.parametrize("Z", [None, [1, 4, 8], [[1], [4]], np.empty((3, 0))])
def test_embedding_shape_only_cases(reporter_module, Z):
    expected = None if Z is None else {"shape": list(np.asarray(Z).shape)}
    assert reporter_module.embedding_stats(Z) == expected
    nd = reporter_module.embedding_stats_nd(Z, max_axes=0)
    assert nd == expected


def test_embedding_missing_values_use_axis_population(reporter_module):
    result = reporter_module.embedding_stats([[1, 4], [np.nan, 7], [9, np.nan]])
    assert result == {
        "shape": [3, 2],
        "x_min": 1,
        "x_max": 9,
        "x_mean": 5,
        "x_std": 4,
        "y_min": 4,
        "y_max": 7,
        "y_mean": 5.5,
        "y_std": 1.5,
    }
    nd = reporter_module.embedding_stats_nd([[3], [11]])
    assert nd == {
        "shape": [2, 1],
        "axis_1_min": 3,
        "axis_1_max": 11,
        "axis_1_mean": 7,
        "axis_1_std": 4,
    }


@pytest.mark.parametrize("function", ["embedding_stats", "embedding_stats_nd"])
def test_embedding_nonnumeric_and_empty_refused(reporter_module, function):
    fn = getattr(reporter_module, function)
    with pytest.raises((ValueError, TypeError)):
        fn([["x", "y"], ["a", "b"]])
    with pytest.raises(ValueError):
        fn(np.empty((0, 2)))


def test_vector_statistics_manual_linear_percentiles(reporter_module):
    # Sorted finite values: 1,2,4,20. Linear p05=1+.15; p95=4+.85*16.
    result = reporter_module.vector_stats([[20, 1, np.nan], [4, np.inf, 2]])
    assert result == pytest.approx(
        {
            "n": 4,
            "min": 1,
            "max": 20,
            "mean": 27 / 4,
            "std": math.sqrt(955) / 4,
            "p05": 1.15,
            "p50": 3,
            "p95": 17.6,
        }
    )


@pytest.mark.parametrize("values", [[], [np.nan, np.inf, -np.inf], None])
def test_vector_unavailable_population(reporter_module, values):
    assert reporter_module.vector_stats(values) is None


@pytest.mark.parametrize("values", [[7], [7, 7, 7], [[7, 7], [7, 7]]])
def test_vector_constant_and_singleton(reporter_module, values):
    result = reporter_module.vector_stats(values)
    assert result == {
        "n": np.asarray(values).size,
        "min": 7,
        "max": 7,
        "mean": 7,
        "std": 0,
        "p05": 7,
        "p50": 7,
        "p95": 7,
    }


def test_numeric_means_columns_and_immutability(reporter_module):
    df = pd.DataFrame(
        {
            "a": [1.0, np.nan, 11],
            "b": [2, 4, 12],
            "s": ["1", "2", "3"],
            "flag": [True, False, True],
        }
    )
    before = df.copy(deep=True)
    assert reporter_module.numeric_means(df) == {"a": 6, "b": 6}
    assert reporter_module.numeric_means(df.iloc[:0]) is None
    assert reporter_module.numeric_means(None) is None
    pd.testing.assert_frame_equal(df, before)


@pytest.mark.parametrize(
    "value, expected",
    [
        (None, "N/A"),
        ("bad", "N/A"),
        (np.nan, "N/A"),
        (np.inf, "N/A"),
        (0, "0.00s"),
        (1.234, "1.23s"),
        (-4.5, "-4.50s"),
        (1e12, "1000000000000.00s"),
    ],
)
def test_duration_format_is_not_physical_validation(reporter_module, value, expected):
    assert reporter_module.fmt_seconds(value) == expected


def test_shapes_and_sensitivity(reporter_module):
    assert reporter_module.shape_or_none(None) is None
    assert reporter_module.shape_or_none([[1], [2]]) == [2, 1]
    assert reporter_module.sensitivity_shapes(None) is None
    assert reporter_module.sensitivity_shapes({9: np.zeros((3, 2)), "x": [1, 2]}) == {
        "9": [3, 2],
        "x": [2],
    }


def test_cluster_counts_and_classes_exact(reporter_module):
    clusters = [7, -1, 2, 7, 2, 7, 2, 7, -1]
    predictions = ["M", "F", "G", "F", "G", "M", "F", "M", "F"]
    assert reporter_module.cluster_top_counts(clusters) == {"7": 4, "2": 3}
    assert reporter_module.cluster_top_counts(clusters, 1) == {"7": 4}
    assert reporter_module.top3_pred_by_cluster(clusters, predictions) == {
        "noise": {"F": 2},
        "2": {"G": 2, "F": 1},
        "7": {"M": 3, "F": 1},
    }
    # No ordering is asserted for tied clusters or tied classes.
    assert reporter_module.cluster_top_counts([8, 3, 8, 3]) == {"8": 2, "3": 2}


@pytest.mark.parametrize("labels, expected", [([], {}), ([-1, -1], {})])
def test_cluster_empty_or_only_noise(reporter_module, labels, expected):
    assert reporter_module.cluster_top_counts(labels) == expected
    preds = ["F"] * len(labels)
    assert reporter_module.top3_pred_by_cluster(labels, preds) == (
        {} if not labels else {"noise": {"F": 2}}
    )


@pytest.mark.parametrize("predictions", [["F"], ["F", "G", "M"]])
def test_cluster_cardinality_refused(reporter_module, predictions):
    with pytest.raises((IndexError, ValueError)):
        reporter_module.top3_pred_by_cluster([0, 1], predictions)


def test_timestamp_utc_controlled_clock(reporter_module, monkeypatch):
    local = datetime(
        2026, 11, 9, 23, 58, 7, 999999, tzinfo=timezone(timedelta(hours=-5))
    )

    class Clock:
        @staticmethod
        def now(tz):
            assert tz == timezone.utc
            return local.astimezone(tz)

    monkeypatch.setattr(reporter_module, "datetime", Clock)
    assert reporter_module.make_timestamp() == "20261110T045807Z"
    # Seconds precision is an observation, not an assertion of unique run IDs.
    assert reporter_module.make_timestamp() == "20261110T045807Z"


def test_run_directory_preserves_content(reporter_module, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for base in [Path("relative"), tmp_path / "absolute"]:
        result = reporter_module.make_run_dir(base, "nb")
        assert result == base / "runs" / "nb"
        sentinel = result / "existing.txt"
        sentinel.write_text("keep", encoding="utf-8")
        assert reporter_module.make_run_dir(base, "nb") == result
        assert sentinel.read_text(encoding="utf-8") == "keep"


def test_run_directory_permission_error_propagates(
    reporter_module, tmp_path, monkeypatch
):
    def denied(*args, **kwargs):
        raise PermissionError("synthetic denied directory")

    monkeypatch.setattr(Path, "mkdir", denied)
    with pytest.raises(PermissionError, match="synthetic denied"):
        reporter_module.make_run_dir(tmp_path, "nb")


def assert_payload(payload, expected):
    assert payload.keys() == expected.keys()
    for key, value in expected.items():
        if isinstance(value, pd.DataFrame):
            pd.testing.assert_frame_equal(payload[key], value)
        elif isinstance(value, np.ndarray):
            np.testing.assert_array_equal(payload[key], value)
        else:
            assert payload[key] == value


def test_joblib_three_destinations_and_run_isolation(reporter_module, tmp_path):
    legacy = tmp_path / "legacy.joblib"
    first = {
        "a": np.array([[7, 2], [31, 8]]),
        "meta": pd.DataFrame({"obsid": [91, 7]}, index=[5, 2]),
        "run": "a",
    }
    run_a, latest = reporter_module.save_joblib_pair(
        first, tmp_path, "a", "latest", legacy
    )
    for path in [run_a, latest, legacy]:
        assert_payload(joblib.load(path), first)
    original = run_a.read_bytes()
    second = {
        "a": np.array([[1, 9]]),
        "meta": pd.DataFrame({"obsid": [300]}),
        "run": "b",
    }
    run_b, latest_b = reporter_module.save_joblib_pair(
        second, tmp_path, "b", "latest", legacy
    )
    assert latest_b == latest
    assert run_a.read_bytes() == original
    for path in [run_b, latest_b, legacy]:
        assert_payload(joblib.load(path), second)


def test_json_txt_four_destinations_and_unicode(reporter_module, tmp_path):
    report = {
        "run": "a",
        "étoile": [np.int64(91), np.array([2, 7])],
        "source": Path("synth.csv"),
    }
    paths = reporter_module.save_json_txt_pair(
        report, tmp_path, "a", "latest", ["étoile", "MSE: 1.25"]
    )
    assert paths == tuple(
        tmp_path / name for name in ["a.json", "latest.json", "a.txt", "latest.txt"]
    )
    for path in paths[:2]:
        assert strict_load(path) == {
            "run": "a",
            "étoile": [91, [2, 7]],
            "source": "synth.csv",
        }
        assert "étoile" in path.read_text(encoding="utf-8")
    for path in paths[2:]:
        assert path.read_text(encoding="utf-8").splitlines() == ["étoile", "MSE: 1.25"]
    old_json, old_text = paths[0].read_bytes(), paths[2].read_bytes()
    newer = reporter_module.save_json_txt_pair(
        {"run": "b"}, tmp_path, "b", "latest", ["run b"]
    )
    assert strict_load(newer[0]) == strict_load(newer[1]) == {"run": "b"}
    assert newer[2].read_bytes() == newer[3].read_bytes()
    assert paths[0].read_bytes() == old_json and paths[2].read_bytes() == old_text


def test_joblib_without_legacy_and_missing_parent(reporter_module, tmp_path):
    run, latest = reporter_module.save_joblib_pair({"x": 7}, tmp_path, "run", "latest")
    assert joblib.load(run) == joblib.load(latest) == {"x": 7}
    with pytest.raises(FileNotFoundError):
        reporter_module.save_joblib_pair({}, tmp_path / "absent", "run", "latest")
