"""Revised R1-R7 red evidence, kept outside normal pytest collection."""

import builtins
import json
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import pytest


@pytest.mark.parametrize(
    "dtype,values",
    [
        ("float64", [1.25, np.nan, 9.5]),
        ("float32", [1.25, np.nan, 9.5]),
        ("Float64", [1.25, np.inf, 9.5]),
        ("float64", [np.nan] * 3),
        ("str", ["F", None, "G"]),
        ("string", ["F", None, "G"]),
        (
            "datetime64[ns]",
            [pd.Timestamp("2026-01-02"), pd.NaT, pd.Timestamp("2026-03-04")],
        ),
    ],
)
def test_r1_table_information(reporter_module, reporter_contracts, dtype, values):
    h = reporter_contracts
    df = pd.DataFrame({"obsid": [91, 7, 300], "value": pd.array(values, dtype=dtype)})
    before = df.copy(deep=True)
    try:
        records = reporter_module.safe_df_records(df)
    except Exception as error:
        h.refusal_mentions(
            error, ["finite", "missing", "nan", "inf", "nat", "table", "dataframe"]
        )
        pd.testing.assert_frame_equal(df, before)
        return
    try:
        encoded = json.dumps(
            records, default=reporter_module.json_default, allow_nan=False
        )
    except Exception as error:
        # The conversion itself must return identifiable missing values, rather
        # than relying on a downstream JSON exception to hide a broken promise.
        pytest.fail(
            f"Table conversion did not produce strict serializable records: {error}"
        )
    actual = json.loads(encoded)
    pd.testing.assert_frame_equal(df, before)
    assert [row["obsid"] for row in actual] == [91, 7, 300]
    assert all(set(row) == {"obsid", "value"} for row in actual)
    for row, value in zip(actual, values):
        if pd.isna(value) or isinstance(value, float) and not np.isfinite(value):
            assert h.unavailable(
                row["value"]
            ), "Missing table value became a fictitious measure"
        elif isinstance(value, pd.Timestamp):
            assert row["value"] == value.isoformat()
        else:
            assert row["value"] == value


@pytest.mark.parametrize(
    "invalid",
    [np.nan, np.float32(np.nan), np.inf, -np.inf, pd.NaT, np.array([2.5, np.inf, 8.0])],
)
@pytest.mark.parametrize("existing", [False, True])
def test_r1_writer_information_and_safe_refusal(
    reporter_module, reporter_contracts, tmp_path, invalid, existing
):
    h = reporter_contracts
    if existing:
        reporter_module.save_json_txt_pair(
            {"run": "A", "metric": 3.25}, tmp_path, "A", "latest", ["A"]
        )
    before = h.snapshot(tmp_path)
    wanted = {
        "run": "B",
        "obsid": [91, 7, 300],
        "finite": [1.25, -9.5],
        "metric": invalid,
    }
    try:
        paths = reporter_module.save_json_txt_pair(
            wanted, tmp_path, "B", "latest", ["B"]
        )
    except Exception as error:
        h.refusal_mentions(
            error, ["finite", "nan", "inf", "nat", "json", "missing", "serializ"]
        )
        assert (
            h.snapshot(tmp_path) == before
        ), "Refusal published or changed an artifact"
        return
    for path in paths[:2]:
        try:
            result = h.strict_json(path)
        except ValueError as error:
            pytest.fail(str(error))
        expected = dict(wanted)
        if invalid is pd.NaT:
            expected["metric"] = None
        h.preserve_tree(result, expected)


@pytest.mark.parametrize("kind", ["joblib", "json", "pca", "embedding", "ae"])
def test_r2_collision_preserves_every_existing_artifact(
    reporter_module, reporter_contracts, reporter_case, kind
):
    c, h = reporter_case, reporter_contracts
    if kind == "joblib":
        c.reports.mkdir()

        def save(which):
            return reporter_module.save_joblib_pair(
                {"run": which, "obsid": [91] if which == "A" else [300, 7]},
                c.reports,
                "same",
                "latest",
                c.reports / "legacy.joblib",
            )

    elif kind == "json":
        c.reports.mkdir()

        def save(which):
            return reporter_module.save_json_txt_pair(
                {"run": which, "obsid": [91] if which == "A" else [300, 7]},
                c.reports,
                "same",
                "latest",
                [which],
            )

    else:
        fn, args, key = {
            "pca": (reporter_module.save_pca_run, c.pca_args, "scores"),
            "embedding": (
                reporter_module.save_umap_tsne_run,
                c.embedding_args,
                "Z_umap",
            ),
            "ae": (
                reporter_module.save_autoencoder_run,
                dict(c.ae_args, Z_ae=c.scores[:, :2]),
                "Z_ae",
            ),
        }[kind]

        def save(which):
            altered = (
                {}
                if which == "A"
                else {
                    "y": np.array(["M"] * 5 + ["F"]),
                    key: np.array(args[key], copy=True) + 100,
                }
            )
            return fn(**dict(args, **altered))

    first = save("A")  # Always-successful positive control on valid inputs.
    before = h.snapshot(c.reports)
    try:
        second = save("B")
    except Exception as error:
        h.refusal_mentions(error, ["collision", "exist", "identifier", "duplicate"])
        assert (
            h.snapshot(c.reports) == before
        ), "Collision refusal changed archives/latest/legacy"
        return
    # Unique successful runs may update aliases but every old archive is immutable.
    archives = {
        name: data
        for name, data in before.items()
        if "latest" not in name
        and name != "legacy.joblib"
        and not name.endswith("_output.joblib")
        and name != "phy3500_embeddings.joblib"
    }
    after = h.snapshot(c.reports)
    assert all(
        after.get(name) == data for name, data in archives.items()
    ), "First archive overwritten"
    first_paths = (
        first
        if isinstance(first, tuple)
        else [
            first[k]
            for k in [
                "json_path",
                "txt_path",
                "joblib_run" if kind == "embedding" else "joblib_path",
            ]
        ]
    )
    second_paths = (
        second
        if isinstance(second, tuple)
        else [
            second[k]
            for k in [
                "json_path",
                "txt_path",
                "joblib_run" if kind == "embedding" else "joblib_path",
            ]
        ]
    )
    for a, b in zip(first_paths, second_paths):
        if "latest" not in a.name:
            assert (
                a != b and b.is_file()
            ), "Successful distinct runs share an archive path"
    if kind == "joblib":
        for path in [*second, c.reports / "legacy.joblib"]:
            assert joblib.load(path) == {"run": "B", "obsid": [300, 7]}
    elif kind == "json":
        for path in second[:2]:
            assert h.strict_json(path) == {"run": "B", "obsid": [300, 7]}
        assert all(p.read_text(encoding="utf-8").strip() == "B" for p in second[2:])
    else:
        family, legacy = {
            "pca": ("phy3500_pca", "phy3500_pca_output.joblib"),
            "embedding": ("phy3500_embeddings", "phy3500_embeddings.joblib"),
            "ae": ("phy3500_autoencoder", "phy3500_autoencoder_output.joblib"),
        }[kind]
        stem = family + ("_latest" if kind == "embedding" else "_run_latest")
        jb_key = "joblib_run" if kind == "embedding" else "joblib_path"
        for path in [
            second[jb_key],
            second["run_dir"] / (stem + ".joblib"),
            c.reports / legacy,
        ]:
            payload = joblib.load(path)
            np.testing.assert_array_equal(payload["y"], ["M"] * 5 + ["F"])
            np.testing.assert_array_equal(payload[key], args[key] + 100)
        report = h.strict_json(second["json_path"])
        counts = (
            report["class_counts"]["y"] if kind == "pca" else report["class_counts"]
        )
        assert counts == {"M": 5, "F": 1}
        report_stem = "phy3500_umap_tsne_run_latest" if kind == "embedding" else stem
        assert h.strict_json(second["run_dir"] / (report_stem + ".json")) == report
        for path in [second["txt_path"], second["run_dir"] / (report_stem + ".txt")]:
            import ast

            assert (
                ast.literal_eval(h.text_value(path.read_text(encoding="utf-8"), "y"))
                == counts
            )
    if kind == "embedding":
        h.embedding_group(
            c.reports,
            [
                dict(c.embedding_args),
                dict(
                    c.embedding_args,
                    y=np.array(["M"] * 5 + ["F"]),
                    Z_umap=c.embedding_args["Z_umap"] + 100,
                ),
            ],
        )


@pytest.mark.parametrize("stage", range(1, 8))
@pytest.mark.parametrize("when", ["before", "after"])
def test_r3_complete_publication_group(
    reporter_module, reporter_contracts, reporter_case, monkeypatch, stage, when
):
    c, h = reporter_case, reporter_contracts
    a = dict(c.embedding_args)
    b = dict(a, y=np.array(["M"] * 5 + ["F"]), Z_umap=a["Z_umap"] + 100)
    reporter_module.save_umap_tsne_run(**a)
    h.embedding_group(c.reports, [a])
    monkeypatch.setattr(reporter_module, "make_timestamp", lambda: "20261109T170305Z")
    original = OSError(f"M27 publication stage {stage} {when}")
    real_dump, real_open = joblib.dump, builtins.open
    writes = []
    fired = False

    def trip(path, operation):
        nonlocal fired
        writes.append(str(path))
        selected = len(writes) == stage and not fired
        if selected:
            fired = True
            if when == "before":
                raise original
        result = operation()
        if selected and when == "after":
            raise original
        return result

    def dump(obj, path, *args, **kwargs):
        return trip(path, lambda: real_dump(obj, path, *args, **kwargs))

    class Writer:
        def __init__(self, path, mode, args, kwargs):
            self.path, self.mode, self.args, self.kwargs = path, mode, args, kwargs

        def __enter__(self):
            nonlocal fired
            writes.append(str(self.path))
            self.selected = len(writes) == stage and not fired
            if self.selected:
                fired = True
                if when == "before":
                    raise original
            self.stream = real_open(self.path, self.mode, *self.args, **self.kwargs)
            return self.stream

        def __exit__(self, *args):
            result = self.stream.__exit__(*args)
            if self.selected and when == "after":
                raise original
            return result

    def opened(path, mode="r", *args, **kwargs):
        if "w" in mode and str(path).endswith((".json", ".txt")):
            return Writer(path, mode, args, kwargs)
        return real_open(path, mode, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(joblib, "dump", dump)
        patch.setattr(builtins, "open", opened)
        try:
            reporter_module.save_umap_tsne_run(**b)
        except Exception as error:
            assert h.chain_has(error, original), "Original publication failure was lost"
    assert fired, "Requested write boundary was not exercised"
    h.embedding_group(c.reports, [a, b])


def test_r4_model_provenance_is_not_fabricated(
    reporter_module, reporter_contracts, reporter_case
):
    c, h = reporter_case, reporter_contracts
    figure = c.figures / "xgb_predictions.png"
    figure.write_bytes(b"synthetic figure")
    result = reporter_module.save_umap_tsne_run(
        **dict(c.embedding_args, y_pred=np.array(["F"] * 6), path_xgb=figure)
    )
    report = h.strict_json(result["json_path"])
    assert (
        Path(report["paths"]["xgboost_prediction_figure"]).resolve() == figure.resolve()
    )
    h.model_is_unavailable(report["xgboost_summary"])


@pytest.mark.parametrize("metric", ["ae", "pca"])
def test_r5_metric_identification_and_cause(
    reporter_module, reporter_contracts, reporter_case, monkeypatch, caplog, metric
):
    c, h = reporter_case, reporter_contracts
    original = ValueError(f"M27 {metric} numerical sentinel 4701")
    pca = SimpleNamespace(
        reconstruction_error=lambda X, n_components: np.array([2.0, 4.0, 6.0]),
        n_components_for_variance=lambda threshold: 3,
    )
    nominal_args = dict(c.ae_args, pca=pca)
    with caplog.at_level("WARNING"):
        nominal = reporter_module.save_autoencoder_run(**nominal_args)
    good = h.strict_json(nominal["json_path"])
    key = "ae_mse_global" if metric == "ae" else "pca_mse_2_components"
    assert good["metrics"][key] == pytest.approx(1 / 36 if metric == "ae" else 4)
    assert not h.metric_diagnostic(
        good, caplog.records, reporter_module.logger.name, metric, original
    ), "Nominal metric falsely declared failed"
    caplog.clear()
    monkeypatch.setattr(reporter_module, "make_timestamp", lambda: "20261109T170305Z")

    def failed(*args, **kwargs):
        raise original

    monkeypatch.setattr(
        c.ae if metric == "ae" else pca,
        "reconstruction_mse" if metric == "ae" else "reconstruction_error",
        failed,
    )
    with caplog.at_level("WARNING"):
        try:
            result = reporter_module.save_autoencoder_run(**nominal_args)
        except Exception as error:
            assert h.chain_has(error, original), "Original metric cause was lost"
            return
    report = h.strict_json(result["json_path"])
    assert h.unavailable(
        report["metrics"][key]
    ), "Failed metric became a finite measurement"
    assert h.metric_diagnostic(
        report, caplog.records, reporter_module.logger.name, metric, original
    ), "No metric-specific diagnostic with an original cause"


def test_r6_nonmonotone_finite_pair_identity(
    reporter_module, reporter_contracts, reporter_case
):
    c, h = reporter_case, reporter_contracts
    Z = np.array([[4.0, 6], [1, 5], [np.inf, 4], [2, 3], [6, 2], [3, 1]])
    args = dict(
        c.ae_args, Z_ae=Z, meta=c.meta.assign(teff_gspphot=[30.0, 10, 50, 20, 60, 40])
    )
    before = h.snapshot(c.reports)
    try:
        with np.errstate(invalid="ignore"):
            result = reporter_module.save_autoencoder_run(**args)
    except Exception as error:
        h.refusal_mentions(error, ["latent", "coordinate", "pair"])
        assert (
            h.snapshot(c.reports) == before
        ), "Scientific refusal left published artifacts"
        return
    report = json.loads(result["json_path"].read_text(encoding="utf-8"))
    value = report["correlations"]["ae_latent_vs_physical_spearman"]["teff_gspphot"][
        "latent_axis_1"
    ]
    # Paired ranks [4,1,2,5,3] / [3,1,2,5,4], squared differences sum2.
    assert h.unavailable(value) or value == pytest.approx(
        0.9, abs=1e-12, rel=0
    ), "Latent/physical pairs are invalid or misassociated"


def test_r7_no_invented_second_latent_axis(
    reporter_module, reporter_contracts, reporter_case
):
    c, h = reporter_case, reporter_contracts
    before = h.snapshot(c.reports)
    args = dict(
        c.ae_args,
        Z_ae=np.arange(6.0)[:, None],
        meta=c.meta.assign(teff_gspphot=[1.0, 2, 3, 4, 5, 6]),
    )
    try:
        result = reporter_module.save_autoencoder_run(**args)
    except Exception as error:
        assert not isinstance(error, IndexError), "Unchecked missing axis"
        h.refusal_mentions(error, ["dimension", "axis", "latent", "shape"])
        assert h.snapshot(c.reports) == before, "Dimensional refusal left artifacts"
        return
    report = h.strict_json(result["json_path"])
    assert report["shapes"]["Z_ae"] == [6, 1]
    correlations = report["correlations"]["ae_latent_vs_physical_spearman"][
        "teff_gspphot"
    ]
    assert correlations["latent_axis_1"] == pytest.approx(1.0, abs=1e-12)
    assert h.unavailable(
        correlations.get("latent_axis_2")
    ), "Invented correlation for absent axis"


def test_positive_valid_report_remains_informative(
    reporter_module, reporter_case, reporter_contracts
):
    c = reporter_case
    result = reporter_module.save_autoencoder_run(**c.ae_args)
    report = reporter_contracts.strict_json(result["json_path"])
    assert report["metrics"]["ae_mse_global"] == pytest.approx(1 / 36)
    np.testing.assert_array_equal(joblib.load(result["joblib_path"])["y"], c.y)


def test_r11_reference_substitution_is_traceable(
    reporter_module, reporter_contracts, reporter_case
):
    c, h = reporter_case, reporter_contracts
    args = dict(
        c.embedding_args,
        y=np.array(["STAR", "QSO", "GALAXY", "STAR", "STAR", "QSO"]),
        y_pred=np.array(["STAR", "STAR", "QSO", "STAR", "GALAXY", "QSO"]),
        y_aligned=np.array(["M", "F", "G", "M", "F"]),
    )
    before = h.snapshot(c.reports)
    try:
        result = reporter_module.save_umap_tsne_run(**args)
    except Exception as error:
        h.refusal_mentions(
            error, ["reference", "align", "population", "cardinality", "length"]
        )
        assert h.snapshot(c.reports) == before
        return
    report = h.strict_json(result["json_path"])
    summary = report["xgboost_summary"]
    value = summary.get("accuracy_vs_reference")
    if h.unavailable(value):
        return
    assert value == 0.5
    sources = [
        text.lower()
        for path, text in h.text_leaves(summary)
        if any(
            "reference" in key.lower() or "provenance" in key.lower() for key in path
        )
    ]
    assert any(
        "original" in text or "labels_norm" in text or "fallback" in text
        for text in sources
    ), "Reference population silently substituted"
