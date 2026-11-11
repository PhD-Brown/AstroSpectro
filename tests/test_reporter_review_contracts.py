"""Discriminating controls for the evidence helpers, without backend claims."""

import logging

import pytest


@pytest.mark.parametrize(
    "value", [None, "NaT", "-Inf", {"value": None, "reason": "nonfinite input", "n": 5}]
)
def test_explicit_unavailability_is_admissible(reporter_contracts, value):
    assert reporter_contracts.unavailable(value)


@pytest.mark.parametrize(
    "value",
    [
        0.0,
        1,
        False,
        "2026-01-02T00:00:00",
        {"error": None},
        {"value": 0.0, "reason": "missing"},
    ],
)
def test_missing_cannot_be_a_finite_measure(reporter_contracts, value):
    assert not reporter_contracts.unavailable(value)


@pytest.mark.parametrize(
    "value", ["figure.png", "figure.pkl", "./figure.png", {"artifact": "figure.pkl"}]
)
def test_unprovided_model_provenance_rejected(reporter_contracts, value):
    with pytest.raises(AssertionError, match="provenance"):
        reporter_contracts.model_is_unavailable({"model_artifact": value})


@pytest.mark.parametrize(
    "diagnostic",
    [
        {"diagnostics": {"error": None}},
        {"metrics": {"pca_reconstruction_error": None}},
        {"diagnostics": {"reason": "PCA reconstruction_error baseline requested"}},
    ],
)
def test_keys_and_unrelated_words_are_not_diagnostics(reporter_contracts, diagnostic):
    assert not reporter_contracts.metric_diagnostic(
        diagnostic, [], "m27_reporter", "pca", ValueError("sentinel")
    )


def test_diagnostic_requires_metric_cause_and_correct_logger(reporter_contracts):
    original = ValueError("numerical sentinel")
    text = "PCA reconstruction unavailable: ValueError numerical sentinel"
    foreign = logging.LogRecord("other_reporter", 40, __file__, 1, text, (), None)
    local = logging.LogRecord("m27_reporter", 30, __file__, 1, text, (), None)
    assert not reporter_contracts.metric_diagnostic(
        {}, [foreign], "m27_reporter", "pca", original
    )
    assert reporter_contracts.metric_diagnostic(
        {}, [local], "m27_reporter", "pca", original
    )
    assert not reporter_contracts.metric_diagnostic(
        {}, [local], "m27_reporter", "ae", original
    )
    assert reporter_contracts.metric_diagnostic(
        {"pca_mse": {"status": "unavailable", "reason": "ValueError"}},
        [],
        "m27_reporter",
        "pca",
        original,
    )


def test_wrapped_original_exception_is_identifiable(reporter_contracts):
    original = OSError("write failed")
    wrapper = RuntimeError("publication aborted")
    wrapper.__cause__ = original
    assert reporter_contracts.chain_has(wrapper, original)
    assert not reporter_contracts.chain_has(RuntimeError("same words"), original)


def test_text_extraction_allows_equivalent_spacing(reporter_contracts):
    assert (
        reporter_contracts.text_value(
            "\n - AE MSE global    :    0.25\n\n", "AE MSE global"
        )
        == "0.25"
    )


@pytest.mark.parametrize(
    "association,accepted",
    [
        ({}, False),
        ({"generation": "20261109T170304Z"}, False),
        ({"generation": "20261109T170305Z"}, True),
        ({"reason": "M27 publication stage 4 before"}, True),
    ],
)
def test_incomplete_marker_matches_failed_generation_or_cause(
    reporter_contracts, tmp_path, association, accepted
):
    import json

    aliases = [
        tmp_path / name
        for name in ["latest.joblib", "legacy.joblib", "latest.json", "latest.txt"]
    ]
    marker = dict(
        status="incomplete", artifacts=[p.name for p in aliases], **association
    )
    (tmp_path / "status.json").write_text(json.dumps(marker), encoding="utf-8")
    assert (
        reporter_contracts.incomplete_marker(
            tmp_path,
            aliases,
            link=("20261109T170305Z", "M27 publication stage 4 before"),
        )
        is accepted
    )
    # A declaration of unavailability is visible in nominal checks too; it is
    # never evidence that an otherwise successful publication is complete.
    assert reporter_contracts.incomplete_marker(tmp_path, aliases)


@pytest.mark.parametrize("generation", [None, "20261109T170304Z"])
def test_generic_or_stale_marker_does_not_bypass_group_contents(
    reporter_contracts, tmp_path, generation
):
    import json

    marker = {
        "status": "incomplete",
        "artifacts": [
            "phy3500_embeddings_latest.joblib",
            "phy3500_embeddings.joblib",
            "phy3500_umap_tsne_run_latest.json",
            "phy3500_umap_tsne_run_latest.txt",
        ],
        "generation": generation,
    }
    (tmp_path / "status.json").write_text(json.dumps(marker), encoding="utf-8")
    with pytest.raises(AssertionError, match="missing members"):
        reporter_contracts.embedding_group(
            tmp_path, [], failure_links=("20261109T170305Z", "numerical write failure")
        )


@pytest.mark.parametrize(
    "metric,correct,inverted",
    [
        ("ae", "Autoencoder", "PCA"),
        ("pca", "PCA", "Autoencoder"),
    ],
)
def test_neutral_cause_requires_independent_metric_attribution(
    reporter_contracts, metric, correct, inverted
):
    original = ValueError("numerical sentinel 4701")

    def record(domain):
        return logging.LogRecord(
            "m27_reporter",
            30,
            __file__,
            1,
            f"{domain} unavailable: {original}",
            (),
            None,
        )

    assert reporter_contracts.metric_diagnostic(
        {}, [record(correct)], "m27_reporter", metric, original
    )
    assert not reporter_contracts.metric_diagnostic(
        {}, [record(inverted)], "m27_reporter", metric, original
    )
    assert not reporter_contracts.metric_diagnostic(
        {}, [record(inverted)], "unrelated_reporter", metric, original
    )
    assert reporter_contracts.metric_diagnostic(
        {metric: {"status": "unavailable", "reason": str(original)}},
        [],
        "m27_reporter",
        metric,
        original,
    )
