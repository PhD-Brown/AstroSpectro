"""Discriminating controls for the evidence helpers, without backend claims."""

import logging

import pytest


@pytest.mark.parametrize(
    "value", [None, "NaT", "-Inf", {"value": None, "reason": "nonfinite input"}]
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
