"""Real CPU XGBoost plus named preprocessing and trusted synthetic persistence.

Included in fast CI: XGBoost is a required dependency, these fits are miniature.
This does not exercise SpectralClassifier.train_and_evaluate or historical models.
"""

import json

import numpy as np
import pytest

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("reverse_columns", [False, True])
def test_real_xgboost_saved_pipeline_inference_parity(
    bridge_real_bundle, bridge_preprocessing_oracle, reverse_columns
):
    c = bridge_real_bundle
    query = c.query.copy()
    if reverse_columns:
        query = query[query.columns[::-1]]
        query["unused_extra"] = [999, 2, 7, 31, 8]
        query.to_csv(c.csv, index=False)
    names = ["feature_gamma", "feature_alpha", "feature_beta"]
    encoded = c.classifier.model_pipeline.predict(query[names])
    direct = c.classifier.label_encoder.inverse_transform(encoded)
    probabilities = c.classifier.model_pipeline.predict_proba(query[names])
    loaded = c.classifier_module.SpectralClassifier.load_model(str(c.model))
    assert loaded.feature_names_used == names
    assert loaded.class_labels == ["F", "G", "M"]
    np.testing.assert_array_equal(loaded.label_encoder.classes_, ["F", "G", "M"])
    np.testing.assert_array_equal(loaded.model_pipeline.predict(query[names]), encoded)
    np.testing.assert_allclose(
        loaded.model_pipeline.predict_proba(query[names]),
        probabilities,
        rtol=0,
        atol=1e-12,
    )
    observation = bridge_preprocessing_oracle
    observation.inputs.clear()
    result = c.run()
    assert len(observation.inputs) == 1
    observation.compare(observation.inputs[0], observation.expected(query))
    np.testing.assert_array_equal(result["y_pred"], direct)
    np.testing.assert_allclose(
        result["confidence"], np.max(probabilities, axis=1), rtol=0, atol=1e-12
    )
    np.testing.assert_array_equal(result["y_aligned"], c.args["y"])
    np.testing.assert_array_equal(result["cl_aligned"], c.args["cluster_labels"])
    np.testing.assert_array_equal(
        result["Z_umap_aligned"][:, 0], [104, 19, 807, 33, 250]
    )
    np.testing.assert_array_equal(result["fg_mask"], np.isin(direct, ["F", "G"]))
    assert result["classes_pred"] == sorted(set(direct))
    assert result["_xgb_ok"] is True and result["has_proba"] is True
    assert result["model_path"] == str(c.model)
    # Numerical equality under identical versions; not cross-version compatibility.
    metadata = json.loads(
        c.model.with_name(c.model.stem + "_meta.json").read_text(encoding="utf-8")
    )
    assert metadata["feature_names_used"] == names
    assert metadata["trained_on_file"] == "synthetic_training_only.csv"
    assert metadata["spectro_only"] is False


def test_real_transformed_rows_distinguish_same_class_observations(
    bridge_preprocessing_oracle,
):
    observation = bridge_preprocessing_oracle
    c = observation.case
    expected = observation.expected(c.query)
    # The old query probabilities may coincide; these physical input markers do not.
    assert len(np.unique(expected, axis=0)) == len(c.query)
    observation.inputs.clear()
    result = c.run()
    assert result["_xgb_ok"]
    observation.compare(observation.inputs[0], expected)


@pytest.mark.parametrize("infinity", [np.inf, -np.inf])
def test_real_pipeline_explicitly_refuses_infinity(
    bridge_real_bundle, caplog, infinity
):
    c = bridge_real_bundle
    c.query.loc[2, "feature_alpha"] = infinity
    c.query.to_csv(c.csv, index=False)
    caplog.clear()
    try:
        result = c.run()
    except ValueError as exc:
        assert any(
            word in str(exc).lower() for word in ("inf", "infinity", "too large")
        )
        return
    assert result["_xgb_ok"] is False and result["has_proba"] is False
    assert result["y_pred"] is None
    messages = " ".join(r.getMessage() for r in caplog.records if r.levelno >= 40)
    assert any(word in messages.lower() for word in ("inf", "infinity", "too large"))
