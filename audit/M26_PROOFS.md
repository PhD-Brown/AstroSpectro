# M2.6 — executable red proof archive

Base/source18302e68cf25289873aa170a27834d99b516f376; bridge Git blob
eab849e4bb80be03a50ebc9128d2737530262358. Production unchanged.
Run explicitly: `python -m pytest -q tests/proofs_dimred_xgboost_bridge.py
--tb=short --junitxml=test-results/m26-proofs.xml`.
Default test collection does not discover proofs_*.py. No xfail/skip.

Windows/Python3.11.9, NumPy2.4.6/pandas3.1.1/sklearn1.9.1/XGBoost3.2.0.
Only local synthetic tmp_path CSVs/models; sockets blocked by pytest defaults.
Final source execution: **19FAIL /19PASS /0ERROR/SKIP/XFAIL**, exit1 expected.
An earlier development run had two fixture failures assigning infinity into
integer pandas columns; the fixture now explicitly casts to float. Those two
were not production defects. No scientific oracle was relaxed.

| Test (parameter family) | Count/result | Classification and independent expectation |
|---|---|---|
| filter_cannot_reassign_predictions_to_other_observations (snr/nan/multiple) | 3FAIL | NEW: real CSV filtering leaves9/9/7 observations but prefix truncation associates different IDs; source248–262. Relevant refusal or correctly paired output accepted. |
| each_input_cardinality_checked_independently (y/clusters,3/7) | 4FAIL | NEW: features=Z=5, only one label vector differs; success output has incompatible cardinality. Same identity boundary as preceding tests. |
| prediction_cardinality_cannot_report_success (0/3/7) | 3FAIL | ROBUSTNESS: injected pipeline output; no real backend cardinality violation established. Output-boundary acceptance criterion, not three demonstrated backend bugs. |
| invalid_probability_output_cannot_be_advertised_as_valid (short/long/nan/inf/negative/above_one/unnormalized) | 7FAIL | ROBUSTNESS: controlled estimator outputs reach actual bridge reduction without validation. Reject or mark unavailable; no decision-rule argmax constraint. Real Gaussian case below independently confirms a reachable nonfinite boundary. |
| real_fitted_imputer_is_not_silently_overridden (present_nan) | 1FAIL | NEW: actual saved named median/scaler/XGBoost pipeline predicts F at ID807; bridge replaces NaN by0 and predicts M. Median alpha1.225 learned from train only. Schema is otherwise identical; real inference, not a mock response. |
| same test (absent_column) | 1PASS | CONTROL:1/3 absent names exceeds30%, so no prediction. Does not approve zero imputation below cutoff. |
| real_degenerate_gaussian_probabilities_cannot_claim_success | 1FAIL | NEW: real supported GaussianNB(var_smoothing=0), zero within-class variance yields5NaN confidence values with has_proba=True and _xgb_ok=True. Local np.errstate hides only expected arithmetic warnings; results explicitly asserted. No historical impact claim. |
| identity_information_missing_from_api_is_observation (six scenarios) | 6PASS | OBSERVATION: independent same-size CSV/Z/y permutations, replaced/duplicate/absent CSV IDs accepted. API has no embedding IDs; not a requirement to infer unavailable identity. |
| nonfinite_preparation_is_characterization (four scenarios) | 4PASS | OBSERVATION: NaN/constant+NaN are zero-filled; both infinities reach spy unchanged. Real imputer pipeline rejects infinities in green integration tests. |
| degenerate_schema_and_population_observations (seven scenarios) | 7PASS | OBSERVATION: empty/duplicate needed names, parser-mangled duplicate CSV headers, strings, N0/N1 and nonconsecutive pandas index. Permissive spy is not real-estimator acceptance. |
| unrelated_feature_can_change_population_observation | 1PASS | OBSERVATION: an unused numeric candidate with1/10NaN removes a row, despite not entering model schema. Identity consequence belongs to filtering boundary; provenance policy remains #23. |

19FAIL are **three confirmed defect families**, plus output-boundary robustness
probes. They are not19bugs. #50 concerns loader joins; #73 HDBSCAN profiles;
#24 training transformations; #38 general persistence. Bridge-specific
identity/preparation/output failures need their own corrective issues after
deduplication. General provenance and optional facade remain known#23/#21.

## Limits of these oracles

The identity proofs allow a relevant ValueError or structured failure, or exact
retained observation values; they do not prescribe ID transport architecture.
The cardinality proofs protect shape consistency, not absent identity detection.
The imputation proof permits refusal or identical direct pipeline labels; it
discriminates this fixture's real median-vs-zero predictions but does not prove
general preprocessing equality if a different transform happens to preserve
the class. Future correction tests should compare the transformed values too.
Invalid-probability proofs permit rejection or unavailable status; silently
invented replacement probabilities are not authorized by this audit.
The Gaussian oracle only checks the advertised confidence boundary, not whether
a finite class chosen under undefined likelihoods is statistically meaningful.
Observations freeze current behavior only at this archival SHA, not on main.
