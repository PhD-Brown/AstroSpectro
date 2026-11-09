# M2.6 — additional proofs after Claude review R1–R7

2026-11-09, Refs #43/#88/#89/#90. Production bridge unchanged, blob
eab849e4bb80be03a50ebc9128d2737530262358 (352 lines). Old proof refs
0da758111daaf7ba33283c1462ec1fb1368c6cff and
7b32abbd1881a487c0ea1bcc7355883e5797ea82 remain immutable ancestors.

This supplementary archive contains R1–R3 actual-loader, preprocessing and
output proofs. Default collection excludes proofs_*.py; execute explicitly.

`python -m pytest -q tests/proofs_dimred_xgboost_bridge_review.py --tb=short`:
**13 FAIL /3 PASS /0 ERROR/SKIP**, 16 cases. S1/default and S2/NB01 reproduce four/nine
and five/nine incorrect associations respectively; S3 has loader10/bridge9.
The real loader's final population and X values are checked against manually
retained IDs; they supply Z/y/clusters. The bridge's actual estimator frame is
then compared row-by-row. The shared-population control succeeds. Initial R1
checkpoint3a5690f had only these first4 cases (3FAIL/1PASS); its scoped controls
are retained separately.

| Proof family | Current source result | Independent invariant |
|---|---|---|
| Actual loader S1/S2/S3 | 3FAIL | Final true loader population supplies embedding identities; bridge may refuse or return precisely those values. |
| Shared true loader population | 1PASS | Correct finite input must succeed; blanket refusal is not accepted. |
| Missing beta/gamma, saved real preprocessing | 2FAIL | Train medians5/5 differ from means4/29/6; actual booster matrix equals manual train-only impute/standardize. Learned state cannot change or be refit. |
| Same alpha-missing ID807, batch5/batch10 | 2FAIL | A finite success must have correctly paired transformed rows/IDs/labels. Missing-row retention versus justified exclusion remains open; finite companion observations cannot disappear. |
| Input y/clusters3/7 versus5 rows | 4FAIL | Exact supplied values and all cardinalities, not merely five returned elements. Cropping7→5 is insufficient. |
| Real zero-variance GaussianNB | 1FAIL | Undefined original probabilities cannot become available confidence, even after nan_to_num. Unavailability or explicit diagnosed refusal admitted. |
| Injected rank3 probabilities | 1FAIL | Unavailability/refusal required; flattening to a scalar confidence is insufficient. |
| Valid vectors/probabilities; finite learned preprocessing | 2PASS | Correct nominal outputs remain accepted. |

The passive observer calls the actual XGBoost predictor unchanged. The expected
matrix uses sorted training medians, math.fsum means and population variances,
independent of sklearn transform output. Fit is forbidden after the fixture's
training. Loaded imputer/scaler state is checked before/after inference.
Float64 arithmetic uses a2e-14 absolute bound for30-row calculations; alternate
floating representations allow8dtype epsilons. Beta at ID807 should be
(5−4)/sqrt(17.6)=0.2383656473, versus zero-fill−0.9534625892; gamma should be
0.0819231921, versus zero-fill−2.3757725695. Means, zero-after-scaling and
inference-batch medians are discriminated even if predicted labels coincide.

For batch5 the missing target is sent as zero; for batch10 the target is
removed but its embedding ID remains in the prefix, and a finite last companion
ID disappears. The oracle does not demand one global missing-row policy; it
demands a correct association under whichever supported policy succeeds.

The caller has not violated an identity precondition in S1/S2: the bridge
introduces the difference. Only an unrelated external permutation remains an
unobservable caller violation. Correct reconciliation or relevant explicit
refusal is allowed; no production architecture is prescribed.

Refusal checking now requires a domain-specific diagnostic (identity/schema/
preprocessing/probability) instead of generic words such as row/feature. A
structured failure additionally needs an error log and no successful prediction
payload. A message alone cannot prove the correctness of arbitrary future
validation; paired positive controls and explicit forbidden-remedy checks are
also necessary. No new exception class is required.
