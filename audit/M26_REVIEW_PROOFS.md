# M2.6 — additional proofs after Claude review R1–R7

2026-11-09, Refs #43/#88/#89/#90. Production bridge unchanged, blob
eab849e4bb80be03a50ebc9128d2737530262358 (352 lines). Old proof refs
0da758111daaf7ba33283c1462ec1fb1368c6cff and
7b32abbd1881a487c0ea1bcc7355883e5797ea82 remain immutable ancestors.

This new archive starts with the R1 actual-loader oracle. Final supplementary
proof results and the final archive SHA will be recorded after R2/R3 additions.
Default collection excludes proofs_*.py; execute explicitly.

`python -m pytest -q tests/proofs_dimred_xgboost_bridge_review.py --tb=short`:
**3 FAIL /1 PASS**. S1/default and S2/NB01 independently reproduce four/nine
and five/nine incorrect associations respectively; S3 has loader10/bridge9.
The real loader's final population and X values are checked against manually
retained IDs; they supply Z/y/clusters. The bridge's actual estimator frame is
then compared row-by-row. The shared-population control succeeds.

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
