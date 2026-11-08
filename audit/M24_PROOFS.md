# M2.4 — archived HDBSCAN proofs and observations

2026-11-08 · Refs #43 · audit only, no production correction.

Run `python -m pytest -q tests/proofs_dimred_hdbscan_analyzer.py --tb=short`
from this archive, using the project dependencies and blocked sockets. The
filename is outside normal pytest discovery. No xfail/skip masks a defect.
Optional real observations use the same file with `-m slow -W error` and a
separately installed hdbscan target. The integration branch removes this file,
this manifest and its JSON only after their pushed commit is read back.

Reference: `main@58180392d16e06ad411073bc48011d076f9128f0`, production blob
`667a2985ea1b6b4f0a660a70f7635715730c0f78` (562 lines). Source is unchanged.
Fixtures: `tests/m24_support.py`, registered by `tests/conftest.py`; actual
loader constants are used without importing the optional Torch facade (#21).
All CSVs/coordinates/metadata are synthetic; no scientific files are edited.

## Executed ledger and classification

Windows AMD64, Python 3.11.9, pytest8.4.2, numpy2.4.6, pandas3.1.1,
scipy1.17.1, sklearn1.9.1; optional hdbscan0.8.44. Exact per-case failure
messages, source/test hashes and version context are in `m24_proof_results.json`.
Ordinary: **34 cases, 10 FAIL / 24 PASS / 0 ERROR/SKIP/XFAIL**, 2 optional
cases deselected. Optional: **2 PASS**; its earlier run deselected the then33
ordinary cases. Final committed rerun is recorded in the integration matrix.
Counts describe cases, not numbers of bugs. All ten failing causes were read.

| Test (prefix `test_` omitted) | Cases/result | Classification and independent expectation |
|---|---|---|
| `failed_fit_state_is_coherent` | 4 FAIL (normal/presentation × initial/refit) | NEW defect. Estimator/cache/counters/palette must describe one fit. Full retention or full invalidation permitted. A first failed fit must not expose an unfitted estimator as current. Current code replaces estimator before success but retains old labels/statistics on refit. No transaction policy chosen. |
| `feature_filter_cannot_reassign_labels` | 2 FAIL (SNR/dropna) | NEW alignment defect. IDs100…109, remove middle104; expected100,101,102,103,105…109, actual100…108. Refusal with relevant ValueError or correct reconciliation allowed. Code cannot reconstruct IDs from current interface; silent truncation is not alignment. This is an intentionally incompatible population that the advertised alignment check fails to reject, not a claim that arbitrary preconditions become valid. |
| `feature_labels_length_checked_independently` | 2 FAIL (8/12 labels vs10 Z rows,8 feature rows) | Same alignment/validation defect. Both mismatches succeed after slicing; 8 or12 labels cannot describe10 observations. Explicit length/identity diagnostic allowed; no architectural remedy selected. |
| `no_available_physical_columns_has_explicit_outcome` | 1 FAIL | NEW robustness defect. Documented restriction to available columns reaches internal pandas `No objects to concatenate`. Counts-only output or explicit diagnostic about unavailable physics/columns both allowed. Policy is open. |
| `all_missing_classes_have_explicit_outcome` | 1 FAIL | NEW robustness defect. Dominant class is undefined and `value_counts().index[0]` raises IndexError. Explicit class diagnostic or missing/placeholder class permitted, no tie/missing policy selected. |
| `equal_length_permuted_csv_is_not_identity_proof` | 1 PASS | OPEN architectural limitation/precondition violation. Equal-length CSV102,100,101 with labels for100,101,102 assigns positive standardized marker to ID100's label although its correct score is negative. API receives no embedding IDs. No green promise to detect this. |
| `physical_equal_lengths_cannot_certify_identity` | 3 PASS (meta/y/labels separately permuted) | OPEN positional precondition. Historic pandas indices alone do not establish identity; paired permutations are green, independent permutations produce changed scientific associations silently. |
| `returned_features_are_raw_despite_standardized_docstring` | 1 PASS | NEW documentation inconsistency, independently observed raw10,20,30 vs standardized means. Correct API choice remains open: document raw output or change it deliberately in a corrective PR. Green tests permit either. |
| `centroid_undefined_domain_observation` | 3 PASS | OPEN empty mask→NaN/warning; 2D Boolean mask→scalar; Inf→Inf. No mandatory domain/exception policy invented. Valid 1D masks have analytic green oracles. |
| `profile_open_policy_observations` | 7 PASS | OPEN tie, empty phys_cols fallback, duplicate phys_cols TypeError, duplicate obsids positional acceptance, strings TypeError, Inf→inf/nan, nullable-all-missing display. These observations preserve limitations, not future mandatory behavior. |
| `sensitivity_zero_length_observation` | 1 PASS | OPEN defensive wrapper domain, artificial empty backend result→ZeroDivisionError. Real backend rejects N=0; no claim this is reachable via a successful real HDBSCAN fit. |
| `algorithmic_noise_does_not_use_physical_measurements` | 1 PASS | NEW methodological/textual overclaim. Noise unconditionally receives an astrophysical anomaly label without spectra/errors/SNR/anomaly diagnostic. No independently established physical classification follows from -1. |
| `heterogeneous_temperatures_emit_an_unobserved_average_type` | 1 PASS | Same interpretation limitation:4000K and8000K emit a solar-type group description from their6000K mean, although neither observation has that temperature. This demonstrates averaging, not a physical-class falsification with independently classified real stars. |
| `current_heuristic_display_observation` | 6 PASS | Current threshold fidelity at/around4000,5500,7500K,2.5/3.5dex,-0.8/-0.3dex. These strings are descriptive observations in the archive, not astrophysical requirements in green CI. |
| `real_failed_refit_state_observation` (slow) | 2 PASS | Real hdbscan0.8.44 confirms fresh estimator + stale labels after a single-row refit failure in each mode. Normal mode also rejected NaN in the initial probe; presentation without prediction_data can map a NaN row to noise, so NaN is NOT used as its failure precondition. |

## Validity and limits

The ordinary backend double proves wrapper boundary/state behavior, not HDBSCAN.
The optional cases confirm the state observation against the installed backend.
No population of real stars, historical incidence or scientific severity P0 is
inferred. Feature filtering follows the actual hardcoded SNR>=10 and missing
fraction<=10%; valid equal-sized prefiltered inputs have separate green oracles.
Equal sizes, `min(N_features,N_umap)` and old pandas indices cannot certify IDs.

`audit/check_m24_proof_controls.py --output <json>` executes the59-case green
suite alongside selected proofs on disposable candidates. Full coherent
retention and full invalidation preserving configuration each PASS63cases;
partial invalidation leaves4proofs FAIL and the59green cases PASS, as required.
Explicit alignment refusal PASS63cases, missing-physics diagnostic PASS60cases,
missing-class placeholder PASS60cases. All6oracle checks PASS, integrity/index
unchanged. Results are in `m24_proof_control_results.json`. These are candidate
compatibility checks, not approved corrections. No source file in the real
checkout is changed. Snapshot arrays/lists/maps are copied by the state oracle.

Missing classes and missing physics are distinct failure causes; none specifies
a preferred missing-value policy. Duplicate physical-column semantics, Boolean
feature eligibility, custom metrics, empty grid schema, data ownership, centroid
nonfinite handling and sensitivity multiplicity remain open where documented.
No actual algorithmic monotonicity, probability normalization or physical
interpretation is imposed.

Primary sources read: [HDBSCAN API](https://hdbscan.readthedocs.io/en/latest/api.html)
(membership strength, noise=-1, optional prediction structures),
[Gaia GSP-Phot methods](https://gea.esac.esa.int/archive/documentation/GDR3/Data_analysis/chap_cu8par/sec_cu8par_apsis/ssec_cu8par_apsis_gspphot.html)
(global metallicity and method limitations), and
[UMAP clustering cautions](https://umap-learn.readthedocs.io/en/latest/clustering.html).
The Gaia source datamodel calls mh_gspphot iron abundance whereas the methods
describe [M/H]. That terminology discrepancy is reported; it is not resolved
by silently relabeling data. HDBSCAN noise is not a spectral anomaly diagnosis.
