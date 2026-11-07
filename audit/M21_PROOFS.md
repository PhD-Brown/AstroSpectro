# M2.1 ? durable proof archive after independent review

This document and `tests/proofs_dimred_data_loader.py` belong to
`audit/43-dimred-proofs`, not the integration diff of PR #56. Checkout the fixed
archive SHA linked by the issue bodies to reproduce with matching fixtures.
Original versions remain in ancestor `1e35c8e8f0cc4aac44ad751729a7c0f54c08c00b`:
all 30 original cases and their fixture versions are retained in Git history.

The revised archive has **46 executed cases: 35 FAIL / 11 PASS / 0 ERROR / 0 SKIP**.
Counts are not bug counts. Positive controls and historical observations are
intentional PASS cases; the hygiene `.part` expectation intentionally fails but
is not a confirmed defect. N7's coherent-rollback remedy is proposed, not yet a
chosen transactional API. #55's validation policies were accepted by the owner
on 2026-11-07; message regexes demand a domain diagnostic rather than accepting
an incidental dependency error. Exact final exception wording is not prescribed.

Command (Windows/Python 3.11.9, pytest 8.4.2, offline):

```powershell
.venv/Scripts/python.exe -m pytest -q tests/proofs_dimred_data_loader.py --tb=line --junitxml=test-results/m21-review-proofs.xml
```

All 35 failures are assertion/expected-error mismatches, never fixture/collection
errors. No production source was edited. Initial development encountered an
annotation insertion SyntaxError; it was corrected before this archived run.

## Original-to-revised correspondence

- Labels, conflicting/NaN keys, stale scaler, invalid values/mode, balancing,
  provenance, infinite flux, invvar, ambiguous basenames and `.part`: retained
  with per-test classifications. Fixture names are prefixed consistently.
- Large-ID case: same rounding counterexample, now explicitly uses configurable
  `merge_key="source_id"`; requires BOTH a missing ID and IDs >2^53. No assertion
  about prevalence in ordinary LAMOST observation IDs.
- Old `empty_feature_selection_returns_empty_batch` and
  `empty_spectral_batch_has_declared_shapes`: superseded by domain-error tests
  under accepted #55; original return-empty expectations remain in history.
- Old `exactly_twenty_percent_missing_is_not_more_than_twenty`: superseded by
  `exactly_twenty_percent_missing_is_rejected` (PASS), also covered in green
  19/20/21% tests using a valid companion row so no empty batch is required.
- Old `degenerate_native_wavelength_is_rejected`: its target was also degenerate,
  so it did not demonstrate #52. Retained historically, retracted as a defect.
  Corrected independent increasing-target control is green
  `test_degenerate_native_grid_with_valid_target_does_not_prove_bug` in PR #56.
  Two nonincreasing-target rejection cases belong to accepted #55 instead.
- Added: identical catalogue duplicates (#50), all-files-missing domain error
  (#55), failed reload and its two actual state observations (#51), parent
  worker diagnostics (#57; serial positive control), and N2/N4/N5/N6/N8
  observations. No original scenario vanished without this disposition.

## Per-test classification and executed result

Test names below omit `test_`. Each source test also carries its own Refs and
classification docstring. PASS here does not promote an observation into a
scientific guarantee.

| Test | Follow-up / classification | PASS | FAIL |
|---|---|---:|---:|
| `catalog_labels_are_joined_exactly_once` | Refs #49 - classification: demonstrated defect. | 0 | 2 |
| `catalog_supplies_one_dimensional_labels` | Refs #49 - classification: demonstrated defect. | 0 | 1 |
| `conflicting_catalog_keys_are_rejected` | Refs #50 - classification: demonstrated defect; remedy proposed. | 0 | 1 |
| `missing_identity_cannot_match_another_missing_identity` | Refs #50 - classification: demonstrated defect. | 0 | 1 |
| `large_observation_ids_are_not_rounded_before_join` | Refs #50 - classification: demonstrated defect. | 0 | 1 |
| `unscaled_reload_does_not_reuse_previous_scaler` | Refs #51 - classification: demonstrated defect. | 0 | 1 |
| `empty_feature_selection_has_domain_error` | Refs #55 - classification: accepted validation contract (2026-11-07); not a production fix. | 0 | 2 |
| `no_usable_features_has_explicit_validation_error` | Refs #55 - classification: accepted validation contract (2026-11-07); not a production fix. | 0 | 1 |
| `nonfinite_features_are_rejected_without_scaling` | Refs #55 - classification: accepted validation contract (2026-11-07); not a production fix. | 0 | 2 |
| `unknown_mode_is_rejected` | Refs #55 - classification: accepted validation contract (2026-11-07); not a production fix. | 0 | 1 |
| `balancing_does_not_silently_drop_missing_labels` | Refs #54 - classification: demonstrated defect; remedy proposed. | 0 | 1 |
| `balance_log_reports_original_population` | Refs #54 - classification: demonstrated defect. | 0 | 1 |
| `spectro_only_excludes_known_gaia_proxies` | Refs #23 - classification: already known. | 0 | 1 |
| `spectro_only_rejects_unknown_provenance` | Refs #23 - classification: already known. | 0 | 1 |
| `empty_spectral_batch_has_domain_error` | Refs #55 - classification: accepted validation contract (2026-11-07); not a production fix. | 0 | 3 |
| `exactly_twenty_percent_missing_is_rejected` | Refs #55 - classification: accepted validation contract (2026-11-07); not a production fix. | 1 | 0 |
| `infinite_flux_does_not_become_extreme_valid_pixels` | Refs #52 - classification: demonstrated defect. | 0 | 2 |
| `zero_inverse_variance_pixels_are_not_trusted` | Refs #34 - classification: already known. | 0 | 1 |
| `duplicate_fits_basenames_do_not_silently_choose_one` | Refs #53 - classification: demonstrated defect. | 0 | 1 |
| `index_does_not_treat_partial_download_as_fits` | Refs none - classification: hygiene observation; no linked defect. | 0 | 1 |
| `target_grid_requires_finite_nonempty_vector` | Refs #55 - classification: accepted validation contract (2026-11-07); not a production fix. | 0 | 3 |
| `identical_catalog_duplicates_cannot_multiply_observations` | Refs #50 - classification: demonstrated defect. | 0 | 1 |
| `nonincreasing_target_grid_is_rejected` | Refs #55 - classification: accepted validation contract (2026-11-07); not a production fix. | 0 | 2 |
| `all_missing_fits_has_domain_error` | Refs #55 - classification: accepted validation contract (2026-11-07); not a production fix. | 0 | 1 |
| `failed_reload_does_not_leave_mixed_state` | Refs #51 - classification: demonstrated defect; remedy proposed. | 0 | 1 |
| `worker_read_errors_reach_parent_handlers` | Refs #57 - classification: demonstrated defect; serial control. | 1 | 2 |
| `zero_snr_threshold_historically_differs_between_loaders` | Refs #55 - classification: historical observation, not a scientific guarantee. | 1 | 0 |
| `csv_numeric_strings_are_inferred_as_features` | Refs D1/#23 - classification: historical observation, not a scientific guarantee. | 1 | 0 |
| `default_grid_pixel_count_is_historical` | Refs N5/N8 - classification: historical observation, not a scientific guarantee. | 1 | 0 |
| `uppercase_fits_index_is_platform_dependent` | Refs N5/N8 - classification: historical observation, not a scientific guarantee. | 1 | 0 |
| `failed_reload_state_historical_observation` | Refs #51 - classification: historical observation of mixed state (N7). | 2 | 0 |
| `invalid_target_current_behavior` | Refs #55 - classification: historical observation, not accepted validation (N4). | 3 | 0 |

## Mutation replay

`python audit/check_m21_mutations.py --output test-results/m21-mutations.json`
ran the baseline and 30 reconstructed mutations in disposable source copies:
**baseline 57 PASS; 29 killed / 1 survived; no collection/fixture errors or skips**.
These are not Claude's exact unpublished 31 candidates. Eight of the nine named
former survivors are now detected; keeping the last duplicate basename remains
an expected green-suite survivor until #53 is corrected. A survivor is not a
scientific validity result. The full script records input hashes and failing
node names in its local JSON output. Production files were untouched.

| Reconstruction | Result | Failing cases |
|---|---|---:|
| `spectral_classes_ignored` | KILLED | 1 |
| `spectral_snr_ignored` | KILLED | 2 |
| `sampling_head` | KILLED | 1 |
| `median_native` | KILLED | 1 |
| `threshold_11` | KILLED | 1 |
| `threshold_30` | KILLED | 2 |
| `threshold_inclusive` | KILLED | 1 |
| `index_keep_last` | SURVIVED | 0 |
| `rejection_log_debug` | KILLED | 2 |
| `features_snr_strict` | KILLED | 4 |
| `spectral_snr_strict` | KILLED | 2 |
| `inner_join` | KILLED | 1 |
| `column_nan_inclusive` | KILLED | 3 |
| `balance_with_replacement` | KILLED | 1 |
| `meta_index_not_reset` | KILLED | 1 |
| `dropna_all_columns` | KILLED | 3 |
| `inverse_identity` | KILLED | 1 |
| `wavelength_one_pixel_shift` | KILLED | 20 |
| `nearest_interpolation` | KILLED | 5 |
| `extrapolate` | KILLED | 6 |
| `nonzero_median` | KILLED | 1 |
| `wrong_flux_row` | KILLED | 23 |
| `nan_fill_removed` | KILLED | 3 |
| `catalog_mask_shifted` | KILLED | 12 |
| `cache_ignored` | KILLED | 1 |
| `spectro_prefix_ignored` | KILLED | 1 |
| `scaler_no_std` | KILLED | 1 |
| `feature_order_reversed` | KILLED | 8 |
| `read_warning_debug` | KILLED | 5 |
| `boolean_not_numeric` | KILLED | 1 |
