# M2.2 PCA - revised proof archive after Claude R1-R4

2026-11-08; Refs #43 / PR #64. Production is unchanged from
`main@3604b3db8a66f19b07569f3c0aafb7f256734e77`.
The durable ref is **audit/43-dimred-pca-proofs**. The integration matrix,
PR and #59/#63 link the fixed SHA of this revised commit.
Matching conftest, producer source, pytest config and PCA source are in this tree.

## Immutable history

Initial archive **a9c84a647d1cb1d35b3ee97e65ff099c560fc2ec** remains an
ancestor, not rewritten: **21 cases, 16 FAIL / 5 PASS**. Its #59 assertion
allowed an insufficient capped count and #63 required a capped table. Those
assertions are superseded below, not silently treated as definitive contracts.
Other original cases are retained, with the same scope/classification.
The initial K=0 inverse-transform assumption was already withdrawn there.

## Execution of the revised archive

From a checkout of this ref, use the installed Python 3.11 environment:

```powershell
python -m pytest -q tests/proofs_dimred_pca_analyzer.py --tb=line --junitxml=test-results/m22-review-proofs.xml
```

Windows/Python 3.11.9, numpy 2.4.6, pandas 3.1.1, scipy 1.17.1,
sklearn 1.9.1, joblib 1.6.0, pytest 8.4.2, pytest-socket 0.7.0.
**41 cases: 27 FAIL / 14 PASS / 0 ERROR / 0 SKIP / 0 XFAIL**, exit 1.
JUnit failure messages were reviewed: no collection/fixture failure is evidence.
In drafting, an observation mistakenly used default K=50 for a 3D input; its
setup was fixed to K=3 before this snapshot. That failure is not counted as a bug.
Only synthetic arrays and trusted locally generated tmp_path payloads are used.

## Revised proof dispositions

Test names omit test_. FAIL totals are not bug totals. OPEN assertions are
proposals awaiting decisions, not accepted API contracts or CI tests.

| Cases | Result | Classification and oracle |
|---|---|---|
| unreachable_variance_cannot_return_nonexistent_component | FAIL | Confirmed #59 mechanism A: K=1 retains 9/14 < 0.9. ValueError or a missing sentinel (None/NaN) is accepted; an integer must exist AND actually attain the requested fraction. Capping at K does not pass. |
| full_rank_target_one_roundoff | FAIL | Confirmed #59 mechanism B: Claude's integer 5x3 example independently has centered rank 3 and positive covariance eigenvalues. In full solver on this PC, cumulative ratio is 0.9999999999999999 and target 1 returns 4. An integer answer must be K=3 by this independent full-rank oracle; explicit diagnostic/missing sentinel also accepted. No universal epsilon selected. Other BLAS/solvers may legitimately return 3 already. |
| variance_oracle_rejects_capping_but_accepts_explicit_diagnostics | PASS | Oracle control: reachable 0.5 passes, wrapping the actual current answer in min(K, answer) is rejected for 0.9. Explicit ValueError/None/NaN responses pass. |
| observe_nan_variance_threshold | PASS | OBSERVATION / OPEN: NaN currently returns K+1. Domain policy still undecided; not a green contract. |
| loaded_configuration_is_complete; loaded_model_can_refit | 2 FAIL | Same confirmed #60 missing random_state/solver. Requested vs effective n_components remains OPEN. |
| mixed_numeric_metadata_correlations[bool/Int64/Float64] | 3 FAIL | Confirmed #61 object block from mixed numeric dtypes; boolean inclusion remains OPEN. Separate from the ordinary float64+int64 green contract in R4. |
| failed_fit_does_not_expose_an_unfitted_estimator_as_fitted[False/True] | 2 FAIL | Confirmed #62. Reject access or retain a genuinely fitted old estimator; restore vs invalidate policy not chosen. |
| default_class_separation_on_two_component_model | FAIL | Confirmed #63 internal IndexError. ValueError validation is accepted; a returned table must have the six exact (class,PC1/PC2) rows, six statistics columns, counts, signed analytic means/population std/medians/quartiles. No clamp policy imposed. |
| class_oracle_accepts_both_remedies_and_checks_statistics | PASS | Oracle control: real valid n_pcs=2 calculation passes, table with a wrong mean fails, ValueError diagnostic passes. |
| observe_missing_class_labels_empty_quantile | PASS | OBSERVATION / OPEN: float labels [1,1,2,2,NaN,NaN] create an empty NaN-class mask and percentile IndexError. Rejection vs announced exclusion not chosen. |
| non_tio_producer_features_not_assigned_to_titanium[name] | 10 FAIL | Confirmed R2 accidental case-insensitive TiO substring, **one mechanism**, corrective issue DRAFT ONLY. AST verifies exact assigned producer keys. Oracle only excludes the molecular/TiO family for these non-TiO quantities; no new destination family is prescribed. |
| observe_open_lick_priorities_and_constant_input | PASS | OBSERVATION / OPEN: hBETA_index -> Balmer, TiO_1_Lick -> Titane, constant fit currently accepted with nonfinite variance fractions. No warning text frozen; these outcomes no longer bind green tests. |
| observe_feature_name_container_contract[array/index/empty] | 3 PASS | N3 / C3 observations: ndarray/Index names raise ambiguous-truth ValueError, empty list falls back to f0..f2. Common inputs but array/Index outside List[str] annotation; policy not selected. |
| observe_negative_count_labels | PASS | N4 / C1 observation: reconstruction curve labels -1 but computes K-1 error; correlation n_pcs=-1 silently omits last PC. These labels are misleading; policy/remedy/priority await scientific decision. |
| reconstruction_rejects_unavailable_component_counts[-1/4] | 2 FAIL | OPEN C1 proposed validation; current slicing semantics retained as evidence, not approved. |
| empty_reconstruction_curve_retains_schema | FAIL | OPEN C2 schema vs rejection. |
| unfitted_default_curve_has_domain_guard | FAIL | Confirmed #62 diagnostic dereference before guard, no numerical corruption claim. |
| feature_name_count_is_validated_at_fit | FAIL | OPEN C3 early validation; mismatch currently fails later. |
| training_cache_not_aliased_to_caller; names_not_aliased_to_caller | 2 FAIL | OPEN C4 snapshot vs documented caller immutability. |
| observe_positional_alignment_is_not_identity_alignment | PASS | NOT A BUG under documented same-order precondition: unpaired metadata reversal changes correlation sign. |
| observe_loaded_implicit_correlations_need_training_data | PASS | Documented limitation: no X_fit after load; supplied scores work. |
| observe_negative_indices_duplicate_names_and_zero_components | PASS | OPEN observations: negative PC labeled PC0; duplicate names retained; zero-width scores rejected by sklearn inverse_transform. |
| observe_incompatible_metadata_and_labels_fail | PASS | Bad lengths fail loudly, no domain error policy certified. |
| observe_controlled_unfitted_and_wrong_payloads | PASS | Current raw errors from trusted malformed/unfitted payloads are observations, not future validation contracts. |

The ten R2 keys are:
feature_ratio_Fe_Mg, feature_ratio_Ca_Fe, feature_ratio_Ba_Fe,
feature_ratio_Sr_Fe, feature_flux_ratio_blue_red, feature_rotation_index,
feature_rotation_proxy, ratio_EW_Ha_Hb, ratio_EW_MgB_NaD, ratio_FWHM_Ha_Hb.
Earlier regex matches protect other ratio names (e.g. ratio_EW_CaK_CaH);
do not attribute every taxonomy ambiguity to TiO. No historical impact is
quantified. Lick priority, case sensitivity and abundance-ratio family remain OPEN.

## Green tests and mutation boundary

Revised green suite: 79 cases, with int64 Pearson/Spearman oracle, default
correlation count 10 at K=11 and default class count 5 at K=6.
Zero-variance rejection and schema ValueError are compatible; Lick precedence
examples are archived observations. The replay removes the former no-op,
separates real mutations from six candidate-remedy compatibility controls,
and never edits the working-tree production module.

Proofs remain outside automatic discovery/CI by filename. No xfail, production
patch, new corrective PR or R2 issue creation. Fixed initial SHA is preserved;
the integration child removes only archived files after remote verification.

