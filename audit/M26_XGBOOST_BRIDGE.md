# M2.6 — XGBoostBridge scientific and technical audit

2026-11-09 · Refs #43 — revised for the second independent Claude review.
Tests and audit only. Independent Claude review and human merge decision required.

## Revision after Claude's first review R1–R7

Read the complete [review 5831007444](https://github.com/PhD-Brown/AstroSpectro/pull/91#pullrequestreview-5831007444),
all seven inline threads, [Claude's handoff](https://github.com/PhD-Brown/AstroSpectro/issues/43#issuecomment-6748623479),
live AGENTS.md v1.3 and #43. Start: clean M26 worktree, local/remote
`1d6b257f923b02e7800bd1f5dcb747ef512e8b3d`, main/source still
`18302e68cf25289873aa170a27834d99b516f376`, bridge blob unchanged.
Same branch and Draft PR #91. No production/notebook/dependency/CI/data edit,
new corrective issue, merge, thread resolution or M2.7 work.

This dated section **supersedes** the original claims about caller identity
preconditions, zero placeholders, dtype preservation, red-oracle completeness
and real-test permutation detection. The [initial report at 1d6b257](https://github.com/PhD-Brown/AstroSpectro/blob/1d6b257f923b02e7800bd1f5dcb747ef512e8b3d/audit/M26_XGBOOST_BRIDGE.md)
and every historical JSON remain available unchanged. The old report's354-line
count was incorrect: the unchanged blob has **352 lines**.

| Review | Correction and independent evidence | Status / remaining limit |
|---|---|---|
| R1 | Real DimRedDataLoader supplies final population, actual X/y/meta and identity markers. S1/default: loader9/bridge9, four wrong associations; S2/NB01:9/9, five wrong associations; S3:10/9. | AP these demonstrated #88 failures. They are caused by independent bridge re-filtering despite a correct caller. Shared-population positive control passes. Transport of population/IDs versus explicit refusal OPEN. |
| R2 | Missing-schema green cases preserve exact present values/names/order but allow historical zero, NaN to fitted pipeline, or diagnosed refusal below cutoff. The existing >30% refusal is a prudent historical bound, not approval of imputation. | Complete NaN passthrough and explicit missing-schema refusal each71PASS. No train statistics exist on the unit spy, so the placeholder observation is not a claim of statistical equivalence. |
| R2 numerical oracle | Passive observation of actual loaded XGBoost input; manual sorted train medians, fsum means and population variance; loaded imputer/scaler statistics checked; fit forbidden after training. Beta median5/mean4 yields+0.2383656473, not zero-fill−0.9534625892; gamma median5/mean29/6 yields+0.0819231921, not zero-fill−2.3757725695. | Two new red matrix-value proofs, not label comparisons. Both reject train mean/post-scale zero, inference median and fabricated7; missing-value passthrough or explicit refusal accepted. No astrophysical interpretation. |
| R2 batch dependence | Same alpha-missing ID807 in batch5/10. At20% it reaches zero fill; at10% it is dropped while its embedding ID remains and a finite companion disappears. | Two red proofs protect association under supported retention or exclusion, without selecting a global NaN policy. #88/#89 qualifications published. |
| R3 | New Gaussian/tensor oracles require unavailable probabilities or explicit refusal, not a finite/scalar invented confidence. Input cardinality compares complete supplied y/clusters too, so7→5 silent cropping remains FAIL. | All three forbidden remedy families still rejected. Proper input guards/output availability pass their respective proofs and positive controls. Refusal helper uses domain diagnostics; unrelated row-buffer ValueError is rejected by three green cases. A diagnostic alone cannot prove an arbitrary validator correct. |
| R4 | Initial campaign had a62-case positive baseline, followed by each recorded selected test function; only some controls ran all62. Its SURVIVED verdict did not establish full-suite survival. | Historical JSON not relabeled. New runner records exact selections/collection, site execution and assertion causes. M04/M08/M09 and fillna7 survive all71 green cases and fail relevant red proofs. M10 now fails both artifact-family tests. No repeated old campaign or artificial score. |
| R5 | Feature-frame comparisons use check_dtype=False/check_exact=True; input-immutability comparisons remain strict. Numeric strings/booleans can be preserved, converted without changing numeric meaning, or explicitly refused. | Full71 cases accept float32/float64 and explicit string refusal. Real transformed comparisons allow8dtype epsilons, minimum absolute2e-14 for30-row float64 arithmetic. This does not guarantee safety for arbitrary large IDs or decision-boundary roundoff. |
| R6 | Both complete confidence vectors (historical row maximum, or probability of predicted class) are admissible. Diagnostic propagation and structured refusal are admissible after load. | Both alternative controls71PASS. Human decides confidence semantics and error policy; no production change/new issue. Unrelated model failures must not be mislabeled as scientific validation. |
| R7 | Real estimator receives a passive predict wrapper calling the original predictor unchanged. Each transformed row is compared to a manual external oracle and rows are distinct even where probabilities/labels coincide. | Row0↔3 permutation:37FAIL/34PASS across71, and3FAIL/2PASS in real-only5 cases. All failures are relevant assertions; old claim of real row-value protection depended on spies. General model/runtime/astrophysical validation remains unproved. |

### Additional immutable archive and remedy checks

New authoritative [archive fe25d1a39e83f46b743e5524cf5b70d2169d6183](https://github.com/PhD-Brown/AstroSpectro/tree/fe25d1a39e83f46b743e5524cf5b70d2169d6183)
contains [16 supplementary cases](https://github.com/PhD-Brown/AstroSpectro/blob/fe25d1a39e83f46b743e5524cf5b70d2169d6183/tests/proofs_dimred_xgboost_bridge_review.py),
matching fixtures and [per-family manifest](https://github.com/PhD-Brown/AstroSpectro/blob/fe25d1a39e83f46b743e5524cf5b70d2169d6183/audit/M26_REVIEW_PROOFS.md):
**13FAIL /3PASS /0ERROR/SKIP** on unchanged production, versus **16PASS**
under a combined admissible synthetic control. This control assumes the
fixture's explicit identity marker in Z[:,0]; it is not a proposed architecture
for arbitrary UMAP coordinates. Both older archives0da7581/7b32abb remain
immutable ancestors; the new proof was pushed, remote-verified and read through
GitHub before removal from the green head. SHA7777550 retains the same proof
bytes; fe25d1a additionally records the reviewed runner/initial results.

Commands from the archived checkout use the existing project Python:

```
python -m pytest -q tests/proofs_dimred_xgboost_bridge_review.py -W error --tb=short
python audit/check_m26_review_mutations.py --phase all_proofs --output test-results/review-proofs.json
```

From the integration head the runner accepts `--proofs-ref fe25d1a39e83f46b743e5524cf5b70d2169d6183`;
it records that file origin explicitly. Running from the archived checkout is
the reference for its matched source/tests/fixtures. No xfail/skip hides a proof.

| Recorded campaign / source snapshot | Scope and actual result |
|---|---|
| [R1](m26_review_r1_results.json), clean3a5690f | Four actual-loader cases. Baseline3FAIL/1PASS; length-only refusal2FAIL/2PASS (S1/S2 still wrong); synthetic identity refusal and shared population4PASS each. |
| [R2 initial](m26_review_r2_initial_results.json), clean7777550 | Three selected preprocessing cases. Baseline, inference median, wrong train mean and fabricated7 each2FAIL/1PASS; present-NaN/refusal controls3PASS. One control's redundant absent-column patch was unexercised, so runner correctly reported an unmet scope expectation despite passing tests. |
| [R2 scoped follow-up](m26_review_r2_followup_results.json), cleanfe25d1a | Same baseline2FAIL/1PASS; remove the irrelevant unexercised patch from the present-NaN control:3PASS, all changed sites executed. Full absent-column passthrough is independently exercised by the71-case green control. |
| [R3](m26_review_r3_results.json), clean7777550 | Seven cases: baseline6FAIL/1PASS. nan_to_num, silent cropping and tensor flattening each remain6FAIL/1PASS. Input guard2FAIL/5PASS, output guard4FAIL/3PASS, both guards7PASS. |
| [Full green initial](m26_review_green_initial_results.json), clean7777550 | All71 cases per main candidate. NaN passthrough, missing-schema refusal, float32/64, string refusal and alternative confidence each71PASS. M04/M08/M09/fillna7 each71PASS, declared GREEN survivors. M10=2FAIL/69PASS; row swap37FAIL/34PASS. Real-only baseline5PASS, same-class swap3FAIL/2PASS. |
| [Green follow-up](m26_review_green_followup_results.json), cleanfe25d1a | Baseline71PASS; conditional structured-refusal control71PASS. Original broad wrapper had3FAIL because it swallowed unrelated row-buffer errors and mislabeled them; tests were not relaxed. Only this control and its positive baseline were replayed. |
| [All-proofs initial](m26_review_proofs_initial_results.json) / [follow-up](m26_review_proof_results.json) | 7777550 thenfe25d1a:13FAIL/3PASS baseline, combined control16PASS in both. Initial control had the same redundant unexecuted absent-column patch; scoped follow-up removes it, all changed sites execute and expectation is met. |

All records contain exact patches, pytest commands, assertions, source/test
hashes, Git blobs, HEAD and dirty state, and actual line traces. AST, identical
selected collection, no import/fixture errors and positive controls are checked.
The initial records remain unchanged, including unsuccessful harness/control
qualifications. A red proof FAIL total is not a count of production bugs.
The old25-mutant campaign was not rerun. The four Claude survivors are now
qualified individually: M04→#89, M08/M09→stronger red oracles, M10→fixed test gap.

### R6 scientific decisions deliberately left open

Historical confidence is max(proba), which need not describe the displayed
decision class. In the asymmetric example displayed M can have P(M)=0.06 while
the maximum is0.9 for another class. The alternative P(predicted class) is
admissible too; tests require one coherent vector policy, not a mixture selected
per row. Neither quantity is astrophysically calibrated. A visualizer must know
which quantity it displays and should not infer probability availability from
the sentinel value1. This is documentation of a consumer boundary, not M2.7 work.

The bridge docstring promises structured failures but CSV/predict/decode paths
currently propagate some errors. A useful exception or a diagnosed structured
refusal can both be scientifically defensible. Tests now admit both, including
infinities, without admitting silent success. Policy choice, population/ID
transport, missing-feature handling and real-catalogue impact remain human
decisions. General provenance, leakage and calibration remain NA; facade
without Torch remains B/#21. New tests protect precise numeric/identity examples,
not scientific validity of the entire system.

### Final verification of the R1–R7 revision

[Exact validation ledger](m26_review_validation_results.json): clean input
snapshot **15f5cc5e90346b6aa4dfc9481ae7b9695455d7df**, production source still
18302e68 and bridge blobeab849e4. Actual commands, OS/Python/package versions,
effective LF hashes, Git blobs and stdout/stderr are recorded. Later evidence
commits do not retroactively become tested HEADs. Windows/Python3.11.9,
NumPy2.4.6, pandas3.1.1, sklearn1.9.1, XGBoost3.2.0; no dependencies changed.

| Actual command / check | Observed result |
|---|---|
| `python -m pytest -q tests/test_dimred_xgboost_bridge.py tests/test_dimred_xgboost_bridge_real.py -W error` | **71PASS**,66 cases in wrapper/loader file and5 real XGBoost CPU cases; no skips/warnings |
| `python -m pytest -q` | **487PASS /62deselected /7 existing Gaia warnings** |
| `python -m pytest -q --import-mode=importlib` | **487PASS /62deselected /7 existing Gaia warnings** |
| Same71 cases in reversed collection order, -W error | **71PASS**; sys.path and four temporary import entries restored by identity |
| Black/Ruff on src/tests/scripts/app.py and both M26 audit runners | PASS; no reformatting of production |
| `python -m pip check`, `git diff --check`, src/config/notebook diff against base | PASS; no production/config/notebook differences |
| Reachability of0da7581,7b32abb,fe25d1a; effective hashes of review campaign inputs | All ancestors; all hashes match their recorded committed snapshots |
| New runner after proof removal, --proofs-ref fe25d1a --phase all_proofs --only all_proof_baseline | **13FAIL/3PASS**, expected; provenance records the archived file origin, not a nonexistent HEAD file |
| New runner from green head, --phase green --only green_baseline | **71PASS**, no archived proof needed for the green selection |

The62exclusions still follow the existing slow/network markers; all5 real
M2.6 CPU cases run in the default suite. The7warnings are the pre-existing
Gaia pandas-inplace deprecation. Source-direct fixtures still do not validate
the optional public facade (#21, already reproduced separately). No new claim
of scientific calibration, astrophysical correctness or measured coverage.
Hosted CI for the final pushed SHA is verified separately in the updated
PR91 description and single #43 handoff. Seven inline replies reference the
review evidence; their threads remain open for Claude. Next action: second
independent review, then human decision. PR stays Draft; M2.6 is not integrated.

## Reference and method

Base/source `18302e68cf25289873aa170a27834d99b516f376`, bridge blob
`eab849e4bb80be03a50ebc9128d2737530262358` (entire source read).
Branch `audit/43-dimred-xgboost-bridge`, separate `AstroSpectro-m26` worktree.
Main checkout and M24/M25 worktrees inspected clean; main checkout remains
`795af2c`. Remote main verified by fetch and ls-remote. PR56/64/72/79/87 merges
and PR87 CI42548715421 success independently checked through GitHub.
No pre-existing dedicated bridge tests found. Earlier matrices M21–M25 are
context, not bridge evidence. Live AGENTS.md v1.3, complete #43 body, comment
history and latest handoffs, all issue bodies and comments for #21/#22/#23/
#24/#28/#38/#50/#51/#73 consulted. All-state issue inventory checked for duplicates.

Local Python3.11.9, NumPy2.4.6, pandas3.1.1, sklearn1.9.1, XGBoost3.2.0,
pytest8.4.2, Black24.4.2, Ruff0.5.0 match relevant Python3.11 lock entries.
Interpreter: `../AstroSpectro/.venv/Scripts/python.exe` (no installation edits).

## Initial significant-path matrix (historical, qualified by the revision above)

AP = evidence on a precise contract; PA = partial; NA = not validated; B = blocked.
All line numbers refer to the unchanged352-line bridge at the base SHA.
Tests prefixed `test_` below are in [unit tests](../tests/test_dimred_xgboost_bridge.py)
unless designated **R** ([real integration](../tests/test_dimred_xgboost_bridge_real.py)),
**P** ([primary proof archive](https://github.com/PhD-Brown/AstroSpectro/blob/0da758111daaf7ba33283c1462ec1fb1368c6cff/tests/proofs_dimred_xgboost_bridge.py)) or
**PS** ([supplemental shape archive](https://github.com/PhD-Brown/AstroSpectro/blob/7b32abbd1881a487c0ea1bcc7355883e5797ea82/tests/proofs_dimred_xgboost_bridge_shapes.py)).
No bridge tests predated this mission. Evidence commands, versions, exact hashes
and execution SHAs are in the ledgers below, not inferred from the active HEAD.

| Code / independent contract | Risk; level | Executed test(s) / result | Status; limitation / issue |
|---|---|---|---|
| imports/path169–178: resolve SRC_DIR and fallback only at import boundary | Import state; U/I | test_path_resolution_side_effect_is_idempotent_observation; test_utils_import_fallback_is_lexical_characterization PASS | AP current insertion/idempotence; intentional global sys.path change recorded. Fixtures restore it; no public-facade claim. |
| discovery180: real utils uses mtime; local fallback lexical | Artifact identity; U/I | test_model_selection_uses_actual_utils_mtime; fallback test PASS | AP distinct mtime-vs-name scenario; mtime ties/OS permission policy untested. utils239–272 read. |
| loading188–205: package/script imports and absent/unloadable model yield structured failure | I/O; U/I | test_classifier_script_import_fallback, test_classifier_both_imports_unavailable, test_early_failures_return_complete_empty_result PASS | AP stubbed import/load failure paths; real joblib success separately R. Missing feature metadata caught during logging; no hostile pickle loaded. |
| CSV210–218: explicit stem wins; fallback lexical last | Provenance; U | test_explicit_csv_wins_and_fallback_is_lexical; early no_csv PASS | AP current discovery; scientifically compatible generation not established. No metadata binding; #20/#23. |
| CSV218 and inference289–296: postload errors visible | Diagnostic; U | test_postload_exceptions_remain_visible (CSV/predict/decode) PASS | AP explicit ParserError/ValueError; broad operational-robustness prose is not a universal exception guarantee. |
| SNR223–226: retain10 and10.001, remove9.999/NaN; absent SNR means no cut | Population; U/P | test_snr_inclusive_boundary_and_absent_snr PASS with matched prefiltered Z/y/clusters | AP boundary/values; fixed r/10 does not reproduce arbitrary loader options. |
| candidates234–242: numeric, varying, <=10%NaN; rows incomplete on these columns removed | Population; U/P | test_candidate_nan_fraction_boundary (0/1/2 of10), test_candidate_variance_and_dtype_control_row_selection PASS | AP independent cutoff/dtype/variance predicates, including nullable Boolean withNA; no universal dtype inference promise. |
| _EXCL_COLS: excluded metadata with1/10NaN cannot delete rows | Provenance; U | test_excluded_metadata_nan_cannot_remove_rows (5columns) PASS | AP selected named exclusions; not a provenance allowlist (#23). |
| numeric representations and model schema | Numeric; U | test_representable_dataframe_dtypes_are_passed_by_name (8dtypes), test_candidate_selection_keeps_named_model_columns PASS | AP actual DataFrame boundary; nullable/string cases use read_csv return injection, not a claim CSV preserves extension dtypes. |
| schema285: named columns exactly in saved order with row values intact | Schema; U/I | test_nominal_named_values_decode_confidence_and_integrity; coherent_permutation; **R** real_xgboost_saved_pipeline_inference_parity PASS | AP finite supported schema; correct dimensions alone are insufficient. Spies record both estimator inputs; CSV and caller arrays/palette untouched. |
| alignment248–262: correctly preselected same-order population remains paired | Identity; U/I/S synthetic | test_coherent_permutation_preserves_observation_values; nonconsecutive_pandas_index PASS | AP conditional pairing; no independent ID verification. |
| alignment248–262: middle deletion or incompatible label cardinality must not yield false associations | Identity; U/I/S synthetic | **P** filter_cannot...3FAIL; each_input_cardinality...4FAIL | PA, confirmed #88; compatible refusal or real reconciliation remains open. |
| equal-size independent permutations, replaced/duplicate/absent CSV IDs | Identity; U/I observation | **P** identity_information_missing_from_api_is_observation6PASS | PA architecture: no embedding IDs in API. These are observations of a violated caller precondition, not green requirements to detect unseen IDs. |
| missing names265–280: technical30% cutoff and current zero placeholders | Schema; U | test_missing_feature_fraction_characterization6PASS | AP characterization only (0/1/2/3/4 missing across3/4/7/10expected); scientific imputation OPEN/#89. |
| preparation285 must preserve the saved fitted transformation or refuse | Statistical; I/P | **P** real_fitted_imputer...present_nan FAIL; absent_column PASS control | PA, confirmed #89 by actual saved XGBoost. NaN20% remains a query row but is zero-filled before trained median. |
| decode289–296: exact saved class mapping or original labels as strings | Taxonomy; U | test_decoding_observed_classes_and_fg_mask; test_no_encoder_preserves_labels_as_strings PASS | AP synthetic mapping G/M/F and observed subset; no new taxonomy or mandatory argmax decision rule. |
| proba299–302: row maxima without altering predictions | Numeric; U/P | test_probability_row_maximum_does_not_override_decision_rule PASS | AP analytic asymmetric matrix; not calibration. |
| unavailable proba303–307: absent/raising method or unusable reduction shape marked unavailable | Availability; U | test_probability_unavailable_has_explicit_flag; test_unusable_probability_shape_is_not_advertised PASS | AP methods absent/ValueError,1D and(N,0); ones sentinel never interpreted as calibrated confidence. |
| output289–330: valid cardinality, finite normalized probability matrix, scalar confidence | Numeric/shape; U/I | **P** prediction_cardinality3FAIL, invalid_probability7FAIL; real_degenerate_gaussian FAIL; **PS** tensor shape FAIL | PA/#90. Ten primary and one supplemental cases are injections, one primary case real GaussianNB; not twelve independent backend bugs. |
| result309–330: stable keys/types/status/classes/F-G mask/model path | API; U | nominal; decoding; repeat_inference_and_second_call_failure PASS | AP successful finite nominal results; invalid-output schema remains #90. |
| _empty_result333–352: full fresh failure structure, no stale class list | State; U | test_empty_structure_is_fresh; early_failures5cases PASS | AP fresh mutable list and all early-return branches exercised. |
| saved classifier2168–2254: trusted same-version roundtrip and bridge inference parity | Persistence; I/P | **R** real_xgboost_saved_pipeline_inference_parity2PASS, real_pipeline_explicitly_refuses_infinity2PASS | AP4CPU tests included in default CI; general train/save/load compatibility remains #38/#28. |
| facade/__init__46/70 and optional Torch | Imports; I observation | Fresh `python -I -X utf8` actual facade import FAIL with NameError nn, Torch actually absent | B for public facade in this environment; already #21, not hidden by source-direct tests. |
| empty/one-row/invalid schema and nonfinite inputs | API; U observations | **P** degenerate_schema_and_population7PASS, nonfinite_preparation4PASS | PA observed spy acceptance only; real infinities refused separately R. No general empty-domain or invalid-schema policy selected. |
| STELLAR_COLORS / unused color_map | Presentation; static/U | Palette source read; nominal palette input unchanged PASS | PA aesthetic constants not scientifically validated; no plotting performed. |

No changes to production, dependencies, CI, notebooks, real models or data.

## Consumer boundaries inspected

`dimred/__init__.py` exports xgboost_predict but imports the known failing optional
Torch facade (#21). Source-direct unit imports do not validate the facade.
NB02 cells3/21/48 load PCA artifact y/meta/features_stem, call the bridge, and
pass results to plots/reporter; NB02 does not reconstruct the loader population.
The committed historical output reports42920 bridge rows versus43019 embedding
rows: this is historical output, not a measured current run or quantified harm.
There is no obsolete bridge call in that notebook: it uses the current API.
The claim in its comment that predictions are aligned is conditional, as above.
`data_loader.load` has configurable SNR band/minimum, classes, balancing and
spectro_only; the bridge hardcodes r>=10 and an independent denylist.
Embedding preserves order provided by its caller. HDBSCAN profiles have their
own positional limitations (#73). Visualizer infers probability availability
from confidence!=1 rather than receiving has_proba. Reporter computes accuracy
when lengths agree, which cannot establish identity. These consumers were read,
not executed end-to-end; M2.7/M2.8 are not started.

Inspected at the base SHA: complete bridge; facade; relevant loader selectors,
load/meta/scaler contracts; embedding fit/return order; HDBSCAN profiles;
visualizer2055–2209; reporter715–1082 (only bridge inputs/consumption, not a
reporter audit); classifier preparation812–1084, pipeline interfaces and
save/load2168–2254; utils.latest_file239–272; NB02 source cells3/21/48 and
relevant historical outputs; pyproject, CI lock, precommit, sole quality workflow
and conftest. No notebook or historical artifact executed/deserialized.

### Exclusion and provenance comparison

[Exact static inventory](m26_exclusion_inventory.json) is from AST literal
extraction plus the real loader constants at the source SHA. Loader default
excludes `label`/`main_class` that the bridge admits as candidates; bridge
excludes `radial_velocity_corr` that the default loader admits. Loader
spectro_only additionally excludes instrumental fields and Gaia prefixes;
bridge has no spectro_only argument. Classifier base exclusions additionally
remove fields such as RA/Dec, instrumental coefficients, label/main_class,
parallax_error and phot_bp/rp_mean_mag. Conversely bridge excludes ag_gspphot,
ebpminrp_gspphot,phot_variable_flag,radial_velocity_corr that classifier base
does not (classifier spectro_only has additional restrictions).

Crucially these bridge exclusions select the **population for dropna**, not
the final inference schema: final columns are taken from saved
feature_names_used, even when a name belongs to _EXCL_COLS. Thus an excluded
candidate name is not proof it cannot reach a model that requests it. An unused
numeric variable with10%NaN can delete an inference row; a numeric label column
could do so too. These differences are demonstrated structurally and by the
archived unrelated-feature observation; they do not quantify real prevalence.
No claim of spectro-only provenance follows from these denylists (#23).

### Independent examples and tolerances

Nominal IDs104/19/807/33/250 have alpha3/10/8/23/4, beta101/31/89/17/53,
gamma3/8/1/6/4. The saved order is gamma,alpha,beta; spy integer codes
alpha mod3=0/1/2/2/1 decode through G/M/F to G/M/F/F/M. Literal per-row
probabilities have maxima .7/.55/.8/.8/.55. Another matrix's maxima are
.8/.6/.7/.9/.5; predictions deliberately disagree with some argmax values,
so a legitimate threshold decision wrapper is not rejected. Values/IDs are
compared, not merely dimensions. Two coherent paired permutations preserve
the association. Candidate missing fractions are hand-computed0/10,1/10,2/10;
missing-name fractions include3/10 (accepted technically) and4/10 (refused).
These decimal comparisons are numerical examples, not astrophysical thresholds
endorsed by this audit. Exact spy checks use identical double literals; actual
same-version probability roundtrip uses rtol0/atol1e-12, from #38's parity
criterion. No observational uncertainty is inferred from that tolerance.

## Scientific limits and open decisions

No IDs for Z/y/clusters enter the bridge API. Equal cardinality cannot prove
identity; independent same-length permutations/population replacement cannot be
detected from these inputs. A coherent externally supplied population/order is
a caller responsibility; **the bridge must still preserve the real loader's
population**. Its independent re-filtering can violate that contract despite
a correct caller, including at equal lengths (R1). The initial precondition
wording was too broad and is superseded by the actual-loader proofs above.
Zero substitution is a current behavior, not a statistically justified policy.
Training/preprocessing (#24), group leakage (#22), feature provenance (#23),
XGBoost compatibility (#28), general persistence (#38), and real-catalogue,
calibration and astrophysical validation remain separate questions.

## Execution checkpoints (historical, not replaced by later totals)

- At base with new unit bytes: `python -m pytest -q tests/test_dimred_xgboost_bridge.py
  -W error --tb=short --junitxml=test-results/m26-unit-first.xml`:54PASS.
  Black/Ruff and diff check PASS. Checkpoint8bdea734 pushed and remote verified.
- Real CPU named pipeline: median imputer and StandardScaler fitted on30
  synthetic training rows, XGBoost12trees/depth2/thread1/seed1729; five query
  rows. No full train_and_evaluate, tuning, calibration or reference run.
  `-m pytest -q tests/test_dimred_xgboost_bridge_real.py -W error`:4PASS.
  Combined unit/real58PASS; Black/Ruff3files and diff check PASS.
  Same-version direct/joblib/bridge probability comparison uses atol1e-12,
  rtol0 (serialization parity, not observational tolerance). Real infinity
  refusal is verified, rather than imposing any universal zero replacement.
  Canonical pipeline.classifier is executed from this worktree's unmodified
  source; temporary sys.modules entries/sys.path are restored by monkeypatch.
- Archive `0da758111daaf7ba33283c1462ec1fb1368c6cff` pushed, remote SHA
  and GitHub proof blob verified. Clean-commit replay19FAIL/19PASS/0ERROR/SKIP,
  saved in `audit/m26_proof_results.json` with actual HEAD, empty dirty state,
  input hashes, command, per-case diagnostics and output. Archive contains its
  matching fixtures/source and classification in `audit/M26_PROOFS.md`.
  New confirmed defect families: [#88](https://github.com/PhD-Brown/AstroSpectro/issues/88)
  identity/cardinality, [#89](https://github.com/PhD-Brown/AstroSpectro/issues/89)
  overriding fitted imputation, [#90](https://github.com/PhD-Brown/AstroSpectro/issues/90)
  invalid outputs.19FAIL are not19bugs. Unit schema probes and real degenerate
  backend evidence are distinguished explicitly in the issues and manifest.
- Additional candidate/exclusion tests use10rows with1NaN, so the exclusion,
  variance and dtype predicates are independently discriminated (5rows would
  exclude the column via20%NaNs and hide those predicates). Green suite now
  **58unit +4real =62PASS** under `-W error`.

## Initial mutation evidence and limits (recorded test selections only)

[Runner](check_m26_mutations.py) copies only relevant source/config/tests to a
TemporaryDirectory; exact anchors must occur once, ast.parse must succeed,
pytest collection must match the positive baseline's selected case names, and
line tracing must show the changed statement executed. It records commands,
stdout/JUnit diagnostics, hashes, Git blobs, start/end HEAD and working-tree
state. The real worktree inputs are hash-guarded; no tracked production edited.

[Initial campaign](m26_mutation_initial_results.json) was run on clean
`553c2fb81bd1cdd57b06bca0f25e4050faefe7f3`: baseline62PASS,25mutants,
5controls. Historical raw classification:21KILLED_BY_ASSERTION,
3RUNTIME_FAILURE,1SURVIVED; controls5PASS. Two runtime labels were overly
conservative: pytest's plain `E assert False is True` diagnostics did not spell
AssertionError. The third was an actual test-order weakness: accepted schema
was indexed before asserting success, so the strict30% mutant caused IndexError.
An explicit acceptance assertion now precedes spy indexing. The historical
JSON is retained unchanged, not retrospectively relabeled.

[Targeted follow-up](m26_mutation_followup_results.json) at clean
`72a9206a7fb004604b21d96125fdf2679d49a205` reruns those3cases with a62PASS
baseline: **3KILLED_BY_ASSERTION**, no errors/skips. Final combined evidence
is **24distinct mutants detected by relevant assertions,1survivor,5controls
PASS**, not a claim of a second complete30case campaign. No unaffected
candidate was rerun unnecessarily. Later additions test only two invalid
probability shapes; the original62case campaign snapshot remains explicit.

| Mutant family (exact replacements are in JSON) | Detection / limit |
|---|---|
| reversed columns/rows; shifted y/clusters/Z | Named-frame/paired-array assertions;5detected |
| strict/lowered SNR, strict10%NaN; variance/dtype/exclusion predicates removed | Independent population/value assertions;6detected |
| strict30% missing cutoff, missing-column value1 | Acceptance + exact current placeholders;2detected; this is technical characterization, not approved imputation |
| ignored decoder; probability rows reversed/axis0/minimum; F-only mask; trained-vs-observed classes | Exact labels/maxima/mask/subset assertions;6detected |
| failure/success flags, probability availability, lexical-vs-mtime model, first-vs-last CSV | Status/discovery assertions;5detected; lexical CSV policy characterized, not endorsed scientifically |
| fillna(0)→fillna(7) on the nominal finite input | **SURVIVED**, reached285 but no missing model value in that green scenario. This is a declared #89 coverage gap; protecting zero fill as scientifically correct would be wrong. |
| DataFrame copy, np.max equivalent, np.unique classes |3equivalent controls PASS |
| preserving pipeline NaNs, explicit mismatched-population refusal |2admissible partial-remedy controls PASS; not production changes, complete fixes or scientific approvals |

Mutations cover bridge semantics, not perturbations of XGBoost internals,
historical training, calibration or unseen architectures. The known truncation
path has red proofs, not an arbitrary green contract. No quota or mutation
percentage is used as scientific validation.

## Initial defects, archives and review boundary (historical delivery)

- NEW [#88](https://github.com/PhD-Brown/AstroSpectro/issues/88), P1: actual
  filtering/truncation and independent input-cardinality failures.
- NEW [#89](https://github.com/PhD-Brown/AstroSpectro/issues/89), P1: actual
  saved median/scaler/XGBoost direct F versus bridge M at ID807.
- NEW [#90](https://github.com/PhD-Brown/AstroSpectro/issues/90), P2: real
  degenerate GaussianNB returns5NaN confidences advertised as available;
  injected malformed outputs document the wider validation boundary.
- KNOWN #21 facade; #23 provenance; #22/#24 training leakage/transformations;
  #28 version-sensitive training/calibration and #38 general persistence.
- OPEN/risk: unbound model/CSV generation, mtime/lexical fallback policy (#20),
  missing-column policy, unsupported needed-name lists, unavailable-proba
  presentation, real population identity and any published-result impact.
  No fourth defect inferred from these unmeasured risks.

Primary [proof archive0da7581](https://github.com/PhD-Brown/AstroSpectro/tree/0da758111daaf7ba33283c1462ec1fb1368c6cff)
contains38cases:19FAIL/19PASS; [manifest](https://github.com/PhD-Brown/AstroSpectro/blob/0da758111daaf7ba33283c1462ec1fb1368c6cff/audit/M26_PROOFS.md)
classifies each family and oracle limits. Its fixtures differ from the latest
unit-file bytes, so execute from that fixed checkout. No proof was weakened
or changed after archival. [Supplemental7b32abb](https://github.com/PhD-Brown/AstroSpectro/blob/7b32abbd1881a487c0ea1bcc7355883e5797ea82/tests/proofs_dimred_xgboost_bridge_shapes.py)
adds1FAIL: a(5,2,3)probability tensor is advertised as(5,3)confidence.
[Replayed from fixed-ref files](m26_shape_proof_results.json), also classified
as a #90 injection, not a backend discovery. Two green cases verify1D/(N,0)
already become unavailable. Final M2.6 suite: **60unit +4real =64PASS**.
Both archive commits remain ancestors; proof files were removed from the
integration head only after push/remote/GitHub accessibility verification.
No branch/worktree was deleted and no history rewritten.

Claude should independently review identity preconditions, the direct median
oracle, Gaussian/tensor boundary qualification, mutation diagnostics and
scientific-policy neutrality. In particular the median proof compares labels
and distinguishes this actual fixture; it is not a general transformed-value
equality oracle. No claim of leak-free training, calibrated probabilities,
catalogue provenance or astrophysical validity follows from any green result.
Full classifier training, cross-version persistence, GPU, real spectra/models,
W&B/services, end-to-end notebooks/plots, coverage measurement, OS permission
failures, invalid input shapes/None and arbitrary model schemas are untested.
The Draft PR remains unmerged; M2.6 is not integrated; no M2.7 work follows.

## Initial local verification and handoff (historical delivery)

[Machine-readable validation](m26_validation_results.json) records a clean input
snapshot **68b86aaa1d967ed2dd42c9ffd208b1d21e11190c**, actual commands, stdout,
stderr, versions, Git blobs and effective LF-normalized SHA256 hashes. A later
documentation/evidence commit is not retroactively described as the tested HEAD.
Production source is still the base18302e68, bridge blobeab849e4. Environment
Windows/Python3.11.9, pytest8.4.2/pytest-socket0.7.0, versions listed above;
final subprocess suite environment PYTHONHASHSEED7/MPLBACKEND=Agg.
Commands below use `../AstroSpectro/.venv/Scripts/python.exe` from the M26 root.

| Command / check | Actually observed |
|---|---|
| `-m pytest -q tests/test_dimred_xgboost_bridge.py tests/test_dimred_xgboost_bridge_real.py -W error` | **64PASS**, no skips/warnings/errors |
| `-m pytest -q` | **480PASS /62deselected /7warnings** |
| `-m pytest -q --import-mode=importlib` | **480PASS /62deselected /7warnings** |
| Same64cases in reverse order, via pytest.main collection plugin, `-W error` | **64PASS**, original sys.path and four temporary import entries restored by identity |
| `-m black --check src tests scripts app.py audit/check_m26_mutations.py` | **38files unchanged**, PASS |
| `-m ruff check src tests scripts app.py audit/check_m26_mutations.py` | PASS |
| `-m pip check` | No broken requirements |
| `git diff --check`; diff against base for src/config/dependencies/CI/notebooks | PASS; no production/config differences |
| `git merge-base --is-ancestor` for both fixed proof SHAs | BothPASS; archives remain reachable |
| Fresh actual `python -I -X utf8` facade import, Torch not installed | FAIL NameError nn at autoencoder87; known#21, separately recorded, not hidden in a global “all tests pass” claim |

The62exclusions are the repository's configured slow/network selection, including
optional UMAP/HDBSCAN/Torch work from earlier missions. **All4real M2.6 XGBoost
tests are in the default suite**, not skipped. No optional slow campaign was
required for this bridge. The7warnings come from existing Gaia DataFrame.drop
inplace deprecation (#16), not the M2.6 tests. Primary proof19FAIL/19PASS and
supplemental1FAIL are deliberately separate from the green CI suite.
No line-coverage percentage was measured.

Every checkpoint was pushed and compared to its remote SHA (8bdea734,
8a8e54ec,0da75811,553c2fb8,72a9206a,7b32abbd,68b86aaa; subsequent
evidence/documentation head is in the Draft PR). The original checkout and old
worktrees remain untouched. Hosted CI is checked **after** PR creation and its
actual run/head/job result belongs in the PR description and single #43
handoff; local tests are not presented as hosted CI. Next action is the first
independent Claude review, with scientific decisions and merge reserved to the
human. Keep #43 open and M2.6 yellow until human integration.
