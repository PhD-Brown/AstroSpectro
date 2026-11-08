# M2.2 — PCAAnalyzer: contracts, tests and residual defects

2026-11-08 · revised after first Claude review · [Refs #43](https://github.com/PhD-Brown/AstroSpectro/issues/43).
Tests/audit only; R1-R4 applied for Claude's second review; human merge decision pending.

## Current revision after the first independent review

[Claude's complete review](https://github.com/PhD-Brown/AstroSpectro/pull/64#pullrequestreview-5818590387)
and all eight inline threads were read with current AGENTS.md and #43. Local
and remote head were f58d52e, clean; main remains 3604b3d. Same work branch and
Draft PR #64, no production edit or scientific decision.

| Request | Change and independent check | Status / remaining policy |
|---|---|---|
| R1.1 constant input | Green test permits **ValueError validation only**, not any exception; if fit succeeds, ratios must be nonfinite and projection/reconstruction exact. RuntimeWarning text/existence not imposed. Current acceptance archived separately. | AP conditional math; zero-total-variance acceptance/rejection OPEN #59 |
| R1.2 Lick priorities | Removed hBETA_index→Balmer and TiO_1_Lick→Titane green expectations. Used feature_TiO5_depth as nominal molecule representative; Dn4000 covers Lick without precedence conflict. Ambiguous outcomes are archive observations. | PA scientific taxonomy; priority/case policy OPEN |
| R1.3 invalid payloads | Missing file still FileNotFoundError. Trusted `{}` / `pca=None` payloads must fail with ValueError/TypeError/KeyError/AttributeError; silently returning an object fails. Raw-access errors describe current behavior, not preferred validation. | AP refusal; explicit schema/diagnostic policy PA |
| R2 accidental TiO | Ten assigned producer keys verified by AST, all wrongly routed to molecular family by IGNORECASE `TiO` matching `tio`. Red oracle excludes that family without choosing the destination. | Confirmed defect; taxonomy PA, issue body prepared below, **not created** |
| R3 / #59 | Truncated 9/14 vs 0.9 oracle checks count existence and fraction attained; capped answer rejected. Separate full-rank integer example uses positive covariance eigenvalues/rank at 100%, not a universal epsilon. Explicit validation/None/NaN are admissible outcomes, not an API choice. | 2 FAIL reproduced; reachable/cap/diagnostic controls PASS. NaN target is OPEN observation |
| R3 / #63 | Proof accepts explicit ValueError or a correct bounded table; checks all indices/counts and six analytic statistics. Controls reject a wrong mean. NaN-label empty percentile observed separately. | Internal IndexError still FAIL; clamp vs reject/missing-label policy OPEN |
| R4 int64 | `(2*x).astype(np.int64)` has correlation +1 under Pearson/Spearman; dtype, column presence and unchanged other correlations asserted. | AP float64+int64; float-only mutant killed by both analytic method cases. Distinct from #61 |
| N1/N2 | No-op removed. Default correlation count at K=11, default class count at K=6 protected without fixing K<5 behavior. | AP defaults=10/5 on valid dimensions; three new real mutants killed |
| N3/N4 | ndarray/Index names, empty list fallback, negative reconstruction/correlation counts and PC0 labeling archived as observations. | PA / C1/C3 OPEN; misleading labels identified, no policy/priority selected |

**Revised archive:** existing ref `audit/43-dimred-pca-proofs` advanced by ordinary
fast-forward push to **`02cad98e64ae9afe926b0e2bc57f800f1d8ff9bf`**:
[proofs](https://github.com/PhD-Brown/AstroSpectro/blob/02cad98e64ae9afe926b0e2bc57f800f1d8ff9bf/tests/proofs_dimred_pca_analyzer.py),
[manifest](https://github.com/PhD-Brown/AstroSpectro/blob/02cad98e64ae9afe926b0e2bc57f800f1d8ff9bf/audit/M22_PROOFS.md),
matching conftest/source/config in that tree. Ref and all three files verified
on GitHub **before removal** from integration; #59/#63 descriptions/permaliens
updated first. Initial a9c84a6 and f58d52e remain ancestors (verified); no
force-push, squash or rewritten SHA. Integration child retains that ancestry
and removes only proof file/manifest. No red probes or xfail in final diff.

**41 archived cases: 27 FAIL / 14 PASS / 0 ERROR / 0 SKIP / 0 XFAIL**, exit 1,
repeated at committed archive. Includes ten cases of **one** R2 mechanism,
six failures of unapproved C1-C4 proposals, and observations/oracle controls.
Not 27 bugs. Initial **21 cases / 16 FAIL / 5 PASS** remain historical at a9c84a6.

### Revised execution ledger (Windows, Python 3.11.9)

Prefix `.venv/Scripts/python.exe` from repository root. Package versions match
the initial ledger below; pip check passed. Final tested SHA and new hosted CI
link/result are in the PR and #43 handoff; old CI is not delta evidence.

| Command / scope | Actual revised result |
|---|---|
| `-m pytest -q tests/test_dimred_pca_analyzer.py -W error --junitxml=test-results/m22-review-green.xml` | **79 PASS**, no warnings/skips/xfails |
| Targeted suite, reversed collection via pytest hook, PYTHONHASHSEED=7, -W error | **79 PASS**, `test-results/m22-review-reverse.xml` |
| `-m pytest -q --junitxml=test-results/m22-review-fast.xml` | **219 PASS / 7 existing Gaia warnings**, no skips/xfails |
| At committed archive `02cad98`: `-m pytest -q tests/proofs_dimred_pca_analyzer.py --tb=no --junitxml=test-results/m22-review-archive.xml` | **27 FAIL / 14 PASS**, messages/JUnit reviewed, no fixture/collection errors |
| `audit/check_m22_mutations.py --output test-results/m22-review-mutations-final.json` | Baseline 79 PASS; **25 real mutants killed / 0 survivors / 0 inconclusive**; **6 compatibility controls PASS**, excluded from mutation total |
| `-m black --check src tests scripts app.py audit/check_m22_mutations.py` | **32 unchanged** with archive present; **31** after removal |
| `-m ruff check src tests scripts app.py audit/check_m22_mutations.py` | All checks passed; archive/replay explicitly included |
| `-m pip check`; `git diff --check`; protected-path diff vs main | No broken requirements; clean final diff; production/config/CI/notebooks/data unchanged |

The 22 initial real mutations remain detected. Additions:
`integer_parameters_excluded`, `default_correlation_nine`, `default_class_four`.
The former `failed_fit_guard_still_inadequate` is removed as a **no-op**, not
counted as a survivor or useful equivalent mutant. Six separately labeled
compatibility controls cover zero-variance rejection, payload ValueError,
Lick-first lookup, case-sensitive TiO candidate, class clamping and class
validation. All preserve green contracts; they are **not approved fixes**.
The first zero-variance candidate applied var to nonnumeric strings and failed
the existing invalid-input test. Converting numeric input before that local
candidate's check corrected it; the unchanged test and complete replay pass.
No production patch or green assertion was weakened to hide that candidate error.

Claude's 28-mutant campaign was inspected as review evidence, not claimed as
our replay. Integer/default survivors now have direct protection; ownership
copies, requested/effective count and threshold policies remain neutral.
Retaining zero-weight family rows is benign, not a new bug.

## Reference and preparation

Production base: **`main@3604b3db8a66f19b07569f3c0aafb7f256734e77`**,
the verified merge of [PR #56](https://github.com/PhD-Brown/AstroSpectro/pull/56).
`git fetch origin` advanced origin/main from 2f2a444 to this SHA. Initial local
branch `audit/43-dimred-data-loading@df72072` was clean, with no untracked files,
and was preserved. `git rev-list --left-right --count HEAD...origin/main`
showed 0/1. Authorized branch **`audit/43-dimred-pca`** starts at origin/main.

Read: current AGENTS.md v1.3 on main; #43 and all 17 comments available at
preparation, including [M1](https://github.com/PhD-Brown/AstroSpectro/issues/43#issuecomment-6699300347)
and its historical versus current mandate; historical audit's dimred/optional
dependency findings and unaudited scope; M2.1 matrix/tests/fixtures/mutation
replay and [second review](https://github.com/PhD-Brown/AstroSpectro/pull/56#pullrequestreview-5814921502);
#21/#23/#24/#49–#57; complete 641-line PCA module, pytest/CI configuration,
and PCA consumer call sites in visualizer, reporter, autoencoder and notebooks.
These readings do not attest to historical runs. Existing PCA tests before
this lot: **none**; M2.1 tests and the existing 140-case Windows suite are on main.

Files in the integration diff:

- `tests/test_dimred_pca_analyzer.py`: **79 green pytest cases**.
- `tests/conftest.py`: isolated source import and analytic PCA fixture only;
  existing loader fixtures unchanged.
- `audit/check_m22_mutations.py`: reproducible disposable-copy replay.
- This matrix. No production, dependency, CI, notebook or data changes.

## Independent oracles and assumptions

The principal fixture has six centered coordinates ±3e1, ±2e2, ±1e3, rotated
by a rational orthogonal basis `[[.6,.8,0],[-.8,.6,0],[0,0,1]]`, then translated
by (10,-7,4). Its **sample** covariance eigenvalues are 18/5, 8/5, 2/5, variance
fractions 9/14, 4/14, 1/14. The tests also compute covariance independently
with `(X-mean).T @ (X-mean)/(N-1)` and numpy eigvalsh. Directions are compared
as outer products; repeated eigenvalues are checked through the subspace
projector, never a prescribed eigenbasis. Signed values use an orientation
factor measured against the known analytic axis, not a fixed sklearn sign.

Reconstruction uses the independently known projector Pk and residual
`(Y-mean) @ (I-Pk)`; per-row MSE is squared Euclidean norm divided by D.
The discarded-eigenvalue identity includes `(N-1)/N`, because covariance uses
N-1 and the MSE mean uses N. Tests distinguish new rows outside the training
subspace from exact reconstruction of the fitted plane.

Pearson's oracle is the defining centered inner product, without scipy's
correlation functions. Monotonic cubic data distinguish Spearman (=1) from
Pearson (<1); a symmetric quadratic has zero correlation with its signed
coordinate. Constants yield undefined coefficients, with expected warnings.
Masking and the **implemented**, not universally optimal, 30-pair boundary
are tested independently of the number of rows in the original table.

Float64 absolute tolerances are 2e-13 for these small, well-conditioned linear
algebra problems and 2e-12 for squared residuals/round trips, with relative
tolerance explicitly zero in numerical comparisons. They allow
accumulated roundoff while remaining far below the separated eigenvalues and
nonzero MSEs. They are not tolerances for arbitrary spectra or ill-conditioned
catalogs. No physical units, causal claims or independent astrophysical
validation are inferred from dimensionless synthetic features.

## Systematic method and path inventory

G denotes the green file above; A denotes the fixed proof archive below.
Test names omit `test_`. U/I/P/S = unit/interface/numerical/scientific-risk
contract. **AP** is restricted to the named executed contract; **PA** is a
remaining gap, not an accepted risk. Each row describes inputs → outputs,
preconditions/dependencies/effects, executed assertions, and limits. All source
line numbers refer to the production base above. No method receives blanket AP.

| Code / input → output | Preconditions, dependencies, effects and risk | Tests / result | Audit disposition |
|---|---|---|---|
| `__init__` 63–81: parameters → empty analyzer | Stores options, no I/O; sklearn validates options at fit | G `constructor_and_unfitted_guards`, `invalid_estimator_parameters_fail`: PASS U | AP construction/selected rejected parameters; arbitrary Python types PA |
| `fit` 83–120: X(N,D), optional names → self | sklearn PCA; replaces estimator, caches caller's X and names; numerical/state/provenance risk | G `fit_covariance_scores_and_variance` (four solvers), `centering_does_not_standardize`, `component_selection_parameters`, `refit_replaces_mean_dimensions_and_names`, invalid input tests: PASS U/P | AP centering, dimensions, covariance, forwarding, successful refit and no input mutation; A failed fit and aliased ownership below; no implicit train/test split promised |
| `fit_transform` 122–131 | Calls fit, then projects; same side effects | G reproducibility and refit tests compare separate fit+transform, loader integration: PASS I/P | AP coherent scores on supplied fit population; fit failure state remains #62 |
| `transform` 133–137 | Requires fit; delegates shape/nonfinite rejection to sklearn; must not refit on new rows | G `reproducibility_translation_permutation_and_no_test_refit`, `invalid_projection_inputs_fail`: PASS U/P/S | AP fixed training mean/basis, paired row permutation, translated data, no learned-state changes; no proof of callers' real splits (#24) |
| `explained_variance_ratio` 143–146 | Fitted sample eigenvalues divided by **total input variance**, including omitted PCs | G covariance and `truncated_model_reconstruction_and_tail_variance`: PASS P | AP full and K=1 denominator; zero-total-variance policy OPEN: ValueError rejection allowed; if fit succeeds, fractions must be undefined |
| `cumulative_variance` 149–151 | Cumulative sum of fractions | G covariance and constant tests: PASS P | AP nondecreasing finite nonconstant case; constant variance undefined if fit is accepted; rejection permitted |
| `n_components_for_variance` 153–179 | Search first inclusive crossing; target must be reachable for a valid count | G `reachable_variance_thresholds`: PASS U/P, exact equality included. A truncated inaccessible target and full-rank 100% roundoff: FAIL | **Confirmed #59**, two mechanisms. Explicit diagnostic/sentinel vs count checked semantically; no universal rounding tolerance selected. NaN target is an OPEN observation |
| `variance_summary` 181–198 | Fitted model → labeled DataFrame; no persistent I/O | G covariance and serialization: PASS U/P/I | AP labels/order/fractions/cumulative/sample eigenvalues. Consumer reporter read, complete report generation PA |
| `loadings` 205–211 | Fitted estimator → (K,D) basis, direct estimator array | G covariance/projector tests: PASS P | AP orthonormal rows, sign-invariant axes/subspaces. Mutable reference intentionally exposed, not an immutable snapshot guarantee |
| `loadings_dataframe` 213–226 | Transpose basis to features×PC; names must match D | G labels/top and serialization: PASS U/I | AP names/order/transposition; A wrong count deferred, duplicate names retained, aliasing OPEN C3/C4 |
| `top_features_per_pc` 228–257 | 0-based PC index, top count → sorted rows; no writes | G `loadings_labels_top_absolute_order_and_sign`: PASS U/P; nonzero negative orientation forced | AP absolute ranking, original sign, exact names, PC labels, n_top=0/oversized. Negative index gives PC0 in A: OPEN; general invalid types/index limits PA |
| `loadings_family_breakdown` 259–315 | Fit, supplied/cached names, PC indices → arrays/taxonomy/colors/aggregated frames; no learned changes | G `family_squared_weights_aliases_overrides_colors`, missing names and bad indices: PASS U/I/P | AP numerical weights/overrides/aliases/selected PCs/totals, **PA family interpretation because of R2**. Naming-based interpretation is not feature provenance (#23) |
| `correlations_with_params` 321–393 | Same row order; scores explicit or recalculated from cached fit X; numeric params only; finite paired values; 30 pairs | G analytic ±/zero/nonlinear/**int64**, paired permutation, score precedence, 29/30/31, masked counts, constants, bad method, empty numeric set, default 10 at K=11: PASS U/P/S | AP ndarray/ordinary numeric columns; confirmed #61 mixed pandas numeric dtypes crash. A independent meta reversal is NOT A BUG under same-order precondition. Input identities absent from API |
| `reconstruction_error` 399–428 | Fitted PCA, compatible finite X, optional truncation → one MSE/row; no X mutation | G `reconstruction_projector_mse_and_curve` (fit/new), rank/subspace, truncated-tail and nonfinite/shape tests: PASS U/P | AP k=0..K on K>0 model; K=0 model inverse fails in A; negative/oversized counts use slice semantics, OPEN C1 |
| `reconstruction_error_vs_n` 430–458 | Fits multiple truncations; mean and population std; default 1..min(K,50) | G reconstruction curve and `default_curve_stops_at_fifty_components`: PASS U/P | AP explicit out-of-order/repeated valid indices and cap. A empty schema OPEN C2; before-fit default AttributeError part of #62 |
| `class_separation` 464–502 | Aligned X/y, n_pcs≤K → MultiIndex frame; per-class count/mean/population std/median/linear quartiles | G `class_statistics_match_signed_analytic_coordinates`: PASS U/P/S, unequal groups/singleton/paired permutation | AP exact statistics for valid counts/default=5 at K=6; **Confirmed #63** default 5 crashes for K=2. Missing labels, mixed types, n_pcs≤0 policy PA/OPEN; incompatible y length fails loudly in A |
| `sklearn_pca` 509–512 | Fit guard → actual sklearn estimator, exposed intentionally | G constructor/covariance/serialization: PASS U/I; A failed initial fit/refit: 2 FAIL | **NEW #62** non-None estimator is insufficient fitted-state test |
| `mean_` 515–518 | Fit guard → training mean array | G independent translated mean, new-row projections and round trip: PASS P/I | AP successful state; same post-failure guard defect #62 |
| `save` 524–530 | joblib dump minimal dictionary; writes supplied path; does not guard fit or create directories | G fitted round trip: PASS I. A unfitted save later fails load: PASS observation | AP trusted local fitted payload; permission/full-disk/interrupted/atomic write scenarios **not executed**; no safety claim for arbitrary joblib files |
| `load` 533–549 | Trusted local joblib → analyzer without X_fit; bypasses constructor | G components/mean/names/variance/new projection/reconstruction/explicit correlation: PASS I. G missing file still FileNotFoundError; malformed controlled payloads must fail loudly with validation or raw access errors, without fixing a future exception type. A implicit correlation after load fails as documented | **NEW #60** missing random_state/solver breaks refit. Payload schema/version validation, named shape consistency and cross-version persistence PA |
| `_assign_family` 601–607 | Regex first match, IGNORECASE, catch-all Other; purely lexical | G 12 nominal lexical representatives: PASS U; A 10 real non-TiO producer keys: FAIL | **PA / confirmed R2 defect**, not scientifically validated taxonomy. AP only for named lexical examples, never Lick priority/case policy; issue draft below, creation pending human decision |
| `_family_color_map` 609–615 | Dict order zipped to stable palette | G exact family order/palette and repeated lookup: PASS U | AP current taxonomy; changing/empty/extended class constants PA |
| `_family_contributions` 617–635 | Valid PC/consistent lengths/nonzero norm → normalized squared contributions, group/sort, drop zero weights | G hand [3,-4,0] → [9/25,16/25], merged family, normalization and no input mutation; indices rejected: PASS U/P | AP nonzero vectors including non-unit helper input. Zero vector/corrupt names not a normal orthonormal basis: PA |
| `_check_fitted` 637–640 | Shared guard must represent an actually fitted model | G 15 unfitted access paths: PASS U; A failed-fit accessor and default curve: FAIL | AP fresh-object guards only; #62 covers demonstrated gaps; no silent fittedness inference from `_pca` alone |

## Interfaces exercised versus only read

The integration loader test performs a real synthetic CSV identity join with
catalog order reversed, scaling, PCA fit/transform using returned feature
names, inverse PCA plus inverse loader scaling, and Pearson correlation with
the correctly aligned metadata. It checks all row values and original feature
order, not just successful imports. This validates the stated same-order
interface for a valid unique-key sample; it does not attribute loader defects
#49–#57 or provenance #23 to PCA. Population scaling in this test is descriptive,
not supervised holdout validation.

The visualizer integration passes the real reconstruction table to
`plot_reconstruction_error` under Agg, checks plotted x/y values, restores
rcParams and closes its figure. The production visualizer is imported directly
because of optional facade issue #21. No user file is written. Complete
visualizer rendering, reporter exports, autoencoder training/comparison,
embedding and notebook execution are **outside this lot**. Their relevant
source interfaces were read: visualizer consumes ratios/loadings/thresholds,
reporter clamps threshold counts before slicing, autoencoder skips counts>K;
these guards do not fix the PCA threshold API itself. Notebook outputs contain
historical K+1 counts; they are context, not rerun evidence or proof of impact.

## Initial archive and findings (historical a9c84a6)

**Before** removing red files from the integration diff, branch
[`audit/43-dimred-pca-proofs`](https://github.com/PhD-Brown/AstroSpectro/tree/audit/43-dimred-pca-proofs)
was pushed and both the ref and files were verified on GitHub at fixed SHA
**`a9c84a647d1cb1d35b3ee97e65ff099c560fc2ec`**:

- [Executable red probes](https://github.com/PhD-Brown/AstroSpectro/blob/a9c84a647d1cb1d35b3ee97e65ff099c560fc2ec/tests/proofs_dimred_pca_analyzer.py).
- [Per-case manifest](https://github.com/PhD-Brown/AstroSpectro/blob/a9c84a647d1cb1d35b3ee97e65ff099c560fc2ec/audit/M22_PROOFS.md)
  and matching fixtures/source in that commit.
- **21 cases: 16 FAIL / 5 PASS / no errors, skips or xfails.** Repeated at the
  committed archive SHA with the same results. This is **not 16 bugs**.

After duplicate searches for PCA, pca_analyzer, variance, correlations,
serialization and fitted state, and checking existing issues, five corrective
issues were created, each with source lines, oracle, command and fixed proof:

| Group | Classification | Issue / disposition |
|---|---|---|
| B1 unattainable target gives K+1 | NEW demonstrated output outside fitted basis | [#59](https://github.com/PhD-Brown/AstroSpectro/issues/59); unavailable-target outcome still to specify |
| B2 incomplete loaded wrapper, public refit crashes | NEW, two failures for one mechanism | [#60](https://github.com/PhD-Brown/AstroSpectro/issues/60) |
| B3 mixed float+bool/nullable numeric params crash | NEW, three dtype cases for one mechanism | [#61](https://github.com/PhD-Brown/AstroSpectro/issues/61) |
| B4 failed fit exposes unfitted estimator; default curve lacks guard | NEW state defect and related diagnostic gap | [#62](https://github.com/PhD-Brown/AstroSpectro/issues/62); old-state retention vs full invalidation OPEN |
| B5 default class separation crashes for K<5 | NEW demonstrated default-call crash | [#63](https://github.com/PhD-Brown/AstroSpectro/issues/63); clamp vs explicit validation OPEN |
| C1 negative/oversized reconstruction count; K=0 model | CONTRACT TO DECIDE | Slice semantics today; no green tests impose an unapproved rejection policy |
| C2 empty curve schema; C3 feature-name count/duplicates/index limits | CONTRACT TO DECIDE | Early validation/schema/rejection proposals remain in the archive/matrix |
| C4 caller-owned X/names alias cached analysis metadata | CONTRACT TO DECIDE | Reversing caller X changes implicit correlations; ownership/snapshot policy not documented |
| Unpaired metadata permutation | NOT A BUG under documented same-order precondition | No ID-aware alignment promised; paired permutation tested green |
| Missing X_fit after load | Documented limitation, explicit scores work | No new issue claiming serialization must retain training data |
| Provenance/import/split and loader issues | ALREADY KNOWN #21/#23/#24/#49–#57 | Boundaries stated; not rediscovered as PCA bugs |

The first exploratory zero-component probe incorrectly assumed sklearn could
inverse-transform zero-width scores. It was corrected to an observation after
direct reproduction: initial 17 FAIL/4 PASS becomes final 16 FAIL/5 PASS. The
manifest records this withdrawn assumption; **no extra bug is claimed**.
All five issues are independent corrections, not a second coordination issue.
No priorities of existing issues were changed, no production fix made.

## Initial adversarial evidence (historical f58d52e)

Initial green tests had 76 cases. Review identified two assertion gaps:
renormalizing retained variance could escape a full-rank test; default
correlation/cap behavior needed discriminating data. Added the explicit K=1
9/14 assertion, nonlinear default-correlation and paired-count checks, and an
analytic 52-dimensional fixture proving the default cap is 50. Then 78 cases before the independent review.
Final self-review also made relative tolerances explicitly zero and selected
the negative analytic orientation in the top-loading test. Merely flipping
sklearn's chosen sign would not guarantee a negative orientation on every
backend. No production changes or additional contract assumptions were needed.

The replay copies only required source/tests/config to a TemporaryDirectory,
checks unique replacement sites, compiles candidates, clears only sandbox
bytecode, executes the complete green PCA suite offline, verifies JUnit
collection/error/skip/count consistency, and rechecks original input SHA-256
hashes. Original working-tree source is never rewritten. JSON includes hashes,
Python version and exact failing test names for every candidate.

**Historical report:** 23 candidates executed, 22 killed and one reported
survivor. Claude correctly identified that last candidate as a **no-op control**,
not a useful mutant: the valid historical denominator is **22 real mutations**.
It is removed from the revised replay below. No mutation score is claimed:

| Mutations executed | Result / discriminating oracle |
|---|---|
| fit_standardizes; seed_ignored; solver_ignored | KILLED: independent covariance/mean/no-standardization and explicit estimator parameter forwarding |
| variance_renormalized; covariance_population_denominator | KILLED: K=1 fraction of total variance; sample eigenvalues |
| threshold_right_search | KILLED: exact reachable boundary is inclusive |
| top_signed_order; top_loses_sign | KILLED: known mixed direction with forced negative orientation, signed returned values |
| family_absolute_weights; family_no_normalization; family_reverse_priority | KILLED: squared [9/25,16/25], non-unit helper vector, ambiguous/catch-all taxonomy |
| correlation_default_pearson; correlation_thirty_excluded; correlation_ignores_score_mask; correlation_ignores_explicit_scores | KILLED: nonlinear rank oracle, exact 30 pairs, separate nonfinite masks, explicit-score reversal |
| reconstruction_omits_mean; reconstruction_sum_not_mean | KILLED: translated independent projector and division by D |
| curve_sample_std; curve_cap_fortynine | KILLED: population std and analytic 52D cap case |
| class_wrong_mask; class_sample_std | KILLED: unequal/singleton groups and analytic oriented score summaries |
| save_reverses_names | KILLED: preserved labels/projections after trusted round trip |
| failed_fit_guard_still_inadequate | Historical no-op, removed and excluded from real-mutation totals. `hasattr(PCA(), "fit")` is true on fitted and unfitted instances; the old survivor/equivalent wording was inaccurate. |

This deliberately limited campaign does not mutate sklearn or claim detection
of all possible numerical/backend failures. Replay and archived proofs are
**outside automatic CI**; they were checked explicitly.

## Initial execution ledger and environment (historical)

Red probes/fixtures executed and frozen at archive SHA **a9c84a6**. Green tests
were subsequently strengthened with explicit absolute tolerances and a chosen
negative orientation, then the targeted/full suites and mutation replay were
repeated. Those probes are retained immutably at a9c84a6; revised probes follow below. All production is
identical to main@3604b3d. Commands below were actually run from the repository
root with `.venv/Scripts/python.exe` (Python 3.11.9, Windows). The revised execution ledger below and PR supersede these initial results.

| Command (after interpreter prefix) | Observed result |
|---|---|
| `-m pytest -q tests/test_dimred_pca_analyzer.py --tb=short --junitxml=test-results/m22-green.xml` | 78 PASS, no warnings/skips/xfails |
| With `PYTHONHASHSEED=7`, targeted suite `-W error`; then `pytest.main` with a collection hook reversing all items, also `-W error` | 78 PASS in each order; strict and reverse JUnit reports under test-results |
| `-m pytest -q --junitxml=test-results/m22-fast.xml` | 218 PASS, 7 existing Gaia/pandas warnings, no skips/xfails |
| `-m pytest -q tests/proofs_dimred_pca_analyzer.py --tb=line --junitxml=test-results/m22-proofs.xml` | 16 FAIL / 5 PASS; exit 1; no collection/fixture errors; explicit proof branch only |
| At committed archive: `-m pytest -q tests/proofs_dimred_pca_analyzer.py --tb=no --junitxml=test-results/m22-archive-proofs.xml` | Same 16 FAIL / 5 PASS, exit 1 |
| `audit/check_m22_mutations.py --output test-results/m22-mutations.json` | Baseline 78 PASS; 22 real mutants killed / 1 no-op control (historical classification corrected) |
| Same replay after final assertion strengthening, output `test-results/m22-mutations-final.json` | Baseline 78 PASS; same 22 real mutants killed / 1 no-op control, unchanged input hashes verified |
| `-m black --check src tests scripts app.py audit/check_m22_mutations.py` | 32 files unchanged when archive present; replay explicitly included |
| `-m ruff check src tests scripts app.py audit/check_m22_mutations.py` | All checks passed |
| `-m black --check tests/proofs_dimred_pca_analyzer.py audit/check_m22_mutations.py` and equivalent `ruff check` | Both archived probes and replay pass explicitly |
| `-m pip check` | No broken requirements |
| `git diff --check`; protected-path diff versus origin/main; remote proof ref/file reads | Clean whitespace, no production/config/notebook/data changes; archive verified before removal |

Versions: pytest 8.4.2, pytest-socket 0.7.0, numpy 2.4.6, pandas 3.1.1,
scipy 1.17.1, scikit-learn 1.9.1, joblib 1.6.0, Black 24.4.2, Ruff 0.5.0.
Integration fixture/render dependencies: astropy 8.0.2, matplotlib 3.11.2.
The seven warnings originate from existing Gaia DataFrame.drop(inplace=...)
calls, not PCA. Expected constant-data warnings are captured by their tests.
Hosted CI is separate Ubuntu/Python 3.11 evidence; its final run/SHA/result is
linked in the PR and #43 handoff after publication, not inferred from local PASS.

## Residual gaps and review boundary

- AP covers the specified synthetic contracts, not a general astrophysical
  validation. No real catalog, FITS data, GPU, W&B or network service was used.
  Line coverage was **not measured**. No claim that every bug was found.
- No singleton-fit (N=1), extreme scale/conditioning/overflow, complex/sparse
  inputs, exhaustive solver/MLE parameter combinations or cross-platform BLAS
  sensitivity study. Randomized solver same-seed reproducibility and forwarding
  were tested; different-seed variability/approximation quality on large data
  was not certified.
- Nominal regex representatives are tested; R2 demonstrates accidental routing
  of ten real producer names. Taxonomy remains PA, with scientific priority and
  case policy unapproved; this is not an exhaustive feature-lexicon validation. Identity,
  scaling leakage and provenance guarantees require the appropriate callers
  and #23/#24 fixes; this wrapper supplies no split or observation IDs.
- OPEN boundary counts, ownership, missing labels/names, index/type validation
  and undefined-variance threshold policies are explicitly unapproved. R1 removed the previously imposed constant-fit, Lick-priority and payload-error
  assumptions; the revised green suite is neutral on those policies.
- No atomic/durable save guarantee, filesystem permission/interrupt tests,
  cross-version joblib migration or deserialization security audit. Only trusted
  locally constructed payloads were loaded.
- Claude's first review reproduced B1-B5. The R1-R4 delta now needs his second
  independent review; no threads resolved by Codex. Corrective PRs stay separate.
  Stop after M2.2; no M2.3 and no merge.

## R2 corrective issue draft - NOT CREATED

Creation and labels require the responsible scientist's decision. Proposed
title: **fix: prevent accidental TiO matches in PCA spectral feature families**.
Proposed labels: `bug`, `science`, `P2` (not applied to any new issue).
The following five sections are the prepared issue body.

### Problème

At [pca_analyzer.py:564-566 and 601-606](https://github.com/PhD-Brown/AstroSpectro/blob/3604b3db8a66f19b07569f3c0aafb7f256734e77/src/dimred/pca_analyzer.py#L565),
the unbounded `TiO` alternative is searched with `re.IGNORECASE`. It matches
the `tio` substring of ratio/rotation (also generic fraction/selection words).
The molecular family precedes Strontium, Continuum, Profiles and Lick, so
the following real assigned feature keys are captured before later rules:

| Exact producer keys | Producer lines at the audited base |
|---|---|
| feature_ratio_Fe_Mg, feature_ratio_Ca_Fe, feature_ratio_Ba_Fe, feature_ratio_Sr_Fe | [573-576](https://github.com/PhD-Brown/AstroSpectro/blob/3604b3db8a66f19b07569f3c0aafb7f256734e77/src/pipeline/feature_engineering.py#L573) |
| feature_flux_ratio_blue_red | [759](https://github.com/PhD-Brown/AstroSpectro/blob/3604b3db8a66f19b07569f3c0aafb7f256734e77/src/pipeline/feature_engineering.py#L759) |
| feature_rotation_proxy; feature_rotation_index | [922](https://github.com/PhD-Brown/AstroSpectro/blob/3604b3db8a66f19b07569f3c0aafb7f256734e77/src/pipeline/feature_engineering.py#L922), [1134](https://github.com/PhD-Brown/AstroSpectro/blob/3604b3db8a66f19b07569f3c0aafb7f256734e77/src/pipeline/feature_engineering.py#L1134) |
| ratio_EW_Ha_Hb, ratio_FWHM_Ha_Hb, ratio_EW_MgB_NaD | [1669](https://github.com/PhD-Brown/AstroSpectro/blob/3604b3db8a66f19b07569f3c0aafb7f256734e77/src/pipeline/feature_engineering.py#L1669), [1673](https://github.com/PhD-Brown/AstroSpectro/blob/3604b3db8a66f19b07569f3c0aafb7f256734e77/src/pipeline/feature_engineering.py#L1673), [1701](https://github.com/PhD-Brown/AstroSpectro/blob/3604b3db8a66f19b07569f3c0aafb7f256734e77/src/pipeline/feature_engineering.py#L1701) |

Minimal example: `_assign_family("feature_ratio_Ba_Fe")` returns the family
beginning `Titane`, even though the quantity is a Ba/Fe ratio, not TiO.
This does **not** mean every name containing ratio is captured: earlier
Ca/Balmer rules protect e.g. `ratio_EW_CaK_CaH` and Halpha wing ratios.
CaH/Lick priority inconsistencies are separate questions, not all consequences
of this substring mechanism. Static source/AST provenance was checked; the
feature producer was not executed or astrophysically validated in this lot.

### Pourquoi

The mathematical squared weights are correct, but their family aggregation
can route non-TiO variables into a molecular-family contribution. Claude found
historical notebook outputs with Titane 7.0%/8.2% on PC1/PC2; those observations
do **not** identify which weights were misrouted. Without the corresponding
run arrays/loadings, **no historical error contribution is quantified** here.

### Attendu

Prevent accidental TiO substring matches for these non-TiO quantities.
Choose case handling, lexical boundaries and destination families explicitly,
with the scientist. Red proof only excludes the molecular/TiO family: it does
not prescribe an abundance-ratio family or any other final destination.
Lick priority and dedicated abundance-ratio grouping remain OPEN. Preserve
legitimate TiO names and normalized numerical family weights.

### Vérification

Fixed [proof archive 02cad98](https://github.com/PhD-Brown/AstroSpectro/blob/02cad98e64ae9afe926b0e2bc57f800f1d8ff9bf/tests/proofs_dimred_pca_analyzer.py),
[manifest](https://github.com/PhD-Brown/AstroSpectro/blob/02cad98e64ae9afe926b0e2bc57f800f1d8ff9bf/audit/M22_PROOFS.md).
`test_non_tio_producer_features_not_assigned_to_titanium[name]`: **10 FAIL**,
one mechanism. AST checks each name is actually assigned in the producer;
the final assertion fails on the inappropriate family, not fixture setup.
Command from that checkout:
`python -m pytest -q tests/proofs_dimred_pca_analyzer.py -k non_tio --tb=line`.
Windows/Python 3.11.9, regex stdlib, pytest 8.4.2; no network or real data.
The case-sensitive candidate in a disposable copy leaves the revised green
suite passing; it is an unapproved control, not a production fix.

### Hors scope

Production change in M2.2; retroactive correction of published figures without
run provenance; full taxonomy redesign; deciding Lick priority/case policy;
quantifying astrophysical importance or causal effects; creating this issue
before the responsible scientist authorizes it.
