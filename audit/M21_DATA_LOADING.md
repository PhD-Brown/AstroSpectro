# M2.1 — data loading and preparation audit

Date: 2026-11-07. Coordination: [Refs #43](https://github.com/PhD-Brown/AstroSpectro/issues/43).
Status: delivered for independent review; no production correction or merge.

## Scope and provenance

The complete 892-line `src/dimred/data_loader.py` was read, including both
classes, all 15 methods/functions, constants, CSV fallback and nested FITS worker.
Production reference: [`main@35c2105a6373e9ca5820f08e33acd383e79afc06`](https://github.com/PhD-Brown/AstroSpectro/blob/35c2105a6373e9ca5820f08e33acd383e79afc06/src/dimred/data_loader.py).
Tests: [`2b083b324609e582c5c905425f1e3c41278cd474`](https://github.com/PhD-Brown/AstroSpectro/commit/2b083b324609e582c5c905425f1e3c41278cd474),
branch `audit/43-dimred-data-loading`. Production has no changes relative to main.

Preparation included current main `AGENTS.md` v1.3, #43 and all 10 comments,
the [consolidated M1 matrix](https://github.com/PhD-Brown/AstroSpectro/issues/43#issuecomment-6699300347)
and cross-review, the historical audit (especially D1, T3 and its unreviewed scope),
#21–#24 and their comments, #34/#35, open PRs and remote branch heads.
The current #43 mandate supersedes the historical backup-cleaner mission.
Only PR #48 was open at preparation; its branch remains at `f531205ff830d43764e4237a4370773ea42ee09a`.

The initial checkout was clean on that branch. The existing local branch `test`
prevents creating `test/43-...` (Git ref namespace collision). The failed switch
left the main tree staged under the old HEAD; equality with origin/main and the
absence of unstaged changes were verified before creating `audit/43-...` from
origin/main. The new branch was clean at `35c2105`; no old branch ref was moved.

Relevant dependencies read: `dimred/__init__.py` (known optional Torch issue #21),
`pipeline/preprocessor.py` FITS array/header/unit contract, catalogue generator
schema/IDs and `processing.py` feature/catalogue boundary, pytest/CI and formatting
configuration. Their complete audits are outside M2.1. `pca_analyzer.py` and the
other dimred algorithms were not audited or changed.

## Test artifacts and independent expectations

- [Green contracts](../tests/test_dimred_data_loader.py): **51 executed cases**.
- [Fixtures](../tests/conftest.py): temporary synthetic CSVs and primary-HDU FITS,
  including real gzip, small fixed arrays and explicit sampling seeds.
- [Red proofs](../tests/proofs_dimred_data_loader.py): **30 executed failures**.
  This filename deliberately does not match default pytest discovery. Run it
  explicitly using the command below. It contains no xfail, skip or production
  monkeypatch. These failures remain reviewable in the Draft PR; a green normal
  CI does **not** mean these contracts pass. Do not merge this proof delivery
  without reviewing the disposition of the red artifacts. The green tests can
  be integrated separately after review; fixes belong to their own issues/PRs.

Oracles are identity-key joins and left-row order, independent class/temperature
markers, population mean and variance (ddof=0), and an affine function of
wavelength evaluated analytically at endpoints and midpoints **in Angstroms**.
Tolerance `2e-13` for interpolated affine spectra and `1e-14` for standardized
features covers float64 arithmetic on tiny deterministic arrays, not astrophysical
measurement uncertainty. Descending native wavelengths are also checked: sorting
inside interpolation is valid for these distinct abscissae. Normalization uses
the median of the interpolated vector, not the native-grid median.

Physical assumptions are deliberately narrow: bare numeric wavelength values
are Angstroms reconstructed as `10**(COEFF0+i*COEFF1)`; no air/vacuum, rest-frame,
WCS or flux-unit conversion is claimed. No observed spectra or science models
are read. The worker-count test uses actual loky processes (1 versus 2 workers),
including rejection of a corrupt FITS between valid records. A test-only worker
initializer blocks socket connect/connect_ex/create_connection; workers verify
that guard is installed. Parent sockets are blocked by pytest-socket. The facade
is bypassed with importlib to isolate this source from #21, not to validate
`import dimred` or hide an import failure in the loader itself.

## Detailed method and behavior matrix

All production line numbers below refer to the fixed main SHA above. Existing
tests for **every row**: none at mission start (23 existing environment/Gaia
tests do not exercise this module). `G` = green file, `R` = red proof file;
names in this table omit the `test_` prefix. U/I/P/S denote unit, integration,
numerical/physical and scientific-risk contracts. AP means proof limited to the
named contract; PA means partial audit or unresolved policy, not a module-wide
certificate. A PASS means no defect observed under those inputs only.

| Code / role, inputs and outputs | Preconditions / contract and dependencies / effects | Executed tests and result | Evidence, anomaly and remaining gap |
|---|---|---|---|
| Constants/import boundary (1–165) | Exclusion families; numpy/pandas/sklearn. No I/O on source import. | G `column_selection_order_nan_boundary_and_exclusions`, default grid; PASS U. R `spectro_only_*`: 2 FAIL. | AP for explicit selection rules; known #23. Torch facade deliberately isolated (#21); full feature registry comparison still pending. |
| `DimRedDataLoader.__init__` (184) | Paths, merge key and seed; initialize names/scaler/cache without reading files. | G `initial_state_and_public_utilities`, `custom_merge_key_and_feature_side_precedence`; PASS U/I. | AP for construction/state; malformed parameter types not exhaustively validated. |
| `_load_features` (343) | Feature CSV → DataFrame; require file. C parser fallback to Python explicitly allows omitted bad rows with warnings. | G `missing_csv_reports_path[features]`, `csv_fallback_retains_good_rows_and_warns[features]`, `truncated_csv_reports_omitted_rows[features]`, `empty_csv_has_explicit_parser_error[features]`; PASS U/I. R large ID: FAIL. | AP for file/parser paths; exact ID inference unsafe (#50). Mixed encodings, permissions and schema variants not exhaustively covered. |
| `_load_catalog` (388) | Catalogue CSV → DataFrame; analogous parser/file contract. | Same G cases `[catalog]`: PASS. R large ID demonstrates asymmetric CSV inference. | AP for named paths; #50. Truncated records intentionally omitted with warning, not misclassified as a new bug. |
| `_merge` (407), unique keys and ordering | Many feature observations → one catalogue identity; left rows and feature-side values preserved. pandas merge, no disk writes. | G `feature_identity_join_and_permutations` (4 orders), `custom_merge_key_and_feature_side_precedence`, `missing_merge_key_fails`, `duplicate_feature_observations_are_not_silently_deduplicated`; PASS. | AP U/I on unique right keys and observed feature repetitions. No assumption that an observation ID is a stellar group ID (#22). |
| `_merge`, missing labels / ambiguous identities | Catalogue can supply labels once; missing identity is not an identity; conflicting catalogue keys cannot multiply rows; large integer identities must stay exact. | R `catalog_labels_are_joined_exactly_once` (2), `catalog_supplies_one_dimensional_labels`, `conflicting_catalog_keys_are_rejected`, `missing_identity_cannot_match_another_missing_identity`, `large_observation_ids_are_not_rounded_before_join`: 6 FAIL U/I/S. | Demonstrated #49/#50. M1 already flagged cardinality risk; label duplication, NaN matching and rounded-ID counterexamples are newly demonstrated here. |
| `load` (206), filters / row alignment | CSV→join→SNR/class→column selection→drop incomplete rows; preserve X/y/meta correspondence. SNR threshold inclusive. Missing SNR logs ignored filter. | G `snr_class_filters_and_missing_metadata_preserve_alignment`, `snr_absent_warns_and_alternate_band_is_used`, `nan_column_then_row_filtering`; PASS I. | AP for these filters, unavailable metadata and order. Conflicting shared columns deliberately retain feature-side precedence; no independent reconciliation added. |
| `_select_feature_columns` (428) | Numeric, non-meta, nonconstant columns in original order; NaN fraction exactly threshold remains; spectro-only adds exclusions. | G `column_selection_order_nan_boundary_and_exclusions` (2), `single_row_and_boolean_constant_selection`; PASS U. R `spectro_only_excludes_known_gaia_proxies`, `spectro_only_rejects_unknown_provenance`: 2 FAIL. | AP for selection mechanics, known #23 for provenance. Bool variable is treated as numeric; object strings not coerced. Exhaustive extractor/classifier registry equality is not established. |
| `load`, scaling / feature names (299–325) | X(N,D) float; feature order retained; population standardization if requested; cache final DataFrame. Fits on full current population. | G `population_scaling_and_inverse_preserve_feature_order`, `scaling_rejects_nonfinite_features` (2); PASS I/P. R unscaled reload: FAIL. | AP descriptive population moments; #51 stale scaler. Train-only predictive use **not** established; full-population scaling/selection risk already identified by M1/#24. No split API here; not a new leakage claim. |
| `load`, empty/invalid boundaries | Validate requested mode, usable features, finite raw X and explicit empty-population policy. | G `spectra_mode_delegation_error_precedes_io`: PASS. R `empty_feature_selection_returns_empty_batch` (2), `no_usable_features_has_explicit_validation_error`, `nonfinite_features_are_rejected_without_scaling` (2), `unknown_mode_is_rejected`: 6 FAIL. | PA; #55. Empty-return expectation is proposed policy, not an existing explicit guarantee. Nonfinite acceptance and ignored invalid mode reproduced. |
| `_balance` (543) | Seeded cap per class, sampling without replacement; small classes preserved. No promise of equal sizes below cap or invariance to input-row permutation. | G `balance_cap_seed_no_replacement_and_alignment`: PASS. R `balancing_does_not_silently_drop_missing_labels`, `balance_log_reports_original_population`: 2 FAIL. Empty balance case also fails. | AP nominal U/I; #54 drops missing labels silently and logs 3→3 for 10→3. #55 empty concat. Policy for missing labels needs explicit rejection or announced exclusion. |
| `_build_meta` (563) | Available physical columns/SNR/labels/coordinates only; fresh copy with RangeIndex. | G `meta_is_independent_copy_with_reset_index`, identity/filter/balance tests: PASS U/I. | AP for copy and position alignment. obsid/fits_name/source_id not returned in meta: provenance limitation, not silently claimed as supported. Feature cache preserves IDs privately; spectral loader has no equivalent output. |
| `get_feature_names` (580) | Before load: RuntimeError. After successful load: names in X-column order. | G initial state, permutation and scaling tests: PASS U/I. | AP; returned list mutability and state after failed reload not specified/tested. |
| `inverse_transform` (586) | Identity without scaler, physical values after a corresponding successful scaled load. | G initial state and scaling round-trip: PASS. R `unscaled_reload_does_not_reuse_previous_scaler`: FAIL. | AP nominal; #51. Invalid reconstruction shapes delegated to sklearn, not exhaustively tested. |
| `SpectralMatrixLoader.__init__` (624) | Directory/catalog paths, seed/jobs; common numeric wavelength vector or DR5 default. No disk mutation. | G `default_spectral_grid_and_recursive_index_cache`: PASS. R `target_grid_requires_finite_nonempty_vector` (empty, 2D, NaN): 3 FAIL. | AP default grid; PA validation #55. Early validation is the proposed remedy; exact API error policy still needs review. Quantities/alternate units not supported by a tested contract. |
| `_build_fits_index` (646) | Recursive name→path lookup; cache reused. Unique `.fits` and `.fits.gz` names resolve. | G recursive gzip/cache test: PASS. R `duplicate_fits_basenames_do_not_silently_choose_one`, `index_does_not_treat_partial_download_as_fits`: 2 FAIL. | #53 confirmed ambiguous basename. `.fits.part` inclusion is observed hygiene risk, not demonstrated wrong scientific assignment (requires catalogue name match). Cache invalidation, links and case sensitivity not exhaustively tested. |
| `inspect_sample_fits` (670) | Return diagnostic text; continue after corrupt files; n<=0 inspects none. No changes to FITS. | G `fits_inspection_valid_invalid_and_sample_bounds`, `inspection_accepts_1d_and_empty_primary_hdu`: PASS I. | AP for 2D/1D/empty/corrupt diagnostics, absent headers, shape/range/NaN count. Ordering of recursive inspection is unspecified; no claim of random sampling. |
| `SpectralMatrixLoader.load` (730), catalogue | SNR/class filters before I/O; seed-limited sample, then resolve files; preserve chosen row order. pandas/joblib. | G `spectral_filters_sampling_and_order`, `spectral_catalog_without_snr_and_missing_catalog_error`: PASS I. R empty spectral batch (3): FAIL. | AP nominal/seed; #55 empty vstack. Duplicate catalogue rows intentionally follow catalogue multiplicity; no object-group split validated. |
| Nested `_load_one` (812), valid FITS | Primary row 0 flux, log10 header λ; affine interpolation, endpoints, median when positive, no scaling when median<=0. Astropy/scipy; read-only file context. | G `fits_analytic_linear_interpolation_and_median` (4), `nonpositive_median_is_left_unscaled` (2): PASS I/P. R zero native λ step: FAIL. | AP analytic cases; #52 degenerate λ. No air/vacuum/rest correction or general WCS interpretation. Negative continuum semantics merely characterized by documented positive-median branch. |
| Nested `_load_one`, invalid FITS / resolution | Missing/corrupt/no-data/1D/3D/no-header rows rejected with diagnostics; valid records retain metadata. | G `invalid_fits_rejected_with_diagnostic_and_alignment` (5), `all_missing_fits_returns_empty_aligned_arrays`: PASS I. | AP local rejection; broad except converts exceptions to NaNs. No fail-fast mode; interruption/permissions/disk races not exhaustively tested. 1D inspection works while spectral load rejects it: supported input is explicitly row-based. |
| `load`, invalid pixels / coverage (865–887) | Out-of-support values are missing, excessive missing coverage rejects rows; residual fill is documented 0. | G `partial_boundary_fill_and_too_short_spectral_coverage`, `nan_flux_masking_and_rejection` (3): PASS. R exact 20%: FAIL; Inf (2): FAIL; zero invvar: FAIL. | #52 Inf becomes max-float; related known quality issue #34 for ignored inverse variance. Exact 20% policy ambiguous (#55). Zero fill is characterized, **not** certified physically neutral or continuum-preserving. |
| `load`, stack / y/meta / workers (851–892) | Serial and parallel outputs and rejection masks must agree; labels and physical metadata correspond to each surviving file. | G `worker_count_does_not_change_spectral_results` with real 2-worker loky and corrupt middle record: PASS I. Other spectral tests check empty/nonempty shapes and metadata. | AP finite synthetic workload; no proof for every job count, oversubscription, native-library platforms, process failures or large files. Stable object/file IDs are absent from returned meta (traceability gap). |

## Findings and red-proof disposition

All **30** red cases are preserved unchanged. Counts below are test cases,
not independent bug counts. Every case is assigned a finding or an explicit
policy/limitation disposition.

| Follow-up | Cases | Classification / reproduced behavior |
|---|---:|---|
| [#49](https://github.com/PhD-Brown/AstroSpectro/issues/49) | 3 | NEW: catalogue class/subclass columns joined twice; y becomes (N,2). |
| [#50](https://github.com/PhD-Brown/AstroSpectro/issues/50) | 3 | M1 cardinality risk now reproduced; NEW missing-key cross-match and >2^53 CSV identity rounding counterexamples. No real catalogue prevalence claimed. |
| [#51](https://github.com/PhD-Brown/AstroSpectro/issues/51) | 1 | NEW: scale=True then False leaves old scaler; inverse transformation corrupts raw second load. |
| [#52](https://github.com/PhD-Brown/AstroSpectro/issues/52) | 3 | NEW: Inf becomes max-float with normalize on/off; COEFF1=0 admitted as a valid spectrum. |
| [#53](https://github.com/PhD-Brown/AstroSpectro/issues/53) | 1 | NEW: ambiguous identical basenames silently resolve to the first enumerated path. |
| [#54](https://github.com/PhD-Brown/AstroSpectro/issues/54) | 2 | NEW: missing labels vanish during balancing without diagnostic; log reports sampled count as original count. Fail-fast missing-label policy remains reviewable. |
| [#55](https://github.com/PhD-Brown/AstroSpectro/issues/55) | 13 | Invalid mode/nonfinite raw features/no usable columns/unvalidated grid inputs reproduced. Empty returns (5 cases), early-grid-validation timing (3) and exact 20% threshold (1) require contract decisions; do not count every assertion as a confirmed independent bug. |
| [#23](https://github.com/PhD-Brown/AstroSpectro/issues/23) | 2 | KNOWN: non-spectral Gaia proxies and unknown provenance admitted by spectro_only. Spectral synthetic_BV is retained in this loader. No complete T3 registry proof. |
| [#34](https://github.com/PhD-Brown/AstroSpectro/issues/34) | 1 | Related KNOWN pixel-quality gap: zero-invvar outlier enters raw matrix unchanged. Bit-specific andmask/ormask policy remains untested. |
| Inspection hygiene / [#37 context](https://github.com/PhD-Brown/AstroSpectro/issues/37) | 1 | SUSPECT: glob includes .fits.part. Observed index entry only; no wrong nominal-name resolution or corrupted scientific result demonstrated. Review whether filtering is wanted. |

Duplicate searches preceded issue creation: GitHub queries for data_loader,
jointure/cardinality/identifiants/duplicate, scaler/interpolation/flux/NaN and
empty/balance/20%/fits_index, plus direct reads of #21–#24, #34/#35. No matching
targeted corrective issue was found for #49–#55. These are corrective follow-ups,
not a second audit tracker. #22 is about stellar grouping, not CSV joins.

Two initial red CSV assertions were **test errors**, not production bugs: they
expected ParserError for an unterminated quoted record despite the explicitly
documented warn-and-omit fallback. They were replaced with checks of the warning
and omitted result, not relabeled as regressions. A third initial test accidentally
wrote a catalogue with no columns while intending a missing-key schema; the
fixture was corrected to retain a non-key column. The final proof assertions
were not loosened to obtain green tests.

## Execution evidence

Actual environment: author's Windows PC, `.venv`, Python **3.11.9** (MSC v1938
64-bit), pytest **8.4.2**, pytest-socket **0.7.0**, NumPy **2.4.6**, pandas
**3.1.1**, SciPy **1.17.1**, Astropy **8.0.2**, scikit-learn **1.9.1**, Black
**24.4.2**, Ruff **0.5.0**. No dependency download or environment installation.
Commands below use repository-root CWD. Initial executions were on main's
production SHA with the added uncommitted tests; they were then repeated on
the exact committed test SHA linked above. JUnit artifacts are local/ignored
under `test-results/`, not scientific data or versioned evidence substitutes.

| Exact command | Result |
|---|---|
| `.venv/Scripts/python.exe -c "import sys,importlib.metadata as m; print(sys.version); print('\n'.join(p+'=='+m.version(p) for p in ['pytest','pytest-socket','numpy','pandas','scipy','astropy','scikit-learn','black','ruff']))"` | Versions above. |
| `.venv/Scripts/python.exe -m pip check` | No broken requirements found. |
| `.venv/Scripts/python.exe -m pytest -q tests/test_dimred_data_loader.py --junitxml=test-results/m21-green.xml` | **51 PASS**, no skip/xfail/warning. |
| `.venv/Scripts/python.exe -m pytest --junitxml=test-results/m21-fast.xml` | **74 PASS**, no skip/xfail/deselection; 7 existing Gaia/pandas Pandas4Warning messages (#16). |
| `.venv/Scripts/python.exe -m pytest -q tests/proofs_dimred_data_loader.py --tb=line --junitxml=test-results/m21-proofs.xml` | **30 FAIL**, exit 1, no skips/xfails; expected audit evidence, not a successful suite. |
| `.venv/Scripts/python.exe -m black --check src tests scripts app.py` | **29 files unchanged**. |
| `.venv/Scripts/python.exe -m ruff check src tests scripts app.py` | All checks passed. |
| `git diff --check` / `git diff --cached --check` | Passed. Commit hooks Black/Ruff also passed on the three test files. |

Black was used to format **only the three new test files** before these
read-only checks. Initial green-file development run: 37 PASS / 3 FAIL
(the test mistakes explained above); subsequent 42 PASS, then 51 PASS after
additional boundary and worker cases. Red proofs first ran 27 FAIL and then
30 FAIL after adding three invalid-grid cases. No red assertion was deleted.

Default pytest configuration blocks sockets and excludes slow/network markers;
the 74 collected cases have no marker-based deselections. The 30 proof cases
are **not automatically discovered** by filename and are separately executed.
Historical 83 tests on PR #48 include its 60 storage tests; this main-based
delivery has the 23 main tests plus 51 new cases. It does not claim to rerun #48.

Hosted CI evidence will be linked in the Draft PR and #43 handoff with its head
SHA. This document's PASS/FAIL counts are local evidence. No numeric line/branch
coverage was measured (`coverage` is not installed); no assertion of exhaustive
physical validity follows from the method inventory or any green CI.

## Residual limits and review handoff

The high-risk normal, malformed, identity, numerical and multiprocessing paths
are represented by implemented tests. Remaining gaps are explicit: stable IDs
in returned meta, all producer provenance columns, real WCS/unit/frame variants,
bit-level masks/sentinels, policy for zero continuum and missing coverage,
permutation-invariant sampling (not promised), invalid parameter types/ranges,
failed-load state, cache refresh/path aliases, platform case rules, interrupted
I/O/worker death and large datasets. These are **not accepted risks by default**;
the reviewer/scientific owner must decide their disposition. No service, GPU,
download, real catalogue, model, notebook execution or scientific run was used.

Next actions are independent Claude review of tests/oracles and classifications,
policy decisions for #55/#54 and extraction/integration of accepted green tests.
Corrections proceed only in the linked issues. M2.1 stops here; no automatic
audit of PCA or other modules, no merge, no edits to main or the PR #48 branch.
