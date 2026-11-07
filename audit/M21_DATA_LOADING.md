# M2.1 — reviewed data-loading coverage and evidence

2026-11-07 · [Refs #43](https://github.com/PhD-Brown/AstroSpectro/issues/43) ·
Draft PR #56, awaiting Claude's second independent review.

Production reference: `main@2f2a444fb5b03e1085c8d439cb3825cee5b1c766`.
The complete 892-line `src/dimred/data_loader.py`, all 15 significant methods /
functions, current AGENTS.md, #43 with all comments, [Claude's complete review
and 18 inline threads](https://github.com/PhD-Brown/AstroSpectro/pull/56#pullrequestreview-5814505492),
and coordination were read. Initial checkout: clean on `1e35c8e` and existing
branch `audit/43-dimred-data-loading`. The loader is unchanged from M1 `35c2105`.
No production, notebook, dependency or CI file was modified. No review thread
was resolved, no PR merged, no M2.2/PCA work started.

## Integration and durable archive

The integration diff contains **57 green cases** in
[test_dimred_data_loader.py](../tests/test_dimred_data_loader.py), prefixed
[fixtures](../tests/conftest.py), this matrix, and an explicit optional
[mutation replay](check_m21_mutations.py). **No red proof file remains in the diff.**

Before removal, branch **`audit/43-dimred-proofs`** was pushed and verified
accessible on GitHub at fixed SHA **`68cb36828440671bee57e90ff763be4ebfbaaf93`**:

- [Corrected executable proofs](https://github.com/PhD-Brown/AstroSpectro/blob/68cb36828440671bee57e90ff763be4ebfbaaf93/tests/proofs_dimred_data_loader.py).
- [Per-test classification, results and historical correspondence](https://github.com/PhD-Brown/AstroSpectro/blob/68cb36828440671bee57e90ff763be4ebfbaaf93/audit/M21_PROOFS.md).
- Matching fixtures/source are in the same commit. Issues **#49–#55** and
  **#57** received those fixed links **before removal**. #23/#34 also receive
  updated references. No additional corrective PR was created.
- Ancestor `1e35c8e8f0cc4aac44ad751729a7c0f54c08c00b` retains **all 30 original
  cases**, assertions and fixtures. No history was rewritten. The revised
  archive has **46 cases: 35 FAIL / 11 PASS / 0 ERROR / 0 SKIP**; these include
  positive controls, observations and one failing hygiene preference, not 35 bugs.

Run proofs from that archived checkout, not the final PR checkout. Every test
has a Refs/classification docstring. The manifest maps original scenarios to
retained proofs, accepted-contract replacements or retracted interpretations.
Retraction does not delete the original input. No xfail masks an error.

## Accepted policies and independent oracles

The owner's 2026-11-07 instruction validates #55: **N=0 and D=0 require explicit
domain errors**; **missing fraction >=20% rejects a spectrum**; NaN SNR cannot
satisfy an **active** quality filter; target grid must be **1D, nonempty, finite,
strictly increasing**. A compatible explicit no-filter mode remains to be
designed separately: this audit does not reinterpret `snr_min=0` or fix production.
Archive message regexes distinguish domain diagnostics from incidental dependency
errors; exact final wording is not prescribed. Linear versus logarithmic target
sampling remains open to independent physical study.

Oracles: left-row identity and independent class/temperature markers, population
moments (ddof=0), and analytical affine flux in Angstroms. Float64 tolerances
`2e-13` (interpolation), `1e-14` (scaling), and `np.spacing(WL_MAX)` (historical
grid step) reflect arithmetic precision, not observational uncertainty. A target
covering only the red half makes its median strictly different from the native
median. Coverage tests use **19, 20 and 21 out-of-support pixels out of 100**,
avoiding NaN propagation ambiguity. Valid companion spectra prevent rejection
tests from imposing an empty return. Zero fill and nonpositive-median behavior
remain **characterizations**, not claims of physical neutrality.

Filters now have an exact no-sampling oracle: SNR=10 retained, 9 excluded, QSO
excluded, order and metadata exact. Seed 17 repeats; seed 23 changes the subset;
`head(3)` fails. Active SNR=10 excludes NaN in both loaders. The alternate-band
test retains two variable rows, avoiding its former (N,0) dependency. Default
instrumental admission is annotated as historical D1/#23, not an allowlist.

## Method and significant-path matrix

All line numbers refer to unchanged production. **Existing tests before M2.1:
none for this module** (23 original environment/Gaia cases were unrelated).
G = green file, A = fixed archive above; test names omit `test_`. U/I/P/S =
unit/integration/numerical/scientific-risk proof. AP is restricted to the named
contract; PA denotes a gap, not accepted risk. No method gets AP merely by reading.

| Code and role / I→O | Preconditions, dependencies, effects | Executed tests / results | Evidence and limits |
|---|---|---|---|
| Constants/import boundary (1–165) | numpy/pandas/sklearn; exclusion rules, no source-import I/O | G column selection and grid: PASS U. A spectro-only: 2 FAIL. | AP mechanics; known #23. importlib isolates source from optional Torch facade #21, not a test of `import dimred`. |
| `DimRedDataLoader.__init__` (184) | Paths/key/seed → empty state; no file reads | G initial state, custom key, fixture independence: PASS U/I. | AP construction. Prefixed fixture names and per-instance CSV directories avoid future collisions. General parameter types not exhausted. |
| `_load_features` (343) | CSV→frame; file required; C→Python fallback permits bad-row omission **with warnings** | G missing/empty/truncated/malformed CSV: PASS U/I. A ID inference: FAIL. | AP parser paths, #50 exact identity. N6 CSV numeric strings is an archived observation; encodings/permissions not exhaustive. |
| `_load_catalog` (388) | Catalogue CSV→frame; analogous fallback and errors | Same G catalogue cases: PASS. A asymmetric key inference: FAIL. | AP named I/O paths, not real catalogue validation. |
| `_merge` (407), nominal | Preserve left feature observations, order and feature-side precedence; pandas, no writes | G 4 permutations, custom key, missing key and repeated features: PASS U/I. | AP unique-right-key contract. Observation IDs do not prove stellar grouping (#22). |
| `_merge`, adversarial | Labels once; missing identity is not identity; no right-key amplification or rounding | A labels: 3 FAIL (#49); conflicting/identical keys, NaN and large key: 4 FAIL (#50). | M1 cardinality risk reproduced; N3 identical values also amplify. Large-ID rounding requires BOTH missing key and >2^53, now explicit source_id path; no ordinary obsid prevalence claimed. |
| Feature `load` (206), filters/NaN | CSV→merge→quality/classes→columns→drop incomplete rows | G identity filters, absent/alternate SNR with variable rows, column-then-row missingness, active NaN SNR: PASS I. | AP paths above; precedence characterized, not independently reconciled. N2 zero-threshold API remains #55 design work. |
| `_select_feature_columns` (428) | Numeric non-meta/nonconstant, input order, inclusive NaN threshold | G order/threshold/bool/single-row mechanics: PASS U. A provenance: 2 FAIL. | AP mechanics, #23/D1. Default instrumentals historical; object dtype in memory differs from CSV inference (N6). Full producer registry T3 absent. |
| Feature `load`, scaling/cache | Float X and y, fit scaler to current population, cache final frame | G population moments/inverse/order and scaled Inf rejection: PASS I/P. A stale scaler + failed transaction: FAIL; N7 observations PASS. | #51. Descriptive fit is not train-only proof (#24), stellar split proof, or failure-transaction guarantee. |
| Feature `load`, empty/invalid | Explicit N=0/D=0 domain diagnostics, valid mode, finite raw matrix | G spectra delegation before I/O: PASS. A empty×2, D=0, Inf×2, mode: FAIL (#55). | N=0 already raises loudly in sklearn/concat, but lacks a domain error. No green (N,0) load assertion remains. |
| `_balance` (543) | Seeded per-class cap, no replacement, small classes kept | G cap/seed/unique/alignment: PASS. A missing-label loss + original-count log: 2 FAIL. | #54. Rejection vs announced exclusion is still a proposed remedy. Equal sizes and permutation-invariant sampling not promised. |
| `_build_meta` (563) | Available physical/SNR/class/coordinate fields → independent RangeIndex copy | G copy/reset/identity/filter/balance: PASS U/I. | AP positional alignment. Stable IDs absent from returned meta: traceability gap; only feature private cache keeps IDs. |
| `get_feature_names` (580) | Before load RuntimeError; after success X-order names | G initial/order/scaling: PASS. A N7 failed-load consistency: FAIL. | AP successful calls; mutability and failure atomicity not certified. |
| `inverse_transform` (586) | Identity without scaler; reverse corresponding scaled batch | G identity/round-trip: PASS. A scaled→unscaled reload: FAIL. | #51; invalid shapes delegated to sklearn, not exhaustive. |
| `SpectralMatrixLoader.__init__` (624) | Paths/jobs/seed; explicit vector or default numeric grid | G historical default step: PASS U. A target empty/2D/NaN×3 + nonincreasing×2: FAIL. | #55 accepted validation. Default 3921 pixels observed; this is not physical endorsement of linear sampling (N5). |
| `_build_fits_index` (646) | Recursive basename→path cache; unique FITS/gzip | G recursive gzip/cache: PASS. A ambiguous basename FAIL (#53); .part preference FAIL; uppercase observation PASS on Windows. | .part is hygiene, not demonstrated wrong assignment. N8 Linux behavior from Claude; aliases/cache refresh not exhaustive. |
| `inspect_sample_fits` (670) | Text diagnostics, n<=0 none, corrupt input continuation; read only | G exact count, 1D/2D/empty/corrupt/header paths: PASS I. | AP diagnostics; recursive order unspecified, no random sampling claim. |
| Spectral `load` (730), catalogue/sample | Filters before FITS I/O, seeded sample, chosen order preserved | G exact no-sampling filters, distinct seeds vs head, reverse order, missing catalogue/SNR field: PASS I. A empty catalogue/filter/zero sample: domain-error FAIL. | AP now discriminated by mutation. No stellar split or real producer validation. |
| Nested `_load_one` (812), valid arrays | Primary row 0, log10 header λ, scipy linear interpolation, positive median | G affine endpoints/midpoints, native-vs-target median, native descending support: PASS I/P. Nonpositive-median characterization PASS. | Bare Å assumed, no WCS/air-vacuum/rest/flux-unit conversion. Descending native support differs from required increasing target. |
| `_load_one`, malformed files | Reject missing/corrupt/no-data/1D/3D/header errors; context closes FITS | G invalid structures with valid controls: PASS I. Corrected native zero step with valid target: PASS. | COEFF1=0 bug claim retracted from #52; coverage rejection is not a universal native-grid validator. Permissions/interruption not exhaustive. |
| Spectral `load`, coverage/pixels (864–880) | Reject >=20%; residual fill 0 | G 19/20/21%, NaN fractions and support with valid controls: PASS I/P. A Inf→max-float×2 FAIL (#52), invvar=0 FAIL (known #34). | Exact bound now accepted; misleading `>20%` production comments/log remain #55 follow-up. Masks/physical fill validity unproved. |
| Spectral `load`, stack/y/meta/workers | One rejection mask, order, real loky workers | G serial/two-worker corrupt-middle parity: PASS I. A diagnostic control jobs=1 PASS; jobs=2/4 FAIL (#57); all-files-missing domain error FAIL (#55). | Parity does not prove logging. Worker socket guards verified; no real service/GPU. Worker death/large workloads untested. |

## N1–N8 and impact qualification

| Finding | Recheck / disposition |
|---|---|
| N1 worker diagnostics | Parent handler sees file error at jobs=1, loses it at 2/4; valid output survives. New [#57](https://github.com/PhD-Brown/AstroSpectro/issues/57), after duplicate search worker/diagnostics/loky/logging. Raw worker stderr may still contain warning. Default API jobs=4 justifies P2 observability, not a corruption claim. |
| N2 SNR NaN | Threshold-zero historical observation PASS: features N=2, spectra N=3. New green active-threshold-10 case rejects NaN in both. Compatible explicit no-filter mode to design in #55. |
| N3 identical catalogue keys | Archive FAIL: N=2→3 despite equal metadata; added to #50. Oracle does not prescribe rejection versus explicit deduplication. |
| N4 invalid target | Two nonincreasing rejection proofs FAIL. Three current observations PASS: empty (0,0)+warning, 2D IndexError, NaN accepted as zero. Accepted validation #55; noisy failures distinguished from silent acceptance. |
| N5 default grid | 3921 pixels observed, docstring approximately 3909. Claude raised sampling/flux conservation/aliasing questions. **No independent physical study here, no confirmed linear-grid bug or corrective issue claiming one.** The median oracle does not settle this question. |
| N6 CSV inference | Numeric-looking strings become numeric features (archive PASS); the in-memory object test is narrower. D1/#23, not evidence of a specific real leaked ID. |
| N7 failed reload | names=[] and previous cached frame reproduced. Without balancing scaler is a **new unfitted instance**; with balancing concat fails before replacement and old scaler stays. This refines Claude's summary. Rollback-vs-full-reset remains #51 policy; rollback proof FAIL, two state observations PASS. |
| N8 case sensitivity | Uppercase FITS inclusion reproduced on Windows. Absence under Linux reported by Claude, not independently rerun in this archive here. No renaming or new priority imposed. |

**No existing priority was changed.** Usage inspection: PCA notebook passes
`RAW_DATA_DIR` to SpectralMatrixLoader (near line 1936), API defaults jobs=4;
#53 ambiguity depends on directory organization, e.g. a parent with duplicate
basenames. #49 needs a producer missing class; usual pipeline tables carry it.
#51 needs reload/inverse sequence. These qualify exposure without real prevalence
measurements or an unapproved P1→P2 decision.

## Mutation evidence

Claude **reported** 22/31 killed, 9 survivors on the old 51 cases; his script was
not published. Our versioned replay reconstructs **30 described mutations**, not
that exact experiment: **29 killed / 1 survived**, all 57 green cases run per
candidate, no collection/fixture errors/skips. Eight of nine named former
survivors are killed (classes, SNR, head, native median, three threshold variants,
rejection log). `index_keep_last` remains an expected survivor until #53's fix.
Every substitution and execution is versioned; archive manifest lists outcomes.

The replay uses disposable copies of source/tests/fixtures/pytest config,
validates exactly one replacement site, checks execution failures, records input
hashes/failing test names in local JSON, and never edits repository production.
It is explicit, outside the automatic fast suite. A survivor is a coverage
observation, not automatically a bug; numeric line/branch coverage is unmeasured.

## Actual execution

Windows PC, existing `.venv`: Python **3.11.9**, pytest **8.4.2**, pytest-socket
**0.7.0**, numpy **2.4.6**, pandas **3.1.1**, scipy **1.17.1**, astropy **8.0.2**,
sklearn **1.9.1**, joblib **1.6.0**, Black **24.4.2**, Ruff **0.5.0**. No install
or download. Revised tests/proofs are at archive SHA `68cb368`; final split only
removes archived files and rewrites this matrix. Final head/hosted CI evidence
is linked in PR #56 and #43 handoff separately from local results.

| Exact command (repository root) | Result |
|---|---|
| `.venv/Scripts/python.exe -m pytest -q tests/test_dimred_data_loader.py --tb=short` | 57 PASS, no warnings/skips/xfails. |
| `.venv/Scripts/python.exe -m pytest -q --junitxml=test-results/m21-review-fast.xml` | 140 PASS, 7 existing Gaia/pandas warnings, no skips/xfails. |
| Archive checkout: `.venv/Scripts/python.exe -m pytest -q tests/proofs_dimred_data_loader.py --tb=line --junitxml=test-results/m21-review-proofs.xml` | 35 FAIL / 11 PASS / no errors/skips, exit 1. JUnit verifies classifications, not bug count. |
| `.venv/Scripts/python.exe audit/check_m21_mutations.py --output test-results/m21-mutations.json` | Baseline 57 PASS; 29/30 reconstructed mutations detected, 1 expected survivor. |
| `.venv/Scripts/python.exe -m black --check src tests scripts app.py audit/check_m21_mutations.py` | Unchanged: 31 files before split, 30 after proof removal. |
| `.venv/Scripts/python.exe -m ruff check src tests scripts app.py audit/check_m21_mutations.py` | All checks passed. |
| `.venv/Scripts/python.exe -m pip check` | No broken requirements found. |
| `git diff --check`, `git diff --cached --check`, `git diff origin/main...HEAD --check` | Passed; production/config/notebook diff empty. |

Local JUnit/JSON live under ignored `test-results/`. Final 57 green cases replace
51: one all-missing empty-return case moved to accepted-error archive, seven
cases added. Other rejection tests retain valid controls. Old return-empty and
20%-accepted expectations were superseded **by the owner's decision**, not to
hide failures. Original COEFF1=0 double-degenerate construction was a test error,
corrected and retracted. All original evidence remains reachable by durable ref
and history; no assertion was silently discarded before preservation.

## Limits and handoff

Stable IDs in returned meta, full producer provenance registry, bit masks,
sentinels, WCS/units/frames, transactional failure policy, cache refresh/aliases,
physical missing-data/interpolation policy, native variants, worker interruption
and large workloads remain limited or untested. Historical characterizations do
not grant scientific approval; N5 requires independent physical study. Tests
use only synthetic temporary CSV/FITS, blocked sockets (including guarded real
workers), no model, service, GPU, real catalogue, notebook or scientific run.
Corrections stay in their issues. Await second Claude review and human merge
decision; do not start M2.2 automatically.
