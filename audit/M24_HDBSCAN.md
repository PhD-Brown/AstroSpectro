# M2.4 — HDBSCANAnalyzer contracts and scientific limits

2026-11-08 · [Refs #43](https://github.com/PhD-Brown/AstroSpectro/issues/43).
Tests/audit only; submitted for independent Claude review and human decision.
No M2.5 work. A green suite is not astrophysical validation.

## Revision after Claude's first review R1–R10

Read the [complete review 5820017025](https://github.com/PhD-Brown/AstroSpectro/pull/79#pullrequestreview-5820017025),
all ten inline threads and [handoff](https://github.com/PhD-Brown/AstroSpectro/issues/43#issuecomment-6728219398),
live AGENTS.md and #43. Both worktrees were clean, head/remote e11ad1b,
main58180392, proof ref a27028b. Same worktree/branch/Draft PR79, no new audit
branch, production change or thread resolution. This revision supersedes the
initial AP claims for mean-versus-median discrimination and returned-row identity.

| Review | Correction / independent expectation | Evidence / open policy |
|---|---|---|
| R1 | `asymmetric_physical_cluster_and_noise_are_arithmetic_means`: Teff1000/1200/2000 gives1400±529, noise9000/9100/9800 gives9300. `asymmetric_feature_mean_and_complete_returned_population`: x0/1/5/10/4, mean4,var62/5, scores−2/sqrt(62/5),6/sqrt(62/5). | Three separately scoped median substitutions detected. ddof1 physical vs ddof0 scaler preserved; no classification inferred. |
| R2 | Two complete returned frames compared row-by-row to raw **or** independently standardized observations, one coherent policy for the whole frame. NaN at ID2 retains0,1,3,4,5,6,7,8,9; nonmonotone second feature protects pairing. | Reversed rows, independently sorted columns and head deletion detected; an additional mixed-units candidate is tested. Correct standardized #77 return and mismatch refusal #73 remain admissible. |
| R3 | Explicit missing-statistic helper accepts nan/N/A/dashes, checks a defined mean exactly and refuses a numeric-zero singleton dispersion. | Alternate dash/N/A candidates remain green; undefined→0 is an incorrect guard candidate. No unique missing display required. |
| R4 | Removed exact insufficient-Gaia prose. Checks cluster/count and absence of the existing unsupported taxonomy tokens when no physical parameter was selected. | Reformulated cautious text remains admissible. This lexical guard is not a universal validator of scientific language. |
| R5 | RGBA validity, ID coverage, distinct colors for at most3clusters, dedicated noise, stability under paired order permutations in both modes. | No colormap/LUT/rank formula or exact grey. Original `palette_swapped_ranks` is now an admissible aesthetic alternative, excluded from incorrect-mutation totals; other original sites retained. |
| R6 | Both modes accept relevant ValueError refusal or explicit−1 for invalid rows, with valid-row partition verified against a finite-only real backend fit. | No physical low-density interpretation of invalid coordinates. Normal prediction_data=True kept; presentation option OPEN. Real controls cover prediction_data=True and uniform finite refusal. |
| R7 | Three M2.4 fixtures moved into conftest, standalone support/plugin removed. | Whole suite `--import-mode=importlib`385PASS on Windows, no path hack or pytest config change. Normal mode checked separately. |
| R8 | Joint mismatch proofs renamed/qualified; new archive controls isolate equal features=Z with ndarray vsSeries lengths8/12, same-length shifted Series and physical y Series. | Wrong associations demonstrated for direct Series inputs; annotation/nominal loader remain ndarray. Extending supported Series behavior is OPEN and attached to#73. No green architectural promise. |
| R9 | Primary paper confirms biased GSP-Phot[M/H] needs calibration/translation to LAMOST[Fe/H]; red-clump and hot-dwarf counterexamples qualify current thresholds. | Sources and scope below; no replacement taxonomy or physical threshold asserted by CI. Qualification attached to#78. |
| R10 | New schema2 names `git_head_at_run`, working-tree state, actual content hashes and Git blobs. Historical payload retained separately, without changing749224f into a later run SHA. | Historical inputs match committed d402922; revised campaigns run on committed input snapshot1b9fa1157146eb508e9b590704ef0a7b214b87e6. Final ledger distinguishes executions. |

Independent R2 central-NaN oracle: retained a sum43,sumsq281; b sum37,sumsq221.
Means43/9,37/9; population variances680/81,620/81. Cluster2 averages yield
−31/sqrt680 and−4/sqrt620; cluster7 yields13/(2sqrt680),7/(2sqrt620).
The embedding encodes the retained identities; labels already match this valid
population. Exact frame rows and aligned labels are checked together. Raw and
standardized matrices are accepted as complete alternatives, not column-wise
choices. Existing float64 tolerances3e−15 are unchanged.

Historical results in the later initial-ledger sections describe pre-review
executions only. New executions and corrected classifications are recorded in
the final revision ledger below and the PR, never substituted into old runs.

## Reference and scope

Audited `origin/main@58180392d16e06ad411073bc48011d076f9128f0`, fetched at
start, matching merge PR#72. Complete 562-line `src/dimred/hdbscan_analyzer.py`
read, Git blob **667a2985ea1b6b4f0a660a70f7635715730c0f78**. Work branch
`audit/43-dimred-hdbscan` starts at that main in separate worktree
`C:/Users/alexb/Documents/Projets/AstroSpectro-m24`. Original workspace remains
on clean `audit/43-dimred-embedding@59ba28d6a5d664e36b37b195fefe7484d5b85a8c`.
No stash/reset/cleanup, no main push, production change, notebook execution,
scientific data/model changes, dependency/lock/CI edits or merge.

Read live AGENTS.md v 1.3, complete#43/dashboard and relevant comments through
GPT's M2.4 preparation, merged PR#56/#64/#72 and matrices M 21/M 22/M 23. The
matrices' pending-review language is historical: all three merges were checked.
Open backlog, especially#21/#23/#24/#49–#71, was read for scope/deduplication.
Searches over all issue states for hdbscan/physical/align/feature_profiles found
no existing HDBSCAN-specific corrective ticket. #50 covers loader joins, #62/#67
PCA/embedding failed fits; neither is a duplicate of this module's state defect.

Consumers inspected: facade `dimred/__init__.py`, notebook
`notebooks/dimred/phy3500_02_umap_tsne.ipynb` source/calls (no execution), loader
selection/filter/scaler/meta interfaces, embedding producer and visualizer
HDBSCAN/sensitivity/feature-profile interfaces. No prior dedicated HDBSCAN tests
existed. Earlier embedding tests consuming cluster labels do not validate fitting.
Torch facade optionality remains known#21. Tests load actual source using
importlib; CSV helper uses actual loader constants under a lightweight package
boundary. These fixtures do not certify `import dimred` in a minimal environment.

## Contracts, tests and evidence matrix

G=`tests/test_dimred_hdbscan_analyzer.py`, R=`tests/test_dimred_hdbscan_real.py`,
A=archived `tests/proofs_dimred_hdbscan_analyzer.py`. Test names omit `test_`.
AP=acquired evidence for the stated contract only; PA=partial/open; NA=not
validated; B=blocked. U/I/P/S follow AGENTS.md. No line-coverage measurement.

| Code / input, state and output | Contract and independent oracle | Executed evidence / level | Residual limit |
|---|---|---|---|
| Constants/import boundary 1–45 | Six Gaia columns; numpy/pandas/Matplotlib; backend lazy. No catalog I/O on source import. | G initial/optional-import tests PASS U; real source blob checked. AP lazy HDBSCAN error, PA facade. | #21 packaging; names/units and spectral provenance not certified by denylist. |
| `__init__`64–90 | Stores 75/20/euclidean/eom or configured values; normal and presentation estimators/labels initiallyNone, zero counters, empty instance-owned lists/maps. | G `initial_state_and_instance_isolation`, `fit_forwarding_and_success_state`: PASS U. AP named defaults/state. | Constructor itself does not validate types/ranges; backend checks on fit. |
| `fit`94–132 | Z(N,2) positional numeric coordinates; HDBSCAN kwargs match configured size/samples/metric/selection, prediction_data=True; returnself, store estimator/labels then recompute stats/palette. No coordinate physical unit inferred. | G forwarding, preservation, successive fit, optional dependency/failure propagation PASS U. R actual EOM/leaf, degenerate/invalid/probability cases PASS I/P. AP successful wrapper contracts; PA failure-state coherence. | A four controlled failures plus2real observations show replaced estimator with stale cache; transaction choice OPEN. Valid shapes beyond documented 2 D, sparse/precomputed/custom metrics not exhaustive. |
| `fit_presentation`134–172 | Independent model and labels; size 300 default or override, same samples/metric/selection; sorted IDs without-1, tab 20 map; returns presentation labels. No prediction_data=True promise. | G `presentation_is_separate` PASS U; normal→presentation→normal→presentation replacement, ID coverage, RGBA validity and stable pairing, with unchanged opposite mode. Real nonfinite outcomes admit refusal or explicit masking. AP successful state, PA failure paths. | Empty/tiny real inputs fail; no empty-success contract inferred from mocks. No minimum granularity/count monotonicity guaranteed by changing size. |
| `_compute_stats`363–367 | Unique sorted labels except-1; countIDs; exact number of-1, not number of unassigned classes. | G `counts_and_palette_ids` all noise/no-noise/noncontiguous/multi/single plus `helpers_recompute_and_successive_fit_replace_state` PASS U/P. AP. | Labels assumed backend integer vector. Malformed synthetic backend values not a supported scientific input contract. |
| `_build_palette`369–377 | Current implementation uses turbo and tab20, with a noise entry. | G verifies ID coverage, finite RGBA, distinctness for small populations and permutation stability in both modes. AP these mapping properties. | Palette, grey and sampling formula are aesthetic choices, not fixed by green tests. No uniqueness claim beyond palette capacity. |
| `centroid`358–361 | Arithmetic barycenter of selected points; no medoid/density/physical-center claim; inputs unchanged. | G 3 analytic subsets including singleton and elongated cloud; translation, wrong-length mask PASS U/P. AP valid 1D-mask arithmetic. A empty/2D-mask/Inf observations PASS. | PA empty/nonfinite/matrix-mask policy OPEN. Float coordinates describe representation space only. |
| `physical_profile_table`176–338: preconditions/alignment | Requires labels; ndarray length mismatch is refused. meta index is reset; ndarray y assigns positionally, Series y aligns by pandas index (R8). | G before-fit and length errors, identity markers with nonconsecutive indices, paired row permutation, arbitrary ID renaming PASS U/I/S. A independent meta/y/label permutations and duplicate IDs expose limitations. AP positional behavior, PA identity validation. | Same length cannot certify correspondence. API has no fit IDs, so independent permutation violates documented alignment precondition; no new guaranteed detection imposed. |
| physical statistics 220–276 | Group means and pandas sample std(ddof=1), exact counts and unique dominant class; NaN excluded from numeric moments; singleton std remainsundefined, noise row has mean only. | G `physical_profiles_analytic_statistics_and_identity` float+nullable; custom/noisy/all noise/singletons/missing columns/partial missing class, local-change/noise-isolation PASS U/P. AP named finite/missing numerical contracts. | All available physical columns absent triggers internal pandas ValueError; all class labels missing triggers IndexError. Tie/missing/Inf/non-numeric/duplicate-column handling remains PA with archive observations. No silent assertion of std=0 for singleton. |
| physical formatting/interpretation 239–338 | Teff 0 decimals, others 2; noise mean, own count; interp same cluster index/count. Heuristics depend on mean Teff/logg/mh. | G manual expected strings/counts PASS U/P. A 6 exact-threshold display observations PASS. AP arithmetic/display fidelity; **NA astrophysical classification**. | Cluster mean can mix heterogeneous stars; gravity thresholds alone do not establish evolutionary stage. Noise→spectral anomaly is unproven. Gaia metallicity terminology requires qualification. No green assertion mandates heuristic prose. |
| `print_physical_profile`340–354 | Prints supplied table and interpretation rows; returnsNone; no DataFrame modification. | G `profile_print_transmits_tables_without_mutating`, `print_empty_supplied_tables` PASS U. AP structure/data. | Scientific force of title/prose remains unvalidated; whitespace/borders not treated as contracts. |
| `sensitivity_df`380–382 | Returns an empty DataFrame; module function has no analyzer reference and does not populate it. | G default property before/after sweep and after result mutation PASS U. AP existing independent state. | Placeholder/lacuna, not a persisted experiment table. Empty schema/ownership policy OPEN; no invented relationship. |
| `compute_sensitivity`388–429 | Default 50,100,150,300,500 or caller order; each estimator gets samples 20(default), euclidean/eom; counts exclude-1, percent round(100*Nnoise/N,2). | G full/default/custom/duplicate/empty grids, all noise/no-noise/1 of 6/2 of 6 fractions, trial inputs and propagation PASS U/P; R direct-backend sweep PASS I. AP positive-N contracts. | Empty grid returns no-column frame, schema OPEN. Invalid entries delegated to backend; no monotonicity/determinism beyond selected examples. Artificial N=0 backend response divides by 0; real backend refuses N=0. |
| `compute_feature_profiles`432–530: CSV/selection | Path stem.csv; lazy real StandardScaler/loader constants, fallback local import, FileNotFoundError. Active SNR>=10 includesboundary/excludesmissing; numeric variable columns, missing<=10%, metadata/prefix exclusions; dropna on retainedfeatures. | G curated exact columns vs spectro_only loader, thresholds 0/10/20%, SNR boundary, no SNR, empty/constant/Inf populations, missing file/fallback PASS U/I. AP observed mechanisms; PA spectral provenance. | Matches spectro_only=True on controlled population, not default loader (instrumental policy differs). Loader permits filters/classes/balancing/sampling/custom merge key; this function cannot inherit those. #23 denylist is not a provenance certificate; unknown plateid remains admitted. |
| feature alignment 532–545 | Current code compares feature count with Z, truncates positionally on mismatch; no IDs accepted. | A SNR/dropna remove middle104:2FAIL; joint features/Z/label mismatch:2FAIL (redundant for label-cardinality validation). Equal-size ndarray mismatches are already refused; Series observations isolate index alignment. **PA defect proven**, no AP provenance. | Wrong association demonstrated on synthetic mismatched populations; calling this “conservative intersection”/“perfect alignment” is unjustified. Reject mismatch or propagate IDs/selection record are alternatives. No historical rate measured. |
| feature scaling 547–561 | Default centering and population std(ddof=0) fit on all retainedrows **including noise**; grouped z-score means exclude noise only at aggregation. Column order and returned observation identities checked under either complete raw or standardized #77 policy. | G asymmetric means, exact returned populations including central NaN, scaler boundary, paired permutation, single feature/all-noise/post-dropna constant PASS I/P. AP descriptive numeric contracts. | Full-population reference is descriptive, not a train/test evaluation. Train-only required if reused predictively; none is run here. df_feat returnedraw despite standardized docstring; output API choice OPEN, archive raw observation PASS. |
| Interfaces | Loader CSV→scaledX→realPCA→controlled clustering→physical/feature profiles; actual HDBSCAN separate. Visualizer/notebook accept normal/presentation arrays and sensitivity tables. | G `loader_pca_cluster_profile_observation_identity` PASS I with shuffled catalog join and 1000/1200 vs 6000/6200 K oracle. AP tested boundary. | No real UMAP→HDBSCAN→astrophysical validation, large data, notebook execution or visualizer rendering claim. M2.8 owns visualization audit. |

## Independent numerical oracles

Physical means/std are hand-derived. C2 temperatures 1000/1400 give 1200 and
sample std=sqrt(80000)=282.8427…→283; C7 6000/6200/6400 gives 6200±200.
Gravity values 1/3 give 2±sqrt 2;4/5/NaN gives 4.5±sqrt(0.5). Dominant classes
have unambiguous majorities, so no tie-break policy is frozen. Integer counts
and string formatting are exact. Singleton/all missing dispersions are left
undefined rather than replaced by zero. Nullable Float64/float64 compare alike
for the finite/missing cases tested; nullable-all missing display is archived.

Feature example a=[0,2,6,8], b=[1,5,1,9], labels=[7,-1,2,2]. Population means
4/4 and variances 10/11 include noise. Expected cluster 2 scores 3/sqrt 10 and
1/sqrt 11; cluster 7 -4/sqrt 10 and-3/sqrt 11. Positive affine changes of units and
paired row permutations preserve standardized profiles. After dropping the
first of 10 rows with oneNaN, values 1…9 give mean 5,var 20/3; this distinguishes
correct dropping from head truncation. A feature varying before dropna but
constant afterward must scale tozero using StandardScaler's constant handling.
Absolute tolerance 3 e-15 for O(1) standardized means,2 e-14 for translated
centroids allows float64 rounding; relative tolerance 0. No observational
uncertainty or significance threshold is inferred. Text rounding is not a
high-precision statistical oracle by itself; exact feature means supplement it.

Sensitivity independently uses 1/6→16.67%,2/6→33.33%,0→0%,6/6→100%; duplicate
grid entries must retain trial order/multiplicity for current sequence output.
No monotonic cluster-count assertion and no sum-to-one probability assertion.

## Real HDBSCAN versus controlled backend

Ordinary venv has no hdbscan. Installed **hdbscan 0.8.44** with
`uv pip install --python ../AstroSpectro/.venv/Scripts/python.exe --target test-results/m24-optional-deps --no-deps hdbscan`.
Only scoped PYTHONPATH selects it. Project venv/dependencies/CI unchanged.
Windows 10 AMD64/Python 3.11.9; numpy 2.4.6/scipy 1.17.1/sklearn 1.9.1.
Two 12-point Gaussian groups centered(-10,0)/(10,0), std 0.1, RNG 17, plus(0,100),
with size 4/samples 2. Tiny dense fits use actual backend defaults, including
core_dist_n_jobs=4/approx_min_span_tree=True, no GPU. No global random seed
or cross-platform partition guarantee is asserted.

**21 optional tests PASS**: EOM/leaf forwarding/partitions/membership strengths,
presentation, duplicates/constant all noise, empty/singleton/1D/3D/nonfinite/
strings/invalidmetric/size/samples/selection, direct-backend sensitivity,
paired permutation(seed 23), and presentation NaN/+Inf/-Inf noise remapping.
EOM has 2 clusters/1 noise on this fixture; leaf subdivides one group (initial
assumption that leaf must also yield 2 was rejected). Assertions compare
co-membership matrices, not literal label integers; each group stays in its
separated cloud. Real 2 D fitting is not evidence of physical populations.

Observed before review: normal prediction_data=True rejects nonfinite input
on 0.8.44, while presentation without it may map those rows to -1. Revised tests
do not require this asymmetry: relevant refusal or explicitly masked invalid
rows is admissible in either mode. Presentation prediction_data is OPEN. Constant geometry yields 8 noise/0 clusters in this
fixture/version; no universal allconstant policy is inferred. Optional 2 real
failed-refit observations confirm the cache defect on singleton failure.

Membership strengths are finite in[0,1], zero fornoise, and have one value per
observation. They are density/persistence-related scores, not probabilities of
astrophysical class and need not sum to 1. External API and installed signature/
docstring were inspected separately from mocks.

## Scientific interpretation and primary sources

Teff uses kelvin; logg is log 10 surface gravity in cgs; distance is pc; Gaia
photometric colors/magnitudes are not embedding coordinate units. GSP-Phot
parameter estimates have modeling assumptions and systematic errors. This goes
beyond a naming discrepancy: [Andrae et al., section 3.5.3](https://arxiv.org/html/2206.06138v1#S3.SS5.SSS3)
find biased GSP-Phot[M/H] and describe an empirical calibration that also
translates to the LAMOST DR6 [Fe/H] scale. Applying -0.8/-0.3 to an uncalibrated
mean does not validate a Galactic population classification. No calibration
was run here; calibrated abundances and group homogeneity remain unverified.
The requested [ESA calibration page](https://www.cosmos.esa.int/web/gaia/dr3-gspphot-metallicity-calibration)
returned HTTP 451 in this environment; its content was not independently read.
The linked primary paper supplies the bias and translation evidence directly.

[Wang & Chen, section 2.2 and Table 1](https://arxiv.org/html/2108.13605v1#S2.SS2)
select APOGEE red-clump candidates with 2.3 <= logg <= 2.5 plus temperature/quality
criteria. This is a counterexample to logg<2.5 implying supergiants, not a
universal red-clump definition or an application of their cuts to Gaia.
The [Pecaut/Mamajek dwarf sequence, version 2022.04.16](https://www.pas.rochester.edu/~emamajek/EEM_dwarf_UBVIJHK_colors_Teff.txt)
lists F0V 7220 K and B9V 10700 K: the unbounded condition Teff > 7500 K includes stars
hotter than A/F. The sequence is for dwarfs; no universal MK classifier or
new production boundary is proposed. These qualifications belong to #78.

The [HDBSCAN API](https://hdbscan.readthedocs.io/en/latest/api.html) defines noise
and membership strength geometrically. A low-density point can arise from
observational quality, sampling, embedding/hyperparameters or true rarity;
the emitted anomaly label requires independent spectral checks. The
[UMAP clustering guide](https://umap-learn.readthedocs.io/en/latest/clustering.html)
and [FAQ](https://umap-learn.readthedocs.io/en/latest/faq.html) caution about
density and recommend distinguishing clustering from 2 D visualization. Cluster
mean Teff/logg/mh thresholds do not establish homogeneous MK classes, luminosity
classes, populationI/II/halo or individual evolution. Neither uncertainty,
SNR, selection nor coordinate/parameter sensitivity is propagated into those
phrases. No astrophysical truth assertion lives in green tests.

For pandas moments see [GroupBy.std](https://pandas.pydata.org/pandas-docs/version/1.5/reference/api/pandas.core.groupby.GroupBy.std.html)
and the installed signature(ddof=1). For scaling see
[StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)
(population variance). Sources clarify contracts; latest documentation is not
proof that archived real runs used these versions or preprocessing.

## Defects, archives, mutations and execution ledger

Proof source and results were pushed/read back at fixed
**[d402922d3835046d850ad13d57d0b3275b886063](https://github.com/PhD-Brown/AstroSpectro/tree/d402922d3835046d850ad13d57d0b3275b886063)**:
[executable](https://github.com/PhD-Brown/AstroSpectro/blob/d402922d3835046d850ad13d57d0b3275b886063/tests/proofs_dimred_hdbscan_analyzer.py),
[manifest](https://github.com/PhD-Brown/AstroSpectro/blob/d402922d3835046d850ad13d57d0b3275b886063/audit/M24_PROOFS.md),
[per-case results](https://github.com/PhD-Brown/AstroSpectro/blob/d402922d3835046d850ad13d57d0b3275b886063/audit/m24_proof_results.json),
[oracle controls](https://github.com/PhD-Brown/AstroSpectro/blob/d402922d3835046d850ad13d57d0b3275b886063/audit/m24_proof_control_results.json).
At that committed checkout: **10 FAIL /24 PASS /2 optional deselected**, then
separately **2 real observations PASS /34 ordinary deselected**. Zero collection/
fixture errors, skips or xfails. The documentation-only descendant adds issue
links; final archive ref/SHA is also recorded in the PR/handoff. Both archives
and initial `749224f` remain ancestors of integration, without force-push.

| Finding / classification | Follow-up | Evidence and open remedy |
|---|---|---|
| NEW alignment/length validation defect | [#73 P1](https://github.com/PhD-Brown/AstroSpectro/issues/73) | Four FAIL, plus equal-length permutation observation. Wrong synthetic row-label association, no historical incidence measured. Reject mismatches or reconcile by identities; identity transport OPEN. |
| NEW incoherent failed-fit state | [#74 P1](https://github.com/PhD-Brown/AstroSpectro/issues/74) | Four FAIL, two real observations. Retain or invalidate coherently, configuration preserved; neither remedy chosen. |
| NEW no available physical columns | [#75 P2](https://github.com/PhD-Brown/AstroSpectro/issues/75) | One FAIL on internal concat diagnostic. Counts-only table vs explicit domain refusal OPEN. |
| NEW all classes missing | [#76 P2](https://github.com/PhD-Brown/AstroSpectro/issues/76) | One FAIL on IndexError; explicit class diagnostic vs missing dominant class OPEN. |
| NEW raw output/docstring inconsistency | [#77 P2](https://github.com/PhD-Brown/AstroSpectro/issues/77) | PASS observation raw 10/20/30 vs mean z=0; raw documentation or deliberate standardized API both admissible. |
| NEW unqualified physical interpretations | [#78 P2](https://github.com/PhD-Brown/AstroSpectro/issues/78) | Text/method evidence, noise and heterogeneous 4000/8000 K observations, six threshold displays. These are not a calibrated taxonomy; metallicity terminology qualified above. |
| KNOWN packaging/provenance/train-only/joins/other audit defects | #21/#23/#24/#49–#71 | No duplicate corrective issue and no correction here. Source-level import isolation and denylist testing do not resolve these. |
| OPEN / not a demonstrated new bug | archive observations | Empty/matrix/nonfinite centroid, duplicate/empty phys_cols, missing/tied class policy, independent positional permutation without IDs, empty sweep schema, artificial zero-length backend result, unknown-column provenance. |

No additional suspicion is promoted to a confirmed defect without evidence.
False initial assumptions were retained in the local execution history:2D
Boolean indexing can return a scalar; plateid is not in the actual instrumental
denylist; hand variance for selected[0,1,4,5,6,7,8,9] is 9 (not 10); leaf need
not reproduce EOM's two clusters; presentation NaN need not fail. These were
oracle/precondition mistakes, not production bugs or assertions weakened to
hide errors. Red/green split preserves open decisions rather than fixing them.

### Initial mutations and oracle controls (historical)

[Replay](check_m24_mutations.py) uses AST-scoped function/line sites, exactly
one edit per mutation, syntax parsing, intact baseline, JUnit counts and
failure names. Only disposable copies are changed. Input content hashes use
CRLF→LF normalization, and the tracked index is checked before/after. Import,
fixture/collection errors or differing case counts are INCONCLUSIVE, not kills.
The complete successful run has no such errors; all real source hashes match.

First 56-case campaign at `749224f`: **52 KILLED /2 SURVIVED** out of 54, plus 6
separate controls PASS. The survivors reverse symmetric presentation labels
or remove a Gaia exclusion masked by other prefix exclusions. Added asymmetric
labels and mh_gspphot (not caught by those prefixes), plus default physical
column statistics. Current 59-case suite: **54 KILLED /0 SURVIVED /0 INCONCLUSIVE**,
same 6 controls PASS. The two sites were explicitly replayed against the Git
test file at `749224f` and both SURVIVED. This is a reproducible before/after
comparison, not a claim of exhaustive mutation coverage or scientific truth.

Results in [m24_mutation_results.json](m24_mutation_results.json) contain final
site/failure metadata and the historical two-site comparison. Initial full
results remain archived at `749224f`, outside the integration diff. Six controls:
equivalent np.unique IDs, neutral comment, mismatch refusal, standardized first
return, prefix qualifying cluster heuristic prose, normal-fit retention. The
prose-prefix control does **not** fix the unconditional noise description or
certify a physical label. These 6 are excluded from the 54 mutation denominator.

Separate archived oracle validation runs green 59 plus targeted red proofs:
coherent retention and full invalidation 63 PASS each, partial invalidation
59 PASS/4 expectedFAIL (control PASS), mismatch refusal 63 PASS, missing-physics
diagnostic 60 PASS, missing-class placeholder 60 PASS. All 6 control outcomes PASS,
with intact real inputs/index. They do not approve a corrective API.

### Initial local commands actually executed (historical)

Actual Python executable throughout:
`C:/Users/alexb/Documents/Projets/AstroSpectro/.venv/Scripts/python.exe`.
Windows `10.0.26300` AMD64, Python 3.11.9, pytest 8.4.2/pytest-socket 0.7.0,
numpy 2.4.6, pandas 3.1.1, scipy 1.17.1, sklearn 1.9.1, astropy 8.0.2,
joblib 1.6.0, Matplotlib 3.11.2, Black 24.4.2, Ruff 0.5.0. Optional target 0.8.44.
Commands below use that Python, from worktree root; outputs under ignored
`test-results/` are local, not scientific artifacts or a second task tracker.

| Command | Observed result |
|---|---|
| `-m pytest -q --junitxml=test-results/m24-baseline.xml` before additions |323 PASS/8 deselected/7 existing Gaia warnings |
| `-m pytest -q tests/test_dimred_hdbscan_analyzer.py -W error --junitxml=test-results/m24-green-enriched.xml` |59 PASS; no warning/skip/xfail |
| `-m pytest -q --junitxml=test-results/m24-fast-enriched.xml` |382 PASS/29 deselected/7 existing Gaia warnings |
| `-m pytest -q tests/test_dimred_hdbscan_analyzer.py -m 'integration and not slow' -W error --junitxml=test-results/m24-integration.xml` |4 PASS/55 deselected |
| Scoped optional PYTHONPATH, `-m pytest -q tests/test_dimred_hdbscan_real.py -m slow -W error --junitxml=test-results/m24-real-final.xml` |21 PASS |
| `audit/check_m24_mutations.py --reverse --output test-results/m24-reverse.json` |59 PASS, reversed collection/hashseed 7/-W error |
| `audit/check_m24_mutations.py --tests-ref 749224fc5f3237290e005c6ff5a98423e06adf81 --only presentation_reverses_labels feature_no_gaia_exclusions --output test-results/m24-mutations-before.json` |56-case baseline PASS, both mutations SURVIVED |
| `audit/check_m24_mutations.py --output test-results/m24-mutations-final.json` |59-case baseline PASS,54 mutations KILLED,6 controls PASS |
| Archived `audit/check_m24_proof_controls.py --output test-results/m24-proof-controls-final.json` |6 oracle outcomes PASS, partial invalidation deliberately 4 FAIL; all green unaffected |
| At committed `d402922`: `-m pytest -q tests/proofs_dimred_hdbscan_analyzer.py --tb=no --junitxml=test-results/m24-proofs-at-d402922.xml` |10 FAIL/24 PASS/2 deselected/0 ERROR/SKIP/XFAIL |
| Same committed archive, scoped optional target, proof file `-m slow -W error --junitxml=test-results/m24-real-proofs-at-d402922.xml` |2 PASS/34 deselected |
| `-m black --check src tests scripts app.py audit/check_m24_mutations.py audit/check_m24_proof_controls.py` |37 files unchanged, archive script included |
| Same Ruff scope; `-m pip check`; `git diff --check` |PASS, no broken requirements, clean diff |

The 7 warnings are unchanged Gaia/pandas inplace-deprecation warnings at
`src/tools/gaia_crossmatcher.py:1210` (#16), outside this audit. Strict-Werror
passes on M2.4. Hosted CI uses Python 3.11/Ubuntu and excludes slow/network;
it does not install/run optional HDBSCAN. Final SHA-specific CI is checked and
linked in the Draft PR/handoff after the final integration commit.

### Integration diff and remaining gaps

Only G/R test files, M2.4 helper fixtures, two-line plugin registration in
conftest, this matrix, replay and final mutation JSON remain for integration.
Proof source/manifests/results/oracle-control script and initial mutation JSON
remain in reachable archived ancestors, removed from the green tree. No
production/config/dependency/CI/notebook/data changes. Archive lint/replay
are checked explicitly; CI lint paths do not include audit scripts.

Not validated: real star populations, calibrated evolutionary/chemical classes,
full notebook/UMAP-to-cluster pipeline, large/sparse/precomputed/custom metrics,
version-independent tie/degenerate partitions, backend/GPU/platform performance,
I/O interruption/permission/encoding exhaustiveness, covariance/measurement
errors and selection corrections, line coverage. No hidden B status: optional
backend was available; remaining contracts are PA/NA/open, not silently blocked.

Review questions for Claude: test oracles vs implementation, identity/precondition
distinction, transaction alternatives, missing physics/classes outcomes,
population-vs-sample variance, dependency-version nonfinite behavior, green
compatibility with legitimate remedies, mutation classification and whether
the scientific documentation sufficiently qualifies its claims. No M2.4
completion/integration claim before that review and human merge decision.

## Initial archive and integration boundary (historical)

The durable proof branch advanced by fast-forward to
**[a27028bb9dfb617d5cddc2a7fac6eb08a195142a](https://github.com/PhD-Brown/AstroSpectro/tree/a27028bb9dfb617d5cddc2a7fac6eb08a195142a)**.
Its [manifest with issue correspondence](https://github.com/PhD-Brown/AstroSpectro/blob/a27028bb9dfb617d5cddc2a7fac6eb08a195142a/audit/M24_PROOFS.md),
proof source, fixtures, both result JSONs and oracle-control script were read
back from GitHub **before** removal from integration. Source/tests/results are
identical to d402922, whose committed executions are recorded above. The
source/proof blob IDs were verified, and the remote ref matched the full SHA.
No force-push/history rewrite. The proof ref is an ancestor of the final head.
All six issues link the fixed executed d402922 proofs; a27028b adds their links
to the manifest. Both are durable, not ephemeral references to a moving branch.

The final diff versus starting main contains exactly seven paths: two green/
optional test files, helper, conftest registration, matrix, mutation script and
mutation JSON. The source blob remains `667a2985ea1b6b4f0a660a70f7635715730c0f78`.
Final commit SHA and hosted CI are recorded in the PR/handoff so this document
does not need a self-referential commit ID. M2.4 awaits independent review.

## Revision execution and provenance ledger

Revised code/test input snapshot:
`1b9fa1157146eb508e9b590704ef0a7b214b87e6`. Commands use the same Windows
Python3.11.9 environment and isolated hdbscan0.8.44 target described above.
Source remains blob667a2985ea1b6b4f0a660a70f7635715730c0f78. Replays copy
only synthetic source/test trees and never mutate production in this worktree.
No dependencies, CI, notebooks, data or models were edited.

| Actual command (project Python prefix) | Revised execution |
|---|---|
| `-m pytest -q tests/test_dimred_hdbscan_analyzer.py -W error --junitxml=test-results/m24-review-green-final.xml` |62PASS, no warnings/skips/xfails |
| `-m pytest -q --import-mode=importlib --junitxml=test-results/m24-review-importlib-at-1b9fa11.xml` |385PASS/29deselected/7existing Gaia warnings |
| `-m pytest -q --junitxml=test-results/m24-review-fast-final.xml` |385PASS/29deselected/7existing Gaia warnings |
| `-m pytest -q -m 'integration and not slow' --junitxml=test-results/m24-review-integration-final.xml` |54PASS/360deselected, whole-repository integration selection |
| Scoped target, `audit/check_m24_mutations.py --real --only presentation_prediction_data uniform_finite_refusal --output test-results/m24-review-real-controls-final.json` |Intact21PASS; both legitimate controls21PASS each, no skip/xfail |
| `-m pytest -q tests/proofs_dimred_hdbscan_analyzer.py --tb=no --junitxml=test-results/m24-review-proofs-at-1b9fa11.xml` |10FAIL/33PASS/2optional deselected/0ERROR/SKIP/XFAIL |
| Same proof file with scoped target, `-m slow -W error --junitxml=test-results/m24-review-real-proofs-at-1b9fa11.xml` |2PASS/43deselected |
| `audit/check_m24_proof_controls.py --output test-results/m24-review-proof-controls-final.json` |7oracle checks PASS: retention/invalidation/noise-only palette66PASS each; partial invalidation62PASS/4expectedFAIL; refusal66PASS; diagnostics63PASS each |
| `-m black --check src tests scripts app.py audit/check_m24_mutations.py audit/m24_review_candidates.py audit/check_m24_proof_controls.py`; same Ruff scope |PASS,37Black files with archive present |
| `-m pip check`; `git diff --check` |PASS; no broken requirements |

The complete revised mutation/ordering comparison is recorded in the final
JSON. It distinguishes the
original54 sites, six semantic review reconstructions and two extra guards
(undefined dispersion→zero; mixed raw/standardized returned columns). They
are distinct scoped edits, not the unavailable exact19-case Claude script.
`palette_swapped_ranks`, one of the original54, is explicitly reclassified as
admissible underR5. Its survival is expected; aesthetic controls are not
incorrect mutations. The six original controls remain identifiable by name.

Replay at committed input snapshot1b9fa11:
`audit/check_m24_mutations.py --output test-results/m24-review-mutations-final.json`
gives **53legacy incorrect sites KILLED +1admissible palette PASS**, six semantic
R1/R2 sites KILLED, two extra guard sites KILLED, six original controls PASS
and eleven review controls PASS. Thus61distinct incorrect sites are detected;
the18other outcomes (including the palette alternative) are excluded. Zero
SURVIVED among incorrect candidates, zero INCONCLUSIVE/import/collection errors.
Every incorrect site has named failing assertion(s) in the JSON. No claim to
the exact independent19-case script is made.

The six semantic reconstructions were replayed with only the green test file
from e11ad1b: intact59PASS and all6SURVIVED; current intact62PASS detects all6.
`--reverse --output test-results/m24-review-reverse.json` gives62PASS under
reversed collection, hashseed7 and-Werror. All source/input hashes and tracked
index guards remain unchanged. No production file in the real tree was edited.

### R10: historical run is not rewritten

The original schema1 JSON, including `base_sha=749224f`, remains immutable at
[e11ad1b](https://github.com/PhD-Brown/AstroSpectro/blob/e11ad1bbfca82a399b7e45049c5e6cfde1e6d25f/audit/m24_mutation_results.json).
That SHA field was the active Git HEAD during an execution with uncommitted
test/replay edits, **not** the commit containing all tested bytes. All eight
recorded input hashes were verified against d402922. The exact full historical
working-tree status was not captured by schema1; input-byte differences from
749224f are reconstructible and listed in the revised JSON. The later commit
containing these inputs is distinguished from the run's Git HEAD.

New schema2 executions record the actual active HEAD, working-tree status,
normalized input hashes, working-content Git blob IDs, selected group and
per-case JUnit failures. Current tested input files are committed at1b9fa11;
untracked/restored documents and result files at run start do not alter them.
The historical-test comparison replaces **only** the green test file in a
sandbox with its e11ad1b bytes; current conftest remains, and its exact hash is
recorded. It is not mislabeled as a full old-checkout execution.

Final archive SHA, final integration HEAD and SHA-specific hosted CI are
published in the updated Draft PR and #43 handoff. No review thread is
resolved by Codex. M2.4 remains pending Claude's second review and human merge.
