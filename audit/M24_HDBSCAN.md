# M2.4 — HDBSCANAnalyzer contracts and scientific limits

2026-11-08 · [Refs #43](https://github.com/PhD-Brown/AstroSpectro/issues/43).
Tests/audit only; submitted for independent Claude review and human decision.
No M2.5 work. A green suite is not astrophysical validation.

## Reference and scope

Audited `origin/main@58180392d16e06ad411073bc48011d076f9128f0`, fetched at
start, matching merge PR#72. Complete562-line `src/dimred/hdbscan_analyzer.py`
read, Git blob **667a2985ea1b6b4f0a660a70f7635715730c0f78**. Work branch
`audit/43-dimred-hdbscan` starts at that main in separate worktree
`C:/Users/alexb/Documents/Projets/AstroSpectro-m24`. Original workspace remains
on clean `audit/43-dimred-embedding@59ba28d6a5d664e36b37b195fefe7484d5b85a8c`.
No stash/reset/cleanup, no main push, production change, notebook execution,
scientific data/model changes, dependency/lock/CI edits or merge.

Read live AGENTS.md v1.3, complete#43/dashboard and relevant comments through
GPT's M2.4 preparation, merged PR#56/#64/#72 and matrices M21/M22/M23. The
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
| Constants/import boundary1–45 | Six Gaia columns; numpy/pandas/Matplotlib; backend lazy. No catalog I/O on source import. | G initial/optional-import tests PASS U; real source blob checked. AP lazy HDBSCAN error, PA facade. | #21 packaging; names/units and spectral provenance not certified by denylist. |
| `__init__`64–90 | Stores75/20/euclidean/eom or configured values; normal and presentation estimators/labels initiallyNone, zero counters, empty instance-owned lists/maps. | G `initial_state_and_instance_isolation`, `fit_forwarding_and_success_state`: PASS U. AP named defaults/state. | Constructor itself does not validate types/ranges; backend checks on fit. |
| `fit`94–132 | Z(N,2) positional numeric coordinates; HDBSCAN kwargs match configured size/samples/metric/selection, prediction_data=True; returnself, store estimator/labels then recompute stats/palette. No coordinate physical unit inferred. | G forwarding, preservation, successive fit, optional dependency/failure propagation PASS U. R actual EOM/leaf, degenerate/invalid/probability cases PASS I/P. AP successful wrapper contracts; PA failure-state coherence. | A four mock failures plus2 real observations show replaced estimator with stale cache; transaction choice OPEN. Valid shapes beyond documented2D, sparse/precomputed/custom metrics not exhaustive. |
| `fit_presentation`134–172 | Independent model and labels; size300 default or override, same samples/metric/selection; sorted IDs without-1, tab20 map; returns presentation labels. No prediction_data=True promise. | G `presentation_is_separate` PASS U; normal→presentation→normal→presentation replacement, exact ID/color mapping and unchanged opposite mode. R real presentation PASS I; nonfinite remapping compared directly. AP successful state, PA failure paths. | Empty/tiny real inputs fail; no empty-success contract inferred from mocks. No minimum granularity/count monotonicity guaranteed by changing size. |
| `_compute_stats`364–368 | Unique sorted labels except-1; countIDs; exact number of-1, not number of unassigned classes. | G `counts_and_palette_ids` allnoise/no-noise/noncontiguous/multi/single plus `helpers_recompute_and_successive_fit_replace_state` PASS U/P. AP. | Labels assumed backend integer vector. Malformed synthetic backend values not a supported scientific input contract. |
| `_build_palette`370–378 | turbo sampled in sorted-ID order; dedicated grey alpha0.25 for-1; presentation tab20 separately. | G counts/presentation check exact rank→color using external Matplotlib API, membership/dimensions/distinctness and order permutation PASS U. AP current mapping. | Exact palette colors are visualization conventions, not significance/class semantics; more than20 presentation groups can repeat a tab20 color. |
| `centroid`359–362 | Arithmetic barycenter of selected points; no medoid/density/physical-center claim; inputs unchanged. | G3 analytic subsets including singleton and elongated cloud; translation, wrong-length mask PASS U/P. AP valid 1D-mask arithmetic. A empty/2D-mask/Inf observations PASS. | PA empty/nonfinite/matrix-mask policy OPEN. Float coordinates describe representation space only. |
| `physical_profile_table`176–338: preconditions/alignment | Requires labels, equal lengths; copied/reset-index meta then positional labels/y. Old pandas index is ignored. | G before-fit and length errors, identity markers with nonconsecutive indices, paired row permutation, arbitrary ID renaming PASS U/I/S. A independent meta/y/label permutations and duplicate IDs expose limitations. AP positional behavior, PA identity validation. | Same length cannot certify correspondence. API has no fit IDs, so independent permutation violates documented alignment precondition; no new guaranteed detection imposed. |
| physical statistics220–276 | Group means and pandas sample std(ddof=1), exact counts and unique dominant class; NaN excluded from numeric moments; singleton std remainsundefined, noise row has mean only. | G `physical_profiles_analytic_statistics_and_identity` float+nullable; custom/noisy/allnoise/singletons/missing columns/partial missing class, local-change/noise-isolation PASS U/P. AP named finite/missing numerical contracts. | All available physical columns absent triggers internal pandas ValueError; all class labels missing triggers IndexError. Tie/missing/Inf/non-numeric/duplicate-column handling remains PA with archive observations. No silent assertion of std=0 for singleton. |
| physical formatting/interpretation239–338 | Teff0 decimals, others2; noise mean, own count; interp same cluster index/count. Heuristics depend on meanTeff/logg/mh. | G manual expected strings/counts PASS U/P. A6 exact-threshold display observations PASS. AP arithmetic/display fidelity; **NA astrophysical classification**. | Cluster mean can mix heterogeneous stars; gravity thresholds alone do not establish evolutionary stage. Noise→spectral anomaly is unproven. Gaia metallicity terminology requires qualification. No green assertion mandates heuristic prose. |
| `print_physical_profile`340–355 | Prints supplied table and interpretation rows; returnsNone; no DataFrame modification. | G `profile_print_transmits_tables_without_mutating`, `print_empty_supplied_tables` PASS U. AP structure/data. | Scientific force of title/prose remains unvalidated; whitespace/borders not treated as contracts. |
| `sensitivity_df`380–382 | Returns an empty DataFrame; module function has no analyzer reference and does not populate it. | G default property before/after sweep and after result mutation PASS U. AP existing independent state. | Placeholder/lacuna, not a persisted experiment table. Empty schema/ownership policy OPEN; no invented relationship. |
| `compute_sensitivity`388–429 | Default50,100,150,300,500 or caller order; each estimator gets samples20(default), euclidean/eom; counts exclude-1, percent round(100*Nnoise/N,2). | G full/default/custom/duplicate/empty grids, allnoise/no-noise/1of6/2of6 fractions, trial inputs and propagation PASS U/P; R direct-backend sweep PASS I. AP positive-N contracts. | Empty grid returns no-column frame, schema OPEN. Invalid entries delegated to backend; no monotonicity/determinism beyond selected examples. Artificial N=0 backend response divides by0; real backend refuses N=0. |
| `compute_feature_profiles`432–530: CSV/selection | Path stem.csv; lazy real StandardScaler/loader constants, fallback local import, FileNotFoundError. Active SNR>=10 includesboundary/excludesmissing; numeric variable columns, missing<=10%, metadata/prefix exclusions; dropna on retainedfeatures. | G curated exact columns vs spectro_only loader, thresholds0/10/20%, SNR boundary, noSNR, empty/constant/Inf populations, missing file/fallback PASS U/I. AP observed mechanisms; PA spectral provenance. | Matches spectro_only=True on controlled population, not default loader (instrumental policy differs). Loader permits filters/classes/balancing/sampling/custom merge key; this function cannot inherit those. #23 denylist is not a provenance certificate; unknown plateid remains admitted. |
| feature alignment532–545 | Current code compares feature count with Z, truncates positionally on mismatch; no IDs accepted. | A SNR/dropna remove middle104:2 FAIL, independent label lengths8/12 vs10:2 FAIL; equal-size permutation observation PASS S. **PA defect proven**, no AP provenance. | Wrong association demonstrated on synthetic mismatched populations; calling this “conservative intersection”/“perfect alignment” is unjustified. Reject mismatch or propagate IDs/selection record are alternatives. No historical rate measured. |
| feature scaling547–561 | Default centering and population std(ddof=0) fit on all retainedrows **including noise**; grouped z-score means exclude noise only at aggregation. Output feature order preserved. | G2-feature analytic profiles, scaler boundary, affine units/paired permutation, feature order reversal, singlefeature/allnoise/postdropna constant PASS I/P. AP descriptive numeric contracts. | Full-population reference is descriptive, not a train/test evaluation. Train-only required if reused predictively; none is run here. df_feat returnedraw despite standardized docstring; output API choice OPEN, archive raw observation PASS. |
| Interfaces | Loader CSV→scaledX→realPCA→controlled clustering→physical/feature profiles; actual HDBSCAN separate. Visualizer/notebook accept normal/presentation arrays and sensitivity tables. | G `loader_pca_cluster_profile_observation_identity` PASS I with shuffled catalog join and1000/1200 vs6000/6200K oracle. AP tested boundary. | No real UMAP→HDBSCAN→astrophysical validation, large data, notebook execution or visualizer rendering claim. M2.8 owns visualization audit. |

## Independent numerical oracles

Physical means/std are hand-derived. C2 temperatures1000/1400 give1200 and
sample std=sqrt(80000)=282.8427…→283; C7 6000/6200/6400 gives6200±200.
Gravity values1/3 give2±sqrt2;4/5/NaN gives4.5±sqrt(0.5). Dominant classes
have unambiguous majorities, so no tie-break policy is frozen. Integer counts
and string formatting are exact. Singleton/allmissing dispersions are left
undefined rather than replaced by zero. NullableFloat64/float64 compare alike
for the finite/missing cases tested; nullable-allmissing display is archived.

Feature example a=[0,2,6,8], b=[1,5,1,9], labels=[7,-1,2,2]. Population means
4/4 and variances10/11 include noise. Expected cluster2 scores3/sqrt10 and
1/sqrt11; cluster7 -4/sqrt10 and-3/sqrt11. Positive affine changes of units and
paired row permutations preserve standardized profiles. After dropping the
first of10 rows with oneNaN, values1…9 give mean5,var20/3; this distinguishes
correct dropping from head truncation. A feature varying before dropna but
constant afterward must scale tozero using StandardScaler's constant handling.
Absolute tolerance3e-15 for O(1) standardized means,2e-14 for translated
centroids allows float64 rounding; relative tolerance0. No observational
uncertainty or significance threshold is inferred. Text rounding is not a
high-precision statistical oracle by itself; exact feature means supplement it.

Sensitivity independently uses1/6→16.67%,2/6→33.33%,0→0%,6/6→100%; duplicate
grid entries must retain trial order/multiplicity for current sequence output.
No monotonic cluster-count assertion and no sum-to-one probability assertion.

## Real HDBSCAN versus controlled backend

Ordinary venv has no hdbscan. Installed **hdbscan0.8.44** with
`uv pip install --python ../AstroSpectro/.venv/Scripts/python.exe --target test-results/m24-optional-deps --no-deps hdbscan`.
Only scoped PYTHONPATH selects it. Project venv/dependencies/CI unchanged.
Windows10 AMD64/Python3.11.9; numpy2.4.6/scipy1.17.1/sklearn1.9.1.
Two12-point Gaussian groups centered(-10,0)/(10,0), std0.1, RNG17, plus(0,100),
with size4/samples2. Tiny dense fits use actual backend defaults, including
core_dist_n_jobs=4/approx_min_span_tree=True, no GPU. No global random seed
or cross-platform partition guarantee is asserted.

**21 optional tests PASS**: EOM/leaf forwarding/partitions/membership strengths,
presentation, duplicates/constant allnoise, empty/singleton/1D/3D/nonfinite/
strings/invalidmetric/size/samples/selection, direct-backend sensitivity,
paired permutation(seed23), and presentation NaN/+Inf/-Inf noise remapping.
EOM has2 clusters/1 noise on this fixture; leaf subdivides one group (initial
assumption that leaf must also yield2 was rejected). Assertions compare
co-membership matrices, not literal label integers; each group stays in its
separated cloud. Real2D fitting is not evidence of physical populations.

Normal prediction_data=True rejects nonfinite input while generating prediction
structures on this version; without it, presentation may map bad rows to-1.
This difference is tested, not presented as a wrapper validation guarantee for
other backend versions. Constant geometry yields8noise/0clusters in this
fixture/version; no universal allconstant policy is inferred. Optional2 real
failed-refit observations confirm the cache defect on singleton failure.

Membership strengths are finite in[0,1], zero fornoise, and have one value per
observation. They are density/persistence-related scores, not probabilities of
astrophysical class and need not sum to1. External API and installed signature/
docstring were inspected separately from mocks.

## Scientific interpretation and primary sources

Teff useskelvin; logg is log10 surface gravity in cgs; distance ispc; Gaia
photometric colors/magnitudes are not embedding coordinate units. GSP-Phot
parameter estimates have modeling assumptions, selection effects and systematic
uncertainties. The [ESA methods](https://gea.esac.esa.int/archive/documentation/GDR3/Data_analysis/chap_cu8par/sec_cu8par_apsis/ssec_cu8par_apsis_gspphot.html)
describe global[M/H]; the [source datamodel](https://gea.esac.esa.int/archive/documentation/GDR3/Gaia_archive/chap_datamodel/sec_dm_main_source_catalogue/ssec_dm_gaia_source.html)
calls mh_gspphot ironabundance. This documentary discrepancy is disclosed:
neither universal[M/H]=[Fe/H] nor a silent relabeling is established here.
No calibrated metallicity/abundance analysis was performed.

The [HDBSCAN API](https://hdbscan.readthedocs.io/en/latest/api.html) defines noise
and membership strength geometrically. A low-density point can arise from
observational quality, sampling, embedding/hyperparameters or true rarity;
the emitted anomaly label requires independent spectral checks. The
[UMAP clustering guide](https://umap-learn.readthedocs.io/en/latest/clustering.html)
and [FAQ](https://umap-learn.readthedocs.io/en/latest/faq.html) caution about
density and recommend distinguishing clustering from2D visualization. Cluster
meanTeff/logg/mh thresholds do not establish homogeneous MK classes, luminosity
classes, populationI/II/halo or individual evolution. Neither uncertainty,
SNR, selection nor coordinate/parameter sensitivity is propagated into those
phrases. No astrophysical truth assertion lives in green tests.

For pandas moments see [GroupBy.std](https://pandas.pydata.org/pandas-docs/version/1.5/reference/api/pandas.core.groupby.GroupBy.std.html)
and the installed signature(ddof=1). For scaling see
[StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)
(population variance). Sources clarify contracts; latest documentation is not
proof that archived real runs used these versions or preprocessing.

## Defects, archives, mutations and execution ledger

Fixed archive SHA, corrective issue links, final campaign classifications and
final local/hosted execution results are recorded below after verification.
Known#21/#23/#24/#49–#71 remain separate; none is corrected by this PR.

Review questions for Claude: test oracles vs implementation, identity/precondition
distinction, transaction alternatives, missing physics/classes outcomes,
population-vs-sample variance, dependency-version nonfinite behavior, green
compatibility with legitimate remedies, mutation classification and whether
the scientific documentation sufficiently qualifies its claims. No M2.4
completion/integration claim before that review and human merge decision.
