# M2.7 — RunReporter scientific and artifact audit

2026-11-09 · Refs #43 · Delivered for first independent Claude review.
Tests/evidence/documentation only; production remains unchanged.

## Traceability and initial risk matrix

Base/source: `1e2266385eab287d3a1c54c2e919e012293fa44b`.
Module: `src/dimred/run_reporter.py`, blob
`b966d9b90f056baa37c0ccbb68135c71ff65de92`, 1558 lines, 19 functions.
Dedicated branch `audit/43-dimred-run-reporter`, separate `AstroSpectro-m27`
worktree. Main checkout clean at `795af2c8`; never checked out or updated.
Live main and M2.6 merge PR91 verified; no preexisting M2.7 branch/PR.
AGENTS v1.3, full #43 body and comment history retrieved (latest M2.6 handoffs
read), previous M22/M23/M25/M26 matrices and notebook save/load cells examined.

Interpreter for all documented local commands: existing
`../AstroSpectro/.venv/Scripts/python.exe` (Windows, Python3.11.9).
NumPy2.4.6, pandas3.1.1, scipy1.17.1, sklearn1.9.1, joblib1.6.0,
pytest8.4.2, pytest-socket0.7.0, Black24.4.2, Ruff0.5.0; pip check PASS.
The unqualified system Python has different packages and is not used to run tests.

Only preexisting reporter test: `test_embeddings_reporter_round_trip`
in `tests/test_dimred_embedding.py`: actual EmbeddingEngine/PCA wrapper with
controlled UMAP backend, real trusted synthetic joblib. It does not execute
the real UMAP algorithm or protect NB01/NB03.
Tests load actual reporter source by file, isolating optional facade #21.
Reporter imports NumPy/pandas; joblib imported lazily in persistence helper.

| Function | Inputs / calculation / output | Effects, consumers, initial gaps / risks |
|---|---|---|
| `class_counts(labels)` | np.unique counts, string keys | NB01/02/03; mixed/missing types policy unspecified |
| `safe_df_records(df,max_rows=None)` | copy, head, where, records | All table reports; promised NaN->None needs dtype-dependent strict JSON proof |
| `json_default(obj)` | NumPy scalar/array, Path, datetime conversions | JSON writer; unsupported types refused, nonfinite values not normalized |
| `embedding_stats(Z)` | first two axes, nan-aware min/max/mean/population std | NB02/03; empty/all-NaN/Inf need characterization |
| `embedding_stats_nd(Z,max_axes=3)` | bounded axis summaries | NB02 3D; dimensional validity and bounds |
| `shape_or_none(x)` | asarray shape; exception->None | NB02; shape is not identity or validation |
| `sensitivity_shapes(mapping)` | string keys->shapes | NB02; metadata only, not reproducibility |
| `numeric_means(df)` | numeric column means | NB02 stability; missing/nonfinite policy |
| `fmt_seconds(value)` | finite float, .2f or N/A | TXT NB02/03; format does not validate negative durations |
| `vector_stats(arr)` | flatten finite subset, n, mean/std, linear percentiles | Confidence; finite n differs from input population |
| `cluster_top_counts(labels,top_n=10)` | exclude -1, counts, descending sizes | NB02; integer labels; ties no scientific ordering |
| `top3_pred_by_cluster(c,p)` | positional mask and top3 labels, includes noise | NB02; lengths and identities must agree |
| `make_timestamp()` | UTC second precision string | All runs; no uniqueness guarantee |
| `make_run_dir(reports_dir,sub)` | reports/runs/sub Path | mkdir parents/exist_ok, shared directory, not per timestamp |
| `save_joblib_pair(obj,run_dir,stem_run,stem_latest,legacy_path=None)` | run/latest Paths | 2/3 sequential dumps, potential overwrite/partial state |
| `save_json_txt_pair(report,run_dir,stem_run,stem_latest,text_lines)` | four Paths | JSON run/latest then TXT run/latest, no transaction |
| `save_pca_run(*,...)` | score subsets and variance tables | NB01 -> NB02 legacy joblib; threshold bounding, row/feature order |
| `save_umap_tsne_run(*,...)` | optional embeddings/cluster/XGB/stability/zoom | NB02 -> NB03 legacy/latest; positional accuracy, figure inventory, missing provenance |
| `save_autoencoder_run(*,...)` | explicit/fallback latent/history, AE/PCA MSE, Spearman, interpolation | NB03; swallowed metric errors, finite pairs, paths, provenance |

Initial priorities: strict JSON; same-second overwrite; mixed latest after
failure; figures from shared feature-stem directories; incorrect model-path
attribution; positional population assumptions; nonfinite/masked metrics.
These are hypotheses until executed below. No missing manifest field is
invented retroactively. No new dependency, production or notebook change.

## Initial verification

First run: 55 PASS / 2 FAIL. The two failures exposed inferred pandas3 string
missing values surviving as NaN, independently of the anticipated float64 case.
Both dtypes are retained in explicit red proofs, not xfail. Green table contract
uses object text (supported) plus nullable numeric/bool columns; no production fix.
AP/PA/NA/B results and artifact/proof/mutation matrices follow below.

## Executed contract matrix

AP = acquired evidence for the stated finite/synthetic contract or demonstrated
defect, never general module/scientific approval. PA = partial, NA = no validation,
B = blocked (none needed for this bounded synthetic audit). U = unit, I = interface,
P = numerical, S = scientific protocol. Test names below omit the `test_` prefix
and live in `tests/test_dimred_run_reporter.py`, unless marked **proof** or **probe**.
All19 current signatures, line numbers and defaults, including the complete
keyword-only notebook signatures, are recorded verbatim by AST in
[boundary results](m27_boundary_results.json). The19-row inventory above plus
this behavior matrix accounts for the whole module; no line coverage measured.

| Contract / actual calculation | Independent oracle / test | Level | Result / status | Limit / issue |
|---|---|---|---|---|
| Class frequencies including absent classes, empty, bool/integer | Manual 3M/2F/1G; `class_counts_manual` | U/P | PASS/AP | No all-label taxonomy implied |
| Mixed and noncomparable/missing labels | **probe** mixed coerced to str counts; object/None TypeError | U | observed/PA | Supported domain not generalized |
| DataFrame copy, numeric nullable/bool/text fields, row selection0/2/all, nonconsecutive index | Explicit records + strict json.dumps; `dataframe_records_finite_and_nullable`, `none_and_empty` | U/I | PASS/AP | Text forced object; native pandas3 str failure not hidden |
| Missing float64/native str -> None promised by docstring | **proof** 2 cases, identity/strict JSON, no modification | U/I | FAIL/AP defect | #92; Float64/Int64/string/boolean nullable pass |
| max_rows negative/invalid; arbitrary columns/index | **probe** -1 retains all but last; invalid float/str TypeError | U | observed/PA | pandas head semantics, not scientific policy |
| NumPy int/float/array, Path, naive/aware datetime, Timestamp conversion | Explicit nested JSON values; `json_default_exact` | U/I | PASS/AP | ISO retains supplied offset; no implicit UTC conversion |
| Unsupported object/set/complex/np.bool_ | `json_default_unsupported_types_refused` | U | PASS/AP | Explicit TypeError, no silent stringification |
| Nonfinite scalar/nested array JSON output | **proof** strict parser rejects4 outputs | U/I | FAIL/AP defect | #92; refuse or explicit policy, no forced replacement |
| Embedding first2/all3 axes, shape and max_axes truncation | Z rows(1,4,9),(2,7,3),(8,1,6); population std sqrt86/3,sqrt6,sqrt6 | U/P | PASS/AP | `embedding_axes_independent_population_oracle`; not geometry validity |
| None,1D,1column,0columns; NaNs ignored axiswise | `embedding_shape_only_cases`, `missing_values_use_axis_population` | U/P | PASS/AP | Shape-only return is not validation; max_axes0 |
| Empty/nonnumeric embeddings | `embedding_nonnumeric_and_empty_refused` | U | PASS/AP | Explicit failure; no empty statistic invented |
| All-NaN/Inf, invalid max_axes | **probes**, warnings and actual values retained | U/P | observed/PA | NaN/Inf not interpretable measurements; #92 output |
| Shape/sensitivity summaries | Exact dimension maps; `shapes_and_sensitivity`; ragged **probe** | U | PASS/AP | Lossy metadata; shape helper failure->None |
| Numeric-only means, missing values, input immutability | Manual means6/6; `numeric_means_columns_and_immutability` | U/P | PASS/AP | All missing/Inf propagate as nonfinite in **probe**, PA |
| Duration conversion and finite formatting | `duration_format_is_not_physical_validation` | U | PASS/AP | Negative durations formatted; validity not promised |
| Finite vector population, flattening, population n/std and linear percentiles | Sorted1,2,4,20: n4,mean27/4,std sqrt955/4,p05=1.15,p50=3,p95=17.6 | U/P | PASS/AP | `vector_statistics_manual_linear_percentiles`; counts finite subset |
| Constant/singleton/empty/all-invalid vectors | `vector_constant_and_singleton`, `unavailable_population` | U/P | PASS/AP | No confidence calibration implied |
| Integer cluster counts exclude noise, topN, empty/noise, ties | Manual 7:4,2:3; exact class counts; tie-permissive tests | U/P | PASS/AP | `cluster_counts_and_classes_exact`, `cluster_top3_ties_allow_any_equal_frequency_order` |
| Cluster/prediction length mismatch | `cluster_cardinality_refused` | U | PASS/AP | Equal length still does not demonstrate identity |
| Unusual cluster labels / negative topN / missing predictions | **probes** float labels collapse through int(), strings fail, missing preds omitted | U | observed/PA | HDBSCAN actually emits integers; unsupported domain not declared new defect |
| UTC clock conversion, seconds granularity | Controlled local-5h time gives next-day045807Z; `timestamp_utc_controlled_clock` | U | PASS/AP | Same second repeats; uniqueness not inferred |
| Relative/absolute run directory, missing parents, reuse, preservation, permission | `run_directory_preserves_content`, `permission_error_propagates`; file collision **probe** | U/I | PASS/AP | Permission simulated; no physical disk exhaustion/ACL experiment |
| Trusted joblib run/latest/legacy roundtrip and distinct-stem isolation | Arrays plus DataFrame exact values/index; `joblib_three_destinations_and_run_isolation`, `without_legacy_and_missing_parent` | I | PASS/AP | Same versions/process only; no integrity/authenticity claim |
| Four UTF8 JSON/TXT files, returned paths, latest update and old run retained | `json_txt_four_destinations_and_unicode`, strict parser, semantic text lines | I | PASS/AP | Equivalent extra TXT spacing allowed |
| Helper failures and retry | Inject failures at dumps1/2/3 and file opens1/2/3/4; retry with distinct ID | I | PASS/AP | `joblib_failure_propagates_and_retry_recovers`, `json_txt_failure_propagates_and_retry_recovers`; not transaction safety |
| Same timestamp/stem cannot replace archived experiment | **proofs** helper2 + notebook3 | I | FAIL/AP defect | #93, reject collision or unique path acceptable |
| Latest group after interrupted NB02 publication | **proofs** JSONlatest/TXTlatest failures | I | FAIL/AP defect | #94; exception propagates but aliases mix A/B |
| NB01 score rows/labels/meta/features, thresholds80/90/95/99 and bounds | Deterministic cumulative .8/.9/.96/1 plus 2-column spectra; `pca_notebook_roundtrip_thresholds_and_rows` | I/P | PASS/AP | Fake PCA boundary actually computes thresholds; oracle specified independently |
| NB01 fitted real PCA | Six paired coordinates eigen-ratio9:4:1, squared scores; `pca_reporter_real_fitted_sign_invariant` | I/P | PASS/AP | Full local PCA fit; no sign assumption or astrophysical validation |
| NB02 UMAP-only/TSNE-only/both/3D-only/none and optional sections | `embedding_optional_modes_roundtrip`, `no_input_refused_before_writes`, `without_labels_metadata_or_scores` | I | PASS/AP | Synthetic arrays, not refitting UMAP/TSNE; preexisting M23 test retained |
| NB02 clusters/presentation params/noise/empty/all noise | Exact integer populations; `embedding_all_sections_exact`, `degenerate_cluster_summaries` | I/P | PASS/AP | Wrapper reporting, not HDBSCAN validation |
| NB02 normalization, inclusive zoom boundaries and nonfinite exclusions/refusal | Manually selected rows0/1/2; `embedding_all_sections_exact`, `zoom_nonfinite_is_excluded_or_explicitly_refused` | I/P | PASS/AP | Missing/error zoom policies not generally chosen |
| NB02 XGB counts/classes/fg_mask/confidence summaries/confusion | Counts F3/G2/M1, flag sum5, mean confidence.5, accuracy3/6; exact3x3 matrix | I/P | PASS/AP | `embedding_all_sections_exact`; reporter summarizes supplied mask/confidence, no definition/calibration imposed |
| Positional accuracy versus actual identity | Same6 rows with first2 labels permuted:1 vs4/6; `accuracy_is_positional_not_identity_proof` | I/S | arithmetic PASS/AP; identity PA | #50/#88; no IDs accepted for reference population, cannot detect external equal-size permutations |
| Model path vs prediction figure | **proof** PNG is both paths.prediction_figure and summary.model_path | I/S | FAIL/AP defect | #95; no actual model-path input exists |
| NB02 stability/sensitivity/timing/params/negative control/3D exports | Exact table/shape/engine mapping; `embedding_all_sections_exact` | I/P | PASS/AP | params passed through; missing effective params #68, not invented |
| NB03 explicit Z/history/reconstruction versus encode/history fallback | Asymmetric losses [8,1,4,6], projection residual2/18=1/9 | I/P | PASS/AP | `autoencoder_history_latent_and_mse`; supplied X_recon is stored, metric comes from method actually called |
| NB03 actual PCA MSE2 and n95, explicit count, empty history | Discarded energy2/18, all-components error0; real fitted PCA | I/P | PASS/AP | `autoencoder_real_pca_correlations_and_interpolation`, `empty_history_and_explicit_pca_count` |
| NB03 Spearman nonmonotone and ties | Ranks1,2,4,3,5,6 ->33/35; average tied ranks with scalar fsum covariance | I/P | PASS/AP | `autoencoder_tied_rank_oracle`; only positional input agreement assumed |
| NB03 insufficient/constant/nonfinite correlation cases | Two valid params ->None; constant ->None+warnings; latent Inf ->23/35 | U/I/P | observed/PA; Inf FAIL/AP defect | #97; latent two finite pairs currently yield1 despite6 physical valid; minimum pair policy open |
| NB03 one coordinate | **proof** unguarded axis2 IndexError | I | FAIL/AP robustness gap | #98; NB03 nominal latent_dim2, not historical impact |
| NB03 swallowed AE/PCA metric failures | Injected numerical ValueError; **proof** null without diagnostic | I | FAIL/AP defect | #96, propagation/status/log alternative open |
| NB03 interpolation, poles, tables, model path and other embedding arrays | Endpoint displacement(3,4)->5; exact idx4/1,Teff3000/7000, rows/tables | I/P | PASS/AP | `autoencoder_real_pca_correlations_and_interpolation`; input units/provenance supplied, not inferred |

## Artifacts, consumers and reproducibility boundary

Each notebook emits three trusted joblib copies plus two JSON and two TXT files
on a successful nominal call. `run_dir` groups by notebook family, **not** by
timestamp. Unique timestamps preserve old stems; repeated timestamps overwrite
them (#93). Latest/legacy are full copies, not symbolic/atomic pointers.

| Entry | Directory / timestamped stem | Latest / legacy | Payload / report content actually saved | Reuse / missing provenance |
|---|---|---|---|---|
| NB01 | `runs/phy3500_pca/phy3500_pca_run_<UTC>.{joblib,json,txt}` | same stem `_latest`; root `phy3500_pca_output.joblib` | Full/truncated feature+spectra scores, labels, metadata+index, ordered feature names, variance summary, counts, n95, feature stem; JSON paths/tables/inventory | NB02 loads legacy `scores_95pct,y,meta,n_components_95pct,features_stem`; no fitted PCA/scaler, input X, spectral grid, stable data/model hash or joblib internal timestamp |
| NB02 | `runs/phy3500_umap_tsne/phy3500_embeddings_<UTC>.joblib`; `phy3500_umap_tsne_run_<UTC>.json/txt` | respective `_latest`; root `phy3500_embeddings.joblib` | UMAP/TSNE/3D/negative control, original y/meta/clusters/presentation labels, predictions/confidence, stability/sensitivity, supplied params/times, feature stem, zoom; joblib timestamp; JSON summary+PCA path+figures | NB03 loads root legacy; one other NB03 cell constructs a latest path relative to MODELS_DIR (static consumer boundary, not executed). No fitted engines, input PCA scores, effective aligned reference labels/clusters/fg_mask or observation IDs for predictions preserved as separate aligned population |
| NB03 | `runs/phy3500_autoencoder/phy3500_autoencoder_run_<UTC>.{joblib,json,txt}` | same stem `_latest`; root `phy3500_autoencoder_output.joblib` | Latent/reconstruction, y/meta/index, ordered features, history, optional tables/PCA scores/embeddings/interpolation, poles/n95/fit time/MSE; JSON PCA MSE/correlations/model-path string | Post-analysis possible under matching input/schema. Checkpoint path is only a reference, not verified bytes; no model/optimizer/input matrix/scaler in this payload, no internal joblib timestamp |

No entry records a source Git SHA, cryptographic input/output hash, manifest
schema version or software dependency versions. Audit ledgers do record hashes
and environment for this review; those must not be attributed to reporter output.
Paths are serialized as supplied (absolute or relative), not resolved/validated
or content-addressed. NB02 includes only `engine.params_used`; #68 documents
missing effective seeds/dimensions/defaults upstream. NB01/NB03 do not record
full PCA/AE configurations. Report filenames and JSON/TXT timestamps identify
the second; they are not collision-resistant IDs or integrity proofs.

The reports retain some numeric results without all information to independently
recreate them. In particular NB02 stores original y/clusters with predictions,
but not `y_aligned`/`cl_aligned`; if those supplied populations differ, exact
reported accuracy/cluster correspondence cannot be reconstructed from joblib
alone. The suite verifies arithmetic for received vectors and preserves raw
inputs; it does **not** claim an independently reidentifiable experiment. A
manifest/ID-transport design needs a scientific decision (#20/#50/#68/#88).
No red test invents nonexistent provenance for old runs.

All figure inventory fields enumerate existing immediate `.png` files; NB02
also enumerates `.html`. They do not validate image pixels or linkage to an
execution. The46 probes demonstrate `old_run.png` included in a new NB02 report.
Notebooks reuse `reports/figures/phy3500/<features_stem>`, so the actual caller
can share old figures across runs and notebooks. This is **PA/open provenance**,
not a demonstrated ability to infer which old figure belongs to which run.
Provided HTML paths/model paths need not exist; file authenticity was not tested.

Re-executing a reporter after an injected failure can repair ordinary aliases
when all writes succeed. There is no recovery reader, rollback, completion
marker or invalidation mechanism. A partial group is not a validated run (#94).
Loading local synthetic joblib proves roundtrip under identical versions; it
does not prove cross-version compatibility, tamper resistance, trust or science.

## Adversarial findings and policies

All seven NEW findings have archived executed proofs and separate open issues:
[JSON #92](https://github.com/PhD-Brown/AstroSpectro/issues/92),
[collisions #93](https://github.com/PhD-Brown/AstroSpectro/issues/93),
[partial publication #94](https://github.com/PhD-Brown/AstroSpectro/issues/94),
[model path #95](https://github.com/PhD-Brown/AstroSpectro/issues/95),
[diagnostics #96](https://github.com/PhD-Brown/AstroSpectro/issues/96),
[finite correlations #97](https://github.com/PhD-Brown/AstroSpectro/issues/97),
[latent dimension #98](https://github.com/PhD-Brown/AstroSpectro/issues/98).
Priorities P1 for lost archives/mixed aliases, P2 for other bounded findings;
frequency and scientific impact on historical runs are unmeasured.

CONNU: optional facade #21; group independence #22; feature origin #23;
classifier persistence #38; loader identity #50; effective embedding params #68;
bridge populations/preprocessing/probabilities #88/#89/#90. #69 is analogous
finite-pair logic in another module, not a duplicate of reporter #97. M25
training-state/NaN issues #80–#83 and train-loss semantics #84 are not recreated.
The reporter copies train loss and separately calls reconstruction_mse; these
are not silently asserted to be the same per-observation quantity.

NON-BUG / open input policy: bool JSON np.scalar refusal, negative formatted
duration, pandas negative head, noninteger cluster coercion outside HDBSCAN's
integer labels, equal-frequency order, all-NaN statistics as raw unavailable
values, array histories giving epochs=None (real AE returns list histories).
Probes observe nonfinite loss accepted and serialized (#92/#82), and missing
labels dropped in pandas value_counts. Their scientific interpretation is PA;
green contracts do not require these policies to remain forever.

SUSPECT / not measured: simultaneous writers/races, filesystem power loss,
catalog size/memory stress, real corrupt archives, changes in backend versions,
inconsistent arbitrary supplied summary tables or externally reordered
metadata. Their frequency or correct handling is not inferred from the suite.

OPEN: reject versus unavailable/status for invalid metrics; finite-pair count
and pairwise/complete-case masking; IDs/reference population transport;
confidence definition from M26; collision strategy; transaction/manifest format;
software/hash provenance; figure ownership. No green assertion chooses one
unresolved scientific policy. Constant correlations are not physical evidence.

## Mutation evidence

[Final replay](m27_mutation_results.json): baseline84PASS, **18KILLED / 0SURVIVED**,
six separately counted admissible controls PASS. Every changed source site is
unique, AST parsed, present in the executed-line trace, with the same84-case
collection and no ERROR/SKIP; failures are assertions rather than imports.
All source/test protected LF hashes remain unchanged during replay. Disposable
trees contain only required source/test/config files; no real data are copied.

| Mutants | Violated independent contract | Final result |
|---|---|---|
| M01 mean->median; M02 sample std; M03 wrong reduction axis | Asymmetric analytic means/population variance/axis order | KILLED |
| M04 percentile5->25; M05 vector mean->median; M06 include noise | Manual finite population/linear quantiles/integer noise counts | KILLED |
| M07 PCA score-row roll; M08 labels roll; M09 wrong retained components | Exact marked rows/labels; specified threshold subsets | KILLED |
| M10 latest-path substitution; M11 wrong galaxy alias; M12 permuted confusion predictions | Existing expected alias path, manual normalized labels/confusion classes | KILLED |
| M13 best loss->last; M14 PCA2 evaluated with1; M15 latent rows rolled | Asymmetric history, discarded energy, exact coordinate/row order | KILLED |
| M16 model-path->figures; M17 wrong Spearman axis; M18 wrong joblib timestamp | Supplied model provenance, rank oracle, cross-format timestamp | KILLED |

Controls C01 equivalent sum/n, C02 dictionary ordering, C03 extra TXT spacing,
C04 explicit strict JSON refusal, C05 float32 embedding-summary arithmetic,
C06 alternative unsupported-type diagnostic all pass84 tests each. Controls
are not incorrect mutants or evidence of calibration/provenance validation.
Floating arithmetic uses pytest approx at ordinary1e-6 scale, exact preservation
for stored arrays/labels, and1e-14/1e-25 checks for the exactly diagonal synthetic
PCA integration (near-zero numerical residual bound1e-28). These are fixture
precision tolerances, not inferred observational uncertainty.

The [initial ledger](m27_mutation_initial_results.json) is retained honestly:
M18 initially SURVIVED because timestamp in the payload was not compared.
The test was strengthened; [targeted follow-up](m27_mutation_followup_results.json)
then the full final replay kill it. Initial native pytest assertion text lacked
the literal `AssertionError`, so the runner initially marked some numeric FAILs
NON_ASSERTION_FAILURE; final classification also recognizes `E   assert`.
Initial M10 included FileNotFoundError in two alias-load checks; explicit
existence assertions now fail before loading, and all final M10 failures are
assertions. The historical ledgers are not relabeled as final evidence.

## Durable red archive

Fixed ref `audit/43-dimred-run-reporter-proofs` at
**[51fe4d1ff3645aea28cda59f0260acc86fa0c834](https://github.com/PhD-Brown/AstroSpectro/tree/51fe4d1ff3645aea28cda59f0260acc86fa0c834)**:
[proof source](https://github.com/PhD-Brown/AstroSpectro/blob/51fe4d1ff3645aea28cda59f0260acc86fa0c834/tests/proofs_dimred_run_reporter.py),
[manifest](M27_PROOFS.md), initial ledger in that tree. Pushed and SHA read
back; source fetched through GitHub before normal removal from the green head.
The archive is also an ancestor. Committed replay: **18FAIL/1PASS/0ERROR/SKIP**
in [archive replay ledger](m27_archive_replay_results.json). No18-bug claim.

From the delivered green head, `check_m27_mutations.py --proofs` reconstructs
proof+fixture bytes from this fixed ref into a disposable tree and executes
the unchanged production. Direct replay can use the archived checkout instead.
No proof is xfailed or reclassified as a passing production contract.

## Validation and remaining limits

All commands use the interpreter above. [Validation ledger](m27_validation_results.json)
contains actual commands, stdout/stderr, environment/versions and tested-byte
hashes; its HEAD is the active parent during testing, distinct from effective
working-tree file hashes. Later documentation/result commits do not alter
tested production/tests. Final delivery SHA, remote equality and hosted CI are
linked in the PR and #43 handoff after they exist.

| Executed command | Result |
|---|---|
| `python -m pytest -q tests/test_dimred_run_reporter.py -W error` | **84PASS**, no warnings/errors/skips |
| `check_m27_mutations.py --reverse` (reversed collection, hash seed7) | **84PASS**, no warnings/errors/skips |
| `python -m pytest -q` | **602PASS/62deselected/7 preexisting Gaia warnings**, Windows |
| `python -m pytest -q --import-mode=importlib` | **602PASS/62deselected/7 same warnings** |
| Full bounded mutation replay | **18KILLED**, baseline +6controls PASS |
| Explicit committed archive replay | **18FAIL/1PASS/0ERROR/SKIP**, expected |
| `check_m27_boundaries.py` | **19signatures/46observations**, warnings/invalid policy recorded separately |
| Black/Ruff (source/tests/scripts/app +3M27 audit helpers), pip check, git diff/check and protected-path equality | **PASS**; Black41 files unchanged, Ruff all checks passed, no broken requirements |

NA: actual astronomical catalogs/FITS/runs, held-out evaluation, absence of
training leakage/duplicate stars, probabilistic calibration, astrophysical
validation, full ordered notebook execution, GPU/W&B/network, images/HTML visual
quality, cross-version joblib/model compatibility, integrity/authenticity,
exhaustive filesystem failures and line coverage. No external catalog/model
loaded, no raw data touched, no real figure modified. Tests use tmp_path;
mutation/probe helpers only remove their own synthetic TemporaryDirectory.
All writes in production stay exactly as received in the base blob.

The three notebooks' current save/load cells, figure directories and argument
contracts were read without execution. API source consumers and previous audit
matrices were inspected; this does not repeat M2.8/M6 notebook/visualizer audits.
Successful writing, exact roundtrip, cross-artifact consistency, observation
identity, data/model provenance, reproducibility and astrophysical validity
remain separate levels of evidence. First independent Claude review is next;
Draft status and human merge decision remain required. M2.8 is not started.
