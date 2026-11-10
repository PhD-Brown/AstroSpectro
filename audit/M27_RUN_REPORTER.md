# M2.7 — RunReporter scientific and artifact audit

2026-11-09 · Refs #43 · In progress, tests/evidence/documentation only.

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
