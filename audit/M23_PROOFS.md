# M2.3 — executable proof manifest

2026-11-08 · Refs #43. Production base `87bc1aca831c934e926eeb0756c7248ec0603c7a`.
The archive ref is `audit/43-dimred-embedding-proofs`; its immutable SHA is linked
from the integration matrix and corrective issues after publication. All source,
fixtures and tests are in that same tree. No xfail and no production patch.

Run from this archived checkout with the project's Python 3.11 environment:

```powershell
.venv/Scripts/python.exe -m pytest -q tests/proofs_dimred_embedding.py --tb=short --junitxml=test-results/m23-proofs.xml
```

The exit code is **1**: expected red assertions and reproduced runtime failures.
**24 cases: 16 FAIL, 8 PASS, 0 ERROR/SKIP/XFAIL**. These are not 16 bugs: counts
include repetitions of mechanisms and proposals for still-open boundary policies.
The machine-readable JUnit-derived case ledger is `audit/m23_proof_results.json`.
No optional UMAP integration is needed to replay this archive.

| Case (prefix `test_` omitted) | Cases/result | Classification and cause / oracle |
|---|---:|---|
| `params_used_records_seed_and_output_dimension[random_state/n_components-umap/tsne]` | 4 FAIL | NEW metadata defect: effective seed=17 and output dimension=3 never appear. Separate cases assert each field. Reporter serializes this incomplete map. No complete estimator-default policy selected. |
| `tsne_n_jobs_override_is_forwarded` | 1 FAIL | NEW forwarding defect: explicit `n_jobs=-1` plus kwargs n_jobs=1 raises duplicate-key TypeError before fit. |
| `tsne_solver_kwarg_reaches_estimator` | 1 FAIL | Same forwarding issue group, second cause: wrapper selector `method` collides with sklearn solver `method`. This is a minimal API obstruction proof; eventual regression must follow the selected replacement API, not require this impossible call forever. |
| `failed_fit_cannot_mix_model_and_cached_result[False/True-umap/tsne]` | 4 FAIL | NEW state defect: helper replaces `_model`/parameters before fit succeeds; initial failure exposes unfitted model; refit combines new unfitted model with old embedding/time. Controlled estimator failure after constructor, not fixture errors. Retention and invalidation both accepted by oracle. |
| `real_tsne_failed_refit_does_not_expose_old_coordinates` | 1 FAIL | Same state defect using actual sklearn, not a mock: fit on 16x4, then NaN refit; old coordinates survive next to a new failed model. |
| `3d_correlations_use_finite_pairs[umap/tsne]` | 2 FAIL | NEW finite-pair defect: 102 metadata values, one embedding-axis NaN, 101 valid perfectly ordered pairs; reports '?' / 0 instead of rho=1. Explicit ValueError rejection is also allowed. Injected estimator output characterizes wrapper boundary; real TSNE normally rejects nonfinite input. No incidence claim about real catalogs. |
| `stability_rejects_unknown_metric` | 1 FAIL | NEW validation defect: a garbage metric silently produces Procrustes because argument is never read. No additional metric implementation required. |
| `open_nonpositive_seed_count_has_explicit_outcome[0/-1]` | 2 FAIL | OPEN remedy / weak validation defect: raw IndexError from `embeddings[0]`. Example oracle permits explicit ValueError or empty report. Final n_seeds domain/schema is undecided; these do not enter green CI. |
| `observe_single_hyperparameter_convention` | 1 PASS | NOT A BUG under the explicit one-hyperparameter source convention: only first key is varied, second ignored. Warning/rejection vs expanded API remains OPEN. |
| `observe_duplicate_grid_values_collapse` | 1 PASS | OPEN result-key policy: duplicate values run twice and last result wins, one key. No Cartesian-grid or multiplicity promise invented. |
| `observe_empty_grid_mapping` | 1 PASS | OPEN validation: `{}` raises raw IndexError; `{'key': []}` is separately green and returns empty results. |
| `observe_pca_slice_boundaries[0/-1]` | 2 PASS | OPEN validation: Python slicing delegated (0 reaches empty backend input, -1 omits last column); not an accepted domain policy. |
| `observe_transform_requires_preselected_features` | 1 PASS | OPEN convenience policy: fit truncates PCA features, transform delegates all supplied columns. Preselection works; passing the original full width fails. Same-width fit/transform is protected in green. |
| `counterexample_univariate_permutation_keeps_bimodality` | 1 PASS | Scientific-documentation counterexample: permutation keeps two-point support and exact multiset of pairwise distances. Does not measure an actual UMAP output or assert any definition of uniformity in latent space. |
| `counterexample_pca_scores_are_not_implicitly_whitened` | 1 PASS | Confirmed documentation error: actual default PCA scores have sample variances 3.6, 1.6, 0.4, rather than 1. Existing M2.2 analytic fixture supplies independent covariance. |

The green suite is intentionally compatible with explicit validation, corrected
metadata and coherent failed-fit policies. Aliasing/copy ownership, invalid PCA
counts, zero seeds, multi-key/duplicate grids, transform preselection and scientific
thresholds must be decided in corrective work, not inferred from these observations.
