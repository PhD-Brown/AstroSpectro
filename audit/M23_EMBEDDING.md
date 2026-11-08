# M2.3 — embedding contracts, tests and residual defects

2026-11-08 · [Refs #43](https://github.com/PhD-Brown/AstroSpectro/issues/43).
Tests and audit only. Independent Claude review and human merge decision pending.

## Reference, preparation and scope

Production examined: **`main@87bc1aca831c934e926eeb0756c7248ec0603c7a`**,
`src/dimred/embedding.py`, **790 lines** (`str.splitlines()`; supplied context reported 791), blob
`57ecb8c8627c03bc02cec2eeff2258a01afda929`.
`git fetch origin` confirmed the supplied main SHA. GitHub confirms #56 merged
at 3604b3d and #64 merged at 87bc1ac. Initial local checkout was clean on
`audit/43-dimred-pca@d9609d1fb31739af07f6554563c79dbe31d4beea`; it was
preserved. Authorized branch `audit/43-dimred-embedding` starts at origin/main;
the old local main ref was not reset or rewritten. No production, dependency,
CI, notebook, data or reference scientific artifact is changed.

Read current AGENTS.md v1.3 on main first, #43 and all **22** comments present
at preparation, [M1](https://github.com/PhD-Brown/AstroSpectro/issues/43#issuecomment-6699300347),
historical audit, M21/M22 matrices/tests/fixtures/replays and #21/#23/#24/#59–#65
with their comments. Entire embedding source, its notebook call cells, relevant
visualizer/reporter consumers, HDBSCAN input boundary and CI/configuration were
inspected. #21 optional facade, #23 provenance and #24 training leakage remain
separate. No PCA re-audit or M2.4 clustering audit was performed.

Existing tests before M2.3: **none targeted at embedding.py**. The unchanged
baseline suite executed locally: **219 PASS / 7 existing Gaia Pandas4Warning**.
Source imports deliberately bypass `dimred/__init__.py` (#21); this does not
validate the public optional-dependency facade.

## Independent expectations and limits

Four separate questions apply: wrapper/algorithm calculations, conditional
technical reproducibility, measured stability, and scientific interpretation.
Passing one does not prove another. These synthetic tests have no physical
units and do not validate real Gaia matches, spectra, clusters or held-out
scientific performance. No line-coverage measurement is claimed.

- API spies capture exact estimator kwargs and row values, not just successful
  calls. They run the real wrapper and explicitly **do not validate UMAP/t-SNE**.
- Stability uses a fixed sampled population, seeds `0..n_seeds-1`, and seed 0
  as reference. `self.random_state` controls sampling, not that seed sequence.
  Similarities (translation, scale, rotation/reflection) have zero residual;
  anisotropic scaling generally does not. The independent optimum is
  `1 - sum(svd(A.T @ B))**2` for centered unit-Frobenius A/B.
  This matches the least-squares similarity objective, not an astrophysical
  hypothesis test. [SciPy's definition](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.procrustes.html).
- Numerical tolerances: 2e-27 for near-zero residuals on simple similarities;
  3e-14 for the independent SVD disparity, allowing float64 accumulation over
  at most 5000 rows; 2e-15 for exact rank ±1. Embedding reproducibility uses
  absolute 2e-6 with rtol=0, restricted to one dependency/process/thread setup.
  No coordinate equality across versions/architectures is required.
- Negative control preserves each column's empirical marginal, input shape and
  caller data. A deterministic correlated example loses its perfect association;
  no universal bound on correlation after finite random shuffling is asserted.
  The tested <0.5 coefficient belongs only to that fixed example.
- All 3D arrays/tables are checked against independent observation markers.
  Monotone nonlinear metadata distinguish Spearman from Pearson. Signed maximum
  absolute correlation, finite metadata masking, the **implemented** 100-pair
  threshold, selected population, optional columns, plotting quantiles and
  row-aligned class/cluster labels are checked. This threshold is a software
  convention, not an independently justified robustness criterion.
- PCA scores are supplied explicitly. Unsupervised exploratory fitting is not
  proof of train-only transformations or absence of object overlap (#24).
  Real PCA→t-SNE integration checks only interface dimensions and reproducibility.

## Full significant-behavior matrix

G = `tests/test_dimred_embedding.py`; A = the separately published proof archive.
Test names below omit `test_`. U/I/P/S = unit/interface/numerical/scientific-risk
contract. **AP** always means a named executed contract; **PA** means remaining
limits, never tacit acceptance. Every method has more than nominal inspection.

| Code / inputs → outputs | Preconditions, defaults, dependencies and side effects | Tests / executed assertions | Disposition and residual limits |
|---|---|---|---|
| Module imports/defaults 1–83 | numpy/pandas mandatory; umap/sklearn/Plotly local imports; project defaults differ from library defaults | G import isolation, missing umap, defaults/overrides: PASS U | AP import boundary; facade already #21. Scientific prose has confirmed whitening error and broader overclaims below. |
| `__init__` 85–104 | method umap/tsne, dimension 2, seed 42, backend kwargs → empty model/cache/time | G `constructor_and_unfitted_guards`, `unknown_method_rejected`: PASS U | AP selector/initial state. Other parameter validation deferred; backend method collision confirmed. Arbitrary types not exhausted. |
| `fit_transform` 110–160 | X(N,D), optional first k columns → Z(N,C); timing and cache updated after backend fit | G `fit_dispatch_dimensions_parameters_and_time` (None/1/3/oversized k), `backend_defaults_and_successful_refit`, real PCA integration and invalid TSNE inputs: PASS U/I | AP dispatch, successful refit, dimensions, kwargs, timing and caller integrity. Failed-fit mixed state confirmed in A; zero/negative k OPEN. No new PCA is fitted by this method. |
| `transform` 162–178 | Fitted UMAP, input width matching fitted data → new coordinates; TSNE has no transform | G `transform_delegates_without_refit`, `tsne_transform_is_explicitly_unsupported`, unfitted guards; optional real UMAP: PASS U/I | AP no refit/cache mutation and explicit unsupported TSNE. Fit-time PCA slicing is not reapplied to new full-width scores; A observes this, convenience policy OPEN. New-sample preprocessing remains caller responsibility. |
| `stability_report` 184–272 | X, default 5 seeds, optional PCA slice, metric string; at most 5000 sampled once, fresh engine per seed, DataFrame seed/disparity/method | G `stability_sampling_seeds_geometry_and_parent_isolation` (8/5000/5013 rows ×1/2/3 seeds), `alignment_failure_is_missing_not_false_zero`, `stability_fit_failure_propagates`, fitted-parent isolation: PASS U/P | AP sample identities/seed sequence/reference/independent SVD/similarities, input and parent integrity, alignment NaN and fit propagation. Metric ignored; zero/negative seeds raw IndexError. Exceptions during alignment broadly swallowed. Degenerate reference still gets artificial reference distance0; no valid comparison evidence from it. |
| `negative_control` 279–310 | copy X; independent column shuffles using seed+999; fresh same-method/dimension/seed/kwargs engine, optional k slice | G `negative_control_marginals_permutation_and_parameters` (2 methods ×4 shapes/k), fitted-parent isolation: PASS U/S | AP marginals, nonmutation, determinism and changed-seed response, correlated example, constant/one-column/small-N wrapper boundaries. No guarantee of uniform embedding or loss of all structure. Algorithm feasibility on tiny inputs remains backend-dependent. |
| `parameter_sensitivity` 316–358 | Default values5/15/30/50/100 for neighbors/perplexity; documented source convention: first hyperparameter only; one fresh engine/trial; key=name=value | G `sensitivity_grid_override_and_isolation`, TSNE invalid/incompatible grid propagation, fitted-parent isolation: PASS U/I | AP exact kwargs precedence, output keys/arrays, no shared arrays, seed/PCA count, None vs empty values. Multi-key mapping, duplicate values and `{}` observed in A; policies OPEN, no Cartesian-grid contract invented. Defaults need not be feasible for small N; error propagates. |
| `embedding` 364–369 | last successful coordinate array, guard before success | G guard, fit/refit, transform/analysis parent integrity: PASS U | AP cached values; mutable ownership/copy policy not selected. Failed refit exposes stale coordinates in A. |
| `params_used` 371–374 | empty before fit; helper's merged project defaults/user kwargs after constructor begins | G known default/override fields and reporter round trip: PASS U/I | PA complete provenance: seed/dimension and forced TSNE jobs omitted. Completeness defect in A; no claim that caller dictionaries are immutable snapshots. |
| `_fit_umap` 380–396 | lazy umap import; merged defaults/user kwargs; model/params assigned before fit | G spy exact defaults/kwargs, dependency error, optional real fit/transform/invalid/constant cases: PASS U/I | AP wrapper merge/import plus local optional algorithm integration. Failed-fit state remains defect. Large/degenerate graphs, sparse/precomputed/custom metrics, spectral initialization failures and disconnected NaN output not exhaustively tested. |
| `_fit_tsne` 398–413 | sklearn TSNE; merged defaults; C/seed explicit, n_jobs=-1 forced; creates model before fit | G defaults, override fields, real 2D/3D fits and nonfinite/dimension/perplexity/iteration errors: PASS U/I | AP normal Barnes-Hut path, finite output in small controlled runs. n_jobs override TypeError; solver `method` obstructed at wrapper selector. Exact solver not certified through public configuration. |
| `compute_umap3d` 415–591 | all PCA scores; separate 3D engine, explicit seed 42/neighbors 15/min_dist 0.1; positional meta/y/clusters; figures_dir must exist; optional physical exports | G `3d_tables_alignment_sampling_and_exports`, threshold99/100/101, rank/nonlinear and paired/independent permutations, constants, invalid lengths/subsample, missing Plotly: PASS U/P | AP full backend input, 3D params, signed selection, tables, classes/noise labels, sampled row identities, path requests/quantiles. Actual UMAP 3D rendering not run; plotting tables tested with controlled coordinates, actual UMAP 2D fit tested separately. Coordinate finite mask incomplete in A. No ID join/length identity proof is possible from equal-sized arrays. |
| `compute_tsne3d` 593–752 | all scores; direct TSNE(C3,perplexity 30,seed 42,jobs -1,verbose1), own timing; analogous metadata/correlation/class/T_eff exports | Same G 3D matrix + `real_tsne3d_plotly_export`: PASS U/I/P | AP exact arguments, shape, full table and paths, actual small TSNE and HTML generation. Deliberately separate fixed pipeline config from parent kwargs; not promised inheritance. Same finite-pair defect as UMAP. Real method uses sklearn iteration defaults and can be expensive on full catalogs. |
| `compare_embeddings` 755–790 | methods default[umap,tsne], k20, seed 42 → ordered method→Z map; sequential independent engines | G `compare_order_dimensions_and_parameter_forwarding`: PASS U | AP default/subset/reversed/empty lists, PCA slicing, per-engine seed and distinct result arrays. Contrary to prose, executions are sequential, not parallel. Repeated method keys collapse (OPEN multiplicity policy); failure propagates without partial result return. |
| PCA/visualizer/reporter/clustering boundary | aligned scores/metadata/labels; no automatic object identity reconciliation | G real PCA→TSNE, stability table→visualizer PNG, TSNE3D→Plotly HTML, actual reporter joblib/JSON round trip: PASS I | AP tested payload rows/params/shapes/array values. HDBSCAN fit accepts Z rows and returns labels; only their consumer table is tested here. Real HDBSCAN and full notebook execution deferred to their lots. |

## Defects, observations and decisions

**Durable archive:** [`audit/43-dimred-embedding-proofs@c16571ecb55e12b9fa7b4dda2a62c7d96ff41b0b`](https://github.com/PhD-Brown/AstroSpectro/tree/c16571ecb55e12b9fa7b4dda2a62c7d96ff41b0b).
[Executable proofs](https://github.com/PhD-Brown/AstroSpectro/blob/c16571ecb55e12b9fa7b4dda2a62c7d96ff41b0b/tests/proofs_dimred_embedding.py),
[per-case manifest](https://github.com/PhD-Brown/AstroSpectro/blob/c16571ecb55e12b9fa7b4dda2a62c7d96ff41b0b/audit/M23_PROOFS.md),
[JUnit-derived results](https://github.com/PhD-Brown/AstroSpectro/blob/c16571ecb55e12b9fa7b4dda2a62c7d96ff41b0b/audit/m23_proof_results.json)
and matching fixtures were pushed and read back from GitHub **before removal**
from the green integration tree. Re-executed at committed c16571e:
**24 cases / 16 FAIL / 8 PASS / 0 ERROR/SKIP/XFAIL**. The archive is an ancestor
of integration; history was not rewritten. Proof counts are not bug counts.
Searches for embedding/umap/procrustes/stability/params_used/fit and
the existing #21/#23/#24/#59–#65 found no duplicate targeted embedding ticket.
No production fixes belong in M2.3.

1. **NEW — [#66 t-SNE option collisions](https://github.com/PhD-Brown/AstroSpectro/issues/66).** `n_jobs` appears twice; `method` is already
   bound by the wrapper selector. Public promise of directly forwarded kwargs
   is not true for these valid sklearn options. Solver-option naming/backward
   compatibility must be decided in corrective work.
2. **NEW — [#67 mixed state after failed fit](https://github.com/PhD-Brown/AstroSpectro/issues/67).** Both backends replace estimator and
   params before success; old embedding/time remain after failed refit. Real
   TSNE+NaN reproduces it. Retain last coherent model or invalidate everything:
   both remedies remain open; no green assertion chooses one.
3. **NEW — [#68 incomplete parameter provenance](https://github.com/PhD-Brown/AstroSpectro/issues/68).** `params_used` omits explicit seed,
   dimension and TSNE jobs; reporter persists this map unchanged. The issue is
   reproducibility metadata, not a demonstrated historical-result change.
4. **NEW — [#69 incomplete 3D finite-pair masking](https://github.com/PhD-Brown/AstroSpectro/issues/69).** Only metadata finiteness is
   checked; one NaN in axis3 masks a valid 101-pair rho1 as '?' / 0. Rejecting
   nonfinite coordinates vs excluding pairs remains a corrective API choice.
5. **NEW — [#70 stability argument validation](https://github.com/PhD-Brown/AstroSpectro/issues/70).** Unsupported `metric` silently gives
   Procrustes. Nonpositive seed counts hit internal indexing. Define explicit
   accepted domain/diagnostic; no extra metric implementation is requested.
6. **NEW — [#71 documentation error / interpretation limitations](https://github.com/PhD-Brown/AstroSpectro/issues/71).** PCA is not
   implicitly whitened: actual analytic scores have variances3.6 / 1.6 / 0.4.
   Independent column permutations preserve marginal structure (including a
   univariate bimodal example); they do not guarantee a uniform latent result.
   Axis correlations are pairwise and do not establish a physical parameter
   independent of T_eff. Ordinary UMAP is not the separately trained neural
   Parametric UMAP. Topology/density/global-distance preservation are not
   universal guarantees. The 0.05 Procrustes label is an unexplained convention,
   not calibrated significance or external scientific validation.

**OPEN / NOT A BUG under current documented preconditions:** positional row
alignment without IDs; single-key sensitivity convention; duplicate key/value
collapse; invalid/empty grid API; PCA slice bounds; full-width transform after
truncated fit; constant fit acceptance; output-path creation/atomicity; ties and
undefined/zero correlations. With no valid winning correlation the current
result '?' / 0 conflates no information and zero association; redesign is open.
Inf-valued physical colors are only dropna-filtered in export tables, so finite
plot-value/range validation remains a static risk, not a separately proven bug.
All figure exports use fixed names and may partially complete before an I/O
error; interruption/permission/disk-full scenarios were not exercised.

Primary references: [sklearn TSNE parameters and perplexity](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html),
[PCA whitening](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html),
[UMAP parameters](https://umap-learn.readthedocs.io/en/latest/parameters.html),
[Parametric UMAP](https://umap-learn.readthedocs.io/en/latest/parametric_umap.html).
Library defaults/requirements are also checked against the actually installed
versions; current web documentation does not certify archived results.

## Executions and optional environment

Windows, Python **3.11.9**, pytest 8.4.2, pytest-socket 0.7.0, numpy 2.4.6,
pandas 3.1.1, scipy 1.17.1, sklearn 1.9.1, Plotly 7.1.0, threadpoolctl 3.8.0,
Black 24.4.2, Ruff 0.5.0. Baseline `.venv` has **no umap-learn**; neither
pyproject.toml nor requirements/ci.txt declares it. CI remains unchanged.

Optional integration installed **umap-learn 0.5.12 / pynndescent 0.7.0** using
`uv pip install --target test-results/m23-optional-deps --no-deps umap-learn pynndescent`.
Its imports use scoped PYTHONPATH with existing numba 0.68.1/llvmlite 0.50.0;
no project/CI or venv dependency was changed. First one-case compile/fit run
passed in 28.95s; expanded optional run **8 PASS** in 16.70s. No service/GPU,
scientific catalog or real figures directory was used. All exports use tmp_path.

Prefix below: `.venv/Scripts/python.exe` from repository root.

| Command / scope | Executed result |
|---|---|
| `-m pytest -q --junitxml=test-results/m23-baseline.xml` before additions | 219 PASS / 7 existing Gaia warnings |
| `-m pytest -q tests/test_dimred_embedding.py -W error --junitxml=test-results/m23-green-final.xml` | 94 PASS / 8 optional cases deselected; no skips/xfails |
| Scoped optional PYTHONPATH, `-m pytest -q tests/test_dimred_embedding.py -m slow -W error --junitxml=test-results/m23-optional-umap-final.xml` | 8 PASS / 94 ordinary cases deselected; final 17.03s |
| `-m pytest -q --junitxml=test-results/m23-fast-final.xml` | 313 PASS / 8 optional deselected / 7 existing Gaia warnings, 8.68s |
| At committed c16571e, `-m pytest -q tests/proofs_dimred_embedding.py --tb=no --junitxml=test-results/m23-proofs-at-archive.xml` | 16 FAIL / 8 PASS / no errors/skips/xfails; causes reviewed individually |
| `audit/check_m23_mutations.py --output test-results/m23-mutations-final.json` | Baseline 94 PASS; 30 KILLED / 0 SURVIVED / 0 INCONCLUSIVE; 5 separate controls PASS |
| `audit/check_m23_mutations.py --reverse --output test-results/m23-reverse.json` | Reversed collection, PYTHONHASHSEED=7, -W error: 94 PASS |
| `-m black --check src tests scripts app.py audit/check_m23_mutations.py`; same Ruff scope | PASS, 33 unchanged with proof file / 32 after its removal; Ruff PASS |
| `-m pip check`; `git diff --check`; protected-path diff vs origin/main | No broken requirements; clean diff; production/config/CI/notebooks/data diff empty |

The initial replay on an 82-case snapshot detected all 30 mutations but failed
its final integrity guard because tests were concurrently strengthened to 94.
It is **not** the final mutation evidence and no successful-source-integrity
claim rests on it. The stable replay records hashes of real inputs before/after,
validates source syntax, executes baseline and each mutation on disposable
copies, checks equal case counts and JUnit errors, and reports a comment-only
neutral and compatibility controls separately. Source in the real workspace was never mutated.
No arbitrary mutation score or completeness claim is made.

Final replay evidence is versioned in `audit/m23_mutation_results.json`, including
per-mutant failing test names and SHA-256 of the protected real inputs. The
five controls are comment-only, complete metadata, n_jobs override, skipping
constant correlations without a warning, and rejecting a zero-size plot.
They demonstrate green-suite compatibility with these alternatives, **not
approval of corrective APIs/scientific policies**. No-op/neutral controls are
excluded from the 30 mutation count. Hashes were checked before and after the
complete stable replay. Source parsing and equal collection counts passed.

Archive/replay are outside automatic CI. Final integration and hosted CI SHAs,
run status and counts are recorded in the PR and #43 handoff. Hosted CI certifies
only its configured tests/lint/pip checks; optional UMAP was executed locally.

## Review handoff

Claude should examine the independent Procrustes/rank oracles, positional-row
preconditions, failed-fit alternatives, completeness of parameter propagation,
optional-library boundaries, mutation result classification and proof semantics.
Outstanding policies above require scientific coordination where relevant.
No merge, no correction of #59–#65, no M2.4. Scientific validity of real embeddings
and historical interpretations remains unmeasured.

## Hosted CI evidence on the first integration head

[Run 42505334680](https://github.com/PhD-Brown/AstroSpectro/actions/runs/42505334680)
completed successfully for `6325a15546728c830228d92fd385324d9766cb26`; merge
checkout `e364b5d545ed401e868247f767d0f514a28ff210`. Job logs were read:
Ubuntu 24.04.5 / Python 3.11.18, **310 PASS / 3 Windows-storage SKIP /
8 optional UMAP deselected / 7 existing Gaia warnings**, pip check PASS, Black
31 unchanged, Ruff PASS. [JUnit artifact 13627008348](https://github.com/PhD-Brown/AstroSpectro/actions/runs/42505334680/artifacts/13627008348).
This documentation-only follow-up corrects the locally counted line total and
function line anchors; source blob is identical to the supplied reference.
Tests, fixtures and mutation input hashes are unchanged. The final head's new
hosted CI is checked separately and linked in the Draft PR/#43 handoff.
