# M2.9 — Public dimred facade and dependency boundaries

## Scope and provenance

Audit/tests only on `audit/43-dimred-facade`, separate `AstroSpectro-m29`
worktree. Base: `main@db513004fcda5a5cad3ac17a2b9d08fc28946b54`, verified
after fetch; #116 is merged with that merge SHA and audited head `fbe7526`.
`AGENTS.md` v1.3, complete #43 body, recent handoffs including comment
6887479751, #21 and its M2.5 comment, and `M25_AUTOENCODER.md` were consulted.
The principal checkout was clean and remains on main; previous worktrees and
proof refs were preserved. No production/dependency/notebook/workflow/data edit.

Executable inputs and source blobs are recorded in [m29_experiments.json](m29_experiments.json),
including SHA256 of actual local bytes before/after, interpreter, Git HEAD/status,
commands, subprocess output and versions. A local byte hash is not a Git blob.
Final experiment input commit: `ae54346d49499b915f1895ad82cecb2cf14b6cd7`.
The final delivery SHA/CI is recorded in the PR and #43 handoff, avoiding a
self-referential commit hash here.

## Independent public inventory (U/I)

AST inventory: **14 unique names**, not 13; **96 lines** by `str.splitlines()`
on the actual source (the supplied 97-line estimate is not reused). The expectation
map in `tests/facade_probe.py` is independent of parsing the facade. Controlled
sibling wiring tests execute the real facade with synthetic producers; real
CPU-Torch subprocesses separately verify identities against actual producers.

| Public name | Actual producer | Proof |
|---|---|---|
| `DimRedDataLoader` | `data_loader.DimRedDataLoader` | U existence; synthetic and real identity |
| `PCAAnalyzer` | `pca_analyzer.PCAAnalyzer` | Same; isolated source fit without Torch |
| `EmbeddingEngine` | `embedding.EmbeddingEngine` | Same; late UMAP/Plotly refusal |
| `DimRedVisualizer` | `dimred_visualizer.DimRedVisualizer` | Same; global-style red proof |
| `SpectralAutoencoder` | `autoencoder.SpectralAutoencoder` | Same; #21 red import proof |
| `save_pca_run` | `run_reporter.save_pca_run` | U existence; synthetic and real identity |
| `save_umap_tsne_run` | `run_reporter.save_umap_tsne_run` | Same |
| `save_autoencoder_run` | `run_reporter.save_autoencoder_run` | Same |
| `HDBSCANAnalyzer` | `hdbscan_analyzer.HDBSCANAnalyzer` | Same; late fit/presentation refusal |
| `compute_feature_profiles` | `hdbscan_analyzer.compute_feature_profiles` | U existence; synthetic and real identity |
| `compute_sensitivity` | `hdbscan_analyzer.compute_sensitivity` | Same; late backend refusal tested |
| `xgboost_predict` | `xgboost_bridge.load_and_predict` | Actual alias identity, not name-only |
| `tester_candidat` | `autoencoder.tester_candidat` | U existence; synthetic and real identity |
| `latent_arithmetic` | `autoencoder.latent_arithmetic` | Same |

`import *` yields exactly these names (apart from Python's `__builtins__` in the
execution namespace). Every value is the same callable as its source producer.
Repeated package imports and repeated named imports preserve module/object
identity. Unknown internal attributes are not treated as public API.
`SpectralMatrixLoader`, `compare_embeddings`, `CLASS_COLORS`, `CLASS_MARKERS`
and the unaliased `load_and_predict` remain legitimate submodule imports;
their omission from facade `__all__` is not a demonstrated defect.

Metadata: `dimred.__version__ == '0.3.0'`, author `AstroSpectro — PHY-3500`.
The facade comment describes the version as the PHY-3500 API/notebook trace.
Distribution metadata describes the broader pipeline, `astrospectro==0.1.0`.
Different version units are documented in source: **no version-mismatch bug
demonstrated**, and no test requires these versions to equal each other.

## Import graph and declaration boundaries

Current eager order: loader → PCA → embedding → visualizer → AE → reporter →
HDBSCAN helpers → XGBoost bridge → AE diagnostic helpers. This is a source
inventory, not a required future ordering. No test fixes the import order.

| Imported producer | Relevant eager imports | Significant deferred imports |
|---|---|---|
| loader | NumPy, pandas, sklearn.preprocessing | Astropy FITS, joblib workers for spectral loading |
| PCA | NumPy, pandas, sklearn.decomposition, scipy.stats | joblib persistence |
| embedding | NumPy, pandas | UMAP, sklearn TSNE, Plotly, scipy.stats at 3D use |
| visualizer | NumPy, pandas, Matplotlib pyplot/patches/ticker | scipy KDE, sklearn trustworthiness; package-local colors/bridge |
| AE | NumPy, pandas; attempted torch/nn/optim/utils | Matplotlib diagnostic plotting |
| reporter | NumPy, pandas, stdlib JSON/path/time | joblib persistence; no W&B import |
| HDBSCAN helpers | NumPy, pandas, Matplotlib pyplot | hdbscan at fit/presentation/sensitivity; loader in feature profiles |
| bridge | NumPy, pandas, stdlib sys/path | utils and pipeline classifier at inference; XGBoost etc through classifier |

`pyproject.toml` and `requirements/ci.txt` require NumPy, pandas, scipy,
scikit-learn, Matplotlib, Plotly and inference libraries. Thus **Plotly is a
required distribution dependency with a tolerant local boundary**, not a
declared optional extra. Torch, UMAP, HDBSCAN and W&B are absent from both;
only the `test` extra is declared. No real facade dependency on W&B was found.
Legacy `requirements.txt` remains UTF-16 and includes Plotly but not these
four extensions: already in #21, no packaging change here.

## Dependency absence matrix (I/S)

Each probe runs `python -I -B` with explicit absolute helper/src paths, alien
temporary cwd, private Matplotlib/cache paths and CPU-only visibility. The
child verifies no `dimred` or blocked root is preloaded. A meta-path blocker
raises for the dependency **and every descendant**. Root/submodule positive
controls run before the operation; actual operation traces independently
demonstrate the blocked import. No reliance on the developer venv's absence.

| Absence / operation | Observed | Expected / classification | Evidence |
|---|---|---|---|
| Torch / `import dimred` | `NameError: nn`, AE line87 | Independent package API should remain reachable; known #21 | RED fresh subprocess |
| Torch / `from dimred import PCAAnalyzer` | Same before requested PCA binding | PCA must be usable without AE extension; known #21 | RED |
| Torch / `from dimred import SpectralAutoencoder` | Same | Import may succeed or explicitly refuse missing Torch; NameError inadmissible | RED, alternative accepted |
| Torch / `from dimred.pca_analyzer import PCAAnalyzer` | Same | Parent facade executes first, unlike source isolation | RED |
| Torch / dotted loader and embedding imports | Same | Unrelated features blocked; extension of #21 evidence, not new bugs | Two RED cases |
| All five roots blocked / seven non-AE source files loaded in isolation | Success; PCA tiny synthetic fit also reached | Available source implementations do not inherently require AE | GREEN; bypass is explicit |
| UMAP/HDBSCAN/Plotly/W&B blocked / actual facade with real Torch | Success, all14 identities/star/repeats checked | Independent facade API remains reachable | GREEN real CPU |
| UMAP / embedding.fit_transform | ImportError with `pip install umap-learn` | Explicit late refusal attributable to UMAP | GREEN isolated + real facade |
| HDBSCAN / fit, fit_presentation and compute_sensitivity | ImportError with `pip install hdbscan` | Explicit late refusal attributable to backend | GREEN isolated + real facade |
| NumPy/pandas/scipy/sklearn/Matplotlib / corresponding public feature import | ImportError family identifying blocked required dependency | Required dependencies are not treated as optional extensions; no misleading Torch NameError | GREEN five fresh processes |
| Plotly / compute_umap3d and compute_tsne3d | ImportError with `pip install plotly`, before estimator/export | Local tolerance is legitimate despite mandatory distribution declaration | GREEN isolated + real facade |
| XGBoost/LightGBM/CatBoost blocked / real facade | Success, exports retain source identity | Inference backend must not be needed merely to import its callable | GREEN real CPU |
| W&B blocked / facade and isolated reporter | Success; no actual operation import attempt | No tracking requirement established at this boundary | GREEN import, static graph |

Direct `importlib.import_module('dimred.pca_analyzer')` also executes the
parent `__init__`. `spec_from_file_location` with an unrelated module name
does not. Seven isolated-source cases establish source independence only;
they never claim that dotted Python imports work without Torch. The M2.1–8
fixtures often use this bypass (or a synthetic package/Torch namespace), so
their green results do not prove a repaired facade.

## Consumers and packaging (I)

Repository-wide tracked-file scan is retained in the experiment JSON, including
source/doc examples and comments identified as such. The three PHY-3500 notebooks
contain15 dimred import statements, with zero-based cell locations recorded.
NB01 uses facade loader/PCA/visualizer, spectral loader and reporter submodules;
NB02 uses facade embedding/visualizer plus comparison, colors, HDBSCAN, bridge
and reporting; NB03 uses facade loader/PCA/visualizer plus AE/helpers/reporter.
Static imports resolve to existing producers/constants. Real subprocesses
exercise public exports, submodule-first PCA and AE requests, and the additional
spectral-loader/comparison/color/unaliased-bridge paths. Notebook cells outside
these imports are **not executed**. No standalone PHY-3500 Python script using
the facade was found. Internal visualizer/HDBSCAN submodule imports are inventoried;
their method behavior remains with M2.4/M2.8.

Setuptools discovers top-level packages under `src` plus `utils.py`. Distribution
name does not create `astrospectro.dimred`. The checker copies Python source and
build metadata to a temporary tree, runs pip wheel with `--no-index --no-deps
--no-build-isolation`, installs the wheel into a temporary `--target`, and starts
another isolated process **without inserting source/src**. It verifies installed
`dimred.__file__`, real exports/identities/star, and both distribution/subproject
versions. Wheel contents have `dimred/__init__.py`; no `astrospectro/dimred`.
This is a real wheel build/install against the available locked dependencies;
no clean dependency resolver or new editable-install environment was tested.
The existing CI uses editable installation. Additional future namespaces and
different compatible import implementation strategies remain admissible.

## Side effects (S, import boundary only)

Successful import probes preserve `sys.path` after deliberate src insertion,
root logger level/handlers, and deny Python-audited network operations, catalog/
checkpoint reads and file writes outside private Matplotlib/cache directories.
Positive controls intercept an attempted DNS lookup and write before either
reaches a real service/file. Attempt lists survive caught exceptions. `-B`
suppresses bytecode writes. No application artifacts/catalog/model/network
access was observed in successful probes. Legitimate cache creation is allowed.

Real Torch CPU is eagerly loaded; its CUDA runtime is not initialized in these
processes (`torch.cuda.is_initialized()` false). Matplotlib pyplot is eager,
with Agg deliberately selected for headless execution. This does not establish
default interactive-backend behavior or arbitrary GPU-build behavior. W&B,
UMAP, HDBSCAN and Plotly are not eagerly needed. Module presence alone is not
called a defect. No startup-time/memory threshold is invented.

**New observed unwanted global effect:** a plain facade import applies plotting
style from `dimred_visualizer.py:48–68` to the caller's Matplotlib state. The
red proof sets grid=False/dpi91, imports the package with real Torch and observes
grid=True/dpi150. Even a PCA import traverses the same initializer. A failed
Torch-free import can also reach that update before AE fails (static order;
this second combination is not a separate measured red case).

### Separate P2 issue proposal (not created)

- **Title:** `fix: preserve caller Matplotlib style when importing dimred`
- **Problem:** unchanged `main@db513004`, visualizer lines48–68, reached from
  facade line45, overwrites caller grid/DPI on import; executable red proof and
  before/after values above. No duplicate returned by all-state GitHub searches
  for rcParams, matplotlib and style; import search identified #21 and unrelated
  #13/#43. This is separate from missing-extension handling in #21 and M2.8's
  figure-specific defects.
- **Why:** unrelated PCA/loader imports can alter subsequent figures in an
  existing notebook/session without explicit styling action.
- **Expected:** importing the API preserves caller plotting settings; explicit
  style application or local rendering context remains a design choice.
- **Verification:** the preserved red style test passes after a future fix;
  visualizer contracts remain separately verified. Temporary control A06 passes
  by deferring style application, without selecting a complete implementation.
- **Out of scope:** packaging/#21, changing numerical/scientific figures,
  notebook edits, backend policy or a production fix in M2.9.

## Adversarial campaign and proof preservation

`python audit/check_m29_facade.py` executes only temporary source copies.
10 targeted incorrect candidates: missing export, duplicate export, wrong class,
wrong alias, missing bound attribute, incomplete placeholder, sys.path mutation,
root logging mutation, masked PCA producer failure, eager optional UMAP import.
The first nine require their **specific assertion** after the modified path is
executed; errors in setup/collection are not kills. M10 is separately labelled an
expected import-boundary failure at the deliberately inserted import, verified
with a blocked UMAP positive control; it is not a numerical runtime-exception kill.
Baseline wiring passes.

Six admissible alternatives: reordered `__all__`, equivalent alias binding,
private extra state, lazy `__getattr__` facade, conditional Torch class bases,
explicitly deferred style application. First four run synthetic + real identity
checks and preserve an attributable PCA producer error, including deferred lazy
errors. Last two demonstrate that the corresponding red oracle admits a remedy;
they do not constitute approved or complete production fixes. Machine output
retains per-candidate commands/results and unchanged production/test hashes.

The 7 desired-behavior red tests live at `audit/proofs_m29_facade.py`, outside
default `testpaths=['tests']` and pytest filename discovery. Run explicitly:

```powershell
$cpu = '../AstroSpectro-m25/.venv-torch/Scripts/python.exe'
& $cpu -m pytest -q audit/proofs_m29_facade.py
& $cpu audit/check_m29_facade.py
```

Expected unchanged-production result: **7 FAIL**, six import paths for the
single known #21 defect and one global-style proposal. FAIL count is not bug
count. No xfail/skip converts these expectations into green approval. These
proofs remain versioned in this delivery and its eventual history; no red file
was removed, and no extra archive branch is needed. The PR's immutable input
SHA and JSON provide replay provenance. Earlier exploratory harness failures
(null-device guard, notebook import parser and an input-hash abort during edits)
are development failures, not production findings or mutation kills.

## Execution ledger and limits

Windows/Python3.11.9. Fast interpreter is `../AstroSpectro/.venv/Scripts/python.exe`
(no Torch). Optional CPU interpreter is
`../AstroSpectro-m25/.venv-torch/Scripts/python.exe` (Torch2.10.0+cpu).
NumPy2.4.6, pandas3.1.1, sklearn1.9.1, Matplotlib3.11.2, Plotly7.1.0,
pytest8.4.2, setuptools80.9.0, pip25.3. Optional versions are live-recorded in JSON.
No primary venv install or dependency change was performed.

| Command | Result |
|---|---|
| Fast `python -m pytest -q tests/test_dimred_facade.py` | 25 PASS; 12 slow deselected |
| Fast same with `--import-mode=importlib` | 25 PASS; 12 deselected |
| CPU `python -m pytest -q -m slow tests/test_dimred_facade.py` | 12 PASS; 25 deselected |
| Fast explicit `-m slow` same file | 12 SKIP (Torch absent), 25 deselected; not PASS |
| Fast `python -m pytest -q` | 723 PASS; 74 deselected; 7 pre-existing Gaia warnings |
| Fast `python -m pytest -q -m 'integration and not slow'` | 99 PASS; 698 deselected |
| CPU explicit red proof file | 7 FAIL; expected unchanged defects, no ERROR/SKIP/XFAIL |
| CPU experiment checker | 10 KILLED; 6 alternatives admitted (14 executions); wheel build/install PASS |
| Black `--check src tests scripts app.py` plus two M29 audit scripts | PASS (47 files) |
| Ruff same paths; pip check (fast and CPU); git diff --check | PASS |
| Protected production/config/notebook paths against base | Empty diff |

Hosted results belong to the exact PR head and are recorded after publication
in the PR/#43; a local result is never described as hosted CI. The official CI
excludes `slow`, so it does not execute the real Torch facade tests or red proofs.

| Contract family | Status | Practical limit |
|---|---|---|
| Export inventory/identity/star/metadata | AP U/I | Snapshot contract, not a numerical re-audit |
| Fresh imports, repetition, blocked optional roots, known failure attribution | AP I | Tested Windows/Python3.11/CPU dependency set; real extras-complete environment NA |
| Packaging wheel and top-level resolution | AP I for tested wheel | Dependency resolution, independent clean editable env, other Python/OS NA locally |
| Consumer compatibility | PA | Static15 statements + real object/submodule imports; no notebook run |
| Network/destructive I/O/GPU guards | AP S for audited Python events/CPU process | Not C-extension syscall tracing, arbitrary filesystem operation or GPU-build validation |
| Global plotting style | AP observation, desired contract RED | Future explicit-style policy and implementation await review |
| Numerical/physical contracts | No new P tests justified | Producer identity links to M2.1–8; no new algorithm/astrophysical validation |
| Science, provenance, held-out data and historical runs | NA | No catalogs, models, GPU, W&B, real scientific evaluation or leakage proof |
| Torch-free public facade | B by known #21 | Source isolation is not a workaround certified as public API |

For Claude: challenge blocker effectiveness and fresh-process preconditions,
synthetic versus actual identity evidence, dotted versus isolated imports,
proof admissibility under alternate lazy/packaging implementations, global-state
guards and their limits, and separation of #21 from the P2 style proposal.
M2.9 remains pending independent review and human merge decision; M2 is not
declared integrated, #43 stays open, no M3 work follows this delivery.
