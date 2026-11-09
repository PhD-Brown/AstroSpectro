# M2.5 — SpectralAutoencoder: numerical contracts and scientific limits

## Revision after Claude's first review R1–R8

Read [review5820620989](https://github.com/PhD-Brown/AstroSpectro/pull/87#pullrequestreview-5820620989),
all eight full inline threads, #43 and Claude's handoff6729881618, and live
AGENTS.md. Local/remote start ae1e2be, clean; main795af2c unchanged. Same M2.5
worktree/branch/Draft PR87. No production, dependency, CI, notebook or data edit.
This section supersedes the initial evidence claims where noted below.

| Review / checkpoint | Corrected contract and independent expectation | Evidence and remaining limits |
|---|---|---|
| R1 / A:046e78d | Actual hidden encoder: fixed affine weights and BN statistics give pre-activations[-3.5,2.25]/[-1.5,.25], ReLU outputs[0,2.25]/[0,.25], latent−2/0. Repeated eval inference disables dropout. | CPU30PASS at A; C03 Identity and C04 ReLU-before-BN fail the numerical assertion. AP this hidden encoder example; decoder hidden activation is not independently valued by this oracle. Corrected inaccurate finite-difference comment: final-bias gradient is analytic. |
| R2 / B:0374bf3 | Validation5,6,4,7,8,9 with patience2 stops after5epochs and restores weight3; improvement at3 must reset patience. Real registered buffer seen_batches increments once per train forward;5,2,3,4 with patience0 restores weight2 AND buffer2. | CPU32PASS at B; C02 and C01b (named parameters plus strict=False) fail assertions. AP reset and complete controlled state restoration. Real BN buffers remain separately covered in inference/save-load; no claim that bufferless old tests proved restoration. Tie policy remains OPEN. |
| R3 / C:d944ca5 | At each actual Adam.step, global gradient norm≤1+1e−5. At least one pre-clip norm>1.001; actual parameters update. Identical seeded reference training without spies produces exactly the same histories and full state. | CPU33PASS at C; C05 step-before-clip fails the step-time norm assertion. AP order/consumed norm for the chosen real network; first-step Adam formula alone does not prove clipping. |
| R4 / C:d944ca5 | Ranked distances0..99, labels2 at0..14/41..99 and7 at15..40: votes k20=15:5 for2; k50=24:26 for7; k60=34:26 for2. Paired input permutation preserves the winner. | G31PASS at C; C06/C07 fail vote assertion. AP discriminates20/50/60; no promise of distinguishing k49/51 when that requires an undecided tie policy. |
| R5 / C:d944ca5 | STAR rows Teff4500/logg3.0 and5.5 with remote latent coordinates are excluded. Their temperatures lie inside the K window, so Teff cannot mask a gravity-bound error. | C08 inclusive gravity fails all three population cases. AP these half-open numerical boundaries; no universal astrophysical threshold validated. |
| R6 /9a7dab4 | Validation rows compared as paired sorted rows, not RNG return order; floating output dtype, not mandatory float32; actual population samples observed and independently summed; real plateau test discovers configured patience then checks finite, nonincreasing LR and an actual reduction. | A01/A02/A03/A09 all PASS, while counts/disjointness/coverage, output values, unique200-row in-population subsets, repeatability and barycentres remain checked. No historical artifact was found requiring one RNG consumption order or patience10; these internal configurations are not frozen by green tests. |
| R7 | Current mutation runner distinguishes assertion failures, runtime exceptions, errors/collection, expected-refusal tests and skips; compound patches require all anchors. Start/end HEAD, dirty state, effective historical test hashes, versions and actual commands recorded. | Original JSON remains unchanged and historical. New results use m25_review_mutation_results.json. Eight former survivors and four rejected alternatives are replayed against old tests using --tests-ref, without changing the checkout. Final clean-input replay is recorded below after execution. |
| R8 | Historical archive59eb774 remains immutable. New checker reconstructs its files from Git into temporary copies and executes original proofs plus stronger observations. | Six limit/remedy checks PASS in development: drop-all gives zero optimizer steps/NaN train loss but passes old proof; model-only invalidation leaves stale metadata but passes old proof. Safe batch/refusal and full preservation/invalidation satisfy stronger criteria. Silent imputation passes #82's old oracle; scientific policy OPEN. |

### Corrected proof status and contracts

- AP for the added hidden-encoder numerical example, late patience reset,
  controlled parameters AND registered-buffer restoration, consumed clipped
  gradients,20/50/60neighbor discrimination, and the two gravity upper edges.
- PA for untested architectures/devices, independent hidden-decoder nonlinear
  values, physical interpretation, tie policies, general invalid inputs and
  future remediation completeness. Initial AP language below was too broad
  for the old nonlinear/early-stopping/clipping/boundary tests; R1–R5 provide
  the newly stated limited evidence, not a retrospective upgrade of old runs.
- A01: same validation population/order-free mean. A02: values/shapes/floating
  dtype. A03: sample membership/count/uniqueness/repeatability and fsum mean of
  actually selected rows; shared RNG42 and independent RNG42 streams accepted.
  These observations instrument the NumPy sampling boundary; they do not
  establish a general guarantee for arbitrary alternative sampling APIs.
- A09: the scheduler must observe validation loss and function on a constant
  plateau; internal patience is discovered at runtime. The old12×.01 then.005
  trajectory was implementation characterization, not a scientific invariant.
  No verified historical consumer requires patience10. Modes/factors/rate floors
  are checked for a meaningful minimization scheduler, not fixed numerically.
- Real Torch fixture restores RNG, thread count, deterministic mode AND its
  warn_only flag. Agg remains a deliberate process-wide headless backend,
  as before; no interactive notebook is executed in the pytest process.

### R8: limits of the immutable red oracles

[Archive59eb774](https://github.com/PhD-Brown/AstroSpectro/tree/59eb774eb618cdf639cde3c0358da319ee4bcffa)
and its11FAIL/14PASS ledger are unchanged. A PASS under a candidate remedy is
not sufficient evidence that every scientific/state criterion has been met.

| Issue | Counter-control reproduced in temporary archive copies | Strengthened acceptance for a future corrective PR |
|---|---|---|
| #80 | drop_last=True makes both old cases PASS; N2 leaves one train row, zero optimizer steps and NaN train_loss. Old oracle only requires finite validation loss. | A successful fit must have finite histories and at least one actual optimization step; do not silently drop the entire train. Alternatively refuse the impossible batch explicitly before training. Strategy/drop_last policy remains OPEN. The checker also accepts a guarded partial-batch policy/refusal. |
| #81 | Invalidating only _model makes the old proof PASS while history_/fit_time_ remain from the old model. | Retain the entire last completed state (all weights/buffers/history/time) or invalidate model and corresponding metadata completely. Partial invalidation is inadequate. Checker accepts both complete alternatives. |
| #82 | np.nan_to_num before fit makes both old NaN/Inf cases PASS. | This demonstrates only numeric success under imputation. It does not authorize silent imputation or establish astrophysical validity; reject/explicit treatment/provenance policy remains for the scientific owner. |

`audit/check_m25_review_proof_limits.py` contains only temporary-copy controls,
not a production fix or new default red suite. Its results record archive file
hashes, checker hash, actual local HEAD/dirty state, versions and per-case
stdout. No historical file/ref is rewritten; no additional red archive needed.
Qualifications are published on #80/#81/#82 and linked in the final handoff.

### Saving checkpoints and next action

Verified remote checkpoints: A046e78d, B0374bf3, Cd944ca5, R69a7dab4.
Each followed targeted passing tests, explicit staging, ordinary commit/push
and remote SHA verification. Code/tests and intermediate decisions were saved
before the next block. R7/R8 code/doc checkpoint follows targeted validation;
remaining work: one final full mutation replay and suites at committed inputs,
hosted CI at the final head, eight inline replies left open, and one #43 handoff.

## Initial delivery and historical execution ledger

2026-11-08 · [Refs #43](https://github.com/PhD-Brown/AstroSpectro/issues/43).
Audit/tests only; independent Claude review and human merge decision pending.

## Reference, workspace and evidence boundaries

Fetched `origin/main@795af2c8fbe61432dda44480013d86f34ed43e02`, verified
merged PR #79 and parents `58180392d16e06ad411073bc48011d076f9128f0` and
`6e1df68e4275d9d52d9aee8853c4925ccbdddddc`. Source read in full: 1,233 lines,
Git blob **b06bfa75a2e0a8b132d0a0fb8e0a899b6afed0ad**, unchanged in this PR.
New worktree `C:/Users/alexb/Documents/Projets/AstroSpectro-m25`, branch
`audit/43-dimred-autoencoder`; original clean M2.3 checkout at `59ba28d` and
M2.4 worktree preserved. No production/dependency/CI/notebook/data/model edit.
No historical checkpoint loaded, real catalog used, W&B call or GPU training.

Read live AGENTS.md v1.3, #43/dashboard/all comments, #21/all comments, merged
PR #79 and its second review 5820282096, and M21–M24 matrices/conclusions.
Reviewed the all-state issue inventory before drafting new corrective issues.
Historical Draft statements in older matrices are not current merge status.
No previous dedicated AE tests existed. Previous tests bypassed the defective
facade for unrelated dimred modules; they did not validate an autoencoder.

G = `tests/test_dimred_autoencoder.py` (31 cases, no installed Torch needed).
R = `tests/test_dimred_autoencoder_real.py` (29 optional CPU cases).
A = separate red/observation archive, linked in the delivery ledger below.
G executes the complete source with **inert import-only Torch names** to make
the NumPy methods and plotting helpers reachable despite #21. It cannot build,
train or validate a neural network. The names are restored immediately after
each source import. Actual absent-Torch imports run in fresh `python -I`
subprocesses with a meta-path blocker; real Torch tests use the installed
library. Neither mechanism is presented as a substitute for the other.

AP = evidence acquired for the specific stated contract; PA = partial/open;
NA = not validated; B = blocked. U/I/P/S follow AGENTS.md; P here means
numerical/geometrical tests, **not physical spectral validation**. No coverage
percentage measured; green results are not global astrophysical validation.

## Complete definition and contract inventory

All module definitions, including nested helpers, appear below. Inputs are
positional float matrices unless stated; no transport of observation identities,
feature names, physical units or scaler provenance is built into the AE API.

| Definition / source lines | Inputs, preconditions, state, outputs and side effects | Independent evidence / level / status | Residual risks and missing validation |
|---|---|---|---|
| Module imports 55–71 | numpy/pandas/logging/time; Torch imports guarded, but class bases evaluated eagerly. Matplotlib imported only inside diagnostics. Missing Torch logs a warning. | G independent loader/PCA source imports PASS U; R fresh actual import/module+facade PASS I; A absent-Torch module/package/PCA/invocation fail with NameError at87. PA / #21. | `_check_torch` cannot be reached by normal import without Torch. Other facade exports blocked by same mechanism. No facade changes (M2.9). |
| `_check_torch`74–80 | Raises ImportError with install instruction when availability false; no return payload. | G flag-off helper and constructor refusals PASS U, AP reachable guard only. | G's inert import bypass does not repair or validate optional packaging. |
| `_Encoder.__init__`90–112 | D, hidden list, L, dropout; Linear→BatchNorm→ReLU→Dropout for hidden widths, terminal linear. Allocates trainable parameters and BN buffers. | R analytic path parameter counts and geometry for D1/L1/overcomplete/custom/empty internal list; gradients finite PASS U/P, AP examples. | Wrapper discards empty hidden list; invalid negative widths/dropout rejected when built. Zero widths, all numeric dtypes/devices not exhaustive. No scientific ban on overcomplete L>D. |
| `_Encoder.forward`114–116 | Tensor to latent tensor; train/eval behavior inherited from modules. | R manually weighted asymmetric map and finite gradients PASS U/P, AP. | Public 2D shape validation absent; backend may accept higher ranks for linear-only networks. |
| `_Decoder.__init__`122–143 | L, reversed hidden widths, D; same hidden blocks, terminal linear. | R independent dimension/parameter accounting, negative outputs in controlled oracle PASS U/P, AP. | No output positivity or physical-spectrum admissibility promised. |
| `_Decoder.forward`145–147 | Latent tensor→features, with autograd when enabled. | R manual affine reconstruction and final-bias analytic gradient=4 PASS U/P, AP. | No physical units or continuum/line constraints. |
| `_AutoencoderNet.__init__`153–165 | Builds both paths; decoder hidden order reversed. | R custom hidden[4,2], D1, L>D, empty internal list PASS U/P, AP sampled architectures. | Exact internal layer sequence audited as declared implementation; equivalent forwarding accepted in mutation control. |
| `_AutoencoderNet.forward`167–171 | Returns `(reconstruction, latent)` in row order. | R actual Torch backward/shape tests and controlled inference PASS U/P, AP. | No reconstruction accuracy guarantee from output shape. |
| `_AutoencoderNet.encode/decode`173–177 | Delegates to encoder/decoder tensors without public NumPy conversion. | R controlled weights and public methods exercise both PASS U/P, AP. | Training-mode singleton BN fails; device/dtype delegated to Torch. |
| `SpectralAutoencoder.__init__`216–249 | Stores D,L, hidden/default256/128/64, dropout; selects CPU/CUDA; `_model/history_/fit_time_` None. Calls Torch guard. | R explicit CPU, architecture and roundtrip PASS U/I; G before-fit guard PASS U. PA public configuration. | `hidden_dims=[]` silently becomes default; dimensions/dropout validation delayed until fit. CUDA autodetection path/GPU NA. Repr is informational, not provenance. |
| `fit`255–454: build/split | Replaces `_model` before validation; n_val=max(1,floor(N*f)); local NumPy RNG42; distinct val indices, sorted train complement; float32 conversion/device copy. | R `split_disjoint_complete_fixed_and_loader_targets` three fractions, two external seeds, exact validation rows+coverage+target equality PASS I/S, AP row split. | No object/group IDs or scaler provided. A duplicate-object example crosses boundary; no claim of independent object evaluation. N0/N1/f1/f2/negative fraction observed, policy PA. At least1 validation even f≤0. |
| `fit`: DataLoader | Targets equal inputs, shuffle=True, drop_last=False; batch_size forwarded. | R real loader, repeated seeded training/dropout, batch>N PASS I/P. A final singleton and one-row training fail in actual BN. PA small-batch domain. | No universal drop_last policy selected. Batch≤0 rejected by backend; batching/validation may partially replace state. |
| `fit`: Adam/MSE/clipping | Real Adam with lr/weight_decay; zero gradients each batch; MSELoss; clipping norm1; train log averages batch losses equally. | R actual first Adam step from p0,target2 gives p≈.1; train4,val3.61; gradient-reset/clip/decay spies wrap real Torch PASS I/P. AP optimizer steps; PA epoch reporting. | A unequal batches yield236.708328 vs observation MSE228 at lr0. Equal-batch training tests do not hide this. No epoch trajectory or accuracy threshold imposed. |
| `fit`: scheduler | Optional ReduceLROnPlateau(min,.5,patience10,min_lr1e-6), step after logging used LR. | R real constant-loss plateau14 epochs: first12 LR.01, last2 .005; controlled scheduler `.1,.05,.025`; disabled constructor forbidden PASS I/P, AP. | Scheduler version pinned by environment, no cross-version trajectory claim. |
| `fit`: early stopping/snapshot | Strict validation improvement; cloned full state dict incl buffers; patience reset/increment;0 disables stopping; restores best state. | R controlled validation5,2,3,4,6,7 and independently changing weights; patience1/2/0, restored weight2 vs final3/4/6 PASS U/I/P, AP. | Controlled losses validate control flow; actual small fit/roundtrip separately exercise real layers/Adam. No astronomical convergence claim. |
| `fit`: validation/history/logging | Eval mode and no_grad; records train/val/LR; returns history and stores elapsed fit_time. Optional W&B log after early-stop check. | R mode hooks, LR calls, histories, repeated successful refit PASS I/P, AP normal completion. | A NaN/Inf returns success with NaN losses. A failed second fit exposes new random model+old history. Zero/negative epochs produce empty history and random usable model (observation, open domain). W&B early-stop last epoch logging and interruption not executed with real W&B; policy PA. |
| `encode`457–475 | Requires `_model`; eval/no_grad; float32 input→device→latent NumPy CPU. Changes model mode, no input mutation. | R manually [−.5,−7.5,11.5] with unequal weights; hooks see eval/no_grad/no_requires_grad; row/input order and repeatability PASS I/P, AP. | Model non-None is insufficient after failed/zero fit. Oversized batches all transferred at once. GPU/float16/complex inputs NA. |
| `decode`477–495 | Same guard/mode/conversion; latent N,L→N,D NumPy. | R exact affine values including negatives; wrong widths refused PASS I/P, AP. | No physical unit or spectrum constraint; empty/nonfinite policy PA. |
| `reconstruct`497–501 | Composition decode(encode(X)); two conversions and mode changes. | R hand-calculated outputs plus composition and real BN/dropout stability PASS I/P, AP. | Composition identity alone does not establish useful representation. |
| `reconstruction_mse`506–530 | Rowwise feature-mean squared residual; per_sample vector or mean float. | G residuals [3,3,49/3], exact reconstruction, asymmetric features and original inputs PASS P, AP. | A one dominant row distinguishes mean/median. Empty/nonfinite/malformed inputs not guaranteed sanitized. Squared standardized features have no universal physical interpretation. |
| `reconstruction_summary`532–582 | Optional y grouped by sorted unique values; n, mean, std ddof0, median, linear percentile95; no y gives `all`. | G squared residuals0,1,4,16:21/4, sqrt651/4,2.5,14.2; class2/9 independent moments PASS U/P, AP finite positional labels. | Absent classes omitted. A NaN label/empty summary warn then IndexError; wrong-length y raises indexing error. Missing/mixed types and useful diagnostic policy PA. No identity-aware alignment. |
| `compare_with_pca`584–637 | Uses AE MSE and fitted PCA reconstruction_error on same X; skips n>K, preserves requested order, default1/2/3/5/10/20/50; output method/count/mse. | G actual PCA paired orthogonal axes gives14/9,0,5/9,1/9; independent forwarding and mean-vs-median stub PASS I/P, AP valid dimensions. | No refit, preprocessing or evaluation split enforced. n≤0/unfitted PCA delegated; scientific comparability PA. AE need not outperform PCA; representations' dimensions stated per row. |
| `latent_interpolation`643–684 | Accepts vector or1-row endpoints; encodes first row; linspace0..1; decodes trajectory; returns Z,X. | G independent affine endpoints[1,5]/[−9,7], steps1/2/5 incl middle/order/decoded values PASS P, AP geometry. | A steps0/−1 and multirow truncation observed. Identical/nonfinite endpoints policy PA. Continuous latent line is not evidence of stellar evolution/physical validity; docstring overclaim tracked separately. |
| `save`690–712 | Requires model; torch.save weights, D,L,hidden,dropout,history,time to caller path. Disk side effect only tmp_path in tests. | R actual trained tiny model and all weights/buffers roundtrip PASS I, AP. | No feature schema/scaler/data hashes/split/seed/optimizer state persisted; cannot resume exact optimizer training or prove feature identity. Atomic-write/permission failures NA. |
| `load`715–752 | Torch required; CPU map, `weights_only=False`; reconstructs architecture, strict weights, eval; optional history/time; chosen device. | R own local trusted payloads: missing/corrupt/incomplete/shape/weights refusal; optional+extra metadata; exact outputs/state PASS I, AP CPU. | Static trust-boundary risk: unrestricted pickle; only locally created payloads loaded. No malicious payload/exploit test. No universal safe-load claim. GPU transfer/other versions NA. |
| `_check_fitted`758–763 | RuntimeError if `_model is None`. | G encode/decode/reconstruct/save before model PASS U, AP initial guard. | A failed fit can pass guard; preserve or invalidate state remains open. |
| `__repr__`765–768 | Displays D→hidden→L; no training/scientific guarantee. | Read/statical inventory PA; no brittle formatting test. | Does not attest actual parameters or fit success. |
| `tester_candidat`774–1013 | Cast/atleast2d raw row; reference scaler or raw-copy fallback; encode/reconstruct/MSE; nearest50 Euclidean neighbors; vote excludes−1; fixed heuristic thresholds. Creates3panel figure, show(), optional save. Returns z,mse,cluster,verdict,scaled/recon. | G actual StandardScaler and controlled AE boundary, nontrivial residual MSE5, declared feature-bar order, no scaler, noise-only, noise-majority, noncontiguous labels, <50 and scrambled120row nearest-neighbor oracle PASS U/I/P, AP calculations. | AE stub does not validate neural inference (R does separately). No raw feature identity/order detection beyond caller precondition. Equal-size permutations undetectable; ties not scientifically defined. Metadata/y arguments currently unused; teff optional plotting. Teff finite mask and actual tmp_path PNG export tested; plt.show suppressed/Agg. Glyph/layout/pixel quality unvalidated. Fixed.97/3 are not measured quantiles of arbitrary datasets. |
| `latent_arithmetic`1016–1233 | Reads Gaia Teff/logg columns, STAR finite mask; selects6 windows; returns populations,tests,fig; analogies/Euclidean distances/reference-ratio with denominator floor1e−8; optional save/show. | G 1/23/207 rows per group, asymmetric positions+offsets,2analogies, independent norms, excluded QSO/NaN/outside rows, same-seed resampling, absent populations/columns PASS U/I/P, AP calculations. | Positional Z/meta/y must correspond; no object IDs. <20 only warning, not exclusion. Zero/near-zero reference ratio floor is uncalibrated policy, PA. Latent axis metric/nonidentifiability and calibrated astrophysical interpretation NA. No green verdict/taxonomy threshold asserted. |
| Nested `select_pop`1056–1080 | Half-open Teff/logg windows, max200 without replacement using shared local RNG42; returns mean and total qualifying count; absent→None,0; prints warning/count/2coordinates. | G first-window lower limits, excluded upper/outside, sampling cap207→200 and means, deterministic repeat PASS U/P, AP selected examples. | Total found count vs sampled200 not returned in public populations; numeric count/print formatting not frozen. Nullable/misaligned meta/y, nonfinite Z,1D latent visualization PA. |

## Mathematical oracles and tolerances

For N×D residuals, row MSE is sum of D squares divided by D; global MSE is
sum of N row MSEs divided by N. G uses different errors per row and unequal
features. Summary std is population ddof0, percentile is NumPy's default
linear rank `(n−1)*.95`; both are declared numerical conventions, not inferred
uncertainties of a physical population. Float64 analytic comparisons use
absolute tolerances1e−14/1e−15 for small arithmetic; Torch's hand-set binary
weights/inputs yield exact float32 values. Adam first-step tolerance2e−7 and
MSE1e−6 allow float32 rounding, not a trained-loss target.

Parameter count per affine path is the sum of `(width_in * width_out + width_out)`
over consecutive widths d0..dk, plus
2Σ(hidden widths) for trainable BN scale/bias. BN running mean/variance and
num_batches_tracked are buffers, verified in the checkpoint roundtrip but
excluded from trainable counts. G's geometry/scientific helpers use controlled
coordinates with independent manual vector calculations; no learned latent
axis is assigned a physical meaning.

## Split, preprocessing, identity and determinism

The actual consumer notebook `notebooks/dimred/phy3500_03_autoencoder.ipynb`
was read as JSON, never executed or modified. Cell5 calls loader.load(scale=True)
before cell9 fits AE. `data_loader.py:321–322` fits StandardScaler on the
entire returned population, then AE makes its internal validation split.
This is a **demonstrated methodological exposure of internal validation to
preprocessing fitted on that population**, tracked by existing #24; it is not
proof of an independent final test-set leak in every caller of `fit`.
There is no scaler fit inside AE and no API proving external preprocessing
provenance. G/R use synthetic matrices only. A repeated-object example shows
row splitting can share object identities across training/validation; existing
#22 covers grouping policy. No GroupShuffleSplit contract imposed here.

`__init__.py` eagerly imports AE; `run_reporter.py:1259–1386` consumes encode,
history, fit_time, reconstruction_mse and persists derived artifacts. Reporter
also reads two latent axes and may suppress errors; full reporter audit is
outside M2.5. `PCAAnalyzer` reconstructed API and M2.2 limits were consulted.
Notebook cell9 loads z2 checkpoint without the z3 cell11's dimension check;
neither checkpoint stores ordered feature names/scaler, so equal D alone is
not provenance. Candidate caller passes raw values using loader.inverse_transform
and feature_names in current order. Actual historical consistency unverified.

Only split/sample NumPy RNG42 is fixed in production. Network initialization,
DataLoader shuffle and dropout consume Torch RNG; production does not set it.
R explicitly sets CPU threads1, manual seed1729 (restores RNG/threads afterwards),
and deterministic algorithms. Repeated training with external seed97 reproduces
history and outputs; seed98 differs while row split remains fixed. This is
environment-specific CPU reproducibility, no cross-version/GPU equality claim.
See [PyTorch2.10 reproducibility](https://docs.pytorch.org/docs/2.10/notes/randomness.html).

## Checkpoint trust and scientific interpretation

`load` explicitly requests `weights_only=False` at line732. The
[PyTorch2.10 documentation](https://docs.pytorch.org/docs/2.10/generated/torch.load.html)
describes the unpickling/trust restriction. The audit never loads unknown,
downloaded or historical models. All malformed checkpoints are generated
locally inside tmp_path from the current synthetic network; no malicious code
or exploit is constructed. Restricting serialization and migrating NumPy
history scalars is a separate compatibility/security design decision.

Candidate thresholds.97 and3 and the latent analogy30%/60% labels lack
population calibration in these functions. Quantiles cannot be inferred from
one candidate MSE, and stellar evolution cannot be inferred from a latent line
or vector analogy. G tests distances, means and votes only; **no green test
enforces the current astrophysical verdict, its thresholds or universal
performance superiority to PCA**. Correct cautious prose remains admissible.

## Delivery ledger

Initial execution inputs: snapshot `47c56c1` (preliminary0328a99 preserved).
Final green-suite/mutation inputs after two export tests: snapshot `1eb0c0b`.
The mutation JSON records actual input hashes, source blob, HEAD and working
tree status. Preflight harness failures (missing copied loader), two interrupted
pilot runs and the initial plotting backend error are development checks,
not mutation kills or production defects. Backend was explicitly fixed to Agg
in the new plotting fixture; production unchanged.

### New and existing issue routing

| Classification | Trace | Scope / policy left open |
|---|---|---|
| Known demonstrated import defect | [#21](https://github.com/PhD-Brown/AstroSpectro/issues/21) | Four fresh-process failures; no duplicate issue. |
| New demonstrated small-batch defect | [#80](https://github.com/PhD-Brown/AstroSpectro/issues/80) | BatchNorm receives singleton; batching or explicit domain refusal not chosen. |
| New demonstrated state inconsistency | [#81](https://github.com/PhD-Brown/AstroSpectro/issues/81) | Failed refit exposes random network and stale metadata; retention/invalidation both accepted. |
| New demonstrated nonfinite success | [#82](https://github.com/PhD-Brown/AstroSpectro/issues/82) | NaN/Inf training completes with NaN history; no imputation policy selected. |
| New demonstrated configuration substitution | [#83](https://github.com/PhD-Brown/AstroSpectro/issues/83) | Explicit [] replaced by default; honoring or rejecting empty configuration acceptable. |
| Demonstrated reporting difference, open metric contract | [#84](https://github.com/PhD-Brown/AstroSpectro/issues/84) | Batch-mean236.708328 versus observation-MSE228; public API does not explicitly promise observation weighting. Red case is a policy probe, not proof of an optimizer bug. |
| Static/documentary scientific overclaims | [#85](https://github.com/PhD-Brown/AstroSpectro/issues/85) | Uncalibrated quantile and physical-latent claims; no empirical astrophysical invalidity measurement. |
| Static checkpoint trust risk | [#86](https://github.com/PhD-Brown/AstroSpectro/issues/86) | Unrestricted unpickling; no exploit/incident asserted. |
| Existing methodological concerns | [#24](https://github.com/PhD-Brown/AstroSpectro/issues/24), [#22](https://github.com/PhD-Brown/AstroSpectro/issues/22), [#20](https://github.com/PhD-Brown/AstroSpectro/issues/20) | Notebook scaler exposure, repeated-object split, incomplete run provenance; no duplicate or production fix. |

All new issues use the five AGENTS.md sections, immutable source/proof links,
severity labels and explicit scope. #80–#83 are behavioral defects; #84–#86
are reporting/documentation/trust tasks, not three additional proven runtime
bugs. Source unchanged; no fixes included.

### Durable red archive

Branch **audit/43-dimred-autoencoder-proofs** pushed at immutable SHA
**59eb774eb618cdf639cde3c0358da319ee4bcffa** and read back through GitHub
before removing owned red files from integration:

- [Manifest](https://github.com/PhD-Brown/AstroSpectro/blob/59eb774eb618cdf639cde3c0358da319ee4bcffa/audit/M25_PROOFS.md).
- [Executable cases](https://github.com/PhD-Brown/AstroSpectro/blob/59eb774eb618cdf639cde3c0358da319ee4bcffa/tests/proofs_dimred_autoencoder.py).
- [Per-case results/stdout](https://github.com/PhD-Brown/AstroSpectro/blob/59eb774eb618cdf639cde3c0358da319ee4bcffa/audit/m25_proof_results.json),
  [JUnit](https://github.com/PhD-Brown/AstroSpectro/blob/59eb774eb618cdf639cde3c0358da319ee4bcffa/audit/m25_proofs.xml).
- [Oracle controls](https://github.com/PhD-Brown/AstroSpectro/blob/59eb774eb618cdf639cde3c0358da319ee4bcffa/audit/check_m25_proof_controls.py),
  [control results](https://github.com/PhD-Brown/AstroSpectro/blob/59eb774eb618cdf639cde3c0358da319ee4bcffa/audit/m25_proof_control_results.json).

**11FAIL/14PASS/0ERROR/0SKIP/0XFAIL**, exit1,12 warnings on the intentionally
NaN-label/empty-summary observations. Replayed with identical outcomes.
Four failures are #21, two #80, one #81, two #82, one #83, one open #84 probe.
PASS observations do not bless the current invalid/empty-input policies.
Seven candidate-remedy oracle controls PASS, selected counts1/1/2/2/1/1/4:
preserve state, invalidate state, singleton refusal, finite refusal, empty
hidden configuration, weighted epoch metric, conditional Torch class bases.
Controls run on temporary copies, not production; no claim of approved fixes.
The archive stays an ancestor of integration and its ordinary merge history.

### Mutation campaign and positive controls

Run `python audit/check_m25_mutations.py` with the isolated CPU interpreter.
[Machine-readable results](m25_mutation_results.json) record the final input
snapshot **1eb0c0b** (full SHA in JSON), all input local-byte SHA256
hashes, source Git blob, platform/Torch and actual dirty/untracked status.
CRLF local-byte hashes are not Git blob IDs. Before/after hashes agree.
No input file changed during this completed campaign; the preliminary runs
were interrupted before this snapshot and are not claimed as evidence.

Each trial copied all9dimred sources, conftest, G/R/config/script into a unique
temporary directory and ran **56 tests**. Two independent fresh-process import
tests and two PNG exports are explicitly excluded from replay with `-k` to avoid
redundant import/render work; all four execute in the normal suites. No collection/fixture/import errors,
skips or missing cases are classified as kills.

**33 incorrect candidates KILLED /0 SURVIVED /0 INCONCLUSIVE**. Covered sites:
split overlap/size/seed; validation/inference train mode; lost best snapshot
or aliasing; patience; history values; gradient reset/clipping/weight_decay;
MAE/summed-feature/global-median errors; summary ddof/mean/quantile; latent
width; reversed features/rows; missing checkpoint weights/wrong architecture;
reversed interpolation; arithmetic sign/reference/population; candidate
noise/distance/scaler/neighbor selection; PCA mean versus median.

Baseline and **six** separate controls PASS: equivalent einsum MSE, equivalent
forward delegation, extra checkpoint metadata, cautious interpolation prose,
cautious anomaly verdict, explicit2D shape refusal. Some replacements touch
multiple occurrences in a function/module (counts in JSON): these are semantic
candidates, not33independent source sites. No arbitrary mutation quota or
claim of complete protection, astrophysical correctness or all-policy coverage.

### Executed checks and environment

Fast interpreter `C:/Users/alexb/Documents/Projets/AstroSpectro/.venv/Scripts/python.exe`
has **no Torch installed**. Optional interpreter
`C:/Users/alexb/Documents/Projets/AstroSpectro-m25/.venv-torch/Scripts/python.exe`
is a new independent venv with `requirements/ci.txt` and
`torch==2.10.0+cpu --index-url https://download.pytorch.org/whl/cpu` (113.6MB
CPU wheel, no CUDA). No lock/dependency/CI edits. Both pip checks pass.
Python3.11.9, Windows10build26300 AMD64, NumPy2.4.6, pandas3.1.1,
sklearn1.9.1, pytest8.4.2, Black24.4.2, Ruff0.5.0. Full optional fixture
restores RNG/CPU-thread/determinism state after each test.

| Actual command / interpreter | Result (green inputs1eb0c0b; archived proofs47c56c1) |
|---|---|
| Fast `python -m pytest -q tests/test_dimred_autoencoder.py` |31PASS (normal/reverse order, `-W error`) |
| Fast `python -m pytest -q` |416PASS/58deselected/7existing Gaia warnings |
| Fast `python -m pytest -q --import-mode=importlib` |416PASS/58deselected/7same Gaia warnings |
| Fast `python -m pytest -q -m 'integration and not slow'` |57PASS/417deselected |
| CPU `python -m pytest -q -m slow tests/test_dimred_autoencoder_real.py` |29PASS; normal, importlib and reverse order |
| Fast explicit optional CPU file (`-m slow`, no Torch installed) |29SKIP,0FAIL; these skips are not real-network validation |
| CPU mutation replay |33KILLED,6positive controls+baseline PASS;56tests per trial; PYTHONHASHSEED7 |
| CPU explicit archived proofs |11FAIL/14PASS as classified above,12warnings; identical replay |
| CPU archived oracle controls |7PASS |
| `python -m black --check src tests scripts app.py` plus both M25audit scripts |PASS (38files while archive present) |
| `python -m ruff check src tests scripts app.py` plus both M25audit scripts |PASS |
| `python -m pip check`, both interpreters; `git diff --check` |PASS |

Earlier exploratory full optional-environment importlib run412PASS used27G
cases before the noise-majority parametrization;414PASS used29G before export
tests. These are historical, not the final416count.
The seven fast-suite warnings are pre-existing `gaia_crossmatcher.py:1210`
Pandas4Warning. Optional M2.5 CPU tests are explicitly `slow`+`integration`,
so the unchanged official CI **does not execute them**. Without Torch they
skip when explicitly selected, rather than impersonating real execution.

First hosted [CI42540318386](https://github.com/PhD-Brown/AstroSpectro/actions/runs/42540318386)
completed successfully at headba9f6d6 (control merge1e6222a), before the two PNG
tests: Linux/Python3.11.18,411PASS/3WindowsSKIP/58deselected/7Gaia warnings;
pip check, Black35files and Ruff PASS. Job logs were read. This is historical
CI for that head, not final-head confirmation.

Final integration head, [Draft PR87](https://github.com/PhD-Brown/AstroSpectro/pull/87),
hosted CI run/SHA and handoff are recorded in PR87 and #43 after the last push;
this file does not fabricate a
self-referential commit hash or a hosted result before that run exists.

### Residual limits and review questions

No real astrophysical evaluation, final held-out catalog, GPU, full notebook
execution, cross-version/device checkpoint, filesystem interruption/permission,
large-batch memory measurement, visual-layout QA, line coverage or real W&B.
Both diagnostic exports write actual synthetic PNGs in tmp_path and check
the PNG signature. Candidate Teff scatter checks finite-only coordinates and
color values. Only missing-font-glyph warnings are filtered during that export;
pixel aesthetics/font completeness/layout remain outside the numeric checks.
Further input-policy cases remain PA as marked in the inventory. No residual
is silently converted into a green requirement or claimed covered by a mock.

For Claude: challenge the red proof preconditions and classifications,
especially #84's open metric; check best-state restoration beyond epoch counts,
absence of fixture pollution under importlib/reverse order, independent MSE/
quantile/PCA/vote/analogy oracles, admissibility controls, and Torch-free CI
versus actual CPU evidence. M2.5 remains 🟡 until review and human decision;
no merge, Ready-for-review transition or M2.6 work.
