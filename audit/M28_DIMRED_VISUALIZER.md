# M2.8 — DimRedVisualizer scientific figure audit

2026-11-10 (Quebec) · Refs #43 · tests, evidence and documentation only.
Worktree `AstroSpectro-m28`, branch `audit/43-dimred-visualizer`.
Verified remote main/base **39dad874610745ccd456d26972e90f6fb3a11405**,
PR99 merged, no open PR at start. Main checkout was clean at1e22663 and is
preserved. Production blob **624b857bca5833c3401204b0a592b9ae6ffc2902**.
Actual inventory: **38 methods**, including initializer and two helpers,
not31. Entire3224-line source read. AGENTS v1.3 fetched from current main.

## Revision after Claude review 5858927009 ? 2026-11-13

This revision resumes PR #116 at `4364c61f56d91f7ec93c5c6072b9f88c46023241`;
all first-pass results below remain historical. Production and the original
archive `ed1d91ec0efb619aeb77a5bf7e4a0ab43fa84cac` are unchanged. The current
review manifest is [M28_REVIEW_PROOFS.md](M28_REVIEW_PROOFS.md). The 53 original
green nodes remain; eight adapter controls are added. No production fix is
part of these tests. All twelve GitHub threads remain reviewer-owned.

| Review | Revised contract and scope |
|---|---|
| R1 | Three STAR bimodal anisotropic mixtures (seeds 31/7/83) plus distinct QSO; inspect fitted observations, actual contour labels, vertices and polygon probability. Retain analytic Gaussian and independent expanded quadrature. K1B/K6/K7/K8 and missing-mode K9 exercise different failures. |
| R2a | Analytic law has dataset/d/n/covariance/factor, coherent evaluate/call/resample; K4 sampling and K5 kernel-support quadrature are admissible solvers. A complete display domain accompanies each solver when testing enclosed polygons. |
| R2b | Cluster ID -> actual scatter RGB -> colorbar, categorical Patch/Line legend or colored annotation. A contradictory colorbar is rejected even if other annotations are correct. |
| R3 | HR bounds derive from input K/dex independently; reconstruction needs nonempty paired arrays; histograms require independently counted observations and bar heights; exports match an explicit same-figure rendering. |
| R4/R5 | Class/count/coordinates, synthesis method/geometry, endpoint identity, PC series/value and exported data collections are linked. Exact French prose is not the contract. |
| R6 | Population annotations accept grouped numbers and N= wording; variance annotations link PC count and full-model cumulative variance in either word order. Omission of a questionable claim remains admissible. |
| R7 | Full visibility OR a verifiable disclosure with actual excluded population/count/support or curve boundary/extreme. Generic warnings and false numbers fail. Zero/negative/nonfinite histogram policies remain open. |
| R8 | Capture real sklearn subsets, recover supplied observation rows and check pairing, independent ranks, values and repeatability. A paired permutation sampler is admissible; no particular RNG choice sequence is required. |
| R9 | Inventory corrected to 38/38 below. HR missing-column sentinel AP; provided-data scatter/orientation and dimensional region geometry PA; astrophysical region taxonomy/boundaries NA. |
| R10 | Independently reproduced linear conditional-density clipping: QSO 3/50 retained, area 1. New characterization and a P2 issue proposal only; distinct from #111 log floor. No new issue created. |
| R11/R12 | D04 reassigns original F row 8, not row 5. Corrected issue anchors #103:3215, #104:2588, #105:1415/1560, #110:185/186, #111:2824, #114:2279 at the unchanged base SHA, with dated documentary notes. |
| R13 | Six mh_gspphot -> Fe/H labels (437/616/1100/2311/2455/2892) remain PA interpretation, not a validated abundance conversion. Fixed 3-PC/183-feature/min_size300 titles and anomaly prose also PA; #65/#78 retain policy ownership. |
| R14 | Explicitly lint all four M2.8 audit scripts locally; CI workflow remains unchanged and does not gain audit-script linting implicitly. Historical absolute Windows paths are preserved as execution provenance. |

The KDE mass tolerance .006 is numerical discretization, not astrophysical
uncertainty. The analytic covariance is diag(4,9); its polar integral is
max(0,1-12*pi*threshold). Mixture quadrature has 0.05 spacing on [-12,18] x
[-14,14], total probability within 1e-4 of one. Polygon membership is computed
independently from actual pre-inline-label paths (even/odd rule for disjoint
components and holes), in addition to density-threshold integration. The
geometry interpolation tolerance is 5% on the 200x200 contour mesh. Experiments
showed that merely correcting levels can leave open 99% arcs on the original
percentile domain; current admissible controls also complete that domain.
These bounds and tolerances do not validate all KDE input distributions.

PNG comparison decodes RGBA against the explicit backend export of the same
returned scientific figure, within the same process (one-channel-unit tolerance).
It does not use a version-specific golden image. Blank and axes-only exports
are rejected; PDF scientific rendering and cross-version agreement remain PA.

Current validation: **61 dedicated PASS with -W error**, complete default,
importlib and reversed suites **693 PASS / 62 deselected / 7 preexisting Gaia
warnings**. Dedicated reversed and shuffled orders (3/11/29) each 61 PASS.
Original M01?M16: **16 KILLED**; original C01?C14: **14 PASS** and F01?F03:
**3 KILLED**. Review campaign: **19 KILLED**, **12 admissible controls PASS**,
and **87/87 joint PASS**, with zero surviving/inconclusive current candidates.
Revised red baseline: **25 FAIL / 1 PASS**, 26 cases. This is evidence on bounded
contracts, not 25 bugs or scientific approval. See the new manifest for exact
ledgers, code hashes, archive reference and replay commands. Historical results
below are not attributed to this pass.

## Delivery contract and evidence boundaries

AP: acquired evidence on the specific tested contract; PA: partial examination;
NA: not validated; B: externally blocked. No module-wide scientific approval
is inferred. No external blocker identified. No production/notebook/data/model,
dependency or workflow changes authorized here. First Claude review and human
merge decision follow the Draft PR. M2.9 is outside scope.

The dedicated tests inspect numeric arrays, artist coordinates, signed values,
histogram areas, interval endpoints, colors, labels and actual exports. Synthetic
observation markers are asymmetric, class sizes unequal, physical values unique,
and metadata indices nonconsecutive. Equal-length external permutations remain
undetectable without IDs: this API accepts positional arrays and never joins
on `obsid`. A matching shape is not identity evidence.

The real PCA integration uses six axial points with covariance eigenvalue ratio
9:4:1; discarded residual energy gives MSE5/9 then1/9. PCA sign is arbitrary:
the loading test uses sign equivariance and exact supplied signed values, never
a preferred global sign. Family MSE uses unequal groups with feature MSEs
1,4 versus9,16,25: means2.5 and50/3; SDs are sample dispersions **among feature
means**, not among stars. Global MSE11 weights the five features equally.

Trustworthiness is executed by actual scikit-learn and checked against a
separate scalar rank-penalty implementation (no distance ties). Identity geometry
gives1; a row permutation changes neighborhoods. Subsampling is paired and
repeatable; this cannot establish the external identity of supplied observations.
[Primary definition and domain](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.trustworthiness.html).

KDE uses actual SciPy plus a separately injected analytic Gaussian with known
covariance diag(4,9). Polar integration gives enclosed mass
`max(0,1-12*pi*threshold)`.
An independent larger401x401 quadrature domain includes tails omitted by the
display grid. The601x601 green oracle control verifies known68/95/99% solutions
within.002. Red numerical comparisons allow.006 for contour-cell discretization,
not observational uncertainty. Density percentiles are not mass quantiles.
[SciPy kernel density API](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html).

## Method inventory — input, transformation, output, contract, risk, test gap

All tests below live in `tests/test_dimred_visualizer.py` unless designated
red (`tests/proofs_dimred_visualizer.py`, explicitly executed only in archive).
No dedicated visualizer suite existed at base. Prior PCA integration tested a
correlation heatmap; AE tests inspected interpolation geometry. These were
limited existing boundaries, not previous coverage of every figure.

| Method | Input -> transformation -> output | Contract/test evidence | Status and residual risk |
|---|---|---|---|
| `__init__` | figure options/output path -> mkdir -> object | `save_defaults_override_no_output_and_write_failure` | AP mkdir/options; filesystem ACL exhaustion NA |
| `_save` | figure/path -> parent mkdir/savefig -> None | `real_export_readable_and_same_semantic_figure`, defaults/error | AP real PNG/PDF and write failure; PDF content rendering PA |
| `_draw_loadings_family_donut` | family weights -> pie angles + legend -> None | `family_weights_bar_and_donut_angles` | AP normalized valid weights; zero weights boundary PA |
| `plot_variance_explained` | PCA ratios/count -> percentages/cumsum -> `(fig,axes)` | real covariance oracle, red truncated annotation | AP numeric; misleading truncation confirmed |
| `plot_loadings_heatmap` | loading frame -> abs-max ranking -> `(fig,ax)` | asymmetric signed matrix + sign reversal | AP tested selection/values; excess requested PC labels PA |
| `plot_loadings_bar` | top loading frame -> signed sorting/colors -> `(fig,ax)` | signed values and variance text | AP numeric; PCA sign has no physical orientation guarantee |
| `plot_loadings_family_donut` | two family frames -> two donut helpers -> `(fig,axes)` | angles/legend weights | AP normalized inputs; hardcoded183-feature title PA |
| `plot_loadings_family_bar` | family union -> percentages -> `(fig,ax)` | disjoint family weights80/20 vs30/70 | AP numeric; empty/zero-group policy PA |
| `plot_correlation_heatmap` | correlation frame -> labels/image/text -> `(fig,ax)` | exact matrix/ticks/finite annotations | AP representation; estimator and metallicity interpretation NA |
| `plot_embedding` | Z/class/scalar -> positional scatter -> `(fig,ax)` | classes/plain/continuous+missing, finite scaling | AP paired valid arrays; quantile/clim choice and invalid coordinates boundary PA |
| `plot_embedding_grid` | Z/y/meta -> class+parameter panels -> `(fig,axes)` | coherent permutation/absent column/missing rows | AP nominal; all-missing/Inf scaling PA |
| `plot_stability` | ordered report -> bars/reference-excluded mean -> `(fig,ax)` | actual producer with controlled fit; asymmetric mean | AP producer order0..N-1; arbitrary reordered reports PA; stability not astrophysics |
| `plot_reconstruction_error` | PCA summary -> mean±SD+elbow -> `(fig,ax)` | actual discarded-energy MSE and band | AP statistics; nonuniform count elbow heuristic PA; log zero hidden |
| `plot_eigenspectra` | PCA vectors/wavelength -> line+markers -> `(fig,axes)` | exact wavelength/values/in-range markers,1/3/10 requested | AP custom marker path; default mathtext and fixed3-PC title PA |
| `plot_hr_teff_regions` | supplied Teff(K)/log g(dex) -> scatter + illustrative polygons -> `(fig,ax)` | green missing-column sentinel; D02 input-domain geometry and D03 actual population | AP missing-column sentinel only; PA supplied scatter/orientation and dimensional polygon envelope; NA stellar taxonomy and astrophysical delimitations (including N08) |
| `plot_hr_diagram_embedding` | meta/Z component -> finite triplets -> `(fig,ax)` | missing on distinct rows, two axes, units/inversion | AP provided values; actual stellar stages NA |
| `plot_training_history` | train/val/LR -> curves/best epoch -> `(fig,axes)` | nonlast minimum, distinct series, real explicit export | AP numeric; direct output_dir mismatch red; train-loss semantics #84 |
| `plot_ae_embedding` | 2D Z/y/meta -> class+physical panels -> `(fig,axes)` | coherent row permutation/missing physical row | AP nominal; no-physical scalar axes red |
| `plot_ae_reconstruction` | X/reconstruction -> seeded examples/MSE -> `(fig,axes)` | row values/exact manual MSE,1/3/6 samples | AP valid; excess sample axes red; zero/negative boundary PA |
| `plot_ae_vs_pca` | method/count/MSE frame -> distinct curves -> `(fig,ax)` | AE.17/.11 versus PCA.9/.03 | AP representation; training/comparison provenance NA |
| `plot_latent_interpolation` | latent/features path -> alpha+variable features -> `(fig,axes)` | endpoints/alphas/nontrivial feature ordering | AP geometry only; physical consistency #85 NA |
| `plot_ae_error_distribution` | MSE/y -> normalized class hist/mean/median -> `(fig,ax)` | area1/class, mean!=median, unequal counts | AP<=4classes; fifth class omission red |
| `plot_hdbscan_clusters` | real analyzer attributes/Z/y -> cluster+class scatters -> `(fig,axes)` | nonconsecutive IDs, noise/one cluster, exact colors/rows | AP points; incompatible colorbar red |
| `plot_hdbscan_hr` | Z/analyzer/meta -> finite HR + T overlay -> `(fig,axes)` or `(None,None)` | missing column sentinel, finite pairs/temperature | AP points; colorbar red, empty HR percentile PA |
| `plot_hdbscan_sensitivity` | sweep frame -> two distinct curves -> `(fig,axes)` | exact counts/noise/x and marked parameters | AP supplied sweep; clustering backend NA here |
| `plot_hdbscan_presentation` | presentation labels/map/meta -> paired plots -> `(fig,axes)` | counts/positions with/without HR | AP geometry; fixedmin_size300 and anomaly/population prose PA (#78) |
| `plot_hdbscan_present_split` | same -> separate UMAP/HR -> `(fig_umap,fig_hr)` with HR=None possible | paired rows/colors; missing HR; orientation | AP normal; percentile clipping and genuine populations PA |
| `plot_hdbscan_feature_heatmap` | cluster means/labels -> intercluster variance rank -> `(fig,ax)` | analytic rank b,a,e + counts/values | AP representation; fixedTop30 prose, fixedcolor clipping PA |
| `plot_hdbscan_feature_profiles` | means/labels/n_show -> populous clusters/top5 -> `(fig,axes)` | signs/ranking/counts | AP nominal2;1 and unused panels red, fixedrange clipping PA |
| `plot_xgboost_umap` | Z/pred/confidence/clusters -> three layers -> `(fig,axes)` | exact prediction/scalar/cluster pairing, None | AP nominal; genuineones misdeclared unavailable red; calibration NA |
| `plot_fg_confusion` | Z/pred -> F/G overlays -> `(fig,ax)` or `(None,None)` | <10 sentinel; F3/G8 exact subsets/counts | AP predicted populations; **not** confusion matrix or uncertainty proof |
| `plot_trustworthiness` | X/Z_u/Z_t/k -> real sklearn/seeded subset -> `(fig,ax)` | independent ranks, identity/permutation, admissible k | AP computation; fixed.75 lower bound hides values red; k0/negative PA |
| `plot_latent_zoomed` | Z/y/meta -> percentile window/panels -> `(fig,axes)` | exact paired physical/class arrays | AP nominal; no-physical scalar axes red; tails/NaN boundary PA |
| `plot_latent_kde` | STAR latent+params -> scipy KDE/grid/contours -> `(fig,axes)` | actual STAR/color pairing/reproducibility; analytic+real red masses | AP tested inputs/defect; no-star/small/singular/allmissing boundary PA |
| `plot_ae_recon_by_spectral_type` | X/R/types/features -> type means -> `(fig,axes)` or `(None,None)` | aligned F10/G12 manual means; red deleted middle row | AP nominal; positional truncation defect, external IDs PA |
| `plot_ae_error_by_family` | X/R/names/taxonomy -> feature then family stats -> `(fig,axes)` | means/SD/manual global weighting/errorbar endpoints | AP valid supplied taxonomy; fallback TiO #65 PA |
| `plot_ae_error_distribution_logscale` | MSE/y -> linear/log density hist -> `(fig,axes)` | nominal areas/medians; red tiny positive MSEs | AP nominal; zero/negative/Inf/NaN and truncation PA |
| `plot_synthesis_grid` | distinct PCA/AE/optionalUMAP -> method x scalar panels -> **fig only** | exact per-method/class/physical association | AP nominal; absent/allmissing panels/provenance PA |

## Consumers and prior findings

NB01 passes actual PCA variance/loadings/correlation/reconstruction tables,
including raw-spectral PCA and wavelength grid. NB02 passes ordered stability
reports, HDBSCAN normal/presentation attributes, profile means and bridge arrays.
NB03 passes AE history/latent/reconstructed standardized features, inferred
spectral types, summary/comparison frames and previous PCA/UMAP embeddings.
`run_reporter` inventories figure paths; it does not validate pixels or scientific
ownership (#95/#100). No direct production call from app/tools visualizer was
found. Notebook calls are read statically, never executed as complete workflows.

Known boundaries: #50 joins, #73 profile filtering, #88 bridge filtering each
precede this visualizer; none guarantees identity for an arbitrary caller.
#65 spectral taxonomy also affects fallback family regexes; no duplicate issue
is needed. #78 noise/population interpretation, #84 training loss aggregation,
#85 interpolation physical claims, #89 preprocessing and #90 probability
definition remain open. M2.8 tests do not turn these into approved policy.

## Initial checkpoint

Initial dedicated suite53PASS (Windows/Python3.11.9, matplotlib3.11.2,
numpy2.4.6, pandas3.1.1, scipy1.17.1, sklearn1.9.1), with real Agg rendering.
Final mutation, archive, issue and validation ledgers follow after execution.
Initial test-fixture import ordering and a tied intercluster variance fixture
were corrected before this checkpoint; no production change. Larger KDE
integration domain corrects an early audit-oracle tail omission, not the
production density thresholds. Historical initial JUnit remains under ignored
test-results; final ledgers distinguish effective hashes from active parent HEAD.

NA: astrophysical validation, real catalog/FITS/model/runs, train/test leakage,
calibration, genuine Gaia source identity, actual UMAP/t-SNE/AE refitting here,
GPU/W&B/network, visual publication review, cross-version image/PDF agreement,
all filesystem failures, notebook execution and line coverage. No claim of
historical figure corruption without its actual run artifact.

## First-pass delivery ? historical results before Claude review

**53 dedicated green cases**, actual Windows/Python3.11.9 execution:
[validation ledger](m28_validation_results.json). Normal and importlib full
suites each **685PASS/62deselected/7preexisting Gaia warnings**. Dedicated suite
`-W error` **53PASS**. [Reversed collection/hashseed7](m28_reverse_results.json)
also53PASS. Black/Ruff, pip check and protected-path diff allPASS. The tests
actually render with Agg; no tight_layout/savefig/artist method is bypassed.
PDF verification is limited to format header/trailer and successful backend
write; PNG is opened/verified with Pillow and checked for informative pixels.
PDF scientific layout and text extraction remain PA, not certified.

The ledgers record the active parent HEAD and effective LF source/test hashes
separately. Effective tested files are unchanged by subsequent documentary
commits. The module is imported from its actual file in an isolated package
namespace, with real bridge local imports. The optional facade is bypassed
(#21), not declared generally importable or audited under M2.9. rcParams and
temporary sys.modules entries are restored; every newly created figure closes.
The backend remains Agg for the offline test process, matching CI configuration.

### Findings and duplicate exclusions

All15 findings have separate corrective issues after open/closed inventory
and visualizer/KDE/colorbar searches. **No production correction in this PR.**
D01 was already identified in M1/#43; the other mechanisms were hypotheses or
new reproductions. They are not15new astrophysical discoveries. P1 for false
KDE probability/observation reassignment; P2 for bounded representation and
robustness findings. Historical incidence/impact is unmeasured.

| Proof ID | Corrective issue | Evidence classification |
|---|---|---|
| D01 | [#101 KDE probability mass](https://github.com/PhD-Brown/AstroSpectro/issues/101) | AP demonstrated; M1 risk reverified numerically |
| D02 | [#102 HR gravity units](https://github.com/PhD-Brown/AstroSpectro/issues/102) | AP polygon geometry defect |
| D03 | [#103 hardcoded population](https://github.com/PhD-Brown/AstroSpectro/issues/103) | AP displayed count mismatch |
| D04 | [#104 reconstruction identity](https://github.com/PhD-Brown/AstroSpectro/issues/104) | AP wrong group mean after truncation |
| D05 | [#105 cluster colorbar](https://github.com/PhD-Brown/AstroSpectro/issues/105) | AP mismatch for nonconsecutive IDs; default contiguous fit unaffected in these examples |
| D06 | [#106 probability availability](https://github.com/PhD-Brown/AstroSpectro/issues/106) | AP invalid inference from ones; provenance policy open |
| D07 | [#107 class-only AE plots](https://github.com/PhD-Brown/AstroSpectro/issues/107) | AP scalar-Axes crash |
| D08 | [#108 profile panels](https://github.com/PhD-Brown/AstroSpectro/issues/108) | AP one-slot crash/unused-slot presentation |
| D09 | [#109 reconstruction panels](https://github.com/PhD-Brown/AstroSpectro/issues/109) | AP unused-slot presentation |
| D10 | [#110 variance annotations](https://github.com/PhD-Brown/AstroSpectro/issues/110) | AP count/percentage mismatch |
| D11 | [#111 small positive MSEs](https://github.com/PhD-Brown/AstroSpectro/issues/111) | AP dropped observations in log bins |
| D12 | [#112 fifth class](https://github.com/PhD-Brown/AstroSpectro/issues/112) | AP omission outside usual <=4LAMOST classes |
| D13 | [#113 default export directory](https://github.com/PhD-Brown/AstroSpectro/issues/113) | AP ignored configured output for two direct exporters; others statically similar |
| D14 | [#114 hidden trustworthiness](https://github.com/PhD-Brown/AstroSpectro/issues/114) | AP computed low values outside displayed range |
| D15 | [#115 default Balmer labels](https://github.com/PhD-Brown/AstroSpectro/issues/115) | AP nominal default rendering failure; NB01 supplies custom labels |

Existing #50/#73/#88 do not repair a renderer's own truncation. #90 is the
producer probability boundary, whereas #106 concerns its consumer's missing
availability input. #59 concerns unattainable producer targets, whereas #110
concerns an attainable full-model target with a truncated display. #65/#78/
#84/#85/#89/#95/#100 remain linked without duplicate issues or implied fixes.

### Mutation and admissible controls

[Final green mutation replay](m28_mutation_results.json): baseline53PASS,
**16KILLED/0SURVIVED**, all selected edited sites reached, matching per-selection
collection and assertion failures only. [Initial replay](m28_mutation_initial_results.json)
is preserved. Each mutant is run on the named targeted selection, not claimed
to have executed the whole repository. No perfect mutation coverage is inferred.

| Mutants | Independent contract exercised |
|---|---|
| M01/M07/M08/M10/M11/M15 | Row/class/HR/cluster/confidence/method/KDE population association |
| M02/M03/M04/M05/M06/M12/M13/M14 | Fraction-to-percent, bestepoch, mean-not-median, distinct AE/PCA MSE, reconstruction, signed profile, normalized density |
| M09 | Real sklearn trustworthiness compared with independent ranks |
| M16 | Blank saved figure rejected by real PNG pixel variation |

[Admissible/false-remedy campaign](m28_evidence_final_results.json) contains
C01–C12/C14PASS and F01/F02KILLED. Its C13/F03 isotropic results are superseded
**only for KDE** by [final anisotropic replay](m28_kde_final_results.json):
C13 integrated thresholds PASS, F03 fixed normal thresholds KILLED (two
assertions, same selected collection). Thus **14 admitted alternatives and
three rejected false remedies**, separate from16incorrect mutants. Baselines,
patches, traces and failure texts are retained. Aesthetic palette change and
alternative canvas export pass. Invalid-length explicit refusal passes while
permanent refusal fails on a valid aligned population; empty HR regions fail.

Initial audit mistakes remain visible: fixture import order, a tied variance
ranking fixture, AST bare-return handling, multiline constant trace attribution
and an insufficiently discriminating isotropic KDE. The final anisotropic
analytic oracle handles density thresholds above the peak with mass0. These
are corrections to audit instrumentation/oracles, never production edits.

### Immutable red archive and boundary observations

Designated single final archive:
**[ed1d91ec0efb619aeb77a5bf7e4a0ab43fa84cac](https://github.com/PhD-Brown/AstroSpectro/tree/ed1d91ec0efb619aeb77a5bf7e4a0ab43fa84cac)**,
ref `audit/43-dimred-visualizer-proofs`, also an ancestor of the integration
head. Pushed SHA verified and proof file read back through GitHub **before**
normal removal from the green head. [Exact archive replay](m28_archive_replay_results.json)
**21FAIL/1PASS/0ERROR/SKIP** on22cases. [Green-head reconstruction replay](m28_green_head_archive_replay_results.json)
has the same results. [Proof manifest](M28_PROOFS.md), issues and
[issue index](m28_issue_index.json) link the same immutable source.

```powershell
python audit/check_m28_evidence.py --mode baseline
python audit/check_m28_evidence.py --mode proofs --proofs-ref ed1d91ec0efb619aeb77a5bf7e4a0ab43fa84cac
python audit/check_m28_evidence.py --mode mutations
python audit/check_m28_boundaries.py --output test-results/m28-boundaries.json
```

The default reconstruction ref activates only when the explicit proof file is
absent. The runner checks that current production equals the selected archive,
then restores exact proof/fixture/green bytes in its own disposable tree.
Earlier checkpointd399880 also remains an ancestor, but is not advertised as a
second final archive. All old ledgers remain byte-identical.

[Final boundary ledger](m28_boundary_final_results.json): **38actual method
signatures/44observations**, plus static notebook calls. Initial38observations
remain in [initial ledger](m28_boundary_results.json). These are PA observations,
not green assertions freezing an invalid-input policy: noSTAR/singleton/two
points/collinear/nonfinite KDE, missing physical data, empty HR, zero/negative/
nonfinite MSE, invalidsamplecount/k, emptyhistory/stability and zero family
weights. Eigenspectrum default parser failure became D15. Out-of-range,
NaN/Inf supplied confidence can be accepted/clipped/masked; this remains PA
at the consumer and linked to #90/#106, not claimed as calibrated probabilities.
Collinear/two-point backend acceptance can depend on floating covariance;
no successful density computation alone certifies a meaningful2D population.

### First-pass questions for Claude ? historical

Challenge numerical oracles and adapter limits first: KDE grid tails and label
meaning; mismatched population refusal versus information unavailable to the
API; colorbar versus categorical alternatives; between-feature dispersion;
annotation grammar/diagnostic keyword bounds; fixture state restoration; actual
render/export limits. Assess fixed183feature/3PC/min_size300 titles, [M/H]
versus[Fe/H] conventions and noise-as-anomaly prose as PA/open interpretation
(#65/#78), not validated taxonomy. Review display clipping/zero/nonfinite and
missing-panel policies before any production correction. No M2.9 work or
review correction is authorized before this first independent review.
