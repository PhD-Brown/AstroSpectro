# M2.8 — DimRedVisualizer scientific figure audit

2026-11-10 (Quebec) · Refs #43 · tests, evidence and documentation only.
Worktree `AstroSpectro-m28`, branch `audit/43-dimred-visualizer`.
Verified remote main/base **39dad874610745ccd456d26972e90f6fb3a11405**,
PR99 merged, no open PR at start. Main checkout was clean at1e22663 and is
preserved. Production blob **624b857bca5833c3401204b0a592b9ae6ffc2902**.
Actual inventory: **38 methods**, including initializer and two helpers,
not31. Entire3224-line source read. AGENTS v1.3 fetched from current main.

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

KDE uses actual SciPy plus a separately injected analytic standard normal.
For the latter, polar integration gives enclosed mass `1-2*pi*threshold`.
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
| `plot_embedding` | Z/class/scalar -> positional scatter -> `(fig,ax)` | classes/plain/continuous+missing, percentiles | AP paired valid arrays; invalid coordinates boundary PA |
| `plot_embedding_grid` | Z/y/meta -> class+parameter panels -> `(fig,axes)` | coherent permutation/absent column/missing rows | AP nominal; all-missing/Inf scaling PA |
| `plot_stability` | ordered report -> bars/reference-excluded mean -> `(fig,ax)` | actual producer with controlled fit; asymmetric mean | AP producer order0..N-1; arbitrary reordered reports PA; stability not astrophysics |
| `plot_reconstruction_error` | PCA summary -> mean±SD+elbow -> `(fig,ax)` | actual discarded-energy MSE and band | AP statistics; nonuniform count elbow heuristic PA; log zero hidden |
| `plot_eigenspectra` | PCA vectors/wavelength -> line+markers -> `(fig,axes)` | exact wavelength/values/in-range markers,1/3/10 requested | AP custom marker path; default mathtext and fixed3-PC title PA |
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

## Checkpoint

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
