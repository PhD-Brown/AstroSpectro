# M2.8 - revised evidence after Claude review 5858927009

> Third-pass update after review 5895743573: this is the preserved second-pass
> manifest. See [targeted current revision](M28_ROUND3_PROOFS.md) for R1, R2b,
> R6/R7 and the proposed R10 normalization policy. Historical results remain unchanged.

Revision date: 2026-11-13. Refs #43 / Draft PR #116. Same worktree
`AstroSpectro-m28`, same branch `audit/43-dimred-visualizer`; production blob
`624b857bca5833c3401204b0a592b9ae6ffc2902` and base
`39dad874610745ccd456d26972e90f6fb3a11405` unchanged.

## Archives and historical separation

- Original archive **ed1d91ec0efb619aeb77a5bf7e4a0ab43fa84cac**, original ref
  `audit/43-dimred-visualizer-proofs`, preserved byte-for-byte. Historical
  baseline: 22 cases, 21 FAIL / 1 PASS. The first-pass JSON ledgers are unchanged.
- Revised archive **a8bc8506804cf4c89e784e458498d3f229ca81cb**,
  immutable tag `audit/m28-review-proofs-2026-11-13`. Seven archived files were
  read back from GitHub and their Git blob IDs verified before normal removal of `tests/proofs_dimred_visualizer.py` from the green head.
- The revised baseline is **26 cases: 25 FAIL / 1 PASS / 0 ERROR / 0 SKIP**,
  executed in [baseline ledger](m28_review_frozen_controls_results.json).
  These are the original D01-D15 groups plus one R10 characterization, not
  25 independent bugs. The passing profile `n_show=0` refusal remains a control.

[Archive verification](m28_review_archive_verification.json) records GitHub
blob IDs, preservation of all first-pass JSON files, original-archive ancestry
and unchanged production. [Red reconstruction replay](m28_review_green_red_replay_results.json)
and [joint reconstruction replay](m28_review_green_joint_replay_results.json)
restore exact archived bytes after the proof is removed from the green head.

## Changes to the proof contracts

D01 expands two cases to five: real Gaussian, analytic Gaussian, three
bimodal/anisotropic/non-Gaussian STAR populations with independent seeds and
QSO observations outside the stellar modes. The estimator's actual dataset
must consist of STAR rows. Contour percentages are read from actual label
formatters, linked to actual contour artists and pre-inline-label geometries.
Broad-grid numerical integration checks both the density superlevel probability
and probability in the actual polygons. Vertices must lie on the estimator's
iso-density. Even/odd polygon containment retains disjoint modes and holes.
The analytic law provides an independent polar integral and a coherent
sampling/evaluation protocol (dataset/d/n/covariance/factor/resample/evaluate).

Numerical tolerances: total broad-grid mass within 1e-4 of one; labelled masses
within .006 (quadrature cell boundaries and contour interpolation, not physical
uncertainty); contour vertex density within 5% on the 200x200 display mesh.
Mixture integration uses spacing .05 on [-12,18] x [-14,14]. Gaussian integration
uses the independent 401x401 covariance-scaled [-8,8] domain. The original green
analytic quadrature check remains. Empirical coverage of observed stars is a
different quantity from probability of the fitted KDE and is not substituted.

A further geometry experiment found that valid mass solvers alone can leave
open 99% arcs on the original percentile display domain. K4, K5 and C13 current
controls therefore also expand that display domain; the solver algorithms
remain different (resampling, kernel-support quadrature, broad-grid quadrature).
The exploratory failure is retained, not counted as a mathematical rejection
of those solvers. Complete contour geometry is part of the revised contract.

D02 derives K/dex envelopes independently from supplied physical values and
dimensionally consistent 10%-range margins. This rejects extending axes to
hide -93 dex. It does not validate astrophysical regional taxonomy. D03 accepts
omission or the actual population, including grouped-number/N= syntax. D04
corrects the explanatory row: original F row 8 is relabelled G after removal
of original G row 4; the independent F means stay 1140/13 and 1060/12.

D05 accepts correct discrete colorbars, categorical Patch/Line legends, or
colored ID annotations. Any displayed categorical colorbar must be faithful;
good annotations cannot excuse a contradictory colorbar. It does not impose
an aesthetic palette. D09 requires actual nonempty original/reconstructed
arrays in every visible measurement panel, with the correct rows and count.
D10 links percentage/PC count to full-model cumulative variance in either
phrase order. D11/D12 capture real histogram inputs and independently count
sample membership in every bin, including the closed right endpoint; bar
heights must represent those counts. A legend-only fifth class is insufficient.

D13 compares decoded PNG scientific content to an explicit export of the same
returned figure (RGBA, within-process, one channel unit). Blank images and
axes-only clones with removed data collections fail. Alternative canvas exports
remain admissible; PDF scientific rendering and cross-version image agreement
remain PA. D14 accepts full visibility or actual boundary/extreme information
for clipped curves, with method identity when needed. Histogram clipping needs
class identity, exact excluded/total counts and support information. Generic
warnings fail. Zero, negative and nonfinite MSE policies remain open.

The original 53 green nodes are preserved and strengthened where requested.
Eight adapter controls add probability consistency, faithful/misleading
categorical legends and exact/misleading clipping disclosures. Green contracts
link XGB/HDB classes and counts to actual coordinates, PCA/UMAP/AE names to
row geometries, interpolation endpoints to supplied labels, and PC legends to
actual bar series. AE/autoencoder naming is semantic, not a French sentence.
Trustworthiness captures the real sklearn subsets, checks shared observation
indices, independently recomputes ranks and verifies repeatability. A different
paired deterministic sampling algorithm is admissible.

## Campaign provenance and interpretation

`check_m28_review.py` copies source/tests into synthetic TemporaryDirectories.
Each ledger retains exact patches, LF hashes, active parent HEAD, collection,
executed statement anchors and assertion texts. Wrong candidates use a passing
admissible joint baseline on the same selection; admissible controls record the
original production baseline (expected red for defective contracts). Every
changed hunk must be executed, collections must match and failures must be
assertions. Import/fixture/backend errors do not count as scientific kills.
A clean passing wrong candidate is reported SURVIVED, not silently accepted.
The joint experiment checks all green and red contracts together. None of these
synthetic remedies changes production or constitutes a proposed production PR.

The `*_initial_results.json`, `m28_review_geometry_results.json` and
`m28_review_geometry_final_results.json` are exploratory checkpoints, not final
validation: the first polygon experiment exposed clipped contours; an early
A_HIST replacement literal was absent; mid-experiment test revisions triggered
the protected-input guard in exploratory runners. Their per-case observations
are retained, but authoritative final runs use frozen inputs and are listed
below. Original first-pass ledgers are never overwritten.

## Frozen validation and exact replay

The validation ledgers distinguish active parent HEAD from actual executed LF
hashes. The archive must match those tested bytes; documentary commits do not
retroactively turn an earlier parent SHA into a tested tree. Windows Python
3.11.9, NumPy 2.4.6, pandas 3.1.1, SciPy 1.17.1, sklearn 1.9.1, Matplotlib 3.11.2,
pytest 8.4.2, Pillow 13.0.0; Agg rendering and sockets blocked by test setup.

| Execution | Actual result / ledger |
|---|---|
| Original 53 node IDs | All retained; 8 new adapter nodes, 61 total |
| Targeted `-W error` | 61 PASS - [frozen validation](m28_review_frozen_validation_results.json) |
| Complete default/importlib suite | Each 693 PASS / 62 deselected / 7 preexisting Gaia warnings - same ledger |
| Complete reversed/hashseed 7 | 693 PASS / 62 deselected / 7 warnings - [order ledger](m28_review_frozen_order_results.json) |
| Targeted reversed + shuffle seeds 3/11/29, `-W error` | Each 61 PASS - same order ledger |
| Black `--check`, Ruff, pip check, diff/check and protected paths | PASS, including all four M2.8 audit scripts - frozen validation ledger |
| M01-M16 | 16 KILLED / 0 SURVIVED / 0 inconclusive, baseline 61 PASS - [frozen mutations](m28_review_frozen_mutations_results.json) |
| Old archive replay | 21 FAIL / 1 PASS / 0 ERROR/SKIP - [old archive verification](m28_review_old_archive_integrity_results.json) |

Review campaign: **19 KILLED / 0 SURVIVED / 0 inconclusive** for incorrect
candidates; **12 admissible controls PASS**, plus **87/87 PASS** for the joint
experiment. Each wrong candidate has a passing comparable admissible baseline,
all edited hunks reached, matching collection and meaningful assertions. The
original-remedy campaign separately gives **14 PASS / 3 false remedies KILLED**.
These are bounded candidate experiments, not counts of distinct production bugs.

| Review candidates | Result |
|---|---|
| K1B / K6 / K7 / K8 / K9 | All KILLED (family-only mass, swapped labels, wrong fitted population, transpose, removed mode) |
| FD02 / FD09 / FD11 / FD12 / FD13 | All KILLED (axes expansion, empty lines, filtered input, legend-only class, blank export) |
| FD03 / FD10 | Both KILLED (misleading count/PC annotation reformulations) |
| N01 / N02 / N03 / N04 / N05 / N06 / N10 | All KILLED (semantic/data association and axes-only export) |
| C13 / K4 / K5 | Each 5 PASS with independent solvers and complete contour display domain |
| AD05 / AC05 | Each 2 PASS for real cluster/HR encoding (legend/discrete colorbar) |
| A_TRUST / AD14 / A_EXPORT / A_METHODS | Paired sampler, precise clipped-curve disclosure, equivalent exports and AE naming PASS |
| A_HIST / AD_HIST / AD_LOG | Full histogram support, precise upper/lower exclusions PASS |
| JOINT | 87 PASS across all 61 green + 26 red cases |

Only the **frozen** ledgers and the archive-verification/replay ledgers are
current delivery evidence. Other `m28_review_*results.json` files retain earlier
inputs and experiments; do not aggregate their totals. In particular, an early
AD_HIST candidate included unrelated joint patches, so its unexecuted hunks made
it INCONCLUSIVE; the final candidate is restricted to the actual histogram
change and passes the execution guard. No such inconclusive result is hidden
or counted as a scientific kill.

The review-specific frozen campaign is recorded in
[m28_review_frozen_results.json](m28_review_frozen_results.json). It includes
actual pre-patch baselines, all edited statement anchors, matching collections,
assertion texts, exact candidate patches, tested hashes and a protected-input
guard. The original C01-C14 and F01-F03 are replayed separately in
[m28_review_frozen_controls_results.json](m28_review_frozen_controls_results.json).
C13 additionally completes the display domain for the polygon contract; this
revision does not alter its broad-grid probability algorithm. Earlier C13-only
solver observations remain in exploratory logs.

On the revised archive checkout, execute red proofs explicitly:

```powershell
python -m pytest -q tests/proofs_dimred_visualizer.py --tb=short
python audit/check_m28_review.py --output test-results/m28-review.json
python audit/check_m28_evidence.py --mode mutations --output test-results/m28-mutations.json
python audit/check_m28_evidence.py --mode proofs --output test-results/m28-controls.json
```

From the green checkout, use the immutable archive SHA below with:

```powershell
python audit/check_m28_evidence.py --mode baseline --proofs-ref a8bc8506804cf4c89e784e458498d3f229ca81cb --output test-results/m28-red-replay.json
python audit/check_m28_review.py --proofs-ref a8bc8506804cf4c89e784e458498d3f229ca81cb --output test-results/m28-review-replay.json
python audit/check_m28_review.py --proofs-ref a8bc8506804cf4c89e784e458498d3f229ca81cb --only JOINT --output test-results/m28-joint.json
```

The runners reconstruct exact archived tests/fixtures/helpers in disposable
trees and check production equality. The red pytest exit 1 is expected; no
xfail hides it. The default baseline ref when the proof is absent is the fixed
review-proof tag. To replay the original archive, pass its full SHA explicitly.
Do not overwrite historical audit ledgers with replay outputs.

CI is a separate hosted execution; its exact head/merge SHA, run link and
configured checks are recorded in the updated PR #116 and #43 handoff after the
new green head is pushed. Old CI run 42952227591 is first-pass history only.
CI does not execute the explicit red proofs or mutation campaigns.

## R10 - separate corrective issue proposal, awaiting human decision

**Proposed title:** `fix: disclose population excluded from linear error histograms`

**Problème:** at the unchanged production base, line 2833 sets the linear
histogram support to `[0, 1.5*P99(STAR)]`; `density=True` normalizes only retained
samples. Supply STAR=`linspace(.01,.1,40)`, QSO=`[.04,.09,.1] + linspace(4,7,47)`.
Upper support .14865 retains **3 of 50 QSO**, drops **47**, and still has **area 1**;
its labelled full-population median is **5.402173913...**, outside the support.
The archived `linear_histogram_truncation_discloses_population` test independently
counts input membership and the actual plotted bar area.

**Pourquoi:** a reader can mistake conditional density for the full class
population. Proposed labels bug/science/P2, limited to this synthetic scenario;
no historical astronomical run/publication impact is established.

**Attendu:** cover the relevant positive class population, or disclose precisely
which class, how many excluded out of its total and the support used. Conditional
normalization is admissible when that limitation is explicitly informative.

**Vérification:** run the archived R10 selection. A_HIST full-support alternative
and AD_HIST exact exclusion disclosure are positive controls. The original
production fails by the meaningful 3/50 assertion, not a runtime error.

**Hors scope:** production fixes, zero/negative/nonfinite policy, catalogues,
models, historical scientific results, and a new issue before human approval.

Duplicate search (open and closed, 2026-11-13): `histogram`, `histogramme`,
`"star_p99"`, `"linear" "density"`, `renormalisation`; relevant #111 and #112
read. #111 is the **logarithmic 1e-4 lower floor**; R10 is the **linear STAR-based
upper bound and conditional renormalization**. #112 is palette/class omission.
No matching separate issue found in these searches; **no new issue created**.

## Residual scope and second review

The method table now contains all 38 actual methods. AP includes the existing
`hr_regions_scatter_preserves_input_triplets_despite_geometry_defect` numeric
triplets/color/orientation and missing-column sentinel; regional margins are PA
and astrophysical region geometry/taxonomy NA.
N08's hot/cold taxonomy shift is not claimed rejected. Historical N09 exception
failure is not promoted to a fresh assertion-based scientific kill. Metallicity
mh_gspphot versus Fe/H, fixed 3-PC/183-feature/min_size300 prose and anomaly
interpretation remain PA/open (#65/#78). No real catalogue/FITS/model run,
external identity proof, calibration, astrophysical validation or line coverage
is claimed. The API remains positional. Diagnostic adapters are bounded and
cannot parse every possible natural-language disclosure.

The CI workflow is unchanged: it checks configured src/tests/scripts/app paths,
not audit scripts. All four M2.8 audit scripts are explicitly linted locally.
All twelve inline threads remain unresolved for Claude's second independent
review. PR #116 remains Draft; M2.8 yellow, #43 open, M2.9 not started.
