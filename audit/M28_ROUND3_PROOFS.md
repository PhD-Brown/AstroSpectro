# M2.8 - targeted third pass after Claude review 5895743573

Refs #43 / Draft PR #116. Examined starting head:
`e924f36f24412a435f5ca54a95cdfad914835ff2`, same branch
`audit/43-dimred-visualizer`, same worktree `AstroSpectro-m28`.
Base `39dad874610745ccd456d26972e90f6fb3a11405`; production blob
`624b857bca5833c3401204b0a592b9ae6ffc2902` unchanged. Read main AGENTS v1.3,
second review and all replies in the five concerned threads before editing.
No production, dependency, workflow, notebook, data or model change.

## Scope and historical evidence

Only R1, R2b, R6, R7 and R10 are revised. Resolved observations are not reopened.
The previous 61 green node IDs and numeric contracts are preserved, with the
fixed variance sentence replaced by its count/variance relation as requested.
Five new green nodes exercise locale parsing, component synonyms, both colorbar
orientations and histogram denominators. Shared-helper regressions require
replaying the previous candidate campaigns; that is not a new audit.

Historical archives remain immutable:

- `ed1d91ec0efb619aeb77a5bf7e4a0ab43fa84cac`, ref
  `audit/43-dimred-visualizer-proofs`, baseline 21 FAIL / 1 PASS (22 cases).
- `a8bc8506804cf4c89e784e458498d3f229ca81cb`, tag
  `audit/m28-review-proofs-2026-11-13`, baseline 25 FAIL / 1 PASS (26 cases).

Their JSON logs are not rewritten. The third snapshot is **8e668780b181948856cacb58650b25a0e03eeddf**, tag
`audit/m28-round3-proofs-2026-11-13`. Eight archived files were read back from
GitHub and their Git blob IDs verified before normal proof removal. The
snapshot is an ancestor of the same PR branch; no second M2.8 branch or PR is created.

## R1 - sampled population and complete probability contours

The existing three seeded mixtures retain 210 observations in one mode and
90 in a second mode, plus 40 distinct non-STAR observations. The oracle still
checks actual fitted STAR membership, actual labels, superlevel probability,
actual polygon probability and iso-density vertices, including the analytic
Gaussian control. On mixture cases it additionally checks the fitted fraction
with x > 3 against **90/300 +/- 0.10**. This cut separates these synthetic modes
with small overlap; it is not an astrophysical discriminator.

For a random half-sample n=150 from N=300 and p=.30, the hypergeometric standard
error is sqrt(p*(1-p)*(N-n)/(n*(N-1))) = about .0265. The .10 envelope is about
3.8 such standard errors, also allowing the small overlap of the two modes.
This is a gross-selection-bias check on these finite seeded fixtures, not a
universal unbiasedness test, confidence guarantee, or mandatory sample size.
The oracle never requires all STAR observations or a particular RNG sequence.
K10 removes the second mode and must fail on this population contract before
incidental interpolation errors; K11 samples half randomly and must pass.
C13, K4 and K5 remain admissible independent mass-solving approaches.

Working contour policy supplied for this pass: a contour advertised as a
complete 68/95/99 percent probability region must be closed and wholly inside
the displayed domain. Probability outside the visible frame cannot silently
be advertised as enclosed by its visible portion. The fixtures allow complete
regions, so the proof verifies closure and actual axis-domain containment as
well as probability. The display-domain completion used by C13/K4/K5 is
retained. A different explicitly partial-region product would need its own
unambiguous claim and validation; this bounded proof does not certify such an
alternative. The existing contour-label adapter still targets clabel; proxy
contour legends suggested in the review are not validated in this targeted pass.
This policy and its limits are linked from #101. No universal
coverage of observed stars, astrophysical validation or calibration is implied.

## R2b - every categorical representation must agree

Colorbars are identified structurally as Matplotlib Colorbar objects. A
physical scalar bar is bound to an actual data-bearing collection distinct
from the cluster collections; standalone ScalarMappables remain categorical
regardless of caption or orientation. Actual cluster ID -> actual point RGB
is compared to every relevant colorbar, legend handle and colored annotation.
A correct representation never excuses a contradictory one. Horizontal and
vertical bars, renamed captions, discrete correct bars, Patch legends and
faithful colored annotations (including count suffixes/bare IDs) are covered.
No aesthetic palette or exact French sentence is imposed. Actual visible
Matplotlib Legend artists are read, including figure-level and additional
legends. Underlying artist labels that are not displayed are not treated as
scientific disclosures. A global contradictory legend is rejected even when
an axis-level legend is correct.

An initial annotation-only control removed HR's bar but the tiny fixture had
no HR annotations under the existing 20-point display threshold. Its failure
was genuine missing information, not an admissible-control success. The final
synthetic control enables annotations for its small clusters; production and
its annotation threshold are unchanged. This checkpoint remains in
`m28_r3_targeted_initial_results.json` and is not current delivery evidence.

## R6/R7 - numeric meaning rather than fixed prose

`numbers()` supports decimal dot or comma, spaced/narrow-space/nonbreaking-space
thousands, unambiguous multiple comma groups, scientific notation and k/K/M
suffixes. A single comma is decimal by the explicit French convention; an
isolated English `1,234` is inherently locale ambiguous and is not guessed to
mean thousands. Token matching folds accents, so tronque/tronquées are read
without losing their numeric content. Bad bounds and counts fail assertions,
not float-conversion errors. Excluded/total roles are checked, not just an
unordered set of numbers.

D03 reads 43k or approximately 0.043M as counts, in addition to grouped integers;
catalogue-release metadata remains distinguished. Component annotations accept
PC/PCs/CP/composantes (and equivalent number/word order) and decimal comma.
Each percentage is linked to cumulative model variance for the count mentioned
or the annotated point. The green PCA 9:4:1 control requires count 2 to announce
100*(9+4)/14 within the existing one-decimal printing tolerance .05 percentage
points; it no longer requires the strings '2 PCs' and '92.9%'. False
'3 composantes -> 85 %' fails the scientific relation.

## R10 - two explicit normalization conventions, proposed test policy

The P2 diagnosis and complete five-section issue proposal in
[M28_REVIEW_PROOFS.md](M28_REVIEW_PROOFS.md) remain separate from #111's log floor.
No new issue is created. Human approval is still required for that issue.

The following is the **proposed working test policy for this pass**, presented
for final scientific acceptance; passing these tests does not ratify it:

1. Conditional density divides bin counts by the number inside the support and
   bin width. Its visible area is 1. If observations are excluded, class,
   excluded/total counts and the relevant support cutoff must be disclosed.
2. Total-class density divides by the full class count and bin width. Its
   visible area is the represented fraction, **3/50 = .06** in the QSO example.
   It is admissible with an explicit density scale, actual support and correctly
   identified total reference population. A label claiming a different
   denominator or simultaneous contradictory convention is rejected.

Both cases use independently counted bin populations and actual bar heights.
The denominator is never inferred only from a nice-looking area or caption.
Total-population labels with conditional heights, conditional labels with
full-population heights and incorrect stated totals fail by assertions.
French decimal-comma disclosures are admissible; wrong bounds/counts remain
rejected. Policies for zero/negative/nonfinite MSE, published historical runs,
[M/H] versus [Fe/H], astrophysical taxonomy, fixed titles and anomalies remain
unchanged and open where previously documented.

### Current R10 issue proposal - do not create without human approval

**Title:** `fix: disclose population and normalization in linear error histograms`

**Problem:** unchanged base source line 2833 selects linear bins up to
1.5*P99(STAR). With STAR=linspace(.01,.1,40) and QSO=[.04,.09,.1] plus
linspace(4,7,47), support ends at .14865. Only 3/50 QSO contribute, yet
conditional density has area 1 and the full-class median 5.402173913 is outside.

**Why:** the displayed density can be mistaken for the complete class.
Proposed bug/science/P2, demonstrated on synthetic data; historical publication
impact is not established. This upper-support/normalization mechanism is
distinct from #111's logarithmic lower floor. Earlier duplicate searches and
Claude's confirmation are recorded in the prior manifest/review; no new
unrelated defect search is made in this pass.

**Expected:** full relevant support, or a correctly identified partial-support
representation under the proposed policy above: conditional area 1 with exact
exclusions, or total-class density with area 3/50, correct denominator, density
scale and support. Scientific acceptance of that test policy remains pending.

**Verification:** archived R10 proof, independent bin counts/bar heights,
A_HIST/AD_HIST/AD_COMMA/A_TOTAL positive controls, and misleading-denominator
candidates. Both normalization conventions must be distinguishable.

**Out of scope:** production fixes, new issue creation without explicit human
approval, zero/negative/nonfinite policy, real data/models/catalogues and
historical scientific results. The issue has not been created.

The displayed-legend adapter controls were run before correction: the two
orientation cases failed to reject a contradictory figure legend, and a valid
proxy-legend-only total normalization was rejected (3 assertion failures,
10 deselected). After correction the full 13 adapter cases pass. This tests
the requested representation contracts, not a newly claimed production bug.
Earlier `m28_r3_*` checkpoints precede this last adapter completion; current
checks use the `m28_r3_frozen_*` ledgers and subsequent archive replays. Their
historical outputs are retained and not aggregated as extra kills.

## Verification and replay

| Local check | Actual result / ledger |
|---|---|
| Previous 61 green node IDs | All retained; 5 added nodes, 66 total |
| M2.8 `-W error` | 66 PASS - `m28_r3_frozen_validation_results.json` |
| Full default and importlib pytest | Each 698 PASS / 62 deselected / 7 preexisting Gaia warnings - same ledger |
| Reversed M2.8, hashseed 7 | 66 PASS - `m28_r3_frozen_reverse_results.json` |
| M01-M16 | 16 KILLED, baseline 66 PASS - `m28_r3_frozen_mutations_results.json` |
| Original C01-C14 / F01-F03 | 14 PASS / 3 KILLED - `m28_r3_frozen_controls_results.json` |
| Explicit red baseline | 25 FAIL / 1 PASS / 0 ERROR/SKIP, 26 cases - same controls ledger |
| Black --check/Ruff, pip check, diff/check, protected paths | PASS; four audit scripts explicitly included - frozen validation ledger |

The authoritative [candidate ledger](m28_r3_frozen_results.json) gives **27
KILLED / 0 SURVIVED / 0 inconclusive**, **18 admissible controls PASS** and two
joint experiments **92/92 PASS each** (66 green + 26 red). This comprises the
previous 19 incorrect candidates plus 8 targeted variants, not 27 production
bugs. Every wrong candidate has a passing admissible baseline, matching
collection, all changed hunks reached and assertion failures. No conversion
or fixture/backend error is counted as a scientific kill.

| Targeted candidate | Actual result |
|---|---|
| K10 | 3 mixture assertion FAIL / 2 Gaussian PASS; fitted mode fraction 0 versus .30, before density interpolation |
| K11 | 5 PASS; valid random half-sample, no universal size mandate |
| FD05 / FD05_H | Each 2 assertion FAIL; renamed contradictory vertical/horizontal bars |
| AC05_H / AD05_ANN | Each 2 PASS; correct horizontal bar / faithful annotations |
| FD03_K / FVAR_COMPONENT | 2 / 1 assertion FAIL for 43k count and false 3-component variance |
| A_VARIANCE | 2 PASS (green analytic PCA and truncated-display proof), decimal comma and composantes |
| AD_COMMA | 2 PASS for exact French exclusions; FD_COMMA_BOUND fails the relevant upper cutoff in R10 (D11 lower-cutoff contract still passes) |
| A_TOTAL | 1 PASS, QSO visible area 3/50 |
| F_TOTAL_LIE / F_CONDITIONAL_LIE | Each 1 assertion FAIL for incompatible heights and declared convention |
| JOINT / JOINT_TOTAL | Each 92 PASS; disclosed conditional / identified total-class convention |

C13/K4/K5 remain 5 PASS each; AD05/AC05 remain 2 PASS each. The original
M01-M16 remain 16 KILLED; original C01-C14 remain 14 PASS and F01-F03 3 KILLED.
The execution guard confirms inputs were unchanged throughout the frozen run.
The disclosure adapter validates the crossed cutoff and supported numeric
roles; it does not certify arbitrary extra prose or every possible annotation.
 Earlier checkpoints
are not substituted for the current run. Active HEAD and LF hashes identify
the actual tested bytes separately. Each wrong
candidate uses a passing admissible baseline on the same selection; collection
and changed-hunk execution must match and failures must be assertions. Import,
fixture, conversion and incidental backend failures do not count as kills.
All modifications of production for these experiments occur only in synthetic
TemporaryDirectories. The real production source is never changed.

On the new snapshot checkout:

```powershell
python -m pytest -q tests/proofs_dimred_visualizer.py --tb=short
python audit/check_m28_review.py --output test-results/m28-round3.json
python audit/check_m28_evidence.py --mode mutations --output test-results/m28-round3-mutations.json
python audit/check_m28_evidence.py --mode proofs --output test-results/m28-round3-controls.json
```

From the green head, add `--proofs-ref 8e668780b181948856cacb58650b25a0e03eeddf` to the baseline/proof
reconstruction commands, or to `check_m28_review.py` (optionally `--only JOINT
JOINT_TOTAL`). The fixed tag is the default baseline ref once the proof file is
absent. Replays restore exact archived tests, fixtures and helpers and verify
production equality. Do not overwrite historical audit JSON outputs.

CI checks only its configured paths and normal green suite; explicit red
proofs, mutation campaigns and lint of the four M2.8 audit scripts are separate
local checks. Hosted logs and their actual head/merge SHA will be linked from
PR #116 and the #43 handoff. This pass stops for Claude's short verification on
K10, FD05, R6/R7 and R10. PR remains Draft, M2.8 yellow, #43 open, M2.9 untouched.

[Archive/blob verification](m28_r3_archive_verification.json), [green-head red replay](m28_r3_green_red_replay_results.json) and [joint reconstructions](m28_r3_green_joint_replay_results.json) preserve exact replay evidence.

After normal removal from the green head, exact archive reconstruction returned **25 FAIL / 1 PASS / 0 ERROR/SKIP**, and both JOINT and JOINT_TOTAL returned **92 PASS**. Protected-input guards passed. Green-head validation repeated **66 PASS with -W error**, both full suites **698 PASS / 62 deselected / 7 preexisting warnings**, Black/Ruff, pip check and protected-path diff PASS (`m28_r3_green_validation_results.json`). Hosted CI is recorded separately in PR #116 and the #43 handoff on the actual pushed green SHA.
