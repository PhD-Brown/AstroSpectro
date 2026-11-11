# M2.8 — immutable red evidence manifest

Refs #43. Base39dad874610745ccd456d26972e90f6fb3a11405;
production visualizer blob624b857bca5833c3401204b0a592b9ae6ffc2902 unchanged.
`tests/proofs_dimred_visualizer.py` is explicitly executed, never discovered
by the normal test naming pattern. It is committed in one final archive,
pushed/read back, then removed normally from the Draft PR head. No xfail.
The designated archive is **ed1d91ec0efb619aeb77a5bf7e4a0ab43fa84cac**; its
ancestry preserves exact conftest, green tests, source, scripts and results.

Final baseline: **22 cases,21FAIL/1PASS/0ERROR/SKIP**. This is **15 grouped
findings**, not21bugs. The passing `n_show=0` case already refuses an impossible
subplot count. It is a control/observation, not a requirement to keep one
particular exception text. Green valid-data cases accompany invalid proofs.

| ID | Red test selection / failed cases | Independent invariant and admitted outcomes |
|---|---|---|
| D01 | `kde_labeled_regions` /2 | Real SciPy and analytic Gaussian covariance diag(4,9). Independent expanded-grid quadrature and analytic `max(0,1-12*pi*t)`; density-height percentiles do not specify enclosed probability. Integrated thresholds accepted. M1 already noted this risk; no separate corrective issue existed. |
| D02 | `hr_region_vertices` /1 | All gravity coordinates must lie in gravity-domain bounds. Current mixed K/dex margin yields−93.3175dex for supplied ordinary data. Nonempty informative regions required; corrected gravity margin admitted. |
| D03 | `hr_population_title` /2 | Displayed count must be supplied population count6/9 or omitted. Hardcoded43019 is neither. Omission admitted; stellar-region taxonomy is not validated. |
| D04 | `reconstruction_cardinality` /1 | Deleted middle observation: actual F mean1140/13, plotted1060/12. Incompatible lengths must not reassign labels by truncation. Pertinent ValueError/RuntimeError refusal or true correspondence admitted; valid aligned F10/G12 still must work. No external equal-size IDs inferred. |
| D05 | `cluster_colorbar` /2 | Nonconsecutive2/7 labels and actual analyzer palette. Displayed colorbar mapping at each CID must match the point RGB; an informative categorical annotation can replace a colorbar. No fixed aesthetic palette imposed. |
| D06 | `actual_unit_confidence` /1 | Values all1 are compatible with actual probabilities. Without an availability flag the renderer cannot assert predict_proba absent. Missing probability still tested as None in green; no calibrated probability claim. |
| D07 | `no_physical_columns` /2 | Class-only AE/zoom inputs have a defined nonempty class plot or relevant explicit refusal; scalar axes TypeError is not a decision. Correct scalar-to-array handling admitted. |
| D08 | `cluster_profiles_unavailable` /2,1PASS | n_show1 must address an actual axis; n_show4/two clusters must hide unused data slots. n_show0 explicit refusal admitted. Top-five signed numeric data remain protected in green. |
| D09 | `more_reconstructions` /1 | Requested4/available2 must not present blank measurement panels. Hide unused or allocate actual count; valid1/3/6 rows still exact. |
| D10 | `truncated_variance_annotation` /1 | Reported3PCs cannot be paired with85% when3 actually explain100%. Correct numeric association or removal/clarification of that claim accepted; no PCA sign convention. |
| D11 | `log_histogram_keeps` /1 | Strictly positive1e-8..1 MSE values cannot disappear below an arbitrary1e-4 bin floor without indication. Positive-support bin choice admitted; truezero/negative policy remains open. |
| D12 | `fifth_class` /1 | All five supplied classes represented or relevant explicit refusal; no silent loss due palette length. Nominal <=4class density arithmetic separately protected. |
| D13 | `default_output_directory` /2 | Configured output_dir must actually create a default file on successful training/comparison plot; direct calls currently bypass_save. Explicit exports do work and are green. |
| D14 | `trustworthiness_axis_contains` /1 | Calculated.5,.55882353,.50396825 cannot all be below a fixed.75 display floor without indication. Full0..1 range admitted; independent numerical computation stays green. |
| D15 | `default_eigenspectrum` /1 | Actual default Balmer mathtext must draw with Agg. Doubled backslashes cause a parser error. Supplied custom marker labels are already green; unescaping default labels admitted. Notebook01 supplies RAIES, so no historical failure asserted. |

## Reproduction and admissibility

On the archived checkout:

```powershell
python -m pytest -q tests/proofs_dimred_visualizer.py --tb=short
```

From the green checkout with the fixed archive SHA:

```powershell
python audit/check_m28_evidence.py --mode baseline --proofs-ref ed1d91ec0efb619aeb77a5bf7e4a0ab43fa84cac
python audit/check_m28_evidence.py --mode proofs --proofs-ref ed1d91ec0efb619aeb77a5bf7e4a0ab43fa84cac
```

The runner reads exact proof/green/fixture bytes from Git into its own synthetic
TemporaryDirectory; production source is unchanged. No real datasets are copied.
Exit1 for explicit proof pytest is expected. The runner itself reports evidence
without treating expected counterexamples as an infrastructure failure.

Mutations M01–M16 are separately scoped to green tests, all reached and KILLED
by assertions with matching collection. They are not15production fixes.
Fourteen admissible C01–C14 candidates exercise percentages, export, aesthetic
colors, explicit alignment refusal, gravity margin, count omission, small MSE
support, scalar axes, profile addressing, full trust range, hidden unused
reconstructions, integrated KDE and valid mathtext. These are experimental
copies, not patches proposed for production. Selections and test counts are in
the ledgers; each candidate was not run against the entire repository.

F01 always-refuse is rejected by the valid aligned mean oracle, F02 empty HR
by explicit informative-region assertion. F03 standard-normal thresholds for
every KDE initially survived an almost isotropic sample. Covariance diag(4,9)
now discriminates it: F03 KILLED, while integrated C13 remains PASS. The final
KDE ledger supersedes only this narrow earlier evidence. Old results remain
unchanged. Analytic mass is clamped to0 above peak density; this avoids an
incorrect negative mass in the audit oracle itself.

## Limits

Policy adapters are bounded: explicit diagnostic keywords, current French count
and variance annotation grammar, Matplotlib colorbar/annotation interfaces and
probability contour order. They do not parse arbitrary scientific language or
all future plotting implementations. A future API/representation change must
update adapters while preserving numerical contracts. In particular diagnostic
wording is not universal proof of correct refusal, class-based availability is
not object identity and a renderable figure is not astrophysical validation.
No green test demands retaining the demonstrated red behavior.
