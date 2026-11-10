# M2.7 red proof manifest

Refs #43. Source/base `1e2266385eab287d3a1c54c2e919e012293fa44b`,
reporter blob `b966d9b90f056baa37c0ccbb68135c71ff65de92`, unchanged.
The proof file is executed explicitly, never collected as a green CI file.
It is archived in this commit's history and removed in a later normal commit.
No xfail, production edit or history rewrite. All data are synthetic and local.

Command: `../AstroSpectro/.venv/Scripts/python.exe audit/check_m27_mutations.py
--proofs --output audit/m27_proof_results.json`.
Actual result: **18 FAIL / 1 PASS / 0 ERROR / 0 SKIP**.
The ledger contains complete failure messages/stdout, executed production lines,
effective test/source LF SHA256 and Python version. These hashes describe the
audit's tested files; they are not fields emitted by production manifests.
Git HEAD in a working-tree execution is its parent, not a claim that modified
tests already existed at that commit. The archive SHA is linked in the final
matrix, issues and handoff after commit/push verification.

| Proof test | Cases / result | Independent invariant and admissible alternatives | Classification |
|---|---|---|---|
| `dataframe_missing_values_are_strict_json` | 2 FAIL | Docstring's missing->None for float64 and pandas3 inferred str; positive first row and immutability retained | NEW JSON conversion defect |
| `report_file_is_strict_json_or_refused` | 4 FAIL | NaN, +/-Inf and nested ndarray cannot emit nonstandard JSON; explicit refusal accepted, no replacement policy imposed | NEW strict JSON boundary, grouped with conversion |
| `archive_collision_cannot_overwrite_first_run` | 2 FAIL | A successful distinct run cannot replace the archived first experiment; reject collision or choose unique path | NEW archive collision |
| `notebook_same_second_preserves_first_experiment` | 3 FAIL | NB01/NB02/NB03 same clock value and changed row labels; original joblib/JSON/TXT bytes retained or explicit refusal | Same collision, not three more bugs |
| `failed_run_cannot_publish_mixed_latest` | 2 FAIL | Baseline A then B; injected OSError before latest JSON or latest TXT; aliases must agree on run | NEW nontransactional publication |
| `prediction_figure_is_not_a_model_path` | 1 FAIL | Notebook passes prediction PNG; preserve figure path but do not claim it identifies a trained model | NEW wrong provenance field |
| `failed_metric_is_diagnosed` | 2 FAIL | AE or PCA numerical ValueError must propagate or have diagnostic log/status; null alone loses failure reason | NEW diagnostics loss; API exception injected, not real backend failure incidence |
| `nonfinite_latent_not_reported_as_ordinary_spearman` | 1 FAIL | Axis1 Inf on row2; finite-pair ranks identical => rho1, current rho23/35. Refuse or mark unavailable accepted | NEW finite-pair validation defect |
| `single_latent_axis_is_reported_or_refused_clearly` | 1 FAIL | One coordinate is either supported or refused with dimensional diagnostic; unguarded IndexError rejected | NEW dimension robustness gap; NB03 currently latent_dim2 |
| `positive_valid_report_is_nonempty` | 1 PASS | Exact 1/9 projection MSE, labels, payload rows | Guard against always-empty/always-refuse patches |

**18 failures are not 18 bugs.** Seven grouped findings are proposed for
separate corrective issues; no fix is applied here. The proofs concern the
current API/schema. A future atomic commit marker/manifest schema must adapt
the latest-group oracle to its explicit availability semantics. The present
two tests demonstrate mixed ordinary aliases without such a marker; they do
not simulate a power loss or all possible filesystem failures.

Metric-failure oracle deliberately excludes file paths and diagnostic key
names from its keyword search. Early local versions accidentally matched
`test_failed_metric...` in tmp paths, then the word `errors` in the helper's
own key; both false positives were corrected before this archive. The final
proof has both failures by assertions. Its log/status check remains bounded,
not a general diagnostic classifier; independent review should challenge it.

The finite Spearman proof excludes Inf only from the first latent axis fixture;
it does not impose pairwise versus complete-case policy for unrelated axes.
The second axis stays finite. A finite result for the remaining first-axis
pairs is unambiguous. No physical validation is inferred from a correlation.

Run collision proofs compare bytes only within the same process/versions and
original archived file, not across platforms or joblib versions. Roundtrip
does not establish cross-version compatibility, integrity authentication or
scientific reproducibility. Only trusted joblib files created here are loaded.

Known issues remain #21/#22/#23/#38/#50/#68/#80–#90. None covers these reporter
write/attribution/finite-pair defects. #20 covers broader run manifests in
classifier/master; dimred provenance remains an open design question. Figure
folder provenance and external equal-size row permutations are documented
limitations, not magically detectable IDs or additional red proofs.
