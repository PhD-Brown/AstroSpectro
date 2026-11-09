# M2.5 — Executable red proofs and boundary observations

Refs #43. This archive is deliberately separate from the integration diff and
the default pytest suite. No xfail, production repair, data/model change or
scientific validation claim. Source audited: main795af2c, blob
`b06bfa75a2e0a8b132d0a0fb8e0a899b6afed0ad`.

Inputs committed at **47c56c1da7a07d59f295d2689e0b907b96b1c2f1**; run HEAD is
recorded in `m25_proof_results.json`. Its input content hashes identify exactly
the proof, fixture, configuration and unchanged production source. JUnit XML
records failures and observed stdout, including boundary diagnostics. This
manifest/results/control script were untracked during the proof executions;
they do not purport to have been committed at47c56c1.

Environment: Windows AMD64, Python3.11.9; isolated `.venv-torch`, PyTorch
**2.10.0+cpu**, cuda=None; NumPy2.4.6, pandas3.1.1, sklearn1.9.1, pytest8.4.2.
CPU threads1 and deterministic algorithms inside real-Torch fixture; fresh
Torch seed1729 per test, restored after. No unknown checkpoint or real data.

```powershell
.\.venv-torch\Scripts\python.exe -m pytest -q -m 'slow or not slow' tests/proofs_dimred_autoencoder.py --tb=no --junitxml=audit/m25_proofs.xml -o junit_logging=all
.\.venv-torch\Scripts\python.exe audit/check_m25_proof_controls.py
```

First command: **11 FAIL /14 PASS /0 ERROR /0 SKIP /0 XFAIL**, exit1,
12 runtime warnings from intentionally empty/NaN summary observations. Replayed
twice with the same outcomes (initial 20case development version11/9 is
superseded by this25case archive). The failure count is **not a bug count**.

| Cases (prefix `test_` omitted) | Outcome | Meaning / independent expectation |
|---|---|---|
| `absent_torch_optional_boundary_has_domain_diagnostic[module/package/pca/invoke]` |4FAIL| Known#21: fresh isolated imports fail at `nn.Module` with NameError before `_check_torch`. A Torch-related ImportError diagnostic or reachable optional API is admissible; invocation cannot be reached on current source. |
| `valid_small_training_does_not_crash_inside_batchnorm[6-4-.1/2-10-.1]` |2FAIL| Five train rows with final batch1, or one train row: internal BN ValueError. Domain validation or defined small-batch training accepted; no prescribed drop_last choice. |
| `failed_fit_cannot_expose_random_model_with_stale_history` |1FAIL| Successful fit then batch_size0 failure leaves a different usable random model. Oracle accepts retaining completed outputs or invalidating inference. |
| `nonfinite_training_cannot_silently_report_success[nan/inf]` |2FAIL| Returns history with NaN train/val loss. Refusal or finite success accepted; no universal imputation policy. |
| `epoch_mse_weights_observations_not_batches` |1FAIL policy probe| Constant-zero decoder, lr0; training squared-value sum1596 over7rows→228. Current equal-batch mean236.708328. No changing-weights ambiguity. The public API does not explicitly promise observation weighting: this is a reporting difference/open metric contract, not an established optimizer bug. |
| `explicit_empty_hidden_configuration_is_not_replaced` |1FAIL| Wrapper [] creates4affine layers instead of1; source-level network supports empty hidden list. Explicit rejection or honoring [] accepted. |
| `observe_split_boundaries` N0/N1/f1/f2/f−.5 |5PASS observations| First four fail in NumPy/DataLoader; negative fraction trains using forced1validation row. This is recorded, not endorsed as a permanent domain contract. |
| `observe_zero_negative_epochs`0/−1 |2PASS observations| Empty histories, randomly initialized encode usable. Open domain/fitted-state semantics. |
| `observe_interpolation_empty_or_invalid_steps`0/−1 |2PASS observations| Mock decoder exposes empty trajectory for0, negative count refused by linspace. Does not validate real decoder on shape(0,). |
| `observe_duplicate_objects_cross_validation_boundary` |1PASS observation| Repeated synthetic object IDs occur on both sides of row split; existing#22, not a claim about real catalogs. |
| `observe_numeric_boundary_policies`NaN-label/length/empty/multirow |4PASS observations| NaN-label and empty summary warn then IndexError; wrong label length indexing error; multirow endpoints silently use first row. Open input policies, no invented alignment detection guarantee. |

Seven **compatibility controls PASS** in temporary source copies: preserve old
state; invalidate state; reject singleton batch explicitly; reject nonfinite
input; honor empty hidden list; weight epoch losses by sample counts; guard
class bases when Torch is absent. They target the relevant proof subsets and
show these oracles permit those remedies. No control is a production patch,
approved scientific policy, mutant kill or proof of general remediation.
Control results record their selected test counts and stdout separately.

All owned red files are archived on a durable ref before removal from the
integration tree. Check out this immutable archive SHA to replay. Ordinary
merge ancestry preserves it; never force-push/delete earlier proof archives.
The green matrix links corrective issues and this fixed archive after upload.
