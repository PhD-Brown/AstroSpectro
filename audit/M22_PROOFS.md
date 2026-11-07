# M2.2 PCA — explicit proof archive

2026-11-07 · Refs #43. Production base:
`3604b3db8a66f19b07569f3c0aafb7f256734e77`, unchanged.
Keep this manifest, `tests/proofs_dimred_pca_analyzer.py`, and its matching
`tests/conftest.py` on the durable branch **`audit/43-dimred-pca-proofs`**.
The integration matrix and corrective issues supply the fixed archive SHA.
This file is intentionally outside the green integration diff.

From a checkout of that archive, using the installed Python 3.11 environment:

```powershell
python -m pytest -q tests/proofs_dimred_pca_analyzer.py --tb=line --junitxml=test-results/m22-proofs.xml
```

Executed Windows/Python 3.11.9, numpy 2.4.6, pandas 3.1.1, scipy 1.17.1,
sklearn 1.9.1, joblib 1.6.0, pytest 8.4.2, pytest-socket 0.7.0.
**21 cases: 16 FAIL / 5 PASS / 0 ERROR / 0 SKIP / 0 XFAIL**, exit 1.
All failures are test-call failures, not collection or fixture errors.
All inputs are synthetic; joblib payloads are generated locally in `tmp_path`.
No production patch or real scientific data is involved.

## Per-case disposition

Names below omit `test_`. A failure is not a count of independent bugs.
NEW means a demonstrated defect; OPEN means a proposed contract, not approved.

| Proof | Result | Classification / independent expectation |
|---|---|---|
| `unreachable_variance_cannot_return_nonexistent_component` | FAIL | NEW B1. K=1 retains 9/14; asking for 0.9 returns 2, outside the fitted basis. An explicit unavailable-target outcome is required; exception/sentinel API remains to decide. No claim that two components would actually reach the target. |
| `loaded_configuration_is_complete` | FAIL | NEW B2. `random_state=17`, `svd_solver=full` disappear after load. |
| `loaded_model_can_refit` | FAIL | Same B2. Calling public `fit` after load raises AttributeError for random_state. Inference alone remains green. |
| `mixed_numeric_metadata_correlations[bool/Int64/Float64]` | 3 FAIL | NEW B3. Individually numeric float + bool/nullable columns become an object matrix; `isfinite` crashes. The float column has perfect linear Pearson correlation regardless of the second column dtype. |
| `failed_fit_does_not_expose_an_unfitted_estimator_as_fitted[False/True]` | 2 FAIL | NEW B4. Failed initial fit/refit leaves a new unfitted PCA; the fitted accessor accepts it. Retaining old state versus clearing all state is OPEN, but returning an unfitted estimator as fitted is inconsistent. |
| `default_class_separation_on_two_component_model` | FAIL | NEW B5: supported fitted K=2 plus default n_pcs=5 gives an internal IndexError. The probe proposes a bounded result; **clamp versus explicit validation is OPEN**. It must be revised to the chosen contract in the corrective PR, not integrated as-is. |
| `reconstruction_rejects_unavailable_component_counts[-1/4]` | 2 FAIL | OPEN C1. Negative uses Python slicing; K+1 silently uses all fitted PCs. Proposes ValueError; this is not a pre-existing specified exception contract. |
| `empty_reconstruction_curve_retains_schema` | FAIL | OPEN C2. Returns a DataFrame with no columns. Schema preservation versus rejection remains to decide. |
| `unfitted_default_curve_has_domain_guard` | FAIL | B4 diagnostic inconsistency: default path dereferences None before the fit guard, whereas a non-empty explicit range reaches the guard. No numerical corruption claimed. |
| `feature_name_count_is_validated_at_fit` | FAIL | OPEN C3. Mismatched names accepted by fit then fail on DataFrame creation. Early validation proposed. |
| `training_cache_not_aliased_to_caller` | FAIL | OPEN C4. Reversing the caller's X after fit reverses implicit correlations because `_X_fit` is aliased. Estimator components remain unchanged. Defensive snapshot versus documented immutability requirement needs a decision. |
| `names_not_aliased_to_caller` | FAIL | OPEN C4. Mutating caller's list relabels loadings. Same ownership question; no evidence that source arrays are modified by fit itself. |
| `observe_positional_alignment_is_not_identity_alignment` | PASS | NOT A BUG under the current same-order precondition. Reversing only meta yields -1 instead of +1. No identity-aware join is promised. |
| `observe_loaded_implicit_correlations_need_training_data` | PASS | Documented limitation: `_X_fit` not persisted, implicit call raises ValueError; supplied scores work in green tests. Improved diagnostic optional. |
| `observe_negative_indices_duplicate_names_and_zero_components` | PASS | OPEN observations: top pc_idx=-1 labels PC0; duplicate feature labels are retained; K=0 fit/transform succeeds but sklearn inverse_transform rejects its (N,0) scores. Do not endorse these as desired API behavior. |
| `observe_incompatible_metadata_and_labels_fail` | PASS | Incompatible 35/34 rows fail loudly; y length mismatch fails loudly. Exact business diagnostics/label validation not certified. |
| `observe_controlled_unfitted_and_wrong_payloads` | PASS | Unfitted save is permitted but load fails; unfitted PCA/list payloads fail loudly. No transactional write or schema/version guarantee. |

## Corrections to exploratory assumptions

The first local run was **17 FAIL / 4 PASS**. Its zero-component observation
incorrectly expected inverse_transform to accept zero-width scores. Reproducing
the sklearn call showed a ValueError. That assumption was withdrawn; the
observation now records the actual limitation without asserting a desired
zero-component policy. It is **not an extra confirmed PCA bug**. No fixture or
collection error was counted as a demonstration. The final run above supersedes
the initial total; all other failures are retained and classified individually.

## Boundary

No tests here are automatic CI tests, no xfail is used, and no scientific
causality or astrophysical validation follows from synthetic correlations.
B1–B5 require separate fixes/reviews. C1–C4 and the other OPEN observations are
not silently accepted specifications. The green matrix records residual gaps.
