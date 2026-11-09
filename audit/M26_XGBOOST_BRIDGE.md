# M2.6 — XGBoostBridge scientific and technical audit

2026-11-09 · Refs #43. Initial inventory; execution ledger follows as work proceeds.
Tests and audit only. Independent Claude review and human merge decision required.

## Reference and method

Base/source `18302e68cf25289873aa170a27834d99b516f376`, bridge blob
`eab849e4bb80be03a50ebc9128d2737530262358` (entire source read).
Branch `audit/43-dimred-xgboost-bridge`, separate `AstroSpectro-m26` worktree.
Main checkout and M24/M25 worktrees inspected clean; main checkout remains
`795af2c`. Remote main verified by fetch and ls-remote. PR56/64/72/79/87 merges
and PR87 CI42548715421 success independently checked through GitHub.
No pre-existing dedicated bridge tests found. Earlier matrices M21–M25 are
context, not bridge evidence. Live AGENTS.md v1.3, complete #43 body, comment
history and latest handoffs, all issue bodies and comments for #21/#22/#23/
#24/#28/#38/#50/#51/#73 consulted. All-state issue inventory checked for duplicates.

Local Python3.11.9, NumPy2.4.6, pandas3.1.1, sklearn1.9.1, XGBoost3.2.0,
pytest8.4.2, Black24.4.2, Ruff0.5.0 match relevant Python3.11 lock entries.
Interpreter: `../AstroSpectro/.venv/Scripts/python.exe` (no installation edits).

## Initial significant-path inventory

AP = evidence on a precise contract; PA = partial; NA = not validated; B = blocked.
Initial statuses below are NA until the corresponding execution is recorded.

| Code path | Independent contract / risk | Level | Initial test plan | Status |
|---|---|---|---|---|
| imports, path insertion, utils fallback (186–200) | Correct source, distinguish mtime utility from lexical fallback; inspect global state | U/I | isolated source, import failures, repeat calls | NA |
| model discovery/loading (201–232) | No model/load failure gives fresh complete failure structure; selected artifact traceable | U/I | real latest_file, alternate classifier import, failures | NA |
| CSV resolution/read (237–247) | Explicit stem prioritized; lexical fallback characterized; parse errors visible | U | multiple files, missing/malformed CSV | NA |
| SNR (252–255) | Inclusive10, absent/NaN SNR; exact retained row values | U/P | boundary examples, matched prefiltered population | NA |
| candidate columns/dropna (262–271) | Independent names/dtypes/variance/missing fraction; no unrelated population assumptions | U/I | 0/10/>10% NaNs, exclusions, constants, nullable types | NA |
| alignment (279–295) | Every prediction associated to correct observation; all cardinalities checked | U/I/S | coherent permutations, middle removals, independent vector mismatch | NA |
| needed schema (298–316) | Names/order/values exact; distinguish compatibility from valid imputation | U/I | 0/30/>30% missing; original inputs unchanged | NA |
| inference/decode (323–332) | Model output decoded using saved mapping, no invented taxonomy | U/I | nonintuitive mapping, no encoder, explicit refusals | NA |
| probabilities (334–345) | Correct row maxima; unavailable marker; invalid output never valid confidence | U/P | analytic matrices, method absent/failing, malformed distributions | NA |
| result (347–371) | Stable complete keys, observed classes, F/G exact mask, no stale call state | U | full nominal result and second failure | NA |
| _empty_result (374–395) | Fresh empty schema, false success/probability flags | U | direct and each early-return path | NA |
| real pipeline/save/load | Same synthetic trusted saved object gives direct/bridge inference parity | I/P | miniature CPU XGBoost, named ColumnTransformer, joblib roundtrip | NA |

Priorities: observation identity and exact schema first, then outputs/failures,
real XGBoost, adversarial proofs, targeted mutations, final matrix and Draft PR.
No changes to production, dependencies, CI, notebooks, real models or data.

## Consumer boundaries inspected

`dimred/__init__.py` exports xgboost_predict but imports the known failing optional
Torch facade (#21). Source-direct unit imports do not validate the facade.
NB02 cells3/21/48 load PCA artifact y/meta/features_stem, call the bridge, and
pass results to plots/reporter; NB02 does not reconstruct the loader population.
The committed historical output reports42920 bridge rows versus43019 embedding
rows: this is historical output, not a measured current run or quantified harm.
`data_loader.load` has configurable SNR band/minimum, classes, balancing and
spectro_only; the bridge hardcodes r>=10 and an independent denylist.
Embedding preserves order provided by its caller. HDBSCAN profiles have their
own positional limitations (#73). Visualizer infers probability availability
from confidence!=1 rather than receiving has_proba. Reporter computes accuracy
when lengths agree, which cannot establish identity. These consumers were read,
not executed end-to-end; M2.7/M2.8 are not started.

## Scientific limits and open decisions

No IDs for Z/y/clusters enter the bridge API. Equal cardinality cannot prove
identity; independent same-length permutations/population replacement cannot be
detected from these inputs. Coherent population/order is a caller precondition.
Zero substitution is a current behavior, not a statistically justified policy.
Training/preprocessing (#24), group leakage (#22), feature provenance (#23),
XGBoost compatibility (#28), general persistence (#38), and real-catalogue,
calibration and astrophysical validation remain separate questions.
