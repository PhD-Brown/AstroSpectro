# M2.7 — revised evidence after Claude R1–R12

2026-11-09 client date · Refs #43 / Draft PR99.
Review5835559133, twelve open threads, inspected at9666d7d.
Production base1e22663, blob b966d9b90f056baa37c0ccbb68135c71ff65de92 unchanged.
Original archive51fe4d1 and every original result ledger remain untouched.

## Checkpoint A: R1–R7

New explicit file `tests/proofs_dimred_run_reporter_review.py`, shared bounded
invariants `tests/reporter_review_contracts.py`, replay
`audit/check_m27_review.py`. Only synthetic temporary files, blocked sockets,
trusted joblib created by the tests. No production or notebook change.

Initial revised baseline: **37FAIL/7PASS/0ERROR/SKIP**,44cases. Counts are
properties of these examples, not bug totals. Seven table variants, twelve
writer cases (fresh/preexisting destinations), five collision APIs, fourteen
before/after write failures, provenance, two metric failures, finite-pair
identity, one-axis support and a positive valid report.

All ten reconstructed false remedies are rejected by assertion or explicit
pytest failure: F1 zeros, F8 late JSON refusal, F2 aliases before collision,
F3 rollback without legacy, F4/F4b fabricated model, F5/F5c empty/unrelated
diagnostic, F6 shifted finite pairs, F7 duplicated absent coordinate.
Each exact selected population has an original-source baseline and matching
collection. Candidate source and executed-line traces are in the ledger.

Nine separate controls pass their selected revised proofs plus a positive:
F9 RuntimeError collision before writing; F10 complete rollback and chained
RuntimeError; F4c model unavailable/figure retained; F11 propagated original
metric cause; F6c finite pairs; F6refuse preliminary scientific refusal;
F7c actual one-axis report; F7refuse preliminary dimension refusal; R1safe
strict preflight before opening any JSON target. No general production policy
is selected by these copies. R1safe is a writer control, not a claim that all
table conversion paths are repaired.

R3 compares population labels/counts, exact coordinates, parsed TXT counts,
JSON and joblib timestamps, ordinary aliases and legacy. Timestamped orphan
files are also checked. An explicit incomplete marker can be recognized if
it links every affected alias; arbitrary future manifest schemas require
review of that adapter. Complete A, complete B or explicit unavailability
are alternatives. Python exceptions do not simulate power loss.

Twenty green helper controls reject fictitious zero values, disguised model
paths, key-only diagnostics, unrelated logger records and lost exception
causes; admit explicit missing values, structured/logged causes, chained
exceptions and whitespace-only TXT changes. Existing reporter84 tests still
pass (`-W error`), combined104PASS. This checkpoint is not the final archive;
R8–R12 and final immutable proof SHA follow after fixture stabilization.
