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

## Final revised snapshot

Final45cases: **38FAIL/7PASS/0ERROR/SKIP** on unchanged production. The added
R11 proof demonstrates a distinct reporter boundary: when a supplied aligned
reference is too short, an ordinary finite accuracy can silently use original
labels without recording the substitution. Missing/unavailable metric, explicit
reference provenance or preliminary relevant refusal are alternatives. This
does not choose a correction in green CI or prove external object identity.

Final replay reconstructs ten false repairs and twelve admissible controls.
Every selected candidate has its own original-source baseline, identical
collection, executed changed source, active LF hash and complete failure text.
**All ten false repairs rejected; all twelve alternatives accepted.** Positive
valid reports are included with the admissible variants. Added alternatives
exercise unique file IDs, an explicit incomplete-group marker and a reporter
log containing the metric plus the actual exception reason. Helper availability
structures can retain numeric metadata such as population n, while finite
replacement values remain forbidden.

R1 tables: seven cases, five failures/two supported missing representations.
R1 writer: twelve cases, ten failures/two explicit NaT-token successes.
The NaT token is classified as an identifiable missing value, not a real date;
the finite neighboring timestamp values must still match ISO dates exactly.
R2: five collision failures. R3: twelve failures and two legitimate complete
groups out of fourteen before/after injections (failure before first write
leaves A complete; failure after final close leaves B complete). R4 one failure,
R5 two failures, R6 one, R7 one, R11 one, plus one informative positive PASS.
These case counts are not bug counts.

The nonmonotone correlation oracle is rho.9 with paired ranks and1e-12 absolute
tolerance; historical31/35 and shifted.1 fail. Pairwise/complete-case policy is
open because the other axis is finite. No generic JSON refusal can masquerade
as a scientific coordinate validator or leave files behind.

New scripts/mutation controls exist only to validate evidence. Appended source
variants run inside owned TemporaryDirectory copies; source/notebooks/data in
the real worktree never change. The original51fe4d1 archive and original result
JSONs remain byte-identical. The final snapshot is committed/pushed on the
existing branch, then the proof file is removed in a later regular commit.
The final fixed SHA and replay command are linked from the current M27 matrix,
corrective issue qualifications and PR99; no new branch or history rewrite.

Final archive: `e0680e022ad08e621508913ed1b041640df48177` (ancestor), pushed and
read back on GitHub before normal removal. Committed replay38FAIL/7PASS,
zero ERROR/SKIP; R11 has its own #100. From the delivered head:
`python audit/check_m27_review.py --proofs-ref e0680e022ad08e621508913ed1b041640df48177 --only baseline`.
Without the selection flag, the scoped false-remedy/positive controls replay
as well. The fixed ref supplies proof+helper+fixture bytes, not just the proof.
