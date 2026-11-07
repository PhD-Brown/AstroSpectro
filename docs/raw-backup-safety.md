# Raw backup and cleanup safety (#36)

`SpectraManager.interactive_cleaner()` requires `CONFIRM` and delegates to
`backup_and_clean_raw()`. Stop downloads and all other writers before use.
An empty source is a no-op, including when `raw_backup/` already exists.

The helper accepts absolute sibling source/backup paths. It rejects overlapping
paths, a source containing the working directory, symbolic links, Windows
reparse points (including junctions in ancestors or source entries), hard-linked
files and non-regular entries. The existing backup base is never removed or
reused. New backups are siblings named `raw_backup_<UTC timestamp>_<UUID>/`.

The operation inventories relative file paths, directories (including empty
ones), byte sizes and SHA-256, copies into a temporary `.raw_backup-*` sibling,
and compares both the copy and the source against the original inventory.
Enumeration/read errors abort the operation. Directory rename finalizes the
backup on the same filesystem, atomically where the filesystem supports it.
The source identity and both inventories are checked again before removing
source contents. The source directory itself stays in place.

Copy, verification and rename failures preserve the originals and older backups.
Temporary copies are removed when their safety checks and deletion succeed;
otherwise the exception identifies the retained temporary path and error.
A finalized backup is checked independently of the source. A copy that fails
this check is labelled `FAILED verification`, retained with an `_UNVERIFIED`
suffix when renaming succeeds, and explicitly excluded from recovery. If the
suffix cannot be applied, the diagnostic identifies the original path and the
marking error; its name alone does not establish validity. An interrupted check
has status `UNVERIFIED (reverification incomplete)`. If only the source changes,
the verified snapshot remains available but does not include later changes.
Deletion failures raise `Cleanup incomplete`, never success. Some originals
may already be missing: stop writers, inspect the remaining files and copy the
verified backup's contents into the source. Keep the backup until recovery is
checked. The widget displays the exception/recovery path, clears confirmation
and re-enables the button.

Interruptions such as `KeyboardInterrupt` and `SystemExit`, and unexpected
exceptions, keep their original type and instance and carry a Python 3.11
exception note with recovery paths. The widget prints these notes before
propagating cancellation. Interrupting a copy leaves its temporary location
explicitly UNVERIFIED; no long cleanup is started while handling cancellation.
An interruption during finalization reports both possible locations. Never use
a temporary or unverified copy for recovery. Interruption during deletion reports
`Cleanup incomplete` and the verified backup; some originals may be missing.

## Proof and review

All new tests in `tests/test_spectra_manager.py` use synthetic files under
`tmp_path`; none point to project data. Sockets remain disabled by pytest's
existing configuration. The two widget regression tests were run before the
fix against `35c2105a6373e9ca5820f08e33acd383e79afc06`: both failed because
the previous backup was lost, while originals survived the simulated copy
failure. They remain in the suite and pass with the fix.

The contracts cover empty sources, missing confirmation, immediate/partial copy
failures, missing/extra files, changed sizes, same-size corruption, missing empty
directories, successful cleanup after finalization, overlapping/invalid paths,
successive backups, rename failure, partial deletion and recovery, failed
temporary cleanup, enumeration errors, source/backup changes during the
operation, symbolic links, hard links and native Windows junctions. Link cases
skip explicitly on hosts that cannot create symlinks; native junction tests
are Windows-only. Command results and the tested commit are recorded in the PR.

The independent review of `bcc8232` led to additional contracts: refuse to
finalize a source changed during copying (M1); preserve a replacement source
directory even with identical contents (M3); detect changes during a file read
(M5); report an incomplete cleanup if files remain (M6); protect a source that
contains the CWD (M10). The repeated general path validation after rename (M8)
was removed as redundant: paths are already fixed absolute siblings, the final
manifests reject links/reparse points in roots and ancestors, and source identity
rejects directory substitution. A symlink substituted after rename is tested
through those remaining contracts. The notebook's cells 7 and 9 now direct
restoration to the exact verified path printed by the widget; outputs are
preserved. Further cases cover real widget success, interruptions in copy,
deletion, temporary cleanup and finalization, and invalid-backup marking errors.

```powershell
.venv/Scripts/python.exe -m pytest -q tests/test_spectra_manager.py
.venv/Scripts/python.exe -m pytest
.venv/Scripts/python.exe -m black --check src/tools/spectra_manager.py tests/test_spectra_manager.py
.venv/Scripts/python.exe -m ruff check src/tools/spectra_manager.py tests/test_spectra_manager.py
git diff --check
```

## Limits

This is neither a snapshot nor a lock against other processes. Rechecks detect
many changes but cannot eliminate the interval between checking and deleting;
concurrent writers or hostile path swaps are unsupported. Rename atomicity
depends on the filesystem; there is no power-loss/fsync durability guarantee.
Validation covers directory structure and regular file contents, not every
filesystem-specific metadata stream, ACL or extended attribute. Hashing adds
multiple full reads, and a complete extra copy requires sufficient free space.
Backups accumulate deliberately; retention and removal require human review.
No real cleanup, scientific pipeline or real-data recovery is validated by these
tests. Independent review and the scientific lead's merge decision remain.
