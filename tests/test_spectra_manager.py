"""Issue #36 safety contracts, using synthetic files exclusively under tmp_path."""

import importlib.util
import os
from pathlib import Path
import shutil

import pytest


spec = importlib.util.spec_from_file_location(
    "spectra_manager_under_test",
    Path(__file__).resolve().parents[1] / "src/tools/spectra_manager.py",
)
manager = importlib.util.module_from_spec(spec)
spec.loader.exec_module(manager)
pytestmark = pytest.mark.unit


@pytest.fixture
def folders(tmp_path):
    raw = tmp_path / "raw"
    old = tmp_path / "raw_backup"
    raw.mkdir()
    old.mkdir()
    (old / "previous.fits").write_bytes(b"previous observation")
    return raw, old


def populate(raw):
    (raw / "plan").mkdir()
    (raw / "plan" / "spectrum.fits").write_bytes(b"synthetic spectrum")
    (raw / "empty_plan").mkdir()
    (raw / "catalog.txt").write_bytes(b"synthetic catalog")


def snapshot(folder):
    return {
        str(path.relative_to(folder)): path.read_bytes() if path.is_file() else None
        for path in folder.rglob("*")
    }


def clean(raw, backup, confirmation="CONFIRM"):
    return manager.backup_and_clean_raw(raw, backup, confirmation=confirmation)


def assert_no_staging(raw):
    assert not list(raw.parent.glob(".raw_backup-*"))


def cleaner_widgets(monkeypatch, raw):
    displayed = []
    monkeypatch.setattr(manager, "display", lambda *items: displayed.extend(items))
    manager.SpectraManager(
        {"RAW_DATA_DIR": str(raw), "LOGS_DIR": str(raw.parent), "TOOLS_DIR": "unused"}
    ).interactive_cleaner()
    box = next(item for item in displayed if isinstance(item, manager.widgets.VBox))
    return box.children[1], box.children[2]


def test_widget_empty_source_keeps_previous_backup(folders, monkeypatch):
    raw, old = folders
    before = snapshot(old)
    confirmation, button = cleaner_widgets(monkeypatch, raw)
    confirmation.value = "CONFIRM"
    button.click()
    assert snapshot(old) == before
    assert list(raw.iterdir()) == []


def test_widget_partial_copy_keeps_previous_backup(folders, monkeypatch, capsys):
    raw, old = folders
    populate(raw)
    original, previous = snapshot(raw), snapshot(old)

    def fail_copy(source, target, **kwargs):
        Path(target).mkdir(exist_ok=True)
        (Path(target) / "partial").write_bytes(b"partial")
        raise OSError("simulated disk full")

    monkeypatch.setattr(manager.shutil, "copytree", fail_copy)
    confirmation, button = cleaner_widgets(monkeypatch, raw)
    confirmation.value = "CONFIRM"
    button.click()
    assert snapshot(raw) == original
    assert snapshot(old) == previous
    assert "simulated disk full" in capsys.readouterr().out
    assert_no_staging(raw)


def test_empty_source_is_a_noop(folders):
    raw, old = folders
    previous = snapshot(old)
    assert clean(raw, old) is None
    assert snapshot(old) == previous
    assert set(raw.parent.iterdir()) == {raw, old}


@pytest.mark.parametrize("confirmation", [None, "", "confirm", True, " CONFIRM "])
def test_confirmation_required_before_io(folders, confirmation, monkeypatch):
    raw, old = folders
    populate(raw)
    original = snapshot(raw)
    monkeypatch.setattr(
        manager.shutil, "copytree", lambda *a, **k: pytest.fail("copy not authorized")
    )
    with pytest.raises(ValueError, match="CONFIRM"):
        clean(raw, old, confirmation)
    assert snapshot(raw) == original
    assert_no_staging(raw)


@pytest.mark.parametrize("partial", [False, True])
def test_copy_failure_preserves_originals_and_previous_backup(
    folders, monkeypatch, partial
):
    raw, old = folders
    populate(raw)
    original, previous = snapshot(raw), snapshot(old)

    def fail_copy(source, target, **kwargs):
        if partial:
            (Path(target) / "partial").write_bytes(b"partial")
        raise PermissionError("simulated copy failure")

    monkeypatch.setattr(manager.shutil, "copytree", fail_copy)
    with pytest.raises(RuntimeError, match="Originals preserved"):
        clean(raw, old)
    assert snapshot(raw) == original
    assert snapshot(old) == previous
    assert_no_staging(raw)


@pytest.mark.parametrize(
    "damage", ["missing", "size", "same_size", "extra", "directory"]
)
def test_invalid_copy_never_authorizes_deletion(folders, monkeypatch, damage):
    raw, old = folders
    populate(raw)
    original, previous = snapshot(raw), snapshot(old)
    copytree = shutil.copytree

    def damaged_copy(source, target, *args, **kwargs):
        copytree(source, target, *args, **kwargs)
        if Path(source) != raw:
            return
        target = Path(target)
        if damage == "missing":
            (target / "catalog.txt").unlink()
        elif damage == "size":
            (target / "catalog.txt").write_bytes(b"short")
        elif damage == "same_size":
            (target / "catalog.txt").write_bytes(b"X" * len(b"synthetic catalog"))
        elif damage == "extra":
            (target / "unexpected").write_bytes(b"extra")
        else:
            (target / "empty_plan").rmdir()

    monkeypatch.setattr(manager.shutil, "copytree", damaged_copy)
    with pytest.raises(RuntimeError, match="Originals preserved"):
        clean(raw, old)
    assert snapshot(raw) == original
    assert snapshot(old) == previous
    assert_no_staging(raw)


def test_valid_copy_is_finalized_before_deletion(folders, monkeypatch):
    raw, old = folders
    populate(raw)
    original, previous = snapshot(raw), snapshot(old)
    rmtree = shutil.rmtree
    unlink = Path.unlink
    seen = []

    def check_backup():
        backups = list(raw.parent.glob("raw_backup_*"))
        assert len(backups) == 1
        assert snapshot(backups[0]) == original
        assert_no_staging(raw)
        seen.append(True)

    def checked_rmtree(path, **kwargs):
        if Path(path).is_relative_to(raw):
            check_backup()
        return rmtree(path, **kwargs)

    def checked_unlink(path, **kwargs):
        if path.is_relative_to(raw):
            check_backup()
        return unlink(path, **kwargs)

    monkeypatch.setattr(manager.shutil, "rmtree", checked_rmtree)
    monkeypatch.setattr(Path, "unlink", checked_unlink)
    backup = clean(raw, old)
    assert backup.parent == old.parent
    assert backup != old
    assert seen
    assert snapshot(backup) == original
    assert snapshot(old) == previous
    assert raw.is_dir() and not list(raw.iterdir())


@pytest.mark.parametrize("relationship", ["same", "inside", "ancestor", "relative"])
def test_unsafe_paths_rejected_before_copy(folders, monkeypatch, relationship):
    raw, old = folders
    populate(raw)
    original, previous = snapshot(raw), snapshot(old)
    target = {
        "same": raw,
        "inside": raw / "backup",
        "ancestor": raw.parent,
        "relative": Path("raw_backup"),
    }[relationship]
    monkeypatch.setattr(
        manager.shutil, "copytree", lambda *a, **k: pytest.fail("unsafe copy")
    )
    with pytest.raises(ValueError):
        clean(raw, target)
    assert snapshot(raw) == original
    assert snapshot(old) == previous


def test_successive_operations_keep_all_backups(folders):
    raw, old = folders
    populate(raw)
    first_contents, previous = snapshot(raw), snapshot(old)
    first = clean(raw, old)
    (raw / "second.fits").write_bytes(b"second observation")
    second_contents = snapshot(raw)
    second = clean(raw, old)
    assert first != second
    assert snapshot(first) == first_contents
    assert snapshot(second) == second_contents
    assert snapshot(old) == previous


def test_finalization_failure_preserves_originals(folders, monkeypatch):
    raw, old = folders
    populate(raw)
    original, previous = snapshot(raw), snapshot(old)

    def fail_rename(*args):
        raise OSError("simulated rename failure")

    monkeypatch.setattr(Path, "rename", fail_rename)
    with pytest.raises(RuntimeError, match="Originals preserved"):
        clean(raw, old)
    assert snapshot(raw) == original
    assert snapshot(old) == previous
    assert_no_staging(raw)


def test_partial_deletion_reports_recovery_from_valid_backup(folders, monkeypatch):
    raw, old = folders
    populate(raw)
    original, previous = snapshot(raw), snapshot(old)
    rmtree = shutil.rmtree

    def fail_delete(path, **kwargs):
        path = Path(path)
        if path.is_relative_to(raw):
            (raw / "catalog.txt").unlink(missing_ok=True)
            raise PermissionError("simulated locked file")
        return rmtree(path, **kwargs)

    monkeypatch.setattr(manager.shutil, "rmtree", fail_delete)
    with pytest.raises(RuntimeError, match="Cleanup incomplete") as error:
        clean(raw, old)
    backup = next(raw.parent.glob("raw_backup_*"))
    assert snapshot(backup) == original
    assert snapshot(old) == previous
    assert not (raw / "catalog.txt").exists()
    assert str(backup) in str(error.value)
    assert str(raw) in str(error.value)
    assert "Restore" in str(error.value)
    assert "simulated locked file" in str(error.value)
    assert_no_staging(raw)


def test_staging_cleanup_failure_is_explicit(folders, monkeypatch):
    raw, old = folders
    populate(raw)
    original = snapshot(raw)

    def fail(*args, **kwargs):
        raise PermissionError("simulated permissions")

    monkeypatch.setattr(manager.shutil, "copytree", fail)
    monkeypatch.setattr(manager.shutil, "rmtree", fail)
    with pytest.raises(RuntimeError, match="Temporary backup retained at") as error:
        clean(raw, old)
    staging = next(raw.parent.glob(".raw_backup-*"))
    assert str(staging) in str(error.value)
    assert snapshot(raw) == original


def test_source_changed_during_copy_aborts(folders, monkeypatch):
    raw, old = folders
    populate(raw)
    copytree = shutil.copytree

    def changing_copy(source, target, *args, **kwargs):
        copytree(source, target, *args, **kwargs)
        if Path(source) == raw:
            (raw / "new.fits").write_bytes(b"new observation")

    monkeypatch.setattr(manager.shutil, "copytree", changing_copy)
    monkeypatch.setattr(
        Path, "rename", lambda *a, **k: pytest.fail("Changed source must not finalize")
    )
    with pytest.raises(RuntimeError, match="Originals preserved"):
        clean(raw, old)
    assert (raw / "new.fits").read_bytes() == b"new observation"
    assert_no_staging(raw)


@pytest.mark.parametrize("location", ["source", "target", "ancestor", "file"])
def test_symbolic_links_rejected(folders, tmp_path, location):
    raw, old = folders
    populate(raw)
    alias = tmp_path / "alias"
    destination = raw if location in {"source", "file"} else old
    if location == "ancestor":
        destination = tmp_path
    try:
        alias.symlink_to(destination, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"Symbolic links unavailable: {error}")
    if location == "file":
        (raw / "linked.fits").symlink_to(old / "previous.fits")
    source = alias if location == "source" else raw
    target = alias if location == "target" else old
    if location == "ancestor":
        source, target = alias / "raw", alias / "raw_backup"
    with pytest.raises(ValueError, match="[Ll]ink|[Rr]eparse"):
        clean(source, target)
    assert (raw / "catalog.txt").read_bytes() == b"synthetic catalog"
    assert (old / "previous.fits").read_bytes() == b"previous observation"
    assert_no_staging(raw)


def test_hard_link_rejected(folders):
    raw, old = folders
    os.link(old / "previous.fits", raw / "linked.fits")
    with pytest.raises(ValueError, match="[Ll]ink"):
        clean(raw, old)
    assert (raw / "linked.fits").read_bytes() == b"previous observation"


def test_widget_requires_confirmation_and_reports_failure(folders, monkeypatch, capsys):
    raw, old = folders
    populate(raw)
    calls = []

    def fail(source, target, *, confirmation):
        calls.append((source, target, confirmation))
        raise RuntimeError("Cleanup incomplete. Restore from verified backup HERE")

    monkeypatch.setattr(manager, "backup_and_clean_raw", fail)
    confirmation, button = cleaner_widgets(monkeypatch, raw)
    button.click()
    assert not calls
    confirmation.value = "CONFIRM"
    button.click()
    output = capsys.readouterr().out
    assert calls and calls[0][2] == "CONFIRM"
    assert "Restore from verified backup HERE" in output
    assert "OK:" not in output
    assert not button.disabled
    assert confirmation.value == ""


@pytest.mark.parametrize("problem", ["missing_source", "file_source", "file_backup"])
def test_invalid_paths_do_not_start_copy(folders, monkeypatch, problem):
    raw, old = folders
    if problem == "missing_source":
        raw = raw.parent / "missing"
    elif problem == "file_source":
        raw.rmdir()
        raw.write_bytes(b"not a directory")
    else:
        old = old.parent / "backup_file"
        old.write_bytes(b"not a directory")
    monkeypatch.setattr(
        manager.shutil, "copytree", lambda *a, **k: pytest.fail("invalid path copy")
    )
    with pytest.raises((ValueError, RuntimeError)):
        clean(raw, old)
    assert_no_staging(raw)


@pytest.mark.parametrize("phase", ["preflight", "verification"])
def test_directory_enumeration_error_cannot_be_hidden(folders, monkeypatch, phase):
    raw, old = folders
    populate(raw)
    original = snapshot(raw)
    scandir = os.scandir

    def fail_scandir(path):
        path = Path(path)
        if (phase == "preflight" and path == raw / "plan") or (
            phase == "verification" and path.name.startswith(".raw_backup-")
        ):
            raise PermissionError("simulated enumeration denied")
        return scandir(path)

    with monkeypatch.context() as patch:
        patch.setattr(manager.os, "scandir", fail_scandir)
        with pytest.raises(RuntimeError, match="[Cc]leanup not started") as error:
            clean(raw, old)
        assert "simulated enumeration denied" in str(error.value)
    assert snapshot(raw) == original


def test_no_previous_backup_is_required(folders):
    raw, _ = folders
    (raw / "synthetic.fits").write_bytes(b"synthetic")
    backup = clean(raw, raw.parent / "new_backup")
    assert (backup / "synthetic.fits").read_bytes() == b"synthetic"
    assert not list(raw.iterdir())


@pytest.mark.parametrize("change", ["source", "backup", "both"])
def test_change_after_finalization_prevents_deletion(folders, monkeypatch, change):
    raw, old = folders
    populate(raw)
    rename = Path.rename

    def changing_rename(path, target):
        result = rename(path, target)
        changed = raw if change == "source" else Path(target)
        (changed / "catalog.txt").write_bytes(b"changed after rename")
        if change == "both":
            (raw / "catalog.txt").write_bytes(b"source also changed")
        return result

    monkeypatch.setattr(Path, "rename", changing_rename)
    with pytest.raises(RuntimeError, match="Originals preserved") as error:
        clean(raw, old)
    backup = next(raw.parent.glob("raw_backup_*"))
    assert str(backup) in str(error.value)
    assert (raw / "plan" / "spectrum.fits").read_bytes() == b"synthetic spectrum"
    if change in {"backup", "both"}:
        assert "FAILED verification" in str(error.value)
        assert "do not use for recovery" in str(error.value)
        assert backup.name.endswith("_UNVERIFIED")
    else:
        assert "Source contents changed" in str(error.value)
        assert "Verified snapshot" in str(error.value)
        assert not backup.name.endswith("_UNVERIFIED")
    assert_no_staging(raw)


@pytest.mark.skipif(os.name != "nt", reason="Windows junction contract")
@pytest.mark.parametrize("location", ["source", "target", "entry"])
def test_windows_junction_rejected(folders, location):
    import _winapi

    raw, old = folders
    populate(raw)
    junction = (raw if location == "entry" else raw.parent) / "junction"
    _winapi.CreateJunction(str(old), str(junction))
    source = junction if location == "source" else raw
    target = junction if location == "target" else old
    with pytest.raises(ValueError, match="[Rr]eparse"):
        clean(source, target)
    assert (old / "previous.fits").read_bytes() == b"previous observation"
    assert_no_staging(raw)


@pytest.mark.parametrize("phase", ["copy", "deletion"])
@pytest.mark.parametrize(
    "exception_type", [KeyboardInterrupt, SystemExit, RuntimeError]
)
def test_interruption_has_recovery_note(folders, monkeypatch, phase, exception_type):
    raw, old = folders
    populate(raw)
    original, previous = snapshot(raw), snapshot(old)
    interruption = exception_type("simulated interruption")

    def interrupt(*args, **kwargs):
        if phase == "copy":
            (Path(args[1]) / "partial").write_bytes(b"partial")
        else:
            (raw / "catalog.txt").unlink(missing_ok=True)
        raise interruption

    target = "copytree" if phase == "copy" else "rmtree"
    monkeypatch.setattr(manager.shutil, target, interrupt)
    with pytest.raises(exception_type) as caught:
        clean(raw, old)
    assert caught.value is interruption  # Preserve cancellation/exit semantics.
    diagnostic = "\n".join(getattr(caught.value, "__notes__", []))
    assert str(raw) in diagnostic
    assert snapshot(old) == previous
    if phase == "copy":
        assert snapshot(raw) == original
        staging = next(raw.parent.glob(".raw_backup-*"))
        assert str(staging) in diagnostic
        assert "UNVERIFIED" in diagnostic
        assert "do not use for recovery" in diagnostic
        assert "Originals preserved" in diagnostic
    else:
        backup = next(raw.parent.glob("raw_backup_*"))
        assert snapshot(backup) == original
        assert not (raw / "catalog.txt").exists()
        assert str(backup) in diagnostic
        assert "Cleanup incomplete" in diagnostic
        assert "Restore" in diagnostic


def test_source_replaced_by_identical_directory_is_not_deleted(folders, monkeypatch):
    raw, old = folders
    populate(raw)
    original = snapshot(raw)
    rename = Path.rename
    moved = raw.parent / "original_source"

    def replacing_rename(path, target):
        result = rename(path, target)
        if path.name.startswith(".raw_backup-"):
            rename(raw, moved)
            shutil.copytree(moved, raw)
        return result

    monkeypatch.setattr(Path, "rename", replacing_rename)
    with pytest.raises(RuntimeError, match="[Ss]ource.*changed"):
        clean(raw, old)
    assert snapshot(raw) == original
    assert snapshot(moved) == original
    assert_no_staging(raw)


def test_file_changed_while_hashing_is_rejected(folders, monkeypatch):
    raw, old = folders
    populate(raw)
    previous = snapshot(old)
    file_digest = manager.hashlib.file_digest
    target = raw / "catalog.txt"

    def changing_digest(stream, algorithm):
        result = file_digest(stream, algorithm)
        if Path(stream.name) == target:
            before = target.stat()
            target.write_bytes(b"X" * before.st_size)
            os.utime(
                target, ns=(before.st_atime_ns, before.st_mtime_ns + 2_000_000_000)
            )
        return result

    monkeypatch.setattr(manager.hashlib, "file_digest", changing_digest)
    with pytest.raises(ValueError, match="changed while reading"):
        clean(raw, old)
    assert target.read_bytes() == b"X" * len(b"synthetic catalog")
    assert (raw / "plan" / "spectrum.fits").read_bytes() == b"synthetic spectrum"
    assert snapshot(old) == previous
    assert_no_staging(raw)


def test_nonempty_source_after_cleanup_reports_failure(folders, monkeypatch):
    raw, old = folders
    (raw / "spectrum.fits").write_bytes(b"synthetic")
    original = snapshot(raw)
    unlink = Path.unlink

    def recreating_unlink(path, **kwargs):
        unlink(path, **kwargs)
        if path.parent == raw:
            path.write_bytes(b"synthetic")

    monkeypatch.setattr(Path, "unlink", recreating_unlink)
    with pytest.raises(RuntimeError, match="Cleanup incomplete"):
        clean(raw, old)
    assert snapshot(raw) == original
    assert snapshot(next(raw.parent.glob("raw_backup_*"))) == original


def test_source_containing_cwd_rejected_before_copy(folders, monkeypatch):
    raw, old = folders
    populate(raw)
    original = snapshot(raw)
    monkeypatch.chdir(raw / "plan")
    monkeypatch.setattr(
        manager.shutil, "copytree", lambda *a, **k: pytest.fail("CWD must be protected")
    )
    with pytest.raises(ValueError, match="working directory"):
        clean(raw, old)
    assert snapshot(raw) == original
    assert_no_staging(raw)


def test_source_link_substituted_after_rename_is_rejected(folders, monkeypatch):
    raw, old = folders
    populate(raw)
    original = snapshot(raw)
    rename = Path.rename
    moved = raw.parent / "original_source"
    probe = raw.parent / "link_probe"
    try:
        probe.symlink_to(raw, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"Symbolic links unavailable: {error}")
    probe.unlink()

    def linked_rename(path, target):
        result = rename(path, target)
        if path.name.startswith(".raw_backup-"):
            rename(raw, moved)
            raw.symlink_to(moved, target_is_directory=True)
        return result

    monkeypatch.setattr(Path, "rename", linked_rename)
    with pytest.raises(RuntimeError, match="[Ll]ink|[Rr]eparse"):
        clean(raw, old)
    assert snapshot(moved) == original
    assert_no_staging(raw)


def test_widget_success_prints_real_verified_backup(folders, monkeypatch, capsys):
    raw, old = folders
    populate(raw)
    original = snapshot(raw)
    confirmation, button = cleaner_widgets(monkeypatch, raw)
    confirmation.value = "CONFIRM"
    button.click()
    backup = next(raw.parent.glob("raw_backup_*"))
    output = capsys.readouterr().out
    assert str(backup) in output
    assert "OK:" in output
    assert snapshot(backup) == original
    assert not list(raw.iterdir())


def test_widget_interruption_displays_recovery_and_propagates(
    folders, monkeypatch, capsys
):
    raw, old = folders
    populate(raw)

    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt()

    monkeypatch.setattr(manager.shutil, "rmtree", interrupt)
    confirmation, button = cleaner_widgets(monkeypatch, raw)
    confirmation.value = "CONFIRM"
    with pytest.raises(KeyboardInterrupt):
        button.click()
    backup = next(raw.parent.glob("raw_backup_*"))
    output = capsys.readouterr().out
    assert "Cleanup incomplete" in output
    assert str(backup) in output
    assert "OK:" not in output
    assert not button.disabled
    assert confirmation.value == ""


def test_failed_backup_marking_error_preserves_invalid_status(folders, monkeypatch):
    raw, old = folders
    populate(raw)
    original = snapshot(raw)
    rename = Path.rename

    def corrupt_then_refuse_marking(path, target):
        if not path.name.startswith(".raw_backup-"):
            raise PermissionError("simulated marking denied")
        result = rename(path, target)
        (Path(target) / "catalog.txt").write_bytes(b"corrupted")
        return result

    monkeypatch.setattr(Path, "rename", corrupt_then_refuse_marking)
    with pytest.raises(RuntimeError) as error:
        clean(raw, old)
    backup = next(raw.parent.glob("raw_backup_*"))
    assert str(backup) in str(error.value)
    assert "FAILED verification" in str(error.value)
    assert "do not use for recovery" in str(error.value)
    assert "simulated marking denied" in str(error.value)
    assert "Verified snapshot" not in str(error.value)
    assert snapshot(raw) == original


def test_reverification_interruption_never_claims_valid_backup(folders, monkeypatch):
    raw, old = folders
    populate(raw)
    original = snapshot(raw)
    file_digest = manager.hashlib.file_digest

    def interrupt_finalized_digest(stream, algorithm):
        if Path(stream.name).relative_to(raw.parent).parts[0].startswith("raw_backup_"):
            raise KeyboardInterrupt()
        return file_digest(stream, algorithm)

    monkeypatch.setattr(manager.hashlib, "file_digest", interrupt_finalized_digest)
    with pytest.raises(KeyboardInterrupt) as error:
        clean(raw, old)
    backup = next(raw.parent.glob("raw_backup_*"))
    diagnostic = "\n".join(error.value.__notes__)
    assert str(backup) in diagnostic
    assert "UNVERIFIED (reverification incomplete)" in diagnostic
    assert "do not use for recovery" in diagnostic
    assert "Verified snapshot" not in diagnostic
    assert snapshot(raw) == original


def test_interruption_during_staging_cleanup_has_diagnostic(folders, monkeypatch):
    raw, old = folders
    populate(raw)
    original = snapshot(raw)

    def fail_copy(*args, **kwargs):
        raise OSError("simulated copy failure")

    def interrupt_cleanup(*args, **kwargs):
        raise KeyboardInterrupt()

    monkeypatch.setattr(manager.shutil, "copytree", fail_copy)
    monkeypatch.setattr(manager.shutil, "rmtree", interrupt_cleanup)
    with pytest.raises(KeyboardInterrupt) as error:
        clean(raw, old)
    staging = next(raw.parent.glob(".raw_backup-*"))
    diagnostic = "\n".join(error.value.__notes__)
    assert str(staging) in diagnostic
    assert "UNVERIFIED" in diagnostic
    assert "Originals preserved" in diagnostic
    assert snapshot(raw) == original


def test_interruption_just_after_rename_reports_both_possible_paths(
    folders, monkeypatch
):
    raw, old = folders
    populate(raw)
    original = snapshot(raw)
    rename = Path.rename

    def interrupt_after_rename(path, target):
        rename(path, target)
        raise KeyboardInterrupt()

    monkeypatch.setattr(Path, "rename", interrupt_after_rename)
    with pytest.raises(KeyboardInterrupt) as error:
        clean(raw, old)
    backup = next(raw.parent.glob("raw_backup_*"))
    diagnostic = "\n".join(error.value.__notes__)
    assert str(backup) in diagnostic
    assert "UNVERIFIED" in diagnostic
    assert "do not use for recovery" in diagnostic
    assert snapshot(raw) == original
