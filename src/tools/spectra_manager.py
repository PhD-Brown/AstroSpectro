"""AstroSpectro — Spectra management (Notebook / CLI tools).

This module provides the ``SpectraManager`` class, a high-level utility for
driving spectra-related operations from a notebook (or script):

- orchestration of the **download** of new DR5 spectra via
  ``src/tools/dr5_downloader.py`` (with a progress bar and a **single session
  log**),
- lightweight *ops* actions: **cleanup / backup** of the ``data/raw/``
  directory,
- exploration shortcuts (e.g. path to the latest log, reading the last few
  lines, etc. — when implemented in this file).

Conventions
-----------
- Base paths (``RAW_DATA_DIR``, ``PROCESSED_DIR``, ``CATALOG_DIR``,
  ``MODELS_DIR``, ``LOGS_DIR``, ``TOOLS_DIR``, …) are supplied by
  ``utils.setup_project_env()``.
- The **DR5 downloader** is launched as a subprocess and accepts ``--limit``,
  ``--max-spectres``, ``--log-to``, ``--append`` and ``--progress`` to force
  the progress bar inside a notebook.  When ``--log-to`` is provided, a
  **single log file** is produced under ``logs/`` (no more “double log”).
- Jupyter widgets (``ipywidgets``) power the Notebook UX while remaining
  compatible with pure CLI execution.

Inputs / Outputs
----------------
Inputs :
    ``paths`` — dict of key project directories, typically returned by
    ``utils.setup_project_env()``.

Outputs / Side-effects :
    - ``.fits.gz`` files downloaded into ``data/raw/<PLAN>/``,
    - per-session text logs under ``logs/download_log_YYYYMMDDTHHMMSSZ.txt``,
    - optionally: a ``data/raw_backup/`` before cleaning ``data/raw/``.

Public API
----------
- ``SpectraManager(paths)`` — instantiate with the project environment.
- ``interactive_downloader()`` — widget to launch a DR5 download session
  (progress bar, single log in ``logs/``).
- ``interactive_cleaner()`` — widget to empty ``data/raw/`` with backup and
  confirmation.

Examples
--------
>>> from utils import setup_project_env
>>> from tools.spectra_manager import SpectraManager
>>> paths = setup_project_env()
>>> manager = SpectraManager(paths)
>>> # In a notebook:
>>> manager.interactive_downloader()
"""

from __future__ import annotations

import os
import re
import shlex
import sys
import shutil
import subprocess
from datetime import datetime, timezone

import ipywidgets as widgets
from IPython.display import display, clear_output, Markdown

PRESETS = {
    "cautious": dict(
        workers=8, scrape=4, fast=False, validate=True, per_plan=50, dry_run=False
    ),
    "balanced": dict(
        workers=16, scrape=8, fast=True, validate=True, per_plan=0, dry_run=False
    ),
    "max": dict(
        workers=24, scrape=12, fast=True, validate=False, per_plan=0, dry_run=False
    ),
}


class SpectraManager:
    """
    High-level manager for spectra-related operations (Notebook / CLI).

    This class serves as a facade for:
      1) launching **DR5 downloads** of new spectra from a notebook, with a
         **progress bar** and a **single session log**,
      2) running lightweight maintenance operations (e.g. **cleaning**
         ``data/raw/`` after a **backup**),
      3) providing a few log reading / access utilities.

    Attributes
    ----------
    paths : dict[str, str]
        Dictionary of project root paths.  Must contain at least:

        - ``'RAW_DATA_DIR'`` — raw spectra directory (``data/raw/``),
        - ``'LOGS_DIR'``     — log directory (``logs/``),
        - ``'TOOLS_DIR'``    — tools script directory (``src/tools/``).

    Parameters
    ----------
    paths : dict
        Project directory dictionary.  Must contain at least
        ``'RAW_DATA_DIR'``, ``'LOGS_DIR'``, ``'TOOLS_DIR'``.

    Raises
    ------
    KeyError
        If a required key is missing.
    """

    REQUIRED_KEYS = ("RAW_DATA_DIR", "LOGS_DIR", "TOOLS_DIR")

    # Pre-compiled regex to capture "XX%" in the progress line
    _RE_PCT = re.compile(r"(\d+)%")

    def __init__(self, paths: dict) -> None:
        """
        Initialise the manager with project paths.

        Parameters
        ----------
        paths : dict
            Path dictionary, typically returned by
            ``utils.setup_project_env()``.  Keys used here include:
            ``'RAW_DATA_DIR'``, ``'LOGS_DIR'``, ``'TOOLS_DIR'``.

        Raises
        ------
        KeyError
            If a required key is missing from *paths*.
        """
        for k in self.REQUIRED_KEYS:
            if k not in paths:
                raise KeyError(f"Missing key in paths: '{k}'")
        self.paths = paths

    # -------------------------------------------------------------------------
    # 1) Interactive download
    # -------------------------------------------------------------------------

    def interactive_downloader(self) -> None:
        """
        Launch a DR5 download from a notebook (widget + single session log).

        Display two numeric fields (number of plans, maximum spectra),
        a start button, a progress bar, and a live subprocess output
        preview.  The session writes **a single log file** under ``logs/``
        via the ``--log-to`` parameter forwarded to
        ``src/tools/dr5_downloader.py``.

        Notes
        -----
        - Builds and runs the command:
          ``python src/tools/dr5_downloader.py --limit <N> --max-spectres <M>
          --log-to <logs/...txt> --append``
        - Streams stdout from the subprocess to update the progress bar.
          Lines like ``Downloading: 60% | 298/500 [...]`` are detected.
        - On completion, displays the log path and the last few lines.
        - No “double log” is created widget-side: the script writes to the
          file provided via ``--log-to``.

        UI widgets
        ----------
        - **Plans** (IntText): number of plans (``--limit``).
        - **Spectra** (IntText): spectra cap (``--max-spectres``).
        - **Workers**: parallel downloads.
        - **Scrape**: parallelism during plan page analysis.
        - **Fast mode**: ``--delay 0`` + ``--chunk 131072`` (max throughput).
        - **Show progress bar** (Checkbox): forces ``--progress``.
        - **Per plan** ``<N>``: cap files per plan (after deduplication).
        - **Validate FITS**: quick FITS-header check on each download.
        - **Dry-run**: preview only, no disk writes.
        """

        # Basic widgets
        limit_widget = widgets.IntText(value=5, description="Plans:")
        max_spectra_widget = widgets.IntText(value=50, description="Spectres:")
        show_bar = widgets.Checkbox(value=True, description="Show progress bar (tqdm)")
        fast_mode = widgets.Checkbox(value=True, description="Fast mode")
        run_button = widgets.Button(
            description="Start download…",
            button_style="success",
            icon="download",
        )

        # Parallelism
        workers_widget = widgets.IntSlider(
            description="Workers", min=1, max=32, value=16
        )
        scrape_widget = widgets.IntSlider(description="Scrape", min=1, max=16, value=8)

        # Advanced options
        per_plan_widget = widgets.IntText(
            value=0, description="Per plan:"
        )  # 0 = unlimited
        validate_widget = widgets.Checkbox(value=False, description="Validate FITS")
        dry_run_widget = widgets.Checkbox(
            value=False, description="Dry-run (no writes)"
        )
        auto_widget = widgets.Checkbox(value=False, description="Auto‑throttle")
        maxw_widget = widgets.IntText(value=24, description="Max W:")
        minw_widget = widgets.IntText(value=4, description="Min W:")

        # Presets (with event-loop guard)
        preset = widgets.Dropdown(
            options=[
                ("Cautious (8/4)", "cautious"),
                ("Balanced (16/8)", "balanced"),
                ("Max (24/12)", "max"),
            ],
            value="balanced",
            description="Presets:",
        )
        applying_preset = False
        PRESETS = {
            "cautious": dict(
                workers=8,
                scrape=4,
                fast=False,
                validate=True,
                per_plan=50,
                dry_run=False,
            ),
            "balanced": dict(
                workers=16,
                scrape=8,
                fast=True,
                validate=True,
                per_plan=0,
                dry_run=False,
            ),
            "max": dict(
                workers=24,
                scrape=12,
                fast=True,
                validate=False,
                per_plan=0,
                dry_run=False,
            ),
        }

        def apply_preset(name: str) -> None:
            nonlocal applying_preset
            cfg = PRESETS.get(name)
            if not cfg:
                return
            applying_preset = True
            try:
                workers_widget.value = cfg["workers"]
                scrape_widget.value = cfg["scrape"]
                fast_mode.value = cfg["fast"]
                validate_widget.value = cfg["validate"]
                per_plan_widget.value = cfg["per_plan"]
                dry_run_widget.value = cfg["dry_run"]
            finally:
                applying_preset = False

        def on_preset_change(change):
            if change.get("name") != "value" or applying_preset:
                return
            apply_preset(change["new"])

        preset.observe(on_preset_change, names="value")
        apply_preset(preset.value)

        # Display widgets
        progress_display = widgets.HTML(value="")
        log_output = widgets.Output()

        line0 = widgets.HBox([preset])
        line1 = widgets.HBox([limit_widget, max_spectra_widget, show_bar, fast_mode])
        line2 = widgets.HBox([workers_widget, scrape_widget, auto_widget, maxw_widget])
        line3 = widgets.HBox(
            [per_plan_widget, validate_widget, dry_run_widget, minw_widget]
        )

        display(Markdown("### 1. Spectra Download"))
        display(
            Markdown(
                "Use the interface below to start a new download. Progress logs will be displayed in real time.\n**Tips**: Workers = parallel downloads; Scrape = parallelism for plan page analysis; Per plan = cap list per plan; Validate FITS = header check; Dry-run = preview without writing."
            )
        )
        display(line0, line1, line2, line3, run_button, progress_display, log_output)

        # ------- main callback -------------------------------------------------
        def on_run_clicked(_):
            progress_display.value = ""
            with log_output:
                clear_output()

            # Prepare paths / arguments
            script_path = os.path.join(self.paths["TOOLS_DIR"], "dr5_downloader.py")
            os.makedirs(self.paths["LOGS_DIR"], exist_ok=True)
            ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            log_path = os.path.join(self.paths["LOGS_DIR"], f"download_log_{ts}.txt")

            # Build CLI command
            cmd = [
                sys.executable,
                script_path,
                "--limit",
                str(limit_widget.value),
                "--max-spectres",
                str(max_spectra_widget.value),
                "--log-to",
                log_path,
                "--append",
                "--workers",
                str(workers_widget.value),
                "--scrape-workers",
                str(scrape_widget.value),
            ]
            if show_bar.value:
                cmd.append("--progress")
            if fast_mode.value:
                cmd += ["--delay", "0", "--chunk", "131072"]  # 128 KB
            if per_plan_widget.value and per_plan_widget.value > 0:
                cmd += ["--per-plan", str(per_plan_widget.value)]
            if validate_widget.value:
                cmd.append("--validate")
            if dry_run_widget.value:
                cmd.append("--dry-run")
            if auto_widget.value:
                cmd.append("--auto-throttle")
                if maxw_widget.value:
                    cmd += ["--max-workers", str(maxw_widget.value)]
                if "minw_widget" in locals() and minw_widget.value:
                    cmd += ["--min-workers", str(minw_widget.value)]

            # Show message + disable button (prevent double click)
            progress_display.value = "<b>Starting download...</b>"
            run_button.disabled = True

            try:
                print("Command:", shlex.join(cmd))
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    bufsize=1,
                    universal_newlines=True,
                    encoding="utf-8",
                    errors="replace",
                )
                if proc.stdout:
                    for line in iter(proc.stdout.readline, ""):
                        if "Downloading" in line and "%" in line:
                            clean = line.strip()
                            m = self._RE_PCT.search(clean)
                            pct = int(m.group(1)) if m else 0
                            progress_display.value = (
                                "<div style='font-family:monospace;white-space:pre'>"
                                + clean
                                + "</div>"
                                f"<progress value='{pct}' max='100' style='width:100%;height:18px'></progress>"
                            )
                        else:
                            with log_output:
                                print(line, end="")
                rc = proc.wait()
                if rc == 0:
                    if dry_run_widget.value:
                        progress_display.value = "<b>Dry-run complete (no writes).</b>"
                    else:
                        progress_display.value = (
                            "<b>Download completed successfully.</b>"
                        )
                else:
                    progress_display.value = (
                        "<b style='color:#c00'>Download failed.</b>"
                    )
                with log_output:
                    print(f"\nSession log written: {log_path}")
                    try:
                        with open(log_path, "r", encoding="utf-8") as f:
                            tail = f.readlines()[-15:]
                        print("\n--- Log excerpt (tail) ---")
                        for line in tail:
                            print(line, end="")
                    except Exception:
                        pass
            except Exception as e:
                progress_display.value = f"<b style='color:#c00'>Error: {e}</b>"
            finally:
                run_button.disabled = False

        run_button.on_click(on_run_clicked)

    # -------------------------------------------------------------------------
    # 2) Safe cleanup of data/raw (backup + confirmation)
    # -------------------------------------------------------------------------

    def interactive_cleaner(self) -> None:
        """
        Safely empty ``data/raw/`` (widget with confirmation + backup).

        Display a small interactive tool that:

        1. **backs up** the current ``data/raw/`` contents into
           ``data/raw_backup/`` (overwritten on each run),
        2. asks for **explicit confirmation**,
        3. then **deletes** the contents of ``data/raw/``.

        This is useful for starting fresh before a large download session
        while preventing accidental deletion.

        Raises
        ------
        OSError
            If a copy or deletion operation fails.
        """
        display(Markdown("### 2. Cleanup Option"))
        display(
            Markdown(
                "This option can empty `data/raw/` after making a backup copy into `data/raw_backup/`. **Type `CONFIRM`** then click the button."
            )
        )

        confirm = widgets.Text(placeholder="Type 'CONFIRM' here")
        clean_btn = widgets.Button(
            description="Clean the `raw` folder", button_style="danger", icon="trash"
        )
        out = widgets.Output()

        def on_clean(_) -> None:
            with out:
                clear_output()

                if confirm.value.strip() != "CONFIRM":
                    print("Incorrect confirmation. Cleanup cancelled.")
                    return

                raw_dir = self.paths["RAW_DATA_DIR"]
                data_dir = os.path.dirname(raw_dir)
                backup_dir = os.path.join(data_dir, "raw_backup")

                # Minimal safeguards
                if not os.path.isdir(raw_dir) or not os.path.isdir(data_dir):
                    print("Invalid paths; aborted.")
                    return
                if os.path.abspath(raw_dir) == os.path.abspath(backup_dir):
                    print("Backup and raw are identical — aborted.")
                    return

                try:
                    print(f"Backing up '{raw_dir}' -> '{backup_dir}'...")
                    # Remove old backup if it exists
                    if os.path.exists(backup_dir):
                        shutil.rmtree(backup_dir)

                    # Only copy if raw is not empty (otherwise backup is pointless)
                    if os.listdir(raw_dir):
                        shutil.copytree(raw_dir, backup_dir, dirs_exist_ok=True)
                    else:
                        print("(raw is empty, no backup needed)")

                    print("Cleaning 'raw'...")
                    shutil.rmtree(raw_dir, ignore_errors=True)
                    os.makedirs(raw_dir, exist_ok=True)

                    print("OK: 'raw' cleaned. (Backup available)")
                except Exception as e:  # noqa: BLE001
                    print(f"Error during cleanup: {e}")

        clean_btn.on_click(on_clean)

        display(
            widgets.VBox(
                [
                    widgets.Label("WARNING: Destructive action!"),
                    confirm,
                    clean_btn,
                    out,
                ]
            )
        )
