"""AstroSpectro — Smart, resumable LAMOST DR5 spectra downloader.

This module provides a *smart* downloader for LAMOST DR5 spectra.  It:

- reads a list of valid plans from ``data/catalog/valid_plan_urls.csv``,
- excludes plans already completed (ledger ``data/catalog/downloaded_plans.csv``),
- scrapes ``.fits.gz`` links for each plan page and builds a download queue,
- downloads in **round-robin** order (balancing across plans),
- resumes cleanly: existing files are skipped; writes use an atomic
  ``.part`` → ``os.replace()`` strategy,
- maintains a **session log** under ``logs/download_log_YYYYMMDDTHHMMSSZ.txt``,
- marks fully completed plans at the end of the session.

Conventions
-----------
- Project tree (resolved relative to this file):

  - ``data/catalog/valid_plan_urls.csv``   → columns: ``url``
  - ``data/catalog/downloaded_plans.csv``  → columns: ``url``
  - ``data/raw/<plan>/<filename>.fits.gz`` → destination of each file
  - ``logs/*.txt``                         → session logs

- All lengths / sizes use standard SI units.

Inputs / Outputs
-----------------
Inputs : ``valid_plan_urls.csv`` and ``downloaded_plans.csv`` (optional).
Outputs : ``.fits.gz`` files under ``data/raw/`` + one session log file.

Interface
---------
- Python API: ``SmartDownloader(limit_plans, max_spectra, ...).run()``
- CLI: ``python -m src.tools.dr5_downloader --limit 5 --max-spectres 200 --progress``

Examples
--------
>>> dl = SmartDownloader(limit_plans=3, max_spectra=120, progress=True)
>>> dl.run()

Dependencies
------------
``requests``, ``pandas``, ``beautifulsoup4``, ``tqdm``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import zip_longest
from typing import Iterable, List, Optional

import argparse
import os
import sys
import gzip
import random
import math
import statistics
import time
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

__all__ = ["SmartDownloader", "main"]
__version__ = "3.1.0"


@dataclass(slots=True)
class _Paths:
    """Container for paths used by the downloader.

    Paths are resolved relative to this file’s location so that execution
    is reliable from both notebooks and the terminal.
    """

    catalog_dir: str
    raw_data_dir: str
    logs_dir: str


class SmartDownloader:
    """Smart, resumable downloader for LAMOST DR5 spectra.

    Phases
    ------
    1. **Scrape** plan pages → collect ``.fits.gz`` links.
    2. **Round-robin download** (balanced across plans), optional ``tqdm``
       progress bar.
    3. **State update**: mark newly completed plans.

    Parameters
    ----------
    limit_plans : int or None
        Maximum number of **new** plans to process (``None`` = unlimited).
    max_spectra : int or None
        Early stop after this total number of spectra (``None`` = unlimited).
    timeout : int, default=60
        Timeout in seconds applied to HTTP requests.
    delay_between : float, default=0.2
        Short delay in seconds between consecutive downloads (set 0 to go faster).
    retries : int, default=3
        Number of network retries (exponential backoff factor 0.8).
    progress : bool, default=True
        Show a ``tqdm`` progress bar (force-enable in notebooks if needed).
    log_to : str or None
        Path to the session log file (auto-generated if ``None``).
    append : bool, default=False
        If ``True``, append to an existing log instead of overwriting it.
    chunk_size : int, default=65536
        Write chunk size in bytes. 65 536 = 64 kB, 131 072 = 128 kB, etc.
    """

    # --------------------------- initialisation & paths ----------------------------

    def __init__(
        self,
        limit_plans: Optional[int],
        max_spectra: Optional[int],
        *,
        timeout: int = 60,
        delay_between: float = 0.2,
        retries: int = 3,
        progress: bool = True,
        log_to: Optional[str] = None,
        append: bool = False,
        chunk_size: int = 65536,
        workers: int = 1,
        scrape_workers: int = 1,
        per_plan: int | None = None,
        validate: bool = False,
        dry_run: bool = False,
        auto_throttle: bool = False,
        min_workers: int | None = None,
        max_workers: int | None = None,
        target_latency: float = 0.5,  # in seconds (target median)
        tune_step: int = 2,
        tune_err_lo: float = 0.02,
        tune_err_hi: float = 0.05,
        throttle_delay: float = 0.05,
    ) -> None:
        # Robust path resolution (independent of CWD)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(script_dir))
        self.catalog_dir = os.path.join(project_root, "data", "catalog")
        self.raw_data_dir = os.path.join(project_root, "data", "raw")
        self.logs_dir = os.path.join(project_root, "logs")

        os.makedirs(self.catalog_dir, exist_ok=True)
        os.makedirs(self.raw_data_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)

        self.paths = _Paths(self.catalog_dir, self.raw_data_dir, self.logs_dir)

        # Session log file (either user-supplied via --log-to, or auto-generated)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.session_log_path = log_to or os.path.join(
            self.logs_dir, f"download_log_{ts}.txt"
        )
        if log_to and not append:
            # Clean overwrite when the user requests a fixed path
            open(self.session_log_path, "w", encoding="utf-8").close()

        # Session parameters
        self.progress = progress
        self.chunk_size = int(chunk_size)
        self.limit_plans = limit_plans
        self.max_spectra = max_spectra
        self.timeout = int(timeout)
        self.delay_between = float(delay_between)
        self.total_downloaded_this_session = 0
        self.workers = max(1, int(workers))
        self.scrape_workers = max(1, int(scrape_workers))
        self.per_plan = None if per_plan in (None, 0) else int(per_plan)
        self.validate_downloads = bool(validate)
        self.dry_run = bool(dry_run)
        self.retries = int(retries)
        self.plan_urls_in_queue: list[str] = []
        self.auto_throttle = bool(auto_throttle)
        self.min_workers = int(min_workers or 4)  # (lower bound)
        self.max_workers = int(max_workers or self.workers)  # (upper bound)
        self.target_latency = float(target_latency)  # (s)
        self.tune_step = int(tune_step)  # (+2 by default)
        self.tune_err_lo = float(tune_err_lo)  # (< 2%: push harder)
        self.tune_err_hi = float(tune_err_hi)  # (> 5%: scale down)
        self.throttle_delay = float(throttle_delay)  # (minimum delay on 429)

        # HTTP session
        self.session = self._build_session(retries=retries, timeout=timeout)

        # Current state
        self.valid_urls_file = os.path.join(self.catalog_dir, "valid_plan_urls.csv")
        self.downloaded_log_file = os.path.join(
            self.catalog_dir, "downloaded_plans.csv"
        )
        self.plans_to_process = self._load_state()
        self.download_queue: list[list[str]] = []
        self._med_baseline: float | None = None

        # Session header in the log
        self._log("--- New download session ---")
        self._log(f"version={__version__}")
        self._log(
            f"plans={self.limit_plans or 'all'}, max_spectra={self.max_spectra or '∞'}"
        )

    # -------------------------------- I/O utilities ----------------------------------

    def _log(self, msg: str) -> None:
        """Append a line to the session log file (UTF-8, no fail)."""
        try:
            with open(self.session_log_path, "a", encoding="utf-8") as fh:
                fh.write(msg.rstrip() + "\n")
        except Exception:
            # Never break the run because of a logging error
            pass

    def _say(self, msg: str) -> None:
        """Print a message and write it to the session log."""
        print(msg)
        self._log(msg)

    def _say_tqdm(self, msg: str) -> None:
        """Thread-safe variant when a ``tqdm`` progress bar is active."""
        tqdm.write(msg)
        self._log(msg)

    # --------------------------- HTTP & state utilities ----------------------------

    @staticmethod
    def _build_session(*, retries: int, timeout: int) -> requests.Session:
        """Build a ``requests.Session`` with retries and exponential backoff.

        Notes
        -----
        - An explicit **User-Agent** header is set.
        - Timeouts are passed to each ``get()`` call.
        """
        session = requests.Session()
        retry_cfg = Retry(
            total=retries,
            backoff_factor=0.8,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=("GET", "HEAD"),
            raise_on_status=False,
        )
        adapter = HTTPAdapter(
            max_retries=retry_cfg, pool_connections=64, pool_maxsize=64
        )
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        session.headers.update(
            {
                "User-Agent": "AstroSpectro-Downloader/3.1 (+https://example.org)",
                "Accept": "*/*",
                "Connection": "keep-alive",
            }
        )
        # Memorise default timeout as a simple internal attribute
        session.request_timeout = timeout  # type: ignore[attr-defined]
        return session

    def _load_state(self) -> List[str]:
        """Load the list of plans to process.

        Read ``valid_plan_urls.csv``, remove URLs already present in
        ``downloaded_plans.csv``.  Apply ``self.limit_plans`` if set.

        Returns
        -------
        list[str]
            Remaining URLs, in their original order.
        """
        try:
            df_valid = pd.read_csv(self.valid_urls_file)
        except FileNotFoundError:
            print(
                f"ERROR: File not found: '{self.valid_urls_file}'. "
                "Generate it first (list of valid plan URLs)."
            )
            return []

        already_completed: set[str] = set()
        if os.path.exists(self.downloaded_log_file):
            try:
                df_completed = pd.read_csv(self.downloaded_log_file)
                already_completed = set(df_completed["url"].astype(str).tolist())
                print(f"  > {len(already_completed)} plan(s) already completed.")
            except Exception:
                pass

        remaining = df_valid[~df_valid["url"].isin(already_completed)]["url"].tolist()
        if self.limit_plans:
            remaining = remaining[: self.limit_plans]
        return remaining

    def _scrape_plan(self, plan_url: str):
        """Return ``(plan_name, fits_gz_urls)`` for a plan page (thread-safe)."""
        session = self._build_session(retries=3, timeout=self.timeout)
        plan_name = plan_url.rstrip("/").split("/")[-1]
        urls: list[str] = []
        try:
            resp = session.get(plan_url, timeout=session.request_timeout)  # type: ignore[attr-defined]
            resp.raise_for_status()
        except requests.RequestException:
            return plan_name, urls

        soup = BeautifulSoup(resp.text, "html.parser")
        for link in soup.find_all("a"):
            href = (link.get("href") or "").strip()
            if href.endswith(".fits.gz"):
                urls.append(urljoin(plan_url, href))
        return plan_name, urls

    @staticmethod
    def _is_valid_fits_gz(path: str) -> bool:
        """Lightweight validation: readable GZIP header with FITS ``SIMPLE`` keyword."""
        try:
            with gzip.open(path, "rb") as fh:
                head = fh.read(2880)  # 1 bloc FITS
            return b"SIMPLE" in head[:80]
        except Exception:
            return False

    # -------------------------------- phase 1: scrape ---------------------------------

    def _build_download_queue(self) -> None:
        """Phase 1 — scrape plan pages and build ``self.download_queue``.

        - Sequential when ``self.scrape_workers == 1`` (legacy behaviour).
        - Parallel otherwise, with **stable order** (mapped by plan URL).
        - Also populates ``self.plan_urls_in_queue`` for Phase 2/3 alignment.
        """
        if not self.plans_to_process:
            self._say("\nNo plans to process. (Empty list)")
            self.download_queue = []
            self.plan_urls_in_queue = []
            return

        self._say("\n--- [Phase 1/3] Building download queue ... ---")

        # Sequential mode (original order guaranteed)
        if self.scrape_workers == 1:
            selected: list[str] = []
            queue: list[list[str]] = []
            for i, plan_url in enumerate(self.plans_to_process, start=1):
                plan_name, plan_urls = self._scrape_plan(plan_url)
                self._say(
                    f"  > [{i}/{len(self.plans_to_process)}] {plan_name}  ({len(plan_urls)} files)"
                )
                if plan_urls:
                    queue.append(plan_urls)
                    selected.append(plan_url)
                else:
                    self._say("    -> Warning: no .fits.gz found.")
            self.download_queue = queue
            self.plan_urls_in_queue = selected
            self._say("  > Queue built.")
            return

        # Parallel mode (collect into a dict, then rebuild in original order)
        results: dict[str, list[str]] = {}
        with ThreadPoolExecutor(max_workers=self.scrape_workers) as ex:
            futures = {
                ex.submit(self._scrape_plan, url): url for url in self.plans_to_process
            }
            for i, fut in enumerate(as_completed(futures), start=1):
                plan_url = futures[fut]
                try:
                    plan_name, plan_urls = fut.result()
                except Exception as e:  # network/parsing error
                    self._say(f"  > [{i}/{len(futures)}] ERROR: {e}")
                    plan_urls = []
                self._say(
                    f"  > [{i}/{len(futures)}] {plan_url.rstrip('/').split('/')[-1]}  ({len(plan_urls)} files)"
                )
                results[plan_url] = plan_urls

        # Reconstruct in self.plans_to_process order, excluding empty plans
        self.plan_urls_in_queue = [u for u in self.plans_to_process if results.get(u)]
        self.download_queue = [results[u] for u in self.plan_urls_in_queue]
        self._say("  > File construite.")

    # ----------------------------- phase 2: download --------------------------------

    def _iter_missing_files(self) -> Iterable[str]:
        """Iterate all URLs whose files are missing locally (across all plans)."""
        for plan_fits_urls in self.download_queue:
            for file_url in plan_fits_urls:
                plan_name = file_url.split("/")[-2]
                filename = file_url.split("/")[-1]
                dest_dir = os.path.join(self.raw_data_dir, plan_name)
                dest_path = os.path.join(dest_dir, filename)
                if not os.path.exists(dest_path):
                    yield file_url

    def _download_one_threaded(self, url: str) -> bool:
        """Download *url* using a fresh local Session → ``True`` if written.

        Similar to ``_stream_download`` but computes ``dest_path`` internally and avoids
        partager `self.session` entre threads (Requests Session n'est pas thread‑safe).
        """
        plan_name = url.split("/")[-2]
        filename = url.split("/")[-1]
        dest_dir = os.path.join(self.raw_data_dir, plan_name)
        os.makedirs(dest_dir, exist_ok=True)
        dest_path = os.path.join(dest_dir, filename)
        temp_path = f"{dest_path}.part"

        # already present and no validation requested → nothing to do
        if os.path.exists(dest_path) and not self.validate_downloads:
            return False

        session = self._build_session(retries=self.retries, timeout=self.timeout)

        # Resume support
        resume_from = 0
        headers = {}
        if os.path.exists(temp_path):
            resume_from = os.path.getsize(temp_path)
            if resume_from > 0:
                headers["Range"] = f"bytes={resume_from}-"

        # Retry loop with application-level backoff+jitter (on top of urllib3 Retry)
        for attempt in range(self.retries + 1):
            try:
                with session.get(url, stream=True, headers=headers, timeout=session.request_timeout) as r:  # type: ignore[attr-defined]
                    if resume_from and r.status_code == 200:
                        # Server did not honour the Range header → restart from scratch
                        resume_from = 0
                    r.raise_for_status()

                    mode = "ab" if resume_from else "wb"
                    with open(temp_path, mode) as f:
                        for chunk in r.iter_content(chunk_size=self.chunk_size):
                            if chunk:
                                f.write(chunk)
                os.replace(temp_path, dest_path)

                # Lightweight validation (if requested)
                if self.validate_downloads and not self._is_valid_fits_gz(dest_path):
                    self._say_tqdm(f"    -> Validation failed: {filename}. Retrying…")
                    try:
                        os.remove(dest_path)
                    except Exception:
                        pass
                    resume_from = 0
                    headers.pop("Range", None)
                    # retry
                    raise requests.RequestException("validation failed")

                self._log(f"OK {plan_name}/{filename}")
                return True

            except requests.RequestException as e:
                # small backoff jitter on top of the adapter's Retry
                if attempt < self.retries:
                    sleep_s = (0.5 * (2**attempt)) * (0.5 + random.random())
                    try:
                        import time as _t

                        _t.sleep(sleep_s)
                    except Exception:
                        pass
                    continue
                else:
                    # final failure → clean up the .part
                    try:
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
                    except Exception:
                        pass
                    self._say_tqdm(f"    -> ERROR: {filename} ({e})")
                    return False

    def _download_one_stats(self, url: str) -> tuple[bool, float, int]:
        """Like ``_download_one_threaded`` but return ``(success, duration_s, code)``.

        ``code`` is 0 (ok/generic error), 429, or 5xx (500–599).
        """
        start = time.monotonic()
        plan_name = url.split("/")[-2]
        filename = url.split("/")[-1]
        dest_dir = os.path.join(self.raw_data_dir, plan_name)
        os.makedirs(dest_dir, exist_ok=True)
        dest_path = os.path.join(dest_dir, filename)
        temp_path = f"{dest_path}.part"

        if os.path.exists(dest_path) and not self.validate_downloads:
            return (False, time.monotonic() - start, 0)

        session = self._build_session(retries=self.retries, timeout=self.timeout)

        resume_from = 0
        headers = {}
        if os.path.exists(temp_path):
            resume_from = os.path.getsize(temp_path)
            if resume_from > 0:
                headers["Range"] = f"bytes={resume_from}-"

        code = 0
        for attempt in range(self.retries + 1):
            try:
                with session.get(url, stream=True, headers=headers, timeout=session.request_timeout) as r:  # type: ignore[attr-defined]
                    if resume_from and r.status_code == 200:
                        resume_from = 0
                    r.raise_for_status()
                    mode = "ab" if resume_from else "wb"
                    with open(temp_path, mode) as f:
                        for chunk in r.iter_content(chunk_size=self.chunk_size):
                            if chunk:
                                f.write(chunk)
                os.replace(temp_path, dest_path)

                if self.validate_downloads and not self._is_valid_fits_gz(dest_path):
                    self._say_tqdm(f"    -> Validation failed: {filename}. Retrying…")
                    try:
                        os.remove(dest_path)
                    except Exception:
                        pass
                    resume_from = 0
                    headers.pop("Range", None)
                    raise requests.RequestException("validation failed")

                self._log(f"OK {plan_name}/{filename}")
                return (True, time.monotonic() - start, 0)

            except requests.HTTPError as e:
                sc = getattr(e.response, "status_code", 0) or 0
                code = 429 if sc == 429 else (sc if 500 <= sc <= 599 else 0)
                if attempt < self.retries:
                    sleep_s = (0.5 * (2**attempt)) * (0.5 + random.random())
                    time.sleep(sleep_s)
                    continue
                try:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                except Exception:
                    pass
                self._say_tqdm(f"    -> HTTP ERROR {sc}: {filename}")
                return (False, time.monotonic() - start, code)
            except requests.RequestException:
                if attempt < self.retries:
                    sleep_s = (0.5 * (2**attempt)) * (0.5 + random.random())
                    time.sleep(sleep_s)
                    continue
                try:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                except Exception:
                    pass
                self._say_tqdm(f"    -> ERROR: {filename}")
                return (False, time.monotonic() - start, code)

    def _stream_download(self, url: str, dest_path: str) -> bool:
        """Download *url* to *dest_path* with **atomic** write.

        A temporary ``dest_path + '.part'`` file is used during the write and
        is replaced via ``os.replace()`` only on success.

        Returns
        -------
        bool
            ``True`` on success, ``False`` otherwise (logged via ``_say_tqdm``).
        """
        temp_path = f"{dest_path}.part"
        try:
            with self.session.get(url, stream=True, timeout=self.session.request_timeout) as r:  # type: ignore[attr-defined]
                r.raise_for_status()
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                with open(temp_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=self.chunk_size):
                        if chunk:
                            f.write(chunk)
            os.replace(temp_path, dest_path)
            return True
        except requests.RequestException as e:
            # Clean up the .part file if present
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            except Exception:
                pass
            self._say_tqdm(f"    -> ERROR: {os.path.basename(dest_path)} ({e})")
            return False

    def run_download(self) -> None:
        """Download missing files in **round-robin** order.

        Respects ``self.max_spectra`` if set.  Shows a ``tqdm`` progress bar
        when ``self.progress`` is ``True`` (or ``--progress`` was passed on
        the CLI).

        - Parallel when ``self.workers > 1`` (``ThreadPoolExecutor`` + tqdm).
        """
        if not self.download_queue:
            self._say("\n[Phase 2/3] Nothing to do: queue is empty.")
            return

        # 1) Build an interleaved list across plans
        missing_by_plan: list[list[str]] = []
        for plan_urls in self.download_queue:
            plan_missing: list[str] = []
            for url in plan_urls:
                plan_name = url.split("/")[-2]
                filename = url.split("/")[-1]
                dest_dir = os.path.join(self.raw_data_dir, plan_name)
                dest_path = os.path.join(dest_dir, filename)
                if not os.path.exists(dest_path):
                    plan_missing.append(url)
            if plan_missing:
                missing_by_plan.append(plan_missing)

        if not missing_by_plan:
            self._say(
                "\n--- [Phase 2/3] Nothing to download (everything already present)."
            )
            return

        n_active = len(missing_by_plan)
        auto_per_plan = None
        if (
            (self.per_plan in (None, 0))
            and (self.max_spectra is not None)
            and n_active > 0
        ):
            auto_per_plan = max(1, math.ceil(int(self.max_spectra) / n_active))

        limit_per_plan = auto_per_plan if auto_per_plan is not None else self.per_plan
        if limit_per_plan is not None:
            missing_by_plan = [urls[: int(limit_per_plan)] for urls in missing_by_plan]

        self._say("\nMissing file summary (after deduplication and per-plan cap):")
        for plan in missing_by_plan:
            if not plan:
                continue
            plan_name = plan[0].split("/")[-2]
            self._say(f"  - {plan_name}: {len(plan)} files to download")

        interleaved: list[str] = []
        for batch in zip_longest(*missing_by_plan):
            for url in batch:
                if url is None:
                    continue
                interleaved.append(url)
                if self.max_spectra is not None and len(interleaved) >= int(
                    self.max_spectra
                ):
                    break
            if self.max_spectra is not None and len(interleaved) >= int(
                self.max_spectra
            ):
                break

        total = len(interleaved)

        # 2) Dry-run mode: show preview and stop here
        if self.dry_run:
            self._say("\n[Dry-run] Session preview (no disk writes):")
            self._say(
                f"  > {len(missing_by_plan)} plan(s), {total} file(s) to process (after max_spectra cap)"
            )
            return

        # 3) Downloads
        self._say(
            f"\n--- [Phase 2/3] {total} new spectra to download (workers={self.workers}) ---"
        )
        disable_progress = not bool(self.progress)

        if self.workers <= 1:
            # Legacy sequential path
            from time import sleep as _sl

            with tqdm(
                total=total,
                desc="Downloading",
                unit="spectrum",
                leave=False,
                mininterval=0.4,
                disable=disable_progress,
            ) as pbar:
                for url in interleaved:
                    plan_name = url.split("/")[-2]
                    filename = url.split("/")[-1]
                    dest_dir = os.path.join(self.raw_data_dir, plan_name)
                    dest_path = os.path.join(dest_dir, filename)

                    if os.path.exists(dest_path):
                        pbar.update(1)
                        continue

                    if (
                        self.max_spectra
                        and self.total_downloaded_this_session >= self.max_spectra
                    ):
                        self._say_tqdm(
                            f"Limit reached ({self.max_spectra}). Stopping download."
                        )
                        return

                    ok = self._stream_download(url, dest_path)
                    if ok:
                        self._log(f"OK {plan_name}/{filename}")
                        self.total_downloaded_this_session += 1
                    pbar.update(1)
                    if self.delay_between > 0:
                        _sl(self.delay_between)
            return

        # Parallel
        if self.workers > 1:
            if not self.auto_throttle:
                with tqdm(
                    total=total,
                    desc="Downloading",
                    unit="spectre",
                    leave=False,
                    mininterval=0.4,
                    disable=disable_progress,
                ) as pbar:
                    with ThreadPoolExecutor(max_workers=self.workers) as ex:
                        futures = [
                            ex.submit(self._download_one_threaded, url)
                            for url in interleaved
                        ]
                        for fut in as_completed(futures):
                            try:
                                wrote = fut.result()
                                if wrote:
                                    self.total_downloaded_this_session += 1
                            except Exception:
                                pass
                            pbar.update(1)
            else:
                # Auto-throttle: process in batches of self.workers and adjust
                i = 0
                with tqdm(
                    total=total,
                    desc="Downloading",
                    unit="spectre",
                    leave=False,
                    mininterval=0.4,
                    disable=disable_progress,
                ) as pbar:
                    while i < len(interleaved):
                        batch = interleaved[i : i + self.workers]
                        succ = 0
                        latencies: list[float] = []
                        err429 = err5xx = total_b = 0

                        with ThreadPoolExecutor(max_workers=self.workers) as ex:
                            futs = [
                                ex.submit(self._download_one_stats, u) for u in batch
                            ]
                            for fut in as_completed(futs):
                                try:
                                    ok, dt, code = fut.result()
                                    if ok:
                                        succ += 1
                                    latencies.append(dt)
                                    if code == 429:
                                        err429 += 1
                                    elif 500 <= code <= 599:
                                        err5xx += 1
                                except Exception:
                                    pass
                                total_b += 1
                                pbar.update(1)

                        # --- Adjust ONCE per batch ---
                        err_rate = (err429 + err5xx) / max(1, total_b)
                        med = statistics.median(latencies) if latencies else 0.0

                        # counter updated once per batch
                        self.total_downloaded_this_session += succ

                        # EMA baseline of the median latency
                        if self._med_baseline is None:
                            self._med_baseline = med or self.target_latency
                        elif med:
                            self._med_baseline = 0.2 * med + 0.8 * self._med_baseline

                        # Dynamic thresholds vs baseline
                        hi_thresh = max(
                            self.target_latency, (self._med_baseline or 0) * 1.5
                        )
                        lo_thresh = max(self.target_latency, (self._med_baseline or 0))

                        new_workers = self.workers
                        if (
                            err429 > 0
                            or err_rate > self.tune_err_hi
                            or (med and med > hi_thresh)
                        ):
                            new_workers = max(self.min_workers, self.workers // 2)
                            if err429 > 0:
                                self.delay_between = max(
                                    self.delay_between, self.throttle_delay
                                )
                        elif err_rate < self.tune_err_lo and (
                            med == 0.0 or med <= lo_thresh
                        ):
                            new_workers = min(
                                self.max_workers, self.workers + self.tune_step
                            )

                        if new_workers != self.workers:
                            self._say_tqdm(
                                f"Auto-throttle: workers {self.workers} → {new_workers}  "
                                f"(err={err_rate:.1%}, med={med*1000:.0f} ms, base={((self._med_baseline or 0)*1000):.0f} ms)"
                            )
                            self.workers = new_workers

                        i += len(batch)
                # end auto-throttle
            return

    # ------------------------------ phase 3: bookkeeping ----------------------------

    def _update_state(self) -> None:
        """Check which plans are now fully downloaded and update the ledger.

        A plan is considered *complete* when **all** expected files (links found
        in Phase 1) exist under ``data/raw/<plan>/``.
        """
        self._say("\n--- [Phase 3/3] Updating plan state ---")

        newly_completed: List[str] = []
        for plan_url, fits_list in zip(self.plan_urls_in_queue, self.download_queue):
            plan_name = plan_url.rstrip("/").split("/")[-1]
            total = len(fits_list)
            present = sum(
                1
                for u in fits_list
                if os.path.exists(
                    os.path.join(self.raw_data_dir, plan_name, u.split("/")[-1])
                )
            )
            if total > 0 and present == total:
                newly_completed.append(plan_url)
                self._say(f"  > Plan completed: {plan_name}")

        if not newly_completed:
            self._say("  > No new plans completed.")
            return

        df_new = pd.DataFrame({"url": newly_completed})
        df_new.to_csv(
            self.downloaded_log_file,
            mode="a",
            index=False,
            header=not os.path.exists(self.downloaded_log_file),
        )
        self._say(
            f"  > {len(newly_completed)} plan(s) added to '{self.downloaded_log_file}'."
        )

    # --------------------------------- simple facade ---------------------------------

    def run(self) -> None:
        """Run all 3 phases (scrape → download → update) and summarise the session."""
        self._build_download_queue()
        self.run_download()

        # In dry-run mode: do **not** modify state
        if self.dry_run:
            self._say("\n[Dry-run] State not modified (no plan marked completed).")
            self._log("SUMMARY dry_run=True")
            self._say(f"\nSession log written: {self.session_log_path}")
            return

        self._update_state()
        self._log(f"SUMMARY total_downloaded={self.total_downloaded_this_session}")
        self._say(f"\nSession log written: {self.session_log_path}")


# ----------------------------------- CLI -----------------------------------


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    - ``--limit`` : max number of **new** plans to process.
    - ``--max-spectres`` : spectra cap for this session.
    - ``--progress`` : force ``tqdm`` progress bar (useful in notebooks).
    - ``--delay`` : delay (s) between downloads.
    - ``--chunk`` : write chunk size (bytes).
    - ``--log-to`` + ``--append`` : log file control.
    """
    parser = argparse.ArgumentParser(
        description="Smart downloader for LAMOST DR5 spectra"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of NEW plans to process (e.g. 5).",
    )
    parser.add_argument(
        "--max-spectres",
        type=int,
        default=None,
        help="Stop after this total number of downloaded spectra (e.g. 200).",
    )
    parser.add_argument(
        "--log-to",
        type=str,
        default=None,
        help="Path to the session log file (auto-generated if omitted).",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to existing log instead of overwriting.",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Force tqdm progress bar even when stdout is not a TTY (e.g. notebook).",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.2,
        help="Delay (s) between downloads. 0 = fastest.",
    )
    parser.add_argument(
        "--chunk",
        type=int,
        default=65536,
        help="Download chunk size (bytes). 65536=64KB, 131072=128KB, etc.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel downloads (1 = sequential).",
    )
    parser.add_argument(
        "--scrape-workers",
        type=int,
        default=1,
        help="Parallelism for plan page scraping.",
    )
    parser.add_argument(
        "--per-plan",
        type=int,
        default=None,
        help="Per-plan file cap (after deduplication).",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Quickly validate downloaded .fits.gz files (FITS header check).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write nothing: only show what would be done.",
    )
    parser.add_argument(
        "--auto-throttle",
        action="store_true",
        help="Automatically adjust the number of workers based on network/server conditions.",
    )
    parser.add_argument(
        "--min-workers",
        type=int,
        default=None,
        help="Lower bound for auto-throttle (default: 4).",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Upper bound for auto-throttle (default: =workers).",
    )
    parser.add_argument(
        "--target-lat",
        type=float,
        default=0.5,
        help="Target median latency (s). Above this → scale down.",
    )
    return parser.parse_args()


def main(argv: Optional[list[str]] = None) -> int:
    """Programmatic entry point (useful for tests / CI)."""
    args = _parse_args() if argv is None else _parse_args()
    dl = SmartDownloader(
        limit_plans=args.limit,
        max_spectra=args.max_spectres,
        timeout=args.http_timeout if hasattr(args, "http_timeout") else 60,
        delay_between=args.delay,
        retries=args.retries if hasattr(args, "retries") else 3,
        progress=(args.progress or sys.stdout.isatty()),
        log_to=args.log_to,
        append=args.append,
        chunk_size=args.chunk,
        workers=args.workers,
        scrape_workers=args.scrape_workers,
        per_plan=args.per_plan,
        validate=args.validate,
        dry_run=args.dry_run,
        auto_throttle=args.auto_throttle,
        min_workers=args.min_workers,
        max_workers=args.max_workers,
        target_latency=args.target_lat,
    )
    dl.run()
    dl._say(
        f"\n--- Session finished: {dl.total_downloaded_this_session} spectrum/spectra "
        f"downloaded across {len(dl.plans_to_process)} plan(s). ---"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
