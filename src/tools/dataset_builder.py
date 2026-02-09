"""AstroSpectro — Training-batch management (DatasetBuilder).

This module provides a lightweight utility responsible for assembling training
batches of spectra **without ever reusing** files seen in previous sessions.

Responsibilities
----------------
1. Scan the ``raw/`` directory tree to list every available FITS file
   (``.fits.gz``).
2. Maintain a persistent CSV ledger (``catalog/trained_spectra.csv``) of
   spectra already consumed by earlier training runs.
3. Select a new batch of **never-logged** file paths according to a simple
   strategy (``"random"`` or ``"first"``).

Conventions and I/O
-------------------
- Returned paths are **relative** to ``raw_data_dir`` and **normalised** with
  forward slashes (cross-platform compatible).
- The ledger is stored at ``catalog_dir/trained_spectra.csv`` with a single
  ``file_path`` column.
- This module **does not read** FITS content; it only returns file paths.

Public API
----------
- ``get_new_training_batch(batch_size, strategy)`` -> ``list[str]``
- ``update_trained_log(newly_trained_files)`` -> ``None``

Examples
--------
>>> db = DatasetBuilder(catalog_dir="data/catalog", raw_data_dir="data/raw")
>>> batch = db.get_new_training_batch(batch_size=500, strategy="random")
>>> # ... run the processing pipeline on *batch*
>>> db.update_trained_log(batch)  # record consumed files
"""

from __future__ import annotations

import os
import random
from typing import List, Literal, Set

import pandas as pd


class DatasetBuilder:
    """
    Manage training batches to ensure no spectrum is reused across sessions.

    This class encapsulates the logic for scanning available FITS files,
    filtering out previously used spectra via a persistent CSV ledger,
    and returning fresh file-path lists ready for the processing pipeline.

    Parameters
    ----------
    catalog_dir : str, optional
        Directory containing the catalog files, including the trained-spectra
        ledger ``trained_spectra.csv`` (default: ``"../data/catalog/"``).
    raw_data_dir : str, optional
        Root directory of raw FITS data, organised in plan sub-folders
        (default: ``"../data/raw/"``).

    Attributes
    ----------
    catalog_dir : str
        Path to the catalog directory.
    raw_data_dir : str
        Path to the raw data root.
    trained_log_path : str
        Absolute path to the trained-spectra CSV ledger.

    Examples
    --------
    >>> builder = DatasetBuilder("data/catalog", "data/raw")
    >>> batch = builder.get_new_training_batch(500, strategy="random")
    >>> builder.update_trained_log(batch)
    """

    def __init__(
        self,
        catalog_dir: str = "../data/catalog/",
        raw_data_dir: str = "../data/raw/",
    ) -> None:
        self.catalog_dir = catalog_dir
        self.raw_data_dir = raw_data_dir
        self.trained_log_path = os.path.join(self.catalog_dir, "trained_spectra.csv")

    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------

    def _list_available_fits(self) -> List[str]:
        """List all ``.fits.gz`` files available under ``raw_data_dir``.

        Returned paths are **relative** to ``raw_data_dir`` and use forward
        slashes for cross-platform consistency.

        Returns
        -------
        list[str]
            Normalised relative paths, e.g.
            ``'GAC_105N29_B1/spec-55863-GAC_105N29_B1_sp01-001.fits.gz'``.
        """
        available_files: List[str] = []

        for root, _dirs, files in os.walk(self.raw_data_dir):
            for fname in files:
                if fname.endswith(".fits.gz"):
                    full = os.path.join(root, fname)
                    rel = os.path.relpath(full, self.raw_data_dir).replace("\\", "/")
                    available_files.append(rel)

        return available_files

    def _load_trained_log(self) -> Set[str]:
        """Load the set of file paths already recorded in the ledger.

        The CSV is expected to contain a ``file_path`` column. Null values
        are silently ignored.  Malformed or empty files are handled
        gracefully without raising.

        Returns
        -------
        set[str]
            Set of **relative** paths that have already been used.
        """
        if not os.path.exists(self.trained_log_path):
            return set()

        # Robust reading: try comma separator, then fall back to auto-detection
        try:
            try:
                df = pd.read_csv(self.trained_log_path)
            except Exception:
                df = pd.read_csv(self.trained_log_path, sep=None, engine="python")
        except pd.errors.EmptyDataError:
            return set()

        # Identify a plausible path column among known candidates
        candidates = [
            "file_path",
            "path",
            "filepath",
            "fits_path",
            "relpath",
            "raw_path",
        ]
        col = next((c for c in candidates if c in df.columns), None)
        if col is None:
            print(
                f"  > WARNING: '{self.trained_log_path}' has no recognised path column ({candidates}). Ignored."
            )
            return set()

        # Normalise to relative paths (w.r.t. raw_data_dir) with forward slashes
        paths = []
        for p in df[col].dropna().astype(str):
            p = p.strip().replace("\\", "/")
            if not p:
                continue
            if os.path.isabs(p):
                try:
                    p = os.path.relpath(p, self.raw_data_dir).replace("\\", "/")
                except Exception:
                    # Si on ne peut pas relativiser (hors raw/), garde tel quel
                    pass
            paths.append(p)

        return set(paths)

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------

    def get_new_training_batch(
        self,
        batch_size: int = 500,
        strategy: Literal["random", "first"] = "random",
    ) -> List[str]:
        """Select a new batch of **never-trained** files.

        Parameters
        ----------
        batch_size : int, default=500
            Desired number of spectra in the batch.
        strategy : {'random', 'first'}, default='random'
            - ``'random'`` : draw a random sample from the available files.
            - ``'first'``  : take the first ``batch_size`` in scan order.

        Returns
        -------
        list[str]
            Normalised relative paths ready to be passed to the pipeline.
            May be an empty list if no new files are available.
        """
        print("--- Building a new training batch ---")

        all_available = self._list_available_fits()
        already_used = self._load_trained_log()

        print(f"  > {len(all_available)} spectra found under '{self.raw_data_dir}'")
        print(f"  > {len(already_used)} spectra already logged in the ledger.")

        # Keep only paths never seen before
        new_fits = [p for p in all_available if p not in already_used]
        print(f"  > {len(new_fits)} **new** spectra available.")

        if not new_fits:
            print("  > No new spectra to train on. Stopping.")
            return []

        # Adjust batch size if fewer files are available
        if len(new_fits) < batch_size:
            print(
                f"  > Warning: only {len(new_fits)} available < batch_size={batch_size}."
            )
            batch_size = len(new_fits)

        # Apply the selection strategy
        if strategy == "random":
            selected = random.sample(new_fits, k=batch_size)
            print(f"  > Random selection of {batch_size} spectra.")
        else:  # 'first'
            selected = new_fits[:batch_size]
            print(f"  > Selected the first {batch_size} spectra.")

        return selected

    def update_trained_log(self, newly_trained_files: List[str]) -> None:
        """Append newly used file paths to the trained-spectra ledger.

        Duplicates are automatically filtered out before writing.

        Parameters
        ----------
        newly_trained_files : list[str]
            Normalised relative paths (forward slashes) of files consumed
            by the training session that just completed.
        """
        if not newly_trained_files:
            return

        # Load existing ledger into a set for fast deduplication
        if os.path.exists(self.trained_log_path):
            try:
                existing_df = pd.read_csv(self.trained_log_path)
                existing = set(existing_df.get("file_path", pd.Series([])).astype(str))
            except pd.errors.EmptyDataError:
                existing = set()
        else:
            os.makedirs(self.catalog_dir, exist_ok=True)
            existing = set()

        truly_new = [p for p in newly_trained_files if p not in existing]
        if not truly_new:
            print("  > Nothing new to add (all already in ledger).")
            return

        df_new = pd.DataFrame({"file_path": truly_new})
        df_new.to_csv(
            self.trained_log_path,
            mode="a",
            index=False,
            header=not os.path.exists(self.trained_log_path),
        )
        print(
            f"  > {len(truly_new)} new spectra appended to ledger: "
            f"'{self.trained_log_path}'."
        )
