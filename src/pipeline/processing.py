"""
AstroSpectro --- Spectrum processing pipeline (preprocessing + features).
=========================================================================

For each FITS spectrum this module applies the full processing chain:

1. Preprocessing (loading, flux normalisation, inverse-variance sanitisation).
2. Absorption-line detection and association (``PeakDetector``).
3. Spectroscopic feature extraction (``FeatureEngineer``: FWHM, EW,
   ratios, indices, etc.).
4. Photometric / colour features (when available from the catalogue).
5. Final assembly into a DataFrame row, then aggregation into a
   feature table.

Conventions
-----------
- Wavelengths are in Angstroms (Å).
- ``invvar`` (inverse-variance) is cleaned upstream to avoid NaN / Inf
  during ``sqrt`` operations and to produce robust uncertainties (see
  ``utils.sanitize_invvar`` and ``make_stddev_uncertainty_from_invvar``).
- ``match_*`` columns (e.g. ``match_Hbeta_wl``, ``match_Hbeta_prom``)
  come from the line-matching step and **complement** the ``feature_*``
  columns returned by the Feature Engineering module.
- ``feature_*`` vectors are returned in a **stable order``
  (``feature_names`` list).

Parallelism strategy
--------------------
Processing is split into two phases per chunk:

**Phase 1 — Parallel I/O** (``ThreadPoolExecutor``):
  Threads overlap their blocking ``gzip.open`` / ``fits.open`` calls
  because those operations release the GIL.  Results are plain tuples
  held in shared memory — no serialisation through an IPC pipe.

**Phase 2 — Parallel CPU** (``ProcessPoolExecutor`` with worker initializer):
  True multi-core parallelism on all platforms, including Windows where
  the GIL prevents threads from running CPU-bound Python code in
  parallel.  A ``ProcessPoolExecutor`` spawns N worker processes, each
  initialised *once* with its own ``SpectraPreprocessor`` /
  ``PeakDetector`` / ``FeatureEngineer`` via ``_init_cpu_worker``.
  Only the raw numpy arrays are pickled per task (fast buffer protocol).
  The module-level ``_compute_features_worker`` function is used instead
  of a bound method to avoid pickling the ``ProcessingPipeline`` object.
  A single pool is kept alive for the entire batch to amortise the
  process-spawn and module-import cost across all chunks.

Inputs / Outputs
----------------
Inputs:
    - ``file_path`` : path to a FITS file (str)
    - ``catalog_row`` : (optional) catalogue row for photometric / Gaia
      enrichment
    - ``config`` : (optional) parameter dict (detection window, thresholds)

Outputs:
    - ``pd.Series`` : consolidated row containing ``match_*``, ``feature_*``,
      and photometric / Gaia columns when available.
    - An ensemble of such Series is aggregated into a ``pd.DataFrame``.

Public API
----------
- ``ProcessingPipeline.run(batch_paths) -> pd.DataFrame``

Examples
--------
>>> from pipeline.processing import ProcessingPipeline
>>> pp = ProcessingPipeline(raw_data_dir, catalog_df)
>>> df = pp.run(["specA.fits.gz", "specB.fits.gz"])
"""

from __future__ import annotations

import gc
import gzip
import multiprocessing
import os
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from astropy.io import fits
from tqdm import tqdm

from .preprocessor import SpectraPreprocessor
from .peak_detector import PeakDetector, matches_to_series
from .feature_engineering import (
    FeatureEngineer,
    add_gaia_derived_features,
    add_line_composites,
    add_main_sequence_delta,
    add_photometric_composites,
    stabilize_spectral_features,
)
from utils import sanitize_invvar

# ---------------------------------------------------------------------------
# Tuning constants
# ---------------------------------------------------------------------------
# I/O threads: NVMe SSD → 32–64 | SATA SSD → 16–32 | HDD → 4–8
_DEFAULT_N_IO_WORKERS: int = 32

# CPU worker processes: all physical cores by default
_DEFAULT_N_CPU_WORKERS: int = multiprocessing.cpu_count()

# Spectra per chunk.  Each spectrum ≈ 86 KB raw arrays.
# 5 000 × 86 KB ≈ 430 MB — safe on 64 GB RAM.
_DEFAULT_CHUNK_SIZE: int = 5_000

# Spectra dispatched per IPC round-trip to CPU workers.
# Larger → less scheduling overhead; smaller → better load balancing.
_CPU_CHUNKSIZE: int = 16

# ---------------------------------------------------------------------------
# Module-level CPU worker state (set once per worker process by the initializer)
# ---------------------------------------------------------------------------
_w_preprocessor: Optional[SpectraPreprocessor] = None
_w_peak_detector: Optional[PeakDetector] = None
_w_feature_engineer: Optional[FeatureEngineer] = None


def _init_cpu_worker(prominence: float, window: int) -> None:
    """Initialise per-process worker state.

    Called once by ``ProcessPoolExecutor`` when each worker process
    starts.  Creates expensive objects in the worker address space so
    they are reused — not rebuilt — for every task.  Also suppresses the
    median-flux ``UserWarning`` that is expected for low-SNR spectra.

    Parameters
    ----------
    prominence : float
        ``PeakDetector`` prominence threshold.
    window : int
        ``PeakDetector`` detection window (Å).
    """
    global _w_preprocessor, _w_peak_detector, _w_feature_engineer
    warnings.filterwarnings("ignore", category=UserWarning)
    _w_preprocessor = SpectraPreprocessor()
    _w_peak_detector = PeakDetector(prominence=prominence, window=window)
    _w_feature_engineer = FeatureEngineer()


def _compute_features_worker(
    task: Tuple[str, np.ndarray, np.ndarray, np.ndarray],
) -> Optional[Dict[str, float]]:
    """Compute all spectroscopic features from pre-loaded arrays.

    Module-level function (not a bound method) so that only the input
    numpy arrays are pickled for IPC — not the ``ProcessingPipeline``
    object.  Uses per-process globals set by ``_init_cpu_worker``.

    Parameters
    ----------
    task : tuple of (str, ndarray, ndarray, ndarray)
        ``(file_path, wavelength, flux, invvar)`` for one spectrum.

    Returns
    -------
    Optional[Dict[str, float]]
        Feature record dict on success, or ``None`` if computation fails.
    """
    file_path, wavelength, flux, invvar = task
    try:
        # 1) Flux normalisation
        flux_norm = _w_preprocessor.normalize_spectrum(flux)

        # 2) Absorption-line detection / association
        matched_lines = _w_peak_detector.analyze_spectrum(wavelength, flux_norm)

        # 3) Feature engineering (prominence / FWHM / EW / indices)
        features_vector = _w_feature_engineer.extract_features(
            matched_lines, wavelength, flux_norm, invvar
        )

        # 3b) Additional columns from line matching (stable & interpretable)
        order = _w_feature_engineer.base_lines
        match_series = matches_to_series(matched_lines, order=order, prefix="match_")
        match_series = match_series.rename(lambda c: c.replace(" ", ""))

        # 4) Global continuum shape: slope and curvature (Pack D)
        slope = 0.0
        curvature = 0.0
        try:
            cont_mask = (wavelength > 4300) & (wavelength < 6800)
            lam = wavelength[cont_mask]
            flx = flux_norm[cont_mask]
            valid = np.isfinite(lam) & np.isfinite(flx)
            lam, flx = lam[valid], flx[valid]
            if lam.size >= 10:
                coeffs = np.polyfit(lam, flx, deg=2)
                curvature = float(coeffs[0])
                slope = float(coeffs[1])
        except Exception:
            pass

        # 5) Assemble the record row
        record: Dict[str, float] = {"file_path": file_path}

        for feature_name, feature_val in zip(
            _w_feature_engineer.feature_names, features_vector
        ):
            record[feature_name] = float(feature_val)

        record["feature_cont_slope"] = float(slope)
        record["feature_cont_curvature"] = float(curvature)

        record.update(
            {
                k: float(v) if pd.notna(v) else np.nan
                for k, v in match_series.to_dict().items()
            }
        )
        return record

    except Exception:
        return None


# ---------------------------------------------------------------------------
# Pipeline class
# ---------------------------------------------------------------------------


class ProcessingPipeline:
    """Full spectrum processing pipeline (preprocessing + feature extraction).

    Uses a two-phase parallelism strategy per chunk (see module docstring):
    parallel I/O with threads, then parallel CPU with processes.

    Parameters
    ----------
    raw_data_dir : str
        Directory containing gzipped FITS files.
    master_catalog_df : pd.DataFrame
        Metadata catalogue used to enrich spectra with photometric and
        Gaia information.
    n_io_workers : int, optional
        Threads for the I/O phase.  Tune to your storage type.
    n_cpu_workers : int, optional
        Worker processes for the CPU phase (default: all physical cores).
    prominence : float, optional
        ``PeakDetector`` prominence threshold (forwarded to workers).
    window : int, optional
        ``PeakDetector`` window in Å (forwarded to workers).
    """

    def __init__(
        self,
        raw_data_dir: str,
        master_catalog_df: pd.DataFrame,
        n_io_workers: int = _DEFAULT_N_IO_WORKERS,
        n_cpu_workers: int = _DEFAULT_N_CPU_WORKERS,
        prominence: float = 0.15,
        window: int = 8,
    ) -> None:
        self.raw_data_dir = raw_data_dir
        self.n_io_workers = n_io_workers
        self.n_cpu_workers = n_cpu_workers
        self.prominence = prominence
        self.window = window

        # Local instances kept for _process_single_spectrum (backward compat)
        self.preprocessor = SpectraPreprocessor()
        self.peak_detector = PeakDetector(prominence=prominence, window=window)
        self.feature_engineer = FeatureEngineer()

        self.master_catalog = master_catalog_df
        if self.master_catalog is not None and not self.master_catalog.empty:
            if "fits_name" in self.master_catalog.columns:
                self.master_catalog = self.master_catalog.copy()
                self.master_catalog["fits_name_only"] = self.master_catalog[
                    "fits_name"
                ].str.replace(".gz", "", regex=False)
            else:
                print(
                    "WARNING: 'fits_name' column is missing from the supplied catalogue."
                )

    # ------------------------------------------------------------------
    # Phase 1: Parallel I/O
    # ------------------------------------------------------------------

    def _load_spectrum_data(
        self, file_path: str
    ) -> Optional[Tuple[str, np.ndarray, np.ndarray, np.ndarray]]:
        """Load raw spectrum arrays from a gzipped FITS file (I/O only).

        Designed for ``ThreadPoolExecutor``: ``gzip.open`` and
        ``fits.open`` release the GIL so threads overlap disk requests.

        Parameters
        ----------
        file_path : str
            Relative path within ``self.raw_data_dir``.

        Returns
        -------
        tuple or None
            ``(file_path, wavelength, flux, invvar)`` on success, else ``None``.
        """
        full_path = os.path.join(self.raw_data_dir, file_path)
        try:
            with gzip.open(full_path, "rb") as f_gz:
                with fits.open(f_gz, memmap=False) as hdul:
                    wavelength, flux, invvar = self.preprocessor.load_spectrum(hdul)
                    invvar = sanitize_invvar(invvar)
            return (file_path, wavelength, flux, invvar)
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Legacy single-spectrum method (backward compatibility)
    # ------------------------------------------------------------------

    def _process_single_spectrum(self, file_path: str) -> Optional[Dict[str, float]]:
        """Process one spectrum end-to-end (I/O + CPU, single-threaded).

        Kept for backward compatibility.  For batch processing use
        :meth:`run`.

        Parameters
        ----------
        file_path : str
            Relative path within ``self.raw_data_dir``.

        Returns
        -------
        Optional[Dict[str, float]]
            Feature record or ``None`` on failure.
        """
        result = self._load_spectrum_data(file_path)
        if result is None:
            return None
        return _compute_features_worker(result)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(
        self,
        batch_paths: List[str],
        chunk_size: int = _DEFAULT_CHUNK_SIZE,
    ) -> pd.DataFrame:
        """Process a list of spectra and return a feature DataFrame.

        Per chunk:

        1. **I/O phase** — ``ThreadPoolExecutor`` reads all raw arrays
           concurrently (threads overlap blocking I/O, no IPC pickling).
        2. **CPU phase** — ``ProcessPoolExecutor`` (with worker
           initializer) computes features using true OS-level parallelism,
           bypassing the GIL on Windows and Linux alike.

        A single ``ProcessPoolExecutor`` is kept alive for the whole
        batch so worker processes (and their module imports) are created
        only once.

        Parameters
        ----------
        batch_paths : List[str]
            Relative FITS file paths to process.
        chunk_size : int
            Spectra per chunk (default ``_DEFAULT_CHUNK_SIZE``).

        Returns
        -------
        pd.DataFrame
            Computed and enriched feature table.
        """
        all_features_list: List[Dict[str, float]] = []
        total = len(batch_paths)
        n_chunks = (total + chunk_size - 1) // chunk_size

        print(
            f"\n--- Processing pipeline started for {total} spectra "
            f"({n_chunks} chunk(s) of {chunk_size}) ---"
        )
        print(
            f"    I/O workers : {self.n_io_workers} threads  |  "
            f"CPU workers : {self.n_cpu_workers} processes"
        )

        # Single pool kept alive across all chunks — amortises spawn cost
        with ProcessPoolExecutor(
            max_workers=self.n_cpu_workers,
            initializer=_init_cpu_worker,
            initargs=(self.prominence, self.window),
        ) as cpu_pool:

            for chunk_idx_0, i in enumerate(range(0, total, chunk_size)):
                chunk_paths = batch_paths[i : i + chunk_size]
                chunk_num = chunk_idx_0 + 1
                print(
                    f"\n  >> Chunk {chunk_num}/{n_chunks}  "
                    f"({len(chunk_paths)} spectra)"
                )

                # ── Phase 1: Parallel I/O ──────────────────────────────
                loaded_spectra: List[Tuple] = []
                io_failed = 0

                with ThreadPoolExecutor(
                    max_workers=min(self.n_io_workers, len(chunk_paths))
                ) as io_pool:
                    future_to_path = {
                        io_pool.submit(self._load_spectrum_data, fp): fp
                        for fp in chunk_paths
                    }
                    for future in tqdm(
                        as_completed(future_to_path),
                        total=len(chunk_paths),
                        desc=f"  I/O {chunk_num}/{n_chunks}",
                    ):
                        result = future.result()
                        if result is not None:
                            loaded_spectra.append(result)
                        else:
                            io_failed += 1

                if io_failed:
                    print(f"    I/O phase: {io_failed} files failed to load.")
                print(
                    f"    I/O phase complete: {len(loaded_spectra)} spectra in memory."
                )

                # ── Phase 2: Parallel CPU (processes — bypasses the GIL) ─
                # Only (file_path, wl, flux, invvar) tuples cross the IPC
                # boundary.  Numpy arrays use the buffer protocol → fast.
                # Worker processes reuse module-level globals from _init_cpu_worker.
                raw_results = list(
                    tqdm(
                        cpu_pool.map(
                            _compute_features_worker,
                            loaded_spectra,
                            chunksize=_CPU_CHUNKSIZE,
                        ),
                        total=len(loaded_spectra),
                        desc=f"  CPU {chunk_num}/{n_chunks}",
                    )
                )

                chunk_ok = [r for r in raw_results if r is not None]
                all_features_list.extend(chunk_ok)

                del loaded_spectra, raw_results, chunk_ok, chunk_paths
                gc.collect()

        failed_count = total - len(all_features_list)
        if failed_count > 0:
            print(f"  > WARNING: {failed_count} spectra failed and were skipped.")

        features_df = pd.DataFrame(all_features_list)
        features_df = stabilize_spectral_features(features_df)

        # --- Join with master catalogue (if available) ---
        if (
            features_df.empty
            or self.master_catalog is None
            or self.master_catalog.empty
        ):
            print(
                f"\nProcessing pipeline complete. {len(features_df)} spectra processed."
            )
            if "label" not in features_df.columns:
                features_df["label"] = "UNKNOWN"
            return features_df

        features_df["fits_name_only"] = features_df["file_path"].apply(
            lambda x: os.path.basename(x).replace(".gz", "")
        )

        # 1) Left-join with master catalogue
        cat_to_merge = self.master_catalog.drop(columns=["file_path"], errors="ignore")
        merged_df = pd.merge(features_df, cat_to_merge, how="left", on="fits_name_only")

        # 2) Gaia-derived features
        merged_df, new_gaia_cols = add_gaia_derived_features(
            merged_df, min_parallax_snr=5.0
        )
        print(f"  > Gaia-derived features added: {len(new_gaia_cols)} columns")

        # --- Line composites (Pack B) ---
        try:
            merged_df, new_line_cols = add_line_composites(merged_df)
            print(f"  > Line composites added: {len(new_line_cols)} column(s)")
        except Exception as exc:
            print(f"  > WARNING: add_line_composites failed: {exc}")

        # 2b) Main-sequence offset (delta_ms)
        merged_df, ms_coeffs = add_main_sequence_delta(merged_df, min_parallax_snr=10.0)
        if ms_coeffs is None:
            print(
                "  > delta_ms: not enough clean stars to fit the polynomial (kept NaN)."
            )
        else:
            np.save("delta_ms_poly_coeffs.npy", ms_coeffs)
            print("  > delta_ms added (poly deg=3).")

        # --- Photometric colour indices (g-r, r-i) ---
        print("  > Computing photometric colour features...")
        mag_cols = ["magnitude_g", "magnitude_r", "magnitude_i"]
        for col in mag_cols:
            if col in merged_df.columns:
                merged_df[col] = pd.to_numeric(merged_df[col], errors="coerce").replace(
                    99.0, np.nan
                )

        if {"magnitude_g", "magnitude_r"}.issubset(merged_df.columns):
            merged_df["feature_color_gr"] = (
                merged_df["magnitude_g"] - merged_df["magnitude_r"]
            )
        if {"magnitude_r", "magnitude_i"}.issubset(merged_df.columns):
            merged_df["feature_color_ri"] = (
                merged_df["magnitude_r"] - merged_df["magnitude_i"]
            )

        merged_df.fillna({"feature_color_gr": 0, "feature_color_ri": 0}, inplace=True)

        # --- Photometric composites post-merge ---
        try:
            merged_df, updated_photo_cols = add_photometric_composites(merged_df)
            print(
                f"  > Photometric composites updated: "
                f"{len(updated_photo_cols)} column(s)"
            )
        except Exception as exc:
            print(f"  > WARNING: add_photometric_composites failed: {exc}")

        # --- Final cleanup ---
        cols_to_drop = ["fits_name_only", "dist_pc"]
        nan_cols = [c for c in merged_df.columns if merged_df[c].isna().all()]
        cols_to_drop.extend(nan_cols)
        merged_df = merged_df.drop(columns=cols_to_drop, errors="ignore")
        merged_df = merged_df.replace([-9999, -9999.0], np.nan)

        print(
            f"\nProcessing pipeline complete. "
            f"{len(merged_df)} spectra processed and enriched."
        )
        return merged_df
