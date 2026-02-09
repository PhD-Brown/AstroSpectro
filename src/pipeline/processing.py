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
- ``feature_*`` vectors are returned in a **stable order**
  (``feature_names`` list).

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

from typing import Dict, List

import os
import gzip
import numpy as np
import pandas as pd
from astropy.io import fits
from tqdm import tqdm

from .preprocessor import SpectraPreprocessor
from .peak_detector import PeakDetector, matches_to_series
from .feature_engineering import (
    FeatureEngineer,
    add_gaia_derived_features,
    add_main_sequence_delta,
    stabilize_spectral_features,
)
from utils import sanitize_invvar


class ProcessingPipeline:
    """Execute the full processing pipeline for each FITS spectrum.

    This pipeline applies preprocessing, line detection, feature extraction,
    photometric feature addition, and result aggregation into a DataFrame.

    Parameters
    ----------
    raw_data_dir : str
        Directory containing the FITS files (``.fits.gz``).
    master_catalog_df : pd.DataFrame
        Master metadata catalogue used to enrich spectra with photometric
        or Gaia information.

    Attributes
    ----------
    preprocessor : SpectraPreprocessor
        Utility for loading and normalising spectra.
    peak_detector : PeakDetector
        Absorption-line detection and association engine.
    feature_engineer : FeatureEngineer
        Spectroscopic feature extractor.
    master_catalog : pd.DataFrame
        Enriched catalogue (optional) for join operations.
    """

    def __init__(self, raw_data_dir: str, master_catalog_df: pd.DataFrame) -> None:
        self.raw_data_dir = raw_data_dir
        self.preprocessor = SpectraPreprocessor()
        # Default thresholds (higher = fewer false positives)
        self.peak_detector = PeakDetector(prominence=0.15, window=8)
        self.feature_engineer = FeatureEngineer()

        self.master_catalog = master_catalog_df
        if self.master_catalog is not None and not self.master_catalog.empty:
            if "fits_name" in self.master_catalog.columns:
                # Prepare a simple join key (without .gz)
                self.master_catalog = self.master_catalog.copy()
                self.master_catalog["fits_name_only"] = self.master_catalog[
                    "fits_name"
                ].str.replace(".gz", "", regex=False)
            else:
                print(
                    "WARNING: Column 'fits_name' is missing from the supplied catalogue."
                )

    def run(self, batch_paths: List[str]) -> pd.DataFrame:
        """Process a batch of spectra and return a feature DataFrame.

        For each path, loads the spectrum, normalises the flux, detects
        absorption lines, extracts spectroscopic features, computes colour
        features, and optionally joins information from the master
        catalogue.  Features are stabilised and enriched with Gaia-derived
        columns and line composites.

        Parameters
        ----------
        batch_paths : list[str]
            Relative paths (within ``raw_data_dir``) of the FITS files to
            process.

        Returns
        -------
        pd.DataFrame
            Feature table computed and enriched for all spectra in the batch.
        """
        all_features_list: List[Dict[str, float]] = []

        print(f"\n--- Starting processing pipeline for {len(batch_paths)} spectra ---")
        for file_path in tqdm(batch_paths, desc="Processing spectra"):
            full_fits_path = os.path.join(self.raw_data_dir, file_path)
            try:
                # 1) Loading + preprocessing
                with gzip.open(full_fits_path, "rb") as f_gz:
                    with fits.open(f_gz, memmap=False) as hdul:
                        wavelength, flux, invvar = self.preprocessor.load_spectrum(hdul)
                        # Numerical robustness: avoid NaN/inf/<=0 before sqrt()
                        invvar = sanitize_invvar(invvar)

                flux_norm = self.preprocessor.normalize_spectrum(flux)

                # 2) Line detection / association
                matched_lines = self.peak_detector.analyze_spectrum(
                    wavelength, flux_norm
                )

                # 3) Feature engineering (prominence/FWHM/EW/indices)
                features_vector = self.feature_engineer.extract_features(
                    matched_lines, wavelength, flux_norm, invvar
                )

                # 3.b) Additional columns from line matching (stable & interpretable)
                order = self.feature_engineer.base_lines  # canonical line order
                match_series = matches_to_series(
                    matched_lines, order=order, prefix="match_"
                )
                # Harmonise names (no spaces) for consistency
                match_series = match_series.rename(lambda c: c.replace(" ", ""))

                # 4) Feature couleur “Blue/Red” (simple proxy de pente)
                flux_blue = float(
                    np.nanmean(flux_norm[(wavelength > 4000) & (wavelength < 4500)])
                )
                flux_red = float(
                    np.nanmean(flux_norm[(wavelength > 6500) & (wavelength < 7000)])
                )
                color_index = flux_blue / (flux_red + 1e-6)

                # 4.b) Global continuum shape: slope and curvature (Pack D)
                # Fit a degree-2 polynomial over the continuum range
                slope = 0.0
                curvature = 0.0
                try:
                    cont_mask = (wavelength > 4300) & (wavelength < 6800)
                    lam = wavelength[cont_mask]
                    flx = flux_norm[cont_mask]
                    # Use only finite points
                    valid = np.isfinite(lam) & np.isfinite(flx)
                    lam = lam[valid]
                    flx = flx[valid]
                    if lam.size >= 10:
                        # Fit a degree-2 polynomial: flx ~ a*lam^2 + b*lam + c
                        coeffs = np.polyfit(lam, flx, deg=2)
                        curvature = float(coeffs[0])
                        slope = float(coeffs[1])
                except Exception:
                    # Leave slope/curvature at 0 if the fit fails
                    pass

                # 5) Row assembly
                record: Dict[str, float] = {"file_path": file_path}

                # features FE (dans l’ordre canonique)
                for feature_name, feature_val in zip(
                    self.feature_engineer.feature_names, features_vector
                ):
                    record[feature_name] = float(feature_val)

                # local colour feature
                record["feature_color_index_BlueRed"] = float(color_index)
                # continuum shape: slope and curvature
                record["feature_cont_slope"] = float(slope)
                record["feature_cont_curvature"] = float(curvature)

                # columns from matching (wl/prom per line)
                record.update(
                    {
                        k: float(v) if pd.notna(v) else np.nan
                        for k, v in match_series.to_dict().items()
                    }
                )

                all_features_list.append(record)

            except Exception as e:
                print(f"\n    -> ERROR processing {file_path}: {e}")

        features_df = pd.DataFrame(all_features_list)
        features_df = stabilize_spectral_features(features_df)

        # --- Join with master catalogue (if present) ---
        if (
            features_df.empty
            or self.master_catalog is None
            or self.master_catalog.empty
        ):
            print(
                f"\nProcessing pipeline finished. {len(features_df)} spectra processed."
            )
            if "label" not in features_df.columns:
                features_df["label"] = "UNKNOWN"
            return features_df

        # Join key on the features side
        features_df["fits_name_only"] = features_df["file_path"].apply(
            lambda x: os.path.basename(x).replace(".gz", "")
        )

        # 1) Join first
        merged_df = pd.merge(
            features_df, self.master_catalog, how="left", on="fits_name_only"
        )

        # 2) Then Gaia-derived features (columns now exist)
        merged_df, new_gaia_cols = add_gaia_derived_features(
            merged_df, min_parallax_snr=5.0
        )
        print(f"  > Gaia-derived features added: {len(new_gaia_cols)} columns")

        # --- Line composites (Pack B) ---
        try:
            from .feature_engineering import add_line_composites

            merged_df, new_line_cols = add_line_composites(merged_df)
            print(f"  > Line composites added: {len(new_line_cols)} column(s)")
        except Exception as exc:
            print(f"  > WARNING: add_line_composites failed: {exc}")

        # 2bis) Main-sequence offset (delta_ms)
        merged_df, ms_coeffs = add_main_sequence_delta(merged_df, min_parallax_snr=10.0)
        if ms_coeffs is None:
            print(
                "  > delta_ms: not enough clean stars to fit the polynomial (kept as NaN)."
            )
        else:
            np.save("delta_ms_poly_coeffs.npy", ms_coeffs)
            print("  > delta_ms added (poly deg=3).")

        # --- Photometric colour indices (g-r, r-i) ---
        print("  > Creating photometric colour features...")

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

        # Final cleanup
        merged_df = merged_df.drop(columns=["fits_name_only"], errors="ignore")

        print(
            f"\nProcessing pipeline finished. {len(merged_df)} spectra processed and enriched."
        )
        return merged_df
