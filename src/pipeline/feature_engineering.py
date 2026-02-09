"""
AstroSpectro --- Line-centred spectroscopic feature extraction.

This module provides the ``FeatureEngineer`` class, responsible for:

- defining the vocabulary of target absorption lines and the metrics
  computed per line (prominence, FWHM, equivalent width),
- computing simple spectroscopic band-ratio indices,
- returning feature vectors in a **stable order** (``feature_names``).

Conventions
-----------
- Wavelengths and FWHM are in Angstroms (Å).
- Equivalent widths (EW) are in Å, computed on the normalised spectrum.
- Prominences originate from the upstream peak detector.

Inputs / Outputs
----------------
- Input : ``matched_lines`` (dict line -> ``(detected_wl, prominence)``),
  ``wavelength``, ``flux_norm``, ``invvar`` (inverse-variance).
- Output : ``list[float]`` aligned with ``self.feature_names``.

Public API
----------
- ``extract_features(matched_lines, wavelength, flux_norm, invvar)`` -> list[float]
- ``feature_names`` : canonical feature ordering
- (internal) ``_calculate_spectroscopic_indices(...)`` -> dict of indices

Examples
--------
>>> fe = FeatureEngineer()
>>> vec = fe.extract_features(matched_lines, wl, flux_n, invvar)
>>> names = fe.feature_names
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd
import warnings
from scipy.signal import savgol_filter
from specutils import Spectrum, SpectralRegion
from specutils.analysis import gaussian_fwhm, equivalent_width
from astropy.nddata import StdDevUncertainty
from astropy import units as u

from utils import safe_sigma_from_invvar


class FeatureEngineer:
    """Line-centred spectroscopic feature extractor.

    Defines a vocabulary of target absorption lines and per-line metrics,
    computes a handful of simple spectroscopic indices, and builds a feature
    vector in a stable order (``feature_names``).

    Notes
    -----
    - FWHM values are in Angstroms (Å).
    - Equivalent widths (EW) are in Å, computed on the normalised spectrum.
    - Prominences originate from the upstream peak detector.
    """

    def __init__(self) -> None:
        # 1) Target-line vocabulary and metrics definition
        self.base_lines = ["Hα", "Hβ", "CaII K", "CaII H", "Mg_b", "Na_D"]
        self.line_metrics = ["prominence", "fwhm", "eq_width"]

        # Prominence ratios (numerator, denominator)
        self.ratio_definitions: Dict[str, Tuple[str, str]] = {
            "ratio_prom_CaK_Hbeta": (
                "feature_CaIIK_prominence",
                "feature_Hβ_prominence",
            ),
            "ratio_prom_Halpha_Hbeta": (
                "feature_Hα_prominence",
                "feature_Hβ_prominence",
            ),
            "ratio_prom_Mgb_Hbeta": (
                "feature_Mg_b_prominence",
                "feature_Hβ_prominence",
            ),
        }

        # Spectroscopic indices (feature band, continuum bands), in Å
        #
        # Each entry maps an index name to a pair (feature_band, continuum_bands).
        # The feature band is an interval [λ_min, λ_max].  The continuum bands
        # are a list of sub-intervals [λ_min, λ_max].  This structure allows
        # defining more complex indices with multiple continuum regions
        # (e.g. G4300, Hβ index).  Continuum means are computed on the
        # concatenation of all sub-bands.
        self.index_definitions: Dict[str, Tuple[List[float], List[List[float]]]] = {
            # TiO5 (pre-existing): band 7126–7135 Å / continuum 7042–7052 Å
            "TiO5": ([7126.0, 7135.0], [[7042.0, 7052.0]]),
            # Dn4000 (narrow Balmer break) : 4000–4100 Å / continuum 3850–3950 Å
            "Dn4000": ([4000.0, 4100.0], [[3850.0, 3950.0]]),
            # G4300 (CH band) : 4280–4320 Å / continuum 4260–4280 & 4320–4340 Å
            "G4300": ([4280.0, 4320.0], [[4260.0, 4280.0], [4320.0, 4340.0]]),
            # Ca4227 index : 4225–4235 Å / continuum 4215–4225 & 4235–4245 Å
            "Ca4227": ([4225.0, 4235.0], [[4215.0, 4225.0], [4235.0, 4245.0]]),
            # Hβ index : 4840–4870 Å / continuum 4820–4840 & 4870–4890 Å
            "Hbeta_index": ([4840.0, 4870.0], [[4820.0, 4840.0], [4870.0, 4890.0]]),
            # Mg b index : 5160–5190 Å / continuum 5150–5160 & 5190–5200 Å
            "Mgb_index": ([5160.0, 5190.0], [[5150.0, 5160.0], [5190.0, 5200.0]]),
            # CaH2 : 6814–6846 Å / continuum 7042–7056 Å
            "CaH2": ([6814.0, 6846.0], [[7042.0, 7056.0]]),
            # CaH3 : 6960–6990 Å / continuum 7042–7056 Å
            "CaH3": ([6960.0, 6990.0], [[7042.0, 7056.0]]),
        }

        # 2) Generate the complete, ordered list of feature names
        self.feature_names: List[str] = []
        for line in self.base_lines:
            safe = line.replace(" ", "").replace("II", "II")
            for metric in self.line_metrics:
                self.feature_names.append(f"feature_{safe}_{metric}")
        # NB: this ordering is authoritative for the entire pipeline (column stability).

        self.feature_names += [f"feature_{name}" for name in self.ratio_definitions]
        self.feature_names += [
            f"feature_index_{name}" for name in self.index_definitions
        ]

    # -------------------------------------------------------------------------
    # Spectroscopic indices
    # -------------------------------------------------------------------------

    def _calculate_spectroscopic_indices(
        self, wavelength: np.ndarray, flux_norm: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute spectroscopic band-ratio indices.

        Each index is defined as the ratio of the mean flux in a feature
        band to the mean flux in one or more continuum bands.

        Parameters
        ----------
        wavelength : np.ndarray
            Wavelength array in Angstroms.
        flux_norm : np.ndarray
            Normalised flux array (same length as *wavelength*).

        Returns
        -------
        dict[str, float]
            Dictionary of computed indices, keyed by
            ``"feature_index_<name>"``.  On failure, the value is set to
            ``0.0`` for robustness.
        """
        indices: Dict[str, float] = {}
        for name, (feature_band, continuum_bands) in self.index_definitions.items():
            try:
                # Mask for the feature band
                f_mask = (wavelength >= feature_band[0]) & (
                    wavelength <= feature_band[1]
                )
                # Mean flux in the feature band
                f_vals = flux_norm[f_mask]
                mean_f = float(np.nanmean(f_vals)) if f_vals.size > 0 else 0.0

                # Mask(s) for the continuum bands (concatenated)
                c_vals_list: List[np.ndarray] = []
                # 'continuum_bands' is a list of lists [λ_min, λ_max]
                if isinstance(continuum_bands, list):
                    for band in continuum_bands:
                        if band is None:
                            continue
                        # Support for [start, end] entries
                        if isinstance(band, (list, tuple)) and len(band) == 2:
                            c_mask = (wavelength >= band[0]) & (wavelength <= band[1])
                            c_vals_list.append(flux_norm[c_mask])
                        else:
                            # Fallback case: a single contiguous band
                            c_mask = (wavelength >= continuum_bands[0]) & (
                                wavelength <= continuum_bands[1]
                            )
                            c_vals_list.append(flux_norm[c_mask])
                            break
                else:
                    # Simple case (legacy signature): a single interval [λ_min, λ_max]
                    c_mask = (wavelength >= continuum_bands[0]) & (
                        wavelength <= continuum_bands[1]
                    )
                    c_vals_list.append(flux_norm[c_mask])

                # Concatenate all continuum values
                if c_vals_list:
                    c_concat = np.concatenate(c_vals_list)
                    mean_c = float(np.nanmean(c_concat)) if c_concat.size > 0 else 0.0
                else:
                    mean_c = 0.0

                # Index ratio: feature-band mean / continuum mean
                denom = mean_c + 1e-6  # avoid division by zero
                indices[f"feature_index_{name}"] = mean_f / denom if denom != 0 else 0.0
            except Exception:
                # Robustness: on empty band / out-of-grid mask, keep 0.0
                indices[f"feature_index_{name}"] = 0.0
        return indices

    # -------------------------------------------------------------------------
    # Per-line feature extraction
    # -------------------------------------------------------------------------

    def extract_features(
        self,
        matched_lines: Mapping[str, Optional[Tuple[float, float]]],
        wavelength: np.ndarray,
        flux_norm: np.ndarray,
        invvar: np.ndarray,
    ) -> List[float]:
        """Extract the feature vector for a single spectrum.

        For each line defined in ``base_lines``, retrieves the transmitted
        prominence, estimates the full width at half-maximum (FWHM) via a
        local Gaussian fit, and computes the equivalent width (EW) on the
        normalised spectrum.  Prominence ratios and spectroscopic band
        indices are also computed.  The result follows the ordering defined
        by ``self.feature_names``.

        Parameters
        ----------
        matched_lines : Mapping[str, tuple[float, float] | None]
            Mapping from line name to ``(detected_wavelength, prominence)``
            or ``None`` if the line was not detected.
        wavelength : np.ndarray
            Wavelengths in Angstroms, shape ``(N,)``.
        flux_norm : np.ndarray
            Normalised flux, same length as *wavelength*.
        invvar : np.ndarray
            Inverse-variance of the flux, same length as *wavelength*.

        Returns
        -------
        list[float]
            Feature vector aligned with ``self.feature_names``.
        """
        all_line_features: Dict[str, float] = {}

        # Robust uncertainties (remove / avoid NaN / Inf / neg before sqrt)
        sigma = safe_sigma_from_invvar(invvar)  # utils.safe_sigma_from_invvar
        uncertainty = StdDevUncertainty(sigma)

        spec_full = Spectrum(
            spectral_axis=wavelength * u.AA,
            flux=flux_norm * u.adu,
            uncertainty=uncertainty,
        )

        # 1) Per-line metrics
        for line in self.base_lines:
            line_key = line.replace(" ", "")
            prominence = 0.0
            fwhm_val = 0.0
            eq_width_val = 0.0

            match = matched_lines.get(line)
            if match:
                wl_detected, prom = match
                prominence = float(prom)

                try:
                    # Local window ±20 Å around the detection
                    win = 20.0
                    mask = (wavelength > wl_detected - win) & (
                        wavelength < wl_detected + win
                    )

                    emission_like = None
                    if np.any(mask) and mask.sum() > 5:
                        # Light smoothing to stabilise FWHM
                        smoothed = savgol_filter(
                            flux_norm[mask], window_length=5, polyorder=2
                        )
                        # Inversion (absorption -> "emission") for a positive FWHM
                        emission_like = 1.0 - smoothed

                    with warnings.catch_warnings():
                        warnings.filterwarnings(
                            "ignore",
                            message="invalid value encountered in sqrt",
                            category=RuntimeWarning,
                            module=r".*astropy\.units\.quantity.*",
                        )

                        # Local FWHM if the window was usable
                        if emission_like is not None and np.max(emission_like) > 0:
                            local_spec = Spectrum(
                                spectral_axis=wavelength[mask] * u.AA,
                                flux=emission_like * u.adu,
                            )
                            fwhm_val = float(gaussian_fwhm(local_spec).to_value(u.AA))

                        # Equivalent width (EW) on the normalised spectrum
                        ew = equivalent_width(
                            spec_full,
                            regions=SpectralRegion(
                                (wl_detected - win) * u.AA, (wl_detected + win) * u.AA
                            ),
                        )
                        eq_width_val = float(ew.to_value(u.AA))

                except Exception:
                    pass

            all_line_features[f"feature_{line_key}_prominence"] = float(prominence)
            all_line_features[f"feature_{line_key}_fwhm"] = float(fwhm_val)
            all_line_features[f"feature_{line_key}_eq_width"] = float(eq_width_val)

        # 2) Prominence ratios
        eps = 1e-6
        for name, (num_key, den_key) in self.ratio_definitions.items():
            num = all_line_features.get(num_key, 0.0)
            den = all_line_features.get(den_key, 0.0)
            all_line_features[f"feature_{name}"] = float(num) / (float(den) + eps)

        # 3) Spectroscopic indices (band ratios)
        all_line_features.update(
            self._calculate_spectroscopic_indices(wavelength, flux_norm)
        )

        # 4) Ordered final vector
        return [float(all_line_features.get(name, 0.0)) for name in self.feature_names]

    # -------------------------------------------------------------------------
    # Batch helper (optional)
    # -------------------------------------------------------------------------

    def batch_features(
        self,
        matched_lines_list: List[Mapping[str, Optional[Tuple[float, float]]]],
        wavelength: np.ndarray,
        flux_norm: np.ndarray,
        invvar: np.ndarray,
    ) -> np.ndarray:
        """
        Apply ``extract_features`` over a list of spectra.

        Parameters
        ----------
        matched_lines_list : list[Mapping[str, tuple[float, float] | None]]
            Peak annotations, one entry per spectrum.
        wavelength : np.ndarray
            Common wavelength grid.
        flux_norm : np.ndarray
            Common normalised flux array.
        invvar : np.ndarray
            Common inverse-variance array.

        Returns
        -------
        np.ndarray
            Matrix of shape ``(n_spectra, n_features)`` aligned with
            ``feature_names``.
        """
        rows = [
            self.extract_features(ml, wavelength, flux_norm, invvar)
            for ml in matched_lines_list
        ]
        return np.asarray(rows, dtype=float)


def add_gaia_derived_features(
    df: pd.DataFrame, *, min_parallax_snr: float = 5.0
) -> tuple[pd.DataFrame, list[str]]:
    """
    Add Gaia-derived columns to the feature DataFrame.

    Creates simple colours, flux ratios, absolute magnitudes, variability
    indicators, and various flags based on the Gaia DR3 columns present in
    *df*.  New column names are collected and returned alongside the
    enriched DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing Gaia columns.
    min_parallax_snr : float, optional
        Minimum parallax signal-to-noise ratio required for a reliable
        absolute magnitude estimate (default: 5.0).

    Returns
    -------
    df : pd.DataFrame
        The original DataFrame augmented with new columns.
    new_cols : list[str]
        Names of the newly added columns.
    """
    df = df.copy()
    new_cols: list[str] = []
    cols = set(df.columns)

    # ---------------- Simple colours ----------------
    if {"phot_bp_mean_mag", "phot_g_mean_mag"} <= cols and "bp_g" not in df:
        df["bp_g"] = df["phot_bp_mean_mag"] - df["phot_g_mean_mag"]
        new_cols.append("bp_g")
    if {"phot_g_mean_mag", "phot_rp_mean_mag"} <= cols and "g_rp" not in df:
        df["g_rp"] = df["phot_g_mean_mag"] - df["phot_rp_mean_mag"]
        new_cols.append("g_rp")

    # ---------------- Parallax SNR ----------------
    if {"parallax", "parallax_error"} <= cols:
        snr = pd.Series(np.nan, index=df.index)
        ok = df["parallax_error"].notna() & (df["parallax_error"] > 0)
        snr.loc[ok] = df.loc[ok, "parallax"] / df.loc[ok, "parallax_error"]
        df["parallax_snr"] = snr
        new_cols.append("parallax_snr")
    else:
        df["parallax_snr"] = np.nan
        new_cols.append("parallax_snr")

    # ---------------- Absolute magnitude G ----------------
    # A) via parallax (mas) if SNR is sufficient
    if {"phot_g_mean_mag", "parallax"} <= cols:
        G = df["phot_g_mean_mag"]
        p = df["parallax"]
        snr_ok = p.notna() & (p > 0) & (df["parallax_snr"] >= min_parallax_snr)
        M_G_par = pd.Series(np.nan, index=df.index)
        with np.errstate(divide="ignore", invalid="ignore"):
            M_G_par.loc[snr_ok] = G.loc[snr_ok] - 10.0 + 5.0 * np.log10(p.loc[snr_ok])
        df["M_G_parallax"] = M_G_par
        new_cols.append("M_G_parallax")

    # B) via distance_gspphot (pc)
    if {"phot_g_mean_mag", "distance_gspphot"} <= cols:
        G = df["phot_g_mean_mag"]
        d = df["distance_gspphot"]
        ok = d.notna() & (d > 0)
        M_G_dist = pd.Series(np.nan, index=df.index)
        with np.errstate(divide="ignore", invalid="ignore"):
            M_G_dist.loc[ok] = G.loc[ok] - 5.0 * np.log10(d.loc[ok] / 10.0)
        df["M_G_dist"] = M_G_dist
        new_cols.append("M_G_dist")

    # C) Merge prioritising reliable parallax
    if "M_G_parallax" in df.columns or "M_G_dist" in df.columns:
        df["M_G"] = df.get("M_G_parallax", pd.Series(np.nan, index=df.index))
        if "M_G_dist" in df.columns:
            use_dist = df["M_G"].isna() & df["M_G_dist"].notna()
            df.loc[use_dist, "M_G"] = df.loc[use_dist, "M_G_dist"]
        new_cols.append("M_G")

    # ---------------- RPM & tangential velocity ----------------
    if {"pmra", "pmdec"} <= cols:
        mu_mas = np.sqrt(df["pmra"] ** 2 + df["pmdec"] ** 2)
        df["pm_total"] = mu_mas
        new_cols.append("pm_total")
        if "phot_g_mean_mag" in df.columns:
            mu_as = mu_mas / 1000.0
            H = pd.Series(np.nan, index=df.index)
            ok = mu_as.notna() & (mu_as > 0) & df["phot_g_mean_mag"].notna()
            with np.errstate(divide="ignore", invalid="ignore"):
                H.loc[ok] = (
                    df.loc[ok, "phot_g_mean_mag"] + 5.0 * np.log10(mu_as.loc[ok]) + 5.0
                )
            df["H_G"] = H
            new_cols.append("H_G")

        Vt = pd.Series(np.nan, index=df.index)
        if "distance_gspphot" in df.columns:
            d_pc = df["distance_gspphot"]
            mu_as = mu_mas / 1000.0
            ok = mu_as.notna() & (mu_as > 0) & d_pc.notna() & (d_pc > 0)
            Vt.loc[ok] = 4.74047 * mu_as.loc[ok] * d_pc.loc[ok]
        elif "parallax" in df.columns:
            p = df["parallax"]
            ok = mu_mas.notna() & (mu_mas > 0) & p.notna() & (p > 0)
            Vt.loc[ok] = 4.74047 * mu_mas.loc[ok] / p.loc[ok]
        df["v_tan_kms"] = Vt
        new_cols.append("v_tan_kms")

    # ---------------- Extinction & de-reddened ("0") colours ----------------
    if {"phot_g_mean_mag", "ag_gspphot"} <= cols:
        G0 = pd.Series(np.nan, index=df.index)
        ok = df["phot_g_mean_mag"].notna() & df["ag_gspphot"].notna()
        G0.loc[ok] = df.loc[ok, "phot_g_mean_mag"] - df.loc[ok, "ag_gspphot"]
        df["G0"] = G0
        new_cols.append("G0")

    if {"bp_rp", "ebpminrp_gspphot"} <= cols:
        bp_rp0 = pd.Series(np.nan, index=df.index)
        ok = df["bp_rp"].notna() & df["ebpminrp_gspphot"].notna()
        bp_rp0.loc[ok] = df.loc[ok, "bp_rp"] - df.loc[ok, "ebpminrp_gspphot"]
        df["bp_rp0"] = bp_rp0
        new_cols.append("bp_rp0")

    if "M_G" in df.columns and "ag_gspphot" in df.columns:
        MG0 = pd.Series(np.nan, index=df.index)
        ok = df["M_G"].notna() & df["ag_gspphot"].notna()
        MG0.loc[ok] = df.loc[ok, "M_G"] - df.loc[ok, "ag_gspphot"]
        df["M_G0"] = MG0
        new_cols.append("M_G0")

    # ---------------- Flux ratios / logs ----------------
    for num, den, out in [
        ("phot_bp_mean_flux", "phot_rp_mean_flux", "flux_bp_rp_ratio"),
        ("phot_g_mean_flux", "phot_rp_mean_flux", "flux_g_rp_ratio"),
        ("phot_bp_mean_flux", "phot_g_mean_flux", "flux_bp_g_ratio"),
    ]:
        if {num, den} <= cols:
            r = pd.Series(np.nan, index=df.index)
            x, y = df[num], df[den]
            ok = x.notna() & y.notna() & (y != 0)
            r.loc[ok] = x.loc[ok] / y.loc[ok]
            df[out] = r
            new_cols.append(out)

            out_log = out + "_log10"
            lr = pd.Series(np.nan, index=df.index)
            ok2 = r.notna() & (r > 0)
            lr.loc[ok2] = np.log10(r.loc[ok2])
            df[out_log] = lr
            new_cols.append(out_log)

    # ---------------- Simple photometric quality ----------------
    if {"phot_bp_rp_excess_factor", "bp_rp"} <= cols:
        exp = 1.0 + 0.015 * (df["bp_rp"] ** 2)
        df["bp_rp_excess_dev"] = df["phot_bp_rp_excess_factor"] - exp
        new_cols.append("bp_rp_excess_dev")

    # ---------------- Useful flags (cast -> int) ----------------
    if "ruwe" in df.columns:
        df["is_good_ruwe"] = ((df["ruwe"].notna()) & (df["ruwe"] < 1.4)).astype("int8")
        new_cols.append("is_good_ruwe")
    if "astrometric_excess_noise" in df.columns:
        df["has_astrom_excess"] = (df["astrometric_excess_noise"].fillna(0) > 0).astype(
            "int8"
        )
        new_cols.append("has_astrom_excess")
    if "phot_variable_flag" in df.columns:
        df["is_variable_flag"] = (
            df["phot_variable_flag"]
            .fillna("")
            .str.upper()
            .ne("NOT_AVAILABLE")
            .astype("int8")
        )
        new_cols.append("is_variable_flag")

    # ---------------- Missing-value indicators ----------------
    for col in ("parallax", "distance_gspphot"):
        if col in df.columns:
            name = f"{col}_missing"
            df[name] = df[col].isna().astype("int8")
            new_cols.append(name)

    # ---------------- Pack A: enrichissements Gaia “+” ----------------
    # Photometric SNRs (and log10)
    for band in ("g", "bp", "rp"):
        f = f"phot_{band}_mean_flux_over_error"
        if f in df.columns:
            s = pd.to_numeric(df[f], errors="coerce")
            df[f"{band}_snr"] = s
            # log10 only for positive SNRs
            df[f"{band}_snr_log10"] = np.where(s > 0, np.log10(s), np.nan)
            new_cols += [f"{band}_snr", f"{band}_snr_log10"]

    # Absolute magnitude for BP / RP computed like M_G
    def _abs_mag(band: str) -> None:
        """Compute absolute magnitude M_BP or M_RP via parallax or distance."""
        name_par, name_dist = f"M_{band.upper()}_parallax", f"M_{band.upper()}_dist"
        mcol = f"phot_{band}_mean_mag"
        # via parallax if available
        if {mcol, "parallax"}.issubset(df.columns):
            ok = (
                df["parallax"].notna()
                & (df["parallax"] > 0)
                & (df["parallax_snr"] >= 5)
            )
            with np.errstate(divide="ignore", invalid="ignore"):
                df[name_par] = np.where(
                    ok,
                    df[mcol] - 10.0 + 5.0 * np.log10(df["parallax"]),
                    np.nan,
                )
            new_cols.append(name_par)

        if {mcol, "distance_gspphot"}.issubset(df.columns):
            ok = df["distance_gspphot"].notna() & (df["distance_gspphot"] > 0)
            with np.errstate(divide="ignore", invalid="ignore"):
                df[name_dist] = np.where(
                    ok,
                    df[mcol] - 5.0 * np.log10(df["distance_gspphot"] / 10.0),
                    np.nan,
                )
            new_cols.append(name_dist)
        if (name_par in df.columns) or (name_dist in df.columns):
            M_col = f"M_{band.upper()}"
            df[M_col] = df.get(name_par, pd.Series(np.nan, index=df.index))
            if name_dist in df.columns:
                use = df[M_col].isna() & df[name_dist].notna()
                df.loc[use, M_col] = df.loc[use, name_dist]
            new_cols.append(M_col)

    # Compute absolute magnitude for BP / RP
    for band in ("bp", "rp"):
        if f"phot_{band}_mean_mag" in df.columns:
            _abs_mag(band)

    # Absolute colour and evolutionary flags
    if {"M_BP", "M_RP"}.issubset(df.columns):
        df["M_BP_minus_M_RP"] = df["M_BP"] - df["M_RP"]
        new_cols.append("M_BP_minus_M_RP")

    if "delta_ms" in df.columns:
        df["is_giant_like"] = (df["delta_ms"] <= -0.8).astype("int8")
        df["is_subdwarf_like"] = (df["delta_ms"] >= +0.6).astype("int8")
        new_cols += ["is_giant_like", "is_subdwarf_like"]

    # Tangential velocity
    if "v_tan_kms" in df.columns:
        x = df["v_tan_kms"]
        df["v_tan_kms_log10"] = np.where(x > 0, np.log10(x), np.nan)
        new_cols.append("v_tan_kms_log10")
    if "parallax_snr" in df.columns:
        s = df["parallax_snr"]
        df["parallax_snr_log10"] = np.where(s > 0, np.log10(s), np.nan)
        new_cols.append("parallax_snr_log10")

    return df, new_cols


def add_line_composites(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Compute simple combinations of equivalent widths (EW) and FWHM to capture
    Balmer-to-metal ratios and an average depth proxy.

    Inspects existing columns to determine standard line names and produces
    new derived columns.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing line columns (e.g. ``feature_Ha_eq_width``).

    Returns
    -------
    df : pd.DataFrame
        Enriched DataFrame.
    new : list[str]
        Names of the newly added columns.
    """
    df = df.copy()
    new: list[str] = []

    def pick(*names: str) -> Optional[str]:
        """Return the first column name found in *df* from the proposed names."""
        return next((n for n in names if n in df.columns), None)

    # Tolerant aliases (ASCII / Greek).  EWs are positive for absorption.
    Ha_ew = pick("feature_Ha_eq_width", "feature_Hα_eq_width")
    Hb_ew = pick("feature_Hb_eq_width", "feature_Hβ_eq_width")
    Ha_fwhm = pick("feature_Ha_fwhm", "feature_Hα_fwhm")
    Hb_fwhm = pick("feature_Hb_fwhm", "feature_Hβ_fwhm")
    CaK_ew = pick("feature_CaIIK_eq_width")
    CaH_ew = pick("feature_CaIIH_eq_width")
    MgB_ew = pick("feature_Mg_b_eq_width", "feature_Mgb_eq_width")
    NaD_ew = pick("feature_Na_D_eq_width", "feature_NaD_eq_width")

    # --- Balmer ratios ---
    if Ha_ew and Hb_ew:
        a = df[Ha_ew].abs()
        b = df[Hb_ew].abs()
        # ratio EW Ha / Hb
        r = np.where((b.notna()) & (b != 0), a / b, np.nan)
        df["ratio_EW_Ha_Hb"] = r
        new.append("ratio_EW_Ha_Hb")
    if Ha_fwhm and Hb_fwhm:
        # ratio FWHM Ha / Hb
        r = np.where(
            (df[Hb_fwhm].notna()) & (df[Hb_fwhm] != 0),
            df[Ha_fwhm] / df[Hb_fwhm],
            np.nan,
        )
        df["ratio_FWHM_Ha_Hb"] = r
        new.append("ratio_FWHM_Ha_Hb")

    # --- Average depth ~ EW / FWHM (proxy) for Hα / Hβ ---
    if Ha_ew and Ha_fwhm:
        df["depthproxy_Ha"] = np.where(
            df[Ha_fwhm] > 0, df[Ha_ew].abs() / df[Ha_fwhm], np.nan
        )
        new.append("depthproxy_Ha")
    if Hb_ew and Hb_fwhm:
        df["depthproxy_Hb"] = np.where(
            df[Hb_fwhm] > 0, df[Hb_ew].abs() / df[Hb_fwhm], np.nan
        )
        new.append("depthproxy_Hb")

    # --- Metals: Ca H/K, Mg b, Na D ---
    if CaH_ew and CaK_ew:
        # sum of CaHK EWs
        df["EW_CaHK_sum"] = df[CaH_ew].abs() + df[CaK_ew].abs()
        new.append("EW_CaHK_sum")
        # ratio CaK / CaH
        df["ratio_EW_CaK_CaH"] = np.where(
            df[CaH_ew].abs() > 0, df[CaK_ew].abs() / df[CaH_ew].abs(), np.nan
        )
        new.append("ratio_EW_CaK_CaH")
    if MgB_ew and NaD_ew:
        # ratio Mg b / Na D
        df["ratio_EW_MgB_NaD"] = np.where(
            df[NaD_ew].abs() > 0, df[MgB_ew].abs() / df[NaD_ew].abs(), np.nan
        )
        new.append("ratio_EW_MgB_NaD")

    # --- Metals vs Balmer contrast ---
    if (CaH_ew and CaK_ew) and (Ha_ew and Hb_ew):
        metals = df[CaH_ew].abs() + df[CaK_ew].abs()
        balmer = df[Ha_ew].abs() + df[Hb_ew].abs()
        df["contrast_metals_vs_balmer"] = np.where(balmer > 0, metals / balmer, np.nan)
        new.append("contrast_metals_vs_balmer")

    return df, new


def add_main_sequence_delta(
    df: pd.DataFrame, *, min_parallax_snr: float = 10.0, poly_degree: int = 3
) -> tuple[pd.DataFrame, np.ndarray | None]:
    """
    Add ``delta_ms = M_G0 - M_G0_hat((BP-RP)_0)`` to the DataFrame.

    ``M_G0_hat`` is a polynomial fitted on "clean" stars (good RUWE,
    high parallax SNR).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with at least ``bp_rp0``, ``M_G0``, ``parallax_snr``,
        ``is_good_ruwe``.
    min_parallax_snr : float, optional
        Minimum parallax SNR for clean-star selection (default: 10.0).
    poly_degree : int, optional
        Polynomial degree for the main-sequence fit (default: 3).

    Returns
    -------
    df : pd.DataFrame
        Modified DataFrame with a ``delta_ms`` column.
    coeffs : np.ndarray or None
        Polynomial coefficients, or ``None`` if too few clean stars.
    """
    df = df.copy()
    needed = {"bp_rp0", "M_G0", "parallax_snr", "is_good_ruwe"}
    if not needed.issubset(df.columns):
        df["delta_ms"] = np.nan
        return df, None

    mask = (
        (df["is_good_ruwe"] == 1)
        & (df["parallax_snr"] >= min_parallax_snr)
        & df["bp_rp0"].between(-0.2, 3.5)
        & df["M_G0"].between(-5, 15)
    )

    x = df.loc[mask, "bp_rp0"].astype(float)
    y = df.loc[mask, "M_G0"].astype(float)

    if x.count() < 200:  # safety threshold
        df["delta_ms"] = np.nan
        return df, None

    coeffs = np.polyfit(x, y, poly_degree)
    yhat = np.polyval(coeffs, df["bp_rp0"])
    df["delta_ms"] = df["M_G0"] - yhat  # <0: over-luminous (giants), >0: under-luminous

    return df, coeffs


def _signed_log1p(x: pd.Series) -> pd.Series:
    return np.sign(x) * np.log1p(np.abs(x))


def stabilize_spectral_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    ew_cols = [c for c in df.columns if c.endswith("_eq_width")]
    prom_cols = [c for c in df.columns if c.endswith("_prominence")]
    fwhm_cols = [c for c in df.columns if c.endswith("_fwhm")]

    # 2.1 Soft winsorisation (clip extremes)
    def clip_col(s: pd.Series):
        lo, hi = s.quantile(0.005), s.quantile(0.995)
        return s.clip(lower=lo, upper=hi)

    for cols, transform in [
        (ew_cols, _signed_log1p),  # signed-log for EW
        (prom_cols, np.log1p),  # log1p for prominences (>=0)
        (fwhm_cols, np.log1p),  # log1p for FWHM (>=0)
    ]:
        for c in cols:
            df[c] = clip_col(df[c])
            df[c] = transform(df[c].fillna(0.0).astype(float))

    # 2.2 Binary flags "line present"
    for c in prom_cols:
        flag = c.replace("_prominence", "_present")
        df[flag] = (df[c] > 0).astype(np.uint8)

    return df
