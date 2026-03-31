"""AstroSpectro — Physics-based spectroscopic feature extraction (V2).

This module provides the ``FeatureEngineer`` class (unified V1 + V2),
responsible for:

- 174 physics-based features for LAMOST DR5 stellar classification.
- Molecular bands, extended Balmer series, metallic lines, spectral
  indices, continuum analysis, line profiles, and synthetic colours.

The exact feature count is determined automatically on the first call
to ``extract_features`` via ``_populate_feature_names``.

Feature Categories (~174 features)
----------------------------------
- V1 base (lines + ratios + indices)                  : 29 features
- Molecular bands (TiO, VO, CN, CH, CaH, MgH)        : 15 features
- Extended Balmer series (Hγ → H10, ratios, gradient) : 16 features
- Metallic lines (Fe, Mg, Ca, Na, Si, Ti, Cr, Ba, Sr) : 39 features
- Extended spectral indices (Lick, SDSS, CaII triplet) : 10 features
- Continuum analysis (slopes, curvatures, jumps)       : 18 features
- Line profiles (Hα, CaII K)                          : 25 features
- Colours (synthetic + photometric post-merge)         : 7 features
- Composite indices (Teff, logg, [Fe/H] proxies)       : 15 features

Architecture
------------
``extract_features`` is the main entry point (one spectrum → stable
feature vector).  The helper functions ``add_gaia_derived_features``,
``add_photometric_composites``, ``add_line_composites``,
``add_main_sequence_delta``, and ``stabilize_spectral_features`` are
post-processing hooks called by ``processing.py`` after the catalogue
join.

Examples
--------
>>> fe = FeatureEngineer()
>>> vec = fe.extract_features(matched_lines, wl, flux_norm, invvar)
>>> len(fe.feature_names)   # determined at runtime 174
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd
import warnings

from astropy import units as u
from astropy.nddata import StdDevUncertainty
from astropy.stats import sigma_clip
from scipy.signal import savgol_filter
from scipy.stats import kurtosis, skew
from specutils import Spectrum, SpectralRegion
from specutils.analysis import equivalent_width, gaussian_fwhm

from utils import safe_sigma_from_invvar

# NumPy compatibility: trapz → trapezoid (NumPy 2.0+)
try:
    _np_trapz = np.trapezoid  # NumPy >= 2.0
except AttributeError:
    _np_trapz = np.trapz  # NumPy < 2.0

# Suppress cosmetic RuntimeWarnings from np.nanmean on all-NaN arrays
# (expected behaviour when spectral bands fall outside the wavelength grid).
warnings.filterwarnings("ignore", "Mean of empty slice", RuntimeWarning)
warnings.filterwarnings("ignore", "All-NaN slice encountered", RuntimeWarning)
warnings.filterwarnings("ignore", "Degrees of freedom <= 0", RuntimeWarning)
warnings.filterwarnings("ignore", "divide by zero encountered", RuntimeWarning)
warnings.filterwarnings("ignore", "invalid value encountered", RuntimeWarning)


# ============================================================================
# SPECTRAL LINE VOCABULARY
# ============================================================================


@dataclass
class LineDefinition:
    """Metadata for a single spectral line."""

    name: str
    wavelength: float  # Centre in Å
    species: str
    category: str
    strength: str  # 'strong' | 'medium' | 'weak'


class SpectralLineLibrary:
    """Centralised library of all spectral lines used."""

    MOLECULAR_BANDS: Dict[str, LineDefinition] = {
        "TiO_6180": LineDefinition("TiO_6180", 6205.0, "TiO", "molecular", "medium"),
        "TiO_7050": LineDefinition("TiO_7050", 7100.0, "TiO", "molecular", "strong"),
        "TiO_7600": LineDefinition("TiO_7600", 7650.0, "TiO", "molecular", "medium"),
        "TiO_8200": LineDefinition("TiO_8200", 8250.0, "TiO", "molecular", "medium"),
        "TiO_8400": LineDefinition("TiO_8400", 8450.0, "TiO", "molecular", "medium"),
        "TiO_8860": LineDefinition("TiO_8860", 8885.0, "TiO", "molecular", "weak"),
        "VO_7400": LineDefinition("VO_7400", 7475.0, "VO", "molecular", "medium"),
        "VO_7900": LineDefinition("VO_7900", 7975.0, "VO", "molecular", "medium"),
        "CN_4142": LineDefinition("CN_4142", 4160.0, "CN", "molecular", "medium"),
        "CH_4300": LineDefinition("CH_4300", 4305.0, "CH", "molecular", "medium"),
        "CaH_6380": LineDefinition("CaH_6380", 6385.0, "CaH", "molecular", "weak"),
        "CaH_6830": LineDefinition("CaH_6830", 6835.0, "CaH", "molecular", "weak"),
        "CaH_6975": LineDefinition("CaH_6975", 6980.0, "CaH", "molecular", "weak"),
        "MgH_5140": LineDefinition("MgH_5140", 5145.0, "MgH", "molecular", "weak"),
    }

    BALMER_SERIES: Dict[str, LineDefinition] = {
        "Halpha": LineDefinition("Halpha", 6562.8, "H I", "balmer", "strong"),
        "Hbeta": LineDefinition("Hbeta", 4861.3, "H I", "balmer", "strong"),
        "Hgamma": LineDefinition("Hgamma", 4340.5, "H I", "balmer", "medium"),
        "Hdelta": LineDefinition("Hdelta", 4101.7, "H I", "balmer", "medium"),
        "Hepsilon": LineDefinition("Hepsilon", 3970.1, "H I", "balmer", "weak"),
        "H8": LineDefinition("H8", 3889.0, "H I", "balmer", "weak"),
        "H9": LineDefinition("H9", 3835.4, "H I", "balmer", "weak"),
        "H10": LineDefinition("H10", 3797.9, "H I", "balmer", "weak"),
    }

    METAL_LINES: Dict[str, LineDefinition] = {
        "Fe_4383": LineDefinition("Fe_4383", 4383.5, "Fe I", "metal", "medium"),
        "Fe_4531": LineDefinition("Fe_4531", 4531.1, "Fe II", "metal", "weak"),
        "Fe_5270": LineDefinition("Fe_5270", 5270.4, "Fe I", "metal", "medium"),
        "Fe_5335": LineDefinition("Fe_5335", 5335.2, "Fe I", "metal", "medium"),
        "Fe_5406": LineDefinition("Fe_5406", 5406.8, "Fe I", "metal", "weak"),
        "Fe_5709": LineDefinition("Fe_5709", 5709.4, "Fe I", "metal", "weak"),
        "Mg_5167": LineDefinition("Mg_5167", 5167.3, "Mg I", "metal", "medium"),
        "Mg_5173": LineDefinition("Mg_5173", 5172.7, "Mg I", "metal", "medium"),
        "Mg_5184": LineDefinition("Mg_5184", 5183.6, "Mg I", "metal", "medium"),
        "CaII_K": LineDefinition("CaII_K", 3933.7, "Ca II", "metal", "strong"),
        "CaII_H": LineDefinition("CaII_H", 3968.5, "Ca II", "metal", "strong"),
        "Ca_8498": LineDefinition("Ca_8498", 8498.0, "Ca II", "metal", "strong"),
        "Ca_8542": LineDefinition("Ca_8542", 8542.1, "Ca II", "metal", "strong"),
        "Ca_8662": LineDefinition("Ca_8662", 8662.1, "Ca II", "metal", "strong"),
        "Na_D1": LineDefinition("Na_D1", 5895.9, "Na I", "metal", "medium"),
        "Na_D2": LineDefinition("Na_D2", 5889.9, "Na I", "metal", "medium"),
        "Si_4128": LineDefinition("Si_4128", 4128.1, "Si II", "metal", "medium"),
        "Si_4131": LineDefinition("Si_4131", 4130.9, "Si II", "metal", "medium"),
        "Ti_4758": LineDefinition("Ti_4758", 4758.1, "Ti II", "metal", "weak"),
        "Ti_4764": LineDefinition("Ti_4764", 4764.9, "Ti II", "metal", "weak"),
        "Cr_5206": LineDefinition("Cr_5206", 5206.0, "Cr I", "metal", "weak"),
        "Cr_5208": LineDefinition("Cr_5208", 5208.4, "Cr I", "metal", "weak"),
        "Ni_5081": LineDefinition("Ni_5081", 5081.1, "Ni I", "metal", "weak"),
        "Ba_4554": LineDefinition("Ba_4554", 4554.0, "Ba II", "metal", "medium"),
        "Ba_6497": LineDefinition("Ba_6497", 6496.9, "Ba II", "metal", "medium"),
        "Sr_4077": LineDefinition("Sr_4077", 4077.7, "Sr II", "metal", "medium"),
        "Al_3944": LineDefinition("Al_3944", 3944.0, "Al I", "metal", "weak"),
        "Co_5301": LineDefinition("Co_5301", 5301.0, "Co I", "metal", "weak"),
        "V_4379": LineDefinition("V_4379", 4379.2, "V II", "metal", "weak"),
    }


# ============================================================================
# MAIN CLASS
# ============================================================================


class FeatureEngineer:
    """Unified spectroscopic feature extractor.

    Combines the 29 V1 base features (lines, ratios, indices) with V2
    extensions (molecular bands, extended Balmer series, full metallic
    lines, spectral indices, line profiles, continuum, colours).

    Notes
    -----
    - ``feature_names`` is populated automatically on the first call to
      ``extract_features`` via ``_populate_feature_names`` in ``__init__``.
    - Feature order is alphabetical (stable across runs).
    - Catalogue-derived colour features (bp_g, g_rp, etc.) are NaN during
      per-spectrum extraction; they are filled post-merge by
      ``add_photometric_composites``.
    """

    def __init__(self) -> None:
        # --- V1 Base definitions ---
        self.base_lines: List[str] = ["Hα", "Hβ", "CaII K", "CaII H", "Mg_b", "Na_D"]
        self.line_metrics: List[str] = ["prominence", "fwhm", "eq_width"]

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

        self.index_definitions: Dict[str, Tuple[List[float], List[List[float]]]] = {
            "TiO5": ([7126.0, 7135.0], [[7042.0, 7052.0]]),
            "Dn4000": ([4000.0, 4100.0], [[3850.0, 3950.0]]),
            "G4300": ([4280.0, 4320.0], [[4260.0, 4280.0], [4320.0, 4340.0]]),
            "Ca4227": ([4225.0, 4235.0], [[4215.0, 4225.0], [4235.0, 4245.0]]),
            "Hbeta_index": ([4840.0, 4870.0], [[4820.0, 4840.0], [4870.0, 4890.0]]),
            "Mgb_index": ([5160.0, 5190.0], [[5150.0, 5160.0], [5190.0, 5200.0]]),
            "CaH2": ([6814.0, 6846.0], [[7042.0, 7056.0]]),
            "CaH3": ([6960.0, 6990.0], [[7042.0, 7056.0]]),
        }

        # --- V2 Extended definitions ---
        self.line_library = SpectralLineLibrary()
        self._init_molecular_bands()
        self._init_balmer_extended()
        self._init_metal_lines_extended()
        self._init_spectral_indices_extended()

        # --- Populate self.feature_names with a dry run ---
        self.feature_names: List[str] = []
        self._populate_feature_names()

    # -------------------------------------------------------------------------
    # V2 dictionary initialisers
    # -------------------------------------------------------------------------

    def _init_molecular_bands(self) -> None:
        """Populate ``molecular_bands``: band_name → (centre_Å, half_width_Å)."""
        self.molecular_bands: Dict[str, Tuple[float, float]] = {
            "TiO_6180": (6205, 50),
            "TiO_7050": (7100, 100),
            "TiO_7600": (7650, 100),
            "TiO_8200": (8250, 100),
            "TiO_8400": (8450, 100),
            "TiO_8860": (8885, 50),
            "VO_7400": (7475, 150),
            "VO_7900": (7975, 150),
            "CN_4142": (4160, 35),
            "CH_4300": (4305, 25),
            "CaH_6380": (6385, 10),
            "CaH_6830": (6835, 10),
            "CaH_6975": (6980, 10),
            "MgH_5140": (5145, 10),
        }

    def _init_balmer_extended(self) -> None:
        """Balmer Hγ→H10 wavelengths (Hα and Hβ handled by V1 base)."""
        self.balmer_extended: Dict[str, float] = {
            "Halpha": 6562.8,  # included for ratio computation; extracted in V1
            "Hbeta": 4861.3,  # same
            "Hgamma": 4340.5,
            "Hdelta": 4101.7,
            "Hepsilon": 3970.1,
            "H8": 3889.0,
            "H9": 3835.4,
            "H10": 3797.9,
        }

    def _init_metal_lines_extended(self) -> None:
        self.metal_lines_extended: Dict[str, float] = {
            ld.name: ld.wavelength for ld in self.line_library.METAL_LINES.values()
        }

    def _init_spectral_indices_extended(self) -> None:
        """Additional Lick / SDSS spectral indices."""
        self.spectral_indices_extended: Dict[
            str, Tuple[List[float], List[List[float]]]
        ] = {
            "Mg_1": ([5069, 5135], [[5020, 5050], [5160, 5190]]),
            "Mg_2": ([5154, 5197], [[5120, 5150], [5200, 5230]]),
            "NaD_Lick": ([5876, 5909], [[5860, 5875], [5910, 5925]]),
            "TiO_1_Lick": ([5936, 5994], [[5900, 5930], [6000, 6030]]),
            "TiO_2_Lick": ([6190, 6272], [[6150, 6180], [6280, 6310]]),
            "Dn4000_SDSS": ([4050, 4250], [[3750, 3950]]),
            "CaII_triplet": ([8480, 8680], [[8400, 8470], [8690, 8750]]),
            "Paschen_12": ([8730, 8772], [[8700, 8725], [8776, 8800]]),
            "CN_1": ([4142, 4177], [[4080, 4117], [4244, 4284]]),
            "CN_2": ([4216, 4251], [[4080, 4117], [4284, 4320]]),
        }

    def _populate_feature_names(self) -> None:
        """Dry run to lock the canonical alphabetical order into ``feature_names``."""
        dummy_wl = np.linspace(3600, 9000, 500)
        dummy_flux = np.ones(500)
        dummy_invv = np.ones(500)
        dummy_matches: Dict[str, Optional[Tuple[float, float]]] = {
            line: None for line in self.base_lines
        }
        self.extract_features(dummy_matches, dummy_wl, dummy_flux, dummy_invv)

    # =========================================================================
    # V1 base extraction methods
    # =========================================================================

    def _calculate_spectroscopic_indices(
        self, wavelength: np.ndarray, flux_norm: np.ndarray
    ) -> Dict[str, float]:
        """V1 spectral indices (TiO5, Dn4000, G4300, CaH2/3, etc.).

        Returns flux_feature / flux_continuum ratios for each index
        defined in ``index_definitions``.
        """
        indices: Dict[str, float] = {}
        for name, (feature_band, continuum_bands) in self.index_definitions.items():
            try:
                f_mask = (wavelength >= feature_band[0]) & (
                    wavelength <= feature_band[1]
                )
                f_vals = flux_norm[f_mask]
                mean_f = float(np.nanmean(f_vals)) if f_vals.size > 0 else 0.0

                c_vals_list: List[np.ndarray] = []
                if isinstance(continuum_bands, list):
                    for band in continuum_bands:
                        if band is None:
                            continue
                        if isinstance(band, (list, tuple)) and len(band) == 2:
                            c_mask = (wavelength >= band[0]) & (wavelength <= band[1])
                            c_vals_list.append(flux_norm[c_mask])
                        else:
                            c_mask = (wavelength >= continuum_bands[0]) & (
                                wavelength <= continuum_bands[1]
                            )
                            c_vals_list.append(flux_norm[c_mask])
                            break
                else:
                    c_mask = (wavelength >= continuum_bands[0]) & (
                        wavelength <= continuum_bands[1]
                    )
                    c_vals_list.append(flux_norm[c_mask])

                if c_vals_list:
                    c_concat = np.concatenate(c_vals_list)
                    mean_c = float(np.nanmean(c_concat)) if c_concat.size > 0 else 0.0
                else:
                    mean_c = 0.0

                indices[f"feature_index_{name}"] = mean_f / (mean_c + 1e-6)
            except Exception:
                indices[f"feature_index_{name}"] = np.nan
        return indices

    # =========================================================================
    # V2 extension extraction methods
    # =========================================================================

    def _extract_molecular_bands(
        self, wavelength: np.ndarray, flux_norm: np.ndarray
    ) -> Dict[str, float]:
        """Molecular-band absorption indices (TiO, VO, CN, CH, CaH, MgH).

        For each band: ``index = max(0, 1 - flux_band / flux_continuum)``.
        An index > 0 indicates absorption; the ``max(0, ...)`` clamp
        prevents negative emission artefacts.

        Keys generated: ``feature_molecular_{band_name}`` (float, one per band).
        """
        features: Dict[str, float] = {}
        for band_name, (band_center, half_width) in self.molecular_bands.items():
            key = f"feature_molecular_{band_name}"
            try:
                lam_min = band_center - half_width / 2.0
                lam_max = band_center + half_width / 2.0

                mask_band = (wavelength >= lam_min) & (wavelength <= lam_max)
                mask_cont_blue = (wavelength >= lam_min - 100) & (
                    wavelength <= lam_min - 20
                )
                mask_cont_red = (wavelength >= lam_max + 20) & (
                    wavelength <= lam_max + 100
                )

                flux_band = np.nanmedian(flux_norm[mask_band])
                flux_cont_blue = np.nanmedian(flux_norm[mask_cont_blue])
                flux_cont_red = np.nanmedian(flux_norm[mask_cont_red])
                flux_cont = (flux_cont_blue + flux_cont_red) / 2.0

                if np.isfinite(flux_cont) and flux_cont > 0 and np.isfinite(flux_band):
                    features[key] = float(max(0.0, 1.0 - flux_band / flux_cont))
                else:
                    features[key] = np.nan
            except Exception:
                features[key] = np.nan
        return features

    def _extract_balmer_series_full(
        self,
        wavelength: np.ndarray,
        flux_norm: np.ndarray,
        invvar: np.ndarray,
    ) -> Dict[str, float]:
        """EW and FWHM for Hγ, Hδ, Hε, H8, H9, H10 (Hα/Hβ handled by V1 base).

        Also computes:
        - ``feature_balmer_gradient``         : EW vs λ slope (Teff indicator).
        - ``feature_balmer_ratio_gamma_beta`` : EW(Hγ) / EW(Hβ) — T-sensitivity proxy.
        - ``feature_balmer_ratio_delta_gamma``: EW(Hδ) / EW(Hγ) — second Balmer ratio.
        - ``feature_balmer_temperature_index``: normalised combination of both ratios.

        Note: Hα and Hβ are included in ``balmer_extended`` for ratio
        computation, but their EW values come from the V1 base block
        (via ``all_line_features``).
        """
        features: Dict[str, float] = {}
        ew_values: List[float] = []
        ew_by_name: Dict[str, float] = {}

        # Lines to extract in V2 (Hα/Hβ already in V1)
        v2_balmer = {
            k: v
            for k, v in self.balmer_extended.items()
            if k not in ("Halpha", "Hbeta")
        }

        for line_name, line_wl in v2_balmer.items():
            try:
                window = 20.0
                mask = (wavelength >= line_wl - window) & (
                    wavelength <= line_wl + window
                )
                if np.sum(mask) < 5:
                    ew_val, fwhm_val = np.nan, np.nan
                else:
                    spec = Spectrum(
                        spectral_axis=wavelength[mask] * u.AA,
                        flux=flux_norm[mask] * u.dimensionless_unscaled,
                        uncertainty=StdDevUncertainty(
                            safe_sigma_from_invvar(invvar[mask])
                        ),
                    )
                    try:
                        region = SpectralRegion(
                            (line_wl - 10) * u.AA, (line_wl + 10) * u.AA
                        )
                        ew_raw = equivalent_width(spec, regions=region)
                        ew_val = float(abs(ew_raw.value))
                    except Exception:
                        ew_val = np.nan
                    try:
                        fwhm_raw = gaussian_fwhm(spec)
                        fwhm_val = (
                            float(fwhm_raw.value) if fwhm_raw.value > 0 else np.nan
                        )
                    except Exception:
                        fwhm_val = np.nan

                features[f"feature_{line_name}_eq_width"] = ew_val
                features[f"feature_{line_name}_fwhm"] = fwhm_val
                ew_values.append(ew_val)
                ew_by_name[line_name] = ew_val if np.isfinite(ew_val) else np.nan
            except Exception:
                features[f"feature_{line_name}_eq_width"] = np.nan
                features[f"feature_{line_name}_fwhm"] = np.nan
                ew_values.append(np.nan)
                ew_by_name[line_name] = np.nan

        # --- Balmer gradient (EW vs λ slope, Teff indicator) ---
        try:
            valid_pairs = [
                (self.balmer_extended[n], e)
                for n, e in ew_by_name.items()
                if np.isfinite(e) and e > 0
            ]
            if len(valid_pairs) >= 2:
                wls, ews = zip(*valid_pairs)
                features["feature_balmer_gradient"] = float(np.polyfit(wls, ews, 1)[0])
            else:
                features["feature_balmer_gradient"] = np.nan
        except Exception:
            features["feature_balmer_gradient"] = np.nan

        # --- Balmer ratios (computed from ew_by_name + V1 EW for Hβ) ---
        # Note: EW(Hβ) comes from the V1 block and is not available here.
        # These are placeholders; they are overridden in extract_features
        # after all_line_features has been assembled.
        features["feature_balmer_ratio_gamma_beta"] = np.nan
        features["feature_balmer_ratio_delta_gamma"] = np.nan
        features["feature_balmer_temperature_index"] = np.nan

        # Expose Hγ/Hδ EWs for downstream ratio computation
        features["_ew_Hgamma"] = ew_by_name.get("Hgamma", np.nan)
        features["_ew_Hdelta"] = ew_by_name.get("Hdelta", np.nan)
        return features

    def _extract_metal_lines(
        self,
        wavelength: np.ndarray,
        flux_norm: np.ndarray,
        invvar: np.ndarray,
    ) -> Dict[str, float]:
        """Equivalent widths for all metallic lines + chemical composite indices.

        EW keys: ``feature_{line_name}_eq_width`` (29 lines).

        Composite keys:
        - ``feature_ratio_Fe_Mg``, ``feature_ratio_Ca_Fe``,
          ``feature_ratio_Ba_Fe``, ``feature_ratio_Sr_Fe``:
          inter-element group ratios.
        - ``feature_metal_index_combined`` : weighted metallicity index.
        - ``feature_alpha_elements_index`` : sum of α-elements (Mg, Ca, Si, Ti).
        - ``feature_iron_peak_index``      : sum of iron-peak elements.
        - ``feature_s_process_index``      : s-process ratio (Ba, Sr) / Fe.
        - ``feature_Ca_triplet_strength``  : CaII IR triplet strength.
        - ``feature_Mg_triplet_strength``  : Mg b triplet strength.
        """
        features: Dict[str, float] = {}
        ew_dict: Dict[str, float] = {}

        for line_name, line_wl in self.metal_lines_extended.items():
            key = f"feature_{line_name}_eq_width"
            try:
                mask = (wavelength >= line_wl - 15) & (wavelength <= line_wl + 15)
                if np.sum(mask) < 5:
                    ew_val = np.nan
                else:
                    spec = Spectrum(
                        spectral_axis=wavelength[mask] * u.AA,
                        flux=flux_norm[mask] * u.dimensionless_unscaled,
                        uncertainty=StdDevUncertainty(
                            safe_sigma_from_invvar(invvar[mask])
                        ),
                    )
                    try:
                        ew_raw = equivalent_width(
                            spec,
                            regions=SpectralRegion(
                                (line_wl - 8) * u.AA, (line_wl + 8) * u.AA
                            ),
                        )
                        ew_val = float(abs(ew_raw.value))
                    except Exception:
                        ew_val = np.nan

                features[key] = ew_val
                ew_dict[line_name] = ew_val
            except Exception:
                features[key] = np.nan
                ew_dict[line_name] = np.nan

        # --- Chemical composite indices ---
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                fe_avg = np.nanmean(
                    [ew_dict.get("Fe_5270", np.nan), ew_dict.get("Fe_5335", np.nan)]
                )
            mg_avg = np.nanmean(
                [
                    ew_dict.get("Mg_5167", np.nan),
                    ew_dict.get("Mg_5173", np.nan),
                    ew_dict.get("Mg_5184", np.nan),
                ]
            )
            ca_avg = np.nanmean(
                [
                    ew_dict.get("Ca_8498", np.nan),
                    ew_dict.get("Ca_8542", np.nan),
                    ew_dict.get("Ca_8662", np.nan),
                ]
            )
            ba_avg = np.nanmean(
                [ew_dict.get("Ba_4554", np.nan), ew_dict.get("Ba_6497", np.nan)]
            )
            sr_ew = ew_dict.get("Sr_4077", np.nan)
            si_avg = np.nanmean(
                [ew_dict.get("Si_4128", np.nan), ew_dict.get("Si_4131", np.nan)]
            )
            ti_avg = np.nanmean(
                [ew_dict.get("Ti_4758", np.nan), ew_dict.get("Ti_4764", np.nan)]
            )
            cr_avg = np.nanmean(
                [ew_dict.get("Cr_5206", np.nan), ew_dict.get("Cr_5208", np.nan)]
            )
            ni_ew = ew_dict.get("Ni_5081", np.nan)
            co_ew = ew_dict.get("Co_5301", np.nan)

            features["feature_ratio_Fe_Mg"] = fe_avg / (mg_avg + 1e-6)
            features["feature_ratio_Ca_Fe"] = ca_avg / (fe_avg + 1e-6)
            features["feature_ratio_Ba_Fe"] = ba_avg / (fe_avg + 1e-6)
            features["feature_ratio_Sr_Fe"] = sr_ew / (fe_avg + 1e-6)
            features["feature_metal_index_combined"] = float(
                0.4 * np.nan_to_num(fe_avg)
                + 0.3 * np.nan_to_num(mg_avg)
                + 0.2 * np.nan_to_num(ca_avg)
                + 0.1 * np.nan_to_num(si_avg)
            )
            features["feature_alpha_elements_index"] = float(
                np.nan_to_num(mg_avg)
                + np.nan_to_num(ca_avg)
                + np.nan_to_num(si_avg)
                + np.nan_to_num(ti_avg)
            )
            features["feature_iron_peak_index"] = float(
                np.nan_to_num(fe_avg)
                + np.nan_to_num(cr_avg)
                + np.nan_to_num(ni_ew)
                + np.nan_to_num(co_ew)
            )
            features["feature_s_process_index"] = (
                np.nan_to_num(ba_avg) + np.nan_to_num(sr_ew)
            ) / (fe_avg + 1e-6)
            features["feature_Ca_triplet_strength"] = float(np.nan_to_num(ca_avg))
            features["feature_Mg_triplet_strength"] = float(np.nan_to_num(mg_avg))
        except Exception:
            for k in (
                "feature_ratio_Fe_Mg",
                "feature_ratio_Ca_Fe",
                "feature_ratio_Ba_Fe",
                "feature_ratio_Sr_Fe",
                "feature_metal_index_combined",
                "feature_alpha_elements_index",
                "feature_iron_peak_index",
                "feature_s_process_index",
                "feature_Ca_triplet_strength",
                "feature_Mg_triplet_strength",
            ):
                features.setdefault(k, np.nan)
        return features

    def _extract_spectral_indices_extended(
        self, wavelength: np.ndarray, flux_norm: np.ndarray
    ) -> Dict[str, float]:
        """Additional Lick / SDSS spectral indices (Mg_1, Mg_2, NaD, TiO, CaII triplet, etc.).

        Same logic as ``_calculate_spectroscopic_indices``: band / continuum ratio.
        Keys: ``feature_index_{name}``.
        """
        features: Dict[str, float] = {}
        for name, (
            feature_band,
            continuum_bands,
        ) in self.spectral_indices_extended.items():
            try:
                f_mask = (wavelength >= feature_band[0]) & (
                    wavelength <= feature_band[1]
                )
                f_vals = flux_norm[f_mask]
                mean_f = float(np.nanmean(f_vals)) if f_vals.size > 0 else np.nan

                c_arrs = [
                    flux_norm[(wavelength >= b[0]) & (wavelength <= b[1])]
                    for b in continuum_bands
                ]
                c_cat = np.concatenate(c_arrs) if c_arrs else np.array([])
                mean_c = float(np.nanmean(c_cat)) if c_cat.size > 0 else np.nan

                if np.isfinite(mean_f) and np.isfinite(mean_c):
                    features[f"feature_index_{name}"] = mean_f / (mean_c + 1e-6)
                else:
                    features[f"feature_index_{name}"] = np.nan
            except Exception:
                features[f"feature_index_{name}"] = np.nan
        return features

    def _extract_continuum_analysis(
        self, wavelength: np.ndarray, flux_norm: np.ndarray
    ) -> Dict[str, float]:
        """Morphological continuum analysis.

        Extracts local slopes, curvatures, Paschen / 4000 Å jumps, RMS,
        blue/red asymmetry, and UV excess.

        Notes
        -----
        - ``feature_balmer_jump_strength`` and ``feature_brackett_jump_strength``
          are NaN: these jumps (3646 Å and 14 500 Å) fall outside the LAMOST
          wavelength coverage.
        - Slopes are computed with sigma-clipping (σ=2.5) to exclude
          absorption lines that would bias the continuum slope.
        """
        features: Dict[str, float] = {}

        # Spectral jumps
        features["feature_balmer_jump_strength"] = np.nan  # outside LAMOST range
        features["feature_brackett_jump_strength"] = np.nan  # outside LAMOST range

        try:
            f_blue = np.nanmedian(
                flux_norm[(wavelength >= 8120) & (wavelength <= 8180)]
            )
            f_red = np.nanmedian(flux_norm[(wavelength >= 8220) & (wavelength <= 8280)])
            features["feature_paschen_jump_strength"] = float(f_red / (f_blue + 1e-6))
        except Exception:
            features["feature_paschen_jump_strength"] = np.nan

        try:
            mask = (wavelength > 3850) & (wavelength < 4150)
            wave_b, flux_b = wavelength[mask], flux_norm[mask]
            if len(wave_b) > 10:
                dflux = np.gradient(flux_b, wave_b)
                bk_str = float(np.max(dflux))
                idx = np.where(dflux > bk_str / 2.0)[0]
                bk_width = (
                    float(wave_b[idx[-1]] - wave_b[idx[0]]) if len(idx) > 1 else np.nan
                )
            else:
                bk_str, bk_width = np.nan, np.nan
            features["feature_break_4000A_strength"] = bk_str
            features["feature_break_4000A_width"] = bk_width
        except Exception:
            features["feature_break_4000A_strength"] = np.nan
            features["feature_break_4000A_width"] = np.nan

        # Local slopes (sigma-clipped)
        for region_name, lmin, lmax in [
            ("blue", 3800, 4200),
            ("green", 4500, 5500),
            ("red", 5500, 6500),
            ("deep_red", 6500, 7500),
        ]:
            key = f"feature_slope_{region_name}"
            try:
                mask = (wavelength > lmin) & (wavelength < lmax)
                flux_clipped = sigma_clip(flux_norm[mask], sigma=2.5)
                valid = ~flux_clipped.mask
                if np.sum(valid) > 5:
                    features[key] = float(
                        np.polyfit(wavelength[mask][valid], flux_norm[mask][valid], 1)[
                            0
                        ]
                    )
                else:
                    features[key] = np.nan
            except Exception:
                features[key] = np.nan

        # Local curvature (second derivative of Savitzky-Golay smoothed flux)
        for center in [4000, 5000, 6000]:
            key = f"feature_curvature_{center}"
            try:
                mask = (wavelength > center - 100) & (wavelength < center + 100)
                n = np.sum(mask)
                if n > 10:
                    win_len = min(51, (n // 2) * 2 + 1)
                    smooth = savgol_filter(
                        flux_norm[mask], window_length=win_len, polyorder=3
                    )
                    d2 = np.gradient(
                        np.gradient(smooth, wavelength[mask]), wavelength[mask]
                    )
                    features[key] = float(d2[len(smooth) // 2])
                else:
                    features[key] = np.nan
            except Exception:
                features[key] = np.nan

        # Continuum statistics
        try:
            cont_mask = ((wavelength > 4300) & (wavelength < 4400)) | (
                (wavelength > 5500) & (wavelength < 5600)
            )
            features["feature_continuum_rms"] = float(np.std(flux_norm[cont_mask]))
        except Exception:
            features["feature_continuum_rms"] = np.nan

        try:
            features["feature_continuum_peak_wavelength"] = float(
                wavelength[np.argmax(flux_norm)]
            )
            b = np.nanmean(flux_norm[(wavelength > 4000) & (wavelength < 4500)])
            r = np.nanmean(flux_norm[(wavelength > 6500) & (wavelength < 7000)])
            features["feature_continuum_asymmetry"] = float((b - r) / (b + r + 1e-6))
            features["feature_flux_ratio_blue_red"] = float(b / (r + 1e-6))
            uv = np.nanmean(flux_norm[(wavelength > 3850) & (wavelength < 3950)])
            ref = np.nanmean(flux_norm[(wavelength > 4400) & (wavelength < 4600)])
            features["feature_UV_excess_3900"] = float(uv / (ref + 1e-6))
        except Exception:
            for k in (
                "feature_continuum_peak_wavelength",
                "feature_continuum_asymmetry",
                "feature_flux_ratio_blue_red",
                "feature_UV_excess_3900",
            ):
                features.setdefault(k, np.nan)

        try:
            mask = (wavelength > 3800) & (wavelength < 7500)
            clipped = sigma_clip(flux_norm[mask], sigma=2.5)
            valid = ~clipped.mask
            if np.sum(valid) > 10:
                features["feature_continuum_slope_global"] = float(
                    np.polyfit(wavelength[mask][valid], flux_norm[mask][valid], 1)[0]
                )
            else:
                features["feature_continuum_slope_global"] = np.nan
        except Exception:
            features["feature_continuum_slope_global"] = np.nan

        return features

    def _extract_line_profiles(
        self,
        matched_lines: Mapping[str, Optional[Tuple[float, float]]],
        wavelength: np.ndarray,
        flux_norm: np.ndarray,
    ) -> Dict[str, float]:
        """Detailed morphological profiles for Hα and CaII K.

        Per-line metrics (10 per line × 2 lines = 20):
        - ``depth``         : normalised depth at the profile minimum.
        - ``core_width``    : half-depth width (morphological FWHM, Å).
        - ``base_width``    : width at 5 % of the continuum (Å) — rotation proxy.
        - ``asymmetry``     : (blue wing − red wing) / (total + ε).
        - ``skewness``      : statistical skewness of the normalised profile.
        - ``kurtosis``      : kurtosis of the normalised profile.
        - ``wing_blue``     : trapezoidal integral of the blue wing.
        - ``wing_red``      : trapezoidal integral of the red wing.
        - ``wing_ratio``    : wing_blue / wing_red (differential rotation proxy).
        - ``emission_index``: norm[core] − 1 for Hα (> 0 → emission).

        Aggregates (5):
        - ``feature_avg_line_kurtosis``, ``feature_rotation_proxy``,
          ``feature_avg_line_depth``, ``feature_profile_shape_index``,
          ``feature_chromospheric_activity_flag``.
        """
        features: Dict[str, float] = {}
        metrics = [
            "depth",
            "core_width",
            "base_width",
            "asymmetry",
            "skewness",
            "kurtosis",
            "wing_blue",
            "wing_red",
            "wing_ratio",
            "emission_index",
        ]

        for line_name in ["Hα", "CaII K"]:
            safe = line_name.replace(" ", "").replace("α", "alpha")

            # Line not detected → all metrics set to NaN
            if matched_lines.get(line_name) is None:
                for m in metrics:
                    features[f"feature_{safe}_{m}"] = np.nan
                continue

            detected_wl, _ = matched_lines[line_name]
            try:
                mask = (wavelength >= detected_wl - 15) & (
                    wavelength <= detected_wl + 15
                )
                wave_p = wavelength[mask]
                flux_p = flux_norm[mask]

                if len(wave_p) < 10:
                    for m in metrics:
                        features[f"feature_{safe}_{m}"] = np.nan
                    continue

                # Local normalisation by linearly interpolated continuum
                cont = (flux_p[0] + flux_p[-1]) / 2.0
                norm = flux_p / (cont + 1e-6)
                c_idx = int(np.argmin(norm))

                # 1. Depth
                depth = float(1.0 - norm[c_idx])
                features[f"feature_{safe}_depth"] = depth

                # 2. Half-depth width (core_width)
                half_level = (1.0 + norm[c_idx]) / 2.0
                below_half = np.where(norm < half_level)[0]
                if len(below_half) > 1:
                    core_width = float(wave_p[below_half[-1]] - wave_p[below_half[0]])
                else:
                    core_width = np.nan
                features[f"feature_{safe}_core_width"] = core_width

                # 3. Base width — threshold at 5 % below the continuum (norm < 0.95)
                below_cont = np.where(norm < 0.95)[0]
                if len(below_cont) > 1:
                    base_width = float(wave_p[below_cont[-1]] - wave_p[below_cont[0]])
                else:
                    base_width = np.nan
                features[f"feature_{safe}_base_width"] = base_width

                # 4. Blue / red wing asymmetry
                b_wing = float(_np_trapz(1.0 - norm[:c_idx], wave_p[:c_idx]))
                r_wing = float(_np_trapz(1.0 - norm[c_idx:], wave_p[c_idx:]))
                features[f"feature_{safe}_asymmetry"] = (b_wing - r_wing) / (
                    b_wing + r_wing + 1e-6
                )
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    features[f"feature_{safe}_skewness"] = float(skew(norm))
                    features[f"feature_{safe}_kurtosis"] = float(kurtosis(norm))
                features[f"feature_{safe}_wing_blue"] = b_wing
                features[f"feature_{safe}_wing_red"] = r_wing
                features[f"feature_{safe}_wing_ratio"] = b_wing / (r_wing + 1e-6)

                # 5. Emission index (Hα only)
                if line_name == "Hα":
                    features[f"feature_{safe}_emission_index"] = float(
                        norm[c_idx] - 1.0
                    )
                else:
                    features[f"feature_{safe}_emission_index"] = np.nan

            except Exception:
                for m in metrics:
                    features[f"feature_{safe}_{m}"] = np.nan

        # --- Multi-line aggregates ---
        try:
            k_vals = [
                v
                for v in [
                    features.get("feature_Halpha_kurtosis"),
                    features.get("feature_CaIIK_kurtosis"),
                ]
                if pd.notna(v)
            ]
            features["feature_avg_line_kurtosis"] = (
                float(np.mean(k_vals)) if k_vals else np.nan
            )

            w_vals = [
                v
                for v in [
                    features.get("feature_Halpha_wing_ratio"),
                    features.get("feature_CaIIK_wing_ratio"),
                ]
                if pd.notna(v)
            ]
            features["feature_rotation_proxy"] = (
                float(np.std(w_vals)) if len(w_vals) > 1 else np.nan
            )

            d_vals = [
                v
                for v in [
                    features.get("feature_Halpha_depth"),
                    features.get("feature_CaIIK_depth"),
                ]
                if pd.notna(v)
            ]
            features["feature_avg_line_depth"] = (
                float(np.mean(d_vals)) if d_vals else np.nan
            )

            s_vals = [
                v
                for v in [
                    features.get("feature_Halpha_skewness"),
                    features.get("feature_CaIIK_skewness"),
                ]
                if pd.notna(v)
            ]
            features["feature_profile_shape_index"] = (
                float(np.mean(k_vals) - np.mean(s_vals))
                if k_vals and s_vals
                else np.nan
            )

            emi = features.get("feature_Halpha_emission_index", np.nan)
            features["feature_chromospheric_activity_flag"] = (
                1 if pd.notna(emi) and emi > 0.1 else 0
            )
        except Exception:
            for k in (
                "feature_avg_line_kurtosis",
                "feature_rotation_proxy",
                "feature_avg_line_depth",
                "feature_profile_shape_index",
                "feature_chromospheric_activity_flag",
            ):
                features.setdefault(k, np.nan)

        return features

    def _extract_color_features(
        self,
        wavelength: np.ndarray,
        flux_norm: np.ndarray,
        catalog_row: Optional[pd.Series] = None,
    ) -> Dict[str, float]:
        """Photometric and synthetic colour features.

        During per-spectrum extraction (catalog_row=None) only spectrum-derived
        features are computed (``synthetic_BV``).  Catalogue features
        (``bp_g``, ``g_rp``, ``color_ug``, ``color_iz``) are set to NaN and
        are filled later by ``add_photometric_composites`` after the merge.

        Note: the correct Gaia columns are ``phot_bp_mean_mag``,
        ``phot_g_mean_mag``, ``phot_rp_mean_mag`` (not bp_mag/g_mag/rp_mag).
        ``Q_parameter`` and ``X_parameter`` require ``feature_color_gr``
        (available only post-merge); they remain NaN here.
        """
        features: Dict[str, float] = {}

        # --- Photometric colours from catalogue ---
        if catalog_row is not None:
            # SDSS ugriz (LAMOST)
            for c1, c2, name in [
                ("magnitude_u", "magnitude_g", "color_ug"),
                ("magnitude_i", "magnitude_z", "color_iz"),
            ]:
                v1 = catalog_row.get(c1, np.nan)
                v2 = catalog_row.get(c2, np.nan)
                if pd.notna(v1) and pd.notna(v2) and v1 != 99 and v2 != 99:
                    features[f"feature_{name}"] = float(v1 - v2)
                else:
                    features[f"feature_{name}"] = np.nan

            # Gaia colours — use exact master_catalog_gaia column names
            bp = catalog_row.get("phot_bp_mean_mag", np.nan)
            g = catalog_row.get("phot_g_mean_mag", np.nan)
            rp = catalog_row.get("phot_rp_mean_mag", np.nan)
            # Also use pre-computed columns if they exist
            bp_g_pre = catalog_row.get("bp_g", np.nan)
            g_rp_pre = catalog_row.get("g_rp", np.nan)

            if pd.notna(bp_g_pre):
                features["feature_bp_g"] = float(bp_g_pre)
            elif pd.notna(bp) and pd.notna(g):
                features["feature_bp_g"] = float(bp - g)
            else:
                features["feature_bp_g"] = np.nan

            if pd.notna(g_rp_pre):
                features["feature_g_rp"] = float(g_rp_pre)
            elif pd.notna(g) and pd.notna(rp):
                features["feature_g_rp"] = float(g - rp)
            else:
                features["feature_g_rp"] = np.nan
        else:
            # Normal pipeline: catalog_row not available at this stage
            for name in ("color_ug", "color_iz", "bp_g", "g_rp"):
                features[f"feature_{name}"] = np.nan

        # --- Synthetic B-V colour from the spectrum ---
        try:
            b_flux = np.nanmean(flux_norm[(wavelength > 3920) & (wavelength < 4920)])
            v_flux = np.nanmean(flux_norm[(wavelength > 5070) & (wavelength < 5950)])
            features["feature_synthetic_BV"] = float(
                -2.5 * np.log10(b_flux / (v_flux + 1e-6))
            )
        except Exception:
            features["feature_synthetic_BV"] = np.nan

        # --- Q and X require feature_color_gr (post-merge) → NaN here ---
        features["feature_Q_parameter"] = np.nan
        features["feature_X_parameter"] = np.nan

        return features

    def _extract_composite_indices(
        self, features_dict: Dict[str, float]
    ) -> Dict[str, float]:
        """Physical composite indices (Teff, logg, [Fe/H], [α/Fe] proxies).

        Note on Teff_proxy: the original formula depended on
        ``feature_color_gr`` (unavailable before the catalogue merge).
        It is replaced by ``feature_synthetic_BV`` which is derived
        from the spectrum itself.

        The 11 stellar-population indices (thin_disk, halo, etc.) are NaN
        placeholders — they will be implemented in V3 using Gaia kinematic
        models.
        """
        comp: Dict[str, float] = {}

        def gf(name: str, default: float = np.nan) -> float:
            val = features_dict.get(name, default)
            return float(val) if pd.notna(val) else default

        # --- Teff proxy (spectrum-only) ---
        # Uses synthetic_BV instead of color_gr (post-merge only)
        try:
            synth_bv = gf("feature_synthetic_BV", 0.6)
            slope_blue = gf("feature_slope_blue", 0.0)
            curv_4000 = gf("feature_curvature_4000", 0.0)
            comp["feature_Teff_proxy"] = float(
                6500
                - 3000
                * (
                    0.50 * ((synth_bv - 0.6) / 0.8)
                    + 0.35 * (slope_blue * 1000)
                    + 0.15 * (curv_4000 * 1e6)
                )
            )
        except Exception:
            comp["feature_Teff_proxy"] = np.nan

        # --- logg proxy (line wings → surface gravity) ---
        try:
            ha_wr = gf("feature_Halpha_wing_ratio", 1.0) - 1.0
            ca_wr = gf("feature_CaIIK_wing_ratio", 1.0) - 1.0
            comp["feature_logg_proxy"] = float(
                np.clip(4.5 - 2.0 * np.nanmean([ha_wr, ca_wr]), 0.0, 5.0)
            )
        except Exception:
            comp["feature_logg_proxy"] = np.nan

        # --- [Fe/H] proxy (metallic line strength) ---
        try:
            comp["feature_FeH_proxy"] = float(
                np.clip(
                    -1.0 + 1.5 * (gf("feature_metal_index_combined", 0.0) / 10.0),
                    -2.5,
                    0.5,
                )
            )
        except Exception:
            comp["feature_FeH_proxy"] = np.nan

        # --- [α/Fe] proxy ---
        try:
            alpha = gf("feature_alpha_elements_index", 0.0) + 1e-6
            iron = gf("feature_iron_peak_index", 0.0) + 1e-6
            comp["feature_alpha_Fe_proxy"] = float(
                np.clip(np.log10(alpha / iron), -0.5, 0.5)
            )
        except Exception:
            comp["feature_alpha_Fe_proxy"] = np.nan

        # --- Stellar-population composite indices ---

        # 1. activity_index — chromospheric activity proxy (Hα emission + CaII K prominence)
        #    > 0.5 : active star (flare star, T Tauri, RS CVn)
        try:
            emi = gf("feature_Halpha_emission_index", 0.0)
            caK_prom = gf("feature_CaIIK_prominence", 0.0)
            caK_norm = caK_prom / (caK_prom + 1.0)  # soft normalisation → [0, 1]
            comp["feature_activity_index"] = float(
                np.clip(0.6 * max(0.0, emi) + 0.4 * caK_norm, 0.0, 1.0)
            )
        except Exception:
            comp["feature_activity_index"] = np.nan

        # 2. rotation_index — v sin i proxy (line base-width broadening)
        #    1.0 corresponds to ~30 Å base-width (moderately fast rotator)
        try:
            ha_bw = gf("feature_Halpha_base_width", np.nan)
            ca_bw = gf("feature_CaIIK_base_width", np.nan)
            valid = [v for v in [ha_bw, ca_bw] if np.isfinite(v) and v > 0]
            comp["feature_rotation_index"] = (
                float(np.clip(np.mean(valid) / 30.0, 0.0, 2.0)) if valid else np.nan
            )
        except Exception:
            comp["feature_rotation_index"] = np.nan

        # 3. main_sequence_index — P(dwarf) via logg proxy
        #    Sigmoid centred at logg=4.0: dwarfs (logg~4-5) → ~1, giants → ~0
        try:
            logg = gf("feature_logg_proxy", np.nan)
            comp["feature_main_sequence_index"] = (
                float(1.0 / (1.0 + np.exp(-(logg - 4.0) / 0.5)))
                if np.isfinite(logg)
                else np.nan
            )
        except Exception:
            comp["feature_main_sequence_index"] = np.nan

        # 4. giant_index — P(giant) via logg proxy
        #    Sigmoid centred at logg=2.5: giants (logg~0-3) → ~1, dwarfs → ~0
        try:
            logg = gf("feature_logg_proxy", np.nan)
            comp["feature_giant_index"] = (
                float(1.0 / (1.0 + np.exp((logg - 2.5) / 0.5)))
                if np.isfinite(logg)
                else np.nan
            )
        except Exception:
            comp["feature_giant_index"] = np.nan

        # 5. subgiant_index — P(subgiant): Gaussian peak at logg=3.5
        #    Captures the intermediate-gravity population (IV luminosity class)
        try:
            logg = gf("feature_logg_proxy", np.nan)
            comp["feature_subgiant_index"] = (
                float(np.exp(-0.5 * ((logg - 3.5) / 0.5) ** 2))
                if np.isfinite(logg)
                else np.nan
            )
        except Exception:
            comp["feature_subgiant_index"] = np.nan

        # 6. CNO_index — carbon/nitrogen/oxygen processing indicator
        #    Elevated CN and CH bands → AGB dredge-up, carbon stars
        try:
            cn = gf("feature_molecular_CN_4142", 0.0)
            ch = gf("feature_molecular_CH_4300", 0.0)
            comp["feature_CNO_index"] = float(np.nan_to_num(cn) + np.nan_to_num(ch))
        except Exception:
            comp["feature_CNO_index"] = np.nan

        # 7. s_process_enhanced — continuous degree of s-process enrichment
        #    Derived from existing feature_s_process_index (Ba+Sr / Fe ratio).
        #    Clipped to [0, 1]: solar ~ 0.1, s-process star > 0.5
        try:
            s_idx = gf("feature_s_process_index", np.nan)
            comp["feature_s_process_enhanced"] = (
                float(np.clip(s_idx / 2.0, 0.0, 1.0)) if np.isfinite(s_idx) else np.nan
            )
        except Exception:
            comp["feature_s_process_enhanced"] = np.nan

        # 8. metal_poor_index — spectroscopic metal-poor indicator
        #    Inverted metal_index_combined: weak Ca+Fe+Mg → high metal-poor probability
        #    Normalised so that solar (combined ~ 5) → 0, metal-poor → 1
        try:
            met = gf("feature_metal_index_combined", np.nan)
            comp["feature_metal_poor_index"] = (
                float(np.clip(1.0 - met / 5.0, 0.0, 1.0))
                if np.isfinite(met)
                else np.nan
            )
        except Exception:
            comp["feature_metal_poor_index"] = np.nan

        # 9-11. Kinematic disk/halo indices — filled post-merge by
        #        add_gaia_derived_features once v_tan and FeH_proxy are available.
        #        Kept as NaN here so that the feature names are locked into the
        #        canonical order during the dry-run in _populate_feature_names.
        for name in ("thin_disk_index", "thick_disk_index", "halo_index"):
            comp[f"feature_{name}"] = np.nan

        return comp

    # =========================================================================
    # MAIN ENTRY POINT
    # =========================================================================

    def extract_features(
        self,
        matched_lines: Mapping[str, Optional[Tuple[float, float]]],
        wavelength: np.ndarray,
        flux_norm: np.ndarray,
        invvar: np.ndarray,
        catalog_row: Optional[pd.Series] = None,
    ) -> List[float]:
        """Extract the feature vector for a single spectrum.

        Merges V1 and V2 extractions into a single vector ordered in a
        stable fashion (alphabetical key order, locked on the first call).

        Parameters
        ----------
        matched_lines : Mapping[str, Optional[Tuple[float, float]]]
            Line → (detected_λ, prominence) mapping, or None if undetected.
        wavelength : np.ndarray
            Wavelength array in Å.
        flux_norm : np.ndarray
            Normalised flux.
        invvar : np.ndarray
            Per-pixel inverse-variance.
        catalog_row : Optional[pd.Series]
            Master-catalogue row for this spectrum (optional, for photometric
            colours).  In the normal pipeline this is None; catalogue colours
            are filled post-merge by ``add_photometric_composites``.

        Returns
        -------
        List[float]
            Vector aligned with ``self.feature_names`` (stable order).
        """
        sigma = safe_sigma_from_invvar(invvar)
        spec_full = Spectrum(
            spectral_axis=wavelength * u.AA,
            flux=flux_norm * u.adu,
            uncertainty=StdDevUncertainty(sigma),
        )

        # ── 1. V1 base: prominence, FWHM, EW for the 6 target lines ────────
        all_line_features: Dict[str, float] = {}

        for line in self.base_lines:
            line_key = line.replace(" ", "")
            prominence = 0.0
            fwhm_val = np.nan
            eq_width_v = np.nan

            if match := matched_lines.get(line):
                wl_det, prom = match
                prominence = float(prom)
                try:
                    win = 20.0
                    mask = (wavelength > wl_det - win) & (wavelength < wl_det + win)
                    if mask.sum() > 5:
                        emission_like = 1.0 - savgol_filter(
                            flux_norm[mask], window_length=5, polyorder=2
                        )
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", RuntimeWarning)
                            if np.max(emission_like) > 0:
                                fwhm_val = float(
                                    gaussian_fwhm(
                                        Spectrum(
                                            spectral_axis=wavelength[mask] * u.AA,
                                            flux=emission_like * u.adu,
                                        )
                                    ).to_value(u.AA)
                                )
                        ew_raw = equivalent_width(
                            spec_full,
                            regions=SpectralRegion(
                                (wl_det - win) * u.AA, (wl_det + win) * u.AA
                            ),
                        )
                        eq_width_v = float(ew_raw.to_value(u.AA))
                except Exception:
                    pass

            all_line_features[f"feature_{line_key}_prominence"] = prominence
            all_line_features[f"feature_{line_key}_fwhm"] = fwhm_val
            all_line_features[f"feature_{line_key}_eq_width"] = eq_width_v

        # Prominence ratios
        for name, (num_key, den_key) in self.ratio_definitions.items():
            num = float(all_line_features.get(num_key, 0.0))
            den = float(all_line_features.get(den_key, 0.0))
            all_line_features[f"feature_{name}"] = num / (den + 1e-6)

        # V1 spectral indices
        all_line_features.update(
            self._calculate_spectroscopic_indices(wavelength, flux_norm)
        )

        # ── 2. V2 extensions ──────────────────────────────────────────────────
        balmer_dict = self._extract_balmer_series_full(wavelength, flux_norm, invvar)

        # Compute Balmer ratios using V1 EWs for Hα/Hβ
        hg_ew = balmer_dict.pop("_ew_Hgamma", np.nan)
        hd_ew = balmer_dict.pop("_ew_Hdelta", np.nan)
        hb_ew_v1 = abs(
            float(all_line_features.get("feature_Hβ_eq_width", np.nan) or np.nan)
        )

        if np.isfinite(hg_ew) and np.isfinite(hb_ew_v1) and hb_ew_v1 > 0:
            balmer_dict["feature_balmer_ratio_gamma_beta"] = hg_ew / hb_ew_v1
        if np.isfinite(hd_ew) and np.isfinite(hg_ew) and hg_ew > 0:
            balmer_dict["feature_balmer_ratio_delta_gamma"] = hd_ew / hg_ew
        if np.isfinite(
            balmer_dict.get("feature_balmer_ratio_gamma_beta", np.nan)
        ) and np.isfinite(balmer_dict.get("feature_balmer_ratio_delta_gamma", np.nan)):
            r1 = balmer_dict["feature_balmer_ratio_gamma_beta"]
            r2 = balmer_dict["feature_balmer_ratio_delta_gamma"]
            # Normalised index: hot stars → close to 1, cool stars → > 1
            balmer_dict["feature_balmer_temperature_index"] = float((r1 + r2) / 2.0)

        combined_dict: Dict[str, float] = {
            **all_line_features,
            **self._extract_molecular_bands(wavelength, flux_norm),
            **balmer_dict,
            **self._extract_metal_lines(wavelength, flux_norm, invvar),
            **self._extract_spectral_indices_extended(wavelength, flux_norm),
            **self._extract_continuum_analysis(wavelength, flux_norm),
            **self._extract_line_profiles(matched_lines, wavelength, flux_norm),
            **self._extract_color_features(wavelength, flux_norm, catalog_row),
        }

        # ── 3. Composite indices (require the full dict) ─────────────────
        final_dict = {**combined_dict, **self._extract_composite_indices(combined_dict)}

        # ── 4. Return in canonical order ──────────────────────────────────────
        if not self.feature_names:
            # First real call: lock alphabetical order
            self.feature_names = sorted(final_dict.keys())

        return [float(final_dict.get(name, np.nan)) for name in self.feature_names]

    # =========================================================================
    # Batch helper
    # =========================================================================

    def batch_features(
        self,
        matched_lines_list: List[Mapping[str, Optional[Tuple[float, float]]]],
        wavelengths: List[np.ndarray],
        flux_norms: List[np.ndarray],
        invvars: List[np.ndarray],
        catalog_rows: Optional[List[pd.Series]] = None,
    ) -> np.ndarray:
        """Vectorised extraction for a list of spectra.

        Parameters
        ----------
        matched_lines_list : List[Mapping]
            One matched_lines element per spectrum.
        wavelengths, flux_norms, invvars : List[np.ndarray]
            Per-spectrum arrays (wavelength grids may differ).
        catalog_rows : Optional[List[pd.Series]]
            Corresponding catalogue rows (None if unavailable).

        Returns
        -------
        np.ndarray of shape (n_spectra, n_features)
        """
        rows = []
        for i, ml in enumerate(matched_lines_list):
            cat_row = catalog_rows[i] if catalog_rows is not None else None
            rows.append(
                self.extract_features(
                    ml, wavelengths[i], flux_norms[i], invvars[i], cat_row
                )
            )
        return np.asarray(rows, dtype=float)


# ============================================================================
# POST-PROCESSING HELPERS (called by processing.py after catalogue join)
# ============================================================================


def add_gaia_derived_features(
    df: pd.DataFrame, *, min_parallax_snr: float = 5.0
) -> tuple[pd.DataFrame, list[str]]:
    """Compute astrophysical features derived from Gaia DR3 columns.

    Requires that the DataFrame has already been joined with the
    master_catalog_gaia.  Adds bp_g, g_rp, bp_rp0 (dereddened),
    M_G (absolute magnitude), dist_pc, v_tan, etc.

    Parameters
    ----------
    df : pd.DataFrame
        Merged DataFrame (features + catalogue).
    min_parallax_snr : float
        Minimum parallax SNR required to compute distance.

    Returns
    -------
    tuple[pd.DataFrame, list[str]]
        Enriched DataFrame and list of newly added columns.
    """
    df = df.copy()
    new_cols: list[str] = []
    cols = set(df.columns)

    # Raw Gaia colours
    if {"phot_bp_mean_mag", "phot_g_mean_mag"} <= cols and "bp_g" not in df:
        df["bp_g"] = df["phot_bp_mean_mag"] - df["phot_g_mean_mag"]
        new_cols.append("bp_g")
    if {"phot_g_mean_mag", "phot_rp_mean_mag"} <= cols and "g_rp" not in df:
        df["g_rp"] = df["phot_g_mean_mag"] - df["phot_rp_mean_mag"]
        new_cols.append("g_rp")

    # Parallax SNR
    if {"parallax", "parallax_error"} <= cols:
        df["parallax_snr"] = df["parallax"].abs() / (df["parallax_error"].abs() + 1e-9)
        new_cols.append("parallax_snr")
    else:
        df["parallax_snr"] = np.nan

    # Astrometric RUWE flag
    if "ruwe" in cols:
        df["is_good_ruwe"] = (df["ruwe"] < 1.4).astype("int8")
        new_cols.append("is_good_ruwe")

    # Dereddened colour (BP-RP)₀
    if {"bp_rp", "ebpminrp_gspphot"} <= cols:
        df["bp_rp0"] = df["bp_rp"] - df["ebpminrp_gspphot"].fillna(0)
        new_cols.append("bp_rp0")
    elif "bp_rp" in cols:
        df["bp_rp0"] = df["bp_rp"]
        new_cols.append("bp_rp0")

    # GSP-Phot photometric distance
    name_dist = None
    for cand in ("distance_gspphot", "r_med_photogeo", "r_med_geo"):
        if cand in cols:
            name_dist = cand
            break
    if name_dist:
        df["dist_pc"] = pd.to_numeric(df[name_dist], errors="coerce")
        new_cols.append("dist_pc")
    elif "parallax_snr" in df.columns:
        use = df["parallax_snr"] >= min_parallax_snr
        df["dist_pc"] = np.nan
        df.loc[use, "dist_pc"] = 1000.0 / df.loc[use, "parallax"].abs()
        new_cols.append("dist_pc")

    # Tangential velocity
    if {"pmra", "pmdec", "dist_pc"} <= (cols | set(new_cols)):
        pm_total = np.sqrt(df.get("pmra", 0) ** 2 + df.get("pmdec", 0) ** 2)
        df["v_tan_kms"] = 4.74047 * pm_total * df.get("dist_pc", np.nan) / 1000.0
        new_cols.append("v_tan_kms")

    # Absolute magnitude G₀
    def _abs_mag(band: str) -> None:
        col_app = f"phot_{band}_mean_mag"
        M_col = f"M_{band.upper()}"
        if col_app not in df.columns or "dist_pc" not in df.columns:
            return
        use = (
            (df["parallax_snr"].fillna(0) >= min_parallax_snr)
            & df["dist_pc"].notna()
            & (df["dist_pc"] > 0)
        )
        df[M_col] = np.nan
        ag = df.get("ag_gspphot", pd.Series(0.0, index=df.index)).fillna(0)
        ext = ag if band == "g" else pd.Series(0.0, index=df.index)
        df.loc[use, M_col] = (
            df.loc[use, col_app]
            - 5 * np.log10(df.loc[use, "dist_pc"])
            + 5
            - ext.loc[use]
        )
        new_cols.append(M_col)

    for band in ("g", "bp", "rp"):
        if f"phot_{band}_mean_mag" in df.columns:
            _abs_mag(band)

    if "delta_ms" in df.columns:
        df["is_giant_like"] = (df["delta_ms"] <= -0.8).astype("int8")
        df["is_subdwarf_like"] = (df["delta_ms"] >= +0.6).astype("int8")
        new_cols += ["is_giant_like", "is_subdwarf_like"]

    # Log-transforms
    if "v_tan_kms" in df.columns:
        x = df["v_tan_kms"]
        df["v_tan_kms_log10"] = np.where(x > 0, np.log10(x), np.nan)
        new_cols.append("v_tan_kms_log10")
    if "parallax_snr" in df.columns:
        s = df["parallax_snr"]
        df["parallax_snr_log10"] = np.where(s > 0, np.log10(s), np.nan)
        new_cols.append("parallax_snr_log10")

    # --- Kinematic disk/halo membership proxies ---
    # Overwrites the NaN stubs left by _extract_composite_indices when
    # v_tan_kms (from Gaia proper motions) and feature_FeH_proxy are available.
    #
    # Reference thresholds (Bensby et al. 2003 / Reddy et al. 2006):
    #   Thin disk  : v_tan ≲ 50 km/s,  [Fe/H] ≳ −0.3
    #   Thick disk : v_tan ~ 50–150 km/s, [Fe/H] ~ −0.5
    #   Halo       : v_tan ≳ 200 km/s, [Fe/H] ≲ −1.0
    #
    # The formulae are continuous (not binary) to give a smooth probability-like
    # score in [0, 1] suitable for XGBoost feature input.
    if "v_tan_kms" in df.columns and "feature_FeH_proxy" in df.columns:
        vtan = pd.to_numeric(df["v_tan_kms"], errors="coerce")
        feh = pd.to_numeric(df["feature_FeH_proxy"], errors="coerce")

        # Thin disk: fast decay away from v_tan = 0, penalised for low metallicity
        thin_kin = np.exp(-0.5 * (vtan / 40.0) ** 2)
        thin_met = np.clip(1.0 + feh / 0.3, 0.0, 1.0)  # peaks at [Fe/H] = 0
        df["feature_thin_disk_index"] = np.clip(thin_kin * thin_met, 0.0, 1.0)

        # Thick disk: Gaussian centred at v_tan = 80 km/s, intermediate metallicity
        thick_kin = np.exp(-0.5 * ((vtan - 80.0) / 50.0) ** 2)
        thick_met = np.clip(1.0 + feh / 1.0, 0.0, 1.0)  # peaks at [Fe/H] = 0
        df["feature_thick_disk_index"] = np.clip(thick_kin * thick_met, 0.0, 1.0)

        # Halo: linear ramp above v_tan = 150 km/s, boosted by metal-poor indicator
        halo_kin = np.clip((vtan - 150.0) / 200.0, 0.0, 1.0)
        halo_met = np.clip(-feh / 1.5, 0.0, 1.0)  # peaks at [Fe/H] = −1.5
        df["feature_halo_index"] = np.clip(halo_kin * halo_met, 0.0, 1.0)

        new_cols += [
            "feature_thin_disk_index",
            "feature_thick_disk_index",
            "feature_halo_index",
        ]

    return df, new_cols


def add_photometric_composites(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Fill colour features and composite indices after catalogue merge.

    Must be called **after** ``add_gaia_derived_features`` and after the
    photometric colour indices (``feature_color_gr``, ``feature_color_ri``)
    have been computed by ``processing.py``.

    Fills columns left as NaN during per-spectrum extraction:
    - ``feature_bp_g``         from the catalogue column ``bp_g``.
    - ``feature_g_rp``         from the catalogue column ``g_rp``.
    - ``feature_color_ug``     from magnitude_u − magnitude_g.
    - ``feature_color_iz``     from magnitude_i − magnitude_z.
    - ``feature_Q_parameter``  (U−B) − 0.72·(B−V): reddening-free index.
    - ``feature_X_parameter``  (B−V) + 2·(V−R): temperature proxy.

    Parameters
    ----------
    df : pd.DataFrame
        Post-merge DataFrame with catalogue columns available.

    Returns
    -------
    tuple[pd.DataFrame, list[str]]
        Enriched DataFrame and list of modified / added columns.
    """
    df = df.copy()
    modified: list[str] = []

    # --- bp_g and g_rp from Gaia columns ---
    if "bp_g" in df.columns and "feature_bp_g" in df.columns:
        mask = df["feature_bp_g"].isna() & df["bp_g"].notna()
        df.loc[mask, "feature_bp_g"] = df.loc[mask, "bp_g"]
        modified.append("feature_bp_g")

    if "g_rp" in df.columns and "feature_g_rp" in df.columns:
        mask = df["feature_g_rp"].isna() & df["g_rp"].notna()
        df.loc[mask, "feature_g_rp"] = df.loc[mask, "g_rp"]
        modified.append("feature_g_rp")

    # --- color_ug and color_iz from LAMOST photometry ---
    if {"magnitude_u", "magnitude_g"}.issubset(
        df.columns
    ) and "feature_color_ug" in df.columns:
        u_m = pd.to_numeric(df["magnitude_u"], errors="coerce").replace(99.0, np.nan)
        g_m = pd.to_numeric(df["magnitude_g"], errors="coerce").replace(99.0, np.nan)
        computed = u_m - g_m
        mask = df["feature_color_ug"].isna() & computed.notna()
        df.loc[mask, "feature_color_ug"] = computed[mask]
        modified.append("feature_color_ug")

    if {"magnitude_i", "magnitude_z"}.issubset(
        df.columns
    ) and "feature_color_iz" in df.columns:
        i_m = pd.to_numeric(df["magnitude_i"], errors="coerce").replace(99.0, np.nan)
        z_m = pd.to_numeric(df["magnitude_z"], errors="coerce").replace(99.0, np.nan)
        computed = i_m - z_m
        mask = df["feature_color_iz"].isna() & computed.notna()
        df.loc[mask, "feature_color_iz"] = computed[mask]
        modified.append("feature_color_iz")

    # --- Q_parameter and X_parameter (require feature_color_gr) ---
    if "feature_color_gr" in df.columns:
        gr = df["feature_color_gr"]

        if "feature_color_ug" in df.columns and "feature_Q_parameter" in df.columns:
            # Q = (U-G) - 0.72*(G-R)  (adapted from Johnson UBV to LAMOST filters)
            ug = df["feature_color_ug"]
            q = ug - 0.72 * gr
            mask = df["feature_Q_parameter"].isna() & q.notna()
            df.loc[mask, "feature_Q_parameter"] = q[mask]
            modified.append("feature_Q_parameter")

        if "feature_color_ri" in df.columns and "feature_X_parameter" in df.columns:
            ri = df["feature_color_ri"]
            x = gr + 2.0 * ri
            mask = df["feature_X_parameter"].isna() & x.notna()
            df.loc[mask, "feature_X_parameter"] = x[mask]
            modified.append("feature_X_parameter")

    return df, modified


def add_line_composites(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """EW and FWHM combinations (Balmer / metals) — inherited from V1.

    Computes ratios and depth proxies from existing spectroscopic columns.
    Supports both Unicode (Hα/Hβ) and ASCII (Ha/Hb) column names through
    the ``pick`` helper.

    Returns
    -------
    tuple[pd.DataFrame, list[str]]
        Enriched DataFrame and list of newly added columns.
    """
    df = df.copy()
    new: list[str] = []

    def pick(*names: str) -> Optional[str]:
        return next((n for n in names if n in df.columns), None)

    Ha_ew = pick("feature_Ha_eq_width", "feature_Hα_eq_width")
    Hb_ew = pick("feature_Hb_eq_width", "feature_Hβ_eq_width")
    Ha_fwhm = pick("feature_Ha_fwhm", "feature_Hα_fwhm")
    Hb_fwhm = pick("feature_Hb_fwhm", "feature_Hβ_fwhm")
    CaK_ew = pick("feature_CaIIK_eq_width")
    CaH_ew = pick("feature_CaIIH_eq_width")
    MgB_ew = pick("feature_Mg_b_eq_width", "feature_Mgb_eq_width")
    NaD_ew = pick("feature_Na_D_eq_width", "feature_NaD_eq_width")

    if Ha_ew and Hb_ew:
        a = df[Ha_ew].abs()
        b = df[Hb_ew].abs()
        df["ratio_EW_Ha_Hb"] = np.where((b.notna()) & (b != 0), a / b, np.nan)
        new.append("ratio_EW_Ha_Hb")

    if Ha_fwhm and Hb_fwhm:
        df["ratio_FWHM_Ha_Hb"] = np.where(
            (df[Hb_fwhm].notna()) & (df[Hb_fwhm] != 0),
            df[Ha_fwhm] / df[Hb_fwhm],
            np.nan,
        )
        new.append("ratio_FWHM_Ha_Hb")

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

    if CaH_ew and CaK_ew:
        df["EW_CaHK_sum"] = df[CaH_ew].abs() + df[CaK_ew].abs()
        new.append("EW_CaHK_sum")
        df["ratio_EW_CaK_CaH"] = np.where(
            df[CaH_ew].abs() > 0, df[CaK_ew].abs() / df[CaH_ew].abs(), np.nan
        )
        new.append("ratio_EW_CaK_CaH")

    if MgB_ew and NaD_ew:
        df["ratio_EW_MgB_NaD"] = np.where(
            df[NaD_ew].abs() > 0, df[MgB_ew].abs() / df[NaD_ew].abs(), np.nan
        )
        new.append("ratio_EW_MgB_NaD")

    if (CaH_ew and CaK_ew) and (Ha_ew and Hb_ew):
        metals = df[CaH_ew].abs() + df[CaK_ew].abs()
        balmer = df[Ha_ew].abs() + df[Hb_ew].abs()
        df["contrast_metals_vs_balmer"] = np.where(balmer > 0, metals / balmer, np.nan)
        new.append("contrast_metals_vs_balmer")

    return df, new


def add_main_sequence_delta(
    df: pd.DataFrame, *, min_parallax_snr: float = 10.0, poly_degree: int = 3
) -> tuple[pd.DataFrame, np.ndarray | None]:
    """Add ``delta_ms`` = M_G − M_G_hat((BP−RP)₀).

    ``delta_ms`` < 0 : over-luminous star (likely giant).
    ``delta_ms`` > 0 : under-luminous star (possible white dwarf / subdwarf).

    Returns
    -------
    tuple[pd.DataFrame, np.ndarray | None]
        (modified_df, polynomial_coefficients) — coeffs is None if too few
        clean stars are available.
    """
    df = df.copy()
    needed = {"bp_rp0", "M_G", "parallax_snr", "is_good_ruwe"}
    if not needed.issubset(df.columns):
        df["delta_ms"] = np.nan
        return df, None

    mask = (
        (df["is_good_ruwe"] == 1)
        & (df["parallax_snr"] >= min_parallax_snr)
        & df["bp_rp0"].between(-0.2, 3.5)
        & df["M_G"].between(-5, 15)
    )

    x = df.loc[mask, "bp_rp0"].astype(float)
    y = df.loc[mask, "M_G"].astype(float)

    if x.count() < 200:
        df["delta_ms"] = np.nan
        return df, None

    coeffs = np.polyfit(x, y, poly_degree)
    yhat = np.polyval(coeffs, df["bp_rp0"])
    df["delta_ms"] = df["M_G"] - yhat
    return df, coeffs


def _signed_log1p(x: pd.Series) -> pd.Series:
    """Signed log1p: preserves sign for EWs that can be negative."""
    return np.sign(x) * np.log1p(np.abs(x))


def stabilize_spectral_features(df: pd.DataFrame) -> pd.DataFrame:
    """Winsorisation + log transforms to stabilise spectral features.

    - EW         → clip [0.5 %, 99.5 %] then signed-log1p.
    - Prominences and FWHM → clip [0.5 %, 99.5 %] then log1p.
    - Adds binary ``{feature}_present`` flags for prominence columns.

    Note: this transformation is applied before XGBoost training.
    For inference, transformations are handled by the scikit-learn
    pipeline saved inside the model artefact.
    """
    df = df.copy()
    ew_cols = [c for c in df.columns if c.endswith("_eq_width")]
    prom_cols = [c for c in df.columns if c.endswith("_prominence")]
    fwhm_cols = [c for c in df.columns if c.endswith("_fwhm")]

    def clip_col(s: pd.Series) -> pd.Series:
        lo, hi = s.quantile(0.005), s.quantile(0.995)
        return s.clip(lower=lo, upper=hi)

    for cols, transform in [
        (ew_cols, _signed_log1p),
        (prom_cols, np.log1p),
        (fwhm_cols, np.log1p),
    ]:
        for c in cols:
            df[c] = clip_col(df[c])
            df[c] = transform(df[c].fillna(0.0).astype(float))

    for c in prom_cols:
        flag = c.replace("_prominence", "_present")
        df[flag] = (df[c] > 0).astype(np.uint8)

    return df
