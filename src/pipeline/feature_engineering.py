"""
AstroSpectro — Extraction de descripteurs spectroscopiques « ligne-centrés ».

Ce module fournit la classe `FeatureEngineer`, chargée de :
- définir le vocabulaire de raies cibles et les métriques calculées par raie
  (prominence, FWHM, EW),
- calculer des indices/ratios simples sur des bandes de flux,
- retourner des vecteurs de features **dans un ordre stable** (`feature_names`).

Conventions
-----------
- Longueurs d’onde et FWHM en Ångströms (Å).
- EW (equivalent width) en Å, calculée sur le spectre normalisé.
- Les “prominences” proviennent du détecteur de pics amont.

Entrées / Sorties attendues
---------------------------
- Entrée : `matched_lines` (dict raie → (λ_detectée, prominence)), `wavelength`,
  `flux_norm`, `invvar` (inverse-variance).
- Sortie : vecteur `list[float]` aligné sur `self.feature_names`.

API publique (principales méthodes)
-----------------------------------
- `extract_features(matched_lines, wavelength, flux_norm, invvar)` → list[float]
- `feature_names` : ordre canonique des descripteurs
- (interne) `_calculate_spectroscopic_indices(...)` → dict d’indices

Exemple minimal
---------------
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
    """
    Extracteur de descripteurs spectroscopiques « ligne-centrés ».

    Cette classe :
      - définit un vocabulaire de raies cibles et de métriques par raie,
      - calcule quelques indices spectraux simples (bandes de flux),
      - construit un vecteur de features dans un ordre **stable** (self.feature_names).

    Notes
    -----
    - Les largeurs/FWHM sont en Angströms (Å).
    - Les largeurs équivalentes (EW) sont en Angströms, calculées sur le spectre normalisé.
    - Les « prominences » proviennent du détecteur de pics (amplitude/score transmis).
    """

    def __init__(self) -> None:
        # 1) Définition du vocabulaire de raies et métriques
        self.base_lines = ["Hα", "Hβ", "CaII K", "CaII H", "Mg_b", "Na_D"]
        self.line_metrics = ["prominence", "fwhm", "eq_width"]

        # Ratios entre prominences (numérateur, dénominateur)
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

        # Indices spectraux (bande feature, bande continuum), en Å
        self.index_definitions: Dict[str, Tuple[List[float], List[float]]] = {
            "TiO5": ([7126, 7135], [7042, 7052]),
        }

        # 2) Génère la liste complète et ordonnée des noms de features
        self.feature_names: List[str] = []
        for line in self.base_lines:
            safe = line.replace(" ", "").replace("II", "II")
            for metric in self.line_metrics:
                self.feature_names.append(f"feature_{safe}_{metric}")
        # NB: l’ordre ici fait foi pour tout le pipeline (stabilité des colonnes).

        self.feature_names += [f"feature_{name}" for name in self.ratio_definitions]
        self.feature_names += [
            f"feature_index_{name}" for name in self.index_definitions
        ]

    # -------------------------------------------------------------------------
    # Indices spectraux
    # -------------------------------------------------------------------------

    def _calculate_spectroscopic_indices(
        self, wavelength: np.ndarray, flux_norm: np.ndarray
    ) -> Dict[str, float]:
        """
        Calcule quelques indices spectraux simples basés sur des bandes de flux.

        L’indice est défini comme le **rapport** : moyenne(bande_feature) / moyenne(bande_continuum).

        Parameters
        ----------
        wavelength : np.ndarray
            Longueurs d’onde en Å, même taille que `flux_norm`.
        flux_norm : np.ndarray
            Flux normalisé (autour de 1), même taille que `wavelength`.

        Returns
        -------
        Dict[str, float]
            Dictionnaire `{ "feature_index_<nom>": valeur }`. En cas d’échec,
            l’indice vaut 0.0 pour rester robuste.
        """
        indices: Dict[str, float] = {}
        for name, (feature_band, continuum_band) in self.index_definitions.items():
            try:
                f_mask = (wavelength >= feature_band[0]) & (
                    wavelength <= feature_band[1]
                )
                c_mask = (wavelength >= continuum_band[0]) & (
                    wavelength <= continuum_band[1]
                )

                mean_f = float(np.mean(flux_norm[f_mask]))
                mean_c = float(np.mean(flux_norm[c_mask]))
                indices[f"feature_index_{name}"] = mean_f / (mean_c + 1e-6)
            except Exception:
                # Robustesse: en cas de bande vide / masque hors-grille,
                # on garde 0.0 pour éviter de casser la chaîne.
                indices[f"feature_index_{name}"] = 0.0
        return indices

    # -------------------------------------------------------------------------
    # Extraction des features par raie
    # -------------------------------------------------------------------------

    def extract_features(
        self,
        matched_lines: Mapping[str, Optional[Tuple[float, float]]],
        wavelength: np.ndarray,
        flux_norm: np.ndarray,
        invvar: np.ndarray,
    ) -> List[float]:
        """
        Extrait un vecteur de descripteurs pour les raies détectées.

        Pour chaque raie de `self.base_lines`, on renseigne :
          - *prominence* (score transmis via `matched_lines`),
          - *fwhm* (Å) estimée par ajustement gaussien local,
          - *eq_width* (Å) via `specutils.equivalent_width` sur le spectre normalisé.

        On calcule ensuite :
          - des *ratios* de prominences (définis dans `self.ratio_definitions`),
          - des *indices* spectraux simples (cf. `_calculate_spectroscopic_indices`).

        Parameters
        ----------
        matched_lines : Mapping[str, Optional[Tuple[float, float]]]
            Pour chaque raie, un tuple `(lambda_detectée, prominence)` ou `None`.
            Les clés doivent correspondre à `self.base_lines`.
        wavelength : np.ndarray
            Longueurs d’onde en Å.
        flux_norm : np.ndarray
            Flux normalisé (même taille que `wavelength`).
        invvar : np.ndarray
            Inverse-variance du flux (même taille), potentiellement bruitée (NaN/Inf/négative).

        Returns
        -------
        List[float]
            Le vecteur de features **dans l’ordre `self.feature_names`**.
        """
        all_line_features: Dict[str, float] = {}

        # Incertitudes robustes (supprime/évite les NaN/Inf/neg avant sqrt)
        sigma = safe_sigma_from_invvar(invvar)  # utils.safe_sigma_from_invvar
        uncertainty = StdDevUncertainty(sigma)

        spec_full = Spectrum(
            spectral_axis=wavelength * u.AA,
            flux=flux_norm * u.adu,
            uncertainty=uncertainty,
        )

        # 1) Métriques par raie
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
                    # Fenêtre locale ±20 Å autour de la détection
                    win = 20.0
                    mask = (wavelength > wl_detected - win) & (
                        wavelength < wl_detected + win
                    )

                    emission_like = None
                    if np.any(mask) and mask.sum() > 5:
                        # Lissage léger pour stabiliser la FWHM
                        smoothed = savgol_filter(
                            flux_norm[mask], window_length=5, polyorder=2
                        )
                        # Inversion (absorption -> “émission”) pour avoir une FWHM positive
                        emission_like = 1.0 - smoothed

                    with warnings.catch_warnings():
                        warnings.filterwarnings(
                            "ignore",
                            message="invalid value encountered in sqrt",
                            category=RuntimeWarning,
                            module=r".*astropy\.units\.quantity.*",
                        )

                        # FWHM locale si la fenêtre était exploitable
                        if emission_like is not None and np.max(emission_like) > 0:
                            local_spec = Spectrum(
                                spectral_axis=wavelength[mask] * u.AA,
                                flux=emission_like * u.adu,
                            )
                            fwhm_val = float(gaussian_fwhm(local_spec).to_value(u.AA))

                        # Largeur équivalente (EW) sur le spectre normalisé
                        ew = equivalent_width(
                            spec_full,
                            regions=SpectralRegion(
                                (wl_detected - win) * u.AA, (wl_detected + win) * u.AA
                            ),
                        )
                        eq_width_val = float(ew.to_value(u.AA))

                except Exception:
                    # Tolérance locale : on garde les défauts à 0.0 si l’ajustement échoue
                    pass

            all_line_features[f"feature_{line_key}_prominence"] = float(prominence)
            all_line_features[f"feature_{line_key}_fwhm"] = float(fwhm_val)
            all_line_features[f"feature_{line_key}_eq_width"] = float(eq_width_val)

        # 2) Ratios de prominences
        eps = 1e-6
        for name, (num_key, den_key) in self.ratio_definitions.items():
            num = all_line_features.get(num_key, 0.0)
            den = all_line_features.get(den_key, 0.0)
            all_line_features[f"feature_{name}"] = float(num) / (float(den) + eps)

        # 3) Indices spectraux (bandes)
        all_line_features.update(
            self._calculate_spectroscopic_indices(wavelength, flux_norm)
        )

        # 4) Vecteur final ordonné
        return [float(all_line_features.get(name, 0.0)) for name in self.feature_names]

    # -------------------------------------------------------------------------
    # Batch helper (optionnel)
    # -------------------------------------------------------------------------

    def batch_features(
        self,
        matched_lines_list: List[Mapping[str, Optional[Tuple[float, float]]]],
        wavelength: np.ndarray,
        flux_norm: np.ndarray,
        invvar: np.ndarray,
    ) -> np.ndarray:
        """
        Applique `extract_features` à une liste d’objets `matched_lines`.

        Parameters
        ----------
        matched_lines_list : List[Mapping[str, Optional[Tuple[float, float]]]]
            Liste d’annotations de pics (une entrée par spectre).
        wavelength, flux_norm, invvar : np.ndarray
            Séries communes (si tous les spectres sont rééchantillonnés pareil).

        Returns
        -------
        np.ndarray
            Matrice (n_spectres, n_features) construite dans l’ordre `self.feature_names`.
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
    Enrichit le DataFrame avec des features photométriques/astrométriques dérivées Gaia DR3.
    Retourne (df_enrichi, new_cols).
    """
    df = df.copy()
    new_cols: list[str] = []
    cols = set(df.columns)

    # ---------------- Couleurs simples si manquantes ----------------
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

    # ---------------- Magnitude absolue G ----------------
    # A) via parallaxe (mas) si SNR suffisant
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

    # C) Fusion priorisant la parallaxe fiable
    if "M_G_parallax" in df.columns or "M_G_dist" in df.columns:
        df["M_G"] = df.get("M_G_parallax", pd.Series(np.nan, index=df.index))
        if "M_G_dist" in df.columns:
            use_dist = df["M_G"].isna() & df["M_G_dist"].notna()
            df.loc[use_dist, "M_G"] = df.loc[use_dist, "M_G_dist"]
        new_cols.append("M_G")

    # ---------------- RPM & Vitesse tangentielle ----------------
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

    # ---------------- Extinction & couleurs "0" ----------------
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

    # ---------------- Ratios/Logs de flux ----------------
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

    # ---------------- Qualité photométrique simple ----------------
    if {"phot_bp_rp_excess_factor", "bp_rp"} <= cols:
        exp = 1.0 + 0.015 * (df["bp_rp"] ** 2)
        df["bp_rp_excess_dev"] = df["phot_bp_rp_excess_factor"] - exp
        new_cols.append("bp_rp_excess_dev")

    # ---------------- Flags utiles (cast -> int) ----------------
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

    # ---------------- Indicateurs de manquants ----------------
    for col in ("parallax", "distance_gspphot"):
        if col in df.columns:
            name = f"{col}_missing"
            df[name] = df[col].isna().astype("int8")
            new_cols.append(name)

    return df, new_cols


def add_main_sequence_delta(
    df: pd.DataFrame, *, min_parallax_snr: float = 10.0, poly_degree: int = 3
) -> tuple[pd.DataFrame, np.ndarray | None]:
    """
    Ajoute delta_ms = M_G0 - M_G0_hat((BP-RP)_0)
    où M_G0_hat est une polynomiale ajustée sur des étoiles 'propres'.

    Retourne (df_modifié, coeffs) ; coeffs=None si pas assez d'étoiles propres.
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

    if x.count() < 200:  # garde-fou
        df["delta_ms"] = np.nan
        return df, None

    coeffs = np.polyfit(x, y, poly_degree)
    yhat = np.polyval(coeffs, df["bp_rp0"])
    df["delta_ms"] = (
        df["M_G0"] - yhat
    )  # <0: sur-lumineuses (géantes), >0: sous-lumineuses

    return df, coeffs


def _signed_log1p(x: pd.Series) -> pd.Series:
    return np.sign(x) * np.log1p(np.abs(x))


def stabilize_spectral_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Colonnes à transformer (adapte selon ton set)
    ew_cols = [c for c in df.columns if c.endswith("_eq_width")]
    prom_cols = [c for c in df.columns if c.endswith("_prominence")]
    fwhm_cols = [c for c in df.columns if c.endswith("_fwhm")]

    # 2.1 Clip des extrêmes (winsorize soft)
    def clip_col(s: pd.Series):
        lo, hi = s.quantile(0.005), s.quantile(0.995)
        return s.clip(lower=lo, upper=hi)

    for cols, transform in [
        (ew_cols, _signed_log1p),  # signed-log pour EW
        (prom_cols, np.log1p),  # log1p pour prominences (>=0)
        (fwhm_cols, np.log1p),  # log1p pour FWHM (>=0)
    ]:
        for c in cols:
            df[c] = clip_col(df[c])
            df[c] = transform(df[c].fillna(0.0).astype(float))

    # 2.2 Drapeaux binaires "présence de ligne"
    for c in prom_cols:
        flag = c.replace("_prominence", "_present")
        df[flag] = (df[c] > 0).astype(np.uint8)

    return df
