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
