"""AstroSpectro — Prétraitement des spectres (FITS -> arrays propres)

Ce module centralise les opérations de bas niveau nécessaires avant l’extraction
de features :

1) Lecture d’un spectre depuis un HDUList FITS :
   - flux : ligne 0 du tableau
   - inverse-variance (invvar) : ligne 1
   - calcul des longueurs d’onde : 10**(COEFF0 + i * COEFF1)  [wavelength en Å]

2) Normalisation du flux (simple, robuste) pour stabiliser l’analyse.

3) Sécurisation de l’inverse-variance via `sanitize_invvar()` afin d’éviter tout
   warning/NaN lors de la conversion en écart-type (sqrt). Optionnellement,
   on peut retourner directement une `StdDevUncertainty`.

Conventions
-----------
- Longueurs d’onde : Å.
- Les spectres attendus sont déjà “ligne-centré” (absorption en négatif
  après inversion éventuelle plus en amont si besoin).
- Les helpers de sécurité proviennent de `src/utils.py`.

Exemple minimal
---------------
>>> sp = SpectraPreprocessor()
>>> # hdul: astropy.io.fits.HDUList
>>> wl, flux, invvar = sp.load_spectrum(hdul)
>>> flux_n = sp.normalize_spectrum(flux)
>>> spec = sp.prepare(hdul, normalize=True, return_uncertainty=True)
>>> spec.wavelength.shape, spec.flux_norm.shape
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

# Helpers “sécurité invvar” (évite warnings/NaN dans sqrt)
from utils import sanitize_invvar, make_stddev_uncertainty_from_invvar


@dataclass
class ProcessedSpectrum:
    """Conteneur typé pour un spectre prétraité."""

    wavelength: np.ndarray  # (N,) en Å
    flux: np.ndarray  # (N,)
    invvar: np.ndarray  # (N,) inverse-variance (brut)
    flux_norm: Optional[np.ndarray] = None  # (N,) flux normalisé (si demandé)
    invvar_clean: Optional[np.ndarray] = None  # (N,) invvar après “sanitize”
    uncertainty: Optional[object] = None  # StdDevUncertainty (si demandé)


class SpectraPreprocessor:
    """
    Outils de prétraitement : lecture FITS, normalisation, sécurisation invvar.
    """

    # --------------------------------------------------------------------- #
    # Lecture FITS -> (wavelength, flux, invvar)
    # --------------------------------------------------------------------- #
    def load_spectrum(self, hdul) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extrait le spectre (wavelength, flux, invvar) depuis un HDUList FITS.

        Le flux (ligne 0) et l'inverse-variance (ligne 1) sont lus depuis
        `hdul[0].data`, tandis que les longueurs d'onde sont reconstruites
        via les mots-clés `COEFF0` (départ log10) et `COEFF1` (pas log10).

        Args:
            hdul: Objet `astropy.io.fits.HDUList` déjà ouvert.

        Returns:
            tuple(np.ndarray, np.ndarray, np.ndarray):
                - wavelength (N,): longueurs d'onde en Å
                - flux (N,)
                - invvar (N,): inverse-variance brute

        Raises:
            ValueError: si la forme des données est invalide ou si COEFF0/COEFF1
                        sont absents du header.
        """
        header = hdul[0].header
        data = hdul[0].data

        if data.ndim < 2 or data.shape[0] < 2:
            raise ValueError(
                "Le tableau de données FITS est invalide (moins de 2 lignes)."
            )

        flux = data[0]
        invvar = data[1]

        if "COEFF0" in header and "COEFF1" in header:
            loglam_start = header["COEFF0"]
            loglam_step = header["COEFF1"]
            loglam = loglam_start + np.arange(len(flux)) * loglam_step
            wavelength = 10**loglam
        else:
            raise ValueError("Header FITS invalide : COEFF0 ou COEFF1 manquants.")

        return wavelength, flux, invvar

    # --------------------------------------------------------------------- #
    # Normalisation simple & robuste
    # --------------------------------------------------------------------- #
    def normalize_spectrum(
        self, flux: np.ndarray, method: str = "median"
    ) -> np.ndarray:
        """
        Normalise un vecteur de flux.

        Actuellement, seule la normalisation **par la médiane** est proposée
        (méthode robuste pour limiter l'effet des raies et outliers).

        Args:
            flux: Tableau 1D de flux.
            method: Stratégie ("median" uniquement pour l’instant).

        Returns:
            np.ndarray: flux normalisé (même forme que `flux`).
        """
        if method != "median":
            raise ValueError(f"Méthode de normalisation inconnue: {method!r}")

        med = float(np.median(flux))
        return flux / med if med > 0 else flux

    # --------------------------------------------------------------------- #
    # Pipeline court : lecture + options de normalisation & incertitudes
    # --------------------------------------------------------------------- #
    def prepare(
        self,
        hdul,
        normalize: bool = True,
        return_uncertainty: bool = False,
        min_invvar: float = 1e-12,
    ) -> ProcessedSpectrum:
        """
        Prétraite un spectre complet depuis un HDUList FITS.

        Étapes:
            1) Lecture (wavelength, flux, invvar).
            2) Sécurisation de l'inverse-variance (`sanitize_invvar`).
            3) Normalisation du flux (optionnelle).
            4) Construction de l'incertitude (optionnelle) via
               `make_stddev_uncertainty_from_invvar`.

        Args:
            hdul: HDUList FITS.
            normalize: Si True, calcule et renvoie `flux_norm`.
            return_uncertainty: Si True, calcule et renvoie une
                `StdDevUncertainty` basée sur l'invvar nettoyé.
            min_invvar: Planche minimale appliquée à l'invvar lors du nettoyage.

        Returns:
            ProcessedSpectrum: conteneur avec les champs utiles remplis.
        """
        wl, flux, invvar = self.load_spectrum(hdul)

        invvar_clean = sanitize_invvar(invvar, min_val=min_invvar)
        flux_norm = self.normalize_spectrum(flux) if normalize else None

        uncertainty = None
        if return_uncertainty:
            uncertainty = make_stddev_uncertainty_from_invvar(
                invvar_clean, min_val=min_invvar
            )

        return ProcessedSpectrum(
            wavelength=wl,
            flux=flux,
            invvar=invvar,
            flux_norm=flux_norm,
            invvar_clean=invvar_clean,
            uncertainty=uncertainty,
        )
