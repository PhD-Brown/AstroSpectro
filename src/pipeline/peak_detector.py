"""AstroSpectro — Détection de raies d’absorption (pics négatifs)

Ce module expose une petite classe utilitaire, `PeakDetector`, qui:
- détecte les minima locaux (raies en absorption) dans un spectre normalisé,
- associe ces minima à un petit jeu de raies connues (Balmer, Ca II, Mg_b, Na_D),
- retourne un mapping {nom_de_raie -> (lambda_detectée, prominence)}.

Conventions
-----------
- Longueurs d’onde en Angströms (Å).
- On détecte des **minima** en inversant le flux (absorption).
- La fenêtre de tolérance autour des longueurs d’onde théoriques est donnée par `window` (Å).

Entrées / Sorties
-----------------
Entrées:
    wavelength : array-like (N,)
    flux       : array-like (N,) — de préférence déjà normalisé

Sorties:
    - `detect_peaks(...)`      -> (indices_peaks: np.ndarray[int], properties: dict)
    - `match_known_lines(...)` -> dict[str, tuple[float, float] | None]
    - `analyze_spectrum(...)`  -> idem, pipeline complet (détecte + associe)

Exemple
-------
>>> pd = PeakDetector(prominence=0.85, window=28)
>>> matches = pd.analyze_spectrum(wl, flux_norm)
>>> matches["Hβ"]  # -> (lambda_detectee, prominence) ou None
"""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, Optional, Tuple, Sequence

import numpy as np
from scipy.signal import find_peaks


class PeakDetector:
    """Détecteur simple de raies d’absorption (pics inversés).

    Args:
        prominence: Seuil de "proéminence" passé à `scipy.signal.find_peaks`
            (après inversion du flux). Plus grand => moins de pics détectés.
        window: Tolérance (en Å) pour associer un pic détecté à une raie
            théorique connue (± `window` autour de λ_théorique).

    Attributes:
        target_lines: Dictionnaire {nom_de_raie -> λ_théorique (Å)}.
        prominence:  Seuil de proéminence pour la détection.
        window:      Tolérance d’association en Å.

    Notes:
        - On **n’inverse** pas les longueurs d’onde; on inverse uniquement le
          flux pour détecter des minima comme s’il s’agissait de maxima.
        - Les dictionnaires Python préservent l’ordre d’insertion; l’ordre des
          raies renvoyées suit donc celui de `target_lines`.
    """

    def __init__(self, prominence: float = 0.85, window: int = 28) -> None:
        self.prominence = float(prominence)
        self.window = int(window)

        # Vocabulaire minimal de raies (Å)
        self.target_lines: Dict[str, float] = {
            # Balmer (étoiles A/F chaudes)
            "Hα": 6563.0,
            "Hβ": 4861.0,
            # Calcium (généralistes)
            "CaII K": 3933.0,
            "CaII H": 3968.0,
            # Pour G/K plus froides
            "Mg_b": 5175.0,  # Triplet du magnésium (approx. centre)
            "Na_D": 5893.0,  # Doublet du sodium (approx. centre)
        }

    # --------------------------------------------------------------------- #
    # Détection
    # --------------------------------------------------------------------- #
    def detect_peaks(
        self, wavelength: Iterable[float], flux: Iterable[float]
    ) -> Tuple[np.ndarray, Mapping[str, np.ndarray]]:
        """Détecte les **minima** d’un spectre.

        Le flux est inversé (–flux) pour utiliser `find_peaks` comme détecteur
        de minima d’absorption.

        Args:
            wavelength: Longueurs d’onde (Å), 1D.
            flux: Flux (normalisé de préférence), 1D.

        Returns:
            indices: Indices (np.ndarray[int]) des pics détectés dans les arrays
                d’entrée.
            properties: Dictionnaire `scipy.signal.find_peaks` contenant, entre
                autres, `prominences` (float par pic).

        Remarques:
            - Les NaN/Inf sont masqués (retirés) avant détection.
            - `prominence` est directement passé à `find_peaks`.
        """
        wl = np.asarray(wavelength, dtype=float)
        fx = np.asarray(flux, dtype=float)

        # Masquage robuste (évite les surprises avec find_peaks)
        good = np.isfinite(wl) & np.isfinite(fx)
        wl = wl[good]
        fx = fx[good]

        inverted_flux = -fx
        peak_indices, properties = find_peaks(inverted_flux, prominence=self.prominence)
        return peak_indices, properties

    # --------------------------------------------------------------------- #
    # Association aux raies connues
    # --------------------------------------------------------------------- #
    def match_known_lines(
        self,
        peak_indices: np.ndarray,
        peak_wavelengths: np.ndarray,
        properties: Mapping[str, np.ndarray],
    ) -> Dict[str, Optional[Tuple[float, float]]]:
        """Associe les pics détectés aux raies `target_lines`.

        Pour chaque raie de référence, on retient le **candidat le plus
        proéminent** parmi les pics situés à ±`window` Å de λ_théorique.

        Args:
            peak_indices: Indices des pics (issus de `detect_peaks`).
            peak_wavelengths: `wavelength[peak_indices]` (Å).
            properties: Dictionnaire de propriétés renvoyé par `find_peaks`
                — on utilise notamment `properties["prominences"]`.

        Returns:
            dict {nom_de_raie -> (lambda_detectée, prominence) | None}
        """
        matched: Dict[str, Optional[Tuple[float, float]]] = {}
        prominences = np.asarray(properties.get("prominences", []), dtype=float)

        for name, target_wl in self.target_lines.items():
            # indices des pics dans la fenêtre ±window
            mask = np.abs(peak_wavelengths - target_wl) <= self.window
            candidates = np.where(mask)[0]

            if candidates.size > 0:
                # Choisit le candidat le plus "fort" (proéminence max)
                best_local = candidates[np.argmax(prominences[candidates])]
                best_wl = float(peak_wavelengths[best_local])
                best_prom = float(prominences[best_local])
                matched[name] = (best_wl, best_prom)
            else:
                matched[name] = None

        return matched

    # --------------------------------------------------------------------- #
    # Pipeline court
    # --------------------------------------------------------------------- #
    def analyze_spectrum(
        self, wavelength: Iterable[float], flux: Iterable[float]
    ) -> Dict[str, Optional[Tuple[float, float]]]:
        """Pipeline court: détection des minima + association aux raies.

        Args:
            wavelength: Longueurs d’onde (Å), 1D.
            flux: Flux (normalisé), 1D.

        Returns:
            dict {nom_de_raie -> (lambda_detectée, prominence) | None}
            (les clés sont les mêmes que `self.target_lines`).
        """
        idx, props = self.detect_peaks(wavelength, flux)
        if idx.size == 0:
            # Conserve l’ordre des clés de target_lines
            return {name: None for name in self.target_lines}

        wl = np.asarray(wavelength, dtype=float)
        peak_wl = wl[idx]
        return self.match_known_lines(idx, peak_wl, props)


__all__ = ["PeakDetector", "matches_to_dataframe", "matches_to_series"]


def matches_to_dataframe(matches: Mapping[str, Optional[Tuple[float, float]]]):
    """Convertit un dict de correspondances raies -> (λ, prom) en DataFrame.

    Schéma retourné (index = nom de raie):
        - line         : nom de la raie (index)
        - lambda_A     : longueur d'onde détectée (Å) ou NaN si absente
        - prominence   : proéminence du pic (float) ou NaN si absente
        - matched      : booléen, True si une correspondance existe

    Args:
        matches: dict { "Hβ": (4860.9, 0.93), "Na_D": None, ... }

    Returns:
        pd.DataFrame indexé par le nom de la raie.
    """
    try:
        import pandas as pd  # import paresseux pour ne pas imposer la dépendance
    except Exception as e:
        raise RuntimeError(
            "pandas est requis pour matches_to_dataframe(). "
            "Installez-le avec `pip install pandas`."
        ) from e

    rows = []
    for line, payload in matches.items():
        if payload is None:
            wl, prom, ok = np.nan, np.nan, False
        else:
            wl, prom = payload
            ok = True
        rows.append(
            {
                "line": line,
                "lambda_A": float(wl),
                "prominence": float(prom) if ok else np.nan,
                "matched": ok,
            }
        )

    df = pd.DataFrame(rows).set_index("line")
    return df


def matches_to_series(
    matches: Mapping[str, Optional[Tuple[float, float]]],
    order: Optional[Sequence[str]] = None,
    prefix: str = "match_",
):
    """Aplati les correspondances en **une seule ligne** (Series) aux colonnes stables.

    Colonnes générées (pour chaque raie dans `order` ou dans l'ordre du dict):
        - f"{prefix}{RAIE}_wl"
        - f"{prefix}{RAIE}_prom"

    Args:
        matches: dict { "Hβ": (4860.9, 0.93), "Na_D": None, ... }
        order: ordre explicite des raies (sinon l’ordre d’insertion du dict).
        prefix: préfixe des colonnes (par défaut 'match_').

    Returns:
        pandas.Series avec 2*len(order) colonnes. Les absences deviennent NaN.
    """
    try:
        import pandas as pd
    except Exception as e:
        raise RuntimeError(
            "pandas est requis pour matches_to_series(). "
            "Installez-le avec `pip install pandas`."
        ) from e

    names = list(order) if order is not None else list(matches.keys())

    data = {}
    for name in names:
        payload = matches.get(name)
        wl, prom = (np.nan, np.nan) if payload is None else payload
        data[f"{prefix}{name}_wl"] = float(wl) if np.isfinite(wl) else np.nan
        data[f"{prefix}{name}_prom"] = float(prom) if np.isfinite(prom) else np.nan

    return pd.Series(data)
