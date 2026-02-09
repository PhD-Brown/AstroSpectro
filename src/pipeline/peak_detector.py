"""AstroSpectro --- Absorption-line detection (negative peaks).

This module exposes ``PeakDetector``, a lightweight utility class that:

- detects local minima (absorption lines) in a normalised spectrum,
- associates those minima with a small set of known reference lines
  (Balmer, Ca II, Mg b, Na D),
- returns a mapping ``{line_name: (detected_wavelength, prominence)}``.

Conventions
-----------
- Wavelengths are in Angstroms (Å).
- Detection operates on the **inverted** flux (absorption features become
  positive peaks) so that ``scipy.signal.find_peaks`` can be applied directly.
- The tolerance window around each reference wavelength is controlled by the
  *window* parameter (in Å).

Inputs / Outputs
----------------
Inputs:
    wavelength : array-like, shape ``(N,)``
    flux       : array-like, shape ``(N,)`` --- preferably normalised

Outputs:
    - ``detect_peaks(...)``      -> ``(peak_indices, properties)``
    - ``match_known_lines(...)`` -> ``dict[str, tuple[float, float] | None]``
    - ``analyze_spectrum(...)``  -> same as above (full pipeline)

Examples
--------
>>> pd = PeakDetector(prominence=0.85, window=28)
>>> matches = pd.analyze_spectrum(wl, flux_norm)
>>> matches["Hbeta"]  # -> (detected_wavelength, prominence) or None
"""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, Optional, Tuple, Sequence

import numpy as np
from scipy.signal import find_peaks


class PeakDetector:
    """Simple absorption-line detector (inverted peaks).

    Parameters
    ----------
    prominence : float, optional
        Prominence threshold passed to ``scipy.signal.find_peaks`` on the
        inverted flux.  Higher values yield fewer detected peaks
        (default: 0.85).
    window : int, optional
        Tolerance in Angstroms for associating a detected peak with a
        reference line (``+/- window`` around the theoretical wavelength)
        (default: 28).

    Attributes
    ----------
    target_lines : dict[str, float]
        Mapping ``{line_name: theoretical_wavelength (Å)}``.
    prominence : float
        Prominence threshold for detection.
    window : int
        Association tolerance in Angstroms.

    Notes
    -----
    - Only the flux is inverted; wavelengths are left unchanged.
    - Python dicts preserve insertion order, so the returned mapping
      follows the order defined in ``target_lines``.
    """

    def __init__(self, prominence: float = 0.85, window: int = 28) -> None:
        self.prominence = float(prominence)
        self.window = int(window)

        # Minimal line vocabulary (Å)
        self.target_lines: Dict[str, float] = {
            # Balmer series (hot A/F stars)
            "Hα": 6563.0,
            "Hβ": 4861.0,
            # Calcium (general-purpose)
            "CaII K": 3933.0,
            "CaII H": 3968.0,
            # For cooler G/K stars
            "Mg_b": 5175.0,  # Magnesium triplet (approx. centre)
            "Na_D": 5893.0,  # Sodium doublet (approx. centre)
        }

    # --------------------------------------------------------------------- #
    # Detection
    # --------------------------------------------------------------------- #
    def detect_peaks(
        self, wavelength: Iterable[float], flux: Iterable[float]
    ) -> Tuple[np.ndarray, Mapping[str, np.ndarray]]:
        """Detect **minima** in a spectrum.

        The flux is inverted (``-flux``) so that ``find_peaks`` can be used
        as an absorption-minimum detector.

        Parameters
        ----------
        wavelength : array-like
            Wavelengths in Angstroms, 1-D.
        flux : array-like
            Flux (preferably normalised), 1-D.

        Returns
        -------
        indices : np.ndarray of int
            Indices of the detected peaks in the input arrays.
        properties : dict[str, np.ndarray]
            Dictionary returned by ``scipy.signal.find_peaks`` containing,
            among others, ``'prominences'`` (float per peak).

        Notes
        -----
        NaN / Inf values are masked (removed) before detection.
        ``prominence`` is passed directly to ``find_peaks``.
        """
        wl = np.asarray(wavelength, dtype=float)
        fx = np.asarray(flux, dtype=float)

        # Robust masking (avoids surprises with find_peaks)
        good = np.isfinite(wl) & np.isfinite(fx)
        wl = wl[good]
        fx = fx[good]

        inverted_flux = -fx
        peak_indices, properties = find_peaks(inverted_flux, prominence=self.prominence)
        return peak_indices, properties

    # --------------------------------------------------------------------- #
    # Association with known lines
    # --------------------------------------------------------------------- #
    def match_known_lines(
        self,
        peak_indices: np.ndarray,
        peak_wavelengths: np.ndarray,
        properties: Mapping[str, np.ndarray],
    ) -> Dict[str, Optional[Tuple[float, float]]]:
        """Associate detected peaks with the reference ``target_lines``.

        For each reference line, the **most prominent** candidate within
        ``+/- window`` Angstroms of the theoretical wavelength is selected.

        Parameters
        ----------
        peak_indices : np.ndarray
            Indices of the detected peaks (from ``detect_peaks``).
        peak_wavelengths : np.ndarray
            ``wavelength[peak_indices]`` in Angstroms.
        properties : Mapping[str, np.ndarray]
            Properties dict returned by ``find_peaks``; must contain
            ``'prominences'``.

        Returns
        -------
        dict[str, tuple[float, float] | None]
            ``{line_name: (detected_wavelength, prominence)}`` or ``None``
            for unmatched lines.
        """
        matched: Dict[str, Optional[Tuple[float, float]]] = {}
        prominences = np.asarray(properties.get("prominences", []), dtype=float)

        for name, target_wl in self.target_lines.items():
            # peak indices within the +/- window
            mask = np.abs(peak_wavelengths - target_wl) <= self.window
            candidates = np.where(mask)[0]

            if candidates.size > 0:
                # Select the strongest candidate (highest prominence)
                best_local = candidates[np.argmax(prominences[candidates])]
                best_wl = float(peak_wavelengths[best_local])
                best_prom = float(prominences[best_local])
                matched[name] = (best_wl, best_prom)
            else:
                matched[name] = None

        return matched

    # --------------------------------------------------------------------- #
    # Short pipeline
    # --------------------------------------------------------------------- #
    def analyze_spectrum(
        self, wavelength: Iterable[float], flux: Iterable[float]
    ) -> Dict[str, Optional[Tuple[float, float]]]:
        """Full pipeline: detect minima and associate with reference lines.

        Parameters
        ----------
        wavelength : array-like
            Wavelengths in Angstroms, 1-D.
        flux : array-like
            Normalised flux, 1-D.

        Returns
        -------
        dict[str, tuple[float, float] | None]
            Same keys as ``self.target_lines``.  Values are
            ``(detected_wavelength, prominence)`` or ``None``.
        """
        idx, props = self.detect_peaks(wavelength, flux)
        if idx.size == 0:
            # Preserve the key order of target_lines
            return {name: None for name in self.target_lines}

        wl = np.asarray(wavelength, dtype=float)
        peak_wl = wl[idx]
        return self.match_known_lines(idx, peak_wl, props)


__all__ = ["PeakDetector", "matches_to_dataframe", "matches_to_series"]


def matches_to_dataframe(matches: Mapping[str, Optional[Tuple[float, float]]]):
    """Convert a line-match dict to a DataFrame.

    The returned DataFrame is indexed by line name and contains:

    - ``lambda_A``   : detected wavelength (Å) or NaN if unmatched
    - ``prominence`` : peak prominence (float) or NaN if unmatched
    - ``matched``    : boolean flag, True when a match exists

    Parameters
    ----------
    matches : Mapping[str, tuple[float, float] | None]
        e.g. ``{"Hbeta": (4860.9, 0.93), "Na_D": None, ...}``

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by line name.
    """
    try:
        import pandas as pd  # lazy import to avoid hard dependency
    except Exception as e:
        raise RuntimeError(
            "pandas is required for matches_to_dataframe(). "
            "Install it with `pip install pandas`."
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
    """Flatten line matches into a single row (Series) with stable column names.

    For each line in *order* (or dict insertion order), two columns are
    produced: ``{prefix}{LINE}_wl`` and ``{prefix}{LINE}_prom``.

    Parameters
    ----------
    matches : Mapping[str, tuple[float, float] | None]
        e.g. ``{"Hbeta": (4860.9, 0.93), "Na_D": None, ...}``
    order : Sequence[str] or None, optional
        Explicit line ordering.  If ``None``, dict insertion order is used.
    prefix : str, optional
        Column-name prefix (default: ``'match_'``).

    Returns
    -------
    pd.Series
        Series with ``2 * len(order)`` entries.  Missing lines become NaN.
    """
    try:
        import pandas as pd
    except Exception as e:
        raise RuntimeError(
            "pandas is required for matches_to_series(). "
            "Install it with `pip install pandas`."
        ) from e

    names = list(order) if order is not None else list(matches.keys())

    data = {}
    for name in names:
        payload = matches.get(name)
        wl, prom = (np.nan, np.nan) if payload is None else payload
        data[f"{prefix}{name}_wl"] = float(wl) if np.isfinite(wl) else np.nan
        data[f"{prefix}{name}_prom"] = float(prom) if np.isfinite(prom) else np.nan

    return pd.Series(data)
