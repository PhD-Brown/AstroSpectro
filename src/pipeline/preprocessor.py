"""AstroSpectro --- Spectrum preprocessing (FITS to clean arrays).

This module centralises the low-level operations required before feature
extraction:

1. **Reading** a spectrum from a FITS HDUList:
   - flux: row 0 of the data array
   - inverse-variance (invvar): row 1
   - wavelength reconstruction: ``10**(COEFF0 + i * COEFF1)``  [Angstroms]

2. **Normalisation** of the flux (simple, robust) to stabilise downstream
   analysis.

3. **Sanitisation** of the inverse-variance via ``sanitize_invvar()`` to
   prevent warnings / NaN during the square-root conversion.  Optionally
   returns a ``StdDevUncertainty`` directly.

Conventions
-----------
- Wavelengths are in Angstroms (Å).
- Spectra are assumed to be rest-frame (absorption features have negative
  residuals after normalisation if applicable).
- Safety helpers are imported from ``src/utils.py``.

Examples
--------
>>> sp = SpectraPreprocessor()
>>> wl, flux, invvar = sp.load_spectrum(hdul)
>>> flux_n = sp.normalize_spectrum(flux)
>>> spec = sp.prepare(hdul, normalize=True, return_uncertainty=True)
>>> spec.wavelength.shape, spec.flux_norm.shape
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

# Safe invvar helpers (avoid warnings / NaN in sqrt)
from utils import sanitize_invvar, make_stddev_uncertainty_from_invvar


@dataclass
class ProcessedSpectrum:
    """Typed container for a preprocessed spectrum.

    Attributes
    ----------
    wavelength : np.ndarray
        Wavelength grid in Angstroms, shape ``(N,)``.
    flux : np.ndarray
        Raw flux array, shape ``(N,)``.
    invvar : np.ndarray
        Raw inverse-variance array, shape ``(N,)``.
    flux_norm : np.ndarray or None
        Normalised flux, shape ``(N,)``, if normalisation was requested.
    invvar_clean : np.ndarray or None
        Sanitised inverse-variance, shape ``(N,)``, after ``sanitize_invvar``.
    uncertainty : StdDevUncertainty or None
        Standard-deviation uncertainty, if requested.
    """

    wavelength: np.ndarray  # (N,) in Angstroms
    flux: np.ndarray  # (N,)
    invvar: np.ndarray  # (N,) raw inverse-variance
    flux_norm: Optional[np.ndarray] = None  # (N,) normalised flux (if requested)
    invvar_clean: Optional[np.ndarray] = None  # (N,) invvar after sanitisation
    uncertainty: Optional[object] = None  # StdDevUncertainty (if requested)


class SpectraPreprocessor:
    """Preprocessing tools: FITS reading, normalisation, inverse-variance sanitisation.

    This class provides a minimal, stateless API for loading a LAMOST-style FITS
    spectrum, normalising its flux, and optionally computing a robust uncertainty
    array from the inverse-variance extension.
    """

    # --------------------------------------------------------------------- #
    # FITS reading -> (wavelength, flux, invvar)
    # --------------------------------------------------------------------- #
    def load_spectrum(self, hdul) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract wavelength, flux, and inverse-variance from a FITS HDUList.

        Flux (row 0) and inverse-variance (row 1) are read from
        ``hdul[0].data``.  Wavelengths are reconstructed from the header
        keywords ``COEFF0`` (starting log10 wavelength) and ``COEFF1``
        (log10 step).

        Parameters
        ----------
        hdul : astropy.io.fits.HDUList
            An already-opened HDUList.

        Returns
        -------
        wavelength : np.ndarray
            Wavelength grid in Angstroms, shape ``(N,)``.
        flux : np.ndarray
            Flux array, shape ``(N,)``.
        invvar : np.ndarray
            Raw inverse-variance array, shape ``(N,)``.

        Raises
        ------
        ValueError
            If the data shape is invalid (fewer than 2 rows) or if
            ``COEFF0`` / ``COEFF1`` are missing from the header.
        """
        header = hdul[0].header
        data = hdul[0].data

        if data.ndim < 2 or data.shape[0] < 2:
            raise ValueError("Invalid FITS data array (fewer than 2 rows).")

        flux = data[0]
        invvar = data[1]

        if "COEFF0" in header and "COEFF1" in header:
            loglam_start = header["COEFF0"]
            loglam_step = header["COEFF1"]
            loglam = loglam_start + np.arange(len(flux)) * loglam_step
            wavelength = 10**loglam
        else:
            raise ValueError("Invalid FITS header: COEFF0 or COEFF1 missing.")

        return wavelength, flux, invvar

    # --------------------------------------------------------------------- #
    # Simple & robust normalisation
    # --------------------------------------------------------------------- #
    def normalize_spectrum(
        self, flux: np.ndarray, method: str = "median"
    ) -> np.ndarray:
        """
        Normalise a flux vector.

        Currently only **median** normalisation is supported, which is robust
        against emission/absorption lines and outliers.

        Parameters
        ----------
        flux : np.ndarray
            One-dimensional flux array.
        method : {'median'}, optional
            Normalisation strategy (default: ``'median'``).

        Returns
        -------
        np.ndarray
            Normalised flux with the same shape as *flux*.

        Raises
        ------
        ValueError
            If *method* is not ``'median'``.
        """
        if method != "median":
            raise ValueError(f"Unknown normalisation method: {method!r}")

        med = float(np.median(flux))
        return flux / med if med > 0 else flux

    # --------------------------------------------------------------------- #
    # Short pipeline: read + optional normalisation & uncertainties
    # --------------------------------------------------------------------- #
    def prepare(
        self,
        hdul,
        normalize: bool = True,
        return_uncertainty: bool = False,
        min_invvar: float = 1e-12,
    ) -> ProcessedSpectrum:
        """
        Preprocess a complete spectrum from a FITS HDUList.

        Steps
        -----
        1. Read wavelength, flux, and inverse-variance.
        2. Sanitise inverse-variance via ``sanitize_invvar``.
        3. Normalise flux (optional).
        4. Build ``StdDevUncertainty`` (optional) via
           ``make_stddev_uncertainty_from_invvar``.

        Parameters
        ----------
        hdul : astropy.io.fits.HDUList
            Opened FITS HDUList.
        normalize : bool, optional
            If True, compute and return ``flux_norm`` (default: True).
        return_uncertainty : bool, optional
            If True, compute and return a ``StdDevUncertainty`` from the
            cleaned inverse-variance (default: False).
        min_invvar : float, optional
            Minimum floor applied to inverse-variance during sanitisation
            (default: 1e-12).

        Returns
        -------
        ProcessedSpectrum
            Container with the relevant fields populated.
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
