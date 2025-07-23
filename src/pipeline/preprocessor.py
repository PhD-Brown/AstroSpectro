import numpy as np
from astropy.io import fits

class SpectraPreprocessor:
    def __init__(self):
        pass

    def load_spectrum(self, hdul):
        """
        Extrait le spectre (wavelength, flux) et l'incertitude depuis un objet HDUList.
        """
        header = hdul[0].header
        data = hdul[0].data

        # 1. Extraire le flux (ligne 0) et l'inverse de la variance (ligne 1)
        if data.ndim < 2 or data.shape[0] < 2:
             raise ValueError("Le tableau de données FITS est invalide (moins de 2 lignes).")
        flux = data[0]
        invvar = data[1]

        # 2. Calculer la longueur d'onde à partir du header (COEFF0, COEFF1)
        if 'COEFF0' in header and 'COEFF1' in header:
            loglam_start = header['COEFF0']
            loglam_step = header['COEFF1']
            loglam = loglam_start + np.arange(len(flux)) * loglam_step
            wavelength = 10**loglam
        else:
            raise ValueError("Header FITS invalide : COEFF0 ou COEFF1 manquants.")
            
        return wavelength, flux, invvar

    def normalize_spectrum(self, flux):
        """Normalisation simple du flux par la médiane."""
        median_flux = np.median(flux)
        if median_flux > 0:
            return flux / median_flux
        return flux