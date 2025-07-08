import numpy as np
from astropy.io import fits

class SpectraPreprocessor:
    def __init__(self):
        pass

    def load_spectrum(self, hdul):
        """
        Extrait le spectre (wavelength, flux) depuis un objet HDUList déjà ouvert.
        Cette version est spécifiquement conçue pour le format de données DR5.
        """
        header = hdul[0].header
        data = hdul[0].data

        # --- Logique de lecture validée pour le format FITS que tu as ---
        
        # 1. Extraire le flux
        # Le flux est la première ligne de la matrice de données.
        if data.ndim < 1:
             raise ValueError("Le tableau de données est invalide (pas de dimensions).")
        flux = data[0]

        # 2. Calculer la longueur d'onde à partir du header
        # Le header utilise COEFF0 et COEFF1 pour une échelle log-linéaire.
        if 'COEFF0' in header and 'COEFF1' in header:
            loglam_start = header['COEFF0']
            loglam_step = header['COEFF1']
            
            # On crée un tableau de log(lambda)
            loglam = loglam_start + np.arange(len(flux)) * loglam_step
            
            # On le convertit en Angströms
            wavelength = 10**loglam
        else:
            raise ValueError("Header FITS invalide : mots-clés COEFF0 ou COEFF1 manquants.")
            
        return wavelength, flux

    def normalize_spectrum(self, flux):
        """Normalisation simple du flux par la médiane."""
        median_flux = np.median(flux)
        if median_flux > 0:
            return flux / median_flux
        return flux