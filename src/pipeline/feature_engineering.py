import numpy as np
from scipy.stats import skew
from scipy.signal import savgol_filter 
from specutils import Spectrum
from specutils.analysis import gaussian_fwhm
from astropy import units as u

class FeatureEngineer:
    def __init__(self):
        # 1. Les raies de base que nous allons analyser
        self.base_lines = ["Hα", "Hβ", "CaII K", "CaII H", "Mg_b", "Na_D"]
        
        # 2. Les métriques physiques que nous allons extraire pour chaque raie
        self.line_metrics = ["prominence", "fwhm"]
        
        # 3. On génère dynamiquement la liste complète des noms de features
        self.feature_names = []
        # Ajouter les noms pour les métriques de raies (ex: feature_Hα_prominence)
        for line in self.base_lines:
            line_name_safe = line.replace(' ', '')
            for metric in self.line_metrics:
                self.feature_names.append(f"feature_{line_name_safe}_{metric}")
                
                
        # 4. Les ratios que nous allons calculer, basés sur la PROMINENCE
        self.ratio_definitions = {
            "ratio_prom_CaK_Hbeta": ("feature_CaIIK_prominence", "feature_Hβ_prominence"),
            "ratio_prom_Halpha_Hbeta": ("feature_Hα_prominence", "feature_Hβ_prominence"),
            "ratio_prom_Mgb_Hbeta": ("feature_Mg_b_prominence", "feature_Hβ_prominence")
        }
        
        # Ajouter les noms pour les ratios
        self.feature_names += [f"feature_{name}" for name in self.ratio_definitions.keys()]

    def extract_features(self, matched_lines, wavelength, flux_norm):
        """
        Extrait un ensemble riche de features pour chaque raie détectée :
        - Prominence (force)
        - FWHM (largeur)
        - Skewness (asymétrie)
        Puis calcule des ratios basés sur les prominences.
        """
        all_line_features = {}

        # --- Étape 1 : Calculer les 3 métriques pour chaque raie de base ---
        for line in self.base_lines:
            match_data = matched_lines.get(line)
            line_name_safe = line.replace(' ', '')
            
            # Initialiser les features à 0.0 pour ce cycle
            prominence, fwhm_val = 0.0, 0.0

            if match_data:
                wl_detected, prom = match_data
                prominence = prom

                try:
                    window_half_width = 20 # Angströms de chaque côté
                    region_mask = (wavelength > wl_detected - window_half_width) & (wavelength < wl_detected + window_half_width)
                        
                    # On s'assure qu'il y a bien un pic à analyser
                    if np.any(region_mask) and len(wavelength[region_mask]) > 5: # Sav-Gol a besoin d'assez de points
                        flux_window = flux_norm[region_mask]
                        
                        # LISSAGE SAVITZKY-GOLAY ---
                        # On applique un filtre polynomial simple pour réduire le bruit à haute fréquence.
                        # window_length=5, polyorder=2 sont des valeurs de départ robustes.
                        flux_window_smoothed = savgol_filter(flux_window, window_length=5, polyorder=2)
                        emission_like_flux = 1.0 - flux_window_smoothed
                        
                        # On ne calcule que si le pic inversé est significatif
                        if np.max(emission_like_flux) > 0:
                            spectrum_obj = Spectrum(spectral_axis=wavelength[region_mask]*u.AA, flux=emission_like_flux*u.adu)
                            fwhm_result = gaussian_fwhm(spectrum_obj) 
                            fwhm_val = fwhm_result.to_value(u.AA)
                except Exception:
                    pass
            
            all_line_features[f"feature_{line_name_safe}_prominence"] = prominence
            all_line_features[f"feature_{line_name_safe}_fwhm"] = fwhm_val

        # --- Étape 2 : Calculer les 3 ratios basés sur les prominences extraites ---
        epsilon = 1e-6
        for name, (num_key, den_key) in self.ratio_definitions.items():
            num_val = all_line_features.get(num_key, 0.0)
            den_val = all_line_features.get(den_key, 0.0)
            all_line_features[f"feature_{name}"] = num_val / (den_val + epsilon)

        # --- Étape 3 : Retourner le vecteur de features final dans l'ordre défini par self.feature_names ---
        return [all_line_features.get(name, 0.0) for name in self.feature_names]

    def batch_features(self, matched_lines_list):
        """
        Permet de traiter une liste de spectres en batch.
        """
        return np.array([self.extract_features(ml) for ml in matched_lines_list])
