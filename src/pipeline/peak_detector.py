import numpy as np
from scipy.signal import find_peaks

class PeakDetector:
    def __init__(self, prominence=0.85, window=28):
        self.prominence = prominence
        self.window = window
        self.target_lines = {
            "Hα": 6563,
            "Hβ": 4861,
            "CaII K": 3933,
            "CaII H": 3968,
        }

    def detect_peaks(self, wavelength, flux):
        """
        Détecte tous les minima locaux (absorption) dans le spectre.
        Retourne les INDICES des pics, et leurs propriétés.
        """
        inverted_flux = -flux
        # find_peaks retourne les indices et un dictionnaire de propriétés
        peak_indices, properties = find_peaks(inverted_flux, prominence=self.prominence)
        return peak_indices, properties
    
    def match_known_lines(self, peak_indices, peak_wavelengths, properties):
        """
        Associe les pics détectés aux raies connues et retourne un dictionnaire
        contenant la longueur d'onde et la prominence du meilleur match.
        """
        matched_lines = {}
        # 'prominences' est une clé dans le dictionnaire des propriétés
        peak_prominences = properties["prominences"]

        for name, target_wl in self.target_lines.items():
            # Cherche les pics dans la fenêtre de tolérance
            candidate_indices = [
                i for i, wl in enumerate(peak_wavelengths) 
                if abs(wl - target_wl) <= self.window
            ]
            
            if candidate_indices:
                # S'il y a plusieurs candidats, on prend le plus proéminent (le plus fort)
                best_candidate_idx = -1
                max_prominence = -1
                for idx in candidate_indices:
                    if peak_prominences[idx] > max_prominence:
                        max_prominence = peak_prominences[idx]
                        best_candidate_idx = idx
                
                # On stocke la longueur d'onde et la prominence du meilleur pic trouvé
                best_peak_wl = peak_wavelengths[best_candidate_idx]
                best_peak_prominence = peak_prominences[best_candidate_idx]
                matched_lines[name] = (best_peak_wl, best_peak_prominence)
            else:
                matched_lines[name] = None
                
        return matched_lines
    
    def analyze_spectrum(self, wavelength, flux):
        """Pipeline complet de détection et d'association."""
        peak_indices, properties = self.detect_peaks(wavelength, flux)
        if len(peak_indices) == 0:
            return {name: None for name in self.target_lines} # Retourne un dict vide si aucun pic
            
        peak_wavelengths = wavelength[peak_indices]
        matched_lines = self.match_known_lines(peak_indices, peak_wavelengths, properties)
        return matched_lines
