import numpy as np
from scipy.signal import find_peaks

class PeakDetector:
    def __init__(self, prominence=1.4, window=30):
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
        # On retourne `peaks` (les indices) et non `peak_wavelengths`.
        peaks, properties = find_peaks(inverted_flux, prominence=self.prominence)
        return peaks, properties

    def match_known_lines(self, peak_wavelengths):
        matched_lines = {}
        for name, target_wl in self.target_lines.items():
            found = [pw for pw in peak_wavelengths if abs(pw - target_wl) <= self.window]
            if found:
                # On prend le pic le plus proche de la raie cible
                closest_peak = min(found, key=lambda x: abs(x - target_wl))
                matched_lines[name] = closest_peak
            else:
                matched_lines[name] = None
        return matched_lines

    def analyze_spectrum(self, wavelength, flux):
        """
        Pipeline complet : détection puis association aux raies connues.
        """
        # On passe le flux normal, l'inversion est gérée à l'intérieur de detect_peaks
        peak_indices, _ = self.detect_peaks(wavelength, flux) 
        peak_wavelengths = wavelength[peak_indices]
        matched_lines = self.match_known_lines(peak_wavelengths)
        return matched_lines
