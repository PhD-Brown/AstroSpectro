from pipeline.preprocessor import SpectraPreprocessor
from pipeline.peak_detector import PeakDetector
from src.visualizer import SpectraVisualizer

# === Exemple d'utilisation simple (à ajuster avec tes fichiers réels) ===

if __name__ == "__main__":
    data_dir = "data"
    

    
    # Pré-traitement d'un fichier (exemple bidon, car pas de vrai FITS encore)
    preprocessor = SpectraPreprocessor()
    # wavelength, flux = preprocessor.load_spectrum("data/star1.fits")  # À décommenter avec vrais fichiers
    
    # Pour tester : générons un faux spectre pour debug visuel
    import numpy as np
    wavelength = np.linspace(3800, 7500, 1000)
    flux = 1 - 0.2 * np.exp(-0.5 * ((wavelength - 6563) / 2)**2)  # simulons Hα
    
    flux = preprocessor.normalize_spectrum(flux)
    
    # Détection de raies
    detector = PeakDetector(prominence=0.05)
    peaks, properties = detector.detect_peaks(wavelength, flux)
    
    # Visualisation
    visualizer = SpectraVisualizer()
    visualizer.plot_spectrum(wavelength, flux, peaks)
