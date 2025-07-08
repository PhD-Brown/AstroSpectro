import os
import gzip
import numpy as np
import pandas as pd
from astropy.io import fits
from tqdm import tqdm

from .preprocessor import SpectraPreprocessor
from .peak_detector import PeakDetector
from .feature_engineering import FeatureEngineer

class ProcessingPipeline:
    def __init__(self, raw_data_dir, catalog_dir):
        self.raw_data_dir = raw_data_dir
        self.catalog_dir = catalog_dir
        self.preprocessor = SpectraPreprocessor()
        self.peak_detector = PeakDetector(prominence=0.85, window=27) 
        self.feature_engineer = FeatureEngineer()
        self._load_master_catalog()

    def _load_master_catalog(self):
        catalog_path = os.path.join(self.catalog_dir, "master_catalog_temp.csv")
        try:
            self.master_catalog = pd.read_csv(catalog_path, sep="|", on_bad_lines="skip")

            # On utilise directement les noms réels depuis la colonne fits_name
            self.master_catalog['fits_name_only'] = self.master_catalog['fits_name'].str.replace(".gz", "", regex=False)
            
            print("  > Catalogue master DR5 chargé pour la récupération des labels.")
        except FileNotFoundError:
            print(f"  > AVERTISSEMENT : Catalogue master non trouvé à '{catalog_path}'. Les labels ne seront pas extraits.")
            self.master_catalog = None

    def _get_label_for_file(self, fits_filename_only):
        if self.master_catalog is None: return "UNKNOWN"
        
        label_row = self.master_catalog[self.master_catalog['fits_name_only'] == fits_filename_only]
        if not label_row.empty:
            # Pour le DR5, la classe est dans 'subclass'
            label = str(label_row.iloc[0]['subclass'])
            return label[0] if label else "UNKNOWN"
        return "NOT_FOUND_IN_CATALOG"

    def run(self, batch_paths):
        all_features_list = []

        print(f"\n--- Démarrage du pipeline de traitement pour {len(batch_paths)} spectres ---")
        for file_path in tqdm(batch_paths, desc="Traitement des spectres"):
            full_fits_path = os.path.join(self.raw_data_dir, file_path)
            
            try:
                with gzip.open(full_fits_path, 'rb') as f_gz:
                    with fits.open(f_gz, memmap=False) as hdul: # type: ignore
                        wavelength, flux = self.preprocessor.load_spectrum(hdul)
                
                flux_norm = self.preprocessor.normalize_spectrum(flux)
                matched_lines = self.peak_detector.analyze_spectrum(wavelength, flux_norm)
                features_vector = self.feature_engineer.extract_features(matched_lines)
                
                fits_filename_only = os.path.basename(file_path).replace(".gz", "", 1)
                label = self._get_label_for_file(fits_filename_only)

                record = {"fits_name": fits_filename_only, "file_path": file_path, "label": label}
                for line_name, feature_val in zip(self.feature_engineer.lines, features_vector):
                    record[f"feature_{line_name.replace(' ', '')}"] = feature_val
                all_features_list.append(record)

            except Exception as e:
                print(f"\n    -> ERREUR lors du traitement de {file_path}: {e}")

        features_df = pd.DataFrame(all_features_list)
        # Merge complet : ajouter toutes les colonnes du master_catalog au features_df
        merged_df = features_df.merge(
            self.master_catalog,   # on prend toutes les colonnes
            how='left',
            on='fits_name'
        )
        print(f"\nPipeline de traitement terminé. {len(features_df)} spectres traités avec succès.")
        return features_df