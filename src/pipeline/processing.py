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
    def __init__(self, raw_data_dir, master_catalog_df):
        self.raw_data_dir = raw_data_dir
        self.preprocessor = SpectraPreprocessor()
        self.peak_detector = PeakDetector(prominence=0.4, window=15)
        self.feature_engineer = FeatureEngineer()
        
        self.master_catalog = master_catalog_df
        if self.master_catalog is not None and not self.master_catalog.empty:
            if 'fits_name' in self.master_catalog.columns:
                # On crée la clé de jointure une seule fois pour la performance
                self.master_catalog['fits_name_only'] = self.master_catalog['fits_name'].str.replace(".gz", "", regex=False)
            else:
                print("AVERTISSEMENT: La colonne 'fits_name' est manquante dans le catalogue fourni.")

    def run(self, batch_paths):
        all_features_list = []
        for file_path in tqdm(batch_paths, desc="Traitement des spectres"):
            full_fits_path = os.path.join(self.raw_data_dir, file_path)
            try:
                # --- LOGIQUE DE TRAITEMENT COMPLÈTE ---
                # 1. Chargement et Pré-traitement
                with gzip.open(full_fits_path, 'rb') as f_gz:
                    with fits.open(f_gz, memmap=False) as hdul:
                        wavelength, flux, invvar = self.preprocessor.load_spectrum(hdul)
                flux_norm = self.preprocessor.normalize_spectrum(flux)

                # 2. Détection de Raies
                matched_lines = self.peak_detector.analyze_spectrum(wavelength, flux_norm)
                
                # 3. Feature Engineering (C'est la ligne qui manquait !)
                features_vector = self.feature_engineer.extract_features(matched_lines)
                # --- FIN DE LA LOGIQUE DE TRAITEMENT ---
                
                # On stocke les résultats
                record = {"file_path": file_path}
                for line_name, feature_val in zip(self.feature_engineer.lines, features_vector):
                    record[f"feature_{line_name.replace(' ', '')}"] = feature_val
                all_features_list.append(record)

            except Exception as e:
                print(f"\n    -> ERREUR lors du traitement de {file_path}: {e}")

        features_df = pd.DataFrame(all_features_list)
        
        # --- JOINTURE AVEC LE CATALOGUE MASTER ---
        if features_df.empty or self.master_catalog is None or self.master_catalog.empty:
            print(f"\nPipeline de traitement terminé. {len(features_df)} spectres traités.")
            if 'label' not in features_df.columns:
                 features_df['label'] = 'UNKNOWN'
            return features_df

        features_df['fits_name_only'] = features_df['file_path'].apply(lambda x: os.path.basename(x).replace('.gz', ''))

        merged_df = pd.merge(
            features_df,
            self.master_catalog,
            how='left',
            on='fits_name_only'
        )
        
        # Nettoyage final
        merged_df = merged_df.drop(columns=['fits_name_only'])

        print(f"\nPipeline de traitement terminé. {len(merged_df)} spectres traités et enrichis.")
        return merged_df