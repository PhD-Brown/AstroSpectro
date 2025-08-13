"""
AstroSpectro — Pipeline de traitement des spectres (prétraîtement + features)
==============================================================================

Ce module applique, pour chaque spectre FITS, une chaîne de traitement complète :
1) Prétraitement (chargement, normalisation du flux, sécurisation de l’invvar)
2) Détection & association des raies d’absorption (PeakDetector)
3) Extraction de descripteurs spectroscopiques (FeatureEngineer : FWHM, EW, ratios, indices…)
4) Construction de features “couleur/photométrie” (si dispo dans le catalogue)
5) Assemblage final en ligne de DataFrame, puis agrégation en un tableau features

Conventions
-----------
- Longueurs d’onde en Angströms (Å).
- `invvar` (inverse-variance) est **nettoyée/tolérante** en amont pour éviter NaN/Inf lors des `sqrt`
  et produire une incertitude robuste (voir `utils.sanitize_invvar` & `make_stddev_uncertainty_from_invvar`).
- Les colonnes `match_*` (ex: `match_Hβ_wl`, `match_Hβ_prom`) proviennent du matching de raies
  (détection + association); elles **complètent** les colonnes `feature_*` renvoyées par la
  Feature Engineering (prominence/FWHM/EW/indices/ratios).
- Les vecteurs de features `feature_*` sont renvoyés **dans un ordre stable** (liste `feature_names`).

Entrées / Sorties attendues
---------------------------
Entrées :
- `file_path`       : chemin du fichier FITS à traiter (str)
- `catalog_row`     : (optionnel) ligne de catalogue pour enrichissement photométrique / Gaia
- `config`          : (optionnel) dict des paramètres (fenêtre de détection, seuils, etc.)

Sorties :
- `pandas.Series`   : une ligne consolidée contenant :
    * colonnes `match_*`  (longueur d’onde détectée + prominence par raie connue)
    * colonnes `feature_*` (FWHM, EW, indices, ratios… alignés à `FeatureEngineer.feature_names`)
    * colonnes photométriques/GAIA si disponibles (ex: `bp_rp`, `ag_gspphot`, …)
- un ensemble de ces Series est agrégé en `pandas.DataFrame`.

API publique (principales méthodes)
-----------------------------------
- `ProcessingPipeline.process_one(file_path, catalog_row=None) -> pd.Series`
    Traite un spectre unique et renvoie une ligne prête à être concaténée.
- `ProcessingPipeline.process_many(file_paths, catalog_df=None) -> pd.DataFrame`
    Boucle sur une liste de fichiers et retourne le tableau complet des features.
- `matches_to_series(matched_lines: dict) -> pd.Series`
    Helper qui convertit un mapping `{nom_de_raie: (lambda_detectée, prominence)}` en Series
    avec des noms de colonnes stables (`match_<RAIE>_wl`, `match_<RAIE>_prom`).

Remarques d’implémentation
--------------------------
- Le matching de raies est **tolérant** : les raies absentes ne cassent pas la chaîne,
  et les colonnes correspondantes sont remplies avec des NaN (ou des 0 selon le cas).
- La normalisation du flux et la création d’incertitudes utilisent les utilitaires de `utils.py`
  afin d’éviter les warnings de type `invalid value encountered in sqrt`.
- Les features spectroscopiques sont calculées via `FeatureEngineer` qui garantit l’ordre stable
  des colonnes (liste `feature_names` utilisée pour aligner X/y en apprentissage).

Exemple minimal
---------------
>>> from pipeline.processing import ProcessingPipeline
>>> pp = ProcessingPipeline(paths)                        # chemins/répertoires utiles
>>> row = pp.process_one("data/raw/spec-XXXXX.fits.gz")   # ligne de features pour 1 spectre
>>> df  = pp.process_many(["specA.fits.gz", "specB.fits.gz"])  # batch -> DataFrame

"""

from __future__ import annotations

from typing import Dict, List

import os
import gzip
import numpy as np
import pandas as pd
from astropy.io import fits
from tqdm import tqdm

from .preprocessor import SpectraPreprocessor
from .peak_detector import PeakDetector, matches_to_series
from .feature_engineering import FeatureEngineer
from utils import sanitize_invvar


class ProcessingPipeline:
    """Pipeline de traitement “par spectre”.

    Args:
        raw_data_dir: Dossier racine contenant les fichiers FITS (gz).
        master_catalog_df: Catalogue courant (métadonnées par spectre).

    Attributes:
        preprocessor: Opérations de chargement/normalisation par spectre.
        peak_detector: Détection + association des raies.
        feature_engineer: Calcul des descripteurs spectroscopiques.
        master_catalog: Catalogue (potentiellement enrichi Gaia).
    """

    def __init__(self, raw_data_dir: str, master_catalog_df: pd.DataFrame) -> None:
        self.raw_data_dir = raw_data_dir
        self.preprocessor = SpectraPreprocessor()
        # Seuils par défaut — ajuste si besoin (plus haut = moins de faux positifs)
        self.peak_detector = PeakDetector(prominence=0.15, window=8)
        self.feature_engineer = FeatureEngineer()

        self.master_catalog = master_catalog_df
        if self.master_catalog is not None and not self.master_catalog.empty:
            if "fits_name" in self.master_catalog.columns:
                # Prépare une clé de jointure simple (sans .gz)
                self.master_catalog = self.master_catalog.copy()
                self.master_catalog["fits_name_only"] = self.master_catalog[
                    "fits_name"
                ].str.replace(".gz", "", regex=False)
            else:
                print(
                    "AVERTISSEMENT: La colonne 'fits_name' est manquante dans le catalogue fourni."
                )

    def run(self, batch_paths: List[str]) -> pd.DataFrame:
        """Traite un lot de spectres et retourne le DataFrame de features.

        Étapes (par fichier):
          - ouverture FITS (.gz),
          - `load_spectrum` (+ sécurisation `invvar`),
          - normalisation du flux,
          - détection/association de raies,
          - extraction des features “ligne-centrées”,
          - ajout des colonnes `match_*` (wl/prom) stables,
          - feature couleur (bande bleue/rouge),
          - jointure finale au catalogue maître + indices photométriques.

        Args:
            batch_paths: chemins **relatifs** des FITS à traiter (par ex. “B6001/spec-…gz”).

        Returns:
            DataFrame contenant les features calculées (et jointes au catalogue si dispo).
        """
        all_features_list: List[Dict[str, float]] = []

        print(
            f"\n--- Démarrage du pipeline de traitement pour {len(batch_paths)} spectres ---"
        )
        for file_path in tqdm(batch_paths, desc="Traitement des spectres"):
            full_fits_path = os.path.join(self.raw_data_dir, file_path)
            try:
                # 1) Chargement + prétraitement
                with gzip.open(full_fits_path, "rb") as f_gz:
                    with fits.open(f_gz, memmap=False) as hdul:
                        wavelength, flux, invvar = self.preprocessor.load_spectrum(hdul)
                        # Robustesse numérique : évite NaN/inf/<=0 avant sqrt()
                        invvar = sanitize_invvar(invvar)

                flux_norm = self.preprocessor.normalize_spectrum(flux)

                # 2) Détection/association de raies
                matched_lines = self.peak_detector.analyze_spectrum(
                    wavelength, flux_norm
                )

                # 3) Feature engineering (prominence/FWHM/EW/indices)
                features_vector = self.feature_engineer.extract_features(
                    matched_lines, wavelength, flux_norm, invvar
                )

                # 3.b) Colonnes supplémentaires issues du matching (stables & interprétables)
                order = self.feature_engineer.base_lines  # ordre canonique des raies
                match_series = matches_to_series(
                    matched_lines, order=order, prefix="match_"
                )
                # Harmonise les noms (pas d'espaces) pour rester cohérent
                match_series = match_series.rename(lambda c: c.replace(" ", ""))

                # 4) Feature couleur “Blue/Red” (simple proxy de pente)
                flux_blue = float(
                    np.nanmean(flux_norm[(wavelength > 4000) & (wavelength < 4500)])
                )
                flux_red = float(
                    np.nanmean(flux_norm[(wavelength > 6500) & (wavelength < 7000)])
                )
                color_index = flux_blue / (flux_red + 1e-6)  # évite /0

                # 5) Assemblage de la ligne
                record: Dict[str, float] = {"file_path": file_path}

                # features FE (dans l’ordre canonique)
                for feature_name, feature_val in zip(
                    self.feature_engineer.feature_names, features_vector
                ):
                    record[feature_name] = float(feature_val)

                # feature couleur locale
                record["feature_color_index_BlueRed"] = float(color_index)

                # colonnes issues du matching (wl/prom par raie)
                record.update(
                    {
                        k: float(v) if pd.notna(v) else np.nan
                        for k, v in match_series.to_dict().items()
                    }
                )

                all_features_list.append(record)

            except Exception as e:
                print(f"\n    -> ERREUR lors du traitement de {file_path}: {e}")

        features_df = pd.DataFrame(all_features_list)

        # --- Jointure avec le catalogue maître (si présent) ---
        if (
            features_df.empty
            or self.master_catalog is None
            or self.master_catalog.empty
        ):
            print(
                f"\nPipeline de traitement terminé. {len(features_df)} spectres traités."
            )
            if "label" not in features_df.columns:
                # Valeur par défaut — le classifieur filtrera les labels invalides
                features_df["label"] = "UNKNOWN"
            return features_df

        # Clé de jointure côté features
        features_df["fits_name_only"] = features_df["file_path"].apply(
            lambda x: os.path.basename(x).replace(".gz", "")
        )

        merged_df = pd.merge(
            features_df, self.master_catalog, how="left", on="fits_name_only"
        )

        # --- Indices de couleur photométriques (g-r, r-i) ---
        print("  > Création des features de couleur photométrique...")

        # Convertit en numérique, en neutralisant les placeholders (ex. 99.0)
        mag_cols = ["magnitude_g", "magnitude_r", "magnitude_i"]
        for col in mag_cols:
            if col in merged_df.columns:
                merged_df[col] = pd.to_numeric(merged_df[col], errors="coerce").replace(
                    99.0, np.nan
                )

        # Couleurs (si dispo)
        if {"magnitude_g", "magnitude_r"}.issubset(merged_df.columns):
            merged_df["feature_color_gr"] = (
                merged_df["magnitude_g"] - merged_df["magnitude_r"]
            )
        if {"magnitude_r", "magnitude_i"}.issubset(merged_df.columns):
            merged_df["feature_color_ri"] = (
                merged_df["magnitude_r"] - merged_df["magnitude_i"]
            )

        # Laisse l’imputeur du classifieur gérer les NaN, mais on peut
        # remplir ces deux colonnes à 0 si on veut rester conservateur:
        merged_df.fillna({"feature_color_gr": 0, "feature_color_ri": 0}, inplace=True)

        # Nettoyage final
        merged_df = merged_df.drop(columns=["fits_name_only"], errors="ignore")

        print(
            f"\nPipeline de traitement terminé. {len(merged_df)} spectres traités et enrichis."
        )
        return merged_df
