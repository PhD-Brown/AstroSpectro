"""
AstroSpectro — dimred.data_loader
==================================

Chargement et préparation des données pour la réduction de dimension.

Deux modes d'entrée sont supportés :
  1. **features**  : vecteur de descripteurs spectroscopiques engineerés
                     (sortie du pipeline FeatureEngineer, CSV).
  2. **spectra**   : flux bruts interpolés sur une grille wavelength commune
                     (matrice N × P, un pixel par colonne).

Dans les deux cas, la sortie est un triplet (X, y, meta) :
  - X    : np.ndarray (N, D) — matrice d'entrée pour PCA/UMAP
  - y    : np.ndarray (N,)   — étiquettes de classe ('STAR', 'GALAXY', ...)
  - meta : pd.DataFrame (N, .) — paramètres physiques Gaia + SNR, etc.

Paramètres physiques disponibles (si cross-match Gaia présent)
--------------------------------------------------------------
  teff_gspphot, logg_gspphot, mh_gspphot, bp_rp, phot_g_mean_mag,
  distance_gspphot, pmra, pmdec, ruwe, ag_gspphot, ebpminrp_gspphot

Filtres appliqués par défaut
----------------------------
  - SNR minimum configurable (snr_min, défaut 10 sur bande r)
  - Suppression des lignes avec NaN dans les features
  - Optionnel : sous-échantillonnage équilibré par classe (class_balance)

Exemple
-------
>>> loader = DimRedDataLoader(
...     features_path="data/reports/features_20260213.csv",
...     catalog_path="data/catalog/master_catalog_gaia.csv",
... )
>>> X, y, meta = loader.load(mode="features", snr_min=10)
>>> print(X.shape, y.shape)
(42831, 71) (42831,)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


# Colonnes Gaia à conserver dans meta
_GAIA_META_COLS = [
    "teff_gspphot",
    "logg_gspphot",
    "mh_gspphot",
    "bp_rp",
    "bp_g",
    "g_rp",
    "phot_g_mean_mag",
    "distance_gspphot",
    "pmra",
    "pmdec",
    "parallax",
    "ruwe",
    "ag_gspphot",
    "ebpminrp_gspphot",
    "phot_variable_flag",
]

# Colonnes SNR LAMOST
_SNR_COLS = ["snr_u", "snr_g", "snr_r", "snr_i", "snr_z"]

# Colonnes instrumentales et métadonnées observationnelles à exclure
# Synchronisé avec classifier.py (SpectralClassifier.cols_to_exclude)
# Raison : ces colonnes ne décrivent pas la physique stellaire
_INSTRUMENTAL_COLS = [
    # Calibration spectrale (FITS headers LAMOST)
    "crval1",
    "coeff0",
    "coeff1",
    "cd1_1",
    "crpix1",
    "dc_flag",
    "wfit_type",
    # Coordonnées spatiales (3 variantes RA + Dec)
    "ra",
    "dec",
    "ra_obs",
    "dec_obs",
    "spra",
    "spdec",
    # Métadonnées d'observation / site
    "longitude_site",
    "latitude_site",
    "fiber_id",
    "seeing",
    "redshift",
    "redshift_error",
    # Qualité pipeline LAMOST (chi2, flat, PCA sky)
    "skychi2",
    "schi2min",
    "schi2max",
    "offset",
    "offset_v",
    "fibermas",
    "scamean",
    "spid",
    "slit_mod",
    "fstar",
    "nskies",
    "nstd",
    "sflatten",
    "pcaskysb",
    # Champs FITS divers
    "focus_mm",
    "x_value_mm",
    "y_value_mm",
    "objname",
    "tcomment",
    "tsource",
    "tfrom",
    "obs_type",
    "obscomm",
    # Identifiants Gaia (pas physique)
    "match_dist_arcsec",
    "parallax_error",
    # Flags de qualité Gaia (dérivés de ruwe, non spectroscopiques)
    "is_good_ruwe",
    # Autres métadonnées
    "pipeline_version",
    "processing_notes",
    "download_url",
    "spectrum_hash",
    "flux_unit",
    "wavelength_unit",
    "radial_velocity_corr",
]

# Préfixes non-spectroscopiques pour le mode spectro_only
# (exclut photométrie Gaia, cinématique, paramètres stellaires GSP)
# Synchronisé avec classifier.py spectro_only logic
_NON_SPECTRO_PREFIXES = (
    "parallax",
    "pmra",
    "pmdec",
    "ruwe",
    "phot_",
    "bp_rp",
    "bp_g",
    "g_rp",
    "radial_velocity",
    "rv_",
    "teff",
    "logg",
    "fe_h",
    "alpha_fe",
    "dist_",
    "distance_",
    "ag_gspphot",
    "ebpminrp",
    "astrometric_",
)


class DimRedDataLoader:
    """
    Prépare les données (features ou spectres bruts) pour la réduction de
    dimension (PCA, UMAP, t-SNE, autoencodeur).

    Parameters
    ----------
    features_path : str | Path
        Chemin vers le CSV de features engineerées (sortie pipeline AstroSpectro).
    catalog_path : str | Path
        Chemin vers master_catalog_gaia.csv (cross-match Gaia DR3).
    merge_key : str
        Clé de jointure entre features et catalog (défaut: 'obsid').
    random_state : int
        Graine aléatoire pour reproductibilité.
    """

    def __init__(
        self,
        features_path: str | Path,
        catalog_path: str | Path,
        merge_key: str = "obsid",
        random_state: int = 42,
    ) -> None:
        # Chemins d'entrée normalisés en Path pour une API homogène.
        self.features_path = Path(features_path)
        self.catalog_path = Path(catalog_path)
        self.merge_key = merge_key
        self.random_state = random_state

        # Attributs remplis après load()
        self.feature_names_: Optional[list[str]] = None
        self.scaler_: Optional[StandardScaler] = None
        self._df_merged: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # API principale
    # ------------------------------------------------------------------

    def load(
        self,
        mode: Literal["features", "spectra"] = "features",
        snr_min: float = 10.0,
        snr_band: str = "snr_r",
        classes: Optional[list[str]] = None,
        scale: bool = True,
        class_balance: bool = False,
        n_per_class: int = 5000,
        drop_nan_threshold: float = 0.10,
        spectro_only: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """
        Charge et prépare les données.

        Parameters
        ----------
        mode : 'features' | 'spectra'
            'features' : charge le CSV de features engineerées.
            'spectra'  : non implémenté ici (voir SpectralMatrixLoader).
        snr_min : float
            SNR minimal sur la bande `snr_band`. Les spectres sous ce seuil
            sont exclus.
        snr_band : str
            Bande SNR utilisée pour le filtrage ('snr_r' recommandé).
        classes : list[str] | None
            Si fourni, filtre uniquement ces classes ('STAR', 'GALAXY', 'QSO').
            None = toutes les classes.
        scale : bool
            Si True, applique StandardScaler sur X (moyenne=0, variance=1).
            Obligatoire pour PCA/UMAP bien conditionnés.
        class_balance : bool
            Si True, sous-échantillonne au nombre `n_per_class` par classe.
        n_per_class : int
            Taille max par classe si `class_balance=True`.
        drop_nan_threshold : float
            Fraction maximale de NaN par colonne feature avant suppression
            de la colonne (défaut 10%).
        spectro_only : bool, default False
            Si True, conserve uniquement les features spectroscopiques pures
            (dérivées des flux du spectre). Exclut les colonnes SNR, Gaia
            photométriques, coordonnées spatiales, métadonnées d'observation
            et tous les artefacts instrumentaux LAMOST.
            Équivalent au mode ``spectro_only=True`` de SpectralClassifier —
            garantit que PCA/UMAP/AE travaillent sur les mêmes features
            que le pipeline XGBoost supervisé.

        Returns
        -------
        X : np.ndarray (N, D)
            Matrice d'entrée (standardisée si scale=True).
        y : np.ndarray (N,)
            Étiquettes de classe (str).
        meta : pd.DataFrame (N, .)
            Paramètres physiques Gaia + SNR pour coloration des embeddings.
        """
        # Ce loader gère uniquement le mode tabulaire features.
        if mode == "spectra":
            raise NotImplementedError(
                "mode='spectra' → utiliser SpectralMatrixLoader (notebook 01_pca.ipynb)."
            )

        # 1) Chargement
        # Lecture indépendante du jeu de features et du catalogue Gaia.
        df_feat = self._load_features()
        df_cat = self._load_catalog()

        # 2) Fusion
        # Jointure gauche: on conserve toutes les lignes features.
        df = self._merge(df_feat, df_cat)

        # 3) Filtres qualité
        if snr_band in df.columns:
            before = len(df)
            df = df[df[snr_band] >= snr_min].copy()
            logger.info(
                "Filtre SNR (%s >= %.1f) : %d → %d lignes",
                snr_band,
                snr_min,
                before,
                len(df),
            )
        else:
            logger.warning("Colonne '%s' introuvable — filtre SNR ignoré.", snr_band)

        if classes is not None:
            # Sous-ensemble de classes demandé par l'analyse (ex: STAR/GALAXY/QSO).
            df = df[df["class"].isin(classes)].copy()
            logger.info("Filtre classes %s : %d lignes restantes", classes, len(df))

        # 4) Sélection des colonnes features
        # La sélection applique les règles d'exclusion + seuil NaN.
        feat_cols = self._select_feature_columns(
            df, nan_threshold=drop_nan_threshold, spectro_only=spectro_only
        )
        self.feature_names_ = feat_cols

        # 5) Suppression des lignes avec NaN résiduel
        before = len(df)
        # Après sélection de colonnes, on retire les lignes incomplètes restantes.
        df = df.dropna(subset=feat_cols).copy()
        logger.info("Suppression NaN résiduel : %d → %d lignes", before, len(df))

        # 6) Équilibrage optionnel
        if class_balance:
            # Sous-échantillonnage pour limiter les biais de classes majoritaires.
            df = self._balance(df, n_per_class)

        # 7) Construction de X, y, meta
        # X_raw: matrice numérique prête pour la standardisation éventuelle.
        X_raw = df[feat_cols].values.astype(float)
        y = df["class"].values

        if scale:
            # StandardScaler ajusté sur ce run (stocké pour inverse_transform).
            self.scaler_ = StandardScaler()
            X = self.scaler_.fit_transform(X_raw)
        else:
            X = X_raw

        # meta conserve les colonnes utiles à la coloration/interprétation.
        meta = self._build_meta(df)
        # Cache interne du DataFrame final pour analyses complémentaires.
        self._df_merged = df

        logger.info(
            "Données prêtes : X=%s | classes=%s | features=%d",
            X.shape,
            dict(zip(*np.unique(y, return_counts=True))),
            len(feat_cols),
        )
        return X, y, meta

    # ------------------------------------------------------------------
    # Helpers privés
    # ------------------------------------------------------------------

    def _load_features(self) -> pd.DataFrame:
        """Charge le CSV de features en ignorant les colonnes inutiles."""
        if not self.features_path.exists():
            raise FileNotFoundError(
                f"Fichier features introuvable : {self.features_path}\n"
                "Lancer d'abord le pipeline AstroSpectro (master.py ou 00_master_pipeline.ipynb)."
            )
        # low_memory=False évite des dtypes incohérents sur gros CSV hétérogènes.
        df = pd.read_csv(self.features_path, low_memory=False)
        logger.info(
            "Features chargées : %s — %d colonnes", self.features_path.name, df.shape[1]
        )
        return df

    def _load_catalog(self) -> pd.DataFrame:
        """Charge le catalog Gaia cross-matché."""
        if not self.catalog_path.exists():
            raise FileNotFoundError(
                f"Catalog Gaia introuvable : {self.catalog_path}\n"
                "Vérifier le chemin ou lancer gaia_crossmatcher.py."
            )
        # Chargement brut du catalogue; les filtres sont appliqués plus tard.
        df = pd.read_csv(self.catalog_path, low_memory=False)
        logger.info("Catalog Gaia chargé : %d objets", len(df))
        return df

    def _merge(self, df_feat: pd.DataFrame, df_cat: pd.DataFrame) -> pd.DataFrame:
        """Fusionne features et catalog sur la clé commune."""
        key = self.merge_key

        # Colonnes du catalog à joindre (éviter doublons de class/subclass)
        # On ne rapatrie que les colonnes absentes du DataFrame features.
        cat_extra = [c for c in df_cat.columns if c != key and c not in df_feat.columns]
        # On garde aussi class/subclass si absents dans features
        for col in ("class", "subclass"):
            if col not in df_feat.columns and col in df_cat.columns:
                cat_extra.append(col)

        # Jointure sur la clé partagée, sans perdre les lignes features.
        df = df_feat.merge(
            df_cat[[key] + cat_extra],
            on=key,
            how="left",
        )
        logger.info("Merge features ∩ catalog : %d lignes", len(df))
        return df

    def _select_feature_columns(
        self,
        df: pd.DataFrame,
        nan_threshold: float = 0.10,
        spectro_only: bool = False,
    ) -> list[str]:
        """
        Sélectionne les colonnes numériques non-meta à utiliser comme features.

        Exclut : colonnes d'identifiant, colonnes meta Gaia, SNR, étiquettes,
        colonnes constantes, colonnes avec trop de NaN, et — lorsque
        ``spectro_only=True`` — toutes les colonnes non-spectroscopiques
        (photométrie, cinématique, paramètres Gaia, artefacts instrumentaux).

        Le jeu de features résultant est aligné avec celui utilisé par
        ``SpectralClassifier(spectro_only=True)`` pour garantir la comparabilité
        entre les méthodes supervisées (XGBoost) et non-supervisées
        (PCA / UMAP / autoencodeur).

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame fusionné (features + catalog).
        nan_threshold : float
            Fraction max de NaN par colonne avant suppression (défaut 10 %).
        spectro_only : bool
            Si True, applique en plus le filtre spectroscopique strict.
        """
        # ── Exclusions de base (identifiants, étiquettes, strings) ──────────
        exclude = {
            self.merge_key,
            # Identifiants
            "obsid",
            "fits_name",
            "filename_original",
            "plan_id",
            "mjd",
            "jd",
            "designation",
            "object_name",
            # Étiquettes
            "class",
            "subclass",
            "label",
            "main_class",
            # Métadonnées textuelles
            "author",
            "data_version",
            "date_creation",
            "telescope",
            "fiber_type",
            "catalog_object_type",
            "magnitude_type",
            "heliocentric_correction",
            "obs_date_utc",
            "phot_variable_flag",
            # Identifiants Gaia
            "source_id",
            "gaia_ra",
            "gaia_dec",
        }
        # Colonnes Gaia meta (conservées dans meta, pas dans X)
        exclude |= set(_GAIA_META_COLS)

        # ── Mode spectro_only : exclure SNR + instrumentaux + photométrie ────
        if spectro_only:
            # SNR = qualité observationnelle, pas propriété de l'étoile
            exclude |= set(_SNR_COLS)
            # Artefacts instrumentaux et métadonnées LAMOST
            # (synchronisé avec classifier.py cols_to_exclude)
            exclude |= set(_INSTRUMENTAL_COLS)
            # Préfixes photométriques / cinématiques Gaia
            # (synchronisé avec classifier.py spectro_only logic)
            logger.info(
                "Mode spectro_only=True : exclusion SNR + instrumentaux + Gaia photom."
            )
        else:
            # Mode par défaut : exclure uniquement les SNR
            # (comportement historique conservé pour rétrocompatibilité)
            exclude |= set(_SNR_COLS)

        # ── Sélection des colonnes candidates ────────────────────────────────
        feat_cols = []
        for c in df.columns:
            # 1) Exclusion par nom explicite.
            if c in exclude:
                continue
            # 2) Exclusion non-numérique (objets/chaînes/date).
            if not pd.api.types.is_numeric_dtype(df[c]):
                continue
            # En mode spectro_only, exclure aussi par préfixe Gaia/photom.
            if spectro_only and c.startswith(_NON_SPECTRO_PREFIXES):
                logger.debug(
                    "Colonne '%s' exclue (spectro_only, préfixe non-spectro)", c
                )
                continue
            # 3) Exclusion si trop de données manquantes.
            nan_frac = df[c].isna().mean()
            if nan_frac > nan_threshold:
                logger.debug("Colonne '%s' exclue (%.1f%% NaN)", c, 100 * nan_frac)
                continue
            # 4) Exclusion des colonnes constantes (sans signal discriminant).
            if df[c].nunique() <= 1:
                logger.debug("Colonne '%s' exclue (constante)", c)
                continue
            # 5) Colonne retenue comme feature exploitable.
            feat_cols.append(c)

        logger.info(
            "%d features sélectionnées (spectro_only=%s)",
            len(feat_cols),
            spectro_only,
        )
        return feat_cols

    def _balance(self, df: pd.DataFrame, n_per_class: int) -> pd.DataFrame:
        """Sous-échantillonnage aléatoire équilibré par classe."""
        # Graine fixe pour obtenir le même sous-échantillon entre exécutions.
        rng = np.random.default_rng(self.random_state)
        groups = []
        for cls, grp in df.groupby("class"):
            if len(grp) > n_per_class:
                # Échantillonnage sans remise pour plafonner la classe.
                idx = rng.choice(len(grp), size=n_per_class, replace=False)
                grp = grp.iloc[idx]
                logger.info(
                    "Classe '%s' sous-échantillonnée : %d → %d",
                    cls,
                    len(grp) + n_per_class - len(idx),
                    n_per_class,
                )
            groups.append(grp)
        # Reconcatène toutes les classes dans un unique DataFrame.
        return pd.concat(groups, ignore_index=True)

    def _build_meta(self, df: pd.DataFrame) -> pd.DataFrame:
        """Construit le DataFrame meta avec les paramètres physiques disponibles."""
        # Conserve uniquement les colonnes présentes pour rester robuste aux variantes de schéma.
        meta_cols = []
        for col in (
            _GAIA_META_COLS + _SNR_COLS + ["class", "subclass", "ra", "dec", "redshift"]
        ):
            if col in df.columns:
                meta_cols.append(col)
        # Reset index pour alignement parfait avec X et y retournés.
        meta = df[meta_cols].copy().reset_index(drop=True)
        return meta

    # ------------------------------------------------------------------
    # Utilitaires publics
    # ------------------------------------------------------------------

    def get_feature_names(self) -> list[str]:
        """Retourne les noms de features après load()."""
        if self.feature_names_ is None:
            raise RuntimeError("Appeler load() avant get_feature_names().")
        return self.feature_names_

    def inverse_transform(self, X_scaled: np.ndarray) -> np.ndarray:
        """Dé-standardise X si un scaler a été ajusté."""
        if self.scaler_ is None:
            # Aucun scaling appliqué en amont: retour inchangé.
            return X_scaled
        # Retour à l'échelle physique des features d'origine.
        return self.scaler_.inverse_transform(X_scaled)


class SpectralMatrixLoader:
    """
    Charge une matrice de spectres bruts interpolés sur une grille commune.

    Usage typique (notebook 01_pca.ipynb) :
        Lire les FITS depuis data/raw/, interpoler sur une grille λ fixe,
        normaliser et empiler en matrice N × P.

    Parameters
    ----------
    fits_dir : str | Path
        Dossier contenant les fichiers FITS téléchargés (peut contenir
        des sous-dossiers par plan — la recherche est récursive).
    catalog_path : str | Path
        Chemin vers master_catalog_gaia.csv.
    wl_grid : np.ndarray | None
        Grille wavelength cible (Å). Si None, utilise la grille LAMOST DR5
        standard (3690–9100 Å, pas ~1 Å, ~3909 pixels).
    n_jobs : int
        Nombre de cœurs pour le chargement parallèle.
    random_state : int
        Graine aléatoire pour reproductibilité.
    """

    # Grille LAMOST DR5 standard
    WL_MIN = 3690.0  # Å
    WL_MAX = 9100.0  # Å
    WL_STEP = 1.38  # Å environ (DR5 typique)

    def __init__(
        self,
        fits_dir: str | Path,
        catalog_path: str | Path,
        wl_grid: Optional[np.ndarray] = None,
        n_jobs: int = 4,
        random_state: int = 42,
    ) -> None:
        self.fits_dir = Path(fits_dir)
        self.catalog_path = Path(catalog_path)
        # Grille par défaut DR5 si non fournie explicitement.
        self.wl_grid = (
            wl_grid
            if wl_grid is not None
            else np.arange(self.WL_MIN, self.WL_MAX, self.WL_STEP)
        )
        self.n_jobs = n_jobs
        self.random_state = random_state

        # Index FITS construit au premier appel de load()
        self._fits_index: Optional[dict[str, Path]] = None

    def _build_fits_index(self) -> dict[str, Path]:
        """
        Construit un index {filename → chemin complet} en scannant
        récursivement fits_dir. Gère les sous-dossiers par plan.

        Returns
        -------
        dict[str, Path]
            Mapping nom de fichier → chemin absolu.
        """
        # Cache local: évite de rescanner le disque à chaque load().
        if self._fits_index is not None:
            return self._fits_index

        logger.info("Construction de l'index FITS dans %s...", self.fits_dir)
        self._fits_index = {}
        for p in self.fits_dir.rglob("*.fits*"):
            # En cas de doublons, on garde le premier trouvé
            if p.name not in self._fits_index:
                self._fits_index[p.name] = p

        logger.info("Index FITS construit : %d fichiers trouvés", len(self._fits_index))
        return self._fits_index

    def inspect_sample_fits(self, n_samples: int = 3) -> list[str]:
        """
        Inspecte quelques fichiers FITS et retourne des lignes prêtes a afficher.

        Cette methode factorise le bloc de verification rapide utilise dans
        le notebook PCA (section spectres bruts), sans modifier la logique
        scientifique du pipeline.

        Parameters
        ----------
        n_samples : int
            Nombre de fichiers FITS a inspecter.

        Returns
        -------
        list[str]
            Lignes de texte detaillees pour impression dans un notebook.
        """
        import astropy.io.fits as fits

        # Inventaire rapide du répertoire brut.
        fits_paths = list(self.fits_dir.rglob("*.fits*"))
        # Sélection des premiers fichiers pour diagnostic léger.
        sample_paths = fits_paths[: max(0, n_samples)]

        lines = [
            f"Fichiers trouvés dans RAW_DATA_DIR : {len(fits_paths)}",
        ]

        for path in sample_paths:
            lines.append(f"\n--- {path.name} ---")
            try:
                with fits.open(path) as hdul:
                    # Liste des HDU pour vérifier la structure du FITS.
                    lines.append(f"  HDU list : {[h.name for h in hdul]}")

                    data = hdul[0].data
                    shape = data.shape if data is not None else "None"
                    lines.append(f"  Data shape : {shape}")

                    header = hdul[0].header
                    # COEFF0/COEFF1 servent à reconstruire la grille en log-lambda.
                    lines.append(f"  COEFF0 : {header.get('COEFF0', 'ABSENT')}")
                    lines.append(f"  COEFF1 : {header.get('COEFF1', 'ABSENT')}")

                    if data is not None:
                        # Extraction du flux principal, compatible 1D ou 2D.
                        flux = data[0] if data.ndim > 1 else data
                        lines.append(
                            f"  Flux range : [{flux.min():.2f}, {flux.max():.2f}]"
                        )
                        lines.append(
                            f"  NaN count  : {np.isnan(flux).sum()}/{len(flux)}"
                        )
            except Exception as exc:
                # Le diagnostic continue même si un fichier est corrompu.
                lines.append(f"  ERREUR : {exc}")

        return lines

    def load(
        self,
        n_spectra: Optional[int] = None,
        snr_min: float = 10.0,
        classes: Optional[list[str]] = None,
        normalize: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """
        Charge et interpole N spectres depuis les fichiers FITS.

        Parameters
        ----------
        n_spectra : int | None
            Nombre maximum de spectres à charger (sous-échantillon aléatoire).
            None = tous les spectres du catalogue.
        snr_min : float
            SNR minimum sur bande r pour filtrer le catalogue.
        classes : list[str] | None
            Filtre par classe. None = toutes.
        normalize : bool
            Si True, normalise chaque spectre par sa médiane.

        Returns
        -------
        X : np.ndarray (N, len(wl_grid))
            Matrice de flux interpolés et normalisés.
        y : np.ndarray (N,)
            Étiquettes de classe.
        meta : pd.DataFrame
            Paramètres physiques Gaia.
        """
        from joblib import Parallel, delayed
        import astropy.io.fits as fits
        from scipy.interpolate import interp1d

        # 1) Lecture du catalogue maître.
        catalog = pd.read_csv(self.catalog_path)

        # Filtres catalogue
        if snr_min > 0 and "snr_r" in catalog.columns:
            # Qualité minimale du signal avant accès disque FITS.
            catalog = catalog[catalog["snr_r"] >= snr_min]
        if classes is not None:
            # Filtre de classes optionnel (ex. STAR uniquement).
            catalog = catalog[catalog["class"].isin(classes)]
        if n_spectra is not None and len(catalog) > n_spectra:
            # Sous-échantillonnage reproductible pour essais rapides.
            catalog = catalog.sample(n_spectra, random_state=self.random_state)

        # ── FIX : résolution récursive des chemins FITS ──────────────
        fits_index = self._build_fits_index()

        fits_paths = []
        resolved_mask = []
        for _, row in catalog.iterrows():
            # Résolution nom logique (catalogue) -> chemin physique (disque).
            fname = row["fits_name"]
            path = fits_index.get(fname)
            fits_paths.append(path)
            resolved_mask.append(path is not None)

        n_resolved = sum(resolved_mask)
        n_missing = len(resolved_mask) - n_resolved
        if n_missing > 0:
            logger.warning(
                "%d / %d fichiers FITS introuvables dans %s (vérifier fits_name vs fichiers physiques)",
                n_missing,
                len(resolved_mask),
                self.fits_dir,
            )
        if n_resolved == 0:
            logger.error(
                "Aucun fichier FITS résolu ! Contenu attendu (catalogue) : %s | "
                "Contenu trouvé (disque) : %s",
                catalog["fits_name"].iloc[:3].tolist() if len(catalog) > 0 else "vide",
                list(fits_index.keys())[:3] if fits_index else "vide",
            )
        # ─────────────────────────────────────────────────────────────

        wl_grid = self.wl_grid
        n_wl = len(wl_grid)

        def _load_one(path):
            """Charge un spectre FITS et interpole sur la grille commune."""
            if path is None:
                # Fichier non résolu: ligne entièrement NaN, filtrée plus loin.
                return np.full(n_wl, np.nan)
            try:
                with fits.open(path) as hdul:
                    data = hdul[0].data
                    header = hdul[0].header
                    # Convention LAMOST: flux principal dans la première ligne.
                    flux = data[0].astype(float)
                    # Reconstruction de la grille wavelength native depuis l'en-tête.
                    loglam = header["COEFF0"] + np.arange(len(flux)) * header["COEFF1"]
                    wl = 10**loglam

                    # Interpolation sur grille commune
                    interp_fn = interp1d(
                        wl,
                        flux,
                        kind="linear",
                        bounds_error=False,
                        fill_value=np.nan,
                    )
                    flux_interp = interp_fn(wl_grid)

                    if normalize:
                        # Normalisation par médiane pour homogénéiser l'échelle des spectres.
                        med = np.nanmedian(flux_interp)
                        if med > 0:
                            flux_interp /= med

                    return flux_interp
            except Exception as exc:
                # Toute erreur locale retourne NaN pour préserver le batch global.
                logger.warning("Erreur lecture %s : %s", path.name, exc)
                return np.full(n_wl, np.nan)

        logger.info(
            "Chargement de %d spectres FITS (%d résolus, %d jobs)...",
            len(fits_paths),
            n_resolved,
            self.n_jobs,
        )
        rows = Parallel(n_jobs=self.n_jobs, backend="loky")(
            delayed(_load_one)(p) for p in fits_paths
        )

        # 2) Empilement des spectres interpolés en matrice N x P.
        X = np.vstack(rows)

        # Supprime les spectres avec trop de NaN (> 20%)
        # Ce seuil protège les analyses contre des FITS partiellement invalides.
        nan_frac_per_row = np.mean(np.isnan(X), axis=1)
        valid = nan_frac_per_row < 0.20
        n_rejected = (~valid).sum()
        if n_rejected > 0:
            logger.info(
                "Spectres rejetés (>20%% NaN) : %d / %d",
                n_rejected,
                len(X),
            )

        # 3) Applique le masque de validité à X et au catalogue aligné.
        X = X[valid]
        catalog = catalog.iloc[valid].reset_index(drop=True)

        # Remplace les NaN résiduels par 0 (valeur neutre après normalisation)
        # Permet d'alimenter directement PCA/UMAP sans échec de fit.
        X = np.nan_to_num(X, nan=0.0)

        y = catalog["class"].values
        # Meta minimal utile aux colorations scientifiques en aval.
        meta_cols = [
            c
            for c in _GAIA_META_COLS + _SNR_COLS + ["class", "subclass"]
            if c in catalog.columns
        ]
        meta = catalog[meta_cols].reset_index(drop=True)

        logger.info("Matrice spectrale prête : %s", X.shape)
        return X, y, meta
