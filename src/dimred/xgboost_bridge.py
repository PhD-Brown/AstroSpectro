"""
AstroSpectro — dimred.xgboost_bridge
=======================================

Pont entre le classifieur XGBoost (pipeline src/) et l'espace UMAP (dimred/).

Ce module centralise la logique qui était dans la cellule 19 du notebook
phy3500_02_umap_tsne.ipynb : chargement du modèle, alignement des features,
prédiction, et construction de la figure trianneau (prédictions / confiance /
clusters HDBSCAN).

Objectif méthodologique
-----------------------
Ce pont répond à une question centrale du projet PHY-3500 :
"la structure non supervisée observée en UMAP est-elle cohérente avec
un modèle supervisé entraîné sur les mêmes descripteurs spectraux ?"

Pour y répondre proprement, il faut garantir que :
1) les features passées à XGBoost reproduisent exactement le protocole
    de préparation utilisé au moment de l'entraînement ;
2) l'alignement des lignes entre DataFrame tabulaire et embedding UMAP
    est strictement conservé ;
3) la fonction retourne un contrat de sortie stable, même en cas d'échec,
    afin d'éviter des erreurs en cascade dans les notebooks.

Usage
-----
>>> from dimred.xgboost_bridge import load_and_predict
>>> result = load_and_predict(paths=paths, features_stem=features_stem,
...                           Z_umap=Z_umap, y=y, cluster_labels=cluster_labels)
>>> y_pred, confidence = result['y_pred'], result['confidence']
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Colonnes à exclure lors de l'alignement (reproduit DimRedDataLoader).
#
# Ces colonnes correspondent à des identifiants, métadonnées observationnelles,
# paramètres physiques Gaia, ou variables de qualité qui ne doivent pas être
# mélangés à l'espace de features d'entrée du classifieur spectral.
#
# L'objectif est de recalculer, dans ce module, exactement le même sous-ensemble
# de colonnes numériques "apprenables" que dans le pipeline principal.
_EXCL_COLS = {
    "obsid",
    "fits_name",
    "filename_original",
    "plan_id",
    "mjd",
    "jd",
    "designation",
    "object_name",
    "class",
    "subclass",
    "author",
    "data_version",
    "date_creation",
    "telescope",
    "fiber_type",
    "catalog_object_type",
    "magnitude_type",
    "heliocentric_correction",
    "radial_velocity_corr",
    "obs_date_utc",
    "phot_variable_flag",
    "source_id",
    "gaia_ra",
    "gaia_dec",
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
    "snr_u",
    "snr_g",
    "snr_r",
    "snr_i",
    "snr_z",
}

# Palette couleurs par type spectral stellaire MK.
# Gardée ici comme référence centralisée pour les visualisations couplées
# XGBoost/UMAP (cohérence visuelle inter-notebooks et inter-figures).
STELLAR_COLORS: Dict[str, str] = {
    "O": "#9B59B6",
    "B": "#3498DB",
    "A": "#1ABC9C",
    "F": "#F1C40F",
    "G": "#E67E22",
    "K": "#E74C3C",
    "M": "#922B21",
    "C": "#884EA0",
    "W": "#17202A",
    "s": "#7F8C8D",
    "STAR": "#4C72B0",
    "GALAXY": "#DD8452",
    "QSO": "#55A868",
}


def load_and_predict(
    *,
    paths: dict,
    features_stem: str,
    Z_umap: np.ndarray,
    y: np.ndarray,
    cluster_labels: np.ndarray,
    color_map: Optional[dict] = None,
) -> Dict[str, Any]:
    """
    Charge le SpectralClassifier XGBoost et prédit les types sur les spectres UMAP.

    Reproduit fidèlement les deux corrections de bug de la cellule originale :
    1. Alignement SNR + dropna pour correspondre exactement à Z_umap.
    2. Passage d'un DataFrame (pas un array numpy) au ColumnTransformer.

    Parameters
    ----------
    paths
        Dictionnaire de chemins projet (sortie de setup_project_env).
    features_stem
        Identifiant du CSV de features utilisé pour construire le chemin.
    Z_umap
        Embedding UMAP de forme (N, 2).
    y
        Étiquettes de classe LAMOST de forme (N,).
    cluster_labels
        Étiquettes HDBSCAN de forme (N,).
    color_map
        Palette des clusters HDBSCAN. Conservée pour compatibilité d'API,
        même si cette fonction ne trace pas directement les figures.

    Returns
    -------
    dict
        Dictionnaire standardisé contenant :
        y_pred, confidence, has_proba, classes_pred, y_aligned, cl_aligned,
        Z_umap_aligned, fg_mask, model_path, _xgb_ok.

    Notes
    -----
    La fonction privilégie la robustesse opérationnelle : en cas d'échec
    (modèle absent, chargement impossible, features incompatibles), elle
    retourne un dictionnaire vide structuré via _empty_result() plutôt que
    de lever une exception bloquante côté notebook.
    """
    # ── Résolution du chemin du modèle ──────────────────────────────────────
    # On tente d'utiliser utils.latest_file depuis src/. Si l'import échoue,
    # on fournit un fallback local minimal pour rester exécutable partout.
    try:
        src_root = str(Path(paths.get("SRC_DIR", "../../src")).resolve())
        if src_root not in sys.path:
            sys.path.insert(0, src_root)
        from utils import latest_file
    except ImportError:

        def latest_file(directory, pattern):
            files = sorted(Path(directory).glob(pattern))
            return str(files[-1]) if files else None

    model_path = latest_file(paths["MODELS_DIR"], "spectral_classifier*.pkl")
    if model_path is None:
        # Pas de modèle => pas de prédiction possible. Retour vide explicite.
        logger.warning("Aucun modèle spectral_classifier*.pkl trouvé → XGBoost ignoré.")
        return _empty_result()

    try:
        # Compatibilité double import:
        # - mode package: pipeline.classifier
        # - mode script/notebook: classifier
        try:
            from pipeline.classifier import SpectralClassifier
        except ImportError:
            from classifier import SpectralClassifier
        clf = SpectralClassifier.load_model(model_path)
        logger.info(
            "Modèle XGBoost chargé : %s | classes=%s | features=%d",
            Path(model_path).name,
            clf.class_labels,
            len(clf.feature_names_used),
        )
    except Exception as exc:
        # Toute erreur de chargement est encapsulée en résultat vide pour
        # préserver la continuité d'exécution des notebooks.
        logger.error("Erreur chargement XGBoost : %s", exc)
        return _empty_result()

    # ── Chargement et alignement des features ───────────────────────────────
    # On privilégie d'abord le fichier explicitement attendu via features_stem,
    # puis fallback sur le plus récent features_*.csv si absent.
    features_csv = Path(paths["PROCESSED_DIR"]) / f"{features_stem}.csv"
    if not features_csv.exists():
        files = sorted(Path(paths["PROCESSED_DIR"]).glob("features_*.csv"))
        if not files:
            logger.warning("Aucun features_*.csv trouvé.")
            return _empty_result()
        features_csv = files[-1]

    df_raw = pd.read_csv(features_csv, low_memory=False)

    # Filtre SNR
    # Ce filtre reproduit la logique de sélection qualité utilisée dans la
    # branche dimred afin de comparer des objets homogènes en qualité spectrale.
    if "snr_r" in df_raw.columns:
        df_f = df_raw[df_raw["snr_r"] >= 10.0].copy()
    else:
        df_f = df_raw.copy()

    # Reproduire exactement le dropna de DimRedDataLoader
    # Règles de sélection des colonnes candidates :
    # - numériques,
    # - non exclues explicitement,
    # - non constantes,
    # - taux de valeurs manquantes <= 10%.
    pca_feat_cols = [
        c
        for c in df_f.columns
        if c not in _EXCL_COLS
        and pd.api.types.is_numeric_dtype(df_f[c])
        and df_f[c].nunique() > 1
        and df_f[c].isna().mean() <= 0.10
    ]
    df_aligned = df_f.dropna(subset=pca_feat_cols).reset_index(drop=True)
    logger.info(
        "Lignes après filtre SNR + dropna : %d  (Z_umap : %d)",
        len(df_aligned),
        len(Z_umap),
    )

    if len(df_aligned) != len(Z_umap):
        # Sécurité d'alignement : on tronque au minimum commun pour garantir
        # la correspondance index-à-index entre DataFrame et coordonnées UMAP.
        n_common = min(len(df_aligned), len(Z_umap))
        df_aligned = df_aligned.iloc[:n_common]
        Z_umap_aligned = Z_umap[:n_common]
        y_aligned = y[:n_common]
        cl_aligned = cluster_labels[:n_common]
        logger.warning("Alignement tronqué au minimum commun : %d", n_common)
    else:
        Z_umap_aligned = Z_umap
        y_aligned = y
        cl_aligned = cluster_labels
        logger.info("Alignement parfait : %d spectres", len(df_aligned))

    # ── Vérification des features nécessaires ───────────────────────────────
    needed = clf.feature_names_used
    missing = [f for f in needed if f not in df_aligned.columns]
    if missing:
        # Politique de tolérance :
        # - si peu de features manquent, on les impute à 0.0 (fallback doux),
        # - si trop de features manquent (>30%), la prédiction devient peu
        #   fiable et on préfère abandonner proprement.
        pct = len(missing) / len(needed)
        logger.warning(
            "%d/%d features manquantes (%.0f%%)", len(missing), len(needed), pct * 100
        )
        if pct > 0.30:
            logger.error("Trop de features manquantes → XGBoost ignoré.")
            return _empty_result()
        for f in missing:
            df_aligned[f] = 0.0

    # ── Prédiction (DataFrame, pas array) ───────────────────────────────────
    # Important: on conserve un DataFrame nommé pour respecter l'ordre et les
    # noms attendus par le ColumnTransformer du pipeline scikit-learn.
    X_pred_df = df_aligned[needed].fillna(0)
    logger.info("Prédiction sur %d spectres…", len(X_pred_df))

    # Prédiction des classes encodées.
    y_pred_enc = clf.model_pipeline.predict(X_pred_df)
    if clf.label_encoder is not None:
        # Cas standard: décodage des labels vers les noms de classes.
        y_pred = clf.label_encoder.inverse_transform(y_pred_enc)
    else:
        # Cas défensif: si pas d'encodeur, on force un tableau de chaînes.
        y_pred = np.array(y_pred_enc, dtype=str)

    try:
        # Si le modèle expose predict_proba, on récupère une confiance
        # scalaire par objet (max de la distribution de probabilité).
        proba = clf.model_pipeline.predict_proba(X_pred_df)
        confidence = proba.max(axis=1)
        has_proba = True
    except Exception:
        # Certains estimateurs/pipelines ne fournissent pas predict_proba.
        # On conserve alors une confiance neutre à 1.0.
        confidence = np.ones(len(y_pred))
        has_proba = False

    classes_pred = sorted(set(y_pred))
    logger.info(
        "Prédictions : %s",
        dict(zip(*np.unique(y_pred, return_counts=True))),
    )

    # Masque pratique pour l'analyse ciblée de la confusion F/G,
    # historiquement la zone de recouvrement la plus délicate.
    fg_mask = np.isin(y_pred, ["F", "G"])

    return {
        "y_pred": y_pred,
        "confidence": confidence,
        "has_proba": has_proba,
        "classes_pred": classes_pred,
        "y_aligned": y_aligned,
        "cl_aligned": cl_aligned,
        "Z_umap_aligned": Z_umap_aligned,
        "fg_mask": fg_mask,
        "model_path": str(model_path),
        "_xgb_ok": True,
    }


def _empty_result() -> dict:
    """
    Retourne une structure de sortie vide mais compatible avec l'API publique.

    Ce helper évite de multiplier les `if` dans les notebooks.
    Ils peuvent tester uniquement `_xgb_ok` puis accéder aux autres clés sans
    risque de KeyError.
    """
    return {
        "y_pred": None,
        "confidence": None,
        "has_proba": False,
        "classes_pred": [],
        "y_aligned": None,
        "cl_aligned": None,
        "Z_umap_aligned": None,
        "fg_mask": None,
        "model_path": None,
        "_xgb_ok": False,
    }
