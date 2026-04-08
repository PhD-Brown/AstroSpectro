"""
AstroSpectro — dimred.xgboost_bridge
=======================================

Pont entre le classifieur XGBoost (pipeline src/) et l'espace UMAP (dimred/).

Ce module centralise la logique qui était dans la cellule 19 du notebook
phy3500_02_umap_tsne.ipynb : chargement du modèle, alignement des features,
prédiction, et construction de la figure trianneau (prédictions / confiance /
clusters HDBSCAN).

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

# Colonnes à exclure lors de l'alignement (reproduit DimRedDataLoader)
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

# Palette couleurs par type spectral stellaire MK
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
    paths          : dict de chemins du projet (setup_project_env).
    features_stem  : identifiant du CSV de features.
    Z_umap         : embedding UMAP (N, 2).
    y              : étiquettes de classe LAMOST (N,).
    cluster_labels : étiquettes HDBSCAN (N,).
    color_map      : palette {cluster_id: rgba} depuis HDBSCANAnalyzer.

    Returns
    -------
    dict avec les clés :
      y_pred, confidence, classes_pred, y_aligned, cl_aligned,
      Z_umap_aligned, fg_mask, path_xgb, path_fg, _xgb_ok
    """
    # ── Résolution du chemin du modèle ──────────────────────────────────────
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
        logger.warning("Aucun modèle spectral_classifier*.pkl trouvé → XGBoost ignoré.")
        return _empty_result()

    try:
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
        logger.error("Erreur chargement XGBoost : %s", exc)
        return _empty_result()

    # ── Chargement et alignement des features ───────────────────────────────
    features_csv = Path(paths["PROCESSED_DIR"]) / f"{features_stem}.csv"
    if not features_csv.exists():
        files = sorted(Path(paths["PROCESSED_DIR"]).glob("features_*.csv"))
        if not files:
            logger.warning("Aucun features_*.csv trouvé.")
            return _empty_result()
        features_csv = files[-1]

    df_raw = pd.read_csv(features_csv, low_memory=False)

    # Filtre SNR
    if "snr_r" in df_raw.columns:
        df_f = df_raw[df_raw["snr_r"] >= 10.0].copy()
    else:
        df_f = df_raw.copy()

    # Reproduire exactement le dropna de DimRedDataLoader
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
    X_pred_df = df_aligned[needed].fillna(0)
    logger.info("Prédiction sur %d spectres…", len(X_pred_df))

    y_pred_enc = clf.model_pipeline.predict(X_pred_df)
    if clf.label_encoder is not None:
        y_pred = clf.label_encoder.inverse_transform(y_pred_enc)
    else:
        y_pred = np.array(y_pred_enc, dtype=str)

    try:
        proba = clf.model_pipeline.predict_proba(X_pred_df)
        confidence = proba.max(axis=1)
        has_proba = True
    except Exception:
        confidence = np.ones(len(y_pred))
        has_proba = False

    classes_pred = sorted(set(y_pred))
    logger.info(
        "Prédictions : %s",
        dict(zip(*np.unique(y_pred, return_counts=True))),
    )

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
