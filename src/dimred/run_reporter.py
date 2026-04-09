"""
AstroSpectro — dimred.run_reporter
====================================

Sauvegarde des résultats de run pour les trois notebooks PHY-3500.

Ce module centralise toute la logique de sauvegarde et de génération de
rapports qui était auparavant dupliquée dans les cellules finales des
notebooks phy3500_01_pca, phy3500_02_umap_tsne et phy3500_03_autoencoder.

Responsabilités
---------------
- Fonctions utilitaires partagées : sérialisation JSON, comptage de
  classes, statistiques d'embedding, etc.
- Helpers I/O : création de répertoires horodatés, dump joblib, écriture
  JSON + TXT avec version « latest ».
- Trois fonctions de haut niveau, une par notebook :
    - save_pca_run()          → NB01
    - save_umap_tsne_run()    → NB02
    - save_autoencoder_run()  → NB03

Usage
-----
>>> from dimred.run_reporter import save_pca_run
>>> result = save_pca_run(pca=pca, pca_spec=pca_spec, scores=scores, ...)
>>> print(result["summary"])

Principes de conception
-----------------------
1) Robustesse opérationnelle : le module privilégie des sorties stables,
    explicites et traçables pour faciliter le débogage des notebooks.
2) Reproductibilité : chaque run est horodaté en UTC, avec artefacts
    versionnés et alias « latest » pour l'exploitation continue.
3) Lisibilité scientifique : chaque rapport (JSON + TXT) expose à la fois
    les métriques numériques, les dimensions, les chemins et les inventaires
    de figures nécessaires à l'interprétation.
4) Compatibilité descendante : certains artefacts legacy sont conservés pour
    ne pas casser les flux existants du projet AstroSpectro.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Utilitaires partagés
# ══════════════════════════════════════════════════════════════════════════════


def class_counts(labels) -> Dict[str, int]:
    """
    Compte les occurrences de chaque classe dans un tableau d'étiquettes.

    Notes
    -----
    Utilise ``np.unique(..., return_counts=True)``, ce qui garantit :
    - un comptage vectorisé efficace ;
    - un ordre déterministe des clés (ordre trié NumPy), utile pour
      comparer deux rapports de runs.
    """
    classes, counts = np.unique(labels, return_counts=True)
    return {str(cls): int(cnt) for cls, cnt in zip(classes, counts)}


def safe_df_records(
    df: Optional[pd.DataFrame], max_rows: Optional[int] = None
) -> Optional[List[dict]]:
    """
    Convertit un DataFrame en liste de dicts JSON-sérialisables.

    Remplace les NaN par None pour garantir la compatibilité JSON.
    Retourne None si df est None.

    Pourquoi ce helper est critique
    -------------------------------
    Les DataFrames issus des analyses (variance, corrélations, stabilité)
    contiennent souvent des NaN et des types NumPy. Cette fonction normalise
    ces tables pour éviter les erreurs de sérialisation dans ``json.dump``.
    """
    if df is None:
        return None
    # Copie défensive: évite de modifier le DataFrame original du notebook.
    out = df.copy()
    # Tronque éventuellement le tableau pour garder un JSON lisible.
    if max_rows is not None:
        out = out.head(max_rows)
    # Conversion NaN -> None pour être compatible JSON natif.
    out = out.where(pd.notna(out), None)
    # Format final consommable par json.dump.
    return out.to_dict(orient="records")


def json_default(obj: Any) -> Any:
    """
    Sérialiseur JSON pour les types NumPy / Pandas / Path / datetime.

    À passer comme argument ``default`` à ``json.dump``.

    Justification
    -------------
    Les rapports contiennent de nombreux types non natifs JSON (NumPy, Path,
    Timestamp). Centraliser la conversion ici garantit un comportement
    homogène sur les trois notebooks.
    """
    if isinstance(obj, (np.integer, np.floating)):
        # NumPy scalaire -> scalaire Python (int/float).
        return obj.item()
    if isinstance(obj, np.ndarray):
        # Tableau NumPy -> liste imbriquée JSON.
        return obj.tolist()
    if isinstance(obj, Path):
        # Les chemins sont stockés en texte pour traçabilité.
        return str(obj)
    if isinstance(obj, (datetime, pd.Timestamp)):
        # Horodatage ISO lisible et standardisé.
        return obj.isoformat()
    raise TypeError(f"Type non sérialisable : {type(obj)!r}")


def embedding_stats(Z: Optional[np.ndarray]) -> Optional[dict]:
    """
    Statistiques descriptives pour un embedding 2D.

    Retourne min/max/moyenne/écart-type sur les axes x/y afin de documenter
    l'échelle géométrique des projections (UMAP/t-SNE/AE).
    """
    if Z is None:
        return None
    # Normalisation d'entrée: accepte liste, DataFrame ou ndarray.
    arr = np.asarray(Z)
    if arr.ndim != 2 or arr.shape[1] < 2:
        # Cas incomplet: on documente au moins la forme reçue.
        return {"shape": list(arr.shape)}
    # Statistiques de dispersion et d'échelle pour les deux axes principaux.
    return {
        "shape": list(arr.shape),
        "x_min": float(np.nanmin(arr[:, 0])),
        "x_max": float(np.nanmax(arr[:, 0])),
        "y_min": float(np.nanmin(arr[:, 1])),
        "y_max": float(np.nanmax(arr[:, 1])),
        "x_mean": float(np.nanmean(arr[:, 0])),
        "y_mean": float(np.nanmean(arr[:, 1])),
        "x_std": float(np.nanstd(arr[:, 0])),
        "y_std": float(np.nanstd(arr[:, 1])),
    }


def embedding_stats_nd(Z: Optional[np.ndarray], max_axes: int = 3) -> Optional[dict]:
    """
    Statistiques descriptives pour un embedding N-dimensionnel.

    Limite volontairement le nombre d'axes reportés (``max_axes``) pour
    conserver des JSON compacts et lisibles.
    """
    if Z is None:
        return None
    # Conversion systématique pour travailler avec une API NumPy homogène.
    arr = np.asarray(Z)
    if arr.ndim != 2:
        return {"shape": list(arr.shape)}
    out: dict = {"shape": list(arr.shape)}
    # On limite volontairement le nombre d'axes reportés pour rester concis.
    for i in range(min(arr.shape[1], max_axes)):
        vals = arr[:, i]
        out[f"axis_{i+1}_min"] = float(np.nanmin(vals))
        out[f"axis_{i+1}_max"] = float(np.nanmax(vals))
        out[f"axis_{i+1}_mean"] = float(np.nanmean(vals))
        out[f"axis_{i+1}_std"] = float(np.nanstd(vals))
    return out


def shape_or_none(x: Any) -> Optional[list]:
    """
    Retourne ``list(array.shape)`` ou ``None`` si conversion impossible.

    Ce helper évite de dupliquer des ``try/except`` autour des extractions de
    dimensions dans les sections de reporting.
    """
    if x is None:
        return None
    try:
        return list(np.asarray(x).shape)
    except Exception:
        # Si l'objet n'est pas convertible, on n'interrompt pas le reporting.
        return None


def sensitivity_shapes(sensitivity_dict: Optional[dict]) -> Optional[dict]:
    """
    Convertit un dict ``{label: Z}`` en dict ``{label: shape}``.

    Les matrices complètes de sensibilité peuvent être volumineuses ; on
    conserve ici uniquement leurs dimensions pour alléger le rapport.
    """
    if sensitivity_dict is None:
        return None
    # Réduction mémoire: on garde seulement les dimensions des matrices.
    return {str(k): list(np.asarray(v).shape) for k, v in sensitivity_dict.items()}


def numeric_means(df: Optional[pd.DataFrame]) -> Optional[dict]:
    """
    Retourne la moyenne de chaque colonne numérique d'un DataFrame.

    Utilisé comme résumé compact des tables de stabilité lorsque l'on ne veut
    pas relire la table complète dans le JSON.
    """
    if df is None or len(df) == 0:
        return None
    # Résume uniquement les colonnes numériques pour éviter les conversions ambiguës.
    cols = df.select_dtypes(include=[np.number]).columns
    return {str(c): float(df[c].mean()) for c in cols}


def fmt_seconds(value: Any) -> str:
    """
    Formate une durée en secondes vers une chaîne lisible (ou ``N/A``).

    Uniformise l'affichage des temps de calcul dans les rapports TXT.
    """
    if value is None:
        return "N/A"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "N/A"
    return f"{v:.2f}s" if np.isfinite(v) else "N/A"


def vector_stats(arr: Any) -> Optional[dict]:
    """
    Statistiques robustes (min, max, moyenne, écart-type, percentiles).

    Inclut p05/p50/p95 pour mieux caractériser des distributions asymétriques
    (ex. confiance XGBoost, erreurs de reconstruction).
    """
    if arr is None:
        return None
    vals = np.asarray(arr, dtype=float)
    # Filtre robuste: supprime NaN/inf avant calcul des percentiles.
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return None
    return {
        "n": int(vals.size),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals)),
        "p05": float(np.percentile(vals, 5)),
        "p50": float(np.percentile(vals, 50)),
        "p95": float(np.percentile(vals, 95)),
    }


def cluster_top_counts(labels, top_n: int = 10) -> dict:
    """
    Retourne les N clusters les plus peuplés (exclut le bruit ``-1``).

    Cette vue « top clusters » est pratique pour les rapports synthétiques
    sans charger toutes les distributions détaillées.
    """
    vals = np.asarray(labels)
    # Le bruit HDBSCAN (-1) est exclu pour se concentrer sur les clusters réels.
    counts = {str(int(c)): int((vals == c).sum()) for c in np.unique(vals) if c != -1}
    return dict(sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:top_n])


def top3_pred_by_cluster(cluster_arr, pred_arr) -> dict:
    """
    Pour chaque cluster, retourne les 3 prédictions les plus fréquentes.

    Ce résumé permet d'évaluer rapidement la cohérence entre clusters
    non supervisés (HDBSCAN) et classes supervisées (XGBoost).
    """
    out: dict = {}
    c = np.asarray(cluster_arr)
    p = np.asarray(pred_arr)
    for cid in np.unique(c):
        # À chaque cluster, on isole les prédictions correspondantes.
        mask = c == cid
        vc = pd.Series(p[mask]).value_counts().head(3)
        key = "noise" if cid == -1 else str(int(cid))
        out[key] = {str(k): int(v) for k, v in vc.items()}
    return out


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Helpers I/O
# ══════════════════════════════════════════════════════════════════════════════


def make_timestamp() -> str:
    """
    Génère un horodatage UTC au format ``YYYYMMDDTHHMMSSZ``.

    L'usage systématique de l'UTC évite les ambiguïtés de fuseau horaire
    lors de la comparaison de runs exécutés sur des machines différentes.
    """
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def make_run_dir(reports_dir: Path, sub: str) -> Path:
    """
    Crée (si besoin) ``reports_dir/runs/<sub>`` et retourne le chemin.

    Cette convention impose une arborescence stable :
    ``data/reports/runs/<famille_de_notebook>/...``
    """
    run_dir = reports_dir / "runs" / sub
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_joblib_pair(
    obj: Any,
    run_dir: Path,
    stem_run: str,
    stem_latest: str,
    legacy_path: Optional[Path] = None,
) -> tuple[Path, Path]:
    """
    Sauvegarde ``obj`` en joblib dans deux fichiers (horodaté + latest).

    Parameters
    ----------
    obj
        Objet à sérialiser.
    run_dir
        Répertoire de destination.
    stem_run
        Nom de fichier horodaté (sans extension).
    stem_latest
        Nom de fichier « latest » (sans extension).
    legacy_path
        Chemin legacy facultatif (ex. ``phy3500_pca_output.joblib``).

    Returns
    -------
    (path_run, path_latest)

    Notes
    -----
    Le double-écriture (timestamp + latest) offre deux modes d'usage :
    - audit reproductible d'un run précis ;
    - accès rapide au dernier artefact pour les notebooks suivants.
    """
    import joblib

    path_run = run_dir / f"{stem_run}.joblib"
    path_latest = run_dir / f"{stem_latest}.joblib"

    # Écriture horodatée: preuve d'exécution immuable du run courant.
    joblib.dump(obj, path_run)
    # Écriture latest: pointeur pratique vers le plus récent résultat.
    joblib.dump(obj, path_latest)
    if legacy_path is not None:
        # Compatibilité descendante avec d'anciens chemins utilisés dans
        # certains notebooks/scripts historiques.
        joblib.dump(obj, legacy_path)

    logger.info("Joblib sauvegardé : %s", path_run)
    return path_run, path_latest


def save_json_txt_pair(
    report: dict,
    run_dir: Path,
    stem_run: str,
    stem_latest: str,
    text_lines: List[str],
) -> tuple[Path, Path, Path, Path]:
    """
    Sauvegarde le rapport sous forme JSON + TXT (horodaté + latest).

    Design
    ------
    - JSON : format machine-friendly (scripts, dashboards, comparaisons).
    - TXT  : format humain lisible pour rapport de cours et relecture rapide.

    Returns
    -------
    (json_path, json_latest, txt_path, txt_latest)
    """
    json_path = run_dir / f"{stem_run}.json"
    json_latest = run_dir / f"{stem_latest}.json"
    txt_path = run_dir / f"{stem_run}.txt"
    txt_latest = run_dir / f"{stem_latest}.txt"

    # ensure_ascii=False pour conserver les accents français dans les rapports.
    # On écrit en double (run + latest) pour combiner audit et accès rapide.
    for p in (json_path, json_latest):
        with open(p, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=json_default)

    text_content = "\n".join(text_lines)
    # Même logique pour le TXT: archive horodatée + alias latest.
    for p in (txt_path, txt_latest):
        with open(p, "w", encoding="utf-8") as f:
            f.write(text_content)

    logger.info("JSON sauvegardé   : %s", json_path)
    logger.info("TXT sauvegardé    : %s", txt_path)
    return json_path, json_latest, txt_path, txt_latest


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — save_pca_run  (NB01)
# ══════════════════════════════════════════════════════════════════════════════


def save_pca_run(
    *,
    pca,
    pca_spec,
    scores: np.ndarray,
    scores_spec: np.ndarray,
    X: np.ndarray,
    X_spec: np.ndarray,
    y: np.ndarray,
    y_spec: np.ndarray,
    meta: pd.DataFrame,
    meta_spec: pd.DataFrame,
    feature_names: list,
    paths: dict,
    FEATURES_PATH: Path,
    CATALOG_PATH: Path,
    FIGURES_DIR: Path,
    corr_df: Optional[pd.DataFrame] = None,
    sep_df: Optional[pd.DataFrame] = None,
    recon_df: Optional[pd.DataFrame] = None,
) -> dict:
    """
    Sauvegarde complète du run PCA (NB01).

    Génère :
    - Un fichier joblib avec les scores et métadonnées.
    - Un rapport JSON horodaté + version latest.
    - Un rapport TXT lisible horodaté + version latest.

    Rôle dans le pipeline
    ---------------------
    Cette fonction est la clôture analytique de NB01 : elle transforme les
    objets mémoire (scores, PCAAnalyzer, corrélations, tableaux auxiliaires)
    en artefacts persistants, exploitables par NB02/NB03 et auditables a
    posteriori.

    Parameters
    ----------
    pca, pca_spec    : PCAAnalyzer ajustés (features / spectres bruts).
    scores           : matrice de scores PCA sur les features.
    scores_spec      : matrice de scores PCA sur les spectres bruts.
    X, X_spec        : matrices d'entrée standardisées.
    y, y_spec        : étiquettes de classes.
    meta, meta_spec  : DataFrames de paramètres Gaia.
    feature_names    : liste des noms de features.
    paths            : dict de chemins du projet (``setup_project_env``).
    FEATURES_PATH    : chemin vers le CSV de features utilisé.
    CATALOG_PATH     : chemin vers le catalogue master Gaia.
    FIGURES_DIR      : répertoire des figures du run.
    corr_df          : DataFrame de corrélations Spearman (optionnel).
    sep_df           : DataFrame de séparation des classes (optionnel).
    recon_df         : DataFrame d'erreur de reconstruction (optionnel).

    Returns
    -------
    dict avec les clés ``timestamp``, ``run_dir``, ``json_path``,
    ``txt_path``, ``joblib_path``, ``summary`` (résumé texte court).
    """
    reports_dir = Path(paths["REPORTS_DIR"])
    # 1) Identifiant de run (UTC) et destination de stockage.
    timestamp = make_timestamp()
    run_dir = make_run_dir(reports_dir, "phy3500_pca")

    # Borne de sécurité : on ne demande jamais plus de composantes que ce
    # qui existe réellement dans les matrices de scores en entrée.
    # 2) Nombre de composantes conservées pour 95% de variance.
    max_pcs_features = scores.shape[1]
    max_pcs_spectra = scores_spec.shape[1]
    n_keep = min(int(pca.n_components_for_variance(0.95)), max_pcs_features)
    n_keep_spec = min(int(pca_spec.n_components_for_variance(0.95)), max_pcs_spectra)

    # Dictionnaires de seuils standards (80/90/95/99%) pour lecture rapide.
    # 3) Pré-calcul des seuils principaux pour éviter de recalculer plus bas.
    variance_thresholds = {
        f"{int(th * 100)}pct": min(
            int(pca.n_components_for_variance(th)), max_pcs_features
        )
        for th in (0.80, 0.90, 0.95, 0.99)
    }
    variance_thresholds_spec = {
        f"{int(th * 100)}pct": min(
            int(pca_spec.n_components_for_variance(th)), max_pcs_spectra
        )
        for th in (0.80, 0.90, 0.95, 0.99)
    }

    # ── Joblib ──────────────────────────────────────────────────────────────
    # Structure de base consommée ensuite par NB02 (UMAP/t-SNE) et utilisée
    # également pour l'archivage scientifique des résultats PCA.
    # 4) Ce payload contient les matrices utiles au chaînage inter-notebooks.
    output = {
        "scores": scores,
        "scores_95pct": scores[:, :n_keep],
        "y": y,
        "meta": meta,
        "feature_names": feature_names,
        "pca_variance_summary": pca.variance_summary(),
        "n_components_95pct": n_keep,
        "scores_spec": scores_spec,
        "scores_spec_95pct": scores_spec[:, :n_keep_spec],
        "y_spec": y_spec,
        "meta_spec": meta_spec,
        "n_keep_spec": n_keep_spec,
        "features_stem": FEATURES_PATH.stem,
    }
    # 5) Écriture physique des artefacts PCA (run + latest + legacy).
    legacy_path = reports_dir / "phy3500_pca_output.joblib"
    joblib_run, _joblib_latest = save_joblib_pair(
        output,
        run_dir,
        stem_run=f"phy3500_pca_run_{timestamp}",
        stem_latest="phy3500_pca_run_latest",
        legacy_path=legacy_path,
    )

    # ── Rapport JSON ────────────────────────────────────────────────────────
    # Le JSON capture la structure complète et machine-readable du run.
    # 6) Le JSON décrit les entrées, dimensions, métriques et fichiers produits.
    report = {
        "run_timestamp_utc": timestamp,
        "paths": {
            "features_path": str(FEATURES_PATH),
            "catalog_path": str(CATALOG_PATH),
            "figures_dir": str(FIGURES_DIR),
            "joblib_path": str(joblib_run),
            "run_dir": str(run_dir),
        },
        "shapes": {
            "X": list(X.shape),
            "scores": list(scores.shape),
            "X_spec": list(X_spec.shape),
            "scores_spec": list(scores_spec.shape),
            "scores_95pct": list(scores[:, :n_keep].shape),
            "scores_spec_95pct": list(scores_spec[:, :n_keep_spec].shape),
        },
        "class_counts": {
            "y": class_counts(y),
            "y_spec": class_counts(y_spec),
        },
        "pca_features": {
            "variance_explained_total": float(pca.cumulative_variance[-1]),
            "n_components_for_variance": variance_thresholds,
            "variance_summary_top10": safe_df_records(
                pca.variance_summary(), max_rows=10
            ),
            "top_features_pc1": safe_df_records(
                pca.top_features_per_pc(pc_idx=0, n_top=10)
            ),
            "top_features_pc2": safe_df_records(
                pca.top_features_per_pc(pc_idx=1, n_top=10)
            ),
        },
        "pca_spectra": {
            "variance_explained_total": float(pca_spec.cumulative_variance[-1]),
            "n_components_for_variance": variance_thresholds_spec,
            "variance_summary_top10": safe_df_records(
                pca_spec.variance_summary(), max_rows=10
            ),
        },
        "corrélations_pc_gaia_spearman": (
            safe_df_records(corr_df.reset_index()) if corr_df is not None else None
        ),
        "class_séparation": (
            safe_df_records(sep_df.reset_index(drop=True))
            if sep_df is not None
            else None
        ),
        "reconstruction_error": (
            safe_df_records(recon_df.reset_index(drop=True))
            if recon_df is not None
            else None
        ),
        "figures_saved": sorted(str(p) for p in FIGURES_DIR.glob("*.png")),
    }

    # ── Rapport TXT ─────────────────────────────────────────────────────────
    # Le TXT privilégie la lisibilité humaine (enseignant/collègues) avec
    # sections compactes et tableaux principaux directement imprimés.
    # 7) Le TXT reprend les informations clés dans un ordre de lecture narratif.
    text_lines = [
        "AstroSpectro | PHY-3500 PCA Run Report",
        "=" * 72,
        f"Timestamp (UTC)             : {timestamp}",
        f"Features file               : {FEATURES_PATH}",
        f"Catalog file                : {CATALOG_PATH}",
        f"Figures directory           : {FIGURES_DIR}",
        "",
        "[Saved artifacts]",
        f"- Joblib                    : {joblib_run}",
        "",
        "[Shapes]",
        f"- X                         : {X.shape}",
        f"- Scores                    : {scores.shape}",
        f"- X_spec                    : {X_spec.shape}",
        f"- Scores_spec               : {scores_spec.shape}",
        f"- Scores (95% var)          : {scores[:, :n_keep].shape} ({n_keep} PCs)",
        f"- Scores_spec (95% var)     : {scores_spec[:, :n_keep_spec].shape} ({n_keep_spec} PCs)",
        "",
        "[Class counts]",
        f"- y                         : {class_counts(y)}",
        f"- y_spec                    : {class_counts(y_spec)}",
        "",
        "[PCA features]",
        f"- Variance explained        : {pca.cumulative_variance[-1] * 100:.2f}%",
        f"- Components for variance   : {variance_thresholds}",
        "",
        "Variance summary (top 10):",
        pca.variance_summary().head(10).to_string(index=False),
        "",
        "Top features PC1:",
        pca.top_features_per_pc(pc_idx=0, n_top=10).to_string(index=False),
        "",
        "Top features PC2:",
        pca.top_features_per_pc(pc_idx=1, n_top=10).to_string(index=False),
        "",
        "[PCA spectra]",
        f"- Variance explained        : {pca_spec.cumulative_variance[-1] * 100:.2f}%",
        f"- Components for variance   : {variance_thresholds_spec}",
        "",
    ]

    # Ajout conditionnel : seulement si les sections ont été calculées.
    if corr_df is not None:
        text_lines += [
            "Corrélations Spearman (PC × Gaia):",
            corr_df.round(3).to_string(),
            "",
        ]
    if sep_df is not None:
        text_lines += [
            "Séparation classes (résumé):",
            sep_df.round(3).to_string(index=False),
            "",
        ]
    if recon_df is not None:
        text_lines += [
            "Erreur reconstruction (10 premières lignes):",
            recon_df.head(10).round(6).to_string(index=False),
            "",
        ]

    # 8) Écriture finale des deux formats de rapport.
    json_path, _jl, txt_path, _tl = save_json_txt_pair(
        report,
        run_dir,
        stem_run=f"phy3500_pca_run_{timestamp}",
        stem_latest="phy3500_pca_run_latest",
        text_lines=text_lines,
    )

    summary = (
        # Résumé court destiné à l'affichage direct dans le notebook.
        f"Scores PCA sauvegardes -> {joblib_run}\n"
        f"Rapport JSON run        -> {json_path}\n"
        f"Rapport TXT run         -> {txt_path}\n"
        f"  Features (95% var)    : {scores[:, :n_keep].shape} ({n_keep} PCs)\n"
        f"  Spectres (95% var)    : {scores_spec[:, :n_keep_spec].shape} ({n_keep_spec} PCs)"
    )

    return {
        "timestamp": timestamp,
        "run_dir": run_dir,
        "json_path": json_path,
        "txt_path": txt_path,
        "joblib_path": joblib_run,
        "summary": summary,
    }


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — save_umap_tsne_run  (NB02)
# ══════════════════════════════════════════════════════════════════════════════


def save_umap_tsne_run(
    *,
    Z_umap: Optional[np.ndarray],
    Z_tsne: Optional[np.ndarray],
    Z_umap3: Optional[np.ndarray] = None,
    Z_umap_neg: Optional[np.ndarray] = None,
    y,
    meta: Optional[pd.DataFrame],
    scores_95: Optional[np.ndarray],
    umap_engine=None,
    tsne_engine=None,
    paths: dict,
    FIGURES_DIR: Path,
    # Optionnels — sections facultatives du notebook
    pca_output_path: Optional[str] = None,
    cluster_labels=None,
    labels_pres=None,
    clusterer=None,
    y_pred=None,
    confidence=None,
    classes_pred=None,
    y_aligned=None,
    cl_aligned=None,
    fg_mask=None,
    path_xgb: Optional[Path] = None,
    path_fg: Optional[Path] = None,
    axis3_best_param: Optional[str] = None,
    axis3_best_corr: Optional[float] = None,
    path_3d_html: Optional[Path] = None,
    path_3d_teff: Optional[Path] = None,
    path_3d_feh: Optional[Path] = None,
    sensitivity_umap: Optional[dict] = None,
    sensitivity_tsne: Optional[dict] = None,
    stab_umap: Optional[pd.DataFrame] = None,
    stab_tsne: Optional[pd.DataFrame] = None,
    ZOOM_X: Optional[tuple] = None,
    ZOOM_Y: Optional[tuple] = None,
    features_stem: Optional[str] = None,
) -> dict:
    """
    Sauvegarde complète du run UMAP/t-SNE (NB02).

    Tous les paramètres de sections optionnelles (HDBSCAN, XGBoost,
    UMAP-3D, zoom) sont ignorés s'ils sont None — la sauvegarde est
    toujours effectuée même si seul l'embedding UMAP est disponible.

    Stratégie de conception
    -----------------------
    NB02 est modulaire: certaines sections peuvent être sautées selon le
    temps de calcul ou l'objectif pédagogique. Cette fonction est donc
    volontairement tolérante et assemble un rapport partiel cohérent, plutôt
    que d'échouer dès qu'une section optionnelle manque.

    Returns
    -------
    dict avec les clés ``timestamp``, ``run_dir``, ``json_path``,
    ``txt_path``, ``joblib_run``, ``joblib_latest``, ``summary``.
    """
    # Garde-fou minimal : au moins un embedding doit exister.
    if Z_umap is None and Z_tsne is None and Z_umap3 is None:
        raise RuntimeError(
            "Aucun embedding détecté (Z_umap / Z_tsne / Z_umap3). "
            "Exécuter au moins une section d'embedding avant la sauvegarde."
        )

    reports_dir = Path(paths["REPORTS_DIR"])
    # 1) Prépare le contexte de run et les répertoires de sortie.
    timestamp = make_timestamp()
    run_dir = make_run_dir(reports_dir, "phy3500_umap_tsne")
    FIGURES_DIR = Path(FIGURES_DIR)

    # 2) Les temps de fit sont récupérés si les engines exposent ces attributs.
    umap_fit_time = (
        getattr(umap_engine, "fit_time_", None) if umap_engine is not None else None
    )
    tsne_fit_time = (
        getattr(tsne_engine, "fit_time_", None) if tsne_engine is not None else None
    )

    # ── Normalisation étiquettes ─────────────────────────────────────────────
    # Harmonisation des labels textuels (typos/variantes historiques) pour
    # fiabiliser les comptages et comparaisons inter-sections.
    labels_norm = None
    if y is not None:
        labels_norm = np.char.upper(np.asarray(y).astype(str))
        labels_norm = np.char.strip(labels_norm)
        alias_map = {"GLAXAXY": "GALAXY", "GALAXIE": "GALAXY", "QS0": "QSO"}
        for bad, good in alias_map.items():
            labels_norm = np.where(labels_norm == bad, good, labels_norm)

    # ── Résumé zoom (optionnel) ──────────────────────────────────────────────
    # Résume uniquement le nombre de points dans la fenêtre de zoom utilisée
    # dans les figures "pair" du notebook (utile pour interpréter les plots).
    zoom_summary = None
    if (
        labels_norm is not None
        and Z_umap is not None
        and ZOOM_X is not None
        and ZOOM_Y is not None
    ):
        try:
            # Décompacte les bornes pour produire un masque géométrique explicite.
            zx_min, zx_max = map(float, ZOOM_X)
            zy_min, zy_max = map(float, ZOOM_Y)
            zoom_mask = (
                (Z_umap[:, 0] >= zx_min)
                & (Z_umap[:, 0] <= zx_max)
                & (Z_umap[:, 1] >= zy_min)
                & (Z_umap[:, 1] <= zy_max)
            )
            zoom_summary = {
                "zoom_x": [zx_min, zx_max],
                "zoom_y": [zy_min, zy_max],
                "n_points_total": int(zoom_mask.sum()),
                "counts_in_zoom": {
                    "STAR": int(((labels_norm == "STAR") & zoom_mask).sum()),
                    "GALAXY": int(((labels_norm == "GALAXY") & zoom_mask).sum()),
                    "QSO": int(((labels_norm == "QSO") & zoom_mask).sum()),
                },
            }
        except Exception as exc:
            # Une erreur de zoom ne doit jamais bloquer la sauvegarde globale.
            zoom_summary = {"error": str(exc)}

    # ── Résumé HDBSCAN (optionnel) ───────────────────────────────────────────
    # Capture la topologie de clustering sans stocker de structures lourdes.
    hdbscan_summary = None
    if cluster_labels is not None:
        cl = np.asarray(cluster_labels)
        n_total = int(cl.size)
        # Convention HDBSCAN: -1 correspond au bruit (points non assignés).
        n_noise = int((cl == -1).sum())
        hdbscan_summary = {
            "n_points": n_total,
            "n_clusters": int(len(np.unique(cl[cl != -1]))),
            "n_noise": n_noise,
            "pct_noise": float(100.0 * n_noise / max(1, n_total)),
            "largest_clusters": cluster_top_counts(cl, top_n=12),
        }
        if clusterer is not None:
            hdbscan_summary["params"] = {
                "min_cluster_size": getattr(clusterer, "min_cluster_size", None),
                "min_samples": getattr(clusterer, "min_samples", None),
                "metric": getattr(clusterer, "metric", None),
                "cluster_selection_method": getattr(
                    clusterer, "cluster_selection_method", None
                ),
            }
        if labels_pres is not None:
            lp = np.asarray(labels_pres)
            n_noise_p = int((lp == -1).sum())
            hdbscan_summary["presentation_mode"] = {
                "n_clusters": int(len(np.unique(lp[lp != -1]))),
                "n_noise": n_noise_p,
                "pct_noise": float(100.0 * n_noise_p / max(1, lp.size)),
                "largest_clusters": cluster_top_counts(lp, top_n=12),
            }

    # ── Résumé XGBoost (optionnel) ───────────────────────────────────────────
    # Cette section quantifie l'accord supervisé/non supervisé :
    # classes prédites, confiance, exactitude de référence, confusion matrix.
    xgboost_summary = None
    if y_pred is not None:
        yp = np.asarray(y_pred).astype(str)
        xgboost_summary = {
            "n_predictions": int(yp.size),
            "pred_class_counts": class_counts(yp),
            "model_path": str(path_xgb) if path_xgb is not None else None,
        }
        if confidence is not None:
            xgboost_summary["confidence_stats"] = vector_stats(confidence)
        if classes_pred is not None:
            xgboost_summary["classes_pred_list"] = [str(c) for c in classes_pred]
        y_ref = None
        # Priorité à y_aligned (aligné au pont XGBoost) puis fallback labels_norm.
        if y_aligned is not None and len(np.asarray(y_aligned)) == len(yp):
            y_ref = np.asarray(y_aligned).astype(str)
        elif labels_norm is not None and len(labels_norm) == len(yp):
            y_ref = labels_norm
        if y_ref is not None:
            xgboost_summary["accuracy_vs_reference"] = float(np.mean(yp == y_ref))
            # Matrice de confusion sérialisée sous forme index/columns/values.
            cm = pd.crosstab(pd.Series(y_ref, name="true"), pd.Series(yp, name="pred"))
            xgboost_summary["confusion_matrix"] = {
                "index": [str(v) for v in cm.index.tolist()],
                "columns": [str(v) for v in cm.columns.tolist()],
                "values": cm.astype(int).values.tolist(),
            }
        if cl_aligned is not None and len(np.asarray(cl_aligned)) == len(yp):
            xgboost_summary["top3_pred_by_cluster"] = top3_pred_by_cluster(
                cl_aligned, yp
            )
        if fg_mask is not None:
            xgboost_summary["n_pred_F_or_G"] = int(np.asarray(fg_mask).sum())

    # ── Résumé UMAP-3D (optionnel) ───────────────────────────────────────────
    # Conserve les statistiques de l'embedding 3D et les exports HTML liés.
    umap3d_summary = None
    if Z_umap3 is not None:
        umap3d_summary = {
            "embedding_stats": embedding_stats_nd(Z_umap3, max_axes=3),
            "axis3_best_param": (
                str(axis3_best_param) if axis3_best_param is not None else "N/A"
            ),
            "axis3_best_corr": (
                float(axis3_best_corr) if axis3_best_corr is not None else None
            ),
            "html_exports": {
                "umap3d_classes": (
                    str(path_3d_html) if path_3d_html is not None else None
                ),
                "umap3d_teff": str(path_3d_teff) if path_3d_teff is not None else None,
                "umap3d_feh": str(path_3d_feh) if path_3d_feh is not None else None,
            },
        }

    png_files = sorted(str(p) for p in FIGURES_DIR.glob("*.png"))
    html_files = sorted(str(p) for p in FIGURES_DIR.glob("*.html"))

    # Conversion des tableaux de stabilité en listes JSON compatibles.
    stab_umap_records = (
        safe_df_records(stab_umap.reset_index(drop=True))
        if stab_umap is not None
        else None
    )
    stab_tsne_records = (
        safe_df_records(stab_tsne.reset_index(drop=True))
        if stab_tsne is not None
        else None
    )

    # ── Joblib ───────────────────────────────────────────────────────────────
    # Le joblib conserve les objets riches (embeddings + tables) destinés à
    # la réutilisation programmatique dans NB03 et dans les analyses annexes.
    # 3) On sérialise l'état complet utile à la reprise du pipeline.
    embeddings_output = {
        "timestamp_utc": timestamp,
        "Z_umap": Z_umap,
        "Z_tsne": Z_tsne,
        "Z_umap3": Z_umap3,
        "y": y,
        "meta": meta,
        "cluster_labels": cluster_labels,
        "cluster_labels_presentation": labels_pres,
        "xgb_predictions": y_pred,
        "xgb_confidence": confidence,
        "stability_umap": stab_umap,
        "stability_tsne": stab_tsne,
        "umap_params": (
            getattr(umap_engine, "params_used", None)
            if umap_engine is not None
            else None
        ),
        "tsne_params": (
            getattr(tsne_engine, "params_used", None)
            if tsne_engine is not None
            else None
        ),
        "umap_fit_time_s": float(umap_fit_time) if umap_fit_time is not None else None,
        "tsne_fit_time_s": float(tsne_fit_time) if tsne_fit_time is not None else None,
        "Z_umap_negative_control": Z_umap_neg,
        "sensitivity_umap": sensitivity_umap,
        "sensitivity_tsne": sensitivity_tsne,
        "features_stem": features_stem,
        "zoom_x": list(ZOOM_X) if ZOOM_X is not None else None,
        "zoom_y": list(ZOOM_Y) if ZOOM_Y is not None else None,
    }
    legacy_path = reports_dir / "phy3500_embeddings.joblib"
    joblib_run, joblib_latest = save_joblib_pair(
        embeddings_output,
        run_dir,
        stem_run=f"phy3500_embeddings_{timestamp}",
        stem_latest="phy3500_embeddings_latest",
        legacy_path=legacy_path,
    )

    # ── Rapport JSON ─────────────────────────────────────────────────────────
    # Rapport structuré "machine-first" pour audit, script et comparaison.
    # 4) Le JSON est pensé pour être relu sans réexécuter le notebook.
    report = {
        "run_timestamp_utc": timestamp,
        "notebook": "notebooks/dimred/phy3500_02_umap_tsne.ipynb",
        "captured_sections": {
            "has_scores_95": scores_95 is not None,
            "has_Z_umap": Z_umap is not None,
            "has_Z_tsne": Z_tsne is not None,
            "has_Z_umap3": Z_umap3 is not None,
            "has_labels": y is not None,
            "has_meta": meta is not None,
            "has_hdbscan": cluster_labels is not None,
            "has_xgboost": y_pred is not None,
        },
        "paths": {
            "pca_output_path": pca_output_path,
            "figures_dir": str(FIGURES_DIR),
            "embeddings_joblib_path_legacy": str(legacy_path),
            "embeddings_joblib_path_run": str(joblib_run),
            "embeddings_joblib_path_latest": str(joblib_latest),
            "run_dir": str(run_dir),
            "xgboost_prediction_figure": (
                str(path_xgb) if path_xgb is not None else None
            ),
            "xgboost_fg_confusion_figure": (
                str(path_fg) if path_fg is not None else None
            ),
        },
        "shapes": {
            "scores_95": shape_or_none(scores_95),
            "Z_umap": shape_or_none(Z_umap),
            "Z_tsne": shape_or_none(Z_tsne),
            "Z_umap3": shape_or_none(Z_umap3),
            "meta": shape_or_none(meta),
        },
        "class_counts": class_counts(labels_norm) if labels_norm is not None else None,
        "timings_seconds": {
            "umap_fit_time": (
                float(umap_fit_time) if umap_fit_time is not None else None
            ),
            "tsne_fit_time": (
                float(tsne_fit_time) if tsne_fit_time is not None else None
            ),
        },
        "parameters": {
            "umap": (
                getattr(umap_engine, "params_used", None)
                if umap_engine is not None
                else None
            ),
            "tsne": (
                getattr(tsne_engine, "params_used", None)
                if tsne_engine is not None
                else None
            ),
            "zoom_x": list(ZOOM_X) if ZOOM_X is not None else None,
            "zoom_y": list(ZOOM_Y) if ZOOM_Y is not None else None,
        },
        "embedding_stats": {
            "umap": embedding_stats(Z_umap),
            "tsne": embedding_stats(Z_tsne),
            "umap3d": embedding_stats_nd(Z_umap3, max_axes=3),
            "umap_negative_control": embedding_stats(Z_umap_neg),
        },
        "stability": {
            "umap": stab_umap_records,
            "tsne": stab_tsne_records,
            "umap_numeric_means": numeric_means(stab_umap),
            "tsne_numeric_means": numeric_means(stab_tsne),
        },
        "sensitivity": {
            "umap_shapes": sensitivity_shapes(sensitivity_umap),
            "tsne_shapes": sensitivity_shapes(sensitivity_tsne),
        },
        "zoom_pair_summary": zoom_summary,
        "hdbscan_summary": hdbscan_summary,
        "xgboost_summary": xgboost_summary,
        "umap3d_summary": umap3d_summary,
        "artifacts": {
            "png_files": png_files,
            "html_files": html_files,
            "n_png": len(png_files),
            "n_html": len(html_files),
        },
    }

    # ── Rapport TXT ──────────────────────────────────────────────────────────
    # Rapport "humain-first" destiné à la lecture rapide et au support de
    # rédaction du rapport de cours.
    # 5) Même contenu-clé que le JSON, mais organisé en sections narratives.
    text_lines = [
        "AstroSpectro | PHY-3500 UMAP/t-SNE Run Report",
        "=" * 78,
        f"Timestamp (UTC)             : {timestamp}",
        "Notebook                    : notebooks/dimred/phy3500_02_umap_tsne.ipynb",
        f"PCA input file              : {pca_output_path}",
        f"Figures directory           : {FIGURES_DIR}",
        "",
        "[Saved artifacts]",
        f"- Embeddings joblib (legacy): {legacy_path}",
        f"- Embeddings joblib (run)   : {joblib_run}",
        f"- Embeddings joblib (latest): {joblib_latest}",
        "",
        "[Captured sections]",
        f"- has_Z_umap                : {Z_umap  is not None}",
        f"- has_Z_tsne                : {Z_tsne  is not None}",
        f"- has_Z_umap3               : {Z_umap3 is not None}",
        f"- has_hdbscan               : {cluster_labels is not None}",
        f"- has_xgboost               : {y_pred         is not None}",
        "",
        "[Shapes]",
        f"- scores_95                 : {shape_or_none(scores_95)}",
        f"- Z_umap                    : {shape_or_none(Z_umap)}",
        f"- Z_tsne                    : {shape_or_none(Z_tsne)}",
        f"- Z_umap3                   : {shape_or_none(Z_umap3)}",
        f"- meta                      : {shape_or_none(meta)}",
        "",
        "[Class counts]",
        f"- y                         : {class_counts(labels_norm) if labels_norm is not None else 'N/A'}",
        "",
        "[Timing]",
        f"- UMAP fit                  : {fmt_seconds(umap_fit_time)}",
        f"- t-SNE fit                 : {fmt_seconds(tsne_fit_time)}",
        "",
    ]
    # Chaque bloc suivant est ajouté uniquement si les données existent.
    if zoom_summary is not None:
        text_lines += [
            "[Zoom pair summary]",
            f"- zoom_x                    : {zoom_summary.get('zoom_x')}",
            f"- zoom_y                    : {zoom_summary.get('zoom_y')}",
            f"- n_points_total            : {zoom_summary.get('n_points_total')}",
            f"- counts_in_zoom            : {zoom_summary.get('counts_in_zoom')}",
            "",
        ]
    if hdbscan_summary is not None:
        text_lines += [
            "[HDBSCAN summary]",
            f"- n_clusters                : {hdbscan_summary.get('n_clusters')}",
            f"- n_noise                   : {hdbscan_summary.get('n_noise')}",
            f"- pct_noise                 : {hdbscan_summary.get('pct_noise')}",
            f"- largest_clusters          : {hdbscan_summary.get('largest_clusters')}",
            "",
        ]
    if xgboost_summary is not None:
        text_lines += [
            "[XGBoost summary]",
            f"- n_predictions             : {xgboost_summary.get('n_predictions')}",
            f"- pred_class_counts         : {xgboost_summary.get('pred_class_counts')}",
            f"- accuracy_vs_reference     : {xgboost_summary.get('accuracy_vs_reference')}",
            f"- confidence_stats          : {xgboost_summary.get('confidence_stats')}",
            "",
        ]
    if umap3d_summary is not None:
        text_lines += [
            "[UMAP 3D summary]",
            f"- axis3_best_param          : {umap3d_summary.get('axis3_best_param')}",
            f"- axis3_best_corr           : {umap3d_summary.get('axis3_best_corr')}",
            "",
        ]
    if stab_umap is not None:
        text_lines += ["Stabilité UMAP:", stab_umap.round(6).to_string(index=False), ""]
    if stab_tsne is not None:
        text_lines += [
            "Stabilité t-SNE:",
            stab_tsne.round(6).to_string(index=False),
            "",
        ]
    if sensitivity_umap is not None:
        text_lines += [
            "Sensibilité UMAP (shapes):",
            str(sensitivity_shapes(sensitivity_umap)),
            "",
        ]
    if sensitivity_tsne is not None:
        text_lines += [
            "Sensibilité t-SNE (shapes):",
            str(sensitivity_shapes(sensitivity_tsne)),
            "",
        ]
    text_lines += [
        "[Artifacts inventory]",
        f"- PNG count                 : {len(png_files)}",
        f"- HTML count                : {len(html_files)}",
        "",
        "PNG files:",
        *[f"  - {p}" for p in png_files],
        "",
        "HTML files:",
        *[f"  - {p}" for p in html_files],
        "",
    ]

    # 6) Persiste les rapports sur disque (run + latest).
    json_path, _jl, txt_path, _tl = save_json_txt_pair(
        report,
        run_dir,
        stem_run=f"phy3500_umap_tsne_run_{timestamp}",
        stem_latest="phy3500_umap_tsne_run_latest",
        text_lines=text_lines,
    )

    summary = (
        # Résumé court imprimable dans la cellule finale du notebook.
        f"Embeddings sauvegardes      -> {joblib_run}\n"
        f"Rapport JSON run            -> {json_path}\n"
        f"Rapport TXT run             -> {txt_path}\n"
        f"  Z_umap shape              : {shape_or_none(Z_umap)}\n"
        f"  Z_tsne shape              : {shape_or_none(Z_tsne)}\n"
        f"  Figures PNG inventoriées  : {len(png_files)}\n"
        f"  Figures HTML inventoriées : {len(html_files)}"
    )

    return {
        "timestamp": timestamp,
        "run_dir": run_dir,
        "json_path": json_path,
        "txt_path": txt_path,
        "joblib_run": joblib_run,
        "joblib_latest": joblib_latest,
        "summary": summary,
    }


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — save_autoencoder_run  (NB03)
# ══════════════════════════════════════════════════════════════════════════════


def save_autoencoder_run(
    *,
    ae,
    X: np.ndarray,
    y,
    meta: pd.DataFrame,
    feature_names: list,
    paths: dict,
    FEATURES_PATH: Optional[Path] = None,
    CATALOG_PATH: Optional[Path] = None,
    FIGURES_DIR: Path,
    MODELS_DIR: Optional[Path] = None,
    ae_model_path: Optional[Path] = None,
    # Variables optionnelles (sections du notebook)
    Z_ae: Optional[np.ndarray] = None,
    X_recon: Optional[np.ndarray] = None,
    history: Optional[dict] = None,
    comparison_df: Optional[pd.DataFrame] = None,
    summary_df: Optional[pd.DataFrame] = None,  # "summary" dans le notebook
    df_summary: Optional[pd.DataFrame] = None,
    scores_pca: Optional[np.ndarray] = None,
    pca=None,
    n_pcs_95: Optional[int] = None,
    Z_umap: Optional[np.ndarray] = None,
    Z_tsne: Optional[np.ndarray] = None,
    Z_interp: Optional[np.ndarray] = None,
    X_interp: Optional[np.ndarray] = None,
    idx_cold: Optional[int] = None,
    idx_hot: Optional[int] = None,
    teff_cold: Optional[float] = None,
    teff_hot: Optional[float] = None,
    corr_pca: Optional[pd.DataFrame] = None,
) -> dict:
    """
    Sauvegarde complète du run autoencodeur (NB03).

    Gère la résolution des variables optionnelles (history, Z_ae, X_recon)
    depuis l'objet ``ae`` si elles ne sont pas fournies explicitement.

    Objectif scientifique
    --------------------
    NB03 compare plusieurs axes d'analyse (reconstruction, corrélations
    physiques, comparaison PCA, interpolation latente). La fonction assemble
    ces résultats hétérogènes dans un format unique et traçable.

    Returns
    -------
    dict avec les clés ``timestamp``, ``run_dir``, ``json_path``,
    ``txt_path``, ``joblib_path``, ``summary``.
    """
    reports_dir = Path(paths["REPORTS_DIR"])
    # 1) Prépare les chemins de sortie et l'identifiant temporel du run.
    timestamp = make_timestamp()
    run_dir = make_run_dir(reports_dir, "phy3500_autoencoder")
    FIGURES_DIR = Path(FIGURES_DIR)

    # ── Résolution des variables depuis l'objet ae ───────────────────────────
    # Si le notebook ne passe pas explicitement certaines variables, on tente
    # de les reconstruire depuis l'état interne de l'objet autoencodeur.
    # Concrètement: on garantit que Z_ae_current existe toujours.
    Z_ae_current = Z_ae if Z_ae is not None else ae.encode(X)
    # X_recon peut rester None si la reconstruction n'a pas été demandée.
    X_recon_current = X_recon if X_recon is not None else None
    # history_current est pris depuis l'argument, sinon depuis ae.history_.
    history_current = history if history is not None else getattr(ae, "history_", None)

    # Extraction robuste des courbes d'apprentissage (peut être vide).
    train_loss = (
        history_current.get("train_loss", [])
        if isinstance(history_current, dict)
        else []
    )
    val_loss = (
        history_current.get("val_loss", []) if isinstance(history_current, dict) else []
    )

    # 2) Métriques de synthèse directement dérivées des courbes.
    ae_fit_time = getattr(ae, "fit_time_", None)
    epochs_done = len(train_loss) if isinstance(train_loss, list) else None
    best_val_loss = float(min(val_loss)) if len(val_loss) > 0 else None
    final_train_loss = float(train_loss[-1]) if len(train_loss) > 0 else None
    final_val_loss = float(val_loss[-1]) if len(val_loss) > 0 else None

    try:
        # MSE globale calculée via l'API de l'autoencodeur.
        mse_ae_global = float(ae.reconstruction_mse(X))
    except Exception:
        # La sauvegarde reste possible même si cette métrique échoue.
        mse_ae_global = None

    # ── MSE PCA de référence ─────────────────────────────────────────────────
    # Ces métriques servent de baseline linéaire pour situer les performances
    # de reconstruction de l'autoencodeur.
    n_pcs_95_current = n_pcs_95
    mse_pca2 = mse_pca95 = None
    if pca is not None:
        try:
            # Baseline PCA à 2 composantes (référence visuelle classique).
            mse_pca2 = float(pca.reconstruction_error(X, n_components=2).mean())
            if n_pcs_95_current is None:
                # Détermine automatiquement le nombre de composantes à 95%.
                n_pcs_95_current = int(pca.n_components_for_variance(0.95))
            # Baseline PCA "équitable" au seuil 95% de variance.
            mse_pca95 = float(
                pca.reconstruction_error(X, n_components=n_pcs_95_current).mean()
            )
        except Exception:
            pass

    # ── Corrélations Spearman espace latent ──────────────────────────────────
    # Évalue la cohérence physique des axes latents avec les paramètres Gaia.
    # Spearman est privilégié pour sa robustesse aux non-linéarités monotones
    # et aux distributions non gaussiennes.
    phys_cols = ["teff_gspphot", "logg_gspphot", "mh_gspphot", "bp_rp"]
    ae_correlations: dict = {}
    for col in phys_cols:
        # Ignore les colonnes absentes pour supporter des catalogues partiels.
        if col not in meta.columns:
            continue
        # Conversion numérique robuste pour éviter les erreurs sur chaînes/NaN.
        vals = pd.to_numeric(meta[col], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(vals)
        if valid.sum() < 3:
            # Trop peu de points valides: corrélation non interprétable.
            ae_correlations[col] = {"latent_axis_1": None, "latent_axis_2": None}
            continue
        # Corrélation Spearman axe latent 1 <-> paramètre physique.
        r1 = pd.Series(Z_ae_current[valid, 0]).corr(
            pd.Series(vals[valid]), method="spearman"
        )
        # Corrélation Spearman axe latent 2 <-> paramètre physique.
        r2 = pd.Series(Z_ae_current[valid, 1]).corr(
            pd.Series(vals[valid]), method="spearman"
        )
        ae_correlations[col] = {
            "latent_axis_1": float(r1) if pd.notna(r1) else None,
            "latent_axis_2": float(r2) if pd.notna(r2) else None,
        }

    # ── Distance d'interpolation latente ────────────────────────────────────
    # Mesure simple de l'amplitude du trajet entre les deux pôles choisis
    # (étoile froide -> étoile chaude) dans l'espace latent.
    latent_distance = None
    if Z_interp is not None:
        zi = np.asarray(Z_interp)
        if zi.ndim == 2 and zi.shape[0] >= 2:
            # Distance euclidienne entre le début et la fin de l'interpolation.
            latent_distance = float(np.linalg.norm(zi[-1] - zi[0]))

    # ── Joblib ───────────────────────────────────────────────────────────────
    # Archive riche des sorties NB03, réutilisable pour post-analyse ou
    # génération d'illustrations hors notebook.
    # 3) On regroupe toutes les sorties utiles dans une seule structure persistée.
    autoencoder_output = {
        "Z_ae": Z_ae_current,
        "X_recon": X_recon_current,
        "y": y,
        "meta": meta,
        "feature_names": feature_names,
        "history": history_current,
        "comparison_df": comparison_df,
        "reconstruction_summary": summary_df,
        "method_summary": df_summary,
        "scores_pca": scores_pca,
        "Z_umap": Z_umap,
        "Z_tsne": Z_tsne,
        "Z_interp": Z_interp,
        "X_interp": X_interp,
        "idx_cold": int(idx_cold) if idx_cold is not None else None,
        "idx_hot": int(idx_hot) if idx_hot is not None else None,
        "teff_cold": float(teff_cold) if teff_cold is not None else None,
        "teff_hot": float(teff_hot) if teff_hot is not None else None,
        "n_pcs_95": n_pcs_95_current,
        "ae_fit_time_s": float(ae_fit_time) if ae_fit_time is not None else None,
        "ae_mse_global": mse_ae_global,
    }
    legacy_path = reports_dir / "phy3500_autoencoder_output.joblib"
    joblib_run, _joblib_latest = save_joblib_pair(
        autoencoder_output,
        run_dir,
        stem_run=f"phy3500_autoencoder_run_{timestamp}",
        stem_latest="phy3500_autoencoder_run_latest",
        legacy_path=legacy_path,
    )

    # ── Rapport JSON ─────────────────────────────────────────────────────────
    # Rapport structuré englobant métriques, corrélations, tables et traces
    # d'interpolation, prêt pour exploitation programmatique.
    # 4) Le JSON sert de trace complète de l'expérience NB03.
    report = {
        "run_timestamp_utc": timestamp,
        "paths": {
            "features_path": str(FEATURES_PATH) if FEATURES_PATH is not None else None,
            "catalog_path": str(CATALOG_PATH) if CATALOG_PATH is not None else None,
            "figures_dir": str(FIGURES_DIR),
            "models_dir": str(MODELS_DIR) if MODELS_DIR is not None else None,
            "ae_model_path": str(ae_model_path) if ae_model_path is not None else None,
            "autoencoder_joblib_path": str(joblib_run),
            "run_dir": str(run_dir),
        },
        "shapes": {
            "X": list(np.asarray(X).shape),
            "Z_ae": list(np.asarray(Z_ae_current).shape),
            "X_recon": (
                list(np.asarray(X_recon_current).shape)
                if X_recon_current is not None
                else None
            ),
            "scores_pca": (
                list(np.asarray(scores_pca).shape) if scores_pca is not None else None
            ),
            "meta": list(meta.shape),
        },
        "class_counts": class_counts(y),
        "metrics": {
            "ae_fit_time_s": float(ae_fit_time) if ae_fit_time is not None else None,
            "ae_epochs_done": int(epochs_done) if epochs_done is not None else None,
            "ae_best_val_loss": best_val_loss,
            "ae_final_train_loss": final_train_loss,
            "ae_final_val_loss": final_val_loss,
            "ae_mse_global": mse_ae_global,
            "pca_mse_2_components": mse_pca2,
            "pca_mse_n_components_95": mse_pca95,
            "n_pcs_95": n_pcs_95_current,
        },
        "correlations": {
            "ae_latent_vs_physical_spearman": ae_correlations,
            "pca_corr_table": (
                safe_df_records(corr_pca.reset_index())
                if corr_pca is not None
                else None
            ),
        },
        "tables": {
            "comparison_df": (
                safe_df_records(comparison_df) if comparison_df is not None else None
            ),
            "reconstruction_summary": (
                safe_df_records(summary_df) if summary_df is not None else None
            ),
            "method_summary": (
                safe_df_records(
                    df_summary.reset_index().rename(columns={"index": "method"})
                )
                if df_summary is not None
                else None
            ),
        },
        "interpolation": {
            "idx_cold": int(idx_cold) if idx_cold is not None else None,
            "idx_hot": int(idx_hot) if idx_hot is not None else None,
            "teff_cold": float(teff_cold) if teff_cold is not None else None,
            "teff_hot": float(teff_hot) if teff_hot is not None else None,
            "latent_distance": latent_distance,
        },
        "embeddings": {
            "umap_available": Z_umap is not None,
            "tsne_available": Z_tsne is not None,
            "umap_stats": embedding_stats(Z_umap),
            "tsne_stats": embedding_stats(Z_tsne),
            "ae_stats": embedding_stats(Z_ae_current),
        },
        "figures_saved": sorted(str(p) for p in FIGURES_DIR.glob("*.png")),
    }

    # ── Rapport TXT ──────────────────────────────────────────────────────────
    # Version lecture humaine du run (synthèse orientée rapport scientifique).
    # 5) Le TXT condense les résultats clés pour une lecture rapide.
    text_lines = [
        "AstroSpectro | PHY-3500 Autoencoder Run Report",
        "=" * 72,
        f"Timestamp (UTC)             : {timestamp}",
        f"Features file               : {FEATURES_PATH if FEATURES_PATH is not None else 'N/A'}",
        f"Figures directory           : {FIGURES_DIR}",
        "",
        "[Saved artifacts]",
        f"- Autoencoder joblib        : {joblib_run}",
        "",
        "[Shapes]",
        f"- X                         : {np.asarray(X).shape}",
        f"- Z_ae                      : {np.asarray(Z_ae_current).shape}",
        f"- X_recon                   : {np.asarray(X_recon_current).shape if X_recon_current is not None else 'N/A'}",
        "",
        "[Class counts]",
        f"- y                         : {class_counts(y)}",
        "",
        "[Metrics]",
        f"- AE fit time               : {fmt_seconds(ae_fit_time)}",
        f"- AE epochs                 : {epochs_done if epochs_done is not None else 'N/A'}",
        f"- AE best val_loss          : {best_val_loss if best_val_loss is not None else 'N/A'}",
        f"- AE final train_loss       : {final_train_loss if final_train_loss is not None else 'N/A'}",
        f"- AE final val_loss         : {final_val_loss if final_val_loss is not None else 'N/A'}",
        f"- AE MSE global             : {mse_ae_global if mse_ae_global is not None else 'N/A'}",
        f"- PCA MSE (2)               : {mse_pca2 if mse_pca2 is not None else 'N/A'}",
        f"- PCA MSE (n95)             : {mse_pca95 if mse_pca95 is not None else 'N/A'}",
        f"- n_pcs_95                  : {n_pcs_95_current if n_pcs_95_current is not None else 'N/A'}",
        "",
        "[AE latent correlations — Spearman]",
        str(ae_correlations),
        "",
    ]

    # Les tables détaillées sont ajoutées seulement si disponibles.
    if comparison_df is not None:
        text_lines += [
            "Comparison AE vs PCA:",
            comparison_df.round(6).to_string(index=False),
            "",
        ]
    if summary_df is not None:
        text_lines += [
            "Reconstruction summary by class:",
            summary_df.round(6).to_string(index=False),
            "",
        ]
    if df_summary is not None:
        text_lines += [
            "Method summary table:",
            df_summary.to_string(),
            "",
        ]
    if latent_distance is not None:
        text_lines += [
            "Interpolation:",
            f"- idx_cold                  : {idx_cold if idx_cold is not None else 'N/A'}",
            f"- idx_hot                   : {idx_hot  if idx_hot  is not None else 'N/A'}",
            f"- latent_distance           : {latent_distance:.6f}",
            "",
        ]

    # 6) Écriture finale des rapports machine (JSON) et humain (TXT).
    json_path, _jl, txt_path, _tl = save_json_txt_pair(
        report,
        run_dir,
        stem_run=f"phy3500_autoencoder_run_{timestamp}",
        stem_latest="phy3500_autoencoder_run_latest",
        text_lines=text_lines,
    )

    summary = (
        # Résumé compact retourné pour affichage terminal/notebook.
        f"Autoencoder output sauvegarde -> {joblib_run}\n"
        f"Rapport JSON run              -> {json_path}\n"
        f"Rapport TXT run               -> {txt_path}\n"
        f"  Z_ae shape                  : {np.asarray(Z_ae_current).shape}"
    )

    return {
        "timestamp": timestamp,
        "run_dir": run_dir,
        "json_path": json_path,
        "txt_path": txt_path,
        "joblib_path": joblib_run,
        "summary": summary,
    }
