"""
AstroSpectro — dimred.hdbscan_analyzer
========================================

Analyse non-supervisée par densité (HDBSCAN) sur un embedding UMAP.

Ce module encapsule la logique qui était dans les cellules 12, 13 et 16
du notebook phy3500_02_umap_tsne.ipynb.

Responsabilités
---------------
- `HDBSCANAnalyzer` : ajustement, palette, centroïdes, profil physique,
  interprétation astrophysique, version présentation.
- `compute_feature_profiles()` : chargement des features CSV et calcul
  des profils moyens standardisés par cluster.

Usage
-----
>>> from dimred.hdbscan_analyzer import HDBSCANAnalyzer, compute_feature_profiles
>>> hdb = HDBSCANAnalyzer(min_cluster_size=75, min_samples=20)
>>> hdb.fit(Z_umap)
>>> df_table, df_interp = hdb.physical_profile_table(meta, y)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

# Colonnes physiques standard Gaia DR3
_PHYS_COLS = [
    ("teff_gspphot", "T_eff (K)"),
    ("logg_gspphot", "log g (dex)"),
    ("mh_gspphot", "[Fe/H]"),
    ("bp_rp", "G_BP-G_RP"),
    ("phot_g_mean_mag", "G mag"),
    ("distance_gspphot", "Distance (pc)"),
]


class HDBSCANAnalyzer:
    """
    Encapsule HDBSCAN pour l'analyse des populations de l'espace UMAP.

    Parameters
    ----------
    min_cluster_size : int
        Taille minimale d'un cluster HDBSCAN.
    min_samples : int
        Nombre de voisins requis pour un point cœur.
    metric : str
        Métrique de distance (défaut : 'euclidean').
    cluster_selection_method : str
        Méthode de sélection ('eom' ou 'leaf').
    """

    def __init__(
        self,
        min_cluster_size: int = 75,
        min_samples: int = 20,
        metric: str = "euclidean",
        cluster_selection_method: str = "eom",
    ) -> None:
        # Hyperparamètres "analyse fine" conservés comme état de l'analyseur.
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.metric = metric
        self.cluster_selection_method = cluster_selection_method

        # Attributs produits par fit() (version analyse standard).
        self.clusterer_ = None
        self.labels_: Optional[np.ndarray] = None
        self.n_clusters_: int = 0
        self.n_noise_: int = 0
        self.cluster_ids_: List[int] = []
        self.color_map_: Dict[int, tuple] = {}

        # Version présentation (min_cluster_size plus grand)
        self.clusterer_pres_ = None
        self.labels_pres_: Optional[np.ndarray] = None
        self.n_clusters_pres_: int = 0
        self.color_map_pres_: Dict[int, tuple] = {}
        self.ids_pres_: List[int] = []

    # ── Ajustement ──────────────────────────────────────────────────────────

    def fit(self, Z: np.ndarray) -> "HDBSCANAnalyzer":
        """
        Ajuste HDBSCAN sur l'embedding 2D Z.

        Parameters
        ----------
        Z : np.ndarray (N, 2)
            Coordonnées UMAP (ou tout embedding 2D).

        Returns
        -------
        self
        """
        try:
            import hdbscan
        except ImportError:
            raise ImportError("HDBSCAN non installé → pip install hdbscan")

        # prediction_data=True garde les structures utiles à la post-analyse HDBSCAN.
        self.clusterer_ = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric=self.metric,
            cluster_selection_method=self.cluster_selection_method,
            prediction_data=True,
        )
        # fit_predict renvoie -1 pour les points bruit (non assignés à un cluster dense).
        self.labels_ = self.clusterer_.fit_predict(Z)
        # Statistiques agrégées et palette dérivées des labels obtenus.
        self._compute_stats()
        self._build_palette()

        logger.info(
            "HDBSCAN ajusté : %d clusters, %d points bruit (%.1f%%)",
            self.n_clusters_,
            self.n_noise_,
            100 * self.n_noise_ / max(1, len(Z)),
        )
        return self

    def fit_presentation(
        self, Z: np.ndarray, min_cluster_size: int = 300
    ) -> np.ndarray:
        """
        Ajuste une version «présentation» avec un min_cluster_size plus grand.

        Returns
        -------
        labels_pres : np.ndarray — étiquettes de clusters (version présentation).
        """
        try:
            import hdbscan
        except ImportError:
            raise ImportError("HDBSCAN non installé → pip install hdbscan")

        # Variante "présentation": granularité plus grossière via min_cluster_size élevé.
        self.clusterer_pres_ = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=self.min_samples,
            metric=self.metric,
            cluster_selection_method=self.cluster_selection_method,
        )
        self.labels_pres_ = self.clusterer_pres_.fit_predict(Z)
        # IDs triés sans bruit (-1) pour construire une légende stable.
        self.ids_pres_ = sorted(set(self.labels_pres_) - {-1})
        self.n_clusters_pres_ = len(self.ids_pres_)

        cmap_p = plt.get_cmap("tab20", self.n_clusters_pres_)
        self.color_map_pres_ = {cid: cmap_p(i) for i, cid in enumerate(self.ids_pres_)}
        # Couleur dédiée au bruit, volontairement désaturée.
        self.color_map_pres_[-1] = (0.82, 0.82, 0.82, 0.25)

        logger.info(
            "Version présentation : %d clusters · %d bruit (%.1f%%)",
            self.n_clusters_pres_,
            int((self.labels_pres_ == -1).sum()),
            100 * (self.labels_pres_ == -1).sum() / max(1, len(Z)),
        )
        return self.labels_pres_

    # ── Analyse physique ────────────────────────────────────────────────────

    def physical_profile_table(
        self,
        meta: pd.DataFrame,
        y: np.ndarray,
        phys_cols: Optional[List[Tuple[str, str]]] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Construit le tableau de profil physique par cluster.

        Parameters
        ----------
        meta : pd.DataFrame
            Paramètres Gaia DR3 alignés avec les étiquettes HDBSCAN.
        y : np.ndarray
            Étiquettes de classe LAMOST.
        phys_cols : list[(col, label)] | None
            Colonnes à inclure (défaut : `_PHYS_COLS`).

        Returns
        -------
        (df_table, df_interp) :
          - df_table  : profil moyen ± std par cluster
          - df_interp : interprétation astrophysique par cluster
        """
        if self.labels_ is None:
            raise RuntimeError("Appeler fit() avant physical_profile_table().")

        phys_cols = phys_cols or _PHYS_COLS
        # Restreint l'analyse aux colonnes réellement présentes dans meta.
        available = [
            (col_name, label_name)
            for col_name, label_name in phys_cols
            if col_name in meta.columns
        ]
        col_names = [c for c, _ in available]

        # Table de travail alignée sur les labels HDBSCAN et les classes LAMOST.
        df_profile = meta.copy().reset_index(drop=True)
        df_profile["cluster"] = self.labels_
        df_profile["class_lamost"] = y

        # Exclut le bruit pour le résumé principal des populations structurées.
        df_clusters = df_profile[df_profile["cluster"] != -1].copy()
        # MultiIndex (mean/std) aplati ensuite pour des accès plus simples par nom de colonne.
        summary = df_clusters.groupby("cluster")[col_names].agg(["mean", "std"])
        summary.columns = ["_".join(c) for c in summary.columns]
        # Comptage et classe dominante par cluster pour contexte astrophysique rapide.
        counts = df_clusters["cluster"].value_counts().sort_index()
        dominant = (
            df_clusters.groupby("cluster")["class_lamost"]
            .apply(lambda x: x.value_counts().index[0])
            .rename("Classe dominante")
        )

        # Tableau 1 : profil physique
        rows = []
        for cid in sorted(df_clusters["cluster"].unique()):
            row = {
                "Cluster": f"C{cid}",
                "n spectres": int(counts.get(cid, 0)),
                "Classe dom.": dominant.get(cid, "?"),
            }
            for col, label in available:
                m_col, s_col = f"{col}_mean", f"{col}_std"
                if m_col in summary.columns:
                    mu = summary.loc[cid, m_col]
                    sig = summary.loc[cid, s_col]
                    # Format différent pour T_eff (échelle absolue en K) vs autres paramètres.
                    row[label] = (
                        f"{mu:.0f} ± {sig:.0f}"
                        if col == "teff_gspphot"
                        else f"{mu:.2f} ± {sig:.2f}"
                    )
                else:
                    row[label] = "N/A"
            rows.append(row)

        # Ligne bruit
        df_noise = df_profile[df_profile["cluster"] == -1]
        row_noise = {
            "Cluster": "Bruit",
            "n spectres": len(df_noise),
            "Classe dom.": "—",
        }
        for col, label in available:
            if col in df_noise.columns:
                vals = df_noise[col].dropna()
                if len(vals) > 0:
                    # Résumé du bruit en moyenne simple (sans dispersion) pour rester compact.
                    row_noise[label] = (
                        f"{vals.mean():.0f}"
                        if col == "teff_gspphot"
                        else f"{vals.mean():.2f}"
                    )
                else:
                    row_noise[label] = "N/A"
        rows.append(row_noise)
        df_table = pd.DataFrame(rows).set_index("Cluster")

        # Tableau 2 : interprétation astrophysique
        interp_rows = []
        for cid in sorted(df_clusters["cluster"].unique()):
            teff_mu = (
                summary.loc[cid, "teff_gspphot_mean"]
                if "teff_gspphot_mean" in summary.columns
                else None
            )
            logg_mu = (
                summary.loc[cid, "logg_gspphot_mean"]
                if "logg_gspphot_mean" in summary.columns
                else None
            )
            mh_mu = (
                summary.loc[cid, "mh_gspphot_mean"]
                if "mh_gspphot_mean" in summary.columns
                else None
            )
            n = int(counts.get(cid, 0))
            desc = []
            if teff_mu is not None and not pd.isna(teff_mu):
                # Règles heuristiques Teff -> familles spectrales approximatives.
                if teff_mu > 7500:
                    desc.append("étoiles chaudes (A–F)")
                elif teff_mu > 5500:
                    desc.append("étoiles de type solaire (F–G)")
                elif teff_mu > 4000:
                    desc.append("étoiles froides (G–K)")
                else:
                    desc.append("étoiles très froides (K–M)")
            if logg_mu is not None and not pd.isna(logg_mu):
                # Gravité de surface -> classe de luminosité (naines/géantes/supergéantes).
                if logg_mu < 2.5:
                    desc.append("supergéantes")
                elif logg_mu < 3.5:
                    desc.append("géantes")
                else:
                    desc.append("naines (séquence principale)")
            if mh_mu is not None and not pd.isna(mh_mu):
                # Métallicité moyenne -> indication population stellaire galactique.
                if mh_mu < -0.8:
                    desc.append("[Fe/H] très faible — pop. vieille/halo")
                elif mh_mu < -0.3:
                    desc.append("[Fe/H] faible — pop. II")
                else:
                    desc.append("[Fe/H] solaire — pop. I")
            interp_rows.append(
                {
                    "Cluster": f"C{cid}",
                    "n": n,
                    "Interprétation": (
                        ", ".join(desc) if desc else "paramètres Gaia insuffisants"
                    ),
                }
            )
        interp_rows.append(
            {
                "Cluster": "Bruit",
                "n": len(df_noise),
                "Interprétation": "anomalies spectrales, objets atypiques",
            }
        )
        df_interp = pd.DataFrame(interp_rows).set_index("Cluster")
        return df_table, df_interp

    def print_physical_profile(
        self, df_table: pd.DataFrame, df_interp: pd.DataFrame
    ) -> None:
        """Affiche les tableaux de profil physique et d'interprétation."""
        # Impression console structurée pour insertion directe dans notes/rapport.
        print("=" * 80)
        print("  PROFIL PHYSIQUE DES POPULATIONS DÉCOUVERTES PAR HDBSCAN")
        print("  (paramètres Gaia DR3, moyenne ± écart-type par cluster)")
        print("=" * 80)
        print(df_table.to_string())
        print("=" * 80)
        print("\nInterprétation astrophysique automatique :")
        print("-" * 50)
        for cid, row in df_interp.iterrows():
            print(f"  {cid:>6s} (n={row['n']:5d}) : {row['Interprétation']}")

    # ── Helpers internes ────────────────────────────────────────────────────

    def centroid(self, Z: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Centroïde d'un cluster dans l'espace 2D."""
        # Barycentre euclidien simple des points du cluster sélectionné.
        return Z[mask].mean(axis=0)

    def _compute_stats(self) -> None:
        # Exclut -1 (bruit) pour les compteurs de clusters astrophysiquement interprétables.
        self.cluster_ids_ = sorted(set(self.labels_) - {-1})
        self.n_clusters_ = len(self.cluster_ids_)
        self.n_noise_ = int((self.labels_ == -1).sum())

    def _build_palette(self) -> None:
        # Palette continue turbo puis mapping cluster_id -> couleur.
        cmap = plt.get_cmap("turbo", max(1, self.n_clusters_))
        self.color_map_ = {
            cid: cmap(i / max(1, self.n_clusters_ - 1))
            for i, cid in enumerate(self.cluster_ids_)
        }
        # Bruit en gris transparent pour rester en arrière-plan des figures.
        self.color_map_[-1] = (0.82, 0.82, 0.82, 0.25)

    @property
    def sensitivity_df(self) -> pd.DataFrame:
        """Retourne un DataFrame vide — calculé par compute_sensitivity()."""
        return pd.DataFrame()


# ── Fonctions de niveau module ───────────────────────────────────────────────


def compute_sensitivity(
    Z: np.ndarray,
    min_samples: int = 20,
    min_sizes: Optional[List[int]] = None,
) -> pd.DataFrame:
    """
    Calcule la sensibilité du nombre de clusters à min_cluster_size.

    Returns
    -------
    pd.DataFrame avec colonnes [min_cluster_size, n_clusters, n_noise, pct_noise].
    """
    try:
        import hdbscan as _hdbscan
    except ImportError:
        raise ImportError("HDBSCAN non installé → pip install hdbscan")

    if min_sizes is None:
        # Grille typique du notebook pour comparer finesse vs robustesse des clusters.
        min_sizes = [50, 100, 150, 300, 500]

    rows = []
    for mcs in min_sizes:
        # Refit complet pour chaque valeur de min_cluster_size (analyse de sensibilité).
        cl = _hdbscan.HDBSCAN(
            min_cluster_size=mcs,
            min_samples=min_samples,
            metric="euclidean",
            cluster_selection_method="eom",
        ).fit_predict(Z)
        # Convention HDBSCAN: -1 = bruit, exclu du comptage de clusters.
        nc = len(set(cl)) - (1 if -1 in cl else 0)
        nn = int((cl == -1).sum())
        rows.append(
            {
                "min_cluster_size": mcs,
                "n_clusters": nc,
                "n_noise": nn,
                "pct_noise": round(100 * nn / len(cl), 2),
            }
        )
    return pd.DataFrame(rows)


def compute_feature_profiles(
    paths: dict,
    features_stem: str,
    Z_umap: np.ndarray,
    cluster_labels: np.ndarray,
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """
    Charge le CSV de features, applique les mêmes filtres que DimRedDataLoader
    et calcule les profils standardisés moyens par cluster HDBSCAN.

    Provient de la cellule 16 du notebook phy3500_02_umap_tsne.ipynb.

    Parameters
    ----------
    paths           : dict de chemins du projet
    features_stem   : identifiant du fichier features (ex. 'features_20260305T174836Z')
    Z_umap          : embedding UMAP (N, 2) — pour vérification d'alignement
    cluster_labels  : étiquettes HDBSCAN (N,)

    Returns
    -------
    (df_feat, cluster_means, cluster_labels_aligned)
    - df_feat               : DataFrame standardisé des features (features seulement)
    - cluster_means         : DataFrame (clusters × features) — profils moyens
    - cluster_labels_aligned: étiquettes alignées sur df_feat
    """
    from sklearn.preprocessing import StandardScaler

    try:
        # Import via package installé (cas normal dans le projet).
        from dimred.data_loader import (
            _GAIA_META_COLS,
            _SNR_COLS,
            _INSTRUMENTAL_COLS,
            _NON_SPECTRO_PREFIXES,
        )
    except ImportError:
        # Fallback notebook/exec locale si le package dimred n'est pas résolu.
        from data_loader import (
            _GAIA_META_COLS,
            _SNR_COLS,
            _INSTRUMENTAL_COLS,
            _NON_SPECTRO_PREFIXES,
        )

    features_csv = Path(paths["PROCESSED_DIR"]) / f"{features_stem}.csv"
    if not features_csv.exists():
        raise FileNotFoundError(
            f"Fichier features introuvable : {features_csv}\n"
            "Vérifier paths['PROCESSED_DIR'] et features_stem."
        )

    df_raw = pd.read_csv(features_csv, low_memory=False)
    if "snr_r" in df_raw.columns:
        # Même filtre qualité que le loader principal pour cohérence inter-modules.
        df_filt = df_raw[df_raw["snr_r"] >= 10.0].copy()
    else:
        df_filt = df_raw.copy()

    EXCLUDE = {
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
        "label",
        "main_class",
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
        "source_id",
        "gaia_ra",
        "gaia_dec",
    }
    EXCLUDE |= set(_GAIA_META_COLS) | set(_SNR_COLS) | set(_INSTRUMENTAL_COLS)

    # Colonnes retenues: features spectrales numériques, informatives, peu manquantes.
    feat_cols = [
        c
        for c in df_filt.columns
        if c not in EXCLUDE
        and not c.startswith(_NON_SPECTRO_PREFIXES)
        and pd.api.types.is_numeric_dtype(df_filt[c])
        and df_filt[c].nunique() > 1
        and df_filt[c].isna().mean() <= 0.10
    ]
    df_feat = df_filt[feat_cols].dropna().reset_index(drop=True)
    logger.info("Features filtrées : %s", df_feat.shape)

    # Alignement
    n_umap = len(Z_umap)
    if len(df_feat) != n_umap:
        logger.warning(
            "Désalignement : features=%d, Z_umap=%d → troncature.", len(df_feat), n_umap
        )
        # Troncature conservative: garde l'intersection commune dans l'ordre courant.
        n_common = min(len(df_feat), n_umap)
        df_feat = df_feat.iloc[:n_common]
        cluster_labels_aligned = cluster_labels[:n_common]
    else:
        cluster_labels_aligned = cluster_labels
        logger.info("Alignement parfait : %d spectres", n_umap)

    # Standardisation
    # Passage en z-score pour comparer les contributions des features sur une même échelle.
    scaler = StandardScaler()
    X_std = scaler.fit_transform(df_feat.values.astype(float))
    X_std_df = pd.DataFrame(X_std, columns=feat_cols)
    X_std_df["cluster"] = cluster_labels_aligned

    # Moyenne par cluster (hors bruit) = signature spectrale moyenne standardisée.
    cluster_means = (
        X_std_df[X_std_df["cluster"] != -1].groupby("cluster")[feat_cols].mean()
    )
    logger.info(
        "Profils moyens calculés : %s (clusters × features)", cluster_means.shape
    )
    return df_feat, cluster_means, cluster_labels_aligned
