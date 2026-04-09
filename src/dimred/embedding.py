"""
AstroSpectro — dimred.embedding
=================================

Wrappers reproductibles pour les embeddings non-linéaires 2D/3D :
  - UMAP  (McInnes et al. 2018) — paramétrique, topologie globale préservée
  - t-SNE (van der Maaten & Hinton 2008) — clusters locaux, non paramétrique

Principe d'usage (PHY-3500)
----------------------------
On applique ces méthodes **sur les scores PCA** (et non sur X brut) pour :
  1. Réduire le bruit (whitening implicite via PCA).
  2. Accélérer le calcul (on passe de D=71-3000 à K=10-30 avant UMAP/t-SNE).
  3. Stabiliser les résultats (moins sensible aux hyperparamètres).

Références
----------
- McInnes L., Healy J., Melville J. (2018). UMAP: Uniform Manifold
  Approximation and Projection. arXiv:1802.03426.
- van der Maaten L., Hinton G. (2008). Visualizing Data using t-SNE.
  JMLR 9:2579–2605.
- Wattenberg M. et al. (2016). How to Use t-SNE Effectively. Distill.
  https://distill.pub/2016/misread-tsne/

Avertissements
--------------
- Les distances absolues dans un embedding t-SNE ne sont PAS interprétables.
- La densité relative des clusters UMAP est partiellement préservée.
- Toujours comparer plusieurs seeds pour valider la stabilité qualitative.

Exemple
-------
>>> engine = EmbeddingEngine(method="umap", n_components=2, random_state=42)
>>> Z = engine.fit_transform(pca_scores[:, :20])
>>> engine.stability_report(pca_scores[:, :20], n_seeds=5)
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Literal, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class EmbeddingEngine:
    """
    Wrapper unifié pour UMAP et t-SNE avec gestion de la reproductibilité.

    Parameters
    ----------
    method : 'umap' | 'tsne'
        Algorithme d'embedding.
    n_components : int
        Dimension de l'espace de sortie (2 pour visualisation, 3 possible).
    random_state : int
        Graine pour reproductibilité.
    **kwargs :
        Paramètres supplémentaires passés directement à UMAP ou t-SNE.
        UMAP : n_neighbors (15), min_dist (0.1), metric ('euclidean')
        t-SNE : perplexity (30), learning_rate ('auto'), max_iter (1000)
    """

    # Paramètres par défaut recommandés pour données spectrales LAMOST
    _UMAP_DEFAULTS = {
        "n_neighbors": 15,
        "min_dist": 0.1,
        "metric": "euclidean",
        "n_epochs": 200,
        "low_memory": False,
    }

    _TSNE_DEFAULTS = {
        "perplexity": 30,
        "learning_rate": "auto",
        "max_iter": 1000,
        "metric": "euclidean",
        "init": "pca",
    }

    def __init__(
        self,
        method: Literal["umap", "tsne"] = "umap",
        n_components: int = 2,
        random_state: int = 42,
        **kwargs,
    ) -> None:
        if method not in ("umap", "tsne"):
            raise ValueError(f"method doit être 'umap' ou 'tsne', reçu : {method!r}")
        self.method = method
        self.n_components = n_components
        self.random_state = random_state
        # Paramètres utilisateurs conservés tels quels, fusionnés plus tard avec les défauts.
        self.extra_kwargs = kwargs

        # État interne rempli après fit_transform pour réutilisation (transform, reporting).
        self._model = None
        self._embedding: Optional[np.ndarray] = None
        self._params_used: Optional[Dict] = None
        self.fit_time_: Optional[float] = None

    # ------------------------------------------------------------------
    # Ajustement et transformation
    # ------------------------------------------------------------------

    def fit_transform(
        self,
        X: np.ndarray,
        n_pca_components: Optional[int] = None,
    ) -> np.ndarray:
        """
        Calcule l'embedding 2D/3D de X.

        Parameters
        ----------
        X : np.ndarray (N, D)
            Matrice d'entrée (typiquement les scores PCA, pas X brut).
        n_pca_components : int | None
            Si non None, tronque X aux `n_pca_components` premières colonnes.
            Pratique si X est déjà la matrice de scores PCA complète.

        Returns
        -------
        np.ndarray (N, n_components) — coordonnées 2D/3D.
        """
        if n_pca_components is not None:
            # Troncature défensive: garde uniquement les premières composantes PCA voulues.
            X = X[:, :n_pca_components]

        logger.info(
            "Calcul embedding %s : entrée=%s, dim_sortie=%d, seed=%d",
            self.method.upper(),
            X.shape,
            self.n_components,
            self.random_state,
        )

        # Chronométrage wall-time pour comparaison UMAP/t-SNE et suivi notebook.
        t0 = time.perf_counter()
        if self.method == "umap":
            # Le helper remplit aussi self._model et self._params_used.
            Z = self._fit_umap(X)
        else:
            Z = self._fit_tsne(X)

        self.fit_time_ = time.perf_counter() - t0
        # Cache local du dernier embedding pour accès via propriété embedding.
        self._embedding = Z

        logger.info(
            "Embedding %s terminé en %.1f s | forme : %s",
            self.method.upper(),
            self.fit_time_,
            Z.shape,
        )
        return Z

    def transform(self, X_new: np.ndarray) -> np.ndarray:
        """
        Projette de nouveaux points dans l'espace appris.

        Note : t-SNE n'a PAS de transform() — lève NotImplementedError.
        UMAP supporte transform() (mais peut être lent).
        """
        if self._model is None:
            # Empêche l'usage avant apprentissage explicite de la géométrie latente.
            raise RuntimeError("Appeler fit_transform() avant transform().")
        if self.method == "tsne":
            # t-SNE classique ne fournit pas de mapping paramétrique stable vers de nouveaux points.
            raise NotImplementedError(
                "t-SNE est non-paramétrique et ne supporte pas transform().\n"
                "Utiliser UMAP ou recalculer l'embedding complet."
            )
        return self._model.transform(X_new)

    # ------------------------------------------------------------------
    # Analyse de stabilité
    # ------------------------------------------------------------------

    def stability_report(
        self,
        X: np.ndarray,
        n_seeds: int = 5,
        n_pca_components: Optional[int] = None,
        metric: str = "procrustes",
    ) -> pd.DataFrame:
        """
        Évalue la stabilité de l'embedding sur plusieurs seeds aléatoires.

        Pour chaque seed, calcule l'embedding et mesure la similarité
        (via analyse Procrustes) avec l'embedding de référence (seed 0).

        Parameters
        ----------
        X : np.ndarray
            Données d'entrée (scores PCA recommandés).
        n_seeds : int
            Nombre de seeds à tester (5 minimum recommandé pour le rapport).
        n_pca_components : int | None
            Troncature PCA avant embedding.
        metric : str
            'procrustes' : distance Procrustes (0 = identique, 1 = très différent).

        Returns
        -------
        pd.DataFrame avec colonnes ['seed', 'procrustes_distance', 'method'].

        Notes
        -----
        Une distance Procrustes < 0.05 indique une bonne stabilité.
        Citer cette analyse dans la section "Validation" du rapport.
        """
        from scipy.spatial import procrustes

        if n_pca_components is not None:
            # Même espace d'entrée pour tous les seeds afin d'isoler l'effet aléatoire.
            X = X[:, :n_pca_components]

        # Sur-ensemble aléatoire pour accélérer (max 5000 points)
        n = min(len(X), 5000)
        rng = np.random.default_rng(self.random_state)
        idx = rng.choice(len(X), size=n, replace=False)
        X_sub = X[idx]

        # `rows` alimente le DataFrame final; `embeddings` garde les géométries brutes.
        rows = []
        embeddings = []

        for i, seed in enumerate(range(n_seeds)):
            # Nouveau moteur par seed: évite tout état partagé entre exécutions.
            engine_i = EmbeddingEngine(
                method=self.method,
                n_components=self.n_components,
                random_state=seed,
                **self.extra_kwargs,
            )
            Z_i = engine_i.fit_transform(X_sub)
            embeddings.append(Z_i)

        # Référence = premier embedding
        Z_ref = embeddings[0]
        for i, Z in enumerate(embeddings):
            if i == 0:
                dist = 0.0
            else:
                try:
                    # Procrustes aligne translation/rotation/échelle avant distance résiduelle.
                    _, _, dist = procrustes(Z_ref, Z)
                except Exception:
                    # Robustesse: un seed problématique ne casse pas tout le rapport.
                    dist = np.nan
            rows.append(
                {
                    "seed": i,
                    "procrustes_distance": dist,
                    "method": self.method.upper(),
                }
            )

        df = pd.DataFrame(rows)
        # Moyenne calculée hors seed 0 (distance nulle par construction).
        mean_dist = df.loc[df["seed"] > 0, "procrustes_distance"].mean()
        logger.info(
            "Stabilité %s : distance Procrustes moyenne = %.4f (sur %d seeds)",
            self.method.upper(),
            mean_dist,
            n_seeds,
        )
        return df

    # ------------------------------------------------------------------
    # Contrôle négatif
    # ------------------------------------------------------------------

    def negative_control(
        self,
        X: np.ndarray,
        n_pca_components: Optional[int] = None,
    ) -> np.ndarray:
        """
        Calcule un embedding sur des données permutées (contrôle négatif).

        Si l'embedding original montre une structure, l'embedding permuté
        doit montrer une distribution uniforme (validation que la structure
        n'est pas un artefact numérique).

        Returns
        -------
        np.ndarray (N, n_components) — embedding sur X permuté.
        """
        rng = np.random.default_rng(self.random_state + 999)
        # Copie explicite pour préserver X d'entrée.
        X_perm = X.copy()
        # Permutation indépendante de chaque colonne
        for j in range(X_perm.shape[1]):
            # Détruit les corrélations inter-features tout en conservant les marginales 1D.
            rng.shuffle(X_perm[:, j])

        logger.info("Calcul du contrôle négatif (X permuté)...")
        engine_neg = EmbeddingEngine(
            method=self.method,
            n_components=self.n_components,
            random_state=self.random_state,
            **self.extra_kwargs,
        )
        return engine_neg.fit_transform(X_perm, n_pca_components=n_pca_components)

    # ------------------------------------------------------------------
    # Grid de paramètres (sensibilité)
    # ------------------------------------------------------------------

    def parameter_sensitivity(
        self,
        X: np.ndarray,
        param_grid: Optional[Dict] = None,
        n_pca_components: Optional[int] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Calcule des embeddings pour différentes valeurs du paramètre clé.

        Pour UMAP : varie n_neighbors (15, 30, 50, 100).
        Pour t-SNE : varie perplexity (5, 15, 30, 50).

        Returns
        -------
        dict {str(param_value): embedding_array}
        """
        if param_grid is None:
            # Grille par défaut orientée inspection visuelle coarse-to-fine.
            if self.method == "umap":
                param_grid = {"n_neighbors": [5, 15, 30, 50, 100]}
            else:
                param_grid = {"perplexity": [5, 15, 30, 50, 100]}

        results = {}
        # Convention actuelle: un seul hyperparamètre exploré à la fois.
        param_name = list(param_grid.keys())[0]
        param_values = param_grid[param_name]

        for val in param_values:
            # La valeur testée surcharge la config utilisateur courante.
            kw = {**self.extra_kwargs, param_name: val}
            engine = EmbeddingEngine(
                method=self.method,
                n_components=self.n_components,
                random_state=self.random_state,
                **kw,
            )
            Z = engine.fit_transform(X, n_pca_components=n_pca_components)
            results[f"{param_name}={val}"] = Z
            logger.info("  %s=%s : embedding calculé ✓", param_name, val)

        return results

    # ------------------------------------------------------------------
    # Propriétés d'accès
    # ------------------------------------------------------------------

    @property
    def embedding(self) -> np.ndarray:
        """Dernier embedding calculé."""
        if self._embedding is None:
            raise RuntimeError("Appeler fit_transform() d'abord.")
        return self._embedding

    @property
    def params_used(self) -> Dict:
        """Paramètres effectivement utilisés pour le dernier fit."""
        # Retourne un dict vide avant premier fit pour API sûre côté notebook.
        return self._params_used or {}

    # ------------------------------------------------------------------
    # Helpers privés
    # ------------------------------------------------------------------

    def _fit_umap(self, X: np.ndarray) -> np.ndarray:
        """Ajuste UMAP et retourne les coordonnées 2D."""
        try:
            # Import local pour garder le module utilisable même sans umap-learn installé.
            import umap
        except ImportError:
            raise ImportError("UMAP non installé. Exécuter : pip install umap-learn")
        # Priorité aux kwargs utilisateur sur les valeurs par défaut du projet.
        params = {**self._UMAP_DEFAULTS, **self.extra_kwargs}
        self._params_used = params
        # L'objet modèle est conservé pour autoriser transform() sur nouveaux points.
        self._model = umap.UMAP(
            n_components=self.n_components,
            random_state=self.random_state,
            **params,
        )
        return self._model.fit_transform(X)

    def _fit_tsne(self, X: np.ndarray) -> np.ndarray:
        """Ajuste t-SNE et retourne les coordonnées 2D."""
        from sklearn.manifold import TSNE

        # Même logique de fusion: defaults du projet puis surcharge utilisateur.
        params = {**self._TSNE_DEFAULTS, **self.extra_kwargs}
        self._params_used = params
        self._model = TSNE(
            n_components=self.n_components,
            random_state=self.random_state,
            n_jobs=-1,
            **params,
        )
        Z = self._model.fit_transform(X)
        # t-SNE n'a pas de transform() → on le simule pour cohérence API
        return Z

    def compute_umap3d(
        self,
        scores_95: np.ndarray,
        meta: "pd.DataFrame",
        y: np.ndarray,
        cluster_labels: np.ndarray,
        figures_dir,
        random_state: int = 42,
        n_subsample: int = 20_000,
    ) -> dict:
        """
        Calcule UMAP 3D, corrélations physiques et exporte les figures Plotly.

        Centralise la logique de la cellule 22 de phy3500_02_umap_tsne.ipynb.

        Returns
        -------
        dict avec les clés : Z_umap3, axis3_best_param, axis3_best_corr,
        path_3d_html, path_3d_teff, path_3d_feh, umap_3d_fit_time.
        """
        from pathlib import Path as _Path
        from scipy.stats import spearmanr

        try:
            import plotly.express as px
        except ImportError:
            raise ImportError("Plotly non installé → pip install plotly")

        # Normalisation du chemin de sortie pour write_html.
        figures_dir = _Path(figures_dir)
        # UMAP 3D recalculé avec hyperparamètres de référence du pipeline.
        engine_3d = EmbeddingEngine(
            method="umap",
            n_components=3,
            random_state=random_state,
            n_neighbors=15,
            min_dist=0.1,
        )
        Z_umap3 = engine_3d.fit_transform(scores_95)
        logger.info(
            "UMAP 3D terminé en %.1fs | shape : %s",
            engine_3d.fit_time_,
            Z_umap3.shape,
        )

        # Corrélations axes 3D ↔ paramètres physiques
        phys_params = [
            ("teff_gspphot", "T_eff (K)"),
            ("logg_gspphot", "log g"),
            ("mh_gspphot", "[Fe/H]"),
            ("bp_rp", "G_BP-G_RP"),
            ("phot_g_mean_mag", "G mag"),
            ("parallax", "Parallaxe"),
        ]
        axis3_best_corr, axis3_best_param = 0.0, "?"
        for col, label in phys_params:
            if col not in meta.columns:
                # Paramètre absent du catalogue: on l'ignore proprement.
                continue
            vals = meta[col].values.astype(float)
            valid = np.isfinite(vals)
            if valid.sum() < 100:
                # Corrélation non robuste si trop peu de points valides.
                continue
            r3, _ = spearmanr(Z_umap3[valid, 2], vals[valid])
            # On retient le paramètre le plus corrélé en valeur absolue.
            if abs(r3) > abs(axis3_best_corr):
                axis3_best_corr, axis3_best_param = r3, label

        # DataFrame Plotly
        import pandas as _pd

        df_3d = _pd.DataFrame(
            {
                "UMAP1": Z_umap3[:, 0],
                "UMAP2": Z_umap3[:, 1],
                "UMAP3": Z_umap3[:, 2],
                "Classe": y,
            }
        )
        for col, label in phys_params:
            if col in meta.columns:
                df_3d[label] = meta[col].values
        # Labels cluster human-readable pour exploration interactive Plotly.
        df_3d["Cluster"] = ["Bruit" if c == -1 else f"C{c}" for c in cluster_labels]

        rng = np.random.default_rng(random_state)
        N_PLOT = min(n_subsample, len(df_3d))
        # Sous-échantillon uniforme pour limiter le poids des fichiers HTML.
        df_plot = df_3d.iloc[rng.choice(len(df_3d), N_PLOT, replace=False)].copy()
        COLOR_MAP = {"STAR": "#4C72B0", "GALAXY": "#DD8452", "QSO": "#55A868"}

        # Figure 1 : classes LAMOST
        fig_class = px.scatter_3d(
            df_plot,
            x="UMAP1",
            y="UMAP2",
            z="UMAP3",
            color="Classe",
            color_discrete_map=COLOR_MAP,
            opacity=0.6,
            size_max=3,
            title=(
                f"Variété stellaire UMAP 3D — LAMOST DR5<br>"
                f"axe3 ↔ {axis3_best_param} ρ={axis3_best_corr:+.2f}"
            ),
        )
        fig_class.update_traces(marker=dict(size=2))
        fig_class.update_layout(
            scene=dict(
                xaxis_title="UMAP 1",
                yaxis_title="UMAP 2",
                zaxis_title=f"UMAP 3 (↔ {axis3_best_param})",
            ),
            margin=dict(l=0, r=0, b=0, t=50),
        )
        # Export autonome (HTML interactif partageable hors notebook).
        path_html = figures_dir / "umap3d_classes.html"
        fig_class.write_html(str(path_html))

        # Figure 2 : T_eff
        path_teff = None
        if "T_eff (K)" in df_plot.columns:
            fig_teff = px.scatter_3d(
                df_plot.dropna(subset=["T_eff (K)"]),
                x="UMAP1",
                y="UMAP2",
                z="UMAP3",
                color="T_eff (K)",
                color_continuous_scale="plasma",
                range_color=[
                    # Bornes robustes 2-98% pour atténuer outliers Gaia.
                    float(df_plot["T_eff (K)"].quantile(0.02)),
                    float(df_plot["T_eff (K)"].quantile(0.98)),
                ],
                opacity=0.6,
                title="Variété stellaire UMAP 3D — T_eff · LAMOST DR5 × Gaia DR3",
            )
            fig_teff.update_traces(marker=dict(size=2))
            fig_teff.update_layout(margin=dict(l=0, r=0, b=0, t=50))
            path_teff = figures_dir / "umap3d_teff.html"
            fig_teff.write_html(str(path_teff))

        # Figure 3 : [Fe/H]
        path_feh = None
        if "[Fe/H]" in df_plot.columns:
            fig_feh = px.scatter_3d(
                df_plot.dropna(subset=["[Fe/H]"]),
                x="UMAP1",
                y="UMAP2",
                z="UMAP3",
                color="[Fe/H]",
                color_continuous_scale="RdYlBu",
                range_color=[
                    # Même clipping robuste pour homogénéiser les échelles visuelles.
                    float(df_plot["[Fe/H]"].quantile(0.02)),
                    float(df_plot["[Fe/H]"].quantile(0.98)),
                ],
                opacity=0.6,
                title="Variété stellaire UMAP 3D — [Fe/H] · LAMOST DR5 × Gaia DR3",
            )
            fig_feh.update_traces(marker=dict(size=2))
            fig_feh.update_layout(margin=dict(l=0, r=0, b=0, t=50))
            path_feh = figures_dir / "umap3d_feh.html"
            fig_feh.write_html(str(path_feh))

        logger.info("UMAP 3D — axe 3 ↔ %s (ρ=%+.3f)", axis3_best_param, axis3_best_corr)
        # Bundle unique pour réutilisation dans le notebook et dans le reporting.
        return {
            "Z_umap3": Z_umap3,
            "axis3_best_param": axis3_best_param,
            "axis3_best_corr": axis3_best_corr,
            "path_3d_html": path_html,
            "path_3d_teff": path_teff,
            "path_3d_feh": path_feh,
            "umap_3d_fit_time": engine_3d.fit_time_,
        }


def compare_embeddings(
    X: np.ndarray,
    methods: Optional[List[str]] = None,
    n_pca_components: int = 20,
    random_state: int = 42,
) -> Dict[str, np.ndarray]:
    """
    Calcule plusieurs embeddings en parallèle pour comparaison visuelle.

    Parameters
    ----------
    X : np.ndarray
        Matrice d'entrée (scores PCA recommandés).
    methods : list[str] | None
        Liste de méthodes. Défaut : ['umap', 'tsne'].
    n_pca_components : int
        Nombre de composantes PCA à utiliser comme entrée.
    random_state : int
        Graine commune.

    Returns
    -------
    dict {'umap': Z_umap, 'tsne': Z_tsne, ...}
    """
    if methods is None:
        methods = ["umap", "tsne"]

    results = {}
    for method in methods:
        logger.info("--- Calcul %s ---", method.upper())
        # Un moteur indépendant par méthode pour éviter tout couplage d'état.
        engine = EmbeddingEngine(method=method, random_state=random_state)
        Z = engine.fit_transform(X, n_pca_components=n_pca_components)
        results[method] = Z

    return results
