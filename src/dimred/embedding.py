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
        self.extra_kwargs = kwargs

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
            X = X[:, :n_pca_components]

        logger.info(
            "Calcul embedding %s : entrée=%s, dim_sortie=%d, seed=%d",
            self.method.upper(),
            X.shape,
            self.n_components,
            self.random_state,
        )

        t0 = time.perf_counter()
        if self.method == "umap":
            Z = self._fit_umap(X)
        else:
            Z = self._fit_tsne(X)

        self.fit_time_ = time.perf_counter() - t0
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
            raise RuntimeError("Appeler fit_transform() avant transform().")
        if self.method == "tsne":
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
            X = X[:, :n_pca_components]

        # Sur-ensemble aléatoire pour accélérer (max 5000 points)
        n = min(len(X), 5000)
        rng = np.random.default_rng(self.random_state)
        idx = rng.choice(len(X), size=n, replace=False)
        X_sub = X[idx]

        rows = []
        embeddings = []

        for i, seed in enumerate(range(n_seeds)):
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
                    _, _, dist = procrustes(Z_ref, Z)
                except Exception:
                    dist = np.nan
            rows.append(
                {
                    "seed": i,
                    "procrustes_distance": dist,
                    "method": self.method.upper(),
                }
            )

        df = pd.DataFrame(rows)
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
        X_perm = X.copy()
        # Permutation indépendante de chaque colonne
        for j in range(X_perm.shape[1]):
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
            if self.method == "umap":
                param_grid = {"n_neighbors": [5, 15, 30, 50, 100]}
            else:
                param_grid = {"perplexity": [5, 15, 30, 50, 100]}

        results = {}
        param_name = list(param_grid.keys())[0]
        param_values = param_grid[param_name]

        for val in param_values:
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
        return self._params_used or {}

    # ------------------------------------------------------------------
    # Helpers privés
    # ------------------------------------------------------------------

    def _fit_umap(self, X: np.ndarray) -> np.ndarray:
        """Ajuste UMAP et retourne les coordonnées 2D."""
        try:
            import umap
        except ImportError:
            raise ImportError("UMAP non installé. Exécuter : pip install umap-learn")
        params = {**self._UMAP_DEFAULTS, **self.extra_kwargs}
        self._params_used = params
        self._model = umap.UMAP(
            n_components=self.n_components,
            random_state=self.random_state,
            **params,
        )
        return self._model.fit_transform(X)

    def _fit_tsne(self, X: np.ndarray) -> np.ndarray:
        """Ajuste t-SNE et retourne les coordonnées 2D."""
        from sklearn.manifold import TSNE

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


# ------------------------------------------------------------------
# Utilitaire : grille de comparaison méthodes
# ------------------------------------------------------------------


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
        engine = EmbeddingEngine(method=method, random_state=random_state)
        Z = engine.fit_transform(X, n_pca_components=n_pca_components)
        results[method] = Z

    return results
