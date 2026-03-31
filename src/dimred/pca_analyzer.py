"""
AstroSpectro — dimred.pca_analyzer
====================================

Analyse en Composantes Principales (ACP / PCA) appliquée aux spectres
stellaires LAMOST DR5.

Ce module encapsule sklearn.decomposition.PCA avec des méthodes orientées
vers l'interprétation physique des axes principaux :
  - variance expliquée cumulée (critère de coupure)
  - corrélations PC ↔ paramètres physiques (T_eff, log g, [Fe/H], ...)
  - loadings physiques (quelles régions spectrales ou features dominent)
  - biplot simplifié (scores + vecteurs de chargement)

Ce module est volontairement SANS matplotlib : toutes les figures sont
générées par DimRedVisualizer pour séparer calcul et présentation.

Référence
---------
- Jolliffe, I.T. (2002). Principal Component Analysis, 2nd ed. Springer.
- Yip et al. (2004). AJ 128:585 — PCA applied to SDSS galaxy spectra.
- Singh et al. (2023). arXiv:2302.09207 — PCA on LAMOST stellar spectra.

Exemple
-------
>>> pca = PCAAnalyzer(n_components=50, random_state=42)
>>> pca.fit(X)
>>> scores = pca.transform(X)
>>> corr = pca.correlations_with_params(meta[["teff_gspphot", "mh_gspphot"]])
>>> print(pca.n_components_for_variance(0.95))
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from scipy.stats import pearsonr, spearmanr

logger = logging.getLogger(__name__)


class PCAAnalyzer:
    """
    Analyse en Composantes Principales orientée interprétation physique.

    Parameters
    ----------
    n_components : int | float | 'mle'
        Nombre de composantes (int), fraction de variance à conserver (float
        entre 0 et 1), ou 'mle' pour la sélection automatique.
        Recommandé : 50 pour exploration, puis réduire selon variance expliquée.
    random_state : int
        Graine pour reproductibilité (utilisée par sklearn si svd_solver='randomized').
    svd_solver : str
        Solveur SVD. 'auto' (défaut), 'full', 'randomized' (rapide pour grandes matrices).
    """

    def __init__(
        self,
        n_components: int | float | str = 50,
        random_state: int = 42,
        svd_solver: str = "auto",
    ) -> None:
        self.n_components = n_components
        self.random_state = random_state
        self.svd_solver = svd_solver

        self._pca: Optional[PCA] = None
        self.feature_names_: Optional[List[str]] = None
        self._X_fit: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Ajustement
    # ------------------------------------------------------------------

    def fit(
        self,
        X: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> "PCAAnalyzer":
        """
        Ajuste la PCA sur la matrice X.

        Parameters
        ----------
        X : np.ndarray (N, D)
            Matrice d'entrée (doit être centrée-réduite — StandardScaler recommandé).
        feature_names : list[str] | None
            Noms des colonnes de X, pour l'interprétation des loadings.

        Returns
        -------
        self
        """
        self._pca = PCA(
            n_components=self.n_components,
            random_state=self.random_state,
            svd_solver=self.svd_solver,
        )
        self._pca.fit(X)
        self._X_fit = X
        self.feature_names_ = feature_names or [f"f{i}" for i in range(X.shape[1])]

        logger.info(
            "PCA ajustée : %d composantes, %.2f%% variance expliquée (cumulative)",
            self._pca.n_components_,
            100 * self._pca.explained_variance_ratio_.sum(),
        )
        return self

    def fit_transform(
        self,
        X: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> np.ndarray:
        """Ajuste et retourne les scores (projections sur les PCs)."""
        self.fit(X, feature_names)
        return self._pca.transform(X)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Projette X sur les composantes principales déjà ajustées."""
        self._check_fitted()
        return self._pca.transform(X)

    # ------------------------------------------------------------------
    # Variance expliquée
    # ------------------------------------------------------------------

    @property
    def explained_variance_ratio(self) -> np.ndarray:
        """Fraction de variance expliquée par chaque PC."""
        self._check_fitted()
        return self._pca.explained_variance_ratio_

    @property
    def cumulative_variance(self) -> np.ndarray:
        """Variance expliquée cumulée."""
        return np.cumsum(self.explained_variance_ratio)

    def n_components_for_variance(self, threshold: float = 0.95) -> int:
        """
        Retourne le nombre minimal de composantes pour atteindre `threshold`
        de variance expliquée cumulée.

        Parameters
        ----------
        threshold : float
            Fraction de variance cible (ex. 0.90, 0.95, 0.99).

        Returns
        -------
        int : nombre de composantes nécessaires.
        """
        cumvar = self.cumulative_variance
        idx = np.searchsorted(cumvar, threshold)
        n = int(idx) + 1
        logger.info(
            "%.0f%% variance → %d composantes (sur %d)",
            100 * threshold,
            n,
            len(cumvar),
        )
        return n

    def variance_summary(self) -> pd.DataFrame:
        """
        DataFrame récapitulatif : PC, variance individuelle, variance cumulée.

        Utile pour export CSV ou affichage dans le rapport.
        """
        self._check_fitted()
        ratio = self.explained_variance_ratio
        return pd.DataFrame(
            {
                "PC": [f"PC{i+1}" for i in range(len(ratio))],
                "variance_individuelle": ratio,
                "variance_cumulee": np.cumsum(ratio),
                "valeur_propre": self._pca.explained_variance_,
            }
        )

    # ------------------------------------------------------------------
    # Loadings (composantes principales)
    # ------------------------------------------------------------------

    @property
    def loadings(self) -> np.ndarray:
        """
        Matrice de loadings (n_components, n_features).
        Chaque ligne = vecteur propre d'une PC dans l'espace des features.
        """
        self._check_fitted()
        return self._pca.components_

    def loadings_dataframe(self) -> pd.DataFrame:
        """
        DataFrame des loadings avec noms de features et composantes.

        Rows  : features (noms)
        Cols  : PC1, PC2, ...
        """
        self._check_fitted()
        return pd.DataFrame(
            self.loadings.T,
            index=self.feature_names_,
            columns=[f"PC{i+1}" for i in range(self._pca.n_components_)],
        )

    def top_features_per_pc(self, pc_idx: int = 0, n_top: int = 10) -> pd.DataFrame:
        """
        Retourne les `n_top` features les plus influentes pour la PC `pc_idx`
        (indexation 0-based), triées par valeur absolue du loading décroissante.

        Parameters
        ----------
        pc_idx : int
            Indice de la composante (0 = PC1, 1 = PC2, ...).
        n_top : int
            Nombre de features à retourner.

        Returns
        -------
        pd.DataFrame avec colonnes ['feature', 'loading', 'abs_loading'].
        """
        self._check_fitted()
        vec = self.loadings[pc_idx]
        idx_sorted = np.argsort(np.abs(vec))[::-1][:n_top]
        return pd.DataFrame(
            {
                "feature": [self.feature_names_[i] for i in idx_sorted],
                "loading": vec[idx_sorted],
                "abs_loading": np.abs(vec[idx_sorted]),
                "PC": f"PC{pc_idx + 1}",
            }
        ).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Corrélations PC ↔ paramètres physiques
    # ------------------------------------------------------------------

    def correlations_with_params(
        self,
        meta: pd.DataFrame,
        scores: Optional[np.ndarray] = None,
        method: str = "spearman",
        n_pcs: int = 10,
    ) -> pd.DataFrame:
        """
        Calcule les corrélations entre les scores des PCs et les paramètres
        physiques Gaia (T_eff, log g, [Fe/H], bp_rp, ...).

        Parameters
        ----------
        meta : pd.DataFrame
            DataFrame avec les paramètres physiques (N lignes, même ordre que X).
        scores : np.ndarray (N, K) | None
            Scores PCA. Si None, les scores sont recalculés sur X_fit.
        method : 'pearson' | 'spearman'
            Méthode de corrélation. Spearman recommandé (robuste aux outliers).
        n_pcs : int
            Nombre de PCs à analyser.

        Returns
        -------
        pd.DataFrame (n_pcs, n_params) avec les coefficients de corrélation.
            Index : PC1, PC2, ...
            Colonnes : paramètres physiques numériques.
        """
        self._check_fitted()

        if scores is None:
            scores = self._pca.transform(self._X_fit)

        scores = scores[:, :n_pcs]

        if method not in {"spearman", "pearson"}:
            raise ValueError("method must be 'spearman' or 'pearson'.")

        corr_fn = spearmanr if method == "spearman" else pearsonr

        # Colonnes numériques uniquement
        param_cols = [c for c in meta.columns if pd.api.types.is_numeric_dtype(meta[c])]
        params_arr = meta[param_cols].values

        n_pc = scores.shape[1]
        n_param = len(param_cols)
        corr_matrix = np.full((n_pc, n_param), np.nan)

        for i in range(n_pc):
            pc_scores = scores[:, i]
            for j in range(n_param):
                param_vals = params_arr[:, j]
                valid = np.isfinite(param_vals) & np.isfinite(pc_scores)
                if valid.sum() < 30:
                    continue

                r, _ = corr_fn(pc_scores[valid], param_vals[valid])
                corr_matrix[i, j] = r

        df_corr = pd.DataFrame(
            corr_matrix,
            index=[f"PC{i+1}" for i in range(n_pc)],
            columns=param_cols,
        )
        return df_corr

    # ------------------------------------------------------------------
    # Reconstruction & qualité
    # ------------------------------------------------------------------

    def reconstruction_error(
        self, X: np.ndarray, n_components: Optional[int] = None
    ) -> np.ndarray:
        """
        Erreur de reconstruction MSE par spectre pour `n_components` PCs.

        Parameters
        ----------
        X : np.ndarray (N, D)
        n_components : int | None
            Nombre de PCs utilisées pour la reconstruction. None = toutes.

        Returns
        -------
        np.ndarray (N,) : MSE par échantillon.
        """
        self._check_fitted()
        scores = self._pca.transform(X)
        if n_components is not None:
            scores_trunc = scores.copy()
            scores_trunc[:, n_components:] = 0.0
        else:
            scores_trunc = scores
        X_recon = self._pca.inverse_transform(scores_trunc)
        mse = np.mean((X - X_recon) ** 2, axis=1)
        return mse

    def reconstruction_error_vs_n(
        self, X: np.ndarray, n_range: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """
        Erreur de reconstruction moyenne pour différents nombres de composantes.

        Utile pour choisir le nombre optimal de PCs (coude sur la courbe).

        Returns
        -------
        pd.DataFrame avec colonnes ['n_components', 'mse_mean', 'mse_std'].
        """
        if n_range is None:
            n_max = self._pca.n_components_
            n_range = list(range(1, min(n_max + 1, 51)))

        rows = []
        for n in n_range:
            mse = self.reconstruction_error(X, n_components=n)
            rows.append(
                {
                    "n_components": n,
                    "mse_mean": mse.mean(),
                    "mse_std": mse.std(),
                }
            )
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Analyse par classe
    # ------------------------------------------------------------------

    def class_separation(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_pcs: int = 5,
    ) -> pd.DataFrame:
        """
        Calcule les statistiques de séparation des classes dans l'espace PCA.

        Pour chaque PC, retourne la moyenne et l'écart-type des scores
        par classe, utile pour évaluer si PCA sépare STAR/GALAXY/QSO.

        Returns
        -------
        pd.DataFrame multi-index (classe, PC).
        """
        self._check_fitted()
        scores = self._pca.transform(X)[:, :n_pcs]
        classes = np.unique(y)
        rows = []
        for cls in classes:
            mask = y == cls
            for i in range(n_pcs):
                rows.append(
                    {
                        "classe": cls,
                        "PC": f"PC{i+1}",
                        "n": mask.sum(),
                        "mean": scores[mask, i].mean(),
                        "std": scores[mask, i].std(),
                        "median": np.median(scores[mask, i]),
                        "q25": np.percentile(scores[mask, i], 25),
                        "q75": np.percentile(scores[mask, i], 75),
                    }
                )
        return pd.DataFrame(rows).set_index(["classe", "PC"])

    # ------------------------------------------------------------------
    # Accès bas niveau
    # ------------------------------------------------------------------

    @property
    def sklearn_pca(self) -> PCA:
        """Accès direct à l'objet sklearn.decomposition.PCA."""
        self._check_fitted()
        return self._pca

    @property
    def mean_(self) -> np.ndarray:
        """Moyenne estimée (pour reconstruction manuelle)."""
        self._check_fitted()
        return self._pca.mean_

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Sauvegarde le modèle PCA (joblib)."""
        import joblib

        joblib.dump({"pca": self._pca, "feature_names": self.feature_names_}, path)
        logger.info("PCAAnalyzer sauvegardé : %s", path)

    @classmethod
    def load(cls, path: str) -> "PCAAnalyzer":
        """Charge un PCAAnalyzer préalablement sauvegardé."""
        import joblib

        obj = joblib.load(path)
        analyzer = cls.__new__(cls)
        analyzer._pca = obj["pca"]
        analyzer.feature_names_ = obj["feature_names"]
        analyzer.n_components = analyzer._pca.n_components_
        analyzer._X_fit = None
        return analyzer

    # ------------------------------------------------------------------
    # Helpers privés
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if self._pca is None:
            raise RuntimeError("PCAAnalyzer non ajusté — appeler fit() d'abord.")
