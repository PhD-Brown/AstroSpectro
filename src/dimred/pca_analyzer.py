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
import re
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
        # Paramètres utilisateur qui pilotent directement sklearn.PCA.
        self.n_components = n_components
        self.random_state = random_state
        self.svd_solver = svd_solver

        # Attributs renseignés après fit().
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
        # 1) Initialise l'estimateur PCA sklearn avec les hyperparamètres choisis.
        self._pca = PCA(
            n_components=self.n_components,
            random_state=self.random_state,
            svd_solver=self.svd_solver,
        )
        # 2) Ajuste les composantes principales sur la matrice d'entrée.
        self._pca.fit(X)
        # 3) Mémorise X pour d'éventuels recalculs internes (corrélations sans scores fournis).
        self._X_fit = X
        # 4) Aligne les noms de variables: fournis par l'utilisateur ou fallback f0..fD-1.
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
        # Réutilise le même flux que fit() pour garantir un état interne cohérent.
        self.fit(X, feature_names)
        # Projection de X dans l'espace latent PCA appris à l'étape précédente.
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
        # Courbe cumulée monotone: [PC1, PC1+PC2, ...].
        cumvar = self.cumulative_variance
        # Premier indice où la courbe atteint (ou dépasse) le seuil demandé.
        idx = np.searchsorted(cumvar, threshold)
        # Conversion indice 0-based -> nombre de composantes 1-based.
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
        # Alias local pour éviter des accès répétés à la propriété.
        ratio = self.explained_variance_ratio
        # Tableau final: une ligne par composante principale.
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
        # self.loadings est (n_components, n_features) -> transpose pour avoir features en lignes.
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
        # Vecteur de poids de la composante ciblée (taille = n_features).
        vec = self.loadings[pc_idx]
        # Tri décroissant par impact absolu pour isoler les variables dominantes.
        idx_sorted = np.argsort(np.abs(vec))[::-1][:n_top]
        # Restitue à la fois le signe physique (loading) et la force (abs_loading).
        return pd.DataFrame(
            {
                "feature": [self.feature_names_[i] for i in idx_sorted],
                "loading": vec[idx_sorted],
                "abs_loading": np.abs(vec[idx_sorted]),
                "PC": f"PC{pc_idx + 1}",
            }
        ).reset_index(drop=True)

    def loadings_family_breakdown(
        self,
        feature_names: Optional[List[str]] = None,
        pc_indices: tuple[int, ...] = (0, 1),
    ) -> dict:
        """
        Décompose les contributions des loadings par famille spectroscopique.

        Cette méthode est conçue pour factoriser la logique du notebook 01 (section 4.bis)
        sans modifier la logique scientifique: contribution quadratique normalisée
        par famille et par composante principale.

        Returns
        -------
        dict
            Contient loadings, feat_names, families, family_list, color_map,
            ainsi que contributions par PC (et alias contrib_pc1/contrib_pc2).
        """
        self._check_fitted()

        # Source des noms: argument explicite ou noms mémorisés au fit.
        if feature_names is None:
            feat_names = self.feature_names_
        else:
            feat_names = list(feature_names)

        if feat_names is None:
            raise RuntimeError("Noms de features indisponibles: fournir feature_names.")

        # Prépare la taxonomie des familles spectroscopiques.
        loadings = self.loadings
        families = [self._assign_family(f) for f in feat_names]
        family_list = list(self._FAMILIES.keys())
        color_map = self._family_color_map()

        # Calcule les contributions par famille pour chaque PC demandée.
        contributions = {
            pc_idx: self._family_contributions(pc_idx, loadings, feat_names, families)
            for pc_idx in pc_indices
        }

        out = {
            "loadings": loadings,
            "feat_names": feat_names,
            "families": families,
            "family_list": family_list,
            "color_map": color_map,
            "families_map": dict(self._FAMILIES),
            "family_colors": list(self._FAMILY_COLORS),
            "contributions": contributions,
        }
        if 0 in contributions:
            # Alias pratiques historiques utilisés dans certains notebooks.
            out["contrib_pc1"] = contributions[0]
        if 1 in contributions:
            out["contrib_pc2"] = contributions[1]
        return out

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

        # Si les scores ne sont pas fournis, on les recalcule depuis X d'entraînement.
        if scores is None:
            scores = self._pca.transform(self._X_fit)

        # Limite l'analyse aux n_pcs premières composantes.
        scores = scores[:, :n_pcs]

        if method not in {"spearman", "pearson"}:
            raise ValueError("method must be 'spearman' or 'pearson'.")

        # Fonction de corrélation choisie dynamiquement selon l'option demandée.
        corr_fn = spearmanr if method == "spearman" else pearsonr

        # Colonnes numériques uniquement
        param_cols = [c for c in meta.columns if pd.api.types.is_numeric_dtype(meta[c])]
        params_arr = meta[param_cols].values

        n_pc = scores.shape[1]
        n_param = len(param_cols)
        # Initialisation à NaN: conserve explicitement les cas non calculables.
        corr_matrix = np.full((n_pc, n_param), np.nan)

        # Double boucle: chaque PC contre chaque paramètre physique.
        for i in range(n_pc):
            pc_scores = scores[:, i]
            for j in range(n_param):
                param_vals = params_arr[:, j]
                # Masque de validité commun pour ignorer NaN/inf des deux séries.
                valid = np.isfinite(param_vals) & np.isfinite(pc_scores)
                if valid.sum() < 30:
                    # Seuil minimal de robustesse statistique.
                    continue

                # corr_fn retourne (coefficient, p-value); seul le coefficient est conservé.
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
        # Projection PCA complète de X.
        scores = self._pca.transform(X)
        if n_components is not None:
            # Copie défensive puis troncature: on annule les composantes au-delà de n_components.
            scores_trunc = scores.copy()
            scores_trunc[:, n_components:] = 0.0
        else:
            scores_trunc = scores
        # Retour dans l'espace original après troncature éventuelle.
        X_recon = self._pca.inverse_transform(scores_trunc)
        # Erreur quadratique moyenne par ligne (spectre).
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
            # Par défaut: explore de 1 à min(n_components, 50).
            n_max = self._pca.n_components_
            n_range = list(range(1, min(n_max + 1, 51)))

        rows = []
        for n in n_range:
            # Réévalue la reconstruction pour chaque valeur de n.
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
        # On travaille sur les n_pcs premières dimensions latentes.
        scores = self._pca.transform(X)[:, :n_pcs]
        classes = np.unique(y)
        rows = []
        for cls in classes:
            # Sélection des échantillons appartenant à la classe courante.
            mask = y == cls
            for i in range(n_pcs):
                # Statistiques descriptives de la distribution des scores de classe.
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

        # Payload minimal suffisant pour reconstruire l'objet d'analyse.
        joblib.dump({"pca": self._pca, "feature_names": self.feature_names_}, path)
        logger.info("PCAAnalyzer sauvegardé : %s", path)

    @classmethod
    def load(cls, path: str) -> "PCAAnalyzer":
        """Charge un PCAAnalyzer préalablement sauvegardé."""
        import joblib

        obj = joblib.load(path)
        # Bypass __init__: on réhydrate directement l'état persistant.
        analyzer = cls.__new__(cls)
        analyzer._pca = obj["pca"]
        analyzer.feature_names_ = obj["feature_names"]
        # Conserve l'info utile de configuration (nombre de composantes effectif).
        analyzer.n_components = analyzer._pca.n_components_
        # X de fit n'est pas sérialisé: certaines méthodes dépendantes demanderont scores explicites.
        analyzer._X_fit = None
        return analyzer

    # ------------------------------------------------------------------
    # Helpers privés
    # ------------------------------------------------------------------

    _FAMILIES = {
        "Balmer\n(H α/β/γ/δ/ε/8/9/10)": [
            r"H[αβγδε]|Halpha|Hbeta|Hgamma|Hdelta|Hepsilon|H8|H9|H10"
            r"|feature_H[89]|feature_H10|balmer|paschen"
        ],
        "Calcium\n(Ca II H&K + triplet)": [r"CaII|CaH|CaK|Ca_8|Ca_trip|feature_Ca"],
        "Magnésium\n(Mg b + triplet)": [r"Mg_b|Mg_5|MgH|Mg_trip|feature_Mg"],
        "Fer & métaux\n(Fe, Cr, Ni, Co, V, Al)": [
            r"feature_Fe|feature_Cr|feature_Ni|feature_Co|feature_V_"
            r"|feature_Al|iron_peak|metal_index|metal_poor|FeH_proxy|alpha_Fe|alpha_el"
        ],
        "Sodium\n(Na D)": [r"Na_D|feature_Na"],
        "Titane & moléc.\n(TiO, VO, CH, CN, CaH)": [
            r"TiO|VO_|molecular|feature_Ti|CNO|CN_"
        ],
        "Strontium, Baryum\ns-process": [
            r"feature_Sr|feature_Ba|s_process|feature_ratio_Ba|feature_ratio_Sr"
        ],
        "Continuum\n(pente, couleur, break)": [
            r"continuum|slope|break_4000|flux_ratio|synthetic_BV|UV_excess"
            r"|curvature|color_|feature_cont"
        ],
        "Profils de raies\n(asymétrie, ailes, etc.)": [
            r"asymmetr|wing|kurtosis|skewness|core_width|base_width|depth"
            r"|profile_shape|avg_line|rotation"
        ],
        "Indices Lick\n& indices synthétiques": [
            r"feature_index|Dn4000|G4300|Hbeta_index|NaD_Lick|TiO_1_Lick"
            r"|ratio_EW|EW_CaHK|contrast_metals"
        ],
        "Détection peak\n(match_*, present)": [r"match_|feature_.*_present"],
        "Autres": [r".*"],
    }

    _FAMILY_COLORS = [
        "#E8593C",
        "#3B8BD4",
        "#4C9B6F",
        "#B07DB8",
        "#F5A623",
        "#2E86AB",
        "#D4A853",
        "#7F8FA6",
        "#C06C84",
        "#6CAE75",
        "#A3B4C5",
        "#CCCCCC",
    ]

    def _assign_family(self, feat_name: str) -> str:
        # Premier motif regex qui matche détermine la famille retenue.
        for family, patterns in self._FAMILIES.items():
            for pat in patterns:
                if re.search(pat, feat_name, re.IGNORECASE):
                    return family
        return "Autres"

    def _family_color_map(self) -> dict[str, str]:
        # Assigne une couleur stable à chaque famille (ordre dict conservé).
        family_list = list(self._FAMILIES.keys())
        return {
            fam: self._FAMILY_COLORS[i % len(self._FAMILY_COLORS)]
            for i, fam in enumerate(family_list)
        }

    def _family_contributions(
        self,
        pc_idx: int,
        loadings: np.ndarray,
        feat_names: List[str],
        families: List[str],
    ) -> pd.DataFrame:
        if pc_idx < 0 or pc_idx >= loadings.shape[0]:
            raise IndexError(f"pc_idx={pc_idx} hors bornes (0..{loadings.shape[0]-1}).")

        # Contribution quadratique normalisée: somme totale = 1.
        w = loadings[pc_idx] ** 2
        w /= w.sum()
        # Regroupement par famille pour lecture physique synthétique.
        df = pd.DataFrame({"feature": feat_names, "family": families, "weight": w})
        agg = df.groupby("family")["weight"].sum().reset_index()
        # Nettoyage visuel + tri décroissant des contributions.
        agg = agg[agg["weight"] > 0].sort_values("weight", ascending=False)
        return agg

    def _check_fitted(self) -> None:
        # Garde-fou unique pour toutes les méthodes dépendantes du modèle appris.
        if self._pca is None:
            raise RuntimeError("PCAAnalyzer non ajusté — appeler fit() d'abord.")
