"""
AstroSpectro — dimred.dimred_visualizer
=========================================

Figures de qualité publication pour la réduction de dimension.

Toutes les fonctions retournent (fig, ax) ou (fig, axes) et acceptent
un paramètre `save_path` optionnel pour l'export PNG/PDF haute résolution.

Style : fond blanc, palette perceptuellement uniforme (viridis, plasma),
police compatible LaTeX si disponible, annotations en français.

Figures disponibles
-------------------
1. plot_variance_explained()     — courbe de variance expliquée (coude)
2. plot_loadings_heatmap()       — heatmap des loadings (features × PCs)
3. plot_loadings_bar()           — barplot top features pour une PC
4. plot_correlation_heatmap()    — corrélations PC ↔ paramètres physiques
5. plot_embedding()              — scatter 2D coloré (classe ou paramètre)
6. plot_embedding_grid()         — grille multi-coloration sur même embedding
7. plot_stability()              — rapport de stabilité (Procrustes)
8. plot_reconstruction_error()   — erreur de reconstruction vs n_components
9. plot_hr_diagram_embedding()   — HR diagram coloré par coordonnée embedding

Exemple
-------
>>> viz = DimRedVisualizer(figsize=(8, 6), dpi=150)
>>> fig, ax = viz.plot_variance_explained(pca, threshold=0.95)
>>> fig.savefig("reports/figures/pca_variance.pdf", bbox_inches="tight")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

logger = logging.getLogger(__name__)

# Tentative d'activation LaTeX (graceful fallback)
try:
    plt.rcParams.update(
        {
            "text.usetex": False,  # Mettre True si LaTeX installé
            "font.family": "DejaVu Sans",
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "figure.dpi": 150,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
except Exception:
    pass

# Palette de classes LAMOST
CLASS_COLORS = {
    "STAR": "#4C72B0",
    "GALAXY": "#DD8452",
    "QSO": "#55A868",
    "UNKNOWN": "#8172B3",
}

CLASS_MARKERS = {
    "STAR": "o",
    "GALAXY": "s",
    "QSO": "^",
    "UNKNOWN": "x",
}


class DimRedVisualizer:
    """
    Générateur de figures pour l'analyse en réduction de dimension.

    Parameters
    ----------
    figsize : tuple
        Taille par défaut des figures.
    dpi : int
        Résolution des figures (150 pour écran, 300 pour publication).
    cmap_continuous : str
        Colormap pour les paramètres continus (T_eff, log g, ...).
    output_dir : str | Path | None
        Dossier de sortie par défaut pour save_path automatique.
    """

    def __init__(
        self,
        figsize: Tuple[float, float] = (9, 6),
        dpi: int = 150,
        cmap_continuous: str = "plasma",
        output_dir: Optional[str | Path] = None,
    ) -> None:
        self.figsize = figsize
        self.dpi = dpi
        self.cmap_continuous = cmap_continuous
        self.output_dir = Path(output_dir) if output_dir else None
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Variance expliquée
    # ------------------------------------------------------------------

    def plot_variance_explained(
        self,
        pca_analyzer,
        threshold: float = 0.95,
        max_pcs: int = 50,
        save_path: Optional[str] = None,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Courbe de variance expliquée cumulée avec ligne de seuil.

        Parameters
        ----------
        pca_analyzer : PCAAnalyzer
            Objet PCAAnalyzer ajusté.
        threshold : float
            Seuil de variance à marquer (ligne verticale + annotation).
        max_pcs : int
            Nombre maximal de PCs à afficher.
        """
        ratio = pca_analyzer.explained_variance_ratio[:max_pcs]
        cumvar = np.cumsum(ratio)
        pcs = np.arange(1, len(ratio) + 1)

        n_thresh = pca_analyzer.n_components_for_variance(threshold)

        fig, axes = plt.subplots(1, 2, figsize=self.figsize, dpi=self.dpi)

        # Gauche : variance individuelle (barplot)
        ax = axes[0]
        ax.bar(pcs, ratio * 100, color="#4C72B0", alpha=0.75, width=0.85)
        ax.set_xlabel("Composante principale")
        ax.set_ylabel("Variance expliquée (%)")
        ax.set_title("Variance individuelle par PC")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        # Droite : variance cumulée
        ax = axes[1]
        ax.plot(
            pcs,
            cumvar * 100,
            "o-",
            color="#4C72B0",
            ms=4,
            lw=2,
            label="Variance cumulée",
        )
        ax.axhline(
            threshold * 100,
            color="#DD8452",
            ls="--",
            lw=1.5,
            label=f"Seuil {threshold*100:.0f}%",
        )
        ax.axvline(n_thresh, color="#55A868", ls=":", lw=1.5, label=f"{n_thresh} PCs")

        # ← AJOUTER CES DEUX LIGNES
        n_thresh_plot = min(n_thresh, len(cumvar))  # clamp si n_thresh > max_pcs
        idx = n_thresh_plot - 1  # index 0-based dans cumvar

        ax.annotate(
            f"{n_thresh} PCs\n→ {cumvar[idx]*100:.1f}%",  # affiche la vraie valeur
            xy=(n_thresh_plot, cumvar[idx] * 100),
            xytext=(n_thresh_plot + 2, cumvar[idx] * 100 - 8),
            fontsize=9,
            arrowprops=dict(arrowstyle="->", color="gray"),
            color="#55A868",
        )
        ax.set_xlabel("Nombre de composantes")
        ax.set_ylabel("Variance expliquée cumulée (%)")
        ax.set_title("Variance cumulée")
        ax.set_ylim(0, 102)
        ax.legend(fontsize=9)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        fig.suptitle(
            "ACP — Analyse de la variance expliquée (LAMOST DR5)", fontsize=14, y=1.01
        )
        plt.tight_layout()

        self._save(fig, save_path, "pca_variance_explained")
        return fig, axes

    # ------------------------------------------------------------------
    # 2. Heatmap des loadings
    # ------------------------------------------------------------------

    def plot_loadings_heatmap(
        self,
        pca_analyzer,
        n_pcs: int = 10,
        n_features: int = 30,
        save_path: Optional[str] = None,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Heatmap (features × PCs) des loadings de la PCA.

        Sélectionne les `n_features` features les plus influentes
        (max abs loading sur les n_pcs premières composantes).
        """
        loadings_df = pca_analyzer.loadings_dataframe().iloc[:, :n_pcs]

        # Sélection des features les plus importantes
        importance = loadings_df.abs().max(axis=1)
        top_features = importance.nlargest(n_features).index
        loadings_sub = loadings_df.loc[top_features]

        vmax = np.abs(loadings_sub.values).max()
        fig, ax = plt.subplots(
            figsize=(max(8, n_pcs * 0.7), max(6, n_features * 0.28)), dpi=self.dpi
        )

        im = ax.imshow(
            loadings_sub.values,
            aspect="auto",
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
        )
        plt.colorbar(im, ax=ax, label="Loading", shrink=0.8)

        ax.set_xticks(range(n_pcs))
        ax.set_xticklabels([f"PC{i+1}" for i in range(n_pcs)], fontsize=9)
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features, fontsize=8)
        ax.set_title(
            f"Loadings PCA — Top {n_features} features × {n_pcs} premières PCs",
            fontsize=12,
        )
        ax.set_xlabel("Composante principale")
        ax.set_ylabel("Feature spectroscopique")

        plt.tight_layout()
        self._save(fig, save_path, "pca_loadings_heatmap")
        return fig, ax

    # ------------------------------------------------------------------
    # 3. Barplot loadings pour une PC
    # ------------------------------------------------------------------

    def plot_loadings_bar(
        self,
        pca_analyzer,
        pc_idx: int = 0,
        n_top: int = 15,
        save_path: Optional[str] = None,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Barplot horizontal des loadings pour la composante `pc_idx`.

        Colore en bleu (loading positif) et orange (loading négatif).
        """
        df = pca_analyzer.top_features_per_pc(pc_idx=pc_idx, n_top=n_top)
        # Tri par valeur de loading (pas abs) pour lisibilité
        df = df.sort_values("loading")

        colors = ["#DD8452" if v < 0 else "#4C72B0" for v in df["loading"]]
        fig, ax = plt.subplots(figsize=(8, max(4, n_top * 0.35)), dpi=self.dpi)

        ax.barh(
            df["feature"], df["loading"], color=colors, edgecolor="white", height=0.7
        )
        ax.axvline(0, color="black", lw=0.8)
        ax.set_xlabel("Loading")
        ax.set_title(f"PC{pc_idx + 1} — Top {n_top} features (LAMOST DR5)")
        ax.set_ylabel("Feature spectroscopique")

        # Annotation de la variance expliquée
        var = pca_analyzer.explained_variance_ratio[pc_idx] * 100
        ax.text(
            0.98,
            0.02,
            f"Variance : {var:.1f}%",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=9,
            color="gray",
        )

        plt.tight_layout()
        self._save(fig, save_path, f"pca_loadings_pc{pc_idx+1}")
        return fig, ax

    # ------------------------------------------------------------------
    # 4. Heatmap corrélations PC ↔ paramètres physiques
    # ------------------------------------------------------------------

    def plot_correlation_heatmap(
        self,
        corr_df: pd.DataFrame,
        save_path: Optional[str] = None,
        title: str = "Corrélations PC ↔ Paramètres physiques (Spearman)",
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Heatmap des corrélations entre PCs et paramètres Gaia.

        Parameters
        ----------
        corr_df : pd.DataFrame
            Sortie de PCAAnalyzer.correlations_with_params().
            Index : PC1, PC2, ... / Colonnes : paramètres physiques.
        """
        # Renommage des colonnes pour la lisibilité
        rename_map = {
            "teff_gspphot": "T_eff",
            "logg_gspphot": "log g",
            "mh_gspphot": "[Fe/H]",
            "bp_rp": "BP-RP",
            "bp_g": "BP-G",
            "g_rp": "G-RP",
            "phot_g_mean_mag": "G mag",
            "distance_gspphot": "Distance",
            "ag_gspphot": "A_G",
            "ebpminrp_gspphot": "E(BP-RP)",
            "parallax": "Parallaxe",
            "ruwe": "RUWE",
            "pmra": "μ_α",
            "pmdec": "μ_δ",
        }
        df = corr_df.rename(columns=rename_map)

        fig, ax = plt.subplots(
            figsize=(max(8, len(df.columns) * 0.7), max(5, len(df) * 0.5)),
            dpi=self.dpi,
        )
        im = ax.imshow(df.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        plt.colorbar(im, ax=ax, label="Corrélation de Spearman", shrink=0.8)

        ax.set_xticks(range(len(df.columns)))
        ax.set_xticklabels(df.columns, rotation=45, ha="right", fontsize=9)
        ax.set_yticks(range(len(df.index)))
        ax.set_yticklabels(df.index, fontsize=9)
        ax.set_title(title, fontsize=12)

        # Annotations numériques dans les cellules
        for i in range(len(df.index)):
            for j in range(len(df.columns)):
                val = df.values[i, j]
                if np.isfinite(val):
                    ax.text(
                        j,
                        i,
                        f"{val:.2f}",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="black" if abs(val) < 0.6 else "white",
                    )

        plt.tight_layout()
        self._save(fig, save_path, "pca_correlation_heatmap")
        return fig, ax

    # ------------------------------------------------------------------
    # 5. Scatter 2D embedding
    # ------------------------------------------------------------------

    def plot_embedding(
        self,
        Z: np.ndarray,
        y: Optional[np.ndarray] = None,
        color_by: Optional[np.ndarray] = None,
        color_label: str = "",
        title: str = "Embedding 2D",
        method: str = "UMAP",
        alpha: float = 0.5,
        s: float = 3.0,
        save_path: Optional[str] = None,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Scatter plot 2D d'un embedding (UMAP ou t-SNE).

        Mode 1 (y fourni) : coloration par classe (STAR / GALAXY / QSO).
        Mode 2 (color_by fourni) : coloration par paramètre continu (T_eff, ...).

        Parameters
        ----------
        Z : np.ndarray (N, 2)
            Coordonnées 2D de l'embedding.
        y : np.ndarray (N,) | None
            Étiquettes de classe (str).
        color_by : np.ndarray (N,) | None
            Valeurs continues pour coloration (ex. T_eff).
        color_label : str
            Label de la colorbar (si color_by fourni).
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        if y is not None and color_by is None:
            # Coloration par classe
            classes = np.unique(y)
            for cls in classes:
                mask = y == cls
                ax.scatter(
                    Z[mask, 0],
                    Z[mask, 1],
                    c=CLASS_COLORS.get(cls, "#888888"),
                    marker=CLASS_MARKERS.get(cls, "o"),
                    s=s,
                    alpha=alpha,
                    label=cls,
                    rasterized=True,
                )
            ax.legend(markerscale=3, framealpha=0.8, loc="best")

        elif color_by is not None:
            # Coloration par paramètre continu
            valid = np.isfinite(color_by)
            # Fond gris pour les points sans paramètre
            if (~valid).any():
                ax.scatter(
                    Z[~valid, 0],
                    Z[~valid, 1],
                    c="#cccccc",
                    s=s * 0.5,
                    alpha=0.2,
                    rasterized=True,
                    label="N/A",
                )
            sc = ax.scatter(
                Z[valid, 0],
                Z[valid, 1],
                c=color_by[valid],
                cmap=self.cmap_continuous,
                s=s,
                alpha=alpha,
                rasterized=True,
                vmin=np.percentile(color_by[valid], 2),
                vmax=np.percentile(color_by[valid], 98),
            )
            plt.colorbar(sc, ax=ax, label=color_label, shrink=0.8)

        else:
            ax.scatter(
                Z[:, 0], Z[:, 1], s=s, alpha=alpha, color="#4C72B0", rasterized=True
            )

        ax.set_xlabel(f"{method} axe 1", fontsize=11)
        ax.set_ylabel(f"{method} axe 2", fontsize=11)
        ax.set_title(title, fontsize=13)
        ax.set_aspect("equal", "box")

        plt.tight_layout()
        self._save(fig, save_path, f"embedding_{method.lower()}")
        return fig, ax

    # ------------------------------------------------------------------
    # 6. Grille multi-coloration
    # ------------------------------------------------------------------

    def plot_embedding_grid(
        self,
        Z: np.ndarray,
        y: np.ndarray,
        meta: pd.DataFrame,
        method: str = "UMAP",
        params: Optional[List[str]] = None,
        s: float = 2.5,
        alpha: float = 0.45,
        save_path: Optional[str] = None,
    ) -> Tuple[plt.Figure, np.ndarray]:
        """
        Grille de scatter plots : même embedding, colorations différentes.

        La première case colore par classe, les suivantes par paramètre physique.

        Parameters
        ----------
        params : list[str] | None
            Colonnes de meta à utiliser pour coloration.
            Défaut : ['teff_gspphot', 'logg_gspphot', 'mh_gspphot', 'bp_rp'].
        """
        if params is None:
            params = ["teff_gspphot", "logg_gspphot", "mh_gspphot", "bp_rp"]

        param_labels = {
            "teff_gspphot": r"$T_{\rm eff}$ (K)",
            "logg_gspphot": r"$\log g$",
            "mh_gspphot": r"[Fe/H]",
            "bp_rp": r"$G_{BP} - G_{RP}$",
            "bp_g": r"$G_{BP} - G$",
            "g_rp": r"$G - G_{RP}$",
            "phot_g_mean_mag": r"$G$ (mag)",
            "distance_gspphot": r"Distance (pc)",
            "ag_gspphot": r"$A_G$",
        }

        n_plots = 1 + len(params)
        ncols = min(3, n_plots)
        nrows = (n_plots + ncols - 1) // ncols

        fig, axes = plt.subplots(
            nrows, ncols, figsize=(ncols * 5, nrows * 4.5), dpi=self.dpi
        )
        axes = np.array(axes).flatten()

        # Première case : coloration par classe
        ax = axes[0]
        classes = np.unique(y)
        for cls in classes:
            mask = y == cls
            ax.scatter(
                Z[mask, 0],
                Z[mask, 1],
                c=CLASS_COLORS.get(cls, "#888"),
                marker=CLASS_MARKERS.get(cls, "o"),
                s=s,
                alpha=alpha,
                label=cls,
                rasterized=True,
            )
        ax.legend(markerscale=3, fontsize=8, framealpha=0.8)
        ax.set_title("Type spectral (LAMOST)", fontsize=11)
        ax.set_xlabel(f"{method} 1")
        ax.set_ylabel(f"{method} 2")
        ax.set_aspect("equal", "box")

        # Cases suivantes : paramètres physiques
        for k, param in enumerate(params):
            ax = axes[k + 1]
            if param not in meta.columns:
                ax.set_visible(False)
                continue
            vals = meta[param].values.astype(float)
            valid = np.isfinite(vals)

            if (~valid).any():
                ax.scatter(
                    Z[~valid, 0],
                    Z[~valid, 1],
                    c="#dddddd",
                    s=s * 0.4,
                    alpha=0.15,
                    rasterized=True,
                )

            sc = ax.scatter(
                Z[valid, 0],
                Z[valid, 1],
                c=vals[valid],
                cmap=self.cmap_continuous,
                s=s,
                alpha=alpha,
                rasterized=True,
                vmin=np.nanpercentile(vals, 2),
                vmax=np.nanpercentile(vals, 98),
            )
            plt.colorbar(sc, ax=ax, shrink=0.75, label=param_labels.get(param, param))
            ax.set_title(param_labels.get(param, param), fontsize=11)
            ax.set_xlabel(f"{method} 1")
            ax.set_ylabel(f"{method} 2")
            ax.set_aspect("equal", "box")

        # Masquer les axes vides
        for ax in axes[n_plots:]:
            ax.set_visible(False)

        fig.suptitle(
            f"Embedding {method} — LAMOST DR5 · Colorations physiques",
            fontsize=13,
            y=1.01,
        )
        plt.tight_layout()
        self._save(fig, save_path, f"embedding_{method.lower()}_grid")
        return fig, axes

    # ------------------------------------------------------------------
    # 7. Stabilité Procrustes
    # ------------------------------------------------------------------

    def plot_stability(
        self,
        stability_df: pd.DataFrame,
        save_path: Optional[str] = None,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Barplot des distances Procrustes pour l'analyse de stabilité.

        Parameters
        ----------
        stability_df : pd.DataFrame
            Sortie de EmbeddingEngine.stability_report().
        """
        fig, ax = plt.subplots(figsize=(7, 4), dpi=self.dpi)
        method = stability_df["method"].iloc[0]

        seeds = stability_df["seed"].values
        dists = stability_df["procrustes_distance"].values

        colors = ["#55A868" if d < 0.05 else "#DD8452" for d in dists]
        ax.bar(seeds, dists, color=colors, edgecolor="white", width=0.6)
        ax.axhline(0.05, color="gray", ls="--", lw=1.2, label="Seuil 0.05")
        ax.set_xlabel("Seed aléatoire")
        ax.set_ylabel("Distance Procrustes")
        ax.set_title(f"Stabilité de l'embedding {method} (analyse multi-seeds)")
        ax.legend(fontsize=9)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.text(
            0.98,
            0.97,
            f"Moyenne : {dists[1:].mean():.4f}" if len(dists) > 1 else "",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            color="gray",
        )

        plt.tight_layout()
        self._save(fig, save_path, f"stability_{method.lower()}")
        return fig, ax

    # ------------------------------------------------------------------
    # 8. Erreur de reconstruction
    # ------------------------------------------------------------------

    def plot_reconstruction_error(
        self,
        recon_df: pd.DataFrame,
        save_path: Optional[str] = None,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Courbe d'erreur de reconstruction MSE vs nombre de composantes PCA.

        Parameters
        ----------
        recon_df : pd.DataFrame
            Sortie de PCAAnalyzer.reconstruction_error_vs_n().
        """
        fig, ax = plt.subplots(figsize=(8, 5), dpi=self.dpi)

        n = recon_df["n_components"].values
        mse = recon_df["mse_mean"].values
        std = recon_df["mse_std"].values

        ax.plot(n, mse, "o-", color="#4C72B0", lw=2, ms=4, label="MSE moyenne")
        ax.fill_between(n, mse - std, mse + std, alpha=0.2, color="#4C72B0")

        # Détection du coude (différence seconde)
        if len(n) > 3:
            d2 = np.gradient(np.gradient(mse))
            elbow = n[np.argmax(d2)]
            ax.axvline(
                elbow,
                color="#DD8452",
                ls="--",
                lw=1.5,
                label=f"Coude estimé : {elbow} PCs",
            )

        ax.set_xlabel("Nombre de composantes PCA")
        ax.set_ylabel("Erreur de reconstruction (MSE)")
        ax.set_title("Qualité de reconstruction PCA vs nombre de composantes")
        ax.legend(fontsize=9)
        ax.set_yscale("log")

        plt.tight_layout()
        self._save(fig, save_path, "pca_reconstruction_error")
        return fig, ax

    # ------------------------------------------------------------------
    # 9. Diagramme HR coloré par coordonnée d'embedding
    # ------------------------------------------------------------------

    def plot_hr_diagram_embedding(
        self,
        meta: pd.DataFrame,
        Z: np.ndarray,
        component: int = 0,
        method: str = "UMAP",
        save_path: Optional[str] = None,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Diagramme HR (T_eff vs log g) coloré par la coordonnée `component`
        de l'embedding.

        Permet de visualiser si l'embedding encode la séquence principale,
        les géantes, etc. (lien avec la physique stellaire).
        """
        if "teff_gspphot" not in meta.columns or "logg_gspphot" not in meta.columns:
            raise ValueError(
                "Colonnes Gaia 'teff_gspphot' et 'logg_gspphot' requises dans meta."
            )

        teff = meta["teff_gspphot"].values.astype(float)
        logg = meta["logg_gspphot"].values.astype(float)
        z_coord = Z[:, component]

        valid = np.isfinite(teff) & np.isfinite(logg) & np.isfinite(z_coord)

        fig, ax = plt.subplots(figsize=(8, 7), dpi=self.dpi)
        sc = ax.scatter(
            teff[valid],
            logg[valid],
            c=z_coord[valid],
            cmap=self.cmap_continuous,
            s=2,
            alpha=0.5,
            rasterized=True,
        )
        plt.colorbar(sc, ax=ax, label=f"{method} axe {component + 1}")

        ax.invert_xaxis()
        ax.invert_yaxis()
        ax.set_xlabel(r"$T_{\rm eff}$ (K)", fontsize=12)
        ax.set_ylabel(r"$\log g$ (dex)", fontsize=12)
        ax.set_title(
            f"Diagramme HR — coloré par {method} axe {component + 1}\n"
            f"(LAMOST DR5 × Gaia DR3)",
            fontsize=12,
        )

        plt.tight_layout()
        self._save(fig, save_path, f"hr_diagram_{method.lower()}_ax{component+1}")
        return fig, ax

    # ------------------------------------------------------------------
    # Helper : sauvegarde
    # ------------------------------------------------------------------

    def _save(
        self,
        fig: plt.Figure,
        save_path: Optional[str],
        default_name: str,
    ) -> None:
        """Sauvegarde la figure si save_path fourni ou output_dir défini."""
        if save_path is not None:
            path = Path(save_path)
        elif self.output_dir is not None:
            path = self.output_dir / f"{default_name}.png"
        else:
            return

        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches="tight", dpi=self.dpi)
        logger.info("Figure sauvegardée : %s", path)

    # ==================================================================
    # Visualisations Autoencodeur (PHY-3500 Notebook 03)
    # ==================================================================

    def plot_training_history(self, history, save_path=None):
        """Courbes loss train/val + learning rate sur l'entraînement AE."""
        train_loss = history["train_loss"]
        val_loss = history["val_loss"]
        lr_hist = history.get("lr_history", [])
        epochs = range(1, len(train_loss) + 1)
        has_lr = len(lr_hist) > 0

        fig, axes = plt.subplots(
            1, 1 + int(has_lr), figsize=(self.figsize[0], 4), dpi=self.dpi
        )
        if not has_lr:
            axes = [axes]

        ax = axes[0]
        ax.plot(epochs, train_loss, color="#4C72B0", lw=2, label="Train")
        ax.plot(epochs, val_loss, color="#DD8452", lw=2, ls="--", label="Validation")
        best_epoch = int(np.argmin(val_loss)) + 1
        best_val = min(val_loss)
        ax.axvline(
            best_epoch,
            color="#55A868",
            ls=":",
            lw=1.5,
            label=f"Best epoch {best_epoch}",
        )
        ax.annotate(
            f"Best: {best_val:.5f}",
            xy=(best_epoch, best_val),
            xytext=(best_epoch + max(1, len(list(epochs)) // 15), best_val * 1.05),
            fontsize=8,
            color="#55A868",
            arrowprops=dict(arrowstyle="->", color="#55A868"),
        )
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE Loss")
        ax.set_title("Courbe d'apprentissage — Autoencodeur")
        ax.legend(fontsize=9)
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)

        if has_lr:
            ax2 = axes[1]
            ax2.plot(epochs, lr_hist, color="#8172B2", lw=2)
            ax2.set_xlabel("Epoch")
            ax2.set_ylabel("Learning rate")
            ax2.set_title("Évolution du learning rate")
            ax2.set_yscale("log")
            ax2.grid(True, alpha=0.3)

        fig.suptitle(
            "Autoencodeur — Historique d'entraînement (LAMOST DR5)", fontsize=13, y=1.02
        )
        plt.tight_layout()
        if save_path is not None:
            fig.savefig(save_path, bbox_inches="tight", dpi=self.dpi)
            logger.info("Figure sauvegardée : %s", save_path)
        return fig, axes

    def plot_ae_embedding(self, Z, meta, y, save_path=None):
        """Grille espace latent 2D coloré par classe + paramètres physiques."""
        if Z.shape[1] != 2:
            raise ValueError(f"latent_dim doit être 2, reçu {Z.shape[1]}")

        CMAPS = {
            "teff_gspphot": ("T_eff (K)", "plasma"),
            "logg_gspphot": ("log g", "viridis"),
            "mh_gspphot": ("[Fe/H]", "RdYlBu_r"),
            "bp_rp": ("G_BP-G_RP", "RdYlBu_r"),
        }
        available = [(k, v) for k, v in CMAPS.items() if k in meta.columns]
        fig, axes = plt.subplots(
            1, 1 + len(available), figsize=(5 * (1 + len(available)), 5), dpi=self.dpi
        )

        PALETTE = {"STAR": "#4C72B0", "GALAXY": "#DD8452", "QSO": "#55A868"}
        MARKERS = {"STAR": "o", "GALAXY": "s", "QSO": "^"}
        ax = axes[0]
        for cls in np.unique(y):
            mask = y == cls
            ax.scatter(
                Z[mask, 0],
                Z[mask, 1],
                c=PALETTE.get(cls, "gray"),
                marker=MARKERS.get(cls, "o"),
                s=3,
                alpha=0.45,
                linewidths=0,
                label=cls,
                rasterized=True,
            )
        ax.set_xlabel("Latent axe 1")
        ax.set_ylabel("Latent axe 2")
        ax.set_title("Type spectral (LAMOST)")
        ax.legend(fontsize=9, markerscale=4)

        for ax_i, (col, (label, cmap)) in zip(axes[1:], available):
            vals = meta[col].values.astype(float)
            valid = np.isfinite(vals)
            vmin, vmax = np.nanpercentile(vals[valid], [2, 98])
            sc = ax_i.scatter(
                Z[valid, 0],
                Z[valid, 1],
                c=vals[valid],
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                s=3,
                alpha=0.55,
                linewidths=0,
                rasterized=True,
            )
            plt.colorbar(sc, ax=ax_i, label=label, fraction=0.046, pad=0.04)
            ax_i.set_xlabel("Latent axe 1")
            ax_i.set_title(label)

        fig.suptitle(
            "Autoencodeur — Espace latent 2D · Colorations physiques",
            fontsize=14,
            y=1.02,
        )
        plt.tight_layout()
        if save_path is not None:
            fig.savefig(save_path, bbox_inches="tight", dpi=self.dpi)
            logger.info("Figure sauvegardée : %s", save_path)
        return fig, axes

    def plot_ae_reconstruction(
        self, X, X_recon, feature_names, n_samples=5, y=None, save_path=None
    ):
        """Comparaison features originales vs reconstruites pour N échantillons."""
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X), size=min(n_samples, len(X)), replace=False)
        fig, axes = plt.subplots(
            n_samples, 1, figsize=(14, 2.5 * n_samples), dpi=self.dpi
        )
        if n_samples == 1:
            axes = [axes]
        x_pos = np.arange(len(feature_names))
        for i, (ax, si) in enumerate(zip(axes, idx)):
            orig = X[si]
            recon = X_recon[si]
            mse_i = float(np.mean((orig - recon) ** 2))
            ax.plot(x_pos, orig, color="#4C72B0", lw=1.5, alpha=0.9, label="Original")
            ax.plot(
                x_pos,
                recon,
                color="#DD8452",
                lw=1.5,
                alpha=0.9,
                ls="--",
                label="Reconstruit",
            )
            ax.fill_between(x_pos, orig, recon, alpha=0.15, color="#DD8452")
            title = f"Spectre #{si}"
            if y is not None:
                title += f" · {y[si]}"
            title += f"  |  MSE = {mse_i:.5f}"
            ax.set_title(title, fontsize=9)
            ax.set_ylabel("Valeur standardisée")
            if i == n_samples - 1:
                ax.set_xlabel("Feature index")
            if i == 0:
                ax.legend(fontsize=8, loc="upper right")
            ax.grid(True, alpha=0.2)
        fig.suptitle(
            "Autoencodeur — Reconstruction features (LAMOST DR5)", fontsize=13, y=1.01
        )
        plt.tight_layout()
        if save_path is not None:
            fig.savefig(save_path, bbox_inches="tight", dpi=self.dpi)
            logger.info("Figure sauvegardée : %s", save_path)
        return fig, axes

    def plot_ae_vs_pca(self, comparison_df, save_path=None):
        """Comparaison MSE autoencodeur vs PCA en fonction du nombre de composantes."""
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        pca_df = comparison_df[comparison_df["method"] == "PCA"].sort_values(
            "n_components"
        )
        ae_df = comparison_df[comparison_df["method"] != "PCA"]
        ax.plot(
            pca_df["n_components"],
            pca_df["mse_mean"],
            "o-",
            color="#4C72B0",
            lw=2,
            ms=5,
            label="PCA",
        )
        for _, row in ae_df.iterrows():
            ax.axhline(
                row["mse_mean"], color="#DD8452", ls="--", lw=2, label=row["method"]
            )
            ax.annotate(
                f"{row['method']}\nMSE={row['mse_mean']:.5f}",
                xy=(pca_df["n_components"].max() * 0.55, row["mse_mean"] * 1.03),
                color="#DD8452",
                fontsize=9,
            )
        ax.set_xlabel("Nombre de composantes")
        ax.set_ylabel("MSE moyenne")
        ax.set_title("Autoencodeur vs PCA — Qualité de reconstruction")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        if save_path is not None:
            fig.savefig(save_path, bbox_inches="tight", dpi=self.dpi)
            logger.info("Figure sauvegardée : %s", save_path)
        return fig, ax

    def plot_latent_interpolation(
        self,
        Z_interp,
        X_interp,
        feature_names,
        label_a="Étoile A",
        label_b="Étoile B",
        save_path=None,
    ):
        """Trajectoire d'interpolation dans l'espace latent 2D."""
        n_steps = len(Z_interp)
        alphas = np.linspace(0, 1, n_steps)
        fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=self.dpi)
        ax = axes[0]
        sc = ax.scatter(
            Z_interp[:, 0], Z_interp[:, 1], c=alphas, cmap="RdYlBu_r", s=60, zorder=3
        )
        ax.plot(
            Z_interp[:, 0], Z_interp[:, 1], color="gray", lw=1.5, zorder=2, alpha=0.6
        )
        ax.scatter(*Z_interp[0], s=120, c="blue", zorder=4, marker="*", label=label_a)
        ax.scatter(*Z_interp[-1], s=120, c="red", zorder=4, marker="*", label=label_b)
        plt.colorbar(sc, ax=ax, label="α (interpolation)")
        ax.set_xlabel("Latent axe 1")
        ax.set_ylabel("Latent axe 2")
        ax.set_title("Trajectoire dans l'espace latent")
        ax.legend(fontsize=9)

        ax2 = axes[1]
        top_feat = np.argsort(np.std(X_interp, axis=0))[::-1][:20]
        cmap = plt.get_cmap("RdYlBu_r")
        for step, alpha in enumerate(alphas):
            ax2.plot(
                range(len(top_feat)),
                X_interp[step, top_feat],
                color=cmap(alpha),
                lw=1.2,
                alpha=0.7,
            )
        ax2.set_xticks(range(len(top_feat)))
        ax2.set_xticklabels(
            [feature_names[i] for i in top_feat], rotation=45, ha="right", fontsize=7
        )
        ax2.set_ylabel("Valeur standardisée")
        ax2.set_title("Top 20 features le long de l'interpolation")
        ax2.grid(True, alpha=0.2)
        fig.suptitle(
            f"Interpolation latente : {label_a} → {label_b}", fontsize=13, y=1.02
        )
        plt.tight_layout()
        if save_path is not None:
            fig.savefig(save_path, bbox_inches="tight", dpi=self.dpi)
            logger.info("Figure sauvegardée : %s", save_path)
        return fig, axes

    def plot_ae_error_distribution(self, mse_per_sample, y, save_path=None):
        """Distribution des erreurs de reconstruction par classe (histogramme)."""
        classes = np.unique(y)
        palette = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
        fig, ax = plt.subplots(figsize=(9, 5), dpi=self.dpi)
        for cls, color in zip(classes, palette):
            mask = y == cls
            vals = mse_per_sample[mask]
            ax.hist(
                vals,
                bins=60,
                alpha=0.55,
                color=color,
                density=True,
                label=f"{cls} (n={mask.sum()}, μ={vals.mean():.4f})",
            )
            ax.axvline(np.median(vals), color=color, lw=2, ls="--")
        ax.set_xlabel("MSE de reconstruction")
        ax.set_ylabel("Densité")
        ax.set_title("Distribution des erreurs de reconstruction par classe")
        ax.legend(fontsize=9)
        ax.set_xlim(left=0)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        if save_path is not None:
            fig.savefig(save_path, bbox_inches="tight", dpi=self.dpi)
            logger.info("Figure sauvegardée : %s", save_path)
        return fig, ax
