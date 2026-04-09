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
9. plot_eigenspectra()           — eigen-spectres PCA sur grille wavelength
10. plot_hr_diagram_embedding()  — HR diagram coloré par coordonnée embedding

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
import matplotlib.patches as mpatches
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
        # Paramètres de rendu réutilisés par toutes les figures.
        self.figsize = figsize
        self.dpi = dpi
        self.cmap_continuous = cmap_continuous
        # Dossier de sortie optionnel: utilisé si save_path n'est pas fourni.
        self.output_dir = Path(output_dir) if output_dir else None
        if self.output_dir:
            # Création proactive pour éviter un échec silencieux à l'export.
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
        # 1) Restreint l'analyse aux max_pcs premières composantes.
        ratio = pca_analyzer.explained_variance_ratio[:max_pcs]
        # 2) Transforme variance individuelle -> variance cumulée.
        cumvar = np.cumsum(ratio)
        pcs = np.arange(1, len(ratio) + 1)

        # Nombre de PCs requis sur l'ensemble complet du modèle.
        n_thresh = pca_analyzer.n_components_for_variance(threshold)

        # Deux panneaux complémentaires: contribution locale puis accumulation globale.
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
        # La verticale matérialise le nombre de composantes minimal visé.
        ax.axvline(n_thresh, color="#55A868", ls=":", lw=1.5, label=f"{n_thresh} PCs")

        # Clamp visuel: n_thresh peut dépasser max_pcs affiché.
        n_thresh_plot = min(n_thresh, len(cumvar))
        idx = n_thresh_plot - 1

        ax.annotate(
            # Le texte conserve la valeur réelle de n_thresh, même si la vue est tronquée.
            f"{n_thresh} PCs\n→ {cumvar[idx]*100:.1f}%",
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
        # Sous-matrice: colonnes PC1..PCn_pcs.
        loadings_df = pca_analyzer.loadings_dataframe().iloc[:, :n_pcs]

        # Sélection des features les plus importantes
        # Score d'importance: amplitude max absolue observée sur les PCs affichées.
        importance = loadings_df.abs().max(axis=1)
        top_features = importance.nlargest(n_features).index
        loadings_sub = loadings_df.loc[top_features]

        # Échelle symétrique autour de 0 pour comparer signe et amplitude.
        vmax = np.abs(loadings_sub.values).max()
        # Taille de figure adaptative: lisible même quand n_features ou n_pcs augmente.
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
        # Extrait les top features dominantes pour une PC donnée.
        df = pca_analyzer.top_features_per_pc(pc_idx=pc_idx, n_top=n_top)
        # Tri par valeur de loading (pas abs) pour lisibilité
        df = df.sort_values("loading")

        # Couleur encodant le signe physique de la contribution.
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

    def plot_loadings_family_donut(
        self,
        contrib_pc1: pd.DataFrame,
        contrib_pc2: pd.DataFrame,
        color_map: dict,
        save_path: Optional[str] = None,
    ) -> Tuple[plt.Figure, np.ndarray]:
        """Camemberts (donut) des contributions par famille pour PC1 et PC2."""
        # Un panneau par composante pour comparer rapidement les dominances.
        fig, axes = plt.subplots(1, 2, figsize=(16, 7), dpi=self.dpi)

        self._draw_loadings_family_donut(axes[0], contrib_pc1, "PC1", color_map)
        self._draw_loadings_family_donut(axes[1], contrib_pc2, "PC2", color_map)

        fig.suptitle(
            "Décomposition des loadings PCA par famille spectroscopique\n"
            "(contribution quadratique normalisée — LAMOST DR5 × 183 features)",
            fontsize=13,
            fontweight="bold",
            y=1.02,
        )
        plt.tight_layout()
        self._save(fig, save_path, "pca_loadings_family_donut")
        return fig, axes

    def plot_loadings_family_bar(
        self,
        contrib_pc1: pd.DataFrame,
        contrib_pc2: pd.DataFrame,
        color_map: dict,
        save_path: Optional[str] = None,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Barres côte-à-côte des contributions par famille pour PC1 et PC2."""
        # Union ordonnée des familles présentes dans PC1 et/ou PC2.
        all_fams = list(
            dict.fromkeys(
                contrib_pc1["family"].tolist() + contrib_pc2["family"].tolist()
            )
        )
        c1_dict = dict(zip(contrib_pc1["family"], contrib_pc1["weight"]))
        c2_dict = dict(zip(contrib_pc2["family"], contrib_pc2["weight"]))

        # Tri piloté par PC1 pour conserver la lecture "composante principale d'abord".
        all_fams = sorted(all_fams, key=lambda f: c1_dict.get(f, 0), reverse=True)

        x = np.arange(len(all_fams))
        w = 0.35
        vals1 = [c1_dict.get(f, 0) * 100 for f in all_fams]
        vals2 = [c2_dict.get(f, 0) * 100 for f in all_fams]
        # Une couleur de famille partagée entre PC1 et PC2 aide la comparaison directe.
        colors_bar = [color_map.get(f, "#CCCCCC") for f in all_fams]

        fig, ax = plt.subplots(figsize=(14, 6), dpi=self.dpi)
        ax.bar(
            x - w / 2,
            vals1,
            w,
            color=colors_bar,
            alpha=0.92,
            edgecolor="white",
            linewidth=0.5,
            label="PC1",
        )
        ax.bar(
            x + w / 2,
            vals2,
            w,
            color=colors_bar,
            alpha=0.55,
            edgecolor="white",
            linewidth=0.5,
            label="PC2",
            hatch="//",
        )
        # Lecture visuelle: opaque = PC1, hachuré/translucide = PC2.

        ax.set_xticks(x)
        ax.set_xticklabels(
            [f.replace("\n", "\n") for f in all_fams],
            rotation=30,
            ha="right",
            fontsize=8,
        )
        ax.set_ylabel("Contribution (% variance loading²)", fontsize=10)
        ax.set_title(
            "Contribution par famille spectroscopique — PC1 vs PC2\n"
            "(opaque = PC1, hachuré = PC2)",
            fontsize=12,
            fontweight="bold",
        )
        ax.legend(fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(0, max(vals1 + vals2) * 1.15)

        plt.tight_layout()
        self._save(fig, save_path, "pca_loadings_family_bar")
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

        # Dimensions dynamiques pour conserver une grille lisible quel que soit corr_df.
        fig, ax = plt.subplots(
            figsize=(max(8, len(df.columns) * 0.7), max(5, len(df) * 0.5)),
            dpi=self.dpi,
        )
        # Carte divergente centrée sur 0: rouge négatif, bleu positif.
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
                    # Couleur du texte adaptée au contraste local de la cellule.
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
            # Priorité au mode catégoriel dès qu'on a des labels de classe explicites.
            # Coloration par classe
            # Chaque classe est tracée séparément pour gérer couleur + marqueur.
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
                # Clipping robuste aux outliers extrêmes.
                vmin=np.percentile(color_by[valid], 2),
                vmax=np.percentile(color_by[valid], 98),
            )
            plt.colorbar(sc, ax=ax, label=color_label, shrink=0.8)

        else:
            # Fallback monochrome si aucune variable de couleur n'est fournie.
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
        # Grille automatiquement compacte pour éviter des sous-figures trop étirées.

        fig, axes = plt.subplots(
            nrows, ncols, figsize=(ncols * 5, nrows * 4.5), dpi=self.dpi
        )
        # `flatten` uniformise le parcours, que la grille soit 1xN ou NxM.
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
                # Paramètre absent: case masquée explicitement.
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
                # Bornes robustes pour limiter l'effet des valeurs extrêmes.
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

        # Encodage visuel immédiat: vert stable, orange moins stable.
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
            # Convention actuelle: on exclut souvent seed de référence (index 0) de la moyenne.
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
            # Approximation simple du coude via courbure discrète (2e dérivée).
            d2 = np.gradient(np.gradient(mse))
            # Le maximum de courbure est pris comme compromis erreur/complexité.
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
    # 9. Eigen-spectres PCA
    # ------------------------------------------------------------------

    def plot_eigenspectra(
        self,
        pca_analyzer,
        wl_grid: np.ndarray,
        spectral_lines: Optional[dict[str, float]] = None,
        n_components: int = 3,
        save_path: Optional[str] = None,
    ) -> Tuple[plt.Figure, np.ndarray]:
        """
        Trace les `n_components` premiers eigen-spectres de la PCA.

        Parameters
        ----------
        pca_analyzer : PCAAnalyzer
            Objet PCAAnalyzer ajusté sur spectres bruts.
        wl_grid : np.ndarray
            Grille wavelength (Å).
        spectral_lines : dict[str, float] | None
            Dictionnaire label -> longueur d'onde pour annotation des raies.
            Si None, utilise un jeu de raies classiques (Balmer, Ca, Mg, Na).
        n_components : int
            Nombre de composantes à afficher.
        """
        if spectral_lines is None:
            spectral_lines = {
                r"H$\\alpha$": 6563,
                r"H$\\beta$": 4861,
                r"H$\\gamma$": 4340,
                r"H$\\delta$": 4102,
                "Ca II K": 3933,
                "Ca II H": 3968,
                "Mg b": 5175,
                "Na D": 5893,
            }

        # Garde-fou: force une fenêtre [1, n_pcs_disponibles].
        n_components = max(1, min(n_components, pca_analyzer.loadings.shape[0]))

        fig, axes = plt.subplots(
            n_components, 1, figsize=(12, 8), dpi=self.dpi, sharex=True
        )
        if n_components == 1:
            axes = np.array([axes])

        wl = np.asarray(wl_grid)
        for i, ax in enumerate(axes):
            vec = pca_analyzer.loadings[i]
            var_pct = pca_analyzer.explained_variance_ratio[i] * 100
            # Tracé de l'eigenvecteur spectral de la composante i.
            ax.plot(wl, vec, lw=0.7, color=f"C{i}")
            ax.axhline(0, color="gray", lw=0.5, ls="--")
            ax.set_ylabel(f"PC{i+1}  ({var_pct:.1f}%)", fontsize=10)
            ax.grid(True, alpha=0.25)

            ymax = np.abs(vec).max()
            for nom, lam in spectral_lines.items():
                if wl.min() < lam < wl.max():
                    # Marqueurs verticaux des raies classiques pour lecture physique.
                    ax.axvline(lam, color="crimson", lw=0.6, alpha=0.6)
                    ax.text(
                        lam,
                        ymax * 0.88,
                        nom,
                        rotation=90,
                        fontsize=6,
                        color="crimson",
                        ha="center",
                        va="top",
                    )

        axes[-1].set_xlabel("Longueur d'onde (Å)", fontsize=11)
        fig.suptitle(
            "Eigen-spectres PCA — 3 premières composantes (LAMOST DR5)", fontsize=13
        )
        plt.tight_layout()
        self._save(fig, save_path, "pca_eigenspectra")
        return fig, axes

    # ------------------------------------------------------------------
    # 10. Diagramme HR coloré par coordonnée d'embedding
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

        # On n'affiche que les triplets complets (teff, logg, coordonnée embedding).
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

        # Convention HR: T_eff décroissant vers la droite, log g vers le bas.
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

    def _draw_loadings_family_donut(self, ax, contrib_df, title, color_map):
        # Les labels sont préparés séparément pour construire légende et couleurs.
        labels = contrib_df["family"].tolist()
        sizes = contrib_df["weight"].tolist()
        colors = [color_map.get(label, "#CCCCCC") for label in labels]

        _, _, autotexts = ax.pie(
            sizes,
            labels=None,
            colors=colors,
            # Affiche le pourcentage seulement si le secteur est suffisamment grand.
            autopct=lambda p: f"{p:.1f}%" if p > 4 else "",
            pctdistance=0.75,
            startangle=90,
            wedgeprops={"width": 0.52, "edgecolor": "white", "linewidth": 1.2},
        )
        for at in autotexts:
            at.set_fontsize(8)
            at.set_fontweight("bold")
            at.set_color("white")

        ax.text(
            0,
            0,
            title,
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
            color="#2A3B4C",
        )

        patches = [
            mpatches.Patch(
                color=color_val,
                label=f"{family_label.replace(chr(10), ' ')} ({weight_val*100:.1f}%)",
            )
            for family_label, weight_val, color_val in zip(labels, sizes, colors)
            # Coupe les micro-contributions pour éviter une légende illisible.
            if weight_val > 0.01
        ]
        ax.legend(
            handles=patches,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.28),
            ncol=2,
            fontsize=7.5,
            framealpha=0.9,
            edgecolor="#CCCCCC",
        )

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
            # Chemin par défaut construit depuis output_dir + nom canonique de figure.
            path = self.output_dir / f"{default_name}.png"
        else:
            # Aucun chemin d'export défini: on garde uniquement l'affichage en mémoire.
            return

        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches="tight", dpi=self.dpi)
        logger.info("Figure sauvegardée : %s", path)

    # ==================================================================
    # Visualisations Autoencodeur (PHY-3500 Notebook 03)
    # ==================================================================

    def plot_training_history(self, history, save_path=None):
        """Courbes loss train/val + learning rate sur l'entraînement AE."""
        # Extraction des séries minimales attendues du dictionnaire history.
        train_loss = history["train_loss"]
        val_loss = history["val_loss"]
        lr_hist = history.get("lr_history", [])
        # Axe temporel explicite en epochs (1..N).
        epochs = range(1, len(train_loss) + 1)
        # Le panneau LR n'est ajouté que si l'historique existe réellement.
        has_lr = len(lr_hist) > 0

        fig, axes = plt.subplots(
            1, 1 + int(has_lr), figsize=(self.figsize[0], 4), dpi=self.dpi
        )
        if not has_lr:
            # Uniformise l'itération, même sans second panneau LR.
            axes = [axes]

        ax = axes[0]
        # Courbes principales: ajustement (train) vs généralisation (val).
        ax.plot(epochs, train_loss, color="#4C72B0", lw=2, label="Train")
        ax.plot(epochs, val_loss, color="#DD8452", lw=2, ls="--", label="Validation")
        # Best epoch défini sur la val_loss minimale (critère de sélection modèle).
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
        # Log-scale pour conserver lisibilité au début et en fin d'entraînement.
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)

        if has_lr:
            ax2 = axes[1]
            # Historique du scheduler pour interpréter la dynamique d'optimisation.
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
        # Paramètres réellement disponibles dans meta pour éviter les KeyError.
        available = [(k, v) for k, v in CMAPS.items() if k in meta.columns]
        # Une colonne pour les classes + une colonne par paramètre Gaia disponible.
        fig, axes = plt.subplots(
            1, 1 + len(available), figsize=(5 * (1 + len(available)), 5), dpi=self.dpi
        )

        PALETTE = {"STAR": "#4C72B0", "GALAXY": "#DD8452", "QSO": "#55A868"}
        MARKERS = {"STAR": "o", "GALAXY": "s", "QSO": "^"}
        ax = axes[0]
        for cls in np.unique(y):
            mask = y == cls
            # Tracé couche par couche pour respecter marqueurs et palette de classes.
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
            # Clipping percentile 2-98 pour limiter l'influence des outliers.
            vmin, vmax = np.nanpercentile(vals[valid], [2, 98])
            # Panneau continu: même géométrie, couleur = paramètre astrophysique.
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
        # Échantillonnage reproductible des exemples affichés.
        rng = np.random.default_rng(42)
        # Indices d'exemples affichés (sans remise).
        idx = rng.choice(len(X), size=min(n_samples, len(X)), replace=False)
        fig, axes = plt.subplots(
            n_samples, 1, figsize=(14, 2.5 * n_samples), dpi=self.dpi
        )
        if n_samples == 1:
            axes = [axes]
        x_pos = np.arange(len(feature_names))
        for i, (ax, si) in enumerate(zip(axes, idx)):
            # Comparaison directe profil original vs reconstruit pour un spectre.
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
                # Ajoute le label de classe si disponible pour contexte visuel.
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
        # Sépare la référence PCA des lignes autoencodeur(s).
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
            # Chaque AE est représenté par une ligne horizontale de performance.
            ax.axhline(
                row["mse_mean"], color="#DD8452", ls="--", lw=2, label=row["method"]
            )
            ax.annotate(
                # Position d'annotation choisie pour rester lisible sur la courbe PCA.
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
        # Paramètre d'interpolation alpha de 0 (A) à 1 (B).
        n_steps = len(Z_interp)
        alphas = np.linspace(0, 1, n_steps)
        fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=self.dpi)
        ax = axes[0]
        # Panneau gauche: géométrie latente + codage de progression alpha.
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
        # Sélectionne les 20 features qui varient le plus le long de la trajectoire.
        top_feat = np.argsort(np.std(X_interp, axis=0))[::-1][:20]
        cmap = plt.get_cmap("RdYlBu_r")
        for step, alpha in enumerate(alphas):
            # Une courbe par étape d'interpolation pour visualiser l'évolution continue.
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
            # Histogrammes normalisés pour comparer les formes de distribution.
            ax.hist(
                vals,
                bins=60,
                alpha=0.55,
                color=color,
                density=True,
                label=f"{cls} (n={mask.sum()}, μ={vals.mean():.4f})",
            )
            # Médiane tracée pour robuste comparaison inter-classes.
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

    # ==================================================================
    # SECTION II — Méthodes ajoutées (refactorisation cellules notebooks)
    # ==================================================================

    # ── HDBSCAN ────────────────────────────────────────────────────────

    def plot_hdbscan_clusters(
        self,
        Z: np.ndarray,
        hdb,  # HDBSCANAnalyzer
        y: np.ndarray,
        save_path=None,
    ):
        """Panneau double : clusters HDBSCAN annotés + classes LAMOST."""
        import matplotlib.colors as mcolors
        from dimred.dimred_visualizer import CLASS_COLORS, CLASS_MARKERS

        # Récupère les sorties déjà calculées par l'analyseur HDBSCAN.
        n = hdb.n_clusters_
        cl = hdb.labels_
        color_map = hdb.color_map_
        cluster_ids = hdb.cluster_ids_
        n_noise = hdb.n_noise_
        cmap_c = plt.get_cmap("turbo", max(1, n))
        noise_mask = cl == -1

        fig, axes = plt.subplots(1, 2, figsize=(18, 8), dpi=self.dpi)

        # Gauche : clusters annotés
        ax = axes[0]
        # Bruit (-1) en fond léger pour ne pas masquer les structures clusterisées.
        ax.scatter(
            Z[noise_mask, 0],
            Z[noise_mask, 1],
            c=[color_map[-1]],
            s=1.5,
            alpha=0.25,
            rasterized=True,
            linewidths=0,
            zorder=1,
        )
        for cid in cluster_ids:
            mask = cl == cid
            # Nuage principal du cluster cid.
            ax.scatter(
                Z[mask, 0],
                Z[mask, 1],
                c=[color_map[cid]],
                s=2,
                alpha=0.75,
                rasterized=True,
                linewidths=0,
                zorder=2,
            )
            # Annotation centrée sur le barycentre visuel du cluster.
            cx, cy = Z[mask].mean(axis=0)
            ax.text(
                cx,
                cy,
                f"C{cid}",
                fontsize=7,
                fontweight="bold",
                ha="center",
                va="center",
                color="white",
                bbox=dict(
                    boxstyle="round,pad=0.15", fc=color_map[cid], ec="none", alpha=0.85
                ),
            )
        sm = plt.cm.ScalarMappable(
            cmap=cmap_c, norm=mcolors.Normalize(vmin=0, vmax=n - 1)
        )
        sm.set_array([])
        # Colorbar dédiée aux IDs, indépendante de la coloration réelle par map.
        plt.colorbar(
            sm, ax=ax, fraction=0.035, pad=0.02, label="Cluster ID"
        ).ax.tick_params(labelsize=8)
        ax.text(
            0.02,
            0.98,
            # Résumé compact du taux de bruit dans la vue courante.
            f"Bruit : n={n_noise} ({100*n_noise/len(cl):.1f}%)",
            transform=ax.transAxes,
            fontsize=8,
            va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.85),
        )
        ax.set_xlabel("UMAP axe 1")
        ax.set_ylabel("UMAP axe 2")
        ax.set_title(
            f"HDBSCAN — {n} populations identifiées", fontsize=13, fontweight="bold"
        )
        ax.grid(True, alpha=0.2)

        # Droite : classes LAMOST
        ax = axes[1]
        for cls in np.unique(y):
            mask = y == cls
            # Même embedding, mais référence supervisée par classe spectrale.
            ax.scatter(
                Z[mask, 0],
                Z[mask, 1],
                c=CLASS_COLORS.get(cls, "#888"),
                marker=CLASS_MARKERS.get(cls, "o"),
                s=2,
                alpha=0.45,
                label=f"{cls} (n={mask.sum()})",
                rasterized=True,
                linewidths=0,
            )
        ax.set_xlabel("UMAP axe 1")
        ax.set_title("Classes LAMOST (référence)", fontsize=13, fontweight="bold")
        ax.legend(markerscale=4, fontsize=10, framealpha=0.9)
        ax.grid(True, alpha=0.2)

        fig.suptitle(
            f"HDBSCAN sur UMAP — {n} clusters · {n_noise} points bruit "
            f"({100*n_noise/len(cl):.1f}%)\nLAMOST DR5 × Gaia DR3",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()
        self._save(fig, save_path, "umap_hdbscan_clusters")
        return fig, axes

    def plot_hdbscan_hr(
        self,
        Z: np.ndarray,
        hdb,
        meta: "pd.DataFrame",
        save_path=None,
    ):
        """Panneau double : HR coloré par cluster (top-8 annotés) + T_eff."""
        import matplotlib.colors as mcolors

        # Variables de clustering et configuration visuelle.
        cl = hdb.labels_
        cluster_ids = hdb.cluster_ids_
        color_map = hdb.color_map_
        n = hdb.n_clusters_
        noise_mask = cl == -1

        teff_col, logg_col = "teff_gspphot", "logg_gspphot"
        if teff_col not in meta.columns or logg_col not in meta.columns:
            logger.warning("plot_hdbscan_hr : colonnes Gaia manquantes.")
            return None, None

        # Filtre physique valide pour le diagramme HR.
        teff = meta[teff_col].values.astype(float)
        logg = meta[logg_col].values.astype(float)
        valid_hr = np.isfinite(teff) & np.isfinite(logg)

        # Classement des clusters par effectif HR pour annoter les plus robustes.
        counts_hr = {cid: ((cl == cid) & valid_hr).sum() for cid in cluster_ids}
        top8 = sorted(counts_hr, key=counts_hr.get, reverse=True)[:8]

        cmap_c = plt.get_cmap("turbo", max(1, n))
        fig, axes = plt.subplots(1, 2, figsize=(18, 8), dpi=self.dpi)

        ax = axes[0]
        # Fond bruit discret pour contexte sans surcharger la figure.
        ax.scatter(
            teff[noise_mask & valid_hr],
            logg[noise_mask & valid_hr],
            c="lightgray",
            s=1,
            alpha=0.15,
            rasterized=True,
            linewidths=0,
            zorder=1,
        )
        for cid in cluster_ids:
            m = (cl == cid) & valid_hr
            # Trace chaque cluster dans le plan HR.
            ax.scatter(
                teff[m],
                logg[m],
                c=[color_map[cid]],
                s=3,
                alpha=0.65,
                rasterized=True,
                linewidths=0,
                zorder=2,
            )
        for cid in top8:
            m = (cl == cid) & valid_hr
            if m.sum() < 20:
                continue
            # Étiquette des clusters dominants au barycentre médian.
            mt, mg = np.median(teff[m]), np.median(logg[m])
            ax.text(
                mt,
                mg,
                f"C{cid}\nn={m.sum()}",
                fontsize=7.5,
                fontweight="bold",
                ha="center",
                va="center",
                color="white",
                bbox=dict(
                    boxstyle="round,pad=0.25", fc=color_map[cid], ec="none", alpha=0.90
                ),
                zorder=5,
            )
        ax.invert_xaxis()
        ax.invert_yaxis()
        ax.set_xlabel("$T_{eff}$ (K)", fontsize=12)
        ax.set_ylabel("log g (dex)", fontsize=11)
        ax.set_title(
            "Diagramme HR — clusters HDBSCAN (top-8 annotés)",
            fontsize=12,
            fontweight="bold",
        )
        ax.grid(True, alpha=0.2)
        sm2 = plt.cm.ScalarMappable(
            cmap=cmap_c, norm=mcolors.Normalize(vmin=0, vmax=n - 1)
        )
        sm2.set_array([])
        plt.colorbar(
            sm2, ax=ax, fraction=0.035, pad=0.02, label="Cluster ID"
        ).ax.tick_params(labelsize=8)

        ax2 = axes[1]
        valid_teff = np.isfinite(teff)
        # Embedding UMAP coloré par température pour relier espace latent et HR.
        sc = ax2.scatter(
            Z[valid_teff, 0],
            Z[valid_teff, 1],
            c=teff[valid_teff],
            cmap="plasma",
            vmin=np.nanpercentile(teff[valid_teff], 2),
            vmax=np.nanpercentile(teff[valid_teff], 98),
            s=2,
            alpha=0.55,
            rasterized=True,
            linewidths=0,
        )
        plt.colorbar(sc, ax=ax2, label="T_eff (K)", fraction=0.046, pad=0.04)
        ax2.set_xlabel("UMAP axe 1", fontsize=11)
        ax2.set_title("UMAP coloré par T_eff", fontsize=12, fontweight="bold")
        ax2.grid(True, alpha=0.2)

        fig.suptitle(
            "HDBSCAN — Localisation dans le diagramme HR · LAMOST DR5 × Gaia DR3",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()
        self._save(fig, save_path, "umap_hdbscan_hr")
        return fig, axes

    def plot_hdbscan_sensitivity(
        self,
        sensitivity_df: "pd.DataFrame",
        min_cluster_val: int = 75,
        pres_cluster_val: int = 300,
        save_path=None,
    ):
        """Courbes de sensibilité n_clusters et % bruit vs min_cluster_size."""
        df = sensitivity_df
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=self.dpi)

        for ax, col, ylabel, title in [
            (axes[0], "n_clusters", "Nombre de clusters", "Clusters détectés"),
            (axes[1], "pct_noise", "% de points bruit", "Points bruit"),
        ]:
            # Courbe principale issue du sweep hyperparamètre.
            ax.plot(df["min_cluster_size"], df[col], "o-", color="#1A759F", lw=2, ms=7)
            ax.axvline(
                min_cluster_val,
                color="#D4690A",
                ls="--",
                lw=1.5,
                label=f"Analyse fine ({min_cluster_val})",
            )
            ax.axvline(
                pres_cluster_val,
                color="#2D6A4F",
                ls=":",
                lw=1.5,
                label=f"Présentation ({pres_cluster_val})",
            )
            ax.set_xlabel("min_cluster_size", fontsize=11)
            ax.set_ylabel(ylabel, fontsize=11)
            ax.set_title(f"{title} vs min_cluster_size", fontsize=12)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

        fig.suptitle("Sensibilité HDBSCAN — impact de min_cluster_size", fontsize=13)
        plt.tight_layout()
        self._save(fig, save_path, "umap_hdbscan_sensitivity")
        return fig, axes

    def plot_hdbscan_presentation(
        self,
        Z: np.ndarray,
        hdb,
        meta: "pd.DataFrame",
        save_path=None,
    ):
        """Version présentation avec min_cluster_size plus grand (clusters lisibles)."""
        # Attributs dédiés au mode présentation de HDBSCANAnalyzer.
        ids = hdb.ids_pres_
        lp = hdb.labels_pres_
        nc = hdb.n_clusters_pres_
        cmap = hdb.color_map_pres_
        nn = int((lp == -1).sum())
        noise = lp == -1

        teff_col, logg_col = "teff_gspphot", "logg_gspphot"
        has_hr = teff_col in meta.columns and logg_col in meta.columns
        teff = meta[teff_col].values.astype(float) if has_hr else None
        logg = meta[logg_col].values.astype(float) if has_hr else None
        valid_hr = np.isfinite(teff) & np.isfinite(logg) if has_hr else None

        fig, axes = plt.subplots(1, 2, figsize=(18, 8), dpi=self.dpi)

        ax = axes[0]
        # Bruit explicite pour contextualiser les points non assignés.
        ax.scatter(
            Z[noise, 0],
            Z[noise, 1],
            c="lightgray",
            s=1.5,
            alpha=0.3,
            rasterized=True,
            linewidths=0,
            zorder=1,
            label=f"Bruit (n={nn})",
        )
        for cid in ids:
            mask = lp == cid
            # Cluster cid en couleur pleine.
            ax.scatter(
                Z[mask, 0],
                Z[mask, 1],
                c=[cmap[cid]],
                s=3,
                alpha=0.80,
                rasterized=True,
                linewidths=0,
                zorder=2,
                label=f"C{cid} (n={mask.sum()})",
            )
            # Étiquette sur barycentre pour lecture rapide en présentation.
            cx, cy = Z[mask].mean(axis=0)
            ax.text(
                cx,
                cy,
                f"C{cid}",
                fontsize=9,
                fontweight="bold",
                ha="center",
                va="center",
                color="white",
                bbox=dict(boxstyle="round,pad=0.2", fc=cmap[cid], ec="none", alpha=0.9),
                zorder=5,
            )
        ax.legend(markerscale=4, fontsize=9, framealpha=0.9, loc="lower right", ncol=2)
        ax.set_xlabel("UMAP axe 1", fontsize=12)
        ax.set_ylabel("UMAP axe 2", fontsize=12)
        ax.set_title(
            f"HDBSCAN — {nc} populations stellaires", fontsize=14, fontweight="bold"
        )
        ax.grid(True, alpha=0.2)

        ax2 = axes[1]
        if has_hr:
            # Projection des mêmes clusters dans le plan HR.
            ax2.scatter(
                teff[noise & valid_hr],
                logg[noise & valid_hr],
                c="lightgray",
                s=1,
                alpha=0.15,
                rasterized=True,
                linewidths=0,
                zorder=1,
            )
            for cid in ids:
                m = (lp == cid) & valid_hr
                ax2.scatter(
                    teff[m],
                    logg[m],
                    c=[cmap[cid]],
                    s=4,
                    alpha=0.75,
                    rasterized=True,
                    linewidths=0,
                    zorder=2,
                )
                if m.sum() >= 30:
                    # Annotation seulement si cluster assez dense en HR.
                    ax2.text(
                        np.median(teff[m]),
                        np.median(logg[m]),
                        f"C{cid}",
                        fontsize=9,
                        fontweight="bold",
                        ha="center",
                        va="center",
                        color="white",
                        bbox=dict(
                            boxstyle="round,pad=0.2", fc=cmap[cid], ec="none", alpha=0.9
                        ),
                        zorder=5,
                    )
            ax2.invert_xaxis()
            ax2.invert_yaxis()
            ax2.set_xlabel("T_eff (K)", fontsize=12)
            ax2.set_ylabel("log g (dex)", fontsize=12)
            ax2.set_title(
                "Diagramme HR — populations HDBSCAN", fontsize=14, fontweight="bold"
            )
            ax2.grid(True, alpha=0.2)

        fig.suptitle(
            f"Découverte de populations stellaires — HDBSCAN (min_size=300)\n"
            f"{nc} populations · {nn} anomalies ({100*nn/len(lp):.1f}%) · LAMOST DR5 × Gaia DR3",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()
        self._save(fig, save_path, "umap_hdbscan_present")
        return fig, axes

    def plot_hdbscan_feature_heatmap(
        self,
        cluster_means: "pd.DataFrame",
        cluster_labels_aligned: np.ndarray,
        save_path=None,
        n_top: int = 30,
    ):
        """Heatmap clusters × top-n features discriminantes."""
        # Critère de sélection: variance inter-cluster par feature.
        inter_var = cluster_means.var(axis=0)
        top_feats = inter_var.nlargest(n_top).index.tolist()
        heatmap_data = cluster_means[top_feats]
        n_clusters = len(cluster_means)

        fig, ax = plt.subplots(figsize=(18, max(6, n_clusters * 0.5 + 2)), dpi=self.dpi)
        im = ax.imshow(
            heatmap_data.values, aspect="auto", cmap="RdBu_r", vmin=-2.5, vmax=2.5
        )
        plt.colorbar(
            im, ax=ax, label="Score standardisé moyen", fraction=0.02, pad=0.01
        )
        ax.set_xticks(range(len(top_feats)))
        ax.set_xticklabels(
            [
                f.replace("feature_", "").replace("_eq_width", "_EW")[:18]
                for f in top_feats
            ],
            rotation=45,
            ha="right",
            fontsize=7.5,
        )
        ytick_labels = [
            f"C{cid}  (n={(cluster_labels_aligned == cid).sum()})"
            for cid in cluster_means.index
        ]
        ax.set_yticks(range(n_clusters))
        ax.set_yticklabels(ytick_labels, fontsize=8)
        ax.set_xlabel(
            "Top-30 features discriminantes (variance inter-cluster)", fontsize=11
        )
        ax.set_ylabel("Cluster HDBSCAN", fontsize=11)
        ax.set_title(
            "Profil spectroscopique moyen par cluster HDBSCAN\n"
            "(scores standardisés — rouge = excès, bleu = déficit)",
            fontsize=13,
            fontweight="bold",
        )
        plt.tight_layout()
        self._save(fig, save_path, "umap_hdbscan_feature_heatmap")
        return fig, ax

    def plot_hdbscan_feature_profiles(
        self,
        cluster_means: "pd.DataFrame",
        cluster_labels_aligned: np.ndarray,
        save_path=None,
        n_show: int = 12,
    ):
        """Barplots top-5 features caractéristiques par cluster."""
        # Sélectionne les clusters les plus peuplés pour éviter les micro-clusters bruités.
        top_cids = sorted(
            cluster_means.index,
            key=lambda c: (cluster_labels_aligned == c).sum(),
            reverse=True,
        )[:n_show]
        ncols = 4
        nrows = (n_show + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), dpi=120)
        axes_flat = axes.flatten() if n_show > 1 else [axes]

        for ax_i, cid in zip(axes_flat, top_cids):
            profile = cluster_means.loc[cid]
            # Top-5 par amplitude absolue (excès ou déficit).
            top5 = profile.abs().nlargest(5).index.tolist()
            vals = profile[top5].values
            colors = ["#C0392B" if v > 0 else "#2980B9" for v in vals]
            labels = [
                f.replace("feature_", "").replace("_eq_width", "\nEW")[:14]
                for f in top5
            ]
            ax_i.barh(range(5), vals, color=colors, edgecolor="none")
            ax_i.set_yticks(range(5))
            ax_i.set_yticklabels(labels, fontsize=8)
            ax_i.axvline(0, color="gray", lw=0.8, ls="--")
            n_c = (cluster_labels_aligned == cid).sum()
            ax_i.set_title(f"C{cid}  (n={n_c})", fontsize=10, fontweight="bold")
            ax_i.set_xlabel("Score moyen (σ)", fontsize=8)
            ax_i.grid(axis="x", alpha=0.3)
            ax_i.set_xlim(-3, 3)
        for ax_i in axes_flat[n_show:]:
            ax_i.axis("off")
        fig.suptitle(
            "Top-5 features caractéristiques par cluster HDBSCAN\n"
            "(rouge = excès vs population globale, bleu = déficit)",
            fontsize=13,
            fontweight="bold",
        )
        plt.tight_layout()
        self._save(fig, save_path, "umap_hdbscan_feature_profiles")
        return fig, axes

    # ── XGBoost bridge ─────────────────────────────────────────────────

    def plot_xgboost_umap(
        self,
        Z_umap_aligned: np.ndarray,
        y_pred: np.ndarray,
        confidence: np.ndarray,
        cl_aligned: np.ndarray,
        color_map: dict,
        save_path=None,
    ):
        """Trianneau : Prédictions XGBoost / Confiance / Clusters HDBSCAN."""
        from dimred.xgboost_bridge import STELLAR_COLORS

        # Vérifie si une vraie probabilité est disponible.
        classes_pred = sorted(set(y_pred))
        has_proba = confidence is not None and not np.all(confidence == 1.0)
        fig, axes = plt.subplots(1, 3, figsize=(22, 8), dpi=self.dpi)

        ax = axes[0]
        for cls in classes_pred:
            mask = y_pred == cls
            # Couche supervisée: classes prédites en espace UMAP aligné.
            ax.scatter(
                Z_umap_aligned[mask, 0],
                Z_umap_aligned[mask, 1],
                c=[STELLAR_COLORS.get(cls, "#888")],
                s=2,
                alpha=0.55,
                linewidths=0,
                rasterized=True,
                label=f"{cls} (n={mask.sum()})",
            )
        ax.legend(markerscale=4, fontsize=9, framealpha=0.9)
        ax.set_title("Prédictions XGBoost", fontsize=12, fontweight="bold")
        ax.set_xlabel("UMAP axe 1")
        ax.set_ylabel("UMAP axe 2")
        ax.grid(True, alpha=0.2)

        ax = axes[1]
        if has_proba:
            # Carte continue de confiance du classifieur.
            sc = ax.scatter(
                Z_umap_aligned[:, 0],
                Z_umap_aligned[:, 1],
                c=confidence,
                cmap="RdYlGn",
                vmin=0.4,
                vmax=1.0,
                s=2,
                alpha=0.60,
                linewidths=0,
                rasterized=True,
            )
            plt.colorbar(
                sc, ax=ax, label="Confiance max (prob)", fraction=0.046, pad=0.04
            )
            ax.set_title("Confiance XGBoost (P_max)", fontsize=12, fontweight="bold")
        else:
            # Cas fallback: modèle sans predict_proba.
            ax.text(
                0.5,
                0.5,
                "predict_proba\nnon disponible",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=12,
            )
        ax.set_xlabel("UMAP axe 1")
        ax.grid(True, alpha=0.2)

        ax = axes[2]
        noise_m = cl_aligned == -1
        # Couche de bruit HDBSCAN en arrière-plan.
        ax.scatter(
            Z_umap_aligned[noise_m, 0],
            Z_umap_aligned[noise_m, 1],
            c="lightgray",
            s=1.5,
            alpha=0.25,
            rasterized=True,
            linewidths=0,
        )
        for cid in sorted(set(cl_aligned) - {-1}):
            m = cl_aligned == cid
            # Overlay cluster par cluster pour comparaison visuelle avec panneau 1.
            ax.scatter(
                Z_umap_aligned[m, 0],
                Z_umap_aligned[m, 1],
                c=[color_map.get(cid, "#888")],
                s=2,
                alpha=0.65,
                rasterized=True,
                linewidths=0,
            )
        ax.set_title("Clusters HDBSCAN (référence)", fontsize=12, fontweight="bold")
        ax.set_xlabel("UMAP axe 1")
        ax.grid(True, alpha=0.2)

        fig.suptitle(
            "UMAP × XGBoost — Supervisé vs Non-supervisé\nLAMOST DR5 × Gaia DR3",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()
        self._save(fig, save_path, "umap_xgboost_predictions")
        return fig, axes

    def plot_fg_confusion(
        self,
        Z_umap_aligned: np.ndarray,
        y_pred: np.ndarray,
        save_path=None,
    ):
        """Zone de confusion F/G dans l'espace UMAP."""
        # Focus sur sous-espace des classes F et G uniquement.
        fg_mask = np.isin(y_pred, ["F", "G"])
        if fg_mask.sum() < 10:
            logger.info("Pas assez de prédictions F/G pour le plot.")
            return None, None

        fig, ax = plt.subplots(figsize=(9, 8), dpi=self.dpi)
        ax.scatter(
            Z_umap_aligned[:, 0],
            Z_umap_aligned[:, 1],
            c="#EEEEEE",
            s=1,
            alpha=0.3,
            rasterized=True,
            linewidths=0,
        )
        # Puis superpose uniquement F et G pour isoler la région de confusion.
        for cls, col in [("F", "#F1C40F"), ("G", "#E67E22")]:
            m = y_pred == cls
            # Overlay ciblé pour visualiser la zone d'hésitation locale.
            ax.scatter(
                Z_umap_aligned[m, 0],
                Z_umap_aligned[m, 1],
                c=col,
                s=3,
                alpha=0.7,
                linewidths=0,
                rasterized=True,
                label=f"Prédit {cls} (n={m.sum()})",
            )
        ax.legend(markerscale=3, fontsize=11, framealpha=0.9)
        ax.set_xlabel("UMAP axe 1")
        ax.set_ylabel("UMAP axe 2")
        ax.set_title(
            "Zone de confusion F/G — où XGBoost hésite-t-il ?\nLAMOST DR5 × Gaia DR3",
            fontsize=12,
            fontweight="bold",
        )
        ax.grid(True, alpha=0.25)
        plt.tight_layout()
        self._save(fig, save_path, "umap_xgboost_FG_confusion")
        return fig, ax

    # ── Trustworthiness ────────────────────────────────────────────────

    def plot_trustworthiness(
        self,
        X_ref: np.ndarray,
        Z_umap: np.ndarray,
        Z_tsne: np.ndarray,
        save_path=None,
        k_values=None,
        n_subsample: int = 5000,
    ):
        """Courbe de fidélité de voisinage T(k) — UMAP vs t-SNE."""
        from sklearn.manifold import trustworthiness
        import warnings

        if k_values is None:
            k_values = [5, 10, 15, 20, 30, 50, 100]

        rng = np.random.default_rng(42)
        n = min(n_subsample, len(X_ref))
        # Sous-échantillonnage pour maîtriser le coût de calcul T(k).
        idx = rng.choice(len(X_ref), size=n, replace=False)
        X_sub = X_ref[idx]
        Zu_sub = Z_umap[idx]
        Zt_sub = Z_tsne[idx]

        trust_u, trust_t = [], []
        for k in k_values:
            # Certaines versions sklearn peuvent produire des warnings numériques.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                tu = trustworthiness(X_sub, Zu_sub, n_neighbors=k, metric="euclidean")
                tt = trustworthiness(X_sub, Zt_sub, n_neighbors=k, metric="euclidean")
            # Stocke T(k) pour les deux méthodes afin de tracer la courbe comparative.
            trust_u.append(tu)
            trust_t.append(tt)
            logger.info("  k=%3d | UMAP : %.4f | t-SNE : %.4f", k, tu, tt)

        fig, ax = plt.subplots(figsize=(9, 5), dpi=self.dpi)
        ax.plot(
            k_values,
            trust_u,
            "o-",
            color="#1A759F",
            lw=2.2,
            ms=7,
            label=f"UMAP (moy. = {np.mean(trust_u):.4f})",
        )
        ax.plot(
            k_values,
            trust_t,
            "s--",
            color="#E8593C",
            lw=2.2,
            ms=7,
            label=f"t-SNE (moy. = {np.mean(trust_t):.4f})",
        )
        ax.axhline(1.0, color="gray", lw=0.8, ls=":", alpha=0.6)
        ax.fill_between(k_values, trust_u, trust_t, alpha=0.08, color="#7F8FA6")
        ax.set_xlabel("Nombre de voisins k", fontsize=11)
        ax.set_ylabel("Trustworthiness T(k)", fontsize=11)
        ax.set_title(
            f"Fidélité de voisinage — UMAP vs t-SNE\n"
            f"LAMOST DR5 · {n} spectres · entrée : {X_ref.shape[1]} PCs",
            fontsize=12,
            fontweight="bold",
        )
        ax.set_ylim(0.75, 1.01)
        ax.legend(fontsize=10, framealpha=0.9)
        ax.grid(True, alpha=0.25)
        plt.tight_layout()
        self._save(fig, save_path, "umap_trustworthiness")
        return fig, ax

    # ── Autoencodeur : espace latent ──────────────────────────────────

    def plot_latent_zoomed(
        self,
        Z_ae: np.ndarray,
        y: np.ndarray,
        meta: "pd.DataFrame",
        save_path=None,
    ):
        """Espace latent zoomé (P0.5–P99.5) coloré par classe et paramètres Gaia."""
        # Cadre robuste: on coupe les extrêmes via percentiles 0.5 / 99.5.
        x_lo, x_hi = np.percentile(Z_ae[:, 0], [0.5, 99.5])
        y_lo, y_hi = np.percentile(Z_ae[:, 1], [0.5, 99.5])
        pad_x = (x_hi - x_lo) * 0.10
        pad_y = (y_hi - y_lo) * 0.10

        CLASS_PAL = {
            "STAR": "#4C72B0",
            "GALAXY": "#DD8452",
            "QSO": "#55A868",
            "UNKNOWN": "#8172B3",
        }
        CLASS_MRK = {"STAR": "o", "GALAXY": "s", "QSO": "^", "UNKNOWN": "x"}
        PHYS = [
            ("T_eff (K)", "teff_gspphot", "plasma"),
            ("[Fe/H]", "mh_gspphot", "RdYlBu_r"),
            ("log g", "logg_gspphot", "viridis"),
        ]
        available = [
            (label_text, col_name, cmap_name)
            for label_text, col_name, cmap_name in PHYS
            if col_name in meta.columns
        ]

        fig, axes = plt.subplots(1, 1 + len(available), figsize=(22, 5), dpi=self.dpi)
        ax = axes[0]
        for cls in np.unique(y):
            mask = y == cls
            # Vue de référence par classe spectrale.
            ax.scatter(
                Z_ae[mask, 0],
                Z_ae[mask, 1],
                c=CLASS_PAL.get(cls, "#888"),
                marker=CLASS_MRK.get(cls, "o"),
                s=2,
                alpha=0.45,
                linewidths=0,
                label=f"{cls} (n={mask.sum()})",
                rasterized=True,
            )
        ax.set_xlim(x_lo - pad_x, x_hi + pad_x)
        ax.set_ylim(y_lo - pad_y, y_hi + pad_y)
        ax.legend(markerscale=4, fontsize=8, framealpha=0.9)
        ax.set_xlabel("Latent axe 1")
        ax.set_ylabel("Latent axe 2")
        ax.set_title("Type spectral")
        ax.grid(True, alpha=0.25)

        for ax_i, (label, col, cmap) in zip(axes[1:], available):
            vals = meta[col].values.astype(float)
            valid = np.isfinite(vals)
            if (~valid).any():
                # Points sans mesure Gaia affichés en gris très léger.
                ax_i.scatter(
                    Z_ae[~valid, 0],
                    Z_ae[~valid, 1],
                    c="#e0e0e0",
                    s=1,
                    alpha=0.15,
                    rasterized=True,
                    linewidths=0,
                )
            sc = ax_i.scatter(
                Z_ae[valid, 0],
                Z_ae[valid, 1],
                c=vals[valid],
                cmap=cmap,
                vmin=np.nanpercentile(vals[valid], 2),
                vmax=np.nanpercentile(vals[valid], 98),
                s=2,
                alpha=0.55,
                linewidths=0,
                rasterized=True,
            )
            plt.colorbar(
                sc, ax=ax_i, fraction=0.046, pad=0.04, label=label
            ).ax.tick_params(labelsize=8)
            # Même fenêtre de zoom pour permettre la comparaison panneau à panneau.
            ax_i.set_xlim(x_lo - pad_x, x_hi + pad_x)
            ax_i.set_ylim(y_lo - pad_y, y_hi + pad_y)
            ax_i.set_xlabel("Latent axe 1")
            ax_i.set_title(label)
            ax_i.grid(True, alpha=0.25)

        fig.suptitle(
            "Autoencodeur — Espace latent zoomé (population stellaire, P0.5–P99.5)\n"
            "LAMOST DR5 × Gaia DR3",
            fontsize=13,
            fontweight="bold",
        )
        plt.tight_layout()
        self._save(fig, save_path, "ae_latent_grid_zoomed")
        return fig, axes

    def plot_latent_kde(
        self,
        Z_ae: np.ndarray,
        y: np.ndarray,
        meta: "pd.DataFrame",
        save_path=None,
    ):
        """Carte de densité KDE (population STAR) + iso-contours 68/95/99%."""
        from scipy.stats import gaussian_kde

        # KDE restreint à la population stellaire pour une lecture astrophysique propre.
        star_mask = y == "STAR"
        Z_stars = Z_ae[star_mask]
        teff_s = (
            meta["teff_gspphot"].values[star_mask].astype(float)
            if "teff_gspphot" in meta.columns
            else None
        )
        logg_s = (
            meta["logg_gspphot"].values[star_mask].astype(float)
            if "logg_gspphot" in meta.columns
            else None
        )
        mh_s = (
            meta["mh_gspphot"].values[star_mask].astype(float)
            if "mh_gspphot" in meta.columns
            else None
        )

        N_KDE = min(8000, len(Z_stars))
        rng = np.random.default_rng(42)
        # Sous-échantillon pour stabiliser le coût de la KDE.
        idx = rng.choice(len(Z_stars), N_KDE, replace=False)
        logger.info("Calcul KDE sur %d étoiles...", N_KDE)
        kde = gaussian_kde(Z_stars[idx].T, bw_method="scott")

        p1, p99 = np.percentile(Z_stars, [1, 99], axis=0)
        margin = (p99 - p1) * 0.15
        x_grid = np.linspace(p1[0] - margin[0], p99[0] + margin[0], 200)
        y_grid = np.linspace(p1[1] - margin[1], p99[1] + margin[1], 200)
        XX, YY = np.meshgrid(x_grid, y_grid)
        ZZ = kde(np.vstack([XX.ravel(), YY.ravel()])).reshape(XX.shape)

        zz_flat = ZZ.ravel()
        zz_flat = zz_flat[zz_flat > 0]
        levels = (
            []
            if len(zz_flat) == 0
            else sorted(
                set(
                    [
                        float(np.percentile(zz_flat, 1)),
                        float(np.percentile(zz_flat, 5)),
                        float(np.percentile(zz_flat, 32)),
                    ]
                )
            )
        )
        if len(levels) < 2:
            # Si la densité est trop plate, on désactive les contours nominaux.
            levels = []

        params = [
            (teff_s, "T_eff (K)", "plasma", "Densité + T_eff"),
            (logg_s, "log g", "viridis", "Densité + log g"),
            (mh_s, "[Fe/H]", "RdYlBu", "Densité + [Fe/H]"),
        ]
        params = [
            (param_values, label_text, cmap_name, panel_title)
            for param_values, label_text, cmap_name, panel_title in params
            if param_values is not None
        ]

        fig, axes = plt.subplots(1, len(params), figsize=(18, 6), dpi=self.dpi)
        if len(params) == 1:
            axes = [axes]

        for ax, (param, label, cmap, title) in zip(axes, params):
            valid = np.isfinite(param)
            Z_v, P_v = Z_stars[valid], param[valid]
            # Fond densité global + contours de probabilité cumulée.
            ax.contourf(XX, YY, ZZ, levels=20, cmap="Greys", alpha=0.45, zorder=1)
            if levels:
                n_lv = len(levels)
                cs = ax.contour(
                    XX,
                    YY,
                    ZZ,
                    levels=levels,
                    colors=["#4A4A4A", "#2A2A2A", "#0A0A0A"][-n_lv:],
                    linewidths=[0.8, 1.2, 1.8][-n_lv:],
                    linestyles=["--", "-", "-"][-n_lv:],
                    zorder=2,
                )
                ax.clabel(
                    cs,
                    # Labels textuels 99/95/68% alignés aux niveaux calculés.
                    fmt={
                        lv: nm for lv, nm in zip(levels, ["99%", "95%", "68%"][-n_lv:])
                    },
                    fontsize=7,
                    inline=True,
                )
            N_SHOW = min(5000, len(Z_v))
            # Overlay paramétrique sur sous-échantillon pour préserver la fluidité du rendu.
            idx_s = rng.choice(len(Z_v), N_SHOW, replace=False)
            sc = ax.scatter(
                Z_v[idx_s, 0],
                Z_v[idx_s, 1],
                c=P_v[idx_s],
                cmap=cmap,
                s=1.2,
                alpha=0.55,
                rasterized=True,
                zorder=3,
                vmin=float(np.percentile(P_v, 2)),
                vmax=float(np.percentile(P_v, 98)),
            )
            plt.colorbar(sc, ax=ax, label=label, fraction=0.03, pad=0.02)
            ax.set_xlabel("AE axe 1")
            ax.set_ylabel("AE axe 2")
            ax.set_title(title, fontsize=11, fontweight="bold")
            ax.set_xlim(p1[0] - margin[0], p99[0] + margin[0])
            ax.set_ylim(p1[1] - margin[1], p99[1] + margin[1])
            ax.grid(True, alpha=0.15)

        fig.suptitle(
            "Carte de densité KDE — espace latent AE z=2 · population stellaire LAMOST DR5\n"
            "Iso-contours : 68% / 95% / 99% de la densité",
            fontsize=12,
            fontweight="bold",
            y=1.02,
        )
        plt.tight_layout()
        self._save(fig, save_path, "ae_latent_kde")
        return fig, axes

    def plot_ae_recon_by_spectral_type(
        self,
        X: np.ndarray,
        X_recon: np.ndarray,
        spectral_types: np.ndarray,
        feature_names: list,
        save_path=None,
    ):
        """Profil moyen original vs reconstruit par type spectral A/F/G/K/M."""
        RECON_FEATURES = [
            "feature_Hα_eq_width",
            "feature_Hβ_eq_width",
            "feature_Hδ_eq_width",
            "feature_CaIIK_eq_width",
            "feature_CaII_H_eq_width",
            "feature_Mg_b_eq_width",
            "feature_Na_D_eq_width",
            "feature_FeH_proxy",
            "feature_Teff_proxy",
            "feature_synthetic_BV",
            "feature_flux_ratio_blue_red",
            "feature_molecular_TiO_7050",
        ]
        RECON_LABELS = [
            "Hα EW",
            "Hβ EW",
            "Hδ EW",
            "Ca II K",
            "Ca II H",
            "Mg b",
            "Na D",
            "FeH proxy",
            "T_eff proxy",
            "BV synthét.",
            "Flux B/R",
            "TiO 7050",
        ]
        valid_pairs = [
            (label_name, feat_name)
            for label_name, feat_name in zip(RECON_LABELS, RECON_FEATURES)
            if feat_name in feature_names
        ]
        # Aligne labels et index uniquement sur les features effectivement présentes.
        labels_ok = [p[0] for p in valid_pairs]
        feat_idx = [feature_names.index(p[1]) for p in valid_pairs]

        TYPE_ORDER = ["A", "F", "G", "K", "M"]
        TYPE_COLORS = {
            "A": "#5DCAA5",
            "F": "#F5A623",
            "G": "#E8593C",
            "K": "#B07DB8",
            "M": "#3B8BD4",
        }
        TYPE_ORDER = [t for t in TYPE_ORDER if np.sum(spectral_types == t) >= 10]

        if not TYPE_ORDER:
            logger.warning("Aucun type spectral A/F/G/K/M trouvé.")
            return None, None

        # ── Alignement des tailles (XGBoost peut avoir tronqué spectral_types) ──
        n_common = min(len(X), len(X_recon), len(spectral_types))
        if n_common < len(X):
            logger.warning(
                "plot_ae_recon_by_spectral_type : tailles incohérentes "
                "(X=%d, X_recon=%d, spectral_types=%d) → troncature à %d.",
                len(X),
                len(X_recon),
                len(spectral_types),
                n_common,
            )
        X_aligned = X[:n_common]
        X_recon_aligned = X_recon[:n_common]
        spectral_types_aligned = spectral_types[:n_common]
        # Recalculer TYPE_ORDER sur les types vraiment présents après alignement
        TYPE_ORDER = [
            t for t in TYPE_ORDER if np.sum(spectral_types_aligned == t) >= 10
        ]
        if not TYPE_ORDER:
            logger.warning("Aucun type spectral A/F/G/K/M après alignement.")
            return None, None

        n_types = len(TYPE_ORDER)
        x_pos = np.arange(len(feat_idx))
        w = 0.35
        fig, axes = plt.subplots(
            1, n_types, figsize=(4 * n_types, 6), dpi=self.dpi, sharey=True
        )
        if n_types == 1:
            axes = [axes]

        for ax, stype in zip(axes, TYPE_ORDER):
            mask = spectral_types_aligned == stype
            # Profils moyens originaux/reconstruits pour le type spectral courant.
            orig_mean = X_aligned[mask][:, feat_idx].mean(axis=0)
            recon_mean = X_recon_aligned[mask][:, feat_idx].mean(axis=0)
            col = TYPE_COLORS.get(stype, "#888")
            ax.bar(
                x_pos - w / 2,
                orig_mean,
                w,
                color=col,
                alpha=0.85,
                label="Original",
                edgecolor="white",
                linewidth=0.4,
            )
            ax.bar(
                x_pos + w / 2,
                recon_mean,
                w,
                color=col,
                alpha=0.40,
                label="Reconstruit",
                edgecolor=col,
                linewidth=1.2,
            )
            for xi, (o, r) in enumerate(zip(orig_mean, recon_mean)):
                # Segment pointillé = résidu moyen de reconstruction par feature.
                ax.plot([xi, xi], [o, r], color="#333", lw=0.8, ls=":", zorder=4)
            ax.axhline(0, color="gray", lw=0.6, alpha=0.5)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(labels_ok, rotation=45, ha="right", fontsize=7)
            ax.set_title(
                f"Type {stype}\nn={mask.sum():,}",
                fontsize=11,
                fontweight="bold",
                color=col,
            )
            ax.grid(axis="y", alpha=0.2)
            if ax == axes[0]:
                ax.set_ylabel("Score standardisé moyen (σ)", fontsize=9)
                ax.legend(fontsize=8, loc="upper right")

        fig.suptitle(
            "Profil moyen original vs reconstruit par type spectral\n"
            "(scores standardisés — traits pointillés = résidu AE)",
            fontsize=12,
            fontweight="bold",
            y=1.02,
        )
        plt.tight_layout()
        self._save(fig, save_path, "ae_recon_by_spectral_type")
        return fig, axes

    def plot_ae_error_by_family(
        self,
        X: np.ndarray,
        X_recon: np.ndarray,
        feature_names: list,
        pca_analyzer=None,
        save_path=None,
    ):
        """MSE de reconstruction par famille spectroscopique (barh + scatter)."""
        import re
        import pandas as _pd

        # Réutilise les familles définies dans PCAAnalyzer si disponible
        if pca_analyzer is not None and hasattr(pca_analyzer, "_FAMILIES"):
            _FAMILIES = pca_analyzer._FAMILIES
            _FAM_COLORS = pca_analyzer._FAMILY_COLORS
        else:
            _FAMILIES = {
                "Balmer\n(Hα/β/γ/δ)": [
                    r"H[αβγδε]|Halpha|Hbeta|Hgamma|Hdelta|balmer|paschen"
                ],
                "Calcium\n(Ca II H&K)": [r"CaII|CaH|CaK|Ca_8|Ca_trip|feature_Ca"],
                "Magnésium\n(Mg b)": [r"Mg_b|Mg_5|MgH|Mg_trip|feature_Mg"],
                "Fer & métaux": [
                    r"feature_Fe|feature_Cr|FeH_proxy|metal_index|alpha_el"
                ],
                "Sodium\n(Na D)": [r"Na_D|feature_Na"],
                "TiO & moléc.": [r"TiO|VO_|molecular|feature_Ti"],
                "Continuum": [
                    r"continuum|slope|break_4000|flux_ratio|synthetic_BV|color_"
                ],
                "Profils": [r"asymmetr|wing|kurtosis|skewness|depth|rotation"],
                "Autres": [r".*"],
            }
            _FAM_COLORS = [
                "#E8593C",
                "#3B8BD4",
                "#4C9B6F",
                "#B07DB8",
                "#F5A623",
                "#2E86AB",
                "#7F8FA6",
                "#C06C84",
                "#CCCCCC",
            ]

        fam_color_map = {
            f: _FAM_COLORS[i % len(_FAM_COLORS)] for i, f in enumerate(_FAMILIES)
        }

        def _fam(feat):
            # Attribution de famille par premier motif regex correspondant.
            for fam, pats in _FAMILIES.items():
                for p in pats:
                    if re.search(p, feat, re.IGNORECASE):
                        return fam
            return "Autres"

        # Erreur moyenne par feature sur l'ensemble des échantillons.
        mse_per_feat = np.mean((X - X_recon) ** 2, axis=0)
        df_feat_mse = _pd.DataFrame(
            {
                "feature": feature_names,
                "mse": mse_per_feat,
                "family": [_fam(f) for f in feature_names],
            }
        )
        # Agrégation par famille pour résumé interprétable.
        df_fam = (
            df_feat_mse.groupby("family")["mse"]
            .agg(["mean", "std", "count"])
            .rename(columns={"mean": "mse_mean", "std": "mse_std", "count": "n_feats"})
            .reset_index()
            .sort_values("mse_mean")
        )

        fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=self.dpi)
        ax = axes[0]
        fams = df_fam["family"].tolist()
        mses = df_fam["mse_mean"].tolist()
        stds = df_fam["mse_std"].tolist()
        colors = [fam_color_map.get(f, "#CCCCCC") for f in fams]
        ax.barh(
            [f.replace("\n", " ") for f in fams],
            mses,
            xerr=stds,
            color=colors,
            edgecolor="white",
            linewidth=0.5,
            error_kw={"elinewidth": 1, "capsize": 3, "ecolor": "#666"},
        )
        ax.axvline(
            float(np.mean(mse_per_feat)),
            color="#333",
            lw=1.2,
            ls="--",
            label=f"MSE global = {np.mean(mse_per_feat):.4f}",
        )
        ax.set_xlabel("MSE de reconstruction")
        ax.legend(fontsize=9)
        ax.grid(axis="x", alpha=0.25)
        ax.set_title(
            "MSE de reconstruction par famille\n(mieux reconstruit = gauche)",
            fontsize=11,
        )

        ax2 = axes[1]
        for fam, mse, n, c in zip(fams, mses, df_fam["n_feats"].tolist(), colors):
            # Nuage famille: taille de famille vs erreur moyenne.
            ax2.scatter(
                n, mse, s=120, color=c, edgecolors="white", linewidths=0.8, zorder=3
            )
            ax2.annotate(
                fam.replace("\n", " ")[:22],
                (n, mse),
                textcoords="offset points",
                xytext=(6, 0),
                fontsize=6.5,
                color="#333",
            )
        ax2.set_xlabel("Nombre de features dans la famille")
        ax2.set_ylabel("MSE de reconstruction moyenne")
        ax2.set_title("MSE vs taille de la famille", fontsize=11)
        ax2.grid(True, alpha=0.25)

        plt.tight_layout()
        self._save(fig, save_path, "ae_recon_error_by_family")
        return fig, axes

    def plot_ae_error_distribution_logscale(
        self,
        mse_per_sample: np.ndarray,
        y: np.ndarray,
        save_path=None,
    ):
        """Distribution des erreurs linéaire + log."""
        CLASS_COLORS = {
            "STAR": "#4C72B0",
            "GALAXY": "#DD8452",
            "QSO": "#55A868",
            "UNKNOWN": "#8172B3",
        }
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=self.dpi)
        for ax, xscale in zip(axes, ["linear", "log"]):
            for cls in np.unique(y):
                mask = y == cls
                vals = mse_per_sample[mask]
                med = np.median(vals)
                color = CLASS_COLORS.get(cls, "gray")
                if xscale == "log":
                    # Bins logarithmiques pour visualiser les queues longues.
                    bins = np.logspace(
                        np.log10(max(vals.min(), 1e-4)), np.log10(vals.max() + 1), 60
                    )
                else:
                    # Échelle linéaire bornée sur le cœur de distribution (P99 étoiles).
                    star_p99 = (
                        np.percentile(mse_per_sample[y == "STAR"], 99)
                        if np.any(y == "STAR")
                        else vals.max()
                    )
                    bins = np.linspace(0, star_p99 * 1.5, 60)
                ax.hist(
                    vals,
                    bins=bins,
                    alpha=0.55,
                    color=color,
                    density=True,
                    label=f"{cls}  médiane={med:.3f}",
                )
                # Marqueur central robuste pour chaque classe.
                ax.axvline(med, color=color, lw=1.8, ls="--", alpha=0.9)
            ax.set_xlabel("MSE de reconstruction")
            ax.set_ylabel("Densité")
            ax.set_xscale(xscale)
            ax.set_title(
                "Échelle linéaire (P99 étoiles)"
                if xscale == "linear"
                else "Échelle logarithmique (toutes classes)"
            )
            ax.legend(fontsize=9, framealpha=0.9)
            ax.grid(True, alpha=0.3)
        fig.suptitle(
            "Distribution des erreurs de reconstruction — AE vs classes spectrales",
            fontsize=13,
            fontweight="bold",
        )
        plt.tight_layout()
        self._save(fig, save_path, "ae_error_distribution_logscale")
        return fig, axes

    def plot_synthesis_grid(
        self,
        Z_ae: np.ndarray,
        y: np.ndarray,
        meta: "pd.DataFrame",
        scores_pca: np.ndarray,
        Z_umap: Optional[np.ndarray] = None,
        save_path=None,
    ):
        """Figure de synthèse n×3 : PCA / UMAP / Autoencodeur × Classe / T_eff / [Fe/H]."""
        import matplotlib.gridspec as gridspec

        CLASS_PAL = {
            "STAR": "#4C72B0",
            "GALAXY": "#DD8452",
            "QSO": "#55A868",
            "UNKNOWN": "#8172B3",
        }
        CLASS_MRK = {"STAR": "o", "GALAXY": "s", "QSO": "^", "UNKNOWN": "x"}

        METHODS = [("PCA", scores_pca[:, :2], "PC1", "PC2")]
        if Z_umap is not None:
            # UMAP est optionnel selon le notebook exécuté.
            METHODS.append(("UMAP", Z_umap, "UMAP 1", "UMAP 2"))
        METHODS.append(("Autoencodeur", Z_ae, "Latent 1", "Latent 2"))

        PARAMS = [
            ("Classe spectrale", None, None),
            ("T_eff (K)", "teff_gspphot", "plasma"),
            ("[Fe/H]", "mh_gspphot", "RdYlBu_r"),
        ]

        n_rows, n_cols = len(METHODS), len(PARAMS)
        fig = plt.figure(figsize=(5.5 * n_cols, 4.5 * n_rows), dpi=self.dpi)
        gs = gridspec.GridSpec(
            n_rows,
            n_cols,
            figure=fig,
            hspace=0.45,
            wspace=0.35,
            left=0.07,
            right=0.96,
            top=0.91,
            bottom=0.06,
        )

        # Boucle matricielle: chaque ligne = méthode, chaque colonne = mode de coloration.
        for row, (method_name, Z_m, xlabel, ylabel) in enumerate(METHODS):
            for col, (param_label, param_col, cmap_name) in enumerate(PARAMS):
                ax = fig.add_subplot(gs[row, col])
                if param_col is None:
                    # Colonne 1: classes spectrales (catégoriel).
                    for cls in np.unique(y):
                        mask = y == cls
                        ax.scatter(
                            Z_m[mask, 0],
                            Z_m[mask, 1],
                            c=CLASS_PAL.get(cls, "#888"),
                            marker=CLASS_MRK.get(cls, "o"),
                            s=2,
                            alpha=0.45,
                            linewidths=0,
                            label=cls,
                            rasterized=True,
                        )
                    if row == 0:
                        # Une seule légende en haut suffit pour toute la grille.
                        ax.legend(markerscale=4, fontsize=8, framealpha=0.8, loc="best")
                else:
                    if param_col in meta.columns:
                        # Colonnes 2-3: paramètres physiques continus.
                        vals = meta[param_col].values.astype(float)
                        valid = np.isfinite(vals)
                        if (~valid).any():
                            ax.scatter(
                                Z_m[~valid, 0],
                                Z_m[~valid, 1],
                                c="#dddddd",
                                s=1,
                                alpha=0.15,
                                rasterized=True,
                                linewidths=0,
                            )
                        sc = ax.scatter(
                            Z_m[valid, 0],
                            Z_m[valid, 1],
                            c=vals[valid],
                            cmap=cmap_name,
                            vmin=np.nanpercentile(vals[valid], 2),
                            vmax=np.nanpercentile(vals[valid], 98),
                            s=2,
                            alpha=0.55,
                            linewidths=0,
                            rasterized=True,
                        )
                        plt.colorbar(
                            sc, ax=ax, fraction=0.046, pad=0.04, label=param_label
                        ).ax.tick_params(labelsize=7)
                ax.set_aspect("equal", "box")
                ax.tick_params(labelsize=7)
                ax.grid(True, alpha=0.2, linewidth=0.5)
                ax.set_xlabel(xlabel, fontsize=8)
                if col == 0:
                    ax.set_ylabel(
                        f"{method_name}\n{ylabel}", fontsize=9, fontweight="bold"
                    )
                else:
                    ax.set_ylabel(ylabel, fontsize=8)
                if row == 0:
                    ax.set_title(param_label, fontsize=11, fontweight="bold", pad=6)

        fig.suptitle(
            "Synthèse comparative — PCA / UMAP / Autoencodeur\n"
            "Spectres stellaires LAMOST DR5 × Gaia DR3",
            fontsize=14,
            fontweight="bold",
        )
        self._save(fig, save_path, "synthesis_pca_umap_ae")
        return fig
