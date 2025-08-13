"""
AstroSpectro — Visualizer
=========================

Suite d’outils pour **explorer, analyser et interpréter** les spectres LAMOST
dans un notebook Jupyter ou via une mini-app Streamlit.

Fonctions principales
---------------------
- Exploration des headers FITS et affichage propre (Markdown/Streamlit)
- Analyse de spectres (détection de pics, association aux raies, métriques)
- Comparaison multi-spectres (normalisation + décalage visuel)
- Tableaux de bord dataset (zéros, sous-classes, couverture céleste)
- Inspection de modèles entraînés et **interprétabilité SHAP**

Conventions
-----------
- Tous les chemins proviennent de `paths = setup_project_env()` (voir utils.py)
- FITS compressés `.fits.gz` lus en streaming (pas d’extraction sur disque)
- Les avertissements numpy/astropy liés à `sqrt(invvar)` sont neutralisés
  via `utils.make_stddev_uncertainty_from_invvar`.

Dépendances projet
------------------
- pipeline.preprocessor.SpectraPreprocessor
- pipeline.classifier.SpectralClassifier
- pipeline.peak_detector.PeakDetector
- tools.dataset_builder.DatasetBuilder
- utils.make_stddev_uncertainty_from_invvar / utils.check_model_compat
"""

from __future__ import annotations

# --- Standard lib
import base64
import io
import os
import gzip
import glob
import textwrap
import warnings
from typing import Dict, Tuple, Optional
from datetime import datetime, timezone

# --- Scientific / viz
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# --- Astro
from astropy.io import fits
from astropy.nddata import StdDevUncertainty
from astropy import units as u
from specutils import Spectrum, SpectralRegion
from specutils.analysis import centroid, gaussian_fwhm, equivalent_width

# --- Notebook UI
from IPython.display import display, Markdown, HTML
from ipywidgets import interact, widgets
from tqdm.notebook import tqdm

# --- ML / SHAP
import shap

# --- Projet
from pipeline.preprocessor import SpectraPreprocessor
from pipeline.feature_engineering import FeatureEngineer
from pipeline.classifier import SpectralClassifier
from pipeline.peak_detector import PeakDetector
from tools.dataset_builder import DatasetBuilder
from utils import (
    safe_sigma_from_invvar,
)

# --- Constantes (couleurs & limites par défaut) -------------------------------

LINE_COLORS: Dict[str, str] = {
    "Hα": "#FF6B6B",
    "Hβ": "#4ECDC4",
    "CaII K": "#45B7D1",
    "CaII H": "#5A67D8",
    "Mg_b": "#F7B801",
    "Na_D": "#F18701",
}

DEFAULT_XLIM: Tuple[float, float] = (3800.0, 7000.0)
DEFAULT_YLIM: Tuple[float, float] = (-1.0, 2.5)


class AstroVisualizer:
    """
    Outils interactifs pour visualiser et analyser les spectres LAMOST.

    Paramètres
    ----------
    paths : dict
        Dictionnaire de chemins absolus produit par `setup_project_env()`.
        Doit inclure au minimum : RAW_DATA_DIR, CATALOG_DIR, PROCESSED_DIR,
        MODELS_DIR, PROJECT_ROOT.

    Notes
    -----
    - Les méthodes `interactive_*` sont pensées pour un usage **notebook**.
    - L’appli `app()` s’utilise côté **Streamlit**.
    """

    def __init__(self, paths: dict) -> None:
        """
        Initialise le visualiseur avec les chemins du projet.

        Parameters
        ----------
        paths : dict
            Dictionnaire produit par `setup_project_env()`. Doit contenir
            au minimum RAW_DATA_DIR, CATALOG_DIR et PROJECT_ROOT.
        """
        self.paths = paths
        # Scan des .fits.gz disponibles (retours relatifs à RAW_DATA_DIR)
        self.available_spectra: list[str] = self._scan_for_spectra()
        # Modèle non chargé tant qu’on n’en a pas besoin
        self.classifier = None

        print("AstroVisualizer initialisé. Les modèles seront chargés à la demande.")
        # Pré-charge (optionnel) du catalogue des labels
        self.labels_catalog: dict[str, str] = self._load_labels_catalog()
        if self.labels_catalog:
            print(
                f"  > Catalogue de {len(self.labels_catalog)} labels chargé pour l'affichage."
            )

    def _load_labels_catalog(self) -> dict[str, str]:
        """
        Charge un mapping {nom_fichier_fits_sans_ext: subclass} à partir du
        catalogue temporaire généré par le pipeline.

        Returns
        -------
        dict[str, str]
            Dictionnaire (clé = 'spec.../nom_fichier' sans '.fits.gz', valeur = subclass).
            Retourne {} si le fichier n'existe pas ou en cas d'erreur.
        """
        try:
            catalog_path = os.path.join(
                self.paths["CATALOG_DIR"], "master_catalog_temp.csv"
            )
            if not os.path.exists(catalog_path):
                return {}

            df_cat = pd.read_csv(catalog_path, sep="|")

            if "fits_name" not in df_cat.columns or "subclass" not in df_cat.columns:
                # Catalogue pas au format attendu
                return {}

            # Normalise la clé côté catalogue (on retire l’extension .fits.gz)
            # Exemple: 'M5901/spec-...-000.fits.gz' -> 'M5901/spec-...-000'
            key_series = (
                df_cat["fits_name"]
                .astype(str)
                .str.replace(".fits.gz", "", regex=False)
                .str.replace("\\", "/", regex=False)  # robustesse Windows
            )

            labels = pd.Series(df_cat["subclass"].values, index=key_series).to_dict()
            return labels

        except Exception as e:
            print(
                f"  > Avertissement : impossible de charger le catalogue de labels ({e})."
            )
            return {}

    def _scan_for_spectra(self) -> list[str]:
        """
        Parcourt récursivement RAW_DATA_DIR et retourne la liste triée des fichiers
        '.fits.gz' **relatifs** à ce répertoire.

        Returns
        -------
        list[str]
            Chemins relatifs sous RAW_DATA_DIR, séparateur '/' (indépendant OS).
            Exemple: 'M5901/spec-55859-...-000.fits.gz'
        """
        spectra: list[str] = []
        raw_dir = os.path.normpath(self.paths.get("RAW_DATA_DIR", "../data/raw"))

        if not os.path.isdir(raw_dir):
            return []

        for root, _, files in os.walk(raw_dir):
            for fname in files:
                if fname.endswith(".fits.gz"):
                    rel = os.path.relpath(os.path.join(root, fname), raw_dir)
                    # uniformise Windows -> POSIX pour l’affichage/UI
                    spectra.append(rel.replace("\\", "/"))

        return sorted(spectra)

    # ==============================================================================
    # Outil 1 : Affichage des Headers FITS
    # ==============================================================================

    def _format_header_line(self, header, label, key, unit=""):
        """
        Formate proprement une ligne d'en-tête FITS pour affichage Markdown.

        Parameters
        ----------
        header : astropy.io.fits.Header
            En-tête FITS.
        label : str
            Intitulé lisible à afficher (en gras).
        key : str
            Clé du header à extraire.
        unit : str, optional
            Unité à afficher après la valeur (ex.: 'Å').

        Returns
        -------
        str
            Ligne Markdown (avec valeur ou 'N/A' si absente).
        """
        value = header.get(key, "N/A")
        unit_str = f" {unit}" if unit else ""
        return f"- **{label} :** `{value}`{unit_str}\n"

    def _display_formatted_header(self, fits_relative_path):
        """
        Ouvre un FITS compressé (.fits.gz), lit l'en-tête primaire (HDU 0) et
        l'affiche en sections Markdown.

        Parameters
        ----------
        fits_relative_path : str
            Chemin relatif sous RAW_DATA_DIR vers le fichier .fits.gz sélectionné.
        """
        full_path = os.path.join(self.paths["RAW_DATA_DIR"], fits_relative_path)
        try:
            with gzip.open(full_path, "rb") as f_gz:
                with fits.open(f_gz, memmap=False) as hdul:
                    header = hdul[0].header
            md_output = (
                f"### En-tête du fichier : `{os.path.basename(full_path)}`\n---\n"
            )
            sections = {}
            current_section = "Informations Générales"
            sections[current_section] = ""
            for key, value in header.items():
                if key == "COMMENT" and "--------" in str(value):
                    current_section = (
                        str(value).replace("COMMENT", "").replace("-", "").strip()
                    )
                    sections.setdefault(current_section, "")
                elif key and key not in ["COMMENT", "HISTORY", ""]:
                    sections[current_section] += self._format_header_line(
                        header, key, key
                    )
            for title, content in sections.items():
                if content:
                    md_output += f"\n#### {title}\n{content}"
            display(Markdown(md_output))
        except Exception as e:
            display(
                Markdown(f"### Erreur\n**Fichier :** `{full_path}`\n\n**Détail :** {e}")
            )

    def interactive_header_explorer(self):
        """
        Widget notebook pour parcourir les en-têtes FITS d'un clic.

        Affiche un `Dropdown` listant tous les spectres trouvés puis rend un
        Markdown joliment structuré pour l'en-tête du fichier sélectionné.
        """
        if not self.available_spectra:
            print("Aucun spectre trouvé à visualiser.")
            return
        interact(
            self._display_formatted_header, fits_relative_path=self.available_spectra
        )

    # ==============================================================================
    # Outil Notebook 2: Analyseur de Spectre Interactif
    # ==============================================================================

    def _plot_spectrum_analysis(
        self, file_path, prominence, window, xlim, ylim
    ) -> None:
        """
        Charge, normalise et analyse un spectre puis l'affiche (Matplotlib).

        Parameters
        ----------
        file_path : str
            Chemin RELATIF (sous RAW_DATA_DIR) du fichier `.fits.gz` à analyser.
        prominence : float
            Seuil de proéminence pour la détection de pics.
        window : int
            Demi-fenêtre (en Å) utilisée pour les mesures locales autour d’une raie.
        xlim : tuple[float, float]
            Limites X (Å) à appliquer au graphe (peuvent arriver inversées).
        ylim : tuple[float, float]
            Limites Y (flux) à appliquer au graphe (peuvent arriver inversées).

        Notes
        -----
        - Cette fonction trace le spectre normalisé, marque les pics détectés et
        surligne les raies cibles (Balmer, Ca II, Mg_b, Na_D).
        - Aucune métrique locale (FWHM/EW) n’est calculée ici : cela reste du ressort
        de l’outil #3 “Analyseur & Tuner”.
        """
        preprocessor = SpectraPreprocessor()
        peak_detector = PeakDetector(prominence=prominence, window=window)
        full_path = os.path.join(self.paths["RAW_DATA_DIR"], file_path)

        try:
            # --- Lecture sûre du FITS compressé ---
            with gzip.open(full_path, "rb") as f_gz:
                with fits.open(f_gz, memmap=False) as hdul:
                    wavelength, flux, invvar = preprocessor.load_spectrum(hdul)
                    header = hdul[0].header

            # --- Numpy + normalisation ---
            wavelength = np.asarray(wavelength, dtype=np.float64)
            flux = np.asarray(flux, dtype=np.float64)
            flux_norm = preprocessor.normalize_spectrum(flux)

            # --- Détection des pics & matching aux raies cibles ---
            peak_idx, props = peak_detector.detect_peaks(wavelength, flux_norm)
            peak_wl = wavelength[peak_idx]
            matched = peak_detector.match_known_lines(peak_idx, peak_wl, props)
            matched_wl = [data[0] for data in matched.values() if data is not None]

            # --- Tracé ---
            plt.style.use("dark_background")
            fig, ax = plt.subplots(figsize=(18, 7))

            # Spectre normalisé
            ax.plot(
                wavelength,
                flux_norm,
                color="gray",
                alpha=0.75,
                label="Spectre normalisé",
            )

            # Tous les pics détectés
            if peak_idx.size:
                ax.scatter(
                    peak_wl,
                    flux_norm[peak_idx],
                    s=40,
                    marker="v",
                    color="red",
                    label="Pics détectés",
                )

            # Pics associés aux raies cibles (cercles verts)
            if matched_wl:
                j_idx = [np.abs(wavelength - wl).argmin() for wl in matched_wl]
                ax.scatter(
                    wavelength[j_idx],
                    flux_norm[j_idx],
                    s=150,
                    facecolors="none",
                    edgecolors="lime",
                    linewidth=2,
                    label="Pics associés",
                )

            # Lignes verticales aux positions théoriques des raies
            for name, wl in peak_detector.target_lines.items():
                ax.axvline(
                    wl,
                    color=LINE_COLORS.get(name, "dodgerblue"),
                    linestyle="--",
                    alpha=0.85,
                    label=f"Raie {name}",
                )

            # Titre + axes
            true_subclass = header.get("SUBCLASS", "N/A")
            title_obj = header.get("DESIG", os.path.basename(file_path))
            ax.set_title(
                f"Analyse : {title_obj} (Vraie classe : {true_subclass})", fontsize=16
            )
            ax.set_xlabel("Longueur d'onde (Å)")
            ax.set_ylabel("Flux normalisé")
            ax.grid(True, linestyle=":")

            # Limites robustes (on accepte des sliders inversés)
            def _sorted_pair(p, default):
                if not p:
                    return default
                a, b = float(p[0]), float(p[1])
                return (a, b) if a <= b else (b, a)

            x0, x1 = _sorted_pair(xlim, DEFAULT_XLIM)
            y0, y1 = _sorted_pair(ylim, DEFAULT_YLIM)
            ax.set_xlim(x0, x1)
            ax.set_ylim(y0, y1)

            # Légende sans doublons
            handles, labels = ax.get_legend_handles_labels()
            uniq = dict(zip(labels, handles))
            ax.legend(uniq.values(), uniq.keys())

            plt.show()

        except Exception as e:
            print(f"Erreur : {e}")

    def interactive_spectrum_analyzer(self) -> None:
        """
        Notebook: mini-UI pour explorer un spectre et jouer avec la détection.

        Affiche un petit panneau de contrôle (sélecteur de fichier + sliders) et
        redessine la figure à chaque changement via `_plot_spectrum_analysis`.

        Contrôles
        ---------
        - Spectre : dropdown des fichiers `.fits.gz` disponibles (chemin relatif).
        - Prominence : seuil de proéminence pour la détection de pics.
        - Fenêtre (Å) : demi-fenêtre pour les mesures locales.
        - Zoom X (Å) : plage X demandée (Å).
        - Zoom Y (Flux) : plage Y demandée (flux normalisé).
        - Bouton *Réinitialiser* : remet tous les contrôles aux valeurs par défaut.

        Notes
        -----
        - Les sliders de plage (X/Y) sont en `continuous_update=False` pour éviter
        de redessiner à chaque pixel pendant le drag (plus fluide et moins coûteux).
        - Les limites X/Y acceptent les valeurs inversées ; `_plot_spectrum_analysis`
        trie toujours avant d’appliquer.
        """
        if not self.available_spectra:
            print("Aucun spectre trouvé à analyser.")
            return

        # --- Widgets -------------------------------------------------------------
        file_path_widget = widgets.Dropdown(
            options=sorted(self.available_spectra),
            description="Spectre :",
            layout={"width": "max-content"},
        )

        prominence_widget = widgets.FloatSlider(
            min=0.0, max=2.0, step=0.01, value=0.2, description="Prominence:"
        )

        window_widget = widgets.IntSlider(
            min=1, max=50, step=1, value=15, description="Fenêtre (Å):"
        )

        xlim_widget = widgets.FloatRangeSlider(
            value=list(DEFAULT_XLIM),
            min=3500,
            max=9500,
            step=1,
            description="Zoom X (Å):",
            continuous_update=False,
            layout={"width": "500px"},
        )

        ylim_widget = widgets.FloatRangeSlider(
            value=list(DEFAULT_YLIM),
            min=-2,
            max=5,
            step=0.1,
            description="Zoom Y (Flux):",
            continuous_update=False,
            layout={"width": "500px"},
        )

        reset_btn = widgets.Button(
            description="Réinitialiser",
            button_style="warning",
            icon="refresh",
            tooltip="Remettre tous les contrôles aux valeurs par défaut",
            layout={"width": "150px"},
        )

        def _on_reset(_=None):
            prominence_widget.value = 0.2
            window_widget.value = 15
            xlim_widget.value = tuple(DEFAULT_XLIM)
            ylim_widget.value = tuple(DEFAULT_YLIM)

        reset_btn.on_click(_on_reset)

        # --- Liaison contrôles -> tracé -----------------------------------------
        out = widgets.interactive_output(
            self._plot_spectrum_analysis,
            {
                "file_path": file_path_widget,
                "prominence": prominence_widget,
                "window": window_widget,
                "xlim": xlim_widget,
                "ylim": ylim_widget,
            },
        )

        # --- Mise en page --------------------------------------------------------
        controls = widgets.VBox(
            [
                file_path_widget,
                widgets.HBox([prominence_widget, window_widget, reset_btn]),
                xlim_widget,
                ylim_widget,
            ]
        )

        display(widgets.VBox([controls, out]))

    # ==============================================================================
    # Outil 3 : Analyseur & Tuner de Spectre Interactif (pour Notebook)
    # ==============================================================================

    def _plot_peak_detection_notebook(
        self,
        file_path: str,
        prominence: float,
        window: int,
        xlim: tuple[float, float],
        ylim: tuple[float, float],
        model_path: str,
    ) -> None:
        """
        Analyse *interactive* d’un spectre dans le notebook.

        - Charge un FITS compressé (.fits.gz), normalise le flux et détecte les pics.
        - Associe les pics aux raies cibles (Balmer, Ca II, Mg_b, Na_D).
        - Trace le spectre annoté (Matplotlib) + lien d’export PNG.
        - Calcule un tableau de métriques locales pour les pics associés
        (centroïde, FWHM, EW, prominence) et propose l’export CSV/LaTeX.
        - (Optionnel) Charge un modèle .pkl et affiche la **classe prédite**.
        - Tient un **journal de session** et permet l’export d’un **rapport HTML**.

        Parameters
        ----------
        file_path : str
            Chemin *relatif sous* RAW_DATA_DIR du spectre à analyser.
        prominence : float
            Seuil de proéminence (détection de pics).
        window : int
            Demi-fenêtre (en Å) pour les métriques locales autour de chaque raie.
        xlim, ylim : tuple[float, float]
            Limites X (Å) et Y (flux) demandées par les sliders.
        model_path : str
            Chemin absolu du modèle .pkl sélectionné (ou "Aucun").

        Notes
        -----
        - Les warnings `invalid value encountered in sqrt` (Astropy) sont neutralisés
        par usage de `utils.safe_sigma_from_invvar` et d’un `catch_warnings`.
        - Les colonnes attendues par le modèle sont **alignées** automatiquement.
        """
        import builtins

        preprocessor = SpectraPreprocessor()
        peak_detector = PeakDetector(prominence=prominence, window=window)
        full_path = os.path.join(self.paths["RAW_DATA_DIR"], file_path)

        try:
            # --- Lecture du FITS .gz ---
            with gzip.open(full_path, "rb") as f_gz:
                with fits.open(f_gz, memmap=False) as hdul:
                    wavelength, flux, invvar = preprocessor.load_spectrum(hdul)
                    header = hdul[0].header

            true_subclass = header.get("SUBCLASS", "Inconnu")

            # --- Numpy arrays + normalisation ---
            wavelength, flux, invvar = (
                np.asarray(wavelength, dtype=np.float64),
                np.asarray(flux, dtype=np.float64),
                np.asarray(invvar, dtype=np.float64),
            )
            flux_norm = preprocessor.normalize_spectrum(flux)

            # --- Détection de pics & matching aux raies cibles ---
            peak_idx, props = peak_detector.detect_peaks(wavelength, flux_norm)
            peak_wl = wavelength[peak_idx]
            matched = peak_detector.match_known_lines(peak_idx, peak_wl, props)

            # --- Préparation features (pour modèle) ---
            fe = FeatureEngineer()
            vec = fe.extract_features(
                matched, wavelength, flux_norm, invvar
            )  # calcul côté IA

            # --- Prédiction IA (optionnelle) ---
            predicted_label = "Aucun modèle sélectionné"
            if model_path != "Aucun":
                try:
                    clf = SpectralClassifier.load_model(model_path)
                    clf.prediction_target = "main_class"

                    # Map features -> DataFrame aligné sur le modèle
                    current = {name: val for name, val in zip(fe.feature_names, vec)}

                    # Métadonnées éventuelles attendues par le modèle
                    for m in ("redshift", "snr_g", "snr_r", "snr_i", "seeing"):
                        if m in getattr(clf, "feature_names_used", []):
                            try:
                                current[m] = float(header.get(m.upper(), 0.0))
                            except (TypeError, ValueError):
                                current[m] = 0.0

                    X = pd.DataFrame([current])
                    expected = getattr(clf, "feature_names_used", list(X.columns))
                    for c in expected:
                        if c not in X.columns:
                            X[c] = 0.0
                    X = X[expected]

                    yhat = clf.model_pipeline.predict(X)[0]
                    predicted_label = (
                        clf.class_labels[yhat] if clf.model_type == "XGBoost" else yhat
                    )
                except Exception as e:
                    predicted_label = f"Erreur prédiction : {e}"

            # --- Figure Matplotlib ---
            plt.style.use("dark_background")
            plt.figure(figsize=(18, 7))
            plt.plot(
                wavelength,
                flux_norm,
                color="gray",
                alpha=0.7,
                label="Spectre normalisé",
            )

            if len(peak_idx) > 0:
                plt.scatter(
                    peak_wl,
                    flux_norm[peak_idx],
                    color="red",
                    marker="v",
                    s=40,
                    label="Pics détectés",
                )

            matched_wl, matched_flux = [], []
            for name, data in matched.items():
                if data is None:
                    continue
                wl, prom = data
                j = int(np.abs(wavelength - wl).argmin())
                matched_wl.append(wl)
                matched_flux.append(float(flux_norm[j]))
                plt.text(
                    wl,
                    flux_norm[j] + 0.05,
                    name,
                    color=LINE_COLORS.get(name, "white"),
                    fontsize=13,
                    ha="center",
                    weight="bold",
                )

            if matched_wl:
                plt.scatter(
                    matched_wl,
                    matched_flux,
                    s=150,
                    facecolors="none",
                    edgecolors="lime",
                    linewidth=2,
                    label="Pics associés",
                )

            for name, wl in peak_detector.target_lines.items():
                plt.axvline(
                    wl,
                    color=LINE_COLORS.get(name, "dodgerblue"),
                    linestyle="--",
                    alpha=0.8,
                    label=f"Raie {name}",
                )

            # Limites robustes et triées
            def _sorted_pair(p, default):
                if not p:
                    return default
                a, b = float(p[0]), float(p[1])
                return (a, b) if a <= b else (b, a)

            x0, x1 = _sorted_pair(xlim, DEFAULT_XLIM)
            y0, y1 = _sorted_pair(ylim, DEFAULT_YLIM)

            plt.title(
                f"Analyse : {header.get('DESIG', os.path.basename(file_path))} "
                f"(Vraie classe: {true_subclass})",
                fontsize=16,
            )
            plt.xlabel("Longueur d'onde (Å)")
            plt.ylabel("Flux normalisé")
            plt.xlim((x0, x1))
            plt.ylim((y0, y1))
            plt.grid(True, linestyle=":")

            # Légende sans doublons
            handles, labels = plt.gca().get_legend_handles_labels()
            by = dict(zip(labels, handles))
            plt.legend(by.values(), by.keys())

            # Export PNG (base64)
            buf = io.BytesIO()
            plt.savefig(buf, format="png", bbox_inches="tight", dpi=150)
            buf.seek(0)
            b64 = base64.b64encode(buf.read()).decode()
            display(
                HTML(
                    f'<a download="spectre.png" href="data:image/png;base64,{b64}">⬇️ Exporter le graphique (PNG)</a>'
                )
            )
            plt.show()

            # --- Tableau de métriques sur les raies associées ---
            #  -> construit un Spectrum plein cadre avec incertitudes sûres
            sigma = safe_sigma_from_invvar(invvar)  # helper utils.py
            spec_full = Spectrum(
                spectral_axis=wavelength * u.AA,
                flux=flux_norm * u.adu,
                uncertainty=StdDevUncertainty(sigma),
            )

            results: dict[str, dict] = {}
            for name, data in matched.items():
                if data is None:
                    continue
                wl, prom = data
                region = SpectralRegion((wl - window) * u.AA, (wl + window) * u.AA)
                try:
                    with warnings.catch_warnings():
                        warnings.filterwarnings(
                            "ignore",
                            message="invalid value encountered in sqrt",
                            category=RuntimeWarning,
                            module=r".*astropy\.units\.quantity.*",
                        )
                        c = float(centroid(spec_full, regions=region).to_value(u.AA))
                        w = float(
                            gaussian_fwhm(spec_full, regions=region).to_value(u.AA)
                        )
                        ew = float(
                            equivalent_width(spec_full, regions=region).to_value(u.AA)
                        )

                    results[name] = {
                        "Centroïde (Å)": f"{c:.2f}",
                        "Largeur FWHM (Å)": f"{w:.2f}",
                        "Prominence (Force)": f"{prom:.3f}",
                        "Largeur Équiv. (Å)": f"{ew:.3f}",
                    }
                except Exception as e:
                    results[name] = {"Erreur": "Analyse impossible", "Détail": str(e)}

            df_an = None
            if results:
                display(Markdown("#### Analyse quantitative des raies associées"))
                df_an = pd.DataFrame.from_dict(results, orient="index")

                def _hl(row):
                    col = LINE_COLORS.get(row.name)
                    return [
                        f"background-color:{col}; color:black" if col else ""
                        for _ in row
                    ]

                display(df_an.style.apply(_hl, axis=1))

                # Exports CSV / LaTeX
                csv = io.StringIO()
                df_an.to_csv(csv)
                b64_csv = base64.b64encode(csv.getvalue().encode()).decode()
                tex = df_an.to_latex(
                    index=True, caption="Analyse quantitative des raies associées"
                )
                b64_tex = base64.b64encode(tex.encode()).decode()
                display(
                    HTML(
                        f"<div style='margin:8px 0'><a download='analyse_raies.csv' "
                        f"href='data:text/csv;base64,{b64_csv}'>⬇️ CSV</a>"
                        " &nbsp;|&nbsp; "
                        f"<a download='analyse_raies.tex' href='data:text/plain;base64,{b64_tex}'>⬇️ LaTeX</a></div>"
                    )
                )

            # --- Synthèse : classe prédite ---
            display(
                Markdown(f"**Classe spectrale prédite (IA) :** `{predicted_label}`")
            )

            # --- Historique de session + export rapport HTML ---
            if not hasattr(builtins, "log_analyses"):
                builtins.log_analyses = pd.DataFrame(
                    columns=[
                        "Fichier",
                        "Paramètres",
                        "Tableau",
                        "Timestamp",
                        "Classe prédite",
                    ]
                )

            params = {
                "spectre": file_path,
                "prominence": prominence,
                "window": window,
                "xlim": (x0, x1),
                "ylim": (y0, y1),
            }
            builtins.log_analyses = pd.concat(
                [
                    builtins.log_analyses,
                    pd.DataFrame(
                        [
                            {
                                "Fichier": file_path,
                                "Paramètres": params,
                                "Tableau": (
                                    df_an.to_dict()
                                    if isinstance(df_an, pd.DataFrame)
                                    else {}
                                ),
                                "Timestamp": datetime.now().strftime(
                                    "%Y-%m-%d %H:%M:%S"
                                ),
                                "Classe prédite": predicted_label,
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )

            if len(builtins.log_analyses) > 0:
                display(Markdown("#### Historique des analyses (session)"))
                display(
                    builtins.log_analyses[
                        ["Fichier", "Classe prédite", "Paramètres", "Timestamp"]
                    ]
                )

            from ipywidgets import Button

            def _report(_=None):
                html = "<h1>Rapport d'Analyse Spectroscopique</h1>"
                for i, row in builtins.log_analyses.iterrows():
                    html += f"<h2>Analyse #{i+1} — {row['Fichier']}</h2>"
                    if row.get("Classe prédite") is not None:
                        html += f"<p><b>Classe spectrale prédite :</b> {row['Classe prédite']}</p>"
                    html += f"<pre>{row['Paramètres']}</pre>"
                    try:
                        html += pd.DataFrame.from_dict(row["Tableau"]).to_html()
                    except Exception:
                        html += "<p><i>Tableau indisponible.</i></p>"
                    html += f"<p><i>Timestamp : {row['Timestamp']}</i></p><hr>"

                b = io.BytesIO()
                b.write(html.encode())
                b.seek(0)
                b64 = base64.b64encode(b.read()).decode()
                display(
                    HTML(
                        f'<a download="rapport_astro.html" href="data:text/html;base64,{b64}">⬇️ Télécharger le rapport HTML</a>'
                    )
                )

            btn = Button(
                description="Générer le rapport HTML de la session",
                button_style="success",
            )
            btn.on_click(_report)
            display(btn)

        except Exception as e:
            print(f"Erreur lors du traitement du spectre : {e}")

    def interactive_peak_tuner(self):
        """UI ipywidgets pour tuner la détection et (optionnel) prédire avec un modèle .pkl."""
        if not self.available_spectra:
            print("Aucun spectre trouvé.")
            return

        models_dir = self.paths.get("MODELS_DIR", "../data/models/")
        saved = ["Aucun"]
        if os.path.isdir(models_dir):
            files = [
                os.path.join(models_dir, f)
                for f in os.listdir(models_dir)
                if f.endswith(".pkl")
            ]
            saved += sorted(files, key=os.path.getctime, reverse=True)

        ui = widgets.interactive(
            self._plot_peak_detection_notebook,
            file_path=widgets.Dropdown(
                options=self.available_spectra,
                description="Spectre :",
                layout={"width": "max-content"},
            ),
            prominence=widgets.FloatSlider(
                min=0.01, max=1.0, step=0.01, value=0.2, description="Prominence:"
            ),
            window=widgets.IntSlider(
                min=1, max=50, step=1, value=15, description="Fenêtre (Å):"
            ),
            xlim=widgets.FloatRangeSlider(
                value=list(DEFAULT_XLIM),
                min=3500,
                max=9500,
                step=1,
                description="Zoom X (Å):",
                continuous_update=False,
                layout={"width": "500px"},
            ),
            ylim=widgets.FloatRangeSlider(
                value=list(DEFAULT_YLIM),
                min=-2,
                max=5,
                step=0.1,
                description="Zoom Y (Flux):",
                continuous_update=False,
                layout={"width": "500px"},
            ),
            model_path=widgets.Dropdown(
                options=saved,
                description="Modèle IA :",
                layout={"width": "max-content"},
            ),
        )
        display(ui)

    # ==============================================================================
    # Outil 4 : Analyse des Zéros dans les Features ---
    # ==============================================================================

    def analyze_feature_zeros(
        self,
        n_files: int = 100,
        *,
        prominence: float = 0.2,
        window: int = 15,
        eps: float = 1e-12,
        random_state: int = 42,
        export: bool = True,
        show_plot: bool = False,
    ) -> pd.DataFrame:
        """
        Analyse la *sparsité* des features (valeurs nulles/constantes) sur un échantillon de spectres.

        Cette routine prend un sous-ensemble de fichiers disponibles dans `data/raw/`,
        calcule les features via :class:`FeatureEngineer`, puis mesure :
        - le **nombre** et le **taux** de zéros (|x| <= eps) par feature,
        - si une feature est **constante** (écart-type ~ 0 ou nombre de valeurs uniques <= 1),
        - quelques stats rapides (min, max, moyenne, écart-type).

        Résultats affichés dans le notebook + export CSV/LaTeX optionnel.

        Parameters
        ----------
        n_files : int, default 100
            Taille de l’échantillon de spectres à analyser (borné par le nombre disponible).
        prominence : float, default 0.2
            Paramètre de proéminence pour la détection de pics (cohérent avec l’outil interactif).
        window : int, default 15
            Demi-fenêtre en Å pour l’analyse de raies (cohérent avec l’outil interactif).
        eps : float, default 1e-12
            Seuil de « quasi zéro » : on considère |x| <= eps comme nul.
        random_state : int, default 42
            Graine pour l’échantillonnage aléatoire des fichiers.
        export : bool, default True
            Si True, propose l’export CSV et LaTeX du résumé.

        Returns
        -------
        pandas.DataFrame
            Tableau synthétique par feature : `zero_count`, `zero_rate`, `is_constant`,
            `min`, `max`, `mean`, `std`. Trié par `zero_rate` décroissant.

        Notes
        -----
        - Plus `n_files` est grand, plus la mesure de sparsité est fiable (coût CPU en conséquence).
        - Les warnings numériques de `astropy` sont déjà neutralisés dans les helpers
        utilisés par la chaîne d’extraction (voir `FeatureEngineer.extract_features`).
        """
        import random
        import base64
        import io
        from datetime import datetime

        # Récupère et échantillonne la liste des spectres disponibles
        files = list(self.available_spectra)
        if not files:
            display(Markdown("> Aucun spectre trouvé dans `data/raw/`."))
            return pd.DataFrame()

        random.seed(random_state)
        if n_files < len(files):
            files = random.sample(files, k=n_files)

        preprocessor = SpectraPreprocessor()
        detector = PeakDetector(prominence=prominence, window=window)
        fe = FeatureEngineer()

        rows = []
        processed = 0
        errors = 0

        for rel in files:
            try:
                full_path = os.path.join(self.paths["RAW_DATA_DIR"], rel)
                with gzip.open(full_path, "rb") as f_gz:
                    with fits.open(f_gz, memmap=False) as hdul:
                        wl, fl, inv = preprocessor.load_spectrum(hdul)

                fl = preprocessor.normalize_spectrum(fl)
                wl = np.asarray(wl, dtype=np.float64)
                fl = np.asarray(fl, dtype=np.float64)
                inv = np.asarray(inv, dtype=np.float64)

                peak_idx, props = detector.detect_peaks(wl, fl)
                matched = detector.match_known_lines(peak_idx, wl[peak_idx], props)

                vec = fe.extract_features(matched, wl, fl, inv)
                rows.append(vec)
                processed += 1
            except Exception:
                # On continue (certains spectres peuvent être corrompus ou atypiques)
                errors += 1
                continue

        if not rows:
            display(
                Markdown("> Impossible de calculer des features sur l’échantillon.")
            )
            return pd.DataFrame()

        X = np.vstack(rows)
        df = pd.DataFrame(X, columns=fe.feature_names)

        # Masque de (quasi) zéros
        zero_mask = df.abs() <= eps
        zero_count = zero_mask.sum(axis=0)
        zero_rate = (zero_count / len(df)).astype(float)

        # Constantes (std ~ 0 ou une seule valeur distincte)
        std = df.std(ddof=0)
        nunique = df.nunique(dropna=False)
        is_constant = (std <= eps) | (nunique <= 1)

        summary = pd.DataFrame(
            {
                "zero_count": zero_count,
                "zero_rate": zero_rate,
                "is_constant": is_constant,
                "min": df.min(),
                "max": df.max(),
                "mean": df.mean(),
                "std": std,
            }
        ).sort_values(["zero_rate", "is_constant"], ascending=[False, False])

        # Affichage
        display(
            Markdown(
                f"**Échantillon analysé :** {processed} spectre(s)"
                + (f" &nbsp;•&nbsp; **erreurs** : {errors}" if errors else "")
            )
        )
        display(
            Markdown(
                f"- **Features 100% nulles** : {int((summary['zero_rate'] == 1.0).sum())}  \n"
                f"- **Features constantes** : {int(summary['is_constant'].sum())}"
            )
        )

        def _row_style(row):
            # Mise en évidence : tout zéro -> fond rouge, sinon dégradé sur zero_rate
            if row["zero_rate"] == 1.0:
                return ["background-color:#5a1a1a;color:#fff"] * len(row)
            return [
                "background-color:rgba(0,150,255,{:.2f})".format(row["zero_rate"])
            ] + [""] * (len(row) - 1)

        display(Markdown("#### Sparsité des features (tri `zero_rate` ↓)"))
        try:
            display(
                summary.style.apply(_row_style, axis=1).format(
                    {
                        "zero_rate": "{:.2%}",
                        "min": "{:.3g}",
                        "max": "{:.3g}",
                        "mean": "{:.3g}",
                        "std": "{:.3g}",
                    }
                )
            )
        except Exception:
            # fallback sans style si l’environnement ne supporte pas .style
            display(summary)

        # Export (optionnel)
        if export:
            ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
            csv = io.StringIO()
            summary.to_csv(csv)
            b64_csv = base64.b64encode(csv.getvalue().encode()).decode()

            tex = summary.to_latex(
                index=True, caption="Sparsité des features (échantillon)"
            )
            b64_tex = base64.b64encode(tex.encode()).decode()

            display(
                HTML(
                    "<div style='margin:8px 0'>"
                    f"<a download='feature_zero_summary_{ts}.csv' href='data:text/csv;base64,{b64_csv}'>⬇️ CSV</a>"
                    " &nbsp;|&nbsp; "
                    f"<a download='feature_zero_summary_{ts}.tex' href='data:text/plain;base64,{b64_tex}'>⬇️ LaTeX</a>"
                    "</div>"
                )
            )
        return summary

    # ==============================================================================
    # Outil 5 : Carte de Couverture Céleste ---
    # ==============================================================================

    def _generate_position_catalog(
        self, force_regenerate: bool = False
    ) -> pd.DataFrame:
        """
        Construit (ou recharge) un petit catalogue de positions célestes **par plan**
        à partir des headers FITS (.fits.gz). Les positions retournées sont les
        moyennes RA/Dec par `PLANID`.

        Paramètres
        ----------
        force_regenerate : bool, default False
            Si False et si le fichier CSV existe déjà, on le recharge directement.
            Si True, on rescane les FITS et on régénère le CSV.

        Returns
        -------
        pandas.DataFrame
            DataFrame trié par `plan_id` contenant :
            - `plan_id` : identifiant de plan (header `PLANID`)
            - `ra`      : moyenne (deg) des RA des spectres du plan (0..360)
            - `dec`     : moyenne (deg) des Dec des spectres du plan (-90..+90)
            - `spectra_count` : nombre de spectres vus pour ce plan

        Notes
        -----
        - Headers requis : `RA`, `DEC`, `PLANID`. Les entrées sans ces
        informations ou avec coordonnées invalides sont ignorées.
        - Le résultat est mis en cache dans
        `<CATALOG_DIR>/spectra_position_catalog.csv`.
        - Utilisé par `plot_sky_coverage()`.
        """
        catalog_path = os.path.join(
            self.paths["CATALOG_DIR"], "spectra_position_catalog.csv"
        )

        # 1) Cache sur disque
        if os.path.exists(catalog_path) and not force_regenerate:
            print("  > Catalogue de position existant chargé.")
            try:
                df_cached = pd.read_csv(catalog_path)
                # On s’assure des colonnes minimales
                expected = {"plan_id", "ra", "dec"}
                if expected.issubset(df_cached.columns):
                    return df_cached.sort_values("plan_id").reset_index(drop=True)
            except Exception:
                # Si le cache est corrompu, on retente un scan complet
                print("  > Avertissement: cache illisible, rescannage des FITS...")

        # 2) Scan des FITS
        print("  > Scan des fichiers FITS pour générer le catalogue de position...")
        if not self.available_spectra:
            print("  > Aucun fichier FITS trouvé.")
            return pd.DataFrame(columns=["plan_id", "ra", "dec", "spectra_count"])

        rows: list[dict] = []
        invalid = 0

        for rel_path in tqdm(self.available_spectra, desc="Lecture des headers"):
            file_path = os.path.join(self.paths["RAW_DATA_DIR"], rel_path)
            try:
                with gzip.open(file_path, "rb") as f_gz:
                    with fits.open(f_gz, memmap=False) as hdul:
                        h = hdul[0].header

                # Champs indispensables
                if not all(k in h for k in ("PLANID", "RA", "DEC")):
                    continue

                plan_id = h["PLANID"]

                # Convertit & valide les coordonnées (deg)
                try:
                    ra = float(h["RA"]) % 360.0
                    dec = float(h["DEC"])
                except Exception:
                    invalid += 1
                    continue

                if (
                    not np.isfinite(ra)
                    or not np.isfinite(dec)
                    or not (-90.0 <= dec <= 90.0)
                ):
                    invalid += 1
                    continue

                rows.append({"plan_id": plan_id, "ra": ra, "dec": dec})

            except Exception:
                # On ignore silencieusement les fichiers/headers problématiques
                invalid += 1
                continue

        if not rows:
            print("  > Aucune coordonnée exploitable n'a été extraite.")
            return pd.DataFrame(columns=["plan_id", "ra", "dec", "spectra_count"])

        df_positions = pd.DataFrame(rows)

        # 3) Agrégation par plan
        df_plan = (
            df_positions.groupby("plan_id")
            .agg(ra=("ra", "mean"), dec=("dec", "mean"), spectra_count=("ra", "size"))
            .reset_index()
            .sort_values("plan_id")
            .reset_index(drop=True)
        )

        # 4) Sauvegarde cache
        try:
            os.makedirs(self.paths["CATALOG_DIR"], exist_ok=True)
            df_plan.to_csv(catalog_path, index=False)
        except Exception as e:
            print(f"  > Avertissement: impossible d'écrire le CSV de cache ({e}).")

        print(
            f"  > Catalogue de position créé avec {len(df_plan)} plan(s) unique(s)."
            + (f"  [{invalid} entrée(s) ignorée(s)]" if invalid else "")
        )
        return df_plan

    def plot_sky_coverage(
        self, save_path: str | None = None
    ) -> tuple[plt.Figure, plt.Axes] | None:
        """
        Affiche la carte Mollweide de la couverture du ciel (centroïdes RA/Dec par plan).

        Paramètres
        ----------
        save_path : str | None, default None
            Chemin de sauvegarde (PNG). Si None, un fichier timestampé est
            écrit dans `paths['LOGS_DIR']`. Si False/"" est passé, aucune sauvegarde.

        Returns
        -------
        (fig, ax) | None
            La figure et l'axe Matplotlib si des données sont disponibles, sinon None.

        Notes
        -----
        - Utilise le CSV de cache généré par `_generate_position_catalog`.
        - Les longitudes sont converties en système Mollweide
        (RA vers la gauche, centre 0h/24h).
        - La couleur et la taille traduisent `spectra_count` (nombre de spectres par plan).
        """
        # -- 1) Récupération / construction du petit catalogue de positions
        df = self._generate_position_catalog()
        if df.empty:
            print("  > Aucune donnée de position pour la carte de couverture.")
            return None

        # -- 2) Optionnel : image d'arrière-plan (Voie Lactée en projection Mollweide)
        bg_image = None
        try:
            from PIL import Image

            image_path = os.path.join(
                self.paths["PROJECT_ROOT"],
                "static",
                "images",
                "milky_way_mollweide.jpeg",
            )
            if os.path.exists(image_path):
                bg_image = Image.open(image_path)
        except Exception:
            bg_image = None  # pas bloquant

        # -- 3) Préparation coordonnées -> Mollweide
        ra_deg = df["ra"].to_numpy(dtype=float)
        dec_deg = df["dec"].to_numpy(dtype=float)

        # RA (deg) -> radians, inversion pour avoir RA vers la gauche, plage [-π, π]
        lon = np.radians(ra_deg)
        x = np.pi - lon
        x = (x + np.pi) % (2 * np.pi) - np.pi
        y = np.radians(dec_deg)

        counts = df.get("spectra_count", pd.Series(np.ones(len(df)))).to_numpy(
            dtype=float
        )
        cmax = float(np.nanmax(counts)) if np.isfinite(counts).any() else 1.0
        sizes = 14.0 + 60.0 * (counts / cmax)  # taille ~ densité

        # -- 4) Figure
        plt.style.use("dark_background")
        fig = plt.figure(figsize=(20, 10))
        ax = fig.add_subplot(111, projection="mollweide")

        # Fond Voie Lactée si dispo
        if bg_image is not None:
            ax.imshow(
                bg_image,
                extent=[-np.pi, np.pi, -np.pi / 2, np.pi / 2],
                aspect="auto",
                zorder=0,
                alpha=0.6,
            )

        # Nuage de points par plan
        sc = ax.scatter(
            x,
            y,
            c=counts,
            s=sizes,
            cmap="viridis",
            alpha=0.9,
            edgecolors="black",
            linewidths=0.3,
            zorder=1,
        )

        # Grille & habillage
        ax.grid(True, color="w", alpha=0.2, lw=0.5)

        # Ticks RA (en heures) : 150..-150 step 30°
        lon_labels_deg = np.arange(150, -181, -30)  # affichage à gauche
        lon_ticks = np.radians(lon_labels_deg)
        ax.set_xticks(lon_ticks)
        tick_labels = [f"{((L % 360) / 15):.0f}h" for L in (lon_labels_deg % 360)]
        ax.set_xticklabels(tick_labels)

        ax.set_yticks(np.radians(np.arange(-75, 76, 15)))
        ax.set_yticklabels([f"{d}°" for d in np.arange(-75, 76, 15)])

        ax.set_title(
            "Carte de Couverture Céleste (plans DRS — centroïdes RA/Dec)", pad=16
        )

        # Barre de couleurs
        cbar = fig.colorbar(
            sc, ax=ax, orientation="horizontal", pad=0.06, fraction=0.05
        )
        cbar.set_label("Nombre de spectres par plan")

        fig.tight_layout()

        # -- 5) Sauvegarde
        if save_path is not False:
            if not save_path:
                ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                os.makedirs(self.paths["LOGS_DIR"], exist_ok=True)
                save_path = os.path.join(
                    self.paths["LOGS_DIR"], f"sky_coverage_{ts}.png"
                )
            try:
                fig.savefig(save_path, dpi=150, bbox_inches="tight")
                print(f"  > Carte de couverture sauvegardée : {save_path}")
            except Exception as e:
                print(f"  > Avertissement : échec de la sauvegarde ({e}).")

        return fig, ax

    # ==============================================================================
    # Outil 6 : Inspecteur de Modèles Entraînés ---
    # ==============================================================================

    def _analyze_saved_model(self, model_path: str) -> None:
        """
        Inspecte un classifieur sauvegardé (.pkl) et affiche :
        - un résumé Markdown (type de modèle, meilleures hyperparams, classes),
        - les colonnes utilisées à l'entraînement (+ celles retenues après sélection),
        - un graphique d'importance des features si disponible.

        Parameters
        ----------
        model_path : str
            Chemin ABSOLU vers le fichier `.pkl` du classifieur sauvegardé.

        Notes
        -----
        - Le classifieur est chargé via `SpectralClassifier.load_model()`.
        - Si une étape `SelectFromModel` est présente dans le pipeline, les importances
        affichées utilisent les *noms de features retenues* (`selected_features_`)
        lorsqu’ils sont disponibles; sinon on retombe sur `feature_names_used`.
        - Compatible RandomForest / XGBoost (les deux exposent `feature_importances_`).
        """
        try:
            # --- Chargement du modèle complet (objet SpectralClassifier) ------------
            clf_wrapper = SpectralClassifier.load_model(model_path)

            if not hasattr(clf_wrapper, "model_pipeline"):
                display(
                    Markdown(
                        "### Erreur\nLe fichier chargé **ne** contient pas `model_pipeline`."
                    )
                )
                return

            pipe = clf_wrapper.model_pipeline
            model = pipe.named_steps.get("clf", None)
            model_type = getattr(clf_wrapper, "model_type", type(model).__name__)

            # --- Résumé Markdown -----------------------------------------------------
            md = []
            md.append(f"### Analyse du Modèle : `{os.path.basename(model_path)}`")
            md.append(f"- **Type de Modèle :** `{model_type}`")

            # Meilleurs hyperparamètres CV (s’ils ont été conservés)
            best_params = getattr(clf_wrapper, "best_params_", None)
            if best_params:
                md.append("\n#### Meilleurs Hyperparamètres (GridSearchCV)")
                for k, v in best_params.items():
                    md.append(f"- **{k.replace('clf__', '')} :** `{v}`")
            else:
                md.append(
                    "\n*Pas d’hyperparamètres de recherche enregistrés (ou entraînement sans GridSearchCV).*"
                )

            # Classes connues (ordre d'entraînement)
            class_labels = getattr(clf_wrapper, "class_labels", None)
            if class_labels is not None:
                md.append(f"\n- **Nombre de classes :** `{len(class_labels)}`")
                md.append(f"- **Classes :** `{', '.join(map(str, class_labels))}`")

            # Colonnes d’entraînement & sélection éventuelle
            used_cols = getattr(clf_wrapper, "feature_names_used", None) or []
            selected_cols = getattr(clf_wrapper, "selected_features_", None)

            md.append(f"\n- **Colonnes d'entraînement :** `{len(used_cols)}`")
            if selected_cols:
                md.append(
                    f"- **Colonnes retenues après sélection :** `{len(selected_cols)}`"
                )
            display(Markdown("\n".join(md)))

            # --- Importance des features -------------------------------------------
            # Préfère les noms des features *retenues* si la sélection existe,
            # car le classifieur a été entraîné sur l'espace transformé.
            feature_names_for_plot = selected_cols or used_cols

            if hasattr(model, "feature_importances_") and feature_names_for_plot:
                importances = np.asarray(model.feature_importances_, dtype=float)

                # Sécurise l’alignement (longueurs peuvent différer si l’artefact n’a
                # pas conservé les noms de features retenues).
                if importances.shape[0] != len(feature_names_for_plot):
                    # Fallback : on tente malgré tout avec les colonnes d’entraînement.
                    feature_names_for_plot = used_cols[: importances.shape[0]]

                # Trie du plus faible au plus fort pour un barh lisible
                order = np.argsort(importances)
                names_sorted = np.array(feature_names_for_plot)[order]
                imps_sorted = importances[order]

                # Limite l’affichage à 40 features pour garder un graphe lisible
                top_n = min(40, len(imps_sorted))
                names_sorted = names_sorted[-top_n:]
                imps_sorted = imps_sorted[-top_n:]

                plt.style.use("dark_background")
                plt.figure(figsize=(16, max(8, top_n * 0.35)))
                plt.title("Importance des Features (après pipeline)", fontsize=16)
                plt.barh(range(top_n), imps_sorted, align="center")
                plt.yticks(range(top_n), names_sorted)
                plt.xlabel("Importance (Gini)")
                plt.tight_layout()
                plt.show()
            else:
                display(
                    Markdown(
                        "#### Importance des Features\n*Aucune importance accessible pour ce modèle.*"
                    )
                )

            # --- Infos pipeline utiles (optionnel) ----------------------------------
            try:
                steps_txt = " → ".join(pipe.named_steps.keys())
                display(Markdown(f"**Pipeline entraîné :** `{steps_txt}`"))
            except Exception:
                pass

        except Exception as e:
            display(
                Markdown(f"### Erreur lors de l'analyse du modèle\n**Détail :** `{e}`")
            )

    def interactive_model_inspector(self) -> None:
        """
        Ouvre un mini-UI (ipywidgets) pour parcourir les modèles sauvegardés et lancer
        leur inspection avec `_analyze_saved_model`.

        Comportement
        ------------
        - Cherche tous les fichiers `.pkl` dans `self.paths["MODELS_DIR"]` (ou `../models/`).
        - Trie par date de modification décroissante (plus récent en premier).
        - Dropdown affichant :  « <nom_fichier.pkl> — <taille> — <YYYY-MM-DD HH:MM> ».
        - Bouton **Analyser** : appelle `_analyze_saved_model(model_path)`.
        - Bouton **Rafraîchir** : rescane le répertoire (utile après un nouvel entraînement).

        Notes
        -----
        - Nécessite `ipywidgets` (Jupyter).
        - L’analyse est réalisée par `_analyze_saved_model`, qui affiche le résumé et
        l’importance des features quand disponible.
        """
        from ipywidgets import HBox, VBox, Dropdown, Button, HTML
        from IPython.display import display, Markdown

        models_dir = self.paths.get("MODELS_DIR", "../models/")

        header = HTML(
            value=(
                f"<b>Dossier des modèles :</b> <code>{models_dir}</code><br>"
                "Sélectionne un modèle puis clique <b>Analyser</b>."
            )
        )

        def _scan_models() -> list[tuple[str, str]]:
            """Retourne une liste d’options (label lisible, chemin absolu)."""
            if not os.path.exists(models_dir):
                return []

            entries = []
            for f in os.listdir(models_dir):
                if not f.lower().endswith(".pkl"):
                    continue
                full = os.path.join(models_dir, f)
                try:
                    stat = os.stat(full)
                    size_mb = stat.st_size / (1024 * 1024)
                    mtime = datetime.fromtimestamp(stat.st_mtime).strftime(
                        "%Y-%m-%d %H:%M"
                    )
                    label = f"{f} — {size_mb:.2f} MB — {mtime}"
                    entries.append((label, full))
                except Exception:
                    # Si on n'arrive pas à lire les stats, on garde au moins le nom.
                    entries.append((f, full))

            # Tri : plus récent d'abord
            entries.sort(key=lambda x: os.path.getmtime(x[1]), reverse=True)
            return entries

        options = _scan_models()

        if not options:
            display(Markdown(f"### Aucun modèle `.pkl` trouvé dans `{models_dir}`."))
            return

        dd = Dropdown(options=options, description="Modèle :", layout={"width": "75%"})
        btn_analyze = Button(
            description="Analyser", button_style="success", icon="search"
        )
        btn_refresh = Button(description="Rafraîchir", button_style="", icon="refresh")

        out = widgets.Output()

        def on_analyze_clicked(_):
            out.clear_output(wait=True)
            with out:
                try:
                    self._analyze_saved_model(dd.value)
                except Exception as e:
                    display(
                        Markdown(f"### Erreur lors de l'analyse\n**Détail :** `{e}`")
                    )

        def on_refresh_clicked(_):
            new_opts = _scan_models()
            if not new_opts:
                out.clear_output(wait=True)
                with out:
                    display(
                        Markdown(f"### Aucun modèle `.pkl` trouvé dans `{models_dir}`.")
                    )
                return
            dd.options = new_opts  # conserve la sélection si possible

        btn_analyze.on_click(on_analyze_clicked)
        btn_refresh.on_click(on_refresh_clicked)

        ui = VBox(
            [
                header,
                HBox([dd, btn_analyze, btn_refresh]),
                out,
            ]
        )
        display(ui)

    # ==============================================================================
    # Outil 7: Comparaison de Spectres ---
    # ==============================================================================

    def _plot_spectra_comparison(
        self,
        file_paths: list[str],
        normalize: bool = True,
        offset: float = 0.0,
        save_path: str | None = None,
    ) -> tuple[plt.Figure, plt.Axes] | None:
        """
        Superpose et compare plusieurs spectres `.fits.gz`.

        Paramètres
        ----------
        file_paths : list[str]
            Chemins **relatifs** (sous RAW_DATA_DIR) des spectres à tracer.
        normalize : bool, default=True
            Si True, normalise chaque spectre avant affichage (via `SpectraPreprocessor`).
        offset : float, default=0.0
            Décalage vertical ajouté par spectre (i * offset) pour améliorer la lisibilité.
        save_path : str | None, default=None
            Optionnel : chemin de sortie pour sauvegarder la figure (PNG).

        Retour
        ------
        (fig, ax) ou None
            Figure et axes Matplotlib si au moins un spectre est tracé, sinon None.

        Notes
        -----
        - Les fichiers doivent être compressés `.fits.gz`.
        - Les chemins fournis sont relatifs à `self.paths["RAW_DATA_DIR"]`.
        - Si `self.DEFAULT_XLIM` existe, on l’utilise pour la limite en X, sinon (3800, 7000).
        """
        if not file_paths:
            print("Aucun fichier fourni.")
            return None

        # Dédoublonne et nettoie la liste
        uniq_paths: list[str] = []
        for p in file_paths:
            if p and p not in uniq_paths:
                uniq_paths.append(p)
        file_paths = uniq_paths

        preprocessor = SpectraPreprocessor()
        plt.style.use("dark_background")
        fig, ax = plt.subplots(figsize=(18, 9))

        plotted = 0

        for i, rel_path in enumerate(file_paths):
            full_path = os.path.join(self.paths["RAW_DATA_DIR"], rel_path)

            try:
                with (
                    gzip.open(full_path, "rb") as f_gz,
                    fits.open(f_gz, memmap=False) as hdul,
                ):
                    wavelength, flux, _ = preprocessor.load_spectrum(hdul)
            except Exception as e:
                print(f"[!] Ignoré (lecture impossible) : {rel_path} -> {e}")
                continue

            if normalize:
                try:
                    flux = preprocessor.normalize_spectrum(flux)
                except Exception:
                    # On n'interrompt pas l'affichage pour un échec de normalisation
                    pass

            # Décalage vertical constant par courbe (i*offset)
            y = flux + (i * offset if offset else 0.0)

            ax.plot(wavelength, y, lw=1.0, label=os.path.basename(rel_path))
            plotted += 1

        if plotted == 0:
            plt.close(fig)
            print("Aucun spectre n'a pu être chargé/affiché.")
            return None

        # Axes, grille, titre
        ax.set_xlabel("Longueur d'onde (Å)")
        ax.set_ylabel("Flux (unités normalisées)" if normalize else "Flux (adu)")
        ax.set_title("Comparaison de Spectres")
        ax.grid(True, alpha=0.25)

        # Limites X par défaut (si dispo)
        try:
            xlim = getattr(self, "DEFAULT_XLIM", (3800.0, 7000.0))
            ax.set_xlim(xlim)
        except Exception:
            pass

        # Légende
        ax.legend(loc="upper right", frameon=True, fontsize=9)

        fig.tight_layout()

        if save_path:
            try:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                fig.savefig(save_path, dpi=150, bbox_inches="tight")
            except Exception as e:
                print(f"[!] Impossible de sauvegarder la figure ({save_path}) : {e}")

        return fig, ax

    def interactive_spectra_comparator(self) -> None:
        """
        Widget interactif pour comparer plusieurs spectres.

        Interface
        ---------
        - Liste multi-sélection des fichiers (sous `RAW_DATA_DIR`).
        - Case à cocher pour normaliser les spectres.
        - Curseur de décalage vertical (offset) appliqué à chaque courbe.
        - Option d'export PNG dans le dossier `logs/` avec nom auto (ou personnalisé).

        Effets de bord
        --------------
        - Affiche la figure dans la cellule courante.
        - Si l'export est activé, écrit un PNG dans `self.paths["LOGS_DIR"]`.
        """
        import os
        from datetime import datetime, timezone
        import ipywidgets as widgets
        from IPython.display import display

        if not getattr(self, "available_spectra", None):
            print("Aucun spectre trouvé.")
            return

        # --- Widgets -------------------------------------------------------------
        files = widgets.SelectMultiple(
            options=self.available_spectra,
            rows=min(12, max(6, len(self.available_spectra))),
            description="Spectres:",
            layout=widgets.Layout(width="520px"),
        )
        normalize = widgets.Checkbox(value=True, description="Normaliser")
        offset = widgets.FloatSlider(
            value=0.5, min=0.0, max=5.0, step=0.1, description="Décalage Y:"
        )

        export_toggle = widgets.Checkbox(
            value=False, description="Exporter PNG (logs/)"
        )
        export_name = widgets.Text(
            value="",
            placeholder="auto (spectra_compare_YYYYMMDDTHHMMSSZ.png)",
            description="Nom:",
            layout=widgets.Layout(width="420px"),
        )

        run_btn = widgets.Button(
            description="Tracer",
            icon="line-chart",
            button_style="success",
            layout=widgets.Layout(width="150px"),
        )
        out = widgets.Output()

        # --- Callback ------------------------------------------------------------
        def on_run_clicked(_):
            out.clear_output()
            with out:
                sel = list(files.value)
                if not sel:
                    print("Sélectionne au moins un spectre dans la liste.")
                    return

                save_path = None
                if export_toggle.value:
                    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                    fname = export_name.value.strip() or f"spectra_compare_{ts}.png"
                    os.makedirs(self.paths["LOGS_DIR"], exist_ok=True)
                    save_path = os.path.join(self.paths["LOGS_DIR"], fname)

                res = self._plot_spectra_comparison(
                    file_paths=sel,
                    normalize=normalize.value,
                    offset=offset.value,
                    save_path=save_path,
                )
                if res is None:
                    return

                if save_path:
                    print(f"Figure sauvegardée : {save_path}")

        run_btn.on_click(on_run_clicked)

        # --- Mise en page --------------------------------------------------------
        controls_right = widgets.VBox(
            [normalize, offset, export_toggle, export_name, run_btn]
        )
        ui = widgets.VBox(
            [
                widgets.HBox([files, controls_right]),
                out,
            ]
        )

        display(ui)

    # ==============================================================================
    # Outil 8 : Analyseur de Spectre Augmenté (Version Plotly) ---
    # ==============================================================================

    def _display_formatted_header_for_streamlit(self, st, fits_relative_path):
        """
        Affiche les informations de l'en-tête FITS dans Streamlit.
        """
        full_path = os.path.join(self.paths["RAW_DATA_DIR"], fits_relative_path)
        try:
            with gzip.open(full_path, "rb") as f_gz:
                with fits.open(f_gz, memmap=False) as hdul:
                    header = hdul[0].header

            md_output = (
                f"### En-tête du fichier : `{os.path.basename(full_path)}`\n---\n"
            )
            sections = {}
            current_section = "Informations Générales"
            sections[current_section] = ""
            for key, value in header.items():
                if key == "COMMENT" and "--------" in str(value):
                    current_section = (
                        str(value).replace("COMMENT", "", 1).replace("-", "").strip()
                    )
                    if current_section not in sections:
                        sections[current_section] = ""
                elif key and key not in ["COMMENT", "HISTORY", ""]:
                    sections[current_section] += self._format_header_line(
                        header, key, key
                    )
            for title, content in sections.items():
                if content:
                    md_output += f"\n#### {title}\n{content}"
            st.markdown(md_output, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Erreur lors de l'ouverture du header : {e}")

    def _plot_spectrum_analysis_plotly(
        self, st, file_path_or_buffer, prominence, window
    ):
        preprocessor = SpectraPreprocessor()
        peak_detector = PeakDetector(prominence=prominence, window=window)

        try:
            with gzip.open(file_path_or_buffer, "rb") as f_gz:
                with fits.open(f_gz, memmap=False) as hdul:
                    wavelength, flux, invvar = preprocessor.load_spectrum(hdul)
                    header = hdul[0].header

            wavelength, flux, invvar = (
                np.asarray(d, dtype=np.float64) for d in [wavelength, flux, invvar]
            )
            flux_norm = preprocessor.normalize_spectrum(flux)

            peak_indices, properties = peak_detector.detect_peaks(wavelength, flux_norm)
            peak_wavelengths = wavelength[peak_indices]
            matched_lines = peak_detector.match_known_lines(
                peak_indices, peak_wavelengths, properties
            )

            # --- Récupération des pics associés pour les cercles verts ---
            matched_wavelengths = [
                data[0] for data in matched_lines.values() if data is not None
            ]

            # --- Création du Graphique Interactif avec Plotly ---
            fig = go.Figure()

            # 1. Ajouter le spectre
            fig.add_trace(
                go.Scatter(
                    x=wavelength,
                    y=flux_norm,
                    mode="lines",
                    name="Spectre Normalisé",
                    line=dict(color="rgba(200, 200, 200, 0.7)", width=1),
                )
            )

            # 2. Ajouter les pics détectés
            if len(peak_indices) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=peak_wavelengths,
                        y=flux_norm[peak_indices],
                        mode="markers",
                        name="Pics Détectés",
                        marker=dict(color="red", symbol="triangle-down", size=8),
                    )
                )

            # Ajout des cercles vert
            if matched_wavelengths:
                matched_indices = [
                    np.abs(wavelength - wl).argmin() for wl in matched_wavelengths
                ]
                fig.add_trace(
                    go.Scatter(
                        x=wavelength[matched_indices],
                        y=flux_norm[matched_indices],
                        mode="markers",
                        name="Pics Associés",
                        marker=dict(
                            symbol="circle-open",
                            color="lime",
                            size=15,
                            line=dict(width=2),
                        ),
                    )
                )

            # 3. Préparer les formes pour les raies cibles (lignes verticales)
            shapes = []

            # On sépare les raies de Balmer des autres pour les boutons
            _balmer_lines = {
                k: v for k, v in peak_detector.target_lines.items() if "H" in k
            }
            _other_lines = {
                k: v for k, v in peak_detector.target_lines.items() if "H" not in k
            }

            for name, wl in peak_detector.target_lines.items():
                shapes.append(
                    dict(
                        type="line",
                        xref="x",
                        yref="paper",
                        x0=wl,
                        y0=0,
                        x1=wl,
                        y1=1,
                        line=dict(color="rgba(0, 150, 255, 0.5)", width=1, dash="dash"),
                        name=name,  # On stocke le nom pour le filtrage
                    )
                )

            # --- Mise en forme du graphique ---
            subclass = header.get("SUBCLASS", "N/A")
            title = f"Analyse du Spectre : {header.get('DESIG', 'Inconnu')} (Type: {subclass})"
            fig.update_layout(
                title=title,
                xaxis_title="Longueur d'onde (Å)",
                yaxis_title="Flux Normalisé",
                template="plotly_dark",  # Thème sombre
                legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
                shapes=shapes,  # On ajoute les lignes verticales
            )

            # --- Ajout des boutons interactifs ---
            fig.update_layout(
                updatemenus=[
                    dict(
                        type="buttons",
                        direction="right",
                        x=0.5,
                        y=1.15,
                        xanchor="center",
                        yanchor="top",
                        buttons=list(
                            [
                                dict(
                                    label="Toutes les Raies",
                                    method="relayout",
                                    args=["shapes", shapes],
                                ),
                                dict(
                                    label="Raies de Balmer",
                                    method="relayout",
                                    args=[
                                        "shapes",
                                        [s for s in shapes if "H" in s["name"]],
                                    ],
                                ),
                                dict(
                                    label="Raies de Calcium",
                                    method="relayout",
                                    args=[
                                        "shapes",
                                        [s for s in shapes if "Ca" in s["name"]],
                                    ],
                                ),
                                dict(
                                    label="Aucune Raie",
                                    method="relayout",
                                    args=["shapes", []],
                                ),
                            ]
                        ),
                    )
                ]
            )

            st.plotly_chart(fig, use_container_width=True)
            return header  # On retourne le header pour l'afficher

        except Exception as e:
            st.error(f"Erreur lors de l'analyse du spectre : {e}")
            return None

    def app(self):
        """
        La méthode principale pour lancer l'application Streamlit.
        """
        import streamlit as st

        st.set_page_config(page_title="AstroSpectro Visualizer", layout="wide")
        st.title("AstroSpectro - Tableau de Bord d'Analyse")

        # --- Barre Latérale de Contrôle ---
        st.sidebar.header("Source des Données")

        source_option = st.sidebar.radio(
            "Choisir une source de spectre",
            ("Sélectionner un spectre du projet", "Téléverser un fichier FITS"),
        )

        file_to_process = None

        if source_option == "Sélectionner un spectre du projet":
            selected_file_path = st.sidebar.selectbox(
                "Sélectionner un Spectre", self.available_spectra
            )
            if selected_file_path:
                file_to_process = os.path.join(
                    self.paths["RAW_DATA_DIR"], selected_file_path
                )
        else:
            uploaded_file = st.sidebar.file_uploader(
                "Charger un spectre (.fits.gz)", type=["gz"]
            )
            if uploaded_file is not None:
                file_to_process = uploaded_file

        st.sidebar.markdown("---")
        st.sidebar.header("Paramètres de Détection")
        prominence = st.sidebar.slider("Prominence", 0.01, 1.0, 0.2, 0.01)
        window = st.sidebar.slider("Fenêtre (Å)", 1, 50, 15, 1)

        # --- Affichage Principal ---
        if file_to_process:
            header = self._plot_spectrum_analysis_plotly(
                st, file_to_process, prominence, window
            )

            if header:
                st.markdown("---")
                with st.expander("Afficher l'En-tête FITS Complet"):
                    self._display_header_for_streamlit(st, header)

    def _display_header_for_streamlit(self, st, header):
        """
        Affiche un objet header FITS dans Streamlit.
        """
        md_output = "#### Métadonnées Principales\n"

        # Affiche quelques clés importantes
        md_output += (
            f"- **Objet :** `{header.get('OBJECT', header.get('DESIG', 'N/A'))}`\n"
        )
        md_output += f"- **Type :** `{header.get('SUBCLASS', 'N/A')}`\n"
        md_output += f"- **Date Obs :** `{header.get('DATE-OBS', 'N/A')}`\n"
        md_output += f"- **RA / Dec :** `{header.get('RA', 'N/A')}` / `{header.get('DEC', 'N/A')}`\n"
        st.markdown(md_output)

        # Affiche le header complet sous forme de dictionnaire
        st.json(dict(header))

    # ==============================================================================
    # Outil 9 : Tableau de Bord du Dataset Spectral ---
    # ==============================================================================

    def display_dataset_dashboard(self) -> pd.DataFrame:
        """
        Affiche un mini tableau de bord sur l'état actuel du dataset (dans un notebook)
        et retourne un petit résumé sous forme de DataFrame.

        Contenu affiché
        ---------------
        - Compte des spectres disponibles (data/raw) vs. déjà utilisés pour l'entraînement.
        - Nombre de nouveaux spectres potentiellement exploitables.
        - (optionnel) Statistiques par sous-classe si le fichier
        ``<CATALOG_DIR>/master_catalog_temp.csv`` est présent.
        - (optionnel) Comptage par plan (déduit du 1er segment du chemin).

        Returns
        -------
        pandas.DataFrame
            Un tableau récapitulatif avec les compteurs principaux.
        """
        from IPython.display import display, Markdown, HTML
        import io
        import base64

        display(Markdown("### Tableau de Bord du Dataset Spectral"))
        display(
            Markdown(
                "Ce panneau donne un aperçu rapide de votre collection de spectres "
                "et met en évidence d’éventuelles opportunités de nouveaux entraînements."
            )
        )

        summary_rows = []

        # -- 1) Comptage global -------------------------------------------------------
        try:
            # utilise le scan déjà fait au __init__
            available_files = list(getattr(self, "available_spectra", []) or [])
            total_available = len(available_files)

            # log d'entraînement (via DatasetBuilder) s'il existe
            builder = DatasetBuilder(
                catalog_dir=self.paths["CATALOG_DIR"],
                raw_data_dir=self.paths["RAW_DATA_DIR"],
            )
            try:
                trained_log = builder.load_trained_log()
                total_trained = len(trained_log) if trained_log is not None else 0
            except Exception:
                total_trained = 0

            total_new = max(0, total_available - total_trained)

            md = textwrap.dedent(
                f"""
            - **Spectres téléchargés (data/raw)** : **{total_available}**
            - **Spectres déjà utilisés pour l’entraînement** : **{total_trained}**
            - **Nouveaux spectres potentiels** : **{total_new}**
            """
            ).strip()
            display(Markdown(md))

            summary_rows.append({"metric": "total_available", "value": total_available})
            summary_rows.append({"metric": "total_trained", "value": total_trained})
            summary_rows.append({"metric": "total_new_candidates", "value": total_new})
        except Exception as e:
            display(Markdown(f"> Erreur lors du comptage global : `{e}`"))

        # -- 2) Statistiques par sous-classe (si catalogue temp dispo) ----------------
        try:
            cat_path = os.path.join(
                self.paths["CATALOG_DIR"], "master_catalog_temp.csv"
            )
            if os.path.exists(cat_path):
                import pandas as pd

                df_cat = pd.read_csv(cat_path, sep="|")

                if {"subclass"}.issubset(df_cat.columns):
                    counts = (
                        df_cat["subclass"]
                        .astype(str)
                        .value_counts(dropna=False)
                        .rename_axis("subclass")
                        .reset_index(name="count")
                    )

                    display(Markdown("#### Top sous-classes (top 15)"))
                    display(counts.head(15))

                    # Liens d'export CSV/LaTeX (pratique pour un rapport)
                    csv_buf = io.StringIO()
                    counts.to_csv(csv_buf, index=False)
                    b64_csv = base64.b64encode(csv_buf.getvalue().encode()).decode()

                    tex_buf = counts.to_latex(
                        index=False, caption="Fréquence des sous-classes"
                    )
                    b64_tex = base64.b64encode(tex_buf.encode()).decode()

                    display(
                        HTML(
                            "<div style='margin:8px 0'>"
                            "<a download='subclass_counts.csv' href='data:text/csv;base64,{csv}'>⬇️ CSV</a>"
                            " &nbsp;|&nbsp; "
                            "<a download='subclass_counts.tex' href='data:text/plain;base64,{tex}'>⬇️ LaTeX</a>"
                            "</div>".format(csv=b64_csv, tex=b64_tex)
                        )
                    )

                    # Compteurs utiles pour le résumé retourné
                    summary_rows.append(
                        {
                            "metric": "distinct_subclasses",
                            "value": int(counts["subclass"].nunique()),
                        }
                    )
        except Exception as e:
            display(
                Markdown(f"> Impossible de lire le catalogue des sous-classes : `{e}`")
            )

        # -- 3) Comptage par plan (déduit des chemins) --------------------------------
        try:
            if available_files:
                import pandas as pd

                plans = [p.split("/")[0] for p in available_files]
                vc = (
                    pd.Series(plans, name="plan_id")
                    .value_counts()
                    .rename_axis("plan_id")
                    .reset_index(name="count")
                )
                display(Markdown("#### Spectres par plan (top 20)"))
                display(vc.head(20))

                summary_rows.append(
                    {"metric": "distinct_plans", "value": int(vc["plan_id"].nunique())}
                )
        except Exception as e:
            display(Markdown(f"> Erreur lors du comptage par plan : `{e}`"))

        # -- 4) Retour d’un petit DataFrame récap -------------------------------------
        import pandas as pd

        summary_df = pd.DataFrame(summary_rows, columns=["metric", "value"])
        return summary_df

    # ==============================================================================
    # Outil 10 : Analyse d'Interprétabilité des Modèles (SHAP) ---
    # ==============================================================================

    # ---------------------------------------------------------------------
    # Helper : prépare X pour le modèle (aligne colonnes + étapes pipeline)
    # ---------------------------------------------------------------------

    def _prepare_for_model(self, df: pd.DataFrame, clf_wrapper):
        """
        Aligne un DataFrame de features sur les colonnes d'entraînement du modèle,
        puis applique les étapes *amont* du pipeline (imputer/scaler/selector).

        Parameters
        ----------
        df : pd.DataFrame
            Features calculées dans l’espace "brut" (FeatureEngineer).
        clf_wrapper : SpectralClassifier
            Objet chargé via SpectralClassifier.load_model(model_path).

        Returns
        -------
        Xt : np.ndarray
            Matrice transformée **prête** pour .predict() du classifieur.
        feature_names_for_plot : list[str]
            Les noms de colonnes correspondant à Xt (après éventuelle sélection).
        """
        pipe = clf_wrapper.model_pipeline

        # 1) Colonnes d'entraînement attendues
        used_cols = list(getattr(clf_wrapper, "feature_names_used", list(df.columns)))
        X = df.copy()
        for c in used_cols:
            if c not in X.columns:
                X[c] = np.nan
        X = X.reindex(columns=used_cols)

        # 2) Imputer + scaler (si présents)
        imp = pipe.named_steps.get("imputer")
        scl = pipe.named_steps.get("scaler")
        Xt = X.values
        if imp is not None:
            Xt = imp.transform(X)
        if scl is not None:
            Xt = scl.transform(Xt)

        # 3) Sélection de features (si présente dans le pipeline)
        sel = pipe.named_steps.get("feature_selector")
        if sel is not None:
            Xt = sel.transform(Xt)
            feature_names_for_plot = list(
                getattr(clf_wrapper, "selected_features_", [])
                or used_cols[: Xt.shape[1]]
            )
        else:
            feature_names_for_plot = used_cols

        # Sécurise la concordance tailles <-> noms (cas artefact incomplet)
        if len(feature_names_for_plot) != Xt.shape[1]:
            feature_names_for_plot = used_cols[: Xt.shape[1]]

        return Xt, feature_names_for_plot

    # ---------------------------------------------------------------------
    # Lancement d’une analyse SHAP "end-to-end" sur un échantillon
    # ---------------------------------------------------------------------

    def _run_shap_analysis(
        self, model_path: str, sample_n: int = 500
    ) -> pd.DataFrame | None:
        """
        Charge un modèle, calcule un échantillon de features, aligne sur le pipeline
        et produit un tableau des importances SHAP (moyenne |valeur| par feature).
        Exporte aussi CSV/LaTeX dans logs/shap et mémorise les objets utiles aux tracés.

        Parameters
        ----------
        model_path : str
            Chemin absolu du .pkl (SpectralClassifier sauvegardé).
        sample_n : int, default=500
            Taille d’échantillon de spectres (bornée par le disponible).

        Returns
        -------
        pd.DataFrame | None
            Tableau des importances moyennes SHAP (|valeur|) trié décroissant,
            ou None en cas d’impossibilité.
        """
        try:
            # -- 1) Charger le classifieur
            clf_wrapper = SpectralClassifier.load_model(model_path)
            model = clf_wrapper.model_pipeline.named_steps.get("clf")
            if model is None:
                display(Markdown("### Erreur\nPipeline incomplet : pas d'étape `clf`."))
                return None

            # -- 2) Échantillonner des spectres et calculer les features
            files = list(self.available_spectra)
            if not files:
                display(Markdown("> Aucun spectre disponible pour SHAP."))
                return None

            files = files[: max(1, min(sample_n, len(files)))]
            pre = SpectraPreprocessor()
            det = PeakDetector(prominence=0.2, window=15)
            fe = FeatureEngineer()

            rows = []
            for rel in files:
                try:
                    with gzip.open(
                        os.path.join(self.paths["RAW_DATA_DIR"], rel), "rb"
                    ) as f_gz:
                        with fits.open(f_gz, memmap=False) as hdul:
                            wl, fl, inv = pre.load_spectrum(hdul)
                    fl = pre.normalize_spectrum(fl)
                    wl = np.asarray(wl, float)
                    fl = np.asarray(fl, float)
                    inv = np.asarray(inv, float)

                    pidx, props = det.detect_peaks(wl, fl)
                    matched = det.match_known_lines(pidx, wl[pidx], props)
                    vec = fe.extract_features(matched, wl, fl, inv)
                    rows.append(np.asarray(vec, float))
                except Exception:
                    # on ignore les spectres problématiques
                    continue

            if not rows:
                display(
                    Markdown("> Impossible de construire un échantillon de features.")
                )
                return None

            X_df = pd.DataFrame(np.vstack(rows), columns=fe.feature_names)

            # -- 3) Aligner sur le pipeline du modèle (imputer/scaler/selector)
            Xt, names_for_plot = self._prepare_for_model(X_df, clf_wrapper)

            # -- 4) SHAP : TreeExplainer pour RF/XGB, sinon Explainer générique
            try:
                explainer = shap.TreeExplainer(model)
            except Exception:
                explainer = shap.Explainer(model)

            sv = explainer(Xt, check_additivity=False)  # Explanation ou ndarray
            values = getattr(sv, "values", sv)
            values = np.asarray(values)

            # Agrégation éventuelle si multi-sorties (n_samples, n_features, n_outputs)
            if values.ndim >= 3:
                values_2d = values.mean(axis=-1)
            elif values.ndim == 2:
                values_2d = values
            else:
                raise ValueError(f"Forme SHAP inattendue: {values.shape}")

            # Importances : moyenne de |valeur| par feature
            mean_abs = np.abs(values_2d).mean(axis=0)  # (n_features,)
            n_feat = mean_abs.shape[-1]
            if len(names_for_plot) != n_feat:
                names_for_plot = list(names_for_plot)[:n_feat]

            order = np.argsort(mean_abs)[::-1]
            shap_df = pd.DataFrame(
                {
                    "feature": np.array(names_for_plot)[order],
                    "mean_abs_shap": mean_abs[order],
                }
            )

            # -- 5) Exports dans logs/shap/
            ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            out_dir = os.path.join(self.paths["LOGS_DIR"], "shap")
            os.makedirs(out_dir, exist_ok=True)
            try:
                shap_df.to_csv(
                    os.path.join(out_dir, f"shap_importances_{ts}.csv"), index=False
                )
            except Exception:
                pass
            try:
                with open(
                    os.path.join(out_dir, f"shap_importances_{ts}.tex"),
                    "w",
                    encoding="utf-8",
                ) as f:
                    f.write(
                        shap_df.to_latex(
                            index=False, caption="Importances SHAP (|valeur| moyenne)"
                        )
                    )
            except Exception:
                pass

            display(shap_df.head(40))

            # -- 6) Mémo pour tracés ultérieurs
            # on construit une Explanation **2D** pour les plots (beeswarm/bar)
            try:
                base_vals = getattr(sv, "base_values", None)
                if isinstance(base_vals, np.ndarray) and base_vals.ndim >= 2:
                    base_vals = base_vals.mean(axis=-1)  # (n_samples,)
            except Exception:
                base_vals = None

            try:
                # shap.Explanation(values, base_values=None, data=None, feature_names=None)
                sv2d = shap.Explanation(
                    values=values_2d,
                    base_values=base_vals,
                    data=Xt[:, : values_2d.shape[1]],
                    feature_names=list(names_for_plot),
                )
            except Exception:
                sv2d = None  # on retombe sur le bar custom si besoin

            self._last_shap_explanation = sv2d
            self._last_shap_feature_names = list(names_for_plot)
            self._last_shap_importances = shap_df.copy()
            self._last_shap_samples = Xt  # utile si besoin

            return shap_df

        except Exception as e:
            display(Markdown(f"### Erreur lors du calcul/affichage SHAP\n`{e}`"))
            return None

    def interactive_shap_explainer(self) -> None:
        """
        Crée un petit **widget notebook** pour sélectionner un modèle sauvegardé
        (.pkl) et lancer une analyse d'interprétabilité **SHAP**.

        Le workflow :
        1) scan du dossier `data/models/` (ou `self.paths["MODELS_DIR"]`)
        2) choix d'un modèle via un Dropdown
        3) réglage du nombre d'échantillons (pour les graphes SHAP)
        4) clic sur "Analyser" -> appelle `self._run_shap_analysis(...)`
            et affiche les graphes + un tableau d'importances.

        Notes
        -----
        - Utilise la méthode privée `_run_shap_analysis` implémentée plus haut.
        - Les graphes sont générés avec Matplotlib + SHAP (beeswarm).
        - Un CSV des importances peut aussi être exporté dans `logs/`.
        """
        import os
        from datetime import datetime, timezone
        import pandas as pd
        from IPython.display import display, Markdown
        import ipywidgets as widgets

        display(Markdown("### Outil d'Analyse d'Interprétabilité des Modèles (SHAP)"))
        display(
            Markdown(
                "Sélectionnez un modèle entraîné pour visualiser l'importance et "
                "l'impact de chaque feature sur ses prédictions."
            )
        )

        # 1) modèles trouvés (triés du plus récent)
        models_dir = self.paths.get("MODELS_DIR", "../data/models/")
        saved_models: list[str] = []
        if os.path.exists(models_dir):
            saved_models = sorted(
                [
                    os.path.join(models_dir, f)
                    for f in os.listdir(models_dir)
                    if f.endswith(".pkl")
                ],
                key=os.path.getmtime,
                reverse=True,
            )
        if not saved_models:
            print("Aucun modèle entraîné (.pkl) trouvé dans le dossier 'data/models/'.")
            return

        # 2) widgets
        dd_model = widgets.Dropdown(
            options=saved_models, description="Modèle :", layout={"width": "800px"}
        )
        sample_slider = widgets.IntSlider(
            value=500,
            min=100,
            max=2000,
            step=50,
            description="Échantillons :",
            continuous_update=False,
            layout={"width": "400px"},
        )
        topk_slider = widgets.IntSlider(
            value=30,
            min=5,
            max=100,
            step=5,
            description="Top N :",
            continuous_update=False,
            layout={"width": "300px"},
        )
        run_btn = widgets.Button(
            description="Analyser", button_style="success", icon="bar-chart"
        )
        out = widgets.Output()

        # 3) callback
        def _on_click(_):
            out.clear_output()
            with out:
                shap_df = self._run_shap_analysis(
                    model_path=dd_model.value, sample_n=int(sample_slider.value)
                )
                if isinstance(shap_df, pd.DataFrame) and not shap_df.empty:
                    display(Markdown("**Importances SHAP (moyenne |valeur|) :**"))
                    display(shap_df.head(30))
                    k = int(topk_slider.value)
                    display(
                        Markdown(f"**Importances SHAP (Top {k}, moyenne |valeur|) :**")
                    )
                    display(shap_df.head(k))

                    # export (chemin montré pour confirmation)
                    try:
                        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                        out_dir = os.path.join(self.paths["LOGS_DIR"], "shap")
                        os.makedirs(out_dir, exist_ok=True)
                        csv_path = os.path.join(out_dir, f"shap_importances_{ts}.csv")
                        tex_path = os.path.join(out_dir, f"shap_importances_{ts}.tex")
                        shap_df.to_csv(csv_path, index=False)
                        shap_df.to_latex(tex_path, index=False, float_format="%.6f")
                        display(Markdown(f"Export CSV : `{csv_path}`"))
                        display(Markdown(f"Export LaTeX : `{tex_path}`"))
                    except Exception as e:
                        display(
                            Markdown(f"> Impossible d'exporter (CSV/LaTeX) : `{e}`")
                        )

        run_btn.on_click(_on_click)

        # 4) layout
        display(
            widgets.VBox(
                [widgets.HBox([dd_model, sample_slider, topk_slider, run_btn]), out]
            )
        )

    # ---------------------------------------------------------------------
    # Helpers de tracé réutilisables dans d’autres cellules
    # ---------------------------------------------------------------------

    def plot_shap_summary_bar(
        self,
        top_n: int = 20,
        save_path: str | None = None,
    ) -> tuple[plt.Figure, plt.Axes] | None:
        """
        Trace un **bar chart** des importances SHAP (moyenne de la valeur absolue).

        Pré-requis
        ----------
        - Avoir exécuté `_run_shap_analysis(...)` (ou le widget `interactive_shap_explainer()`),
        qui mémorise le dernier résultat dans `self._last_shap_importances`.

        Paramètres
        ----------
        top_n : int, default=20
            Nombre maximum de features à afficher (les plus importantes).
        save_path : str | None, default=None
            Chemin d'export PNG. Si None, la figure n'est pas sauvegardée.
            Le(s) dossier(s) parent(s) sont créés au besoin.

        Returns
        -------
        (matplotlib.figure.Figure, matplotlib.axes.Axes) | None
            La figure et ses axes si un résultat SHAP est disponible, sinon None.

        Notes
        -----
        - Le graphique est trié **du plus important au moins important** (en haut → bas),
        avec une échelle `mean(|SHAP value|)`.
        - Thème sombre pour être cohérent avec le reste des tracés.
        """
        # Récupère le dernier tableau d'importances calculé par _run_shap_analysis
        df = getattr(self, "_last_shap_importances", None)
        if df is None or df.empty:
            print(
                "Aucun résultat SHAP mémorisé. Lance d'abord interactive_shap_explainer()."
            )
            return None

        # Sélectionne le top N
        top = df.head(int(top_n))

        # Mise en forme du plot
        plt.style.use("dark_background")
        fig, ax = plt.subplots(figsize=(12, max(5, 0.40 * len(top))))

        # barh inversé pour avoir la feature la plus importante en haut
        y_labels = top["feature"].astype(str).iloc[::-1]
        x_vals = top["mean_abs_shap"].astype(float).iloc[::-1]
        ax.barh(y_labels, x_vals)

        ax.set_xlabel("mean(|SHAP value|)")
        ax.set_title("SHAP — importances (top)")
        ax.grid(axis="x", alpha=0.30, linestyle="--")

        fig.tight_layout()

        # Sauvegarde optionnelle
        if save_path:
            try:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                fig.savefig(save_path, dpi=150, bbox_inches="tight")
            except Exception as e:
                print(f"[!] Impossible de sauvegarder la figure ({save_path}) : {e}")

        return fig, ax

    def plot_shap_beeswarm(
        self,
        max_display: int = 15,
        save_path: str | None = None,
    ) -> plt.Figure | None:
        """
        Trace un **beeswarm SHAP** (distribution des contributions par feature),
        avec gestion de **fallback** si le beeswarm n’est pas disponible.

        Pré-requis
        ----------
        - Avoir exécuté `_run_shap_analysis(...)` (ou le widget `interactive_shap_explainer()`),
        qui mémorise la dernière explication SHAP dans `self._last_shap_explanation`.

        Paramètres
        ----------
        max_display : int, default=15
            Nombre maximum de features affichées.
        save_path : str | None, default=None
            Chemin d'export PNG. Si None, la figure n'est pas sauvegardée.
            Le(s) dossier(s) parent(s) sont créés au besoin.

        Returns
        -------
        (matplotlib.figure.Figure, matplotlib.axes.Axes) | None
            La figure et ses axes si l’explication SHAP est disponible, sinon None.

        Notes
        -----
        - Si `shap.plots.beeswarm(...)` échoue (p.ex. formes non supportées),
        on essaie `shap.plots.bar(...)`. Si cela échoue aussi, on retombe
        sur notre bar chart custom (`plot_shap_summary_bar`).
        - Thème sombre et mise en page serrée par défaut.
        """
        sv = getattr(self, "_last_shap_explanation", None)
        if sv is None:
            print(
                "Aucune explication SHAP mémorisée. Lance d'abord interactive_shap_explainer() "
                "ou _run_shap_analysis(...)."
            )
            return None

        plt.style.use("dark_background")
        fig = plt.figure(figsize=(10, 6))
        ax = plt.gca()

        # Tentative beeswarm → fallback bar SHAP → fallback bar custom
        try:
            shap.plots.beeswarm(sv, max_display=int(max_display), show=False)
        except Exception as e1:
            try:
                shap.plots.bar(sv, max_display=int(max_display), show=False)
            except Exception as e2:
                print(
                    f"[!] Beeswarm indisponible ({e1}). Fallback bar SHAP indisponible ({e2}). "
                    "On revient au bar chart custom des importances."
                )
                # Dernier recours : notre bar chart basé sur le DF mémorisé
                return self.plot_shap_summary_bar(
                    top_n=int(max_display), save_path=save_path
                )

        fig = plt.gcf()
        ax = plt.gca()
        fig.tight_layout()

        if save_path:
            try:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                fig.savefig(save_path, dpi=150, bbox_inches="tight")
            except Exception as e:
                print(f"[!] Impossible de sauvegarder la figure ({save_path}) : {e}")

        return fig, ax

    # ==============================================================================
    # Outil 11 : Analyse des sous-classes spectrales ---
    # ==============================================================================

    def plot_subclass_distribution(
        self,
        top_n: int = 20,
        normalize: bool = False,
        save_path: Optional[str] = None,
    ) -> Optional[tuple[plt.Figure, plt.Axes]]:
        """
        Affiche (et optionnellement sauvegarde) la **distribution des sous-classes spectrales**
        à partir du catalogue temporaire `master_catalog_temp.csv`.

        Paramètres
        ----------
        top_n : int, default=20
            Nombre maximal de sous-classes affichées (les plus fréquentes).
        normalize : bool, default=False
            Si True, affiche des fréquences relatives (pourcentage). Sinon, des comptes bruts.
        save_path : str | None, default=None
            Chemin d'export PNG. Si None, la figure n'est pas sauvegardée.

        Returns
        -------
        (Figure, Axes) ou None
            Retourne la figure et les axes Matplotlib si tout s'est bien passé.
            Retourne None si le catalogue est introuvable ou vide.

        Notes
        -----
        - Cherche une colonne `'subclass'`. Si absente, essaie `'label'`.
        - Trie par fréquence décroissante.
        """
        import os
        import pandas as pd
        import matplotlib.pyplot as plt
        from IPython.display import display, Markdown

        # 1) Charger le catalogue temporaire
        cat_path = os.path.join(self.paths["CATALOG_DIR"], "master_catalog_temp.csv")
        if not os.path.exists(cat_path):
            display(
                Markdown(
                    "> Catalogue temporaire introuvable : `master_catalog_temp.csv`."
                )
            )
            return None

        try:
            df = pd.read_csv(cat_path, sep="|")
        except Exception as e:
            display(Markdown(f"> Erreur de lecture du catalogue : `{e}`"))
            return None

        if df.empty:
            display(Markdown("> Catalogue vide."))
            return None

        # 2) Identifier la colonne des sous-classes
        label_col = None
        for cand in ("subclass", "label"):
            if cand in df.columns:
                label_col = cand
                break

        if label_col is None:
            display(Markdown("> Aucune colonne `subclass` ou `label` trouvée."))
            return None

        # 3) Comptage & sélection top N
        counts = df[label_col].astype(str).value_counts(dropna=False)
        if counts.empty:
            display(Markdown("> Aucun label exploitable dans le catalogue."))
            return None

        counts = counts.iloc[:top_n].copy()

        # 4) Normalisation éventuelle
        if normalize:
            total = counts.sum()
            values = (counts / total) * 100.0
            ylabel = "Fréquence (%)"
        else:
            values = counts
            ylabel = "Nombre d'exemples"

        # 5) Plot
        plt.style.use("dark_background")
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(counts.index.astype(str), values.values)
        ax.set_title("Distribution des Sous-Classes Spectrales")
        ax.set_xlabel("Sous-classe")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        fig.tight_layout()

        # 6) Sauvegarde optionnelle
        if save_path:
            try:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                fig.savefig(save_path, dpi=150, bbox_inches="tight")
            except Exception as e:
                display(Markdown(f"> Impossible de sauvegarder la figure : `{e}`"))

        return fig, ax

    # ==============================================================================
    # Outil 12 : Comparateur de normalisation de spectre
    # ==============================================================================

    def plot_normalization_comparison(
        self,
        sample_paths: list[str] | None = None,
        *,
        n_samples: int = 2,
        random_state: int | None = None,
        save_path: str | None = None,
    ) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes]] | None:
        """
        Compare visuellement l'effet de la **normalisation** sur deux spectres.

        Deux sous-graphes sont tracés :
        1) flux bruts (“Avant normalisation”)
        2) flux normalisés (“Après normalisation”)

        Si `save_path` est omis, la figure est sauvegardée par défaut dans :
            <PROJECT_ROOT>/website/static/img/avant_apres_normalisation.png

        Paramètres
        ----------
        sample_paths : list[str] | None
            Chemins **relatifs** (sous `RAW_DATA_DIR`) des spectres à utiliser.
            Si None, on prélève aléatoirement `n_samples` fichiers parmi
            `self.available_spectra`.
        n_samples : int, default 2
            Nombre de spectres à comparer si `sample_paths` n'est pas fourni.
        random_state : int | None
            Graine pour l’échantillonnage aléatoire reproductible.
        save_path : str | None
            Chemin d’export PNG. Si None => chemin par défaut. Si "", pas d’export.

        Returns
        -------
        (fig, (ax1, ax2)) | None
            La figure et les axes Matplotlib si au moins 2 spectres sont disponibles,
            sinon `None`.

        Notes
        -----
        - Les chemins fournis ou échantillonnés doivent être relatifs à `RAW_DATA_DIR`
        et pointer vers des fichiers compressés `.fits.gz`.
        - Utilise `SpectraPreprocessor.normalize_spectrum` pour la normalisation.
        """
        import random

        # --- Vérifications préalables ---------------------------------------------
        if not getattr(self, "available_spectra", []):
            print(
                "Veuillez d'abord exécuter la cellule de setup pour initialiser `visualizer`."
            )
            return None

        # Sélection des fichiers (2 par défaut)
        if sample_paths is None:
            pool = list(self.available_spectra)
            if len(pool) < 2:
                print(
                    "Pas assez de spectres disponibles pour générer la figure (il en faut au moins 2)."
                )
                return None
            if random_state is not None:
                random.seed(random_state)
            sample_paths = random.sample(pool, k=max(2, min(n_samples, len(pool))))
        else:
            # Nettoie et s'assure d'avoir au moins 2 éléments distincts
            sample_paths = [p for p in (sample_paths or []) if p]
            sample_paths = list(dict.fromkeys(sample_paths))  # dédoublonne
            if len(sample_paths) < 2:
                print("Fournir au moins deux chemins relatifs de spectres `.fits.gz`.")
                return None

        # --- Chargement & prétraitement -------------------------------------------
        preprocessor = SpectraPreprocessor()
        spectra_data: list[dict] = []

        for rel in sample_paths[:2]:  # on n’affiche que 2 courbes pour la lisibilité
            full_path = os.path.join(self.paths["RAW_DATA_DIR"], rel)
            try:
                with gzip.open(full_path, "rb") as f_gz:
                    with fits.open(f_gz, memmap=False) as hdul:
                        wavelength, flux, _ = preprocessor.load_spectrum(hdul)
                flux_norm = preprocessor.normalize_spectrum(np.asarray(flux, float))
                spectra_data.append(
                    {
                        "wavelength": np.asarray(wavelength, float),
                        "flux_raw": np.asarray(flux, float),
                        "flux_norm": np.asarray(flux_norm, float),
                        "name": os.path.basename(rel),
                    }
                )
            except Exception as e:
                print(f"[!] Lecture impossible pour {rel} : {e}")

        if len(spectra_data) < 2:
            print("Impossible de charger deux spectres valides.")
            return None

        # --- Figure ---------------------------------------------------------------
        plt.style.use("dark_background")
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 6), sharex=True)

        # Avant normalisation (flux bruts)
        ax1.plot(
            spectra_data[0]["wavelength"],
            spectra_data[0]["flux_raw"],
            label="Spectre 1 (Brut)",
            alpha=0.9,
        )
        ax1.plot(
            spectra_data[1]["wavelength"],
            spectra_data[1]["flux_raw"],
            label="Spectre 2 (Brut)",
            alpha=0.9,
        )
        ax1.set_title("Avant Normalisation", fontsize=16)
        ax1.set_ylabel("Flux (unités arbitraires)")
        ax1.grid(True, linestyle=":", alpha=0.5)
        ax1.legend()

        # Après normalisation (flux normalisés)
        ax2.plot(
            spectra_data[0]["wavelength"],
            spectra_data[0]["flux_norm"],
            label="Spectre 1 (Normalisé)",
        )
        ax2.plot(
            spectra_data[1]["wavelength"],
            spectra_data[1]["flux_norm"],
            label="Spectre 2 (Normalisé)",
        )
        ax2.set_title("Après Normalisation", fontsize=16)
        ax2.set_xlabel("Longueur d’onde (Å)")
        ax2.set_ylabel("Flux Normalisé")
        ax2.grid(True, linestyle=":", alpha=0.5)
        ax2.legend()

        fig.suptitle("Impact de la Normalisation sur les Spectres", fontsize=20, y=1.02)
        fig.tight_layout(rect=(0, 0, 1, 0.98))

        # --- Sauvegarde -----------------------------------------------------------
        if save_path is None:
            # Chemin par défaut dans website/static/img/
            save_path = os.path.join(
                self.paths.get("PROJECT_ROOT", "."),
                "website",
                "static",
                "img",
                "avant_apres_normalisation.png",
            )

        if save_path:
            try:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                fig.savefig(save_path, dpi=150, bbox_inches="tight")
                print(f"Figure de normalisation sauvegardée dans : {save_path}")
            except Exception as e:
                print(f"[!] Échec de la sauvegarde ({e})")

        return fig, (ax1, ax2)

    # ==============================================================================
    # Outil 13 : Explorateur et Analyse des Features
    # ==============================================================================

    def feature_explorer(
        self,
        pattern: str | None = None,
        *,
        save_dir: str | None = None,
        max_hists: int = 24,
        corr_top_n: int = 30,
        rf_estimators: int = 300,
        random_state: int = 42,
        do_permutation: bool = True,
    ) -> dict | None:
        """
        Charge et concatène les fichiers CSV de features, effectue une analyse
        exploratoire (EDA), trace des distributions et corrélations, puis entraîne
        un modèle baseline (RandomForest) pour estimer des importances de variables.

        Paramètres
        ----------
        pattern : str | None
            Motif glob des fichiers features (par ex. ".../data/processed/features_*.csv").
            Si None, essaie automatiquement "<PROJECT_ROOT>/data/processed/features_*.csv".
        save_dir : str | None
            Dossier d’export (PNGs + CSV). Si None => "<LOGS_DIR>/features".
        max_hists : int, default=24
            Nombre max. d’histogrammes tracés (colonnes numériques à plus forte variance).
        corr_top_n : int, default=30
            Nombre de colonnes (numériques, plus forte variance) pour la heatmap de corrélation.
        rf_estimators : int, default=300
            n_estimators pour le RandomForestClassifier baseline.
        random_state : int, default=42
            Graine pour la reproductibilité (échantillonnage / RF / split).
        do_permutation : bool, default=True
            Calcule aussi la permutation importance (validation split).

        Returns
        -------
        dict | None
            Un dictionnaire récapitulatif :
            {
            "df": DataFrame concaténé,
            "numeric_cols": [...],
            "label_col": "subclass" | "label" | None,
            "rf_importances": DataFrame | None,
            "perm_importances": DataFrame | None,
            "save_dir": chemin d’export
            }
            Retourne None si aucun fichier n’est trouvé.

        Notes
        -----
        - Détecte automatiquement le séparateur des CSV via `sep=None, engine='python'`.
        - Si aucune colonne label n’est trouvée ('subclass' ou 'label'), les importances
        baseline sont sautées mais l’EDA est tout de même produite.
        - Graphiques enregistrés en PNG (dark theme) si `save_dir` est défini/valide.
        """
        import os
        from textwrap import dedent

        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        from IPython.display import display, Markdown

        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        from sklearn.inspection import permutation_importance

        # ---------- Résolution des chemins ----------
        if pattern is None:
            proj = self.paths.get("PROJECT_ROOT", ".")
            pattern = os.path.join(proj, "data", "processed", "features_*.csv")
        if save_dir is None:
            save_dir = os.path.join(self.paths.get("LOGS_DIR", "./logs"), "features")
        os.makedirs(save_dir, exist_ok=True)

        # ---------- Lecture & concaténation ----------
        files = sorted(glob.glob(pattern))
        if not files:
            display(Markdown(f"> **Aucun fichier** trouvé pour le motif : `{pattern}`"))
            return None

        dfs = []
        for p in files:
            try:
                df_i = pd.read_csv(p, sep=None, engine="python")
                dfs.append(df_i)
            except Exception as e:
                print(f"[!] Lecture impossible: {p} ({e})")

        if not dfs:
            display(Markdown("> Aucun CSV valide n’a pu être chargé."))
            return None

        df = pd.concat(dfs, ignore_index=True)
        display(
            Markdown(
                f"**{len(files)}** fichier(s) chargé(s) — total lignes : **{len(df):,}**"
            )
        )

        # ---------- EDA rapide ----------
        # Résumé types & manquants
        nunique = df.nunique(dropna=True)
        missing = df.isna().sum()
        missing_rate = (missing / len(df)).round(4)

        info_rows = [
            ("rows", len(df)),
            ("cols", df.shape[1]),
            ("numeric_cols", int(df.select_dtypes(include=[np.number]).shape[1])),
            ("non_numeric_cols", int(df.select_dtypes(exclude=[np.number]).shape[1])),
            ("missing_total", int(missing.sum())),
            ("missing_rate_overall", float(missing_rate.mean())),
        ]
        summary_df = pd.DataFrame(info_rows, columns=["metric", "value"])
        display(Markdown("### Résumé global"))
        display(summary_df)

        # Top manquants
        miss_tbl = (
            pd.DataFrame(
                {"missing": missing, "missing_rate": missing_rate, "nunique": nunique}
            )
            .sort_values("missing_rate", ascending=False)
            .head(30)
        )
        display(Markdown("### Colonnes les plus manquantes"))
        display(miss_tbl)

        # describe numérique
        num_df = df.select_dtypes(include=[np.number])
        if not num_df.empty:
            display(Markdown("### Statistiques descriptives (numériques)"))
            display(num_df.describe().T.head(30))

        # ---------- Figures : distributions ----------
        if not num_df.empty and max_hists > 0:
            # Colonnes triées par variance (desc)
            var = num_df.var(numeric_only=True).sort_values(ascending=False)
            cols = list(var.index[:max_hists])

            plt.style.use("dark_background")
            n = len(cols)
            ncols = 4
            nrows = int(np.ceil(n / ncols))
            fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 2.8 * nrows))
            axes = np.atleast_1d(axes).ravel()

            for i, c in enumerate(cols):
                ax = axes[i]
                ax.hist(num_df[c].dropna(), bins=30)
                ax.set_title(c, fontsize=10)
                ax.grid(True, linestyle=":", alpha=0.3)

            for j in range(i + 1, len(axes)):
                axes[j].axis("off")

            fig.suptitle("Distributions (top variance)", fontsize=14)
            fig.tight_layout()
            try:
                fig.savefig(
                    os.path.join(save_dir, "feature_hists.png"),
                    dpi=140,
                    bbox_inches="tight",
                )
            except Exception:
                pass
            plt.show()

        # ---------- Figure : corrélation (top variance) ----------
        corr_cols = list(num_df.var().sort_values(ascending=False).index[:corr_top_n])
        if len(corr_cols) >= 2:
            # Remplace NaN par médiane pour corr; évite colonnes constantes
            Xcorr = num_df[corr_cols].copy()
            for c in Xcorr.columns:
                if Xcorr[c].isna().any():
                    med = Xcorr[c].median()
                    Xcorr[c] = Xcorr[c].fillna(med)
            # supprime colonnes constantes (variance nulle)
            Xcorr = Xcorr.loc[:, Xcorr.var() > 0]
            if Xcorr.shape[1] >= 2:
                C = Xcorr.corr().values

                plt.style.use("dark_background")
                fig, ax = plt.subplots(figsize=(max(8, 0.35 * Xcorr.shape[1]), 8))
                im = ax.imshow(C, vmin=-1, vmax=1, cmap="coolwarm")
                ax.set_title("Corrélations (top variance)", fontsize=14)
                ticks = range(Xcorr.shape[1])
                ax.set_xticks(ticks)
                ax.set_yticks(ticks)
                labels = list(Xcorr.columns)
                # évite de saturer les labels
                if len(labels) <= 30:
                    ax.set_xticklabels(labels, rotation=90, fontsize=8)
                    ax.set_yticklabels(labels, fontsize=8)
                else:
                    ax.set_xticklabels([])
                    ax.set_yticklabels([])
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                fig.tight_layout()
                try:
                    fig.savefig(
                        os.path.join(save_dir, "corr_heatmap.png"),
                        dpi=150,
                        bbox_inches="tight",
                    )
                except Exception:
                    pass
                plt.show()

        # ---------- Baseline importances (RF) ----------
        label_col = None
        for cand in ("subclass", "label"):
            if cand in df.columns:
                label_col = cand
                break

        rf_df = None
        perm_df = None

        if label_col is None:
            display(
                Markdown(
                    "> Aucune colonne label trouvée (`subclass`/`label`). Importances sautées."
                )
            )
        else:
            # On ne garde que les colonnes numériques pour la baseline simple
            X_num = df.select_dtypes(include=[np.number]).copy()
            y = df[label_col].astype(str)

            # supprime colonnes constantes
            X_num = X_num.loc[:, X_num.var() > 0]
            if X_num.empty:
                display(
                    Markdown("> Aucune feature numérique exploitable (variance nulle).")
                )
            else:
                # --- Split train/test (robuste aux classes très rares) ---
                # y est la série/array des labels; X_num les features numériques déjà préparées
                class_counts = pd.Series(y).value_counts(dropna=False)

                if (class_counts < 2).any():
                    rare = class_counts[class_counts < 2]

                    # Message d'info dans le notebook
                    try:
                        display(
                            Markdown(
                                "> Split **non-stratifié** : certaines classes sont trop rares "
                                "("
                                + ", ".join([f"`{k}`={v}" for k, v in rare.items()])
                                + "). "
                                "Ajoutez des exemples ou fusionnez ces classes pour réactiver le split stratifié."
                            )
                        )
                    except Exception:
                        # si display/Markdown pas dispo, on log simplement
                        print(
                            "[Info] Split non-stratifié (classes trop rares) :",
                            dict(rare),
                        )

                    stratify_vec = None  # fallback : pas de stratification
                else:
                    stratify_vec = y  # OK pour la stratification

                # (25% test par défaut; ajuste si besoin)
                X_train, X_test, y_train, y_test = train_test_split(
                    X_num,
                    y,
                    test_size=0.25,
                    random_state=random_state,
                    stratify=stratify_vec,
                )

                pipe = Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        (
                            "rf",
                            RandomForestClassifier(
                                n_estimators=rf_estimators,
                                random_state=random_state,
                                n_jobs=-1,
                            ),
                        ),
                    ]
                )
                pipe.fit(X_train, y_train)
                rf = pipe.named_steps["rf"]

                # Importances Gini
                imp = pd.Series(
                    rf.feature_importances_, index=X_num.columns
                ).sort_values(ascending=False)
                rf_df = imp.reset_index().rename(
                    columns={"index": "feature", 0: "importance"}
                )
                display(Markdown("### Importances (RandomForest, impurity-based)"))
                display(rf_df.head(30))

                # Export CSV
                try:
                    rf_df.to_csv(
                        os.path.join(save_dir, "rf_importances.csv"), index=False
                    )
                except Exception:
                    pass

                # Tracé bar
                self._plot_feature_importances_bar(
                    rf_df,
                    top_n=25,
                    save_path=os.path.join(save_dir, "rf_importances_top25.png"),
                )

                # Permutation importance (sur test)
                if do_permutation:
                    try:
                        # calcule sur les données imputées
                        X_test_imp = pipe.named_steps["imputer"].transform(X_test)
                        result = permutation_importance(
                            rf,
                            X_test_imp,
                            y_test,
                            n_repeats=5,
                            random_state=random_state,
                            n_jobs=-1,
                        )
                        perm = pd.Series(
                            result.importances_mean, index=X_num.columns
                        ).sort_values(ascending=False)
                        perm_df = perm.reset_index().rename(
                            columns={"index": "feature", 0: "perm_importance"}
                        )
                        display(
                            Markdown("### Permutation importance (split de validation)")
                        )
                        display(perm_df.head(30))
                        try:
                            perm_df.to_csv(
                                os.path.join(save_dir, "permutation_importances.csv"),
                                index=False,
                            )
                        except Exception:
                            pass
                    except Exception as e:
                        print(f"[!] Permutation importance impossible : {e}")

        # ---------- Récap ----------
        md = dedent(
            f"""
            ### Exploration terminée
            * Fichiers chargés : **{len(files)}**
            * Lignes concaténées : **{len(df):,}**
            * Label utilisé : **{label_col or "aucun"}**
            * Exports : `{save_dir}`
            """
        )
        display(Markdown(md))

        # Mémo pour réutiliser ailleurs si besoin
        self._last_features_df = df.copy()
        self._last_features_numeric = list(
            df.select_dtypes(include=[np.number]).columns
        )
        self._last_rf_importances = rf_df
        self._last_perm_importances = perm_df

        return {
            "df": df,
            "numeric_cols": self._last_features_numeric,
            "label_col": label_col,
            "rf_importances": rf_df,
            "perm_importances": perm_df,
            "save_dir": save_dir,
        }

    # ----------------------------------------------------------------------
    # Petit helper réutilisable : bar chart des importances (top N)
    # ----------------------------------------------------------------------

    def _plot_feature_importances_bar(
        self,
        importances_df: pd.DataFrame | None = None,
        *,
        top_n: int = 25,
        save_path: str | None = None,
    ) -> tuple[plt.Figure, plt.Axes] | None:
        """
        Trace un bar chart horizontal des importances à partir d’un DataFrame
        de la forme : columns = ['feature', 'importance'].

        Si `importances_df` est None, on utilise `self._last_rf_importances`.
        """
        import os
        import matplotlib.pyplot as plt

        if importances_df is None:
            importances_df = getattr(self, "_last_rf_importances", None)

        if importances_df is None or importances_df.empty:
            print("Aucune importance à tracer.")
            return None

        top = importances_df.head(int(top_n)).copy()
        plt.style.use("dark_background")
        fig, ax = plt.subplots(figsize=(10, max(5, 0.35 * len(top))))
        ax.barh(top["feature"][::-1], top[top.columns[-1]][::-1])
        ax.set_xlabel(top.columns[-1])
        ax.set_title("Importances — top")
        ax.grid(True, axis="x", linestyle=":", alpha=0.3)
        fig.tight_layout()

        if save_path:
            try:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                fig.savefig(save_path, dpi=150, bbox_inches="tight")
            except Exception:
                pass

        return fig, ax
