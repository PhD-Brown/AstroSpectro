import os
import gzip
import base64
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from astropy.io import fits
from tqdm.notebook import tqdm
from IPython.display import display, Markdown, HTML
from ipywidgets import interact, widgets
import plotly.graph_objects as go
from specutils import Spectrum, SpectralRegion
from astropy import units as u
from astropy.nddata import StdDevUncertainty
from specutils.analysis import snr, centroid, fwhm, line_flux
from specutils.manipulation import median_smooth
import glob

# On importe les briques du pipeline pour les utiliser ici
from pipeline.preprocessor import SpectraPreprocessor
from pipeline.classifier import SpectralClassifier
from pipeline.peak_detector import PeakDetector

class AstroVisualizer:
    """
    Une suite d'outils pour visualiser et analyser interactivement
    les données du projet AstroSpectro.
    """
    def __init__(self, paths):
        """Initialise le visualiseur avec les chemins du projet."""
        self.paths = paths
        self.available_spectra = self._scan_for_spectra()

    def _scan_for_spectra(self):
        """Scanne le dossier raw/ pour trouver tous les spectres disponibles."""
        spectra_list = []
        raw_dir = self.paths.get("RAW_DATA_DIR", "../data/raw/")
        if os.path.exists(raw_dir):
            for root, _, files in os.walk(raw_dir):
                for f in files:
                    if f.endswith(".fits.gz"):
                        rel_path = os.path.relpath(os.path.join(root, f), raw_dir)
                        spectra_list.append(rel_path.replace('\\', '/'))
        return sorted(spectra_list)

    # ==============================================================================
    # Outil 1 : Affichage des Headers FITS
    # ==============================================================================
    
    def _format_header_line(self, header, label, key, unit=""):
        """Formate une ligne de manière robuste pour l'affichage en Markdown."""
        value = header.get(key, 'N/A')  # Valeur par défaut si la clé n'existe pas
        unit_str = f" {unit}" if unit else "" # Ajoute l'unité si spécifiée
        return f"- **{label} :** `{value}`{unit_str}\n" # Ajoute un retour à la ligne pour Markdown

    def _display_formatted_header(self, fits_relative_path):
        """Affiche les informations de l'en-tête FITS de manière organisée."""
        full_path = os.path.join(self.paths["RAW_DATA_DIR"], fits_relative_path)
        try:
            with gzip.open(full_path, 'rb') as f_gz:
                with fits.open(f_gz, memmap=False) as hdul:
                    header = hdul[0].header
            
            md_output = f"### En-tête du fichier : `{os.path.basename(full_path)}`\n---\n"
            sections = {}
            current_section = "Informations Générales"
            sections[current_section] = ""
            for key, value in header.items():
                if key == 'COMMENT' and '--------' in str(value):
                    current_section = str(value).replace('COMMENT', '').replace('-', '').strip()
                    if current_section not in sections: sections[current_section] = ""
                elif key and key not in ['COMMENT', 'HISTORY', '']:
                    sections[current_section] += self._format_header_line(header, key, key)
            for title, content in sections.items():
                if content: md_output += f"\n#### {title}\n{content}"
            display(Markdown(md_output))
        except Exception as e:
            display(Markdown(f"### Erreur\n**Fichier :** `{full_path}`\n\n**Détail :** {e}"))
            
    def interactive_header_explorer(self):
        """Crée et affiche un widget interactif pour explorer les headers FITS."""
        if not self.available_spectra:
            print("Aucun spectre trouvé à visualiser.")
            return
        interact(self._display_formatted_header, fits_relative_path=self.available_spectra)

    # ==============================================================================
    # Outil Notebook 2: Analyseur de Spectre Interactif (Matplotlib)
    # ==============================================================================
    
    def _plot_spectrum_analysis(self, file_path, prominence, window, xlim, ylim):
        """
        Charge, analyse et affiche un spectre, en appliquant les limites de zoom.
        """
        preprocessor = SpectraPreprocessor()
        peak_detector = PeakDetector(prominence=prominence, window=window)
        full_path = os.path.join(self.paths["RAW_DATA_DIR"], file_path)
        
        try:
            with gzip.open(full_path, 'rb') as f_gz:
                with fits.open(f_gz, memmap=False) as hdul:
                    wavelength, flux, invvar = preprocessor.load_spectrum(hdul)
                    header = hdul[0].header
            
            wavelength, flux, invvar = (np.asarray(d, dtype=np.float64) for d in [wavelength, flux, invvar])
            flux_norm = preprocessor.normalize_spectrum(flux)
            
            # --- Étape 1: Détection de TOUS les pics ---
            peak_indices, properties = peak_detector.detect_peaks(wavelength, flux_norm)
            peak_wavelengths = wavelength[peak_indices]
            
            # --- Étape 2: Association des pics ---
            matched_lines = peak_detector.match_known_lines(peak_indices, peak_wavelengths, properties)
            
            # --- Récupération des coordonnées des pics associés ---
            matched_wavelengths = [data[0] for data in matched_lines.values() if data is not None]
            
            # --- Affichage Augmenté ---
            plt.style.use('dark_background')
            plt.figure(figsize=(18, 7))
            
            plt.plot(wavelength, flux_norm, color='gray', alpha=0.7, label='Spectre Normalisé')
            if len(peak_indices) > 0:
                plt.scatter(peak_wavelengths, flux_norm[peak_indices], color='red', marker='v', s=40, label='Tous les Pics Détectés')
            
            if matched_wavelengths:
                # On doit retrouver le flux pour les pics associés
                matched_indices = [np.abs(wavelength - wl).argmin() for wl in matched_wavelengths]
                plt.scatter(wavelength[matched_indices], flux_norm[matched_indices], 
                            s=200, facecolors='none', edgecolors='lime', linewidth=2, label='Pics Associés')

            for name, wl in peak_detector.target_lines.items():
                plt.axvline(x=wl, color='dodgerblue', linestyle='--', alpha=0.8, label=f'Raie {name}')
            
            subclass = header.get('SUBCLASS', 'N/A')
            plt.title(f"Analyse du Spectre : {header.get('DESIG', 'Inconnu')} (Type: {subclass})", fontsize=16)
            plt.xlabel("Longueur d'onde (Å)"); plt.ylabel("Flux Normalisé"); plt.xlim(3800, 7000); plt.grid(True, linestyle=':')
            
            # On applique les limites des sliders aux axes du graphique
            plt.xlim(xlim)
            plt.ylim(ylim)
            
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            plt.legend(by_label.values(), by_label.keys())
            plt.show()

        except Exception as e:
            print(f"Erreur : {e}")

    def interactive_spectrum_analyzer(self):
        """
        Crée un widget interactif pour l'analyse de spectre augmentée,
        incluant des contrôles de zoom.
        """
        if not self.available_spectra:
            print("Aucun spectre trouvé à analyser.")
            return
        
        # --- Définition des widgets pour l'interface ---
        
        # Dropdown pour le spectre
        file_path_widget = widgets.Dropdown(
            options=self.available_spectra, 
            description="Spectre :", 
            layout={'width': 'max-content'}
        )
        
        # Sliders pour les paramètres de détection
        prominence_widget = widgets.FloatSlider(
            min=0.01, max=2.0, step=0.01, value=0.2, 
            description='Prominence:'
        )
        window_widget = widgets.IntSlider(
            min=1, max=50, step=1, value=15, 
            description='Fenêtre (Å):'
        )
        
        # --- NOUVEAUX WIDGETS POUR LE ZOOM ---
        # Slider de plage pour l'axe X (longueur d'onde)
        xlim_widget = widgets.FloatRangeSlider(
            value=[3800, 7000], min=3500, max=9500, step=1,
            description='Zoom X (Å):',
            continuous_update=False, # Met à jour seulement quand on relâche le slider
            layout={'width': '500px'}
        )
        
        # Slider de plage pour l'axe Y (flux)
        ylim_widget = widgets.FloatRangeSlider(
            value=[-1, 2.5], min=-2, max=5, step=0.1,
            description='Zoom Y (Flux):',
            continuous_update=False,
            layout={'width': '500px'}
        )
            
        # --- Liaison des widgets à la fonction de plot ---
        interactive_plot = widgets.interactive(
            self._plot_spectrum_analysis, # La fonction de plot est maintenant appelée par 'interactive'
            file_path=file_path_widget,
            prominence=prominence_widget,
            window=window_widget,
            xlim=xlim_widget, # On passe la valeur du slider de zoom X
            ylim=ylim_widget  # On passe la valeur du slider de zoom Y
        )
        
        # Afficher l'interface
        display(interactive_plot)

    # ==============================================================================
    # Outil 3 : Analyseur & Tuner de Spectre Interactif (pour Notebook)
    # ==============================================================================
    
    def _plot_peak_detection_notebook(self, file_path, prominence, window, xlim, ylim):
        """
        Analyse interactive de spectre dans le notebook :
        - Affiche le graphique Matplotlib (avec annotation des raies et export PNG)
        - Calcule et affiche le tableau d’analyse Pandas stylé (avec export CSV/LaTeX)
        - Prédit la classe spectrale via le modèle IA chargé (.pkl)
        - Log toutes les analyses dans la session (historique)
        - Permet d’exporter toutes les analyses de la session dans un rapport HTML
        """

        import io, base64
        from datetime import datetime
        # === 1. Paramètres d'affichage/couleurs pour chaque raie ===
        line_colors = {
            "Hα": "#00FF00",      # Vert fluo (lime)
            "Hβ": "#FFDD00",      # Jaune-orangé
            "CaII K": "#00A2FF",  # Bleu clair
            "CaII H": "#AA00FF",  # Violet
        }

        # === 2. Extraction du spectre et détection des raies ===
        preprocessor = SpectraPreprocessor()
        # On initialise le détecteur de pics avec les paramètres
        peak_detector = PeakDetector(prominence=prominence, window=window)
        # Chemin complet du fichier FITS
        full_path = os.path.join(self.paths["RAW_DATA_DIR"], file_path)

        try: # On ouvre le fichier FITS compressé
            with gzip.open(full_path, 'rb') as f_gz:
                with fits.open(f_gz, memmap=False) as hdul:
                    wavelength, flux, invvar = preprocessor.load_spectrum(hdul)
                    header = hdul[0].header
            # On convertit les données en tableaux NumPy pour traitement 
            wavelength, flux, invvar = (np.asarray(d, dtype=np.float64) for d in [wavelength, flux, invvar])
            flux_norm = preprocessor.normalize_spectrum(flux)
            # On affiche les informations de l'en-tête FITS
            peak_indices, properties = peak_detector.detect_peaks(wavelength, flux_norm)
            peak_wavelengths = wavelength[peak_indices]
            matched_lines = peak_detector.match_known_lines(peak_indices, peak_wavelengths, properties)

            # === 3. Extraction des features pour IA ===
            # (Indépendant de la visualisation, utilisé pour prédiction IA)
            from pipeline.feature_engineering import FeatureEngineer
            feature_engineer = FeatureEngineer()
            features_vector = feature_engineer.extract_features(matched_lines)

            # === 4. Prédiction automatique de la classe spectrale ===
            # Le modèle doit être chargé UNE SEULE FOIS dans le notebook (ex: spectral_classifier)
            predicted_label = None
            if 'spectral_classifier' in globals() and spectral_classifier is not None:
                try:
                    print("Features vector:", features_vector)
                    predicted_label = spectral_classifier.model.predict([features_vector])[0]
                except Exception as e:
                    print("Erreur de prédiction IA :", e)
                    predicted_label = f"Erreur prédiction: {e}"
            
            # === 5. Affichage graphique Matplotlib interactif ===
            matched_wavelengths, matched_flux = [], []
            plt.style.use('dark_background')
            plt.figure(figsize=(18, 7))
            plt.plot(wavelength, flux_norm, color='gray', alpha=0.7, label='Spectre Normalisé')

            # -- Ajout des pics détectés (petits triangles rouges) --
            if len(peak_indices) > 0:
                plt.scatter(peak_wavelengths, flux_norm[peak_indices], color='red', marker='v', s=40, label='Tous les Pics Détectés')
            # -- Ajout des pics associés aux raies connues, et annotations --
            for name, match_data in matched_lines.items():
                if match_data is not None:
                    wl, prom = match_data
                    matched_wavelengths.append(wl)
                    idx = np.abs(wavelength - wl).argmin()
                    matched_flux.append(flux_norm[idx])
                    y_val = flux_norm[idx]
                    # Affiche le nom de la raie au-dessus du pic
                    plt.text(wl, y_val + 0.05, name, color=line_colors.get(name, 'white'), fontsize=13, ha='center', weight='bold')
            if matched_wavelengths:
                plt.scatter(matched_wavelengths, matched_flux, s=150, facecolors='none', edgecolors='lime', linewidth=2, label='Pics Associés')
            # -- Lignes verticales pour toutes les raies cibles --
            for name, wl in peak_detector.target_lines.items():
                plt.axvline(x=wl, color=line_colors.get(name, 'dodgerblue'), linestyle='--', alpha=0.8, label=f'Raie {name}')
            plt.title(f"Analyse des Pics pour {os.path.basename(file_path)}", fontsize=16)
            plt.xlabel("Longueur d'onde (Å)")
            plt.ylabel("Flux Normalisé")
            plt.xlim(xlim); plt.ylim(ylim)
            plt.grid(True, linestyle=':')
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            plt.legend(by_label.values(), by_label.keys())
            # -- Export PNG du plot --
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
            buf.seek(0)
            b64 = base64.b64encode(buf.read()).decode()
            display(HTML(f'<a download="spectre.png" href="data:image/png;base64,{b64}">⬇️ Exporter le graphique (PNG)</a>'))
            plt.show()

            # === 6. Analyse quantitative des raies associées (tableau Pandas stylé) ===
            invvar[invvar <= 0] = 1e-12
            uncertainty = StdDevUncertainty(1 / np.sqrt(invvar))
            spectrum_obj = Spectrum(spectral_axis=wavelength*u.AA, flux=flux_norm*u.Unit("adu"), uncertainty=uncertainty)
            analysis_results = {}
            for name, match_data in matched_lines.items():
                if match_data is not None:
                    wl_detected, prom = match_data
                    analysis_region = SpectralRegion((wl_detected - window) * u.AA, (wl_detected + window) * u.AA)
                    try:
                        center = centroid(spectrum_obj, regions=analysis_region)
                        width = fwhm(spectrum_obj, regions=analysis_region)
                        analysis_results[name] = {
                            "Centroïde (Å)": f"{center.to_value(u.AA):.2f}",
                            "Largeur FWHM (Å)": f"{width.to_value(u.AA):.2f}",
                            "Prominence (Force)": f"{prom:.3f}"
                        }
                    except Exception:
                        analysis_results[name] = {"Erreur": "Analyse impossible"}

            # === 7. Affichage dynamique du tableau d'analyse & exports ===
            if analysis_results:
                display(Markdown("#### Analyse Quantitative des Raies Associées"))
                df_analysis = pd.DataFrame.from_dict(analysis_results, orient='index')
                # Style couleur par raie
                def highlight_row(row):
                    color = line_colors.get(row.name, None)
                    return ['background-color: %s; color: black' % color if color else '' for _ in row]
                display(df_analysis.style.apply(highlight_row, axis=1))
                # Export CSV & LaTeX
                csv_buffer = io.StringIO()
                df_analysis.to_csv(csv_buffer)
                b64_csv = base64.b64encode(csv_buffer.getvalue().encode()).decode()
                href_csv = f'<a download="analyse_raies.csv" href="data:text/csv;base64,{b64_csv}">⬇️ Exporter en CSV</a>'
                latex_str = df_analysis.to_latex(index=True, caption="Analyse Quantitative des Raies Associées")
                b64_latex = base64.b64encode(latex_str.encode()).decode()
                href_latex = f'<a download="analyse_raies.tex" href="data:text/plain;base64,{b64_latex}">⬇️ Exporter en LaTeX</a>'
                display(HTML(f"<div style='margin:8px 0'>{href_csv} &nbsp;|&nbsp; {href_latex}</div>"))

            # === 8. Affichage de la classe spectrale prédite (IA) ===
            if predicted_label is not None:
                display(Markdown(f"**Classe spectrale prédite par le modèle IA :** `{predicted_label}`"))

            # === 9. Log/Historique complet de la session (toutes analyses sauvegardées) ===
            # (Initialise si pas déjà présent)
            if 'log_analyses' not in globals():
                log_analyses = pd.DataFrame(columns=['Fichier', 'Paramètres', 'Tableau', 'Timestamp', 'Classe prédite'])
            # On log tous les paramètres et résultats
            params = {
                "spectre": file_path,
                "prominence": prominence,
                "window": window,
                "xlim": xlim,
                "ylim": ylim
            }
            new_log = {
                'Fichier': file_path,
                'Paramètres': params,
                'Tableau': df_analysis.to_dict() if analysis_results else {},
                'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'Classe prédite': predicted_label
            }
            try:
                log_analyses = pd.concat([log_analyses, pd.DataFrame([new_log])], ignore_index=True)
            except:
                # Si pb de scope dans certains kernels Jupyter
                import builtins
                if not hasattr(builtins, "log_analyses"):
                    builtins.log_analyses = pd.DataFrame(columns=['Fichier', 'Paramètres', 'Tableau', 'Timestamp', 'Classe prédite'])
                builtins.log_analyses = pd.concat([builtins.log_analyses, pd.DataFrame([new_log])], ignore_index=True)
                log_analyses = builtins.log_analyses

            if len(log_analyses) > 0:
                display(Markdown("#### Historique des Analyses de la Session"))
                display(log_analyses[['Fichier', 'Classe prédite', 'Paramètres', 'Timestamp']])

            # === 10. Génération du rapport HTML de la session (plots non inclus, mais facile à ajouter) ===
            from ipywidgets import Button
            def generate_report():
                html = "<h1>Rapport d'Analyse Spectroscopique</h1>"
                for i, row in log_analyses.iterrows():
                    html += f"<h2>Analyse #{i+1} — {row['Fichier']}</h2>"
                    if 'Classe prédite' in row and row['Classe prédite'] is not None:
                        html += f"<p><b>Classe spectrale prédite :</b> {row['Classe prédite']}</p>"
                    html += f"<pre>{row['Paramètres']}</pre>"
                    try:
                        df_html = pd.DataFrame.from_dict(row['Tableau']).to_html()
                        html += df_html
                    except Exception:
                        html += "<p><i>Erreur lors de la conversion du tableau.</i></p>"
                    html += f"<p><i>Timestamp : {row['Timestamp']}</i></p>"
                    html += "<hr>"
                with io.BytesIO() as f:
                    f.write(html.encode())
                    f.seek(0)
                    b64_report = base64.b64encode(f.read()).decode()
                download_link = f'<a download="rapport_astro.html" href="data:text/html;base64,{b64_report}">⬇️ Télécharger le rapport HTML (toutes analyses)</a>'
                display(HTML(download_link))
            btn = Button(description="Générer le rapport HTML de la session", button_style="success")
            btn.on_click(lambda b: generate_report())
            display(btn)

        except Exception as e:
            print(f"Erreur lors du traitement du spectre : {e}")

    def interactive_peak_tuner(self):
        """Widget interactif complet pour le tuning des paramètres dans le notebook."""
        if not self.available_spectra:
            print("Aucun spectre trouvé.")
            return

        interactive_plot = widgets.interactive(
            self._plot_peak_detection_notebook,
            file_path=widgets.Dropdown(options=self.available_spectra, description="Spectre :", layout={'width': 'max-content'}),
            prominence=widgets.FloatSlider(min=0.01, max=1.0, step=0.01, value=0.2, description='Prominence:'),
            window=widgets.IntSlider(min=1, max=50, step=1, value=15, description='Fenêtre (Å):'),
            xlim=widgets.FloatRangeSlider(value=[3800, 7000], min=3500, max=9500, step=1, description='Zoom X (Å):', continuous_update=False, layout={'width': '500px'}),
            ylim=widgets.FloatRangeSlider(value=[-1, 2.5], min=-2, max=5, step=0.1, description='Zoom Y (Flux):', continuous_update=False, layout={'width': '500px'})
        )
        display(interactive_plot)

    # ==============================================================================
    # Outil 4 : Analyse des Zéros dans les Features ---
    # ==============================================================================
    
    def analyze_feature_zeros(self, threshold=90):
        """
        Charge le dernier dataset de features, analyse le pourcentage de zéros
        par colonne, et affiche un graphique.
        """
        print("--- Analyse des Valeurs Nulles (Zéros) dans les Features ---")
        processed_dir = self.paths.get("PROCESSED_DIR", "../data/processed/")
        
        list_of_files = glob.glob(os.path.join(processed_dir, 'features_*.csv'))
        if not list_of_files:
            print("  > Aucun fichier de features trouvé. Veuillez d'abord lancer le pipeline de traitement.")
            return

        latest_file = max(list_of_files, key=os.path.getctime)
        print(f"  > Analyse du fichier : {os.path.basename(latest_file)}")
        df = pd.read_csv(latest_file)
        
        feature_cols = [col for col in df.columns if col.startswith("feature_")]
        if not feature_cols:
            print("  > Aucune colonne de feature trouvée dans le fichier.")
            return

        zero_stats = {col: 100 * (df[col] == 0).sum() / len(df) for col in feature_cols}
        zero_df = pd.DataFrame.from_dict(zero_stats, orient="index", columns=["zero_percentage"]).sort_values("zero_percentage", ascending=False)

        plt.style.use('dark_background')
        plt.figure(figsize=(12, 6))
        zero_df["zero_percentage"].plot(kind="bar", color="steelblue", edgecolor="black")
        plt.axhline(threshold, color="red", linestyle="--", label=f"Seuil critique : {threshold}%")
        plt.ylabel("Pourcentage de valeurs nulles (zéros)")
        plt.title("Distribution des Zéros par Feature")
        plt.legend()
        plt.tight_layout()
        plt.show()

        to_remove = zero_df[zero_df["zero_percentage"] >= threshold].index.tolist()
        if to_remove:
            print(f"\n  > Features à investiguer ou retirer (>= {threshold}% de zéros) :")
            for f in to_remove:
                print(f"    - {f} ({zero_df.loc[f, 'zero_percentage']:.1f}%)")

    # ==============================================================================
    # Outil 5 : Carte de Couverture Céleste ---
    # ==============================================================================
    
    def _generate_position_catalog(self, force_regenerate=False):
        """
        Scanne les FITS, extrait les coordonnées et crée un catalogue de position des plans.
        (Méthode privée appelée par plot_sky_coverage)
        """
        catalog_path = os.path.join(self.paths["CATALOG_DIR"], "spectra_position_catalog.csv")
        
        if os.path.exists(catalog_path) and not force_regenerate:
            print("  > Catalogue de position existant chargé.")
            return pd.read_csv(catalog_path)

        print("  > Scan des fichiers FITS pour générer le catalogue de position...")
        # On utilise la liste déjà scannée pour éviter de le refaire
        if not self.available_spectra:
            print("  > Aucun fichier FITS trouvé.")
            return pd.DataFrame()

        spectra_data = []
        for file_rel_path in tqdm(self.available_spectra, desc="Lecture des headers"):
            file_path = os.path.join(self.paths["RAW_DATA_DIR"], file_rel_path)
            try:
                with gzip.open(file_path, 'rb') as f_gz:
                    with fits.open(f_gz, memmap=False) as hdul:
                        header = hdul[0].header
                        if 'RA' in header and 'DEC' in header and 'PLANID' in header:
                            spectra_data.append({
                                'plan_id': header['PLANID'], 'ra': header['RA'], 'dec': header['DEC']
                            })
            except Exception:
                pass
        
        df_positions = pd.DataFrame(spectra_data)
        if df_positions.empty:
            print("  > Aucune coordonnée n'a pu être extraite.")
            return df_positions
            
        df_plan_positions = df_positions.groupby('plan_id').agg({'ra': 'mean', 'dec': 'mean'}).reset_index()
        df_plan_positions.to_csv(catalog_path, index=False)
        print(f"\n  > Catalogue de position créé avec {len(df_plan_positions)} plans uniques.")
        return df_plan_positions

    def plot_sky_coverage(self):
        """
        Génère une carte de couverture céleste avec Matplotlib, en contrôlant
        l'ordre des couches pour un affichage correct.
  # NOTE : L'affichage d'une image de fond avec `imshow` ou `figimage` sur une projection
        # Mollweide s'est avéré instable et dépendant du backend de rendu.
        # Pour garantir la robustesse, nous affichons la carte sans image de fond.
        # Une future amélioration pourrait utiliser une librairie de plotting astronomique
        # spécialisée comme `aplpy` si cette fonctionnalité est requise.
        """
        print("--- Carte de Couverture Céleste (Matplotlib) ---")
        df_plan_positions = self._generate_position_catalog()
        if df_plan_positions.empty: return

        from PIL import Image

        image_path = os.path.join(self.paths["PROJECT_ROOT"], "static", "images", "milky_way_mollweide.jpeg")
        try:
            bg_image = Image.open(image_path)
        except FileNotFoundError:
            bg_image = None
            
        spectra_counts = [len([f for f in os.listdir(os.path.join(self.paths["RAW_DATA_DIR"], pid)) if f.endswith('.fits.gz')]) 
                          for pid in df_plan_positions['plan_id']]
        df_plan_positions['spectra_count'] = spectra_counts

        # --- Création du graphique Matplotlib ---
        plt.style.use('dark_background')
        fig = plt.figure(figsize=(20, 10))
        ax = fig.add_subplot(111, projection="mollweide")
        
        # --- ORDRE DE DESSIN CONTRÔLÉ AVEC ZORDER ---
        
        # 1. Image de fond (couche 0, la plus basse)
        if bg_image is not None:
            ax.imshow(bg_image, extent=[-np.pi, np.pi, -np.pi/2, np.pi/2], aspect='auto', zorder=0)

        # 2. Grille (couche 5, par-dessus l'image)
        ax.grid(True, linestyle=':', alpha=0.5, color='white', zorder=5)

        # 3. Points de données (couche 10, la plus haute)
        ra_rad = np.deg2rad(df_plan_positions.ra)
        ra_rad[ra_rad > np.pi] -= 2 * np.pi
        dec_rad = np.deg2rad(df_plan_positions.dec)
        sizes = df_plan_positions['spectra_count'] / 5 + 30
        
        edge_color = (1.0, 1.0, 1.0, 0.7)
        scatter = ax.scatter(ra_rad, dec_rad, s=sizes, c=df_plan_positions['spectra_count'], 
                             cmap='autumn', alpha=1.0, edgecolors=edge_color, linewidth=1, zorder=10)
        
        # --- MISE EN FORME ---
        cbar = fig.colorbar(scatter, shrink=0.6, aspect=12, pad=0.08)
        cbar.set_label('Nombre de Spectres par Plan', fontsize=12)
        ax.set_xticklabels(['14h', '16h', '18h', '20h', '22h', '0h', '2h', '4h', '6h', '8h', '10h'], color='white')
        ax.set_title("Couverture du Ciel - Densité des Données Acquises", pad=20, fontsize=18)
        
        # On s'assure que le fond des axes est transparent
        ax.set_facecolor('none')
        fig.patch.set_facecolor('#0E1117')

        # On sauvegarde et on affiche l'image pour un rendu fiable
        temp_image_path = os.path.join(self.paths["PROJECT_ROOT"], "temp_sky_map.png")
        plt.savefig(temp_image_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close(fig)

        from IPython.display import Image
        display(Image(filename=temp_image_path))
        
    # ==============================================================================
    # Outil 6 : Inspecteur de Modèles Entraînés ---
    # ==============================================================================
    
    def _analyze_saved_model(self, model_path):
        """
        Charge un modèle .pkl, affiche ses paramètres et l'importance des features.
        """
        if not os.path.exists(model_path):
            display(Markdown(f"### Fichier modèle non trouvé\n`{model_path}`"))
            return

        try:
            from pipeline.classifier import SpectralClassifier
            spectral_classifier = SpectralClassifier.load_model(model_path)
            model = spectral_classifier.model
            
            md_output = f"### Analyse du Modèle : `{os.path.basename(model_path)}`\n---\n"
            md_output += "#### Hyperparamètres du Modèle\n"
            params = model.get_params()
            for key, value in params.items():
                md_output += f"- **{key} :** `{value}`\n"
            
            display(Markdown(md_output))

            if hasattr(model, 'feature_importances_'):
                # On utilise le chemin PROCESSED_DIR défini dans notre dictionnaire 'paths'
                processed_dir = self.paths.get("PROCESSED_DIR")
                if not processed_dir:
                    raise FileNotFoundError("Chemin 'PROCESSED_DIR' non trouvé dans la configuration.")
                
                list_of_feature_files = glob.glob(os.path.join(processed_dir, 'features_*.csv'))
                
                if not list_of_feature_files:
                    # On affiche une erreur claire si aucun fichier n'est trouvé
                    display(Markdown("#### Importance des Features\n*Avertissement : Aucun fichier de features trouvé pour récupérer les noms de colonnes.*"))
                    return # On arrête ici, on ne peut pas faire le graphique

                latest_features_file = max(list_of_feature_files, key=os.path.getctime)
                df_features = pd.read_csv(latest_features_file)
                feature_names = [col for col in df_features.columns if col.startswith('feature_')]

                # S'assurer que le nombre de features correspond
                if len(feature_names) == len(model.feature_importances_):
                    importances = model.feature_importances_
                    indices = np.argsort(importances)[::-1]

                    plt.style.use('dark_background')
                    plt.figure(figsize=(12, 6))
                    plt.title("Importance des Features pour le Modèle", fontsize=16)
                    plt.bar(range(len(importances)), importances[indices], color="c", align="center")
                    plt.xticks(range(len(importances)), np.array(feature_names)[indices], rotation=45, ha="right")
                    plt.ylabel("Importance (Gini)")
                    plt.tight_layout()
                    plt.show()
                else:
                    display(Markdown(f"#### Importance des Features\n*Avertissement : Le nombre de features du modèle ({len(model.feature_importances_)}) ne correspond pas à celui du fichier de features ({len(feature_names)}).*"))

        except Exception as e:
            display(Markdown(f"### Erreur lors du chargement ou de l'analyse du modèle\n`{os.path.basename(model_path)}`\n\n**Détail :** {e}"))

    def interactive_model_inspector(self):
        """
        Crée un widget interactif pour sélectionner et inspecter un modèle sauvegardé.
        """
        models_dir = self.paths.get("MODELS_DIR", "../models/")
        if not os.path.exists(models_dir):
            print(f"Le dossier des modèles '{models_dir}' n'existe pas.")
            return
            
        saved_models = [os.path.join(models_dir, f) for f in os.listdir(models_dir) if f.endswith(".pkl")]
        
        if not saved_models:
            print("Aucun modèle sauvegardé (.pkl) trouvé.")
            return
        
        interact(self._analyze_saved_model, model_path=saved_models)
        
    # ==============================================================================
    # Outil 7: Comparaison de Spectres ---
    # ==============================================================================
    
    def _plot_spectra_comparison(self, file_paths, normalize=True, offset=0.0):
        """Affiche plusieurs spectres superposés sur un même graphique."""
        preprocessor = SpectraPreprocessor()
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(18, 9))
        
        current_offset = 0
        for i, file_path in enumerate(file_paths):
            full_path = os.path.join(self.paths["RAW_DATA_DIR"], file_path)
            try:
                with gzip.open(full_path, 'rb') as f_gz:
                    with fits.open(f_gz, memmap=False) as hdul:
                        wavelength, flux, _ = preprocessor.load_spectrum(hdul)
                
                if normalize:
                    flux = preprocessor.normalize_spectrum(flux)
                
                ax.plot(wavelength, flux + current_offset, label=os.path.basename(file_path))
                current_offset += offset
            except Exception as e:
                print(f"Impossible de charger {file_path}: {e}")
        
        ax.set_title("Comparaison de Spectres", fontsize=16)
        ax.set_xlabel("Longueur d'onde (Å)")
        ax.set_ylabel("Flux (normalisé + décalage)")
        ax.grid(True, linestyle=':')
        ax.legend()
        plt.show()

    def interactive_spectra_comparator(self):
        """Crée un widget interactif pour comparer plusieurs spectres."""
        if not self.available_spectra:
            print("Aucun spectre trouvé.")
            return

        interact(
            self._plot_spectra_comparison,
            file_paths=widgets.SelectMultiple(
                options=self.available_spectra,
                description='Spectres:',
                rows=10 # Hauteur de la liste
            ),
            normalize=widgets.Checkbox(value=True, description="Normaliser les spectres"),
            offset=widgets.FloatSlider(min=0.0, max=5.0, step=0.1, value=0.5, description="Décalage Y:")
        )

    # ==============================================================================
    # Outil 8 : Analyseur de Spectre Augmenté (Version Plotly) ---
    # ==============================================================================
    
    def _display_formatted_header_for_streamlit(self, st, fits_relative_path):
        """Affiche les informations de l'en-tête FITS dans Streamlit."""
        full_path = os.path.join(self.paths["RAW_DATA_DIR"], fits_relative_path)
        try:
            with gzip.open(full_path, 'rb') as f_gz:
                with fits.open(f_gz, memmap=False) as hdul:
                    header = hdul[0].header
            
            md_output = f"### En-tête du fichier : `{os.path.basename(full_path)}`\n---\n"
            sections = {}
            current_section = "Informations Générales"
            sections[current_section] = ""
            for key, value in header.items():
                if key == 'COMMENT' and '--------' in str(value):
                    current_section = str(value).replace('COMMENT', '', 1).replace('-', '').strip()
                    if current_section not in sections: sections[current_section] = ""
                elif key and key not in ['COMMENT', 'HISTORY', '']:
                    sections[current_section] += self._format_header_line(header, key, key)
            for title, content in sections.items():
                if content: md_output += f"\n#### {title}\n{content}"
            st.markdown(md_output, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Erreur lors de l'ouverture du header : {e}")
    
    def _plot_spectrum_analysis_plotly(self, st, file_path_or_buffer, prominence, window):
        preprocessor = SpectraPreprocessor()
        peak_detector = PeakDetector(prominence=prominence, window=window)
        
        try:
            with gzip.open(file_path_or_buffer, 'rb') as f_gz:
                with fits.open(f_gz, memmap=False) as hdul:
                    wavelength, flux, invvar = preprocessor.load_spectrum(hdul)
                    header = hdul[0].header
            
            wavelength, flux, invvar = (np.asarray(d, dtype=np.float64) for d in [wavelength, flux, invvar])
            flux_norm = preprocessor.normalize_spectrum(flux)
            
            peak_indices, properties = peak_detector.detect_peaks(wavelength, flux_norm)
            peak_wavelengths = wavelength[peak_indices]
            matched_lines = peak_detector.match_known_lines(peak_indices, peak_wavelengths, properties)
            
            # --- Récupération des pics associés pour les cercles verts ---
            matched_wavelengths = [data[0] for data in matched_lines.values() if data is not None]
            
            # --- Création du Graphique Interactif avec Plotly ---
            fig = go.Figure()

            # 1. Ajouter le spectre
            fig.add_trace(go.Scatter(
                x=wavelength, y=flux_norm, mode='lines', name='Spectre Normalisé',
                line=dict(color='rgba(200, 200, 200, 0.7)', width=1)
            ))

            # 2. Ajouter les pics détectés
            if len(peak_indices) > 0:
                fig.add_trace(go.Scatter(
                    x=peak_wavelengths, y=flux_norm[peak_indices], mode='markers', name='Pics Détectés',
                    marker=dict(color='red', symbol='triangle-down', size=8)
                ))
            
            # <<< AJOUT DES CERCLES VERTS >>>
            if matched_wavelengths:
                matched_indices = [np.abs(wavelength - wl).argmin() for wl in matched_wavelengths]
                fig.add_trace(go.Scatter(x=wavelength[matched_indices], y=flux_norm[matched_indices], mode='markers', name='Pics Associés', 
                                         marker=dict(symbol='circle-open', color='lime', size=15, line=dict(width=2))))
            
            # 3. Préparer les formes pour les raies cibles (lignes verticales)
            shapes = []
            # On sépare les raies de Balmer des autres pour les boutons
            balmer_lines = {k: v for k, v in peak_detector.target_lines.items() if 'H' in k}
            other_lines = {k: v for k, v in peak_detector.target_lines.items() if 'H' not in k}

            for name, wl in peak_detector.target_lines.items():
                shapes.append(dict(
                    type="line", xref="x", yref="paper",
                    x0=wl, y0=0, x1=wl, y1=1,
                    line=dict(color="rgba(0, 150, 255, 0.5)", width=1, dash="dash"),
                    name=name # On stocke le nom pour le filtrage
                ))

            # --- Mise en forme du graphique ---
            subclass = header.get('SUBCLASS', 'N/A')
            title = f"Analyse du Spectre : {header.get('DESIG', 'Inconnu')} (Type: {subclass})"
            fig.update_layout(
                title=title,
                xaxis_title="Longueur d'onde (Å)",
                yaxis_title="Flux Normalisé",
                template="plotly_dark", # Thème sombre
                legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
                shapes=shapes # On ajoute les lignes verticales
            )

            # --- Ajout des boutons interactifs ---
            fig.update_layout(
                updatemenus=[
                    dict(
                        type="buttons",
                        direction="right",
                        x=0.5, y=1.15, xanchor="center", yanchor="top",
                        buttons=list([
                            dict(label="Toutes les Raies",
                                 method="relayout",
                                 args=["shapes", shapes]),
                            dict(label="Raies de Balmer",
                                 method="relayout",
                                 args=["shapes", [s for s in shapes if 'H' in s['name']]]),
                            dict(label="Raies de Calcium",
                                 method="relayout",
                                 args=["shapes", [s for s in shapes if 'Ca' in s['name']]]),
                            dict(label="Aucune Raie",
                                 method="relayout",
                                 args=["shapes", []]),
                        ])
                    )
                ]
            )
            
            st.plotly_chart(fig, use_container_width=True)
            return header # On retourne le header pour l'afficher

        except Exception as e:
            st.error(f"Erreur lors de l'analyse du spectre : {e}")
            return None
    
    def app(self):
        """La méthode principale pour lancer l'application Streamlit."""
        import streamlit as st
        
        st.set_page_config(page_title="AstroSpectro Visualizer", layout="wide")
        st.title("AstroSpectro - Tableau de Bord d'Analyse")

        # --- Barre Latérale de Contrôle ---
        st.sidebar.header("Source des Données")
        
        # --- NOUVELLE SECTION : CHOIX DE LA SOURCE ---
        source_option = st.sidebar.radio(
            "Choisir une source de spectre",
            ('Sélectionner un spectre du projet', 'Téléverser un fichier FITS')
        )

        file_to_process = None
        
        if source_option == 'Sélectionner un spectre du projet':
            selected_file_path = st.sidebar.selectbox("Sélectionner un Spectre", self.available_spectra)
            if selected_file_path:
                file_to_process = os.path.join(self.paths["RAW_DATA_DIR"], selected_file_path)
        else:
            uploaded_file = st.sidebar.file_uploader("Charger un spectre (.fits.gz)", type=["gz"])
            if uploaded_file is not None:
                file_to_process = uploaded_file
        # --- FIN DE LA NOUVELLE SECTION ---

        st.sidebar.markdown("---")
        st.sidebar.header("Paramètres de Détection")
        prominence = st.sidebar.slider("Prominence", 0.01, 1.0, 0.2, 0.01)
        window = st.sidebar.slider("Fenêtre (Å)", 1, 50, 15, 1)

        # --- Affichage Principal ---
        if file_to_process:
            header = self._plot_spectrum_analysis_plotly(st, file_to_process, prominence, window)
            
            if header:
                st.markdown("---")
                with st.expander("Afficher l'En-tête FITS Complet"):
                    # On va créer une petite fonction pour afficher le header dans Streamlit
                    self._display_header_for_streamlit(st, header)
    
    # On a besoin d'une fonction qui prend un header, pas un chemin de fichier
    def _display_header_for_streamlit(self, st, header):
        """Affiche un objet header FITS dans Streamlit."""
        md_output = "#### Métadonnées Principales\n"
        # Affiche quelques clés importantes
        md_output += f"- **Objet :** `{header.get('OBJECT', header.get('DESIG', 'N/A'))}`\n"
        md_output += f"- **Type :** `{header.get('SUBCLASS', 'N/A')}`\n"
        md_output += f"- **Date Obs :** `{header.get('DATE-OBS', 'N/A')}`\n"
        md_output += f"- **RA / Dec :** `{header.get('RA', 'N/A')}` / `{header.get('DEC', 'N/A')}`\n"
        st.markdown(md_output)
        # Affiche le header complet sous forme de dictionnaire
        st.json(dict(header))
        
        