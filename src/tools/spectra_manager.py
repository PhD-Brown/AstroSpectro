import sys
import os
import subprocess
import ipywidgets as widgets
from IPython.display import display, clear_output, HTML
from datetime import datetime, timezone
import re
import shutil

class SpectraManager:
    """
    Fournit des interfaces interactives pour gérer le dataset de spectres
    (téléchargement, nettoyage) directement depuis un notebook Jupyter.
    """
    def __init__(self, paths):
        self.paths = paths
        
    def interactive_downloader(self):
        """Affiche un widget pour lancer le script de téléchargement DR5."""
        
        limit_widget = widgets.IntText(value=5, description='Plans:')
        max_spectra_widget = widgets.IntText(value=500, description='Spectres:')
        run_button = widgets.Button(description="Lancer le téléchargement", button_style='success', icon='download')
        progress_display = widgets.HTML(value="")
        log_output = widgets.Output()

        display(widgets.HBox([limit_widget, max_spectra_widget]), run_button, progress_display, log_output)

        def on_run_clicked(b):
            progress_display.value = ""
            with log_output:
                clear_output()
            
            limit = str(limit_widget.value)
            max_spectra = str(max_spectra_widget.value)
            script_path = os.path.join(self.paths["TOOLS_DIR"], "dr5_downloader.py")
            cmd = [sys.executable, script_path, "--limit", limit, "--max-spectres", max_spectra]
            
            progress_display.value = f"<b>Lancement du téléchargement...</b>"
            
            output_lines = []
            try:
                process = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    bufsize=1, universal_newlines=True, encoding='utf-8', errors='replace'
                )
                if process.stdout:
                    for line in iter(process.stdout.readline, ''):
                        output_lines.append(line)
                        if 'Téléchargement:' in line and '%' in line:
                            clean_line = line.strip()
                            match = re.search(r'(\d+)%\|', clean_line)
                            percent = int(match.group(1)) if match else 0
                            progress_bar_html = f"""
                            <div style="font-family: monospace; white-space: pre;">{clean_line}</div>
                            <progress value="{percent}" max="100" style="width: 100%; height: 20px;"></progress>
                            """
                            progress_display.value = progress_bar_html
                        else:
                            with log_output:
                                print(line, end='')
                
                process.wait()
                full_output = "".join(output_lines)

                timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                log_dir = self.paths["LOGS_DIR"] # Utiliser le chemin depuis self.paths
                os.makedirs(log_dir, exist_ok=True)

                if process.returncode == 0:
                    progress_display.value = "<b>Téléchargement terminé avec succès.</b>"
                    log_filename = f"download_log_{timestamp}.txt"
                    log_path = os.path.join(log_dir, log_filename)
                    with open(log_path, "w", encoding="utf-8") as f: f.write(full_output)
                    with log_output: print(f"\nLog de succès sauvegardé dans : {log_path}")
                else:
                    raise subprocess.CalledProcessError(process.returncode, cmd, output=full_output)
            except Exception as e:
                progress_display.value = f"<b>Une erreur est survenue : {e}</b>"

        run_button.on_click(on_run_clicked)

    def interactive_cleaner(self):
        """Affiche un widget sécurisé pour nettoyer le répertoire data/raw."""
        
        confirm_text = widgets.Text(placeholder="Écrire 'CONFIRMER' ici")
        clean_button = widgets.Button(description="Nettoyer le dossier 'raw'", button_style='danger', icon='trash')
        clean_output = widgets.Output()

        def on_clean_clicked(b):
            with clean_output:
                clear_output()
                if confirm_text.value == "CONFIRMER":
                    raw_dir = self.paths["RAW_DATA_DIR"]
                    backup_dir = os.path.join(self.paths["DATA_DIR"], "raw_backup")
                    print(f"Création d'une sauvegarde dans '{backup_dir}'...")
                    try:
                        if os.path.exists(raw_dir):
                            shutil.copytree(raw_dir, backup_dir, dirs_exist_ok=True)
                        print("Nettoyage du dossier 'raw'...")
                        shutil.rmtree(raw_dir)
                        os.makedirs(raw_dir)
                        print("Dossier 'raw' nettoyé et sauvegardé avec succès.")
                    except Exception as e:
                        print(f"Erreur lors du nettoyage : {e}")
                else:
                    print("Confirmation incorrecte. Nettoyage annulé.")

        clean_button.on_click(on_clean_clicked)
        display(widgets.VBox([widgets.Label("ATTENTION : Action destructive !"), confirm_text, clean_button, clean_output]))