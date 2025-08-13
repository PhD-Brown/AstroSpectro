"""
AstroSpectro — Gestion des spectres (outils Notebook/CLI)

Ce module fournit la classe `SpectraManager`, un utilitaire haut niveau
pour piloter les opérations “autour” des données spectrales depuis un
notebook (ou un script) :

- orchestration du **téléchargement** de nouveaux spectres DR5 via le script
  `src/tools/dr5_downloader.py` (avec barre de progression et **log unique**),
- petites actions “ops” : **nettoyage/backup** du répertoire `data/raw/`,
- raccourcis d’exploration (par ex. chemin du dernier log, lecture des
  dernières lignes, etc. — si présents dans ce fichier).

Conventions
-----------
- Les chemins de base (RAW_DATA_DIR, PROCESSED_DIR, CATALOG_DIR, MODELS_DIR,
  LOGS_DIR, TOOLS_DIR, …) sont fournis par `utils.setup_project_env()`.
- Le **téléchargeur DR5** est lancé en sous-processus et accepte `--limit`,
  `--max-spectres`, `--log-to` et `--append` et `--progress` pour forcer
  l’affichage de la barre dans un notebook. Quand `--log-to` est fourni,
  **un seul fichier log** est produit sous `logs/` (plus de “double log”).
  Et est donc écrit **par le script** dans un seul fichier.  # cf. dr5_downloader.py
- Les widgets Jupyter (ipywidgets) sont utilisés pour l’UX Notebook, tout en
  restant compatibles avec une exécution pure CLI.

Entrées / Sorties principales
-----------------------------
Entrées :
- `paths` : dict des répertoires clés du projet, typiquement issu de
  `utils.setup_project_env()`.

Sorties / Effets :
- fichiers `.fits.gz` téléchargés dans `data/raw/<PLAN>/`,
- logs textuels par session dans `logs/download_log_YYYYMMDDTHHMMSSZ.txt`,
- en option : un backup `data/raw_backup/` avant nettoyage de `data/raw/`.

API publique (principales méthodes)
-----------------------------------
- `SpectraManager(paths)` : instanciation avec l’environnement projet.
- `interactive_downloader()` : widget de lancement de téléchargement DR5
  (barre de progression, log unique dans `logs/`).
- `interactive_cleaner()` : widget pour vider `data/raw/` avec backup et
  confirmation (si cette méthode est présente dans le fichier).

Exemple minimal
---------------
>>> from utils import setup_project_env
>>> from tools.spectra_manager import SpectraManager
>>> paths = setup_project_env()
>>> manager = SpectraManager(paths)
>>> # Dans un notebook :
>>> manager.interactive_downloader()
"""

from __future__ import annotations

import os
import re
import shlex
import sys
import shutil
import subprocess
from datetime import datetime, timezone

import ipywidgets as widgets
from IPython.display import display, clear_output, Markdown


class SpectraManager:
    """
    Gestionnaire d’opérations autour des spectres (Notebook/CLI).

    Cette classe sert de “façade” pour :
      1) lancer des **téléchargements DR5** de nouveaux spectres depuis un
         notebook, avec une **barre de progression** et un **log unique**,
      2) exécuter quelques opérations de maintenance (ex. **nettoyer** `data/raw/`
         après **backup**), si la méthode correspondante est disponible,
      3) fournir quelques utilitaires de lecture/accès aux **logs** (selon
         ce qui est implémenté dans ce fichier).

    Attributes:
        paths (dict[str, str]): Dictionnaire des chemins racine du projet.
            Doit contenir a minima :
            - 'RAW_DATA_DIR'   -> dossier des spectres bruts (`data/raw/`)
            - 'LOGS_DIR'       -> dossier des logs (`logs/`)
            - 'TOOLS_DIR'      -> dossier scripts outils (`src/tools/`)
            Et, selon le projet : 'PROCESSED_DIR', 'CATALOG_DIR', 'MODELS_DIR', etc.

    Args:
        paths: Dictionnaire des répertoires du projet. Doit contenir au moins
               'RAW_DATA_DIR', 'LOGS_DIR', 'TOOLS_DIR'.

    Raises:
        KeyError: si une clé obligatoire manque.
    """

    REQUIRED_KEYS = ("RAW_DATA_DIR", "LOGS_DIR", "TOOLS_DIR")

    # Regex pré-compilée pour capter "XX%" dans la ligne de progression
    _RE_PCT = re.compile(r"(\d+)%")

    def __init__(self, paths: dict) -> None:
        """
        Initialise le gestionnaire avec les chemins de projet.

        Args:
            paths: Dictionnaire de chemins. Typiquement la valeur retournée par
                `utils.setup_project_env()`. Les clés utilisées ici incluent :
                'RAW_DATA_DIR', 'LOGS_DIR', 'TOOLS_DIR'.

        Raises:
            KeyError: si une clé requise manque dans `paths`.
        """
        for k in self.REQUIRED_KEYS:
            if k not in paths:
                raise KeyError(f"Clé manquante dans paths: '{k}'")
        self.paths = paths

    # -------------------------------------------------------------------------
    # 1) Téléchargement interactif
    # -------------------------------------------------------------------------

    def interactive_downloader(self) -> None:
        """
        Lance un téléchargement DR5 depuis un notebook (widget + log unique).

        Affiche deux champs numériques (nombre de plans à traiter, nombre maximal
        de spectres à télécharger), un bouton de lancement, une barre de progression
        et un aperçu “live” de la sortie. La session écrit **un seul fichier log**
        dans `logs/` via le paramètre `--log-to` transmis au script
        `src/tools/dr5_downloader.py`.

        Comportement:
            - Construit et lance la commande:
            `python src/tools/dr5_downloader.py --limit <N> --max-spectres <M>
            --log-to <logs/...txt> --append`
            - Streame le stdout du sous-processus pour **mettre à jour la progression**.
            Les lignes de type: `Téléchargement: 60% | 298/500 [...]` sont détectées.
            - À la fin, affiche le **chemin du log** et les **dernières lignes** du fichier.
            - Ne crée plus de “double log” côté widget: c’est le script qui écrit
            dans le fichier passé via `--log-to`.

        Args:
            None

        Returns:
            None. Effets de bord:
            - fichiers .fits.gz téléchargés dans `data/raw/<PLAN>/`
            - log unique `logs/download_log_YYYYMMDDTHHMMSSZ.txt`

        Raises:
            RuntimeError: si le sous-processus retourne un code non nul.
            OSError: s’il est impossible de créer le dossier de logs.

        UI:
            - `Plans` (IntText): nombre de plans à traiter (`--limit`)
            - `Spectres` (IntText): plafond de spectres (`--max-spectres`)
            - `Afficher la barre (tqdm)` (Checkbox): force `--progress`
            - `Mode rapide` (Checkbox): `--delay 0 --chunk 131072`
            - Bouton “Lancer le téléchargement…”
            - Zone de progression + sortie texte “live”

        Effets:
            - appelle `src/tools/dr5_downloader.py` en sous-processus
              avec `--log-to logs/download_log_YYYYMMDDTHHMMSSZ.txt --append`,
              donc **un seul log** pour la session.
        """

        # Widgets
        limit_widget = widgets.IntText(value=5, description="Plans:")
        max_spectra_widget = widgets.IntText(value=50, description="Spectres:")
        show_bar = widgets.Checkbox(value=True, description="Afficher la barre (tqdm)")
        fast_mode = widgets.Checkbox(value=True, description="Mode rapide")
        run_button = widgets.Button(
            description="Lancer le téléchargement…",
            button_style="success",
            icon="download",
        )

        progress_display = widgets.HTML(value="")
        log_output = widgets.Output()

        line1 = widgets.HBox([limit_widget, max_spectra_widget, show_bar, fast_mode])
        display(Markdown("### 1. Téléchargement des Spectres"))
        display(
            Markdown(
                "Utilisez l’interface ci-dessous pour lancer un nouveau téléchargement. "
                "Les logs de progression s’afficheront en temps réel."
            )
        )
        display(line1, run_button, progress_display, log_output)

        # ------- callback principal -------------------------------------------------
        def on_run_clicked(_):
            progress_display.value = ""
            with log_output:
                clear_output()

            # Préparation des chemins/arguments
            script_path = os.path.join(self.paths["TOOLS_DIR"], "dr5_downloader.py")
            os.makedirs(self.paths["LOGS_DIR"], exist_ok=True)
            ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            log_path = os.path.join(self.paths["LOGS_DIR"], f"download_log_{ts}.txt")

            # Construction de la commande CLI
            cmd = [
                sys.executable,
                script_path,
                "--limit",
                str(limit_widget.value),
                "--max-spectres",
                str(max_spectra_widget.value),
                "--log-to",
                log_path,
                "--append",
            ]
            if show_bar.value:
                cmd.append("--progress")
            if fast_mode.value:
                # boost: délais courts et gros chunks (toujours sûrs, fallback dans le script)
                cmd += ["--delay", "0", "--chunk", "131072"]  # 128 KB

            # Affichage d’un message + désactivation du bouton (évite double clic)
            progress_display.value = "<b>Lancement du téléchargement...</b>"
            run_button.disabled = True

            try:
                # Journalisation de la commande (debug lisible)
                print("Commande:", shlex.join(cmd))

                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    bufsize=1,
                    universal_newlines=True,
                    encoding="utf-8",
                    errors="replace",
                )

                if proc.stdout:
                    for line in iter(proc.stdout.readline, ""):
                        # On intercepte les lignes tqdm pour la barre HTML locale :
                        if "Téléchargement" in line and "%" in line:
                            clean = line.strip()
                            m = self._RE_PCT.search(clean)
                            pct = int(m.group(1)) if m else 0
                            progress_display.value = (
                                "<div style='font-family:monospace;white-space:pre'>"
                                f"{clean}</div>"
                                f"<progress value='{pct}' max='100' "
                                "style='width:100%;height:18px'></progress>"
                            )
                        else:
                            # Le reste part dans la zone de logs
                            with log_output:
                                print(line, end="")

                rc = proc.wait()

                # Résumé (extrait de fin + éventuelle ligne SUMMARY)
                if rc == 0:
                    progress_display.value = (
                        "<b>Téléchargement terminé avec succès.</b>"
                    )
                else:
                    progress_display.value = (
                        "<b style='color:#c00'>Échec du téléchargement.</b>"
                    )

                # Affiche le chemin du log de session
                with log_output:
                    print(f"\nLog de session écrit : {log_path}")
                    try:
                        with open(log_path, "r", encoding="utf-8") as f:
                            tail = f.readlines()[-15:]
                        print("\n--- Extrait du log (fin) ---")
                        for line in tail:
                            print(line, end="")
                    except Exception:
                        pass

            except Exception as e:
                progress_display.value = f"<b style='color:#c00'>Erreur : {e}</b>"

        run_button.on_click(on_run_clicked)

    # -------------------------------------------------------------------------
    # 2) Nettoyage sécurisé de data/raw (backup + confirmation)
    # -------------------------------------------------------------------------

    def interactive_cleaner(self) -> None:
        """
        Vide `data/raw/` en toute sécurité (widget avec confirmation + backup).

        Affiche un petit outil interactif qui :
        1) **sauvegarde** le contenu actuel de `data/raw/` dans `data/raw_backup/`
            (écrasé à chaque exécution),
        2) demande une **confirmation explicite**,
        3) **supprime** ensuite le contenu de `data/raw/`.

        Cette option est utile pour repartir d’un état propre avant une grosse
        session de téléchargement, tout en évitant les suppressions accidentelles.

        Returns:
            None. Effets de bord :
            - création/écrasement de `data/raw_backup/`,
            - suppression du contenu de `data/raw/`.

        Raises:
            OSError: si une opération de copie/suppression échoue.
        """
        display(Markdown("### 2. Option de Nettoyage"))
        display(
            Markdown(
                "Cette option peut vider `data/raw/` après avoir fait une copie de sauvegarde "
                "dans `data/raw_backup/`. **Tape `CONFIRMER`** puis clique sur le bouton."
            )
        )

        confirm = widgets.Text(placeholder="Écrire 'CONFIRMER' ici")
        clean_btn = widgets.Button(
            description="Nettoyer le dossier `raw`", button_style="danger", icon="trash"
        )
        out = widgets.Output()

        def on_clean(_) -> None:
            with out:
                clear_output()

                if confirm.value.strip() != "CONFIRMER":
                    print("Confirmation incorrecte. Nettoyage annulé.")
                    return

                raw_dir = self.paths["RAW_DATA_DIR"]
                data_dir = os.path.dirname(raw_dir)
                backup_dir = os.path.join(data_dir, "raw_backup")

                # Gardes-fou minimaux
                if not os.path.isdir(raw_dir) or not os.path.isdir(data_dir):
                    print("Chemins invalides; arrêt.")
                    return
                if os.path.abspath(raw_dir) == os.path.abspath(backup_dir):
                    print("Backup et raw identiques — arrêt.")
                    return

                try:
                    print(f"Backup de '{raw_dir}' -> '{backup_dir}' ...")
                    # On supprime l'ancien backup s'il existe
                    if os.path.exists(backup_dir):
                        shutil.rmtree(backup_dir)

                    # On ne copie que si raw n'est pas vide (sinon backup inutile)
                    if os.listdir(raw_dir):
                        shutil.copytree(raw_dir, backup_dir, dirs_exist_ok=True)
                    else:
                        print("(raw est vide, pas de backup nécessaire)")

                    print("Nettoyage de 'raw' ...")
                    shutil.rmtree(raw_dir, ignore_errors=True)
                    os.makedirs(raw_dir, exist_ok=True)

                    print("OK : 'raw' nettoyé. (Backup disponible)")
                except Exception as e:  # noqa: BLE001
                    print(f"Erreur lors du nettoyage : {e}")

        clean_btn.on_click(on_clean)

        display(
            widgets.VBox(
                [
                    widgets.Label("ATTENTION : Action destructive !"),
                    confirm,
                    clean_btn,
                    out,
                ]
            )
        )
