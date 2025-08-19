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

PRESETS = {
    "cautious": dict(
        workers=8, scrape=4, fast=False, validate=True, per_plan=50, dry_run=False
    ),
    "balanced": dict(
        workers=16, scrape=8, fast=True, validate=True, per_plan=0, dry_run=False
    ),
    "max": dict(
        workers=24, scrape=12, fast=True, validate=False, per_plan=0, dry_run=False
    ),
}


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
            - Workers = nombre de téléchargements parallèles
            - Scrape  = parallélisme pendant l’analyse des pages *plan*
            - Mode rapide = `--delay 0` + `--chunk 131072` (débit maximal)
            - `Afficher la barre (tqdm)` (Checkbox): force `--progress`
            - Bouton “Lancer le téléchargement…”
            - Zone de progression + sortie texte “live”
            - `per-plan <N>`: plafonner le # de fichiers pris PAR plan (après anti-doublon)
            - `validate` : vérifier rapidement que le .fits.gz contient bien un header FITS
            - `dry-run` : ne rien écrire (aperçu seulement)

        Effets:
            - appelle `src/tools/dr5_downloader.py` en sous-processus
              avec `--log-to logs/download_log_YYYYMMDDTHHMMSSZ.txt --append`,
              donc **un seul log** pour la session.
        """

        # Widgets de base
        limit_widget = widgets.IntText(value=5, description="Plans:")
        max_spectra_widget = widgets.IntText(value=50, description="Spectres:")
        show_bar = widgets.Checkbox(value=True, description="Afficher la barre (tqdm)")
        fast_mode = widgets.Checkbox(value=True, description="Mode rapide")
        run_button = widgets.Button(
            description="Lancer le téléchargement…",
            button_style="success",
            icon="download",
        )

        # Parallélisme
        workers_widget = widgets.IntSlider(
            description="Workers", min=1, max=32, value=16
        )
        scrape_widget = widgets.IntSlider(description="Scrape", min=1, max=16, value=8)

        # Options avancées
        per_plan_widget = widgets.IntText(
            value=0, description="Par plan:"
        )  # 0 = illimité
        validate_widget = widgets.Checkbox(value=False, description="Valider FITS")
        dry_run_widget = widgets.Checkbox(
            value=False, description="Dry‑run (pas d’écriture)"
        )
        auto_widget = widgets.Checkbox(value=False, description="Auto‑throttle")
        maxw_widget = widgets.IntText(value=24, description="Max W:")
        minw_widget = widgets.IntText(value=4, description="Min W:")

        # ▼ Presets (avec garde‑fou anti-boucle d’événements)
        preset = widgets.Dropdown(
            options=[
                ("Cautious (8/4)", "cautious"),
                ("Balanced (16/8)", "balanced"),
                ("Max (24/12)", "max"),
            ],
            value="balanced",
            description="Préréglages:",
        )
        applying_preset = False
        PRESETS = {
            "cautious": dict(
                workers=8,
                scrape=4,
                fast=False,
                validate=True,
                per_plan=50,
                dry_run=False,
            ),
            "balanced": dict(
                workers=16,
                scrape=8,
                fast=True,
                validate=True,
                per_plan=0,
                dry_run=False,
            ),
            "max": dict(
                workers=24,
                scrape=12,
                fast=True,
                validate=False,
                per_plan=0,
                dry_run=False,
            ),
        }

        def apply_preset(name: str) -> None:
            nonlocal applying_preset
            cfg = PRESETS.get(name)
            if not cfg:
                return
            applying_preset = True
            try:
                workers_widget.value = cfg["workers"]
                scrape_widget.value = cfg["scrape"]
                fast_mode.value = cfg["fast"]
                validate_widget.value = cfg["validate"]
                per_plan_widget.value = cfg["per_plan"]
                dry_run_widget.value = cfg["dry_run"]
            finally:
                applying_preset = False

        def on_preset_change(change):
            if change.get("name") != "value" or applying_preset:
                return
            apply_preset(change["new"])

        preset.observe(on_preset_change, names="value")
        apply_preset(preset.value)

        # Affichages
        progress_display = widgets.HTML(value="")
        log_output = widgets.Output()

        line0 = widgets.HBox([preset])
        line1 = widgets.HBox([limit_widget, max_spectra_widget, show_bar, fast_mode])
        line2 = widgets.HBox([workers_widget, scrape_widget, auto_widget, maxw_widget])
        line3 = widgets.HBox(
            [per_plan_widget, validate_widget, dry_run_widget, minw_widget]
        )

        display(Markdown("### 1. Téléchargement des Spectres"))
        display(
            Markdown(
                "Utilisez l’interface ci-dessous pour lancer un nouveau téléchargement. "
                "Les logs de progression s’afficheront en temps réel.\n\n"
                "**Astuces** : Workers = téléchargements parallèles; Scrape = parallélisme d’analyse des pages plan; "
                "Par plan = coupe la liste par plan; Valider FITS = vérif header; Dry‑run = aperçu sans écriture."
            )
        )
        display(line0, line1, line2, line3, run_button, progress_display, log_output)

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
                "--workers",
                str(workers_widget.value),
                "--scrape-workers",
                str(scrape_widget.value),
            ]
            if show_bar.value:
                cmd.append("--progress")
            if fast_mode.value:
                cmd += ["--delay", "0", "--chunk", "131072"]  # 128 KB
            if per_plan_widget.value and per_plan_widget.value > 0:
                cmd += ["--per-plan", str(per_plan_widget.value)]
            if validate_widget.value:
                cmd.append("--validate")
            if dry_run_widget.value:
                cmd.append("--dry-run")
            if auto_widget.value:
                cmd.append("--auto-throttle")
                if maxw_widget.value:
                    cmd += ["--max-workers", str(maxw_widget.value)]
                if "minw_widget" in locals() and minw_widget.value:
                    cmd += ["--min-workers", str(minw_widget.value)]

            # Affichage d’un message + désactivation du bouton (évite double clic)
            progress_display.value = "<b>Lancement du téléchargement...</b>"
            run_button.disabled = True

            try:
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
                        if "Téléchargement" in line and "%" in line:
                            clean = line.strip()
                            m = self._RE_PCT.search(clean)
                            pct = int(m.group(1)) if m else 0
                            progress_display.value = (
                                "<div style='font-family:monospace;white-space:pre'>"
                                + clean
                                + "</div>"
                                f"<progress value='{pct}' max='100' style='width:100%;height:18px'></progress>"
                            )
                        else:
                            with log_output:
                                print(line, end="")
                rc = proc.wait()
                if rc == 0:
                    if dry_run_widget.value:
                        progress_display.value = (
                            "<b>Dry‑run terminé (aucune écriture).</b>"
                        )
                    else:
                        progress_display.value = (
                            "<b>Téléchargement terminé avec succès.</b>"
                        )
                else:
                    progress_display.value = (
                        "<b style='color:#c00'>Échec du téléchargement.</b>"
                    )
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
            finally:
                run_button.disabled = False

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
