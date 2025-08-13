"""
# dr5_downloader — Téléchargeur LAMOST DR5 (intelligent, reprenable, rapide)

Ce module fournit un téléchargeur *smart* pour les spectres LAMOST DR5. Il :

- lit une liste de plans valides depuis `data/catalog/valid_plan_urls.csv`,
- exclut les plans déjà complétés (journal `data/catalog/downloaded_plans.csv`),
- *scrape* les liens `.fits.gz` pour chaque plan et construit une file,
- télécharge en **round‑robin** (équilibrage entre plans),
- reprend proprement : fichiers présents ignorés; écriture atomique `.part` → `os.replace()`,
- tient un **log de session** dans `logs/download_log_YYYYMMDDTHHMMSSZ.txt`,
- marque les plans désormais complets en fin de session.

Conventions
-----------
- Arborescence projet (résolue relativement à ce fichier) :
  - `data/catalog/valid_plan_urls.csv`   → colonnes : `url`
  - `data/catalog/downloaded_plans.csv`  → colonnes : `url`
  - `data/raw/<plan>/<filename>.fits.gz` → destination de chaque fichier
  - `logs/*.txt`                         → logs de session
- Toutes les longueurs/tailles sont exprimées en unités SI habituelles.

Entrées / Sorties
-----------------
Entrées : CSV `valid_plan_urls.csv` et `downloaded_plans.csv` (facultatif).
Sorties : fichiers `.fits.gz` sous `data/raw/` + un fichier de log de session.

Interface
---------
- API Python : `SmartDownloader(limit_plans, max_spectra, ...)` → `.run()`
- CLI : `python -m src.tools.dr5_downloader --limit 5 --max-spectres 200 --progress`

Exemple minimal
---------------
>>> # Python
>>> dl = SmartDownloader(limit_plans=3, max_spectra=120, progress=True)
>>> dl.run()

>>> # Ligne de commande
>>> # Dans la racine du projet (après activation du venv)
>>> python -m src.tools.dr5_downloader --limit 3 --max-spectres 120 --progress

Dépendances
-----------
`requests`, `pandas`, `beautifulsoup4`, `tqdm`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import zip_longest
from typing import Iterable, List, Optional

import argparse
import os
import sys
from time import sleep
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm

__all__ = ["SmartDownloader", "main"]
__version__ = "2.0.0"


@dataclass(slots=True)
class _Paths:
    """Conteneur des chemins utilisés par le téléchargeur.

    Les chemins sont résolus relativement à l’emplacement de ce fichier,
    pour permettre une exécution fiable depuis le notebook **ou** le terminal.
    """

    catalog_dir: str
    raw_data_dir: str
    logs_dir: str


class SmartDownloader:
    """Téléchargeur intelligent et reprenable pour les spectres LAMOST DR5.

    Phases
    ------
    1. **Scrape** des pages de plan → collecte des liens `.fits.gz`.
    2. **Téléchargement round‑robin** (équilibrage entre plans), barre `tqdm` optionnelle.
    3. **Mise à jour de l’état** : marquage des plans désormais complets.

    Paramètres
    ----------
    limit_plans : int | None
        Nombre maximal de **nouveaux** plans à traiter (None = illimité).
    max_spectra : int | None
        Arrêt anticipé après ce nombre total de spectres (None = illimité).
    timeout : int, default=60
        Délai (s) appliqué aux requêtes HTTP.
    delay_between : float, default=0.2
        Petit délai (s) entre deux téléchargements (mettre 0 pour accélérer).
    retries : int, default=3
        Nombre de tentatives réseau (backoff exponentiel 0.8).
    progress : bool, default=True
        Afficher une barre `tqdm` (Notebook : cocher/forcer si besoin).
    log_to : str | None
        Chemin du **log unique** de la session (généré si None).
    append : bool, default=False
        Si `True`, on ajoute au log existant au lieu de l’écraser.
    chunk_size : int, default=65536
        Taille des chunks écrits (octets). 65536=64kB, 131072=128kB, etc.
    """

    # --------------------------- initialisation & chemins ---------------------------

    def __init__(
        self,
        limit_plans: Optional[int],
        max_spectra: Optional[int],
        *,
        timeout: int = 60,
        delay_between: float = 0.2,
        retries: int = 3,
        progress: bool = True,
        log_to: Optional[str] = None,
        append: bool = False,
        chunk_size: int = 65536,
    ) -> None:
        # Résolution robuste des chemins (indépendant du CWD)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(script_dir))
        self.catalog_dir = os.path.join(project_root, "data", "catalog")
        self.raw_data_dir = os.path.join(project_root, "data", "raw")
        self.logs_dir = os.path.join(project_root, "logs")

        os.makedirs(self.catalog_dir, exist_ok=True)
        os.makedirs(self.raw_data_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)

        self.paths = _Paths(self.catalog_dir, self.raw_data_dir, self.logs_dir)

        # Fichier de log (soit imposé par --log-to, soit généré)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.session_log_path = log_to or os.path.join(
            self.logs_dir, f"download_log_{ts}.txt"
        )
        if log_to and not append:
            # Écrase proprement si l’utilisateur demande un chemin fixe
            open(self.session_log_path, "w", encoding="utf-8").close()

        # D'abord les paramètres de session
        self.progress = progress
        self.chunk_size = int(chunk_size)
        self.limit_plans = limit_plans
        self.max_spectra = max_spectra
        self.timeout = int(timeout)
        self.delay_between = float(delay_between)
        self.total_downloaded_this_session = 0

        # HTTP session
        self.session = self._build_session(retries=retries, timeout=timeout)

        # État courant
        self.valid_urls_file = os.path.join(self.catalog_dir, "valid_plan_urls.csv")
        self.downloaded_log_file = os.path.join(
            self.catalog_dir, "downloaded_plans.csv"
        )
        self.plans_to_process = self._load_state()
        self.download_queue: list[list[str]] = []

        # En‑tête de session dans le log
        self._log("--- Nouvelle session de téléchargement ---")
        self._log(f"version={__version__}")
        self._log(
            f"plans={self.limit_plans or 'all'}, max_spectres={self.max_spectra or '∞'}"
        )

    # -------------------------------- utilitaires I/O --------------------------------

    def _log(self, msg: str) -> None:
        """Ajoute une ligne au fichier de log de session (UTF‑8, no fail)."""
        try:
            with open(self.session_log_path, "a", encoding="utf-8") as fh:
                fh.write(msg.rstrip() + "\n")
        except Exception:
            # On ne casse jamais le run sur une erreur de log
            pass

    def _say(self, msg: str) -> None:
        """Affiche et écrit dans le log de session."""
        print(msg)
        self._log(msg)

    def _say_tqdm(self, msg: str) -> None:
        """Version sûre quand une barre `tqdm` est affichée."""
        tqdm.write(msg)
        self._log(msg)

    # --------------------------- utilitaires HTTP & état ---------------------------

    @staticmethod
    def _build_session(*, retries: int, timeout: int) -> requests.Session:
        """Construit une `requests.Session` avec *retries* & backoff exponentiel.

        Notes
        -----
        - Un **User‑Agent** explicite est défini.
        - Les timeouts sont passés à chaque `get()` via `timeout=...`.
        """
        session = requests.Session()
        retry_cfg = Retry(
            total=retries,
            backoff_factor=0.8,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=("GET", "HEAD"),
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry_cfg)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        session.headers.update(
            {
                "User-Agent": "AstroSpectro-Downloader/2.0 (+https://example.org)",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            }
        )
        # Timeout par défaut mémorisé (attribut interne simple)
        session.request_timeout = timeout  # type: ignore[attr-defined]
        return session

    def _load_state(self) -> List[str]:
        """Charge la liste des plans à traiter.

        Lit `valid_plan_urls.csv`, puis enlève les `url` déjà présentes dans
        `downloaded_plans.csv`. Applique `self.limit_plans` si défini.

        Returns
        -------
        list[str]
            URLs restantes, dans l’ordre d’apparition.
        """
        try:
            df_valid = pd.read_csv(self.valid_urls_file)
        except FileNotFoundError:
            print(
                f"ERREUR : Fichier introuvable : '{self.valid_urls_file}'. "
                "Génère-le d’abord (liste des plans valides)."
            )
            return []

        already_completed: set[str] = set()
        if os.path.exists(self.downloaded_log_file):
            try:
                df_completed = pd.read_csv(self.downloaded_log_file)
                already_completed = set(df_completed["url"].astype(str).tolist())
                print(f"  > {len(already_completed)} plan(s) déjà complété(s).")
            except Exception:
                pass

        remaining = df_valid[~df_valid["url"].isin(already_completed)]["url"].tolist()
        if self.limit_plans:
            remaining = remaining[: self.limit_plans]
        return remaining

    # -------------------------------- phase 1 : scrape --------------------------------

    def _build_download_queue(self) -> None:
        """Scrape chaque page de plan et construit `self.download_queue`.

        Pour chaque plan, toutes les ancres se terminant par `.fits.gz` sont
        collectées puis **normalisées** en URLs absolues via `urljoin`.
        """
        if not self.plans_to_process:
            self._say("\nAucun plan à traiter. (Liste vide)")
            return

        self._say("\n--- [Phase 1/3] Construction ... ---")
        for i, plan_url in enumerate(self.plans_to_process, start=1):
            plan_name = plan_url.rstrip("/").split("/")[-1]
            self._say(f"  > [{i}/{len(self.plans_to_process)}] {plan_name}")

            try:
                resp = self.session.get(plan_url, timeout=self.session.request_timeout)  # type: ignore[attr-defined]
                resp.raise_for_status()
            except requests.RequestException as e:
                self._say(f"    -> ERREUR : accès impossible ({e}). Plan ignoré.")
                continue

            soup = BeautifulSoup(resp.text, "html.parser")
            plan_fits_urls: List[str] = []
            for link in soup.find_all("a"):
                href = (link.get("href") or "").strip()
                if href.endswith(".fits.gz"):
                    plan_fits_urls.append(urljoin(plan_url, href))

            if plan_fits_urls:
                self.download_queue.append(plan_fits_urls)
            else:
                self._say("    -> Avertissement : aucun .fits.gz trouvé.")

        self._say("  > File construite.")

    # ---------------------------- phase 2 : téléchargement ----------------------------

    def _iter_missing_files(self) -> Iterable[str]:
        """Itère toutes les URLs absentes en local (tous plans confondus)."""
        for plan_fits_urls in self.download_queue:
            for file_url in plan_fits_urls:
                plan_name = file_url.split("/")[-2]
                filename = file_url.split("/")[-1]
                dest_dir = os.path.join(self.raw_data_dir, plan_name)
                dest_path = os.path.join(dest_dir, filename)
                if not os.path.exists(dest_path):
                    yield file_url

    def _stream_download(self, url: str, dest_path: str) -> bool:
        """Télécharge `url` → `dest_path` avec écriture **atomique**.

        Un fichier temporaire `dest_path + '.part'` est utilisé pendant l’écriture,
        puis remplacé via `os.replace()` seulement en cas de succès.

        Returns
        -------
        bool
            `True` si succès, `False` sinon (loggué via `_say_tqdm`).
        """
        temp_path = f"{dest_path}.part"
        try:
            with self.session.get(url, stream=True, timeout=self.session.request_timeout) as r:  # type: ignore[attr-defined]
                r.raise_for_status()
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                with open(temp_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=self.chunk_size):
                        if chunk:
                            f.write(chunk)
            os.replace(temp_path, dest_path)
            return True
        except requests.RequestException as e:
            # Nettoie le .part éventuel
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            except Exception:
                pass
            self._say_tqdm(f"    -> ERREUR : {os.path.basename(dest_path)} ({e})")
            return False

    def run_download(self) -> None:
        """Télécharge les fichiers manquants en **round‑robin**.

        Respecte `self.max_spectra` si défini. Affiche une barre `tqdm` si
        `self.progress` vaut `True` (ou si `--progress` a été passé en CLI).
        """
        if not self.download_queue:
            self._say("\n[Phase 2/3] Rien à faire : file vide.")
            return

        # 1) Manquants par plan
        missing_by_plan: list[list[str]] = []
        for plan_urls in self.download_queue:
            plan_missing: list[str] = []
            for url in plan_urls:
                plan_name = url.split("/")[-2]
                filename = url.split("/")[-1]
                dest_dir = os.path.join(self.raw_data_dir, plan_name)
                dest_path = os.path.join(dest_dir, filename)
                if not os.path.exists(dest_path):
                    plan_missing.append(url)
            if plan_missing:
                missing_by_plan.append(plan_missing)

        if not missing_by_plan:
            self._say("\n--- [Phase 2/3] Rien à télécharger (tout est déjà présent).")
            return

        # 2) Entrelacement + coupe `max_spectra` pendant l’entrelacement
        interleaved: list[str] = []
        for batch in zip_longest(*missing_by_plan):
            for url in batch:
                if url is None:
                    continue
                interleaved.append(url)
                if self.max_spectra is not None and len(interleaved) >= int(
                    self.max_spectra
                ):
                    break
            if self.max_spectra is not None and len(interleaved) >= int(
                self.max_spectra
            ):
                break

        total = len(interleaved)
        self._say(f"\n--- [Phase 2/3] {total} nouveaux spectres à télécharger ---")

        # 3) Téléchargement avec barre (désactivée si progress=False)
        disable_progress = not getattr(self, "progress", True)
        with tqdm(
            total=total,
            desc="Téléchargement",
            unit="spectre",
            leave=False,
            mininterval=0.4,
            disable=disable_progress,
        ) as pbar:
            for url in interleaved:
                plan_name = url.split("/")[-2]
                filename = url.split("/")[-1]
                dest_dir = os.path.join(self.raw_data_dir, plan_name)
                dest_path = os.path.join(dest_dir, filename)

                if os.path.exists(dest_path):
                    pbar.update(1)
                    continue

                if (
                    self.max_spectra
                    and self.total_downloaded_this_session >= self.max_spectra
                ):
                    self._say_tqdm(
                        f"Limite atteinte ({self.max_spectra}). Arrêt du téléchargement."
                    )
                    return

                ok = self._stream_download(url, dest_path)
                if ok:
                    self._log(f"OK {plan_name}/{filename}")
                    self.total_downloaded_this_session += 1
                    pbar.update(1)
                    if self.delay_between > 0:
                        sleep(self.delay_between)

    # ------------------------------ phase 3 : marquage ------------------------------

    def _update_state(self) -> None:
        """Vérifie quels plans sont désormais **complets**, puis journalise.

        Un plan est « complet » si tous les fichiers attendus (liens trouvés
        pendant la phase 1) existent maintenant sous `data/raw/<plan>/`.
        """
        self._say("\n--- [Phase 3/3] Mise à jour de l'état des plans ---")

        newly_completed: List[str] = []
        for plan_url, fits_list in zip(self.plans_to_process, self.download_queue):
            plan_name = plan_url.rstrip("/").split("/")[-1]
            dest_dir = os.path.join(self.raw_data_dir, plan_name)

            if all(
                os.path.exists(os.path.join(dest_dir, url.split("/")[-1]))
                for url in fits_list
            ):
                newly_completed.append(plan_url)
                self._say(f"  > Plan complété : {plan_name}")

        if not newly_completed:
            self._say("  > Aucun nouveau plan complété.")
            return

        df_new = pd.DataFrame({"url": newly_completed})
        df_new.to_csv(
            self.downloaded_log_file,
            mode="a",
            index=False,
            header=not os.path.exists(self.downloaded_log_file),
        )
        self._say(
            f"  > {len(newly_completed)} plan(s) ajouté(s) à '{self.downloaded_log_file}'."
        )

    # --------------------------------- façade simple ---------------------------------

    def run(self) -> None:
        """Enchaîne les 3 phases (scrape → download → update) et résume la session."""
        self._build_download_queue()
        self.run_download()
        self._update_state()
        self._log(f"SUMMARY total_downloaded={self.total_downloaded_this_session}")
        self._say(f"\nLog de session écrit : {self.session_log_path}")


# ----------------------------------- CLI -----------------------------------


def _parse_args() -> argparse.Namespace:
    """Arguments de la ligne de commande.

    - `--limit` : nombre max de **nouveaux** plans à traiter.
    - `--max-spectres` : plafond de spectres pour cette session.
    - `--progress` : forcer l’affichage de la barre `tqdm` (utile en notebook).
    - `--delay` : délai (s) entre deux téléchargements.
    - `--chunk` : taille d’écriture par bloc (octets).
    - `--log-to` + `--append` : contrôle du fichier de log.
    """
    parser = argparse.ArgumentParser(
        description="Smart Downloader pour spectres LAMOST DR5"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Nombre maximal de NOUVEAUX plans à traiter (ex: 5).",
    )
    parser.add_argument(
        "--max-spectres",
        type=int,
        default=None,
        help="Arrêt après ce nombre total de spectres téléchargés (ex: 200).",
    )
    parser.add_argument(
        "--log-to",
        type=str,
        default=None,
        help="Chemin du fichier log unique à utiliser (généré si absent).",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Ajouter au log existant au lieu de l'écraser.",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Forcer l’affichage de la barre tqdm même si stdout n’est pas un TTY (ex: Notebook).",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.2,
        help="Délai (s) entre deux téléchargements. 0 = le plus rapide.",
    )
    parser.add_argument(
        "--chunk",
        type=int,
        default=65536,
        help="Taille des chunks de téléchargement (octets). 65536=64KB, 131072=128KB, etc.",
    )
    return parser.parse_args()


def main(argv: Optional[list[str]] = None) -> int:
    """Point d’entrée *programmable* (utile pour tests/CI)."""
    args = _parse_args() if argv is None else _parse_args()
    dl = SmartDownloader(
        limit_plans=args.limit,
        max_spectra=args.max_spectres,
        progress=(args.progress or sys.stdout.isatty()),
        delay_between=args.delay,
        chunk_size=args.chunk,
        log_to=args.log_to,
        append=args.append,
    )
    dl.run()
    dl._say(
        f"\n--- Session terminée : {dl.total_downloaded_this_session} spectre(s) "
        f"téléchargé(s) sur {len(dl.plans_to_process)} plan(s). ---"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
