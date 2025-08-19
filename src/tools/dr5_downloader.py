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
import gzip
import random
import math
import statistics
import time
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

__all__ = ["SmartDownloader", "main"]
__version__ = "3.1.0"


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
        workers: int = 1,
        scrape_workers: int = 1,
        per_plan: int | None = None,
        validate: bool = False,
        dry_run: bool = False,
        auto_throttle: bool = False,
        min_workers: int | None = None,
        max_workers: int | None = None,
        target_latency: float = 0.5,  # en secondes (médiane visée)
        tune_step: int = 2,
        tune_err_lo: float = 0.02,
        tune_err_hi: float = 0.05,
        throttle_delay: float = 0.05,
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
        self.workers = max(1, int(workers))
        self.scrape_workers = max(1, int(scrape_workers))
        self.per_plan = None if per_plan in (None, 0) else int(per_plan)
        self.validate_downloads = bool(validate)
        self.dry_run = bool(dry_run)
        self.retries = int(retries)
        self.plan_urls_in_queue: list[str] = []
        self.auto_throttle = bool(auto_throttle)
        self.min_workers = int(min_workers or 4)  # (borne basse)
        self.max_workers = int(max_workers or self.workers)  # (borne haute)
        self.target_latency = float(target_latency)  # (s)
        self.tune_step = int(tune_step)  # (+2 par défaut)
        self.tune_err_lo = float(tune_err_lo)  # (< 2% : on pousse)
        self.tune_err_hi = float(tune_err_hi)  # (> 5% : on réduit)
        self.throttle_delay = float(throttle_delay)  # (retard minimal si 429)

        # HTTP session
        self.session = self._build_session(retries=retries, timeout=timeout)

        # État courant
        self.valid_urls_file = os.path.join(self.catalog_dir, "valid_plan_urls.csv")
        self.downloaded_log_file = os.path.join(
            self.catalog_dir, "downloaded_plans.csv"
        )
        self.plans_to_process = self._load_state()
        self.download_queue: list[list[str]] = []
        self._med_baseline: float | None = None

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
        adapter = HTTPAdapter(
            max_retries=retry_cfg, pool_connections=64, pool_maxsize=64
        )
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        session.headers.update(
            {
                "User-Agent": "AstroSpectro-Downloader/3.1 (+https://example.org)",
                "Accept": "*/*",
                "Connection": "keep-alive",
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

    def _scrape_plan(self, plan_url: str):
        """
        Retourne (plan_name, urls .fits.gz) pour une page *plan* (thread‑safe).
        """
        session = self._build_session(retries=3, timeout=self.timeout)
        plan_name = plan_url.rstrip("/").split("/")[-1]
        urls: list[str] = []
        try:
            resp = session.get(plan_url, timeout=session.request_timeout)  # type: ignore[attr-defined]
            resp.raise_for_status()
        except requests.RequestException:
            return plan_name, urls

        soup = BeautifulSoup(resp.text, "html.parser")
        for link in soup.find_all("a"):
            href = (link.get("href") or "").strip()
            if href.endswith(".fits.gz"):
                urls.append(urljoin(plan_url, href))
        return plan_name, urls

    @staticmethod
    def _is_valid_fits_gz(path: str) -> bool:
        """
        Validation légère : header GZIP lisible et carte FITS 'SIMPLE' en tête.
        Ne décompresse pas tout; lit seulement le début.
        """
        try:
            with gzip.open(path, "rb") as fh:
                head = fh.read(2880)  # 1 bloc FITS
            return b"SIMPLE" in head[:80]
        except Exception:
            return False

    # -------------------------------- phase 1 : scrape --------------------------------

    def _build_download_queue(self) -> None:
        """
        Phase 1 — scrape des pages de plan et construction de `self.download_queue`.

        • Séquentiel si `self.scrape_workers == 1` (comportement historique).
        • Parallèle sinon, avec ORDRE STABLE (mappage par URL de plan).
        • Renseigne aussi `self.plan_urls_in_queue` pour garder l'alignement Phase 2/3.
        """
        if not self.plans_to_process:
            self._say("\nAucun plan à traiter. (Liste vide)")
            self.download_queue = []
            self.plan_urls_in_queue = []
            return

        self._say("\n--- [Phase 1/3] Construction ... ---")

        # Mode séquentiel (ordre d'origine garanti)
        if self.scrape_workers == 1:
            selected: list[str] = []
            queue: list[list[str]] = []
            for i, plan_url in enumerate(self.plans_to_process, start=1):
                plan_name, plan_urls = self._scrape_plan(plan_url)
                self._say(
                    f"  > [{i}/{len(self.plans_to_process)}] {plan_name}  ({len(plan_urls)} fichiers)"
                )
                if plan_urls:
                    queue.append(plan_urls)
                    selected.append(plan_url)
                else:
                    self._say("    -> Avertissement : aucun .fits.gz trouvé.")
            self.download_queue = queue
            self.plan_urls_in_queue = selected
            self._say("  > File construite.")
            return

        # Mode parallèle (on collecte d'abord dans un dict puis on reconstruit dans l'ordre d'origine)
        results: dict[str, list[str]] = {}
        with ThreadPoolExecutor(max_workers=self.scrape_workers) as ex:
            futures = {
                ex.submit(self._scrape_plan, url): url for url in self.plans_to_process
            }
            for i, fut in enumerate(as_completed(futures), start=1):
                plan_url = futures[fut]
                try:
                    plan_name, plan_urls = fut.result()
                except Exception as e:  # en cas d'erreur réseau/parsing
                    self._say(f"  > [{i}/{len(futures)}] ERREUR : {e}")
                    plan_urls = []
                self._say(
                    f"  > [{i}/{len(futures)}] {plan_url.rstrip('/').split('/')[-1]}  ({len(plan_urls)} fichiers)"
                )
                results[plan_url] = plan_urls

        # Reconstitution dans l'ordre self.plans_to_process et exclusion des plans vides
        self.plan_urls_in_queue = [u for u in self.plans_to_process if results.get(u)]
        self.download_queue = [results[u] for u in self.plan_urls_in_queue]
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

    def _download_one_threaded(self, url: str) -> bool:
        """Télécharge `url` (nouvelle Session locale) → True si écrit.

        Identique à `_stream_download`, mais calcule `dest_path` et évite de
        partager `self.session` entre threads (Requests Session n'est pas thread‑safe).
        """
        plan_name = url.split("/")[-2]
        filename = url.split("/")[-1]
        dest_dir = os.path.join(self.raw_data_dir, plan_name)
        os.makedirs(dest_dir, exist_ok=True)
        dest_path = os.path.join(dest_dir, filename)
        temp_path = f"{dest_path}.part"

        # si déjà présent et pas de validation demandée → rien à faire
        if os.path.exists(dest_path) and not self.validate_downloads:
            return False

        session = self._build_session(retries=self.retries, timeout=self.timeout)

        # Gestion reprise
        resume_from = 0
        headers = {}
        if os.path.exists(temp_path):
            resume_from = os.path.getsize(temp_path)
            if resume_from > 0:
                headers["Range"] = f"bytes={resume_from}-"

        # Boucle d'essais avec backoff+jitter applicatif (en plus de urllib3 Retry)
        for attempt in range(self.retries + 1):
            try:
                with session.get(url, stream=True, headers=headers, timeout=session.request_timeout) as r:  # type: ignore[attr-defined]
                    if resume_from and r.status_code == 200:
                        # Le serveur n'a pas accepté Range → on repart de zéro
                        resume_from = 0
                    r.raise_for_status()

                    mode = "ab" if resume_from else "wb"
                    with open(temp_path, mode) as f:
                        for chunk in r.iter_content(chunk_size=self.chunk_size):
                            if chunk:
                                f.write(chunk)
                os.replace(temp_path, dest_path)

                # Validation légère (si demandée)
                if self.validate_downloads and not self._is_valid_fits_gz(dest_path):
                    self._say_tqdm(
                        f"    -> Validation échouée : {filename}. Nouvelle tentative…"
                    )
                    try:
                        os.remove(dest_path)
                    except Exception:
                        pass
                    resume_from = 0
                    headers.pop("Range", None)
                    # retente
                    raise requests.RequestException("validation failed")

                self._log(f"OK {plan_name}/{filename}")
                return True

            except requests.RequestException as e:
                # petit backoff jitter en plus du Retry de l'adapter
                if attempt < self.retries:
                    sleep_s = (0.5 * (2**attempt)) * (0.5 + random.random())
                    try:
                        import time as _t

                        _t.sleep(sleep_s)
                    except Exception:
                        pass
                    continue
                else:
                    # échec final → nettoyer le .part
                    try:
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
                    except Exception:
                        pass
                    self._say_tqdm(f"    -> ERREUR : {filename} ({e})")
                    return False

    def _download_one_stats(self, url: str) -> tuple[bool, float, int]:
        """
        Comme `_download_one_threaded`, mais retourne (succès, durée_s, code).
        code = 0 (ok/erreur générique), 429, 5xx (500–599).
        """
        start = time.monotonic()
        plan_name = url.split("/")[-2]
        filename = url.split("/")[-1]
        dest_dir = os.path.join(self.raw_data_dir, plan_name)
        os.makedirs(dest_dir, exist_ok=True)
        dest_path = os.path.join(dest_dir, filename)
        temp_path = f"{dest_path}.part"

        if os.path.exists(dest_path) and not self.validate_downloads:
            return (False, time.monotonic() - start, 0)

        session = self._build_session(retries=self.retries, timeout=self.timeout)

        resume_from = 0
        headers = {}
        if os.path.exists(temp_path):
            resume_from = os.path.getsize(temp_path)
            if resume_from > 0:
                headers["Range"] = f"bytes={resume_from}-"

        code = 0
        for attempt in range(self.retries + 1):
            try:
                with session.get(url, stream=True, headers=headers, timeout=session.request_timeout) as r:  # type: ignore[attr-defined]
                    if resume_from and r.status_code == 200:
                        resume_from = 0
                    r.raise_for_status()
                    mode = "ab" if resume_from else "wb"
                    with open(temp_path, mode) as f:
                        for chunk in r.iter_content(chunk_size=self.chunk_size):
                            if chunk:
                                f.write(chunk)
                os.replace(temp_path, dest_path)

                if self.validate_downloads and not self._is_valid_fits_gz(dest_path):
                    self._say_tqdm(
                        f"    -> Validation échouée : {filename}. Nouvelle tentative…"
                    )
                    try:
                        os.remove(dest_path)
                    except Exception:
                        pass
                    resume_from = 0
                    headers.pop("Range", None)
                    raise requests.RequestException("validation failed")

                self._log(f"OK {plan_name}/{filename}")
                return (True, time.monotonic() - start, 0)

            except requests.HTTPError as e:
                sc = getattr(e.response, "status_code", 0) or 0
                code = 429 if sc == 429 else (sc if 500 <= sc <= 599 else 0)
                if attempt < self.retries:
                    sleep_s = (0.5 * (2**attempt)) * (0.5 + random.random())
                    time.sleep(sleep_s)
                    continue
                try:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                except Exception:
                    pass
                self._say_tqdm(f"    -> ERREUR HTTP {sc} : {filename}")
                return (False, time.monotonic() - start, code)
            except requests.RequestException:
                if attempt < self.retries:
                    sleep_s = (0.5 * (2**attempt)) * (0.5 + random.random())
                    time.sleep(sleep_s)
                    continue
                try:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                except Exception:
                    pass
                self._say_tqdm(f"    -> ERREUR : {filename}")
                return (False, time.monotonic() - start, code)

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
        • Parallèle si `self.workers > 1` (ThreadPoolExecutor + tqdm)
        """
        if not self.download_queue:
            self._say("\n[Phase 2/3] Rien à faire : file vide.")
            return

        # 1) Construire la liste entrelacée entre plans
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

        n_active = len(missing_by_plan)
        auto_per_plan = None
        if (
            (self.per_plan in (None, 0))
            and (self.max_spectra is not None)
            and n_active > 0
        ):
            auto_per_plan = max(1, math.ceil(int(self.max_spectra) / n_active))

        limit_per_plan = auto_per_plan if auto_per_plan is not None else self.per_plan
        if limit_per_plan is not None:
            missing_by_plan = [urls[: int(limit_per_plan)] for urls in missing_by_plan]

        self._say("\nRésumé des manquants (après anti-doublon et per-plan) :")
        for plan in missing_by_plan:
            if not plan:
                continue
            plan_name = plan[0].split("/")[-2]
            self._say(f"  - {plan_name}: {len(plan)} fichiers à télécharger")

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

        # 2) Mode DRY‑RUN : on montre l'aperçu et on s'arrête ici
        if self.dry_run:
            self._say("\n[Dry‑run] Aperçu de la session (aucune écriture disque):")
            self._say(
                f"  > {len(missing_by_plan)} plan(s), {total} fichier(s) à traiter (après max_spectres)"
            )
            return

        # 3) Téléchargements
        self._say(
            f"\n--- [Phase 2/3] {total} nouveaux spectres à télécharger (workers={self.workers}) ---"
        )
        disable_progress = not bool(self.progress)

        if self.workers <= 1:
            # Chemin historique (séquentiel)
            from time import sleep as _sl

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
                        _sl(self.delay_between)
            return

        # Parallèle
        if self.workers > 1:
            if not self.auto_throttle:
                with tqdm(
                    total=total,
                    desc="Téléchargement",
                    unit="spectre",
                    leave=False,
                    mininterval=0.4,
                    disable=disable_progress,
                ) as pbar:
                    with ThreadPoolExecutor(max_workers=self.workers) as ex:
                        futures = [
                            ex.submit(self._download_one_threaded, url)
                            for url in interleaved
                        ]
                        for fut in as_completed(futures):
                            try:
                                wrote = fut.result()
                                if wrote:
                                    self.total_downloaded_this_session += 1
                            except Exception:
                                pass
                            pbar.update(1)
            else:
                # Auto‑throttle : on traite en lots de `self.workers` et on ajuste
                i = 0
                with tqdm(
                    total=total,
                    desc="Téléchargement",
                    unit="spectre",
                    leave=False,
                    mininterval=0.4,
                    disable=disable_progress,
                ) as pbar:
                    while i < len(interleaved):
                        batch = interleaved[i : i + self.workers]
                        succ = 0
                        latencies: list[float] = []
                        err429 = err5xx = total_b = 0

                        with ThreadPoolExecutor(max_workers=self.workers) as ex:
                            futs = [
                                ex.submit(self._download_one_stats, u) for u in batch
                            ]
                            for fut in as_completed(futs):
                                try:
                                    ok, dt, code = fut.result()
                                    if ok:
                                        succ += 1
                                    latencies.append(dt)
                                    if code == 429:
                                        err429 += 1
                                    elif 500 <= code <= 599:
                                        err5xx += 1
                                except Exception:
                                    pass
                                total_b += 1
                                pbar.update(1)

                        # --- Ajustement UNE fois par lot ---
                        err_rate = (err429 + err5xx) / max(1, total_b)
                        med = statistics.median(latencies) if latencies else 0.0

                        # compteur mis à jour une seule fois par lot
                        self.total_downloaded_this_session += succ

                        # Baseline EMA de la latence médiane
                        if self._med_baseline is None:
                            self._med_baseline = med or self.target_latency
                        elif med:
                            self._med_baseline = 0.2 * med + 0.8 * self._med_baseline

                        # Seuils dynamiques vs baseline
                        hi_thresh = max(
                            self.target_latency, (self._med_baseline or 0) * 1.5
                        )
                        lo_thresh = max(self.target_latency, (self._med_baseline or 0))

                        new_workers = self.workers
                        if (
                            err429 > 0
                            or err_rate > self.tune_err_hi
                            or (med and med > hi_thresh)
                        ):
                            new_workers = max(self.min_workers, self.workers // 2)
                            if err429 > 0:
                                self.delay_between = max(
                                    self.delay_between, self.throttle_delay
                                )
                        elif err_rate < self.tune_err_lo and (
                            med == 0.0 or med <= lo_thresh
                        ):
                            new_workers = min(
                                self.max_workers, self.workers + self.tune_step
                            )

                        if new_workers != self.workers:
                            self._say_tqdm(
                                f"Auto-throttle: workers {self.workers} → {new_workers}  "
                                f"(err={err_rate:.1%}, med={med*1000:.0f} ms, base={((self._med_baseline or 0)*1000):.0f} ms)"
                            )
                            self.workers = new_workers

                        i += len(batch)
                # fin auto‑throttle
            return

    # ------------------------------ phase 3 : marquage ------------------------------

    def _update_state(self) -> None:
        """
        Vérifie quels plans sont désormais **complets**, puis journalise.

        Un plan est « complet » si **tous** les fichiers attendus (liens trouvés
        en phase 1) existent sous `data/raw/<plan>/`.
        """
        self._say("\n--- [Phase 3/3] Mise à jour de l'état des plans ---")

        newly_completed: List[str] = []
        for plan_url, fits_list in zip(self.plan_urls_in_queue, self.download_queue):
            plan_name = plan_url.rstrip("/").split("/")[-1]
            total = len(fits_list)
            present = sum(
                1
                for u in fits_list
                if os.path.exists(
                    os.path.join(self.raw_data_dir, plan_name, u.split("/")[-1])
                )
            )
            if total > 0 and present == total:
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

        # En dry‑run : **ne pas** modifier l'état
        if self.dry_run:
            self._say("\n[Dry‑run] État non modifié (aucun plan marqué complété).")
            self._log("SUMMARY dry_run=True")
            self._say(f"\nLog de session écrit : {self.session_log_path}")
            return

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
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Téléchargements parallèles (1 = séquentiel).",
    )
    parser.add_argument(
        "--scrape-workers",
        type=int,
        default=1,
        help="Parallélisme pour le scrape des pages plan.",
    )
    parser.add_argument(
        "--per-plan",
        type=int,
        default=None,
        help="Plafond de fichiers par plan (après anti‑doublon).",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Valider rapidement les .fits.gz téléchargés (header FITS).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Ne rien écrire : montre seulement ce qui serait fait.",
    )
    parser.add_argument(
        "--auto-throttle",
        action="store_true",
        help="Ajuster automatiquement le nombre de workers en fonction du réseau/serveur.",
    )
    parser.add_argument(
        "--min-workers",
        type=int,
        default=None,
        help="Borne basse pour l’auto-throttle (défaut: 4).",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Borne haute pour l’auto-throttle (défaut: =workers).",
    )
    parser.add_argument(
        "--target-lat",
        type=float,
        default=0.5,
        help="Latence médiane visée (s). Si au‑dessus → on réduit.",
    )
    return parser.parse_args()


def main(argv: Optional[list[str]] = None) -> int:
    """Point d’entrée *programmable* (utile pour tests/CI)."""
    args = _parse_args() if argv is None else _parse_args()
    dl = SmartDownloader(
        limit_plans=args.limit,
        max_spectra=args.max_spectres,
        timeout=args.http_timeout if hasattr(args, "http_timeout") else 60,
        delay_between=args.delay,
        retries=args.retries if hasattr(args, "retries") else 3,
        progress=(args.progress or sys.stdout.isatty()),
        log_to=args.log_to,
        append=args.append,
        chunk_size=args.chunk,
        workers=args.workers,
        scrape_workers=args.scrape_workers,
        per_plan=args.per_plan,
        validate=args.validate,
        dry_run=args.dry_run,
        auto_throttle=args.auto_throttle,
        min_workers=args.min_workers,
        max_workers=args.max_workers,
        target_latency=args.target_lat,
    )
    dl.run()
    dl._say(
        f"\n--- Session terminée : {dl.total_downloaded_this_session} spectre(s) "
        f"téléchargé(s) sur {len(dl.plans_to_process)} plan(s). ---"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
