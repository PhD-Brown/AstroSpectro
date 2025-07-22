# dr5_downloader.py (Version 2.0 - Smart & Resumable)

import os
import requests
import pandas as pd
import argparse
from tqdm import tqdm
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from time import sleep
from itertools import zip_longest

class SmartDownloader:
    """
    Un downloader intelligent pour les spectres LAMOST DR5 qui implémente une stratégie
    de téléchargement en round-robin, avec des limites et une gestion d'état.
    """
    def __init__(self, limit_plans, max_spectra):
        # --- Configuration des chemins ---
        self.catalog_dir = "../data/catalog/"
        self.raw_data_dir = "../data/raw/"
        self.valid_urls_file = os.path.join(self.catalog_dir, "valid_plan_urls.csv")
        self.downloaded_log_file = os.path.join(self.catalog_dir, "downloaded_plans.csv")
        
        # --- Paramètres de la session de téléchargement ---
        self.limit_plans = limit_plans
        self.max_spectra = max_spectra
        self.total_downloaded_this_session = 0
        
        # --- Initialisation de l'état ---
        self.plans_to_process = self._load_state()
        self.download_queue = []

        os.makedirs(self.raw_data_dir, exist_ok=True)
        print("--- Initialisation du Smart Downloader ---")
        print(f"  > Plans à traiter : {len(self.plans_to_process)}")
        print(f"  > Limite de plans pour cette session : {self.limit_plans or 'Aucune'}")
        print(f"  > Limite totale de spectres à télécharger : {self.max_spectra or 'Aucune'}")

    def _load_state(self):
        """Charge les URL des plans et filtre ceux déjà complétés."""
        try:
            df_valid = pd.read_csv(self.valid_urls_file)
        except FileNotFoundError:
            print(f"ERREUR : Le fichier des URL valides '{self.valid_urls_file}' n'a pas été trouvé.")
            return []

        already_completed = set()
        if os.path.exists(self.downloaded_log_file):
            df_completed = pd.read_csv(self.downloaded_log_file)
            already_completed = set(df_completed["url"].tolist())
            print(f"  > {len(already_completed)} plan(s) déjà complété(s) et ignoré(s).")
        
        # On ne garde que les plans qui ne sont pas déjà marqués comme complets
        df_remaining = df_valid[~df_valid["url"].isin(already_completed)]
        
        # Appliquer la limite --limit pour la session actuelle
        if self.limit_plans:
            return df_remaining.head(self.limit_plans)["url"].tolist()
        return df_remaining["url"].tolist()

    def _build_download_queue(self):
        """
        Étape 1 : Scrape les pages des plans pour construire la liste de tous les
        fichiers FITS à télécharger, sans encore les télécharger.
        """
        print("\n--- [Phase 1/3] Construction de la file de téléchargement ---")
        for i, plan_url in enumerate(self.plans_to_process):
            plan_name = plan_url.rstrip('/').split('/')[-1]
            print(f"  > [{i+1}/{len(self.plans_to_process)}] Scraping du plan : {plan_name}")
            try:
                resp = requests.get(plan_url, timeout=20)
                resp.raise_for_status()
                soup = BeautifulSoup(resp.text, "html.parser")
                
                plan_fits_urls = []
                for link in soup.find_all("a"):
                    href = link.get("href", "")
                    if href.endswith(".fits.gz"):
                        file_url = urljoin(plan_url, href)
                        plan_fits_urls.append(file_url)
                
                if plan_fits_urls:
                    self.download_queue.append(plan_fits_urls)
                else:
                    print(f"    -> Avertissement : Aucun fichier .fits.gz trouvé pour le plan {plan_name}")
                
            except requests.RequestException as e:
                print(f"    -> ERREUR : Impossible d'accéder au plan {plan_url}. Il sera ignoré. ({e})")
        print("  > File de téléchargement construite avec succès.")

    def run_download(self):
        """
        Étape 2 : Télécharge les fichiers en round-robin avec une barre de progression globale.
        """
        if not self.download_queue:
            print("\nFile de téléchargement vide. Rien à faire.")
            return

        print("\n--- [Phase 2/3] Calcul des spectres à télécharger ---")
        
        # On calcule d'abord le nombre total de fichiers qui manquent VRAIMENT.
        # Cela nous permettra d'initialiser tqdm avec le bon total.
        files_to_download_queue = []
        for plan_fits_urls in self.download_queue:
            for file_url in plan_fits_urls:
                plan_name = file_url.split('/')[-2]
                filename = file_url.split('/')[-1]
                dest_dir = os.path.join(self.raw_data_dir, plan_name)
                dest_path = os.path.join(dest_dir, filename)
                if not os.path.exists(dest_path):
                    files_to_download_queue.append(file_url)
        
        total_to_download = len(files_to_download_queue)
        
        # Appliquer la limite --max-spectres au total calculé
        if self.max_spectra is not None:
            total_to_download = min(total_to_download, self.max_spectra)

        if total_to_download == 0:
            print("  > Tous les spectres cibles sont déjà présents localement.")
            return
            
        print(f"  > {total_to_download} nouveaux spectres à télécharger.")
        print("\n--- [Phase 2/3] Démarrage du téléchargement en Round-Robin ---")

        # Initialisation de la barre de progression avec le total calculé
        pbar = tqdm(total=total_to_download, desc="Téléchargement", unit="spectre")

        # On garde la logique de round-robin
        for file_urls_in_round in zip_longest(*self.download_queue):
            for file_url in file_urls_in_round:
                if file_url is None:
                    continue

                # On arrête si la limite de la session est atteinte
                if self.max_spectra and self.total_downloaded_this_session >= self.max_spectra:
                    pbar.close()
                    print(f"\nLimite de {self.max_spectra} spectres atteinte. Arrêt du téléchargement.")
                    return

                plan_name = file_url.split('/')[-2]
                filename = file_url.split('/')[-1]
                dest_dir = os.path.join(self.raw_data_dir, plan_name)
                dest_path = os.path.join(dest_dir, filename)

                if os.path.exists(dest_path):
                    continue # On ne fait rien, on passe au suivant

                # Si on arrive ici, c'est un fichier à télécharger
                # pbar.set_description(f"Téléchargement de {filename}") # Optionnel: met à jour le texte de la barre
                try:
                    with requests.get(file_url, stream=True, timeout=60) as r:
                        r.raise_for_status()
                        with open(dest_path, "wb") as f:
                            for chunk in r.iter_content(chunk_size=8192):
                                f.write(chunk)
                    self.total_downloaded_this_session += 1
                    pbar.update(1) # On fait avancer la barre de 1
                except requests.RequestException as e:
                    # On peut logger l'erreur sans arrêter la barre de progression
                    pbar.write(f"    -> ERREUR lors du téléchargement de {filename}: {e}")
                
                sleep(0.3)
        
        pbar.close() # On ferme proprement la barre de progression à la fin

    def _update_state(self):
        """
        Étape 3 : Vérifie quels plans sont maintenant complets et met à jour le log.
        """
        print("\n--- [Phase 3/3] Mise à jour de l'état des plans ---")
        newly_completed_plans = []
        for plan_url, fits_list in zip(self.plans_to_process, self.download_queue):
            plan_name = plan_url.rstrip('/').split('/')[-1]
            dest_dir = os.path.join(self.raw_data_dir, plan_name)
            
            # Vérifier si tous les fichiers de ce plan existent maintenant sur le disque
            all_files_exist = True
            for fits_url in fits_list:
                filename = fits_url.split('/')[-1]
                if not os.path.exists(os.path.join(dest_dir, filename)):
                    all_files_exist = False
                    break # Pas la peine de vérifier les autres
            
            if all_files_exist:
                print(f"  > Plan complété : {plan_name}")
                newly_completed_plans.append(plan_url)
        
        if newly_completed_plans:
            df_new = pd.DataFrame({"url": newly_completed_plans})
            df_new.to_csv(self.downloaded_log_file, mode="a", index=False, header=not os.path.exists(self.downloaded_log_file))
            print(f"\n  > {len(newly_completed_plans)} plan(s) marqué(s) comme complets dans '{self.downloaded_log_file}'.")
        else:
            print("  > Aucun nouveau plan complété durant cette session.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smart Downloader pour spectres LAMOST DR5")
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Nombre maximal de NOUVEAUX plans à traiter (ex: 5 pour un test)."
    )
    parser.add_argument(
        "--max-spectres", type=int, default=None,
        help="Arrête le téléchargement après avoir obtenu ce nombre total de spectres."
    )
    args = parser.parse_args()

    # --- Lancement du pipeline ---
    downloader = SmartDownloader(limit_plans=args.limit, max_spectra=args.max_spectres)
    downloader._build_download_queue()
    downloader.run_download()
    downloader._update_state()

    print(
        f"\n--- Session terminée : "
        f"{downloader.total_downloaded_this_session} spectres téléchargés "
        f"dans {len(downloader.plans_to_process)} plan(s). ---"
    )
