import os
import pandas as pd
import random

class DatasetBuilder:
    """
    Gère la constitution de lots d'entraînement en s'assurant
    de ne jamais réutiliser un spectre déjà vu.
    """
    def __init__(self, catalog_dir="../data/catalog/", raw_data_dir="../data/raw/"):
        self.catalog_dir = catalog_dir
        self.raw_data_dir = raw_data_dir
        self.trained_log_path = os.path.join(self.catalog_dir, "trained_spectra.csv")

    def _list_available_fits(self):
        """
        Scanne le dossier raw/ et retourne une liste de tous les chemins relatifs
        des fichiers .fits.gz disponibles.
        """
        available_files = []
        for root, _, files in os.walk(self.raw_data_dir):
            for f in files:
                if f.endswith(".fits.gz"):
                    # On stocke le chemin relatif par rapport à 'raw_data_dir'
                    # ex: 'GAC_105N29_B1/spec-55863-GAC_105N29_B1_sp01-001.fits.gz'
                    rel_path = os.path.relpath(os.path.join(root, f), self.raw_data_dir)
                    available_files.append(rel_path.replace('\\', '/')) # Normalise les slashes pour tous les OS
        return available_files

    def _load_trained_log(self):
        """Charge la liste des spectres déjà utilisés depuis le log."""
        if os.path.exists(self.trained_log_path):
            df = pd.read_csv(self.trained_log_path)
            return set(df["file_path"].tolist())
        return set()

    def get_new_training_batch(self, batch_size=500, strategy="random"):
        """
        Sélectionne un nouveau lot de spectres jamais utilisés.
        
        :param batch_size: Le nombre de spectres désiré.
        :param strategy: 'random' pour un échantillon aléatoire, 'first' pour les premiers disponibles.
        :return: Une liste de chemins de fichiers relatifs prêts pour le pipeline.
        """
        print("--- Constitution d'un nouveau lot d'entraînement ---")
        all_available_fits = self._list_available_fits()
        already_trained_fits = self._load_trained_log()
        
        print(f"  > {len(all_available_fits)} spectres trouvés dans '{self.raw_data_dir}'")
        print(f"  > {len(already_trained_fits)} spectres déjà utilisés dans des entraînements précédents.")
        
        # On ne garde que les fichiers qui n'ont jamais été utilisés
        new_fits_to_use = [f for f in all_available_fits if f not in already_trained_fits]
        
        print(f"  > {len(new_fits_to_use)} spectres nouveaux et disponibles pour l'entraînement.")

        if not new_fits_to_use:
            print("  > Aucun nouveau spectre à entraîner. Arrêt.")
            return []

        # Appliquer la stratégie de sélection et la limite de taille
        if len(new_fits_to_use) < batch_size:
            print(f"  > Avertissement : Moins de spectres disponibles ({len(new_fits_to_use)}) que demandé ({batch_size}).")
            batch_size = len(new_fits_to_use)
        
        if strategy == "random":
            selected_batch = random.sample(new_fits_to_use, batch_size)
            print(f"  > Sélection d'un échantillon aléatoire de {batch_size} spectres.")
        else: # 'first'
            selected_batch = new_fits_to_use[:batch_size]
            print(f"  > Sélection des {batch_size} premiers spectres disponibles.")
            
        return selected_batch

    def update_trained_log(self, newly_trained_files):
        """
        Met à jour le journal avec la liste des fichiers qui viennent d'être utilisés,
        en évitant les doublons.
        """
        if not newly_trained_files:
            return

        # Charger le log existant
        if os.path.exists(self.trained_log_path):
            df_existing = pd.read_csv(self.trained_log_path)
            existing_files = set(df_existing["file_path"].tolist())
        else:
            existing_files = set()

        # Filtrer les nouveaux fichiers déjà présents
        truly_new_files = [f for f in newly_trained_files if f not in existing_files]

        if not truly_new_files:
            print("\n  > Aucun nouveau spectre à ajouter : tous étaient déjà dans le log.")
            return

        df_new = pd.DataFrame({"file_path": truly_new_files})
        df_new.to_csv(
            self.trained_log_path,
            mode="a",
            index=False,
            header=not os.path.exists(self.trained_log_path)
        )
        print(f"\n  > {len(truly_new_files)} nouveaux spectres ajoutés au journal '{self.trained_log_path}'.")
