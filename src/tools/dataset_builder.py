"""AstroSpectro — Gestion des lots d’entraînement (DatasetBuilder)

Ce module fournit une petite brique utilitaire chargée de constituer des lots
de spectres à entraîner **sans jamais réutiliser** des fichiers déjà vus.

Rôles principaux
----------------
1) Scanner l’arborescence `raw/` pour lister tous les fichiers FITS (.fits.gz).
2) Maintenir un *journal* CSV (`catalog/trained_spectra.csv`) des spectres déjà
   utilisés par des entraînements précédents.
3) Sélectionner un nouveau lot de chemins **jamais journalisés**, selon une
   stratégie simple (“random” ou “first”).

Conventions & I/O
-----------------
- Les chemins retournés sont **relatifs** à `raw_data_dir` et **normalisés**
  avec des “/” (compatibles tous OS).
- Le journal est stocké dans `catalog_dir/trained_spectra.csv` avec une seule
  colonne `file_path`.
- Ce module **ne lit pas** les FITS : il renvoie uniquement les chemins.

API publique (méthodes)
-----------------------
- get_new_training_batch(batch_size, strategy) -> list[str]
- update_trained_log(newly_trained_files) -> None

Exemple minimal
---------------
>>> db = DatasetBuilder(catalog_dir="data/catalog", raw_data_dir="data/raw")
>>> batch = db.get_new_training_batch(batch_size=500, strategy="random")
>>> # ... lancer le pipeline sur `batch`
>>> db.update_trained_log(batch)  # journalise les fichiers utilisés
"""

from __future__ import annotations

import os
import random
from typing import List, Literal, Set

import pandas as pd


class DatasetBuilder:
    """Petit gestionnaire de lots d’entraînement.

    Cette classe encapsule la logique pour :
    - lister les FITS disponibles,
    - ignorer ceux déjà utilisés (journal `trained_spectra.csv`),
    - renvoyer des listes de chemins prêts pour le pipeline.

    Attributes
    ----------
    catalog_dir : str
        Répertoire du catalogue (contiendra `trained_spectra.csv`).
    raw_data_dir : str
        Racine des données brutes (sous-dossiers avec .fits.gz).
    trained_log_path : str
        Chemin absolu vers le journal CSV des spectres déjà entraînés.
    """

    def __init__(
        self,
        catalog_dir: str = "../data/catalog/",
        raw_data_dir: str = "../data/raw/",
    ) -> None:
        self.catalog_dir = catalog_dir
        self.raw_data_dir = raw_data_dir
        self.trained_log_path = os.path.join(self.catalog_dir, "trained_spectra.csv")

    # ---------------------------------------------------------------------
    # Helpers internes
    # ---------------------------------------------------------------------

    def _list_available_fits(self) -> List[str]:
        """Liste *tous* les fichiers .fits.gz disponibles dans `raw_data_dir`.

        Les chemins renvoyés sont **relatifs** à `raw_data_dir` et normalisés
        avec des “/” (utile pour rester invariant entre Windows/Linux).

        Returns
        -------
        list[str]
            Chemins relatifs normalisés, ex.:
            ``GAC_105N29_B1/spec-55863-GAC_105N29_B1_sp01-001.fits.gz``.
        """
        available_files: List[str] = []

        # On marche toute l’arborescence sous raw_data_dir
        for root, _dirs, files in os.walk(self.raw_data_dir):
            for fname in files:
                if fname.endswith(".fits.gz"):
                    # Chemin absolu -> chemin relatif -> normalisation des slashes
                    full = os.path.join(root, fname)
                    rel = os.path.relpath(full, self.raw_data_dir).replace("\\", "/")
                    available_files.append(rel)

        return available_files

    def _load_trained_log(self) -> Set[str]:
        """Charge l’ensemble des chemins déjà journalisés.

        Le CSV est attendu avec une colonne `file_path`. Les valeurs nulles sont
        ignorées. Un fichier vide ou mal formé est traité de façon robuste.

        Returns
        -------
        set[str]
            Ensemble de chemins **relatifs** déjà utilisés.
        """
        if not os.path.exists(self.trained_log_path):
            return set()

        # 1) Lecture robuste : tente ',' sinon auto-détection du séparateur
        try:
            try:
                df = pd.read_csv(self.trained_log_path)
            except Exception:
                df = pd.read_csv(self.trained_log_path, sep=None, engine="python")
        except pd.errors.EmptyDataError:
            return set()

        # 2) Trouver une colonne plausible de chemin
        candidates = [
            "file_path",
            "path",
            "filepath",
            "fits_path",
            "relpath",
            "raw_path",
        ]
        col = next((c for c in candidates if c in df.columns), None)
        if col is None:
            print(
                f"  > AVERTISSEMENT : '{self.trained_log_path}' sans colonne de chemin reconnue ({candidates}). Ignoré."
            )
            return set()

        # 3) Normaliser en **relatif à raw_data_dir** + slashes "/"
        paths = []
        for p in df[col].dropna().astype(str):
            p = p.strip().replace("\\", "/")
            if not p:
                continue
            if os.path.isabs(p):
                try:
                    p = os.path.relpath(p, self.raw_data_dir).replace("\\", "/")
                except Exception:
                    # Si on ne peut pas relativiser (hors raw/), garde tel quel
                    pass
            paths.append(p)

        return set(paths)

    # ---------------------------------------------------------------------
    # API publique
    # ---------------------------------------------------------------------

    def get_new_training_batch(
        self,
        batch_size: int = 500,
        strategy: Literal["random", "first"] = "random",
    ) -> List[str]:
        """Sélectionne un nouveau lot de fichiers **jamais entraînés**.

        Parameters
        ----------
        batch_size : int, default=500
            Nombre de spectres souhaité pour le lot.
        strategy : {'random', 'first'}, default='random'
            - ``'random'`` : échantillon aléatoire dans les fichiers disponibles.
            - ``'first'``  : on prend les `batch_size` premiers (ordre du scan).

        Returns
        -------
        list[str]
            Chemins *relatifs* (normalisés) prêts à être passés au pipeline.
            Peut être une liste vide si aucun nouveau fichier n’est disponible.
        """
        print("--- Constitution d'un nouveau lot d'entraînement ---")

        all_available = self._list_available_fits()
        already_used = self._load_trained_log()

        print(f"  > {len(all_available)} spectres trouvés dans '{self.raw_data_dir}'")
        print(f"  > {len(already_used)} spectres déjà utilisés dans le journal.")

        # Filtre : on ne garde que ce qui n’a jamais été vu
        new_fits = [p for p in all_available if p not in already_used]
        print(f"  > {len(new_fits)} spectres **nouveaux** disponibles.")

        if not new_fits:
            print("  > Aucun nouveau spectre à entraîner. Arrêt.")
            return []

        # Ajuste la taille si nécessaire
        if len(new_fits) < batch_size:
            print(
                f"  > Avertissement : {len(new_fits)} disponibles < batch_size={batch_size}."
            )
            batch_size = len(new_fits)

        # Stratégie de sélection
        if strategy == "random":
            selected = random.sample(new_fits, k=batch_size)
            print(f"  > Sélection aléatoire de {batch_size} spectres.")
        else:  # 'first'
            selected = new_fits[:batch_size]
            print(f"  > Sélection des {batch_size} premiers spectres.")

        return selected

    def update_trained_log(self, newly_trained_files: List[str]) -> None:
        """Ajoute au journal les fichiers qui viennent d’être utilisés.

        Les doublons sont évités ; la colonne utilisée est `file_path`.

        Parameters
        ----------
        newly_trained_files : list[str]
            Chemins *relatifs* (normalisés “/”) des fichiers utilisés par
            l’entraînement qui vient d’être exécuté.
        """
        if not newly_trained_files:
            return

        # Log existant -> ensemble pour filtrer vite
        if os.path.exists(self.trained_log_path):
            try:
                existing_df = pd.read_csv(self.trained_log_path)
                existing = set(existing_df.get("file_path", pd.Series([])).astype(str))
            except pd.errors.EmptyDataError:
                existing = set()
        else:
            # S'assure que le répertoire du log existe
            os.makedirs(self.catalog_dir, exist_ok=True)
            existing = set()

        # On ne journalise que les *vraiment* nouveaux
        truly_new = [p for p in newly_trained_files if p not in existing]
        if not truly_new:
            print("  > Aucun nouvel élément à ajouter (tout déjà journalisé).")
            return

        df_new = pd.DataFrame({"file_path": truly_new})
        df_new.to_csv(
            self.trained_log_path,
            mode="a",
            index=False,
            header=not os.path.exists(self.trained_log_path),
        )
        print(
            f"  > {len(truly_new)} nouveaux spectres ajoutés au journal : "
            f"'{self.trained_log_path}'."
        )
