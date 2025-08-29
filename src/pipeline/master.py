"""AstroSpectro — orchestrateur de pipeline.

Ce module définit la classe :class:`MasterPipeline` qui orchestre un
pipeline complet de traitement et d’apprentissage sur des spectres
astronomiques au format FITS. Le workflow comprend la sélection d’un
lot de spectres, la construction et l’enrichissement d’un catalogue
(option Gaia), le pré‑traitement et l’extraction de caractéristiques,
l’entraînement et l’évaluation d’un classifieur spectral, puis la
journalisation des résultats et des artefacts.

Les méthodes publiques de :class:`MasterPipeline` permettent notamment :

* :meth:`select_batch` — sélectionner un lot de spectres non encore utilisés.
* :meth:`generate_and_enrich_catalog` — générer le catalogue maître et
  l’enrichir éventuellement avec Gaia.
* :meth:`process_data` — exécuter le pipeline de pré‑traitement et
  sauvegarder un CSV de features.
* :meth:`run_training_session` — entraîner et évaluer un modèle tout en
  journalisant les artefacts (modèle, rapports, figures).
* :meth:`run_full_pipeline` — enchaîner toutes les étapes pour un run
  « one‑shot ».
* :meth:`interactive_training_runner` — proposer une interface
  interactive (ipywidgets) pour configurer et lancer des entraînements.

Entrées :
    Les répertoires racine pour les données brutes, le catalogue, les
    features, les modèles et les rapports. Un CSV de features prêt à
    l’emploi peut également être chargé.

Sorties :
    Fichiers CSV de features, modèles sérialisés (et métadonnées),
    figures (matrices de confusion, courbes ROC/PR, etc.) et rapport
    JSON de session, ainsi que la configuration de l’interface.

Dépendances :
    scikit‑learn, xgboost, imbalanced‑learn, astropy, ipywidgets, pandas,
    numpy.
"""

from __future__ import annotations
from typing import Any, Optional
import os
import sys
import json
import time
import shutil
import hashlib
import platform
from pathlib import Path
from datetime import datetime, timezone

import joblib
import pandas as pd
import numpy as np

from IPython.display import display, clear_output, HTML, Image, Markdown

try:
    import ipywidgets as W
except Exception:
    W = None

# --- Imports projet ---
from tools.dataset_builder import DatasetBuilder
from pipeline.processing import ProcessingPipeline
from pipeline.classifier import SpectralClassifier
from tools.generate_catalog_from_fits import generate_catalog_from_fits
from tools.gaia_crossmatcher import enrich_catalog_with_gaia

# --- Imports Scikit-learn ---
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
)

# ================================================================
#  Templates de paramètres (base_params, grid, distributions)
#  Ces dictionnaires définissent les paramètres par défaut et les grilles de
#  recherche pour chaque modèle. Ils sont utilisés pour préremplir les champs
#  correspondants dans l'interface lorsqu'un modèle est sélectionné.
# ================================================================

# Pré-configurations de base (paramètres de départ pour chaque modèle)
PRESET_BASE: dict = {
    "ExtraTrees": {
        "bootstrap": True,
        "class_weight": "balanced_subsample",
    },
    "LogRegOVR": {
        "solver": "lbfgs",
        "C": 1.0,
    },
    "KNN": {
        "n_neighbors": 15,
        "weights": "distance",
    },
    "MLP": {
        "hidden_layer_sizes": [256, 128],
        "activation": "relu",
        "alpha": 0.0001,
        "learning_rate": "adaptive",
        "max_iter": 300,
    },
    "LDA": {
        "solver": "svd",
        "shrinkage": None,
    },
    "QDA": {
        "reg_param": 0.0,
    },
    "CatBoost": {
        "learning_rate": 0.05,
        "depth": 6,
        "l2_leaf_reg": 3.0,
        "auto_class_weights": "Balanced",
    },
    "LightGBM": {
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": -1,
        "bagging_fraction": 0.8,
        "feature_fraction": 0.8,
        "reg_alpha": 0.0,
        "reg_lambda": 0.0,
    },
    "SoftVoting": {},
}

# Grilles de recherche (GridSearchCV)
PRESET_GRID: dict = {
    "ExtraTrees": {
        "clf__max_depth": [None, 10, 20, 30],
        "clf__min_samples_leaf": [1, 2, 4],
        "clf__max_features": ["sqrt", "log2", 0.5],
    },
    "LogRegOVR": {
        "clf__base_estimator__C": [0.1, 0.3, 1, 3, 10],
        "clf__base_estimator__solver": ["lbfgs", "liblinear", "saga"],
    },
    "KNN": {
        "clf__n_neighbors": [5, 9, 15, 25],
        "clf__weights": ["uniform", "distance"],
    },
    "MLP": {
        "clf__hidden_layer_sizes": [[128], [256, 128], [512, 256]],
        "clf__alpha": [0.0001, 0.001, 0.01],
    },
    "LDA": {
        "clf__solver": ["svd", "lsqr", "eigen"],
        "clf__shrinkage": [None, "auto", 0.0, 0.1, 0.3, 0.5],
    },
    "QDA": {
        "clf__reg_param": [0.0, 0.001, 0.01, 0.1, 0.3],
    },
    "CatBoost": {
        "clf__depth": [4, 6, 8],
        "clf__learning_rate": [0.03, 0.05, 0.08],
        "clf__l2_leaf_reg": [1.0, 3.0, 5.0],
    },
    "LightGBM": {
        "clf__num_leaves": [31, 63, 127],
        "clf__max_depth": [-1, 8, 12],
        "clf__bagging_fraction": [0.7, 0.8, 0.9],
        "clf__feature_fraction": [0.7, 0.8, 0.9],
        "clf__learning_rate": [0.03, 0.05, 0.08],
        "clf__reg_lambda": [0.0, 0.5, 1.0],
    },
}

# Distributions pour RandomizedSearchCV
PRESET_DISTS: dict = {
    "ExtraTrees": {
        "clf__min_samples_split": [2, 5, 10],
        "clf__min_samples_leaf": [1, 2, 4],
    },
    "LogRegOVR": {
        "clf__base_estimator__C": [0.1, 0.3, 1, 3, 10],
        "clf__base_estimator__solver": ["lbfgs", "liblinear"],
    },
    "KNN": {
        "clf__n_neighbors": [5, 9, 15, 25, 35],
        "clf__weights": ["uniform", "distance"],
    },
    "MLP": {
        "clf__alpha": [0.0001, 0.0005, 0.001, 0.01],
    },
    "LDA": {
        "clf__solver": ["svd", "lsqr", "eigen"],
        "clf__shrinkage": [None, "auto", 0.0, 0.1, 0.3, 0.5],
    },
    "QDA": {
        "clf__reg_param": [0.0, 0.001, 0.01, 0.1, 0.3],
    },
    "CatBoost": {
        "clf__depth": [4, 6, 8, 10],
        "clf__learning_rate": [0.02, 0.03, 0.05, 0.08],
        "clf__l2_leaf_reg": [1.0, 2.0, 3.0, 5.0],
    },
    "LightGBM": {
        "clf__num_leaves": [31, 63, 95, 127],
        "clf__max_depth": [-1, 6, 8, 12],
        "clf__bagging_fraction": [0.6, 0.7, 0.8, 0.9],
        "clf__feature_fraction": [0.6, 0.7, 0.8, 0.9],
        "clf__learning_rate": [0.02, 0.03, 0.05, 0.08],
        "clf__reg_lambda": [0.0, 0.3, 0.6, 1.0],
    },
}

# -----------------------------------------------------------------------------
# Utilitaires pour le tableau de bord d'entraînement
# -----------------------------------------------------------------------------


def _save_preset(path: str, widgets: dict) -> None:
    """
    Sauvegarde dans un fichier JSON les valeurs des widgets spécifiés.

    Les widgets doivent posséder un attribut ``value`` ; cette fonction
    extrait ces valeurs et les sérialise sous forme de dictionnaire JSON.

    Args:
        path (str): Chemin du fichier JSON où écrire les valeurs.
        widgets (dict): Dictionnaire de widgets ipywidgets. Seuls les
            éléments disposant d’un attribut ``value`` seront sérialisés.

    Side Effects:
        Écrit un fichier JSON à l’emplacement indiqué.
    """
    payload = {k: w.value for k, w in widgets.items() if hasattr(w, "value")}
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _load_preset(path: str, widgets: dict) -> None:
    """
    Charge un préréglage depuis un fichier JSON et met à jour les widgets.

    Cette fonction lit un fichier JSON généré par :func:`_save_preset` et
    affecte les valeurs chargées aux widgets correspondants. Les clés
    absentes de ``widgets`` ou les widgets sans attribut ``value`` sont
    ignorés.

    Args:
        path (str): Chemin vers le fichier JSON à charger.
        widgets (dict): Dictionnaire de widgets à mettre à jour ; seules
            les clés présentes dans ce dictionnaire seront modifiées.

    Side Effects:
        Met à jour la propriété ``value`` des widgets concernés.
    """
    if not Path(path).exists():
        return
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    for k, v in payload.items():
        if k in widgets and hasattr(widgets[k], "value"):
            try:
                widgets[k].value = v
            except Exception:
                pass


def _json_default(o):
    """
    Convertit des objets non sérialisables en types natifs pour JSON.

    Cette fonction sert de callback ``default`` pour :func:`json.dump` et
    :func:`json.dumps`. Elle gère notamment les scalaires NumPy et les
    tableaux NumPy en les convertissant en types Python natifs. Pour les
    objets de bibliothèques tierces (par exemple scikit‑learn), elle
    renvoie le nom de la classe comme représentation courte.

    Args:
        o: Objet à convertir en type sérialisable.

    Returns:
        Un objet compatible avec la sérialisation JSON : valeur numérique,
        liste, nom de classe ou chaîne.
    """
    # numpy -> types natifs
    try:
        import numpy as _np

        if isinstance(o, _np.generic):
            return o.item()
        if hasattr(o, "tolist"):
            return o.tolist()
    except Exception:
        pass
    # sklearn / objets divers -> représentation courte
    try:
        name = getattr(o, "__name__", None) or o.__class__.__name__
        return name
    except Exception:
        return str(o)


def _load_runs_table(reports_root: str):
    """
    Lit tous les rapports de session présents dans un dossier de rapports.

    La fonction parcourt récursivement ``reports_root`` à la recherche
    de fichiers ``session_report_*.json``. Pour chaque rapport trouvé,
    elle extrait certaines métriques clés telles que la précision, la
    balanced accuracy, la F1 macro, l’AUC macro, ainsi que le nombre de
    features utilisées. Ces informations sont agrégées dans un
    :class:`pandas.DataFrame` qui est ensuite trié par performance.

    Args:
        reports_root (str): Chemin vers le répertoire contenant les rapports
            JSON des sessions.

    Returns:
        pandas.DataFrame | None: Un DataFrame résumant les runs trié par
            ordre décroissant de performance, ou ``None`` si aucun rapport
            n’est trouvé.
    """
    rows = []
    for js in Path(reports_root).rglob("session_report_*.json"):
        try:
            meta = json.loads(js.read_text(encoding="utf-8"))
            # Récupère l'exactitude et la balanced accuracy depuis les différentes structures
            acc = meta.get("accuracy") or (meta.get("test_metrics") or {}).get(
                "accuracy"
            )
            bal = meta.get("balanced_accuracy") or (meta.get("test_metrics") or {}).get(
                "balanced_accuracy"
            )
            # Récupère le dictionnaire ROC AUC (avec macro/micro et classes) s'il existe
            roc = meta.get("roc_auc") or {}
            # Récupère les Average Precision par classe (si calculées)
            ap = meta.get("avg_precision") or {}
            # Calcule la macro-AP si possible
            ap_macro = None
            try:
                import numpy as _np

                if ap:
                    ap_macro = float(_np.mean(list(ap.values())))
            except Exception:
                ap_macro = None
            # F1 macro depuis le classification_report
            f1_macro = None
            try:
                rep = meta.get("classification_report") or {}
                if isinstance(rep, dict):
                    f1_macro = (rep.get("macro avg") or {}).get("f1-score")
            except Exception:
                f1_macro = None
            features = (
                meta.get("n_candidate_features")
                or (
                    len((meta.get("feature_columns") or []))
                    if isinstance(meta.get("feature_columns"), list)
                    else None
                )
                or (
                    len((meta.get("selected_features") or []))
                    if isinstance(meta.get("selected_features"), list)
                    else None
                )
            )
            rows.append(
                {
                    "ts": meta.get("saved_at_utc")
                    or meta.get("session_id")
                    or Path(js).parent.name,
                    "exp": meta.get("exp_name"),
                    "model": meta.get("model_type"),
                    "features": features,
                    "acc": acc,
                    "bal_acc": bal,
                    "f1_macro": f1_macro,
                    "auc_macro": (roc.get("macro") if isinstance(roc, dict) else None),
                    "ap_macro": ap_macro,
                    "run_dir": str(Path(js).parent),
                }
            )
        except Exception:
            pass
    if rows:
        import pandas as pd

        # Classements par ordre décroissant : balanced accuracy puis F1 macro
        df = pd.DataFrame(rows)
        # Tri en ignorant les valeurs manquantes
        try:
            df_sorted = df.sort_values(
                ["bal_acc", "f1_macro"], ascending=False, na_position="last"
            )
        except Exception:
            df_sorted = df
        return df_sorted
    else:
        return None


class MasterPipeline:
    """
    Orchestrateur principal du pipeline d’entraînement AstroSpectro.

    Cette classe encapsule un workflow reproductible qui va de la sélection
    d’un lot de fichiers FITS jusqu’à l’entraînement et l’évaluation d’un
    classifieur spectral. Les principales étapes comprennent la construction
    et l’enrichissement d’un catalogue à partir des FITS sélectionnés, le
    prétraitement et l’extraction de caractéristiques, l’entraînement et
    l’évaluation d’un modèle supervisé, et la journalisation complète des
    artefacts de session (figures, rapports, modèles). Une interface
    interactive basée sur ipywidgets est fournie pour piloter les runs
    depuis Jupyter ou VS Code.

    Attributes:
        raw_data_dir (str): Dossier contenant les FITS bruts.
        catalog_dir (str): Répertoire de sortie pour le catalogue intermédiaire.
        processed_dir (str): Répertoire de sortie des CSV de features générés.
        models_dir (str): Répertoire de sauvegarde des modèles entraînés.
        reports_dir (str): Répertoire des rapports et figures par session.
        builder (DatasetBuilder): Sélecteur de lots et gestionnaire de journalisation.
        current_batch (list[str]): Liste des chemins relatifs du lot courant.
        master_catalog_df (pandas.DataFrame): Catalogue principal, éventuellement
            enrichi.
        features_df (pandas.DataFrame): Dernier DataFrame de features chargé ou généré.
        master_catalog_path (str): Chemin vers le catalogue maître temporaire.
        gaia_catalog_path (str): Chemin vers le catalogue enrichi via Gaia.
        last_features_path (str | None): Chemin du dernier CSV de features chargé.
        _runs_out (ipywidgets.Output | contextlib.AbstractContextManager): Sortie
            utilisée pour afficher la table des runs dans l’interface.
    """

    def __init__(
        self,
        raw_data_dir: str,
        catalog_dir: str,
        processed_dir: str,
        models_dir: str,
        reports_dir: str,
    ) -> None:
        """
        Initialise l’orchestrateur et prépare l’état interne du pipeline.

        Cette méthode assure l’existence des répertoires de travail,
        instancie un :class:`DatasetBuilder` pointant vers ``raw_data_dir`` et
        ``catalog_dir``, et initialise les attributs du pipeline (lot courant,
        chemins de catalogue temporaire, DataFrames vides). Si ipywidgets est
        disponible, un conteneur de sortie est créé pour l’explorateur de runs.

        Args:
            raw_data_dir (str): Répertoire des fichiers FITS bruts (données source).
            catalog_dir (str): Répertoire de sortie pour le catalogue intermédiaire.
            processed_dir (str): Répertoire de sortie des CSV de features.
            models_dir (str): Répertoire de sortie des modèles sérialisés.
            reports_dir (str): Répertoire de sortie des rapports JSON et figures.

        Raises:
            OSError: En cas d’échec de création d’un des répertoires requis.
        """
        self.raw_data_dir = raw_data_dir
        self.catalog_dir = catalog_dir
        self.processed_dir = processed_dir
        self.models_dir = models_dir
        self.reports_dir = reports_dir

        self.builder = DatasetBuilder(
            raw_data_dir=self.raw_data_dir, catalog_dir=self.catalog_dir
        )

        # État
        self.current_batch: list[str] = []
        self.master_catalog_df: pd.DataFrame = pd.DataFrame()
        self.features_df: pd.DataFrame = pd.DataFrame()

        # Chemins
        self.master_catalog_path = os.path.join(
            self.catalog_dir, "master_catalog_temp.csv"
        )
        self.gaia_catalog_path = os.path.join(
            self.catalog_dir, "master_catalog_gaia.csv"
        )
        self.last_features_path: Optional[str] = None

        # Crée les répertoires au besoin
        for path in [
            self.catalog_dir,
            self.processed_dir,
            self.models_dir,
            self.reports_dir,
        ]:
            os.makedirs(path, exist_ok=True)

        # Output container for the run explorer (prevent stacking) and last run timestamp
        # Utilise ipywidgets.Output si disponible, sinon un nullcontext pour compatibilité hors Jupyter
        if W is not None:
            self._runs_out = W.Output(
                layout=W.Layout(
                    border="1px solid #333", max_height="360px", overflow="auto"
                )
            )
        else:
            # fall-back: simple context manager qui ne fait rien
            from contextlib import nullcontext

            self._runs_out = nullcontext()
        self._last_run_ts: Optional[str] = None

    # --------------------- API publique (Notebook) ---------------------

    def select_batch(
        self, batch_size: int = 500, strategy: str = "random"
    ) -> list[str]:
        """
        Sélectionne un lot de spectres à traiter et le mémorise.

        Cette méthode interroge le :class:`DatasetBuilder` associé pour
        obtenir un ensemble de chemins FITS relatifs au répertoire des
        données brutes. Les spectres déjà marqués comme entraînés dans
        ``trained_spectra.csv`` sont automatiquement exclus afin d’éviter
        les doublons entre sessions.

        Args:
            batch_size (int, optional): Nombre de spectres à inclure dans le lot.
            strategy (str, optional): Stratégie de sélection (p. ex. ``"random"``).
                Reportez‑vous à :class:`DatasetBuilder` pour les stratégies supportées.

        Returns:
            list[str]: Liste des chemins FITS relatifs qui constituent le lot
            courant.

        Side Effects:
            Met à jour :attr:`current_batch` avec les chemins sélectionnés.
        """
        print("\n=== ÉTAPE 1 : SÉLECTION D'UN NOUVEAU LOT ===")
        self.current_batch = self.builder.get_new_training_batch(
            batch_size=batch_size, strategy=strategy
        )
        return self.current_batch

    def generate_and_enrich_catalog(
        self, enrich_gaia: bool = False, **gaia_kwargs: Any
    ) -> None:
        """
        Construit le catalogue maître à partir du lot courant et l’enrichit optionnellement.

        La méthode lit les fichiers FITS du lot sélectionné, génère un catalogue
        maître intermédiaire et, si ``enrich_gaia`` est vrai, effectue un
        croisement avec le catalogue Gaia en utilisant les paramètres fournis.

        Args:
            enrich_gaia (bool, optional): Active l’enrichissement via Gaia.
            **gaia_kwargs: Paramètres additionnels à transmettre à
                :func:`enrich_catalog_with_gaia` (p. ex. rayon de recherche,
                colonnes à joindre).

        Side Effects:
            Met à jour :attr:`master_catalog_df` avec le catalogue généré.
            Écrit un fichier CSV ``master_catalog_temp.csv`` ou ``master_catalog_gaia.csv``
            dans ``catalog_dir`` selon que l’enrichissement Gaia est activé.
        """
        print("\n=== ÉTAPE 2 : GÉNÉRATION ET ENRICHISSEMENT DU CATALOGUE ===")
        if not self.current_batch:
            print(
                "  > Erreur : Aucun lot sélectionné. Veuillez d'abord exécuter `select_batch`."
            )
            return

        full_paths = [
            os.path.join(self.raw_data_dir, path) for path in self.current_batch
        ]
        local_df = generate_catalog_from_fits(
            full_paths, self.master_catalog_path, return_df=True
        )
        print(f"  > Catalogue local de {len(local_df)} spectres créé.")

        if enrich_gaia:
            enriched_df, stats = enrich_catalog_with_gaia(
                input_catalog_df=local_df,
                output_catalog_path=self.gaia_catalog_path,
                overwrite=True,
                **gaia_kwargs,
            )
            self.master_catalog_df = enriched_df
            print(
                f"  > Gaia : {stats.get('matched', 0)}/{stats.get('total', 0)} objets appariés."
            )
        else:
            self.master_catalog_df = local_df

    def process_data(self) -> Optional[pd.DataFrame]:
        """
        Exécute le pipeline de pré‑traitement et génère un CSV de features.

        Cette étape construit un :class:`ProcessingPipeline` à partir du
        catalogue courant, exécute les opérations de nettoyage, de
        normalisation et d’extraction de caractéristiques, puis enregistre
        le DataFrame résultant dans un fichier ``features_<horodatage>.csv``
        sous ``processed_dir``. Les colonnes cibles dérivées sont ajoutées via
        :meth:`_ensure_derived_targets` et le DataFrame final est stocké dans
        :attr:`features_df`.

        Returns:
            pandas.DataFrame | None: Le DataFrame de features généré, ou
            ``None`` si le catalogue est vide.

        Side Effects:
            Met à jour :attr:`features_df` et :attr:`last_features_path`.
            Écrit un fichier ``features_*.csv`` dans ``processed_dir``.
        """
        print("\n=== ÉTAPE 3 : TRAITEMENT DES DONNÉES ET EXTRACTION DES FEATURES ===")
        if self.master_catalog_df.empty:
            print(
                "  > Erreur : Le catalogue est vide. Lance `generate_and_enrich_catalog` d’abord."
            )
            return None

        pipeline = ProcessingPipeline(self.raw_data_dir, self.master_catalog_df)
        self.features_df = pipeline.run(self.current_batch)
        self.features_df = self._ensure_derived_targets(self.features_df)

        if not self.features_df.empty:
            ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            fname = f"features_{ts}.csv"
            self.last_features_path = os.path.join(self.processed_dir, fname)
            self.features_df.to_csv(self.last_features_path, index=False)
            print(
                f"\n  > Dataset de features sauvegardé dans : {os.path.basename(self.last_features_path)}"
            )
            return self.features_df

        return None

    # --- Utilitaires features existants ---------------------------------

    def _list_feature_files(self, limit: int = 50) -> list[str]:
        """
        Retourne la liste triée des fichiers CSV de features disponibles.

        Args:
            limit (int, optional): Nombre maximum de fichiers à retourner. Les
                fichiers sont triés par date de modification décroissante.

        Returns:
            list[str]: Chemins des fichiers ``features_*.csv`` les plus récents.
        """
        base = Path(self.processed_dir)
        if not base.exists():
            return []
        files = sorted(
            base.glob("features_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True
        )
        return [str(p) for p in files[:limit]]

    def load_features_from_csv(
        self, path: str | None = None, use_last: bool = False
    ) -> pd.DataFrame | None:
        """
        Charge un fichier CSV de features existant et reconstitue les cibles dérivées.

        Si ``use_last`` est vrai, la méthode recherche le fichier
        ``features_*.csv`` le plus récent dans ``processed_dir`` et le charge.
        Sinon, elle charge le fichier indiqué par ``path``. Après chargement,
        :meth:`_ensure_derived_targets` est appelée pour garantir la présence
        des colonnes cibles dérivées.

        Args:
            path (str | None, optional): Chemin explicite vers le CSV de features
                à charger. Ignoré si ``use_last`` est ``True``.
            use_last (bool, optional): S’il est ``True``, charge automatiquement
                le dernier fichier ``features_*.csv`` trouvé dans ``processed_dir``.

        Returns:
            pandas.DataFrame | None: Le DataFrame de features chargé, ou ``None`` en cas d’échec.

        Side Effects:
            Met à jour :attr:`features_df` et :attr:`last_features_path`.
            Affiche un résumé des colonnes catégorielles disponibles.
        """
        try:
            if use_last:
                cand = self._list_feature_files(limit=1)
                if not cand:
                    print(
                        "Aucun fichier 'features_*.csv' trouvé dans:",
                        self.processed_dir,
                    )
                    return None
                path = cand[0]
            if not path:
                print("Spécifie un chemin CSV ou passe use_last=True.")
                return None
            df = pd.read_csv(path)
            df = self._ensure_derived_targets(df)

            # Cast utile pour les cibles (évite surprises de types)
            for col in (
                "main_class",
                "sub_class_top25",
                "sub_class_bins",
                "subclass",
                "class",
            ):
                if col in df.columns and df[col].dtype == "object":
                    df[col] = df[col].astype("category")

            self.features_df = df
            self.last_features_path = path
            print(
                f"Features chargées depuis: {path}  ({len(df):,} lignes, {df.select_dtypes(include=['number']).shape[1]} num. features)"
            )
            # Petit hint sur les cibles dispo
            cat_cols = [
                c for c in df.columns if str(df[c].dtype) in ("category", "object")
            ]
            if cat_cols:
                print(
                    "Colonnes catégorielles (cibles potentielles):",
                    ", ".join(cat_cols[:12]),
                    "…",
                )
            return df
        except Exception as e:
            print(f"(erreur) Impossible de charger {path}: {e}")
            return None

    def _ensure_derived_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Crée les cibles dérivées standardisées si elles sont absentes dans un DataFrame.

        Cette fonction ajoute jusqu’à trois colonnes cibles dérivées :

        * ``main_class`` : la lettre spectrale (O, B, A, F, G, K, M, …) extraite
          de la colonne ``subclass`` ou ``class``.
        * ``sub_class_top25`` : les 25 sous‑classes les plus fréquentes dans
          ``subclass`` ; toutes les autres sont regroupées sous ``"Other"``.
        * ``sub_class_bins`` : combinaison de la lettre spectrale avec un bin
          numérique (``"0-4"`` ou ``"5-9"``) selon le chiffre de sous‑classe.

        Args:
            df (pandas.DataFrame): DataFrame des features contenant ``class`` et/ou ``subclass``.

        Returns:
            pandas.DataFrame: Une copie du DataFrame d’entrée complétée avec les colonnes dérivées.
        """
        out = df.copy()

        # ---- main_class
        if "main_class" not in out.columns:
            src = (
                "subclass"
                if "subclass" in out.columns
                else ("class" if "class" in out.columns else None)
            )
            if src is not None:
                letters = (
                    out[src]
                    .astype(str)
                    .str.extract(
                        r"([OBAFGKMLTYCWD])", expand=False
                    )  # 1re lettre de type spectral
                    .str.upper()
                )
                out["main_class"] = pd.Categorical(letters)

        # ---- sub_class_top25
        if "sub_class_top25" not in out.columns and "subclass" in out.columns:
            sub = out["subclass"].astype(str)
            top = sub.value_counts().index[:25]
            out["sub_class_top25"] = pd.Categorical(
                np.where(sub.isin(top), sub, "Other")
            )

        # ---- sub_class_bins (lettre + bin numérique)
        if "sub_class_bins" not in out.columns:
            if "subclass" in out.columns:
                s = (
                    out["subclass"]
                    .astype(str)
                    .str.extract(r"^([OBAFGKMLTYCWD])\s*([0-9])?", expand=True)
                )
                s.columns = ["L", "N"]
                bins = np.where(s["N"].fillna("0").astype(int) <= 4, "0-4", "5-9")
                out["sub_class_bins"] = pd.Categorical(
                    s["L"].fillna("X").str.upper() + "_" + bins
                )
            elif "main_class" in out.columns:
                # fallback si `subclass` absent : un seul bin par lettre
                out["sub_class_bins"] = pd.Categorical(
                    out["main_class"].astype(str).str.upper() + "_0-9"
                )

        # Harmonise en catégoriel
        for c in ("main_class", "sub_class_top25", "sub_class_bins"):
            if c in out.columns and out[c].dtype == "object":
                out[c] = out[c].astype("category")

        return out

    # ---------------------------------------------------------------------
    # Runs explorer helpers (highlight last run and refresh table)
    # ---------------------------------------------------------------------

    def _style_highlight_ts(
        self,
        df: pd.DataFrame,
        last_ts: str | None = None,
        best_ts: str | None = None,
        metric_col: str = "bal_acc",
    ) -> "pd.io.formats.style.Styler":
        """
        Retourne un Styler pandas qui applique un surlignage sur la ligne dont la colonne
        'ts' correspond à highlight_ts. Met également en forme certaines colonnes
        numériques avec trois décimales.

        Args:
            df: DataFrame contenant les runs.
            highlight_ts: Timestamp du run à surligner (str ou None).

        Returns:
            Un objet Styler pandas prêt à être affiché.
        """
        # couleurs
        c_last = "#fff6bf"  # jaune pâle  = dernier run
        c_best = "#d6f5d6"  # vert pâle   = meilleure ligne
        c_cell = "#a3e6a3"  # vert + fort = meilleure cellule du metric

        # repère l’index de la meilleure ligne selon metric_col
        m = pd.to_numeric(df.get(metric_col, pd.Series(dtype="float")), errors="coerce")
        if best_ts is None and not m.isna().all():
            try:
                best_ts = df.loc[m.idxmax(), "ts"]
            except Exception:
                best_ts = None

        def _hl(row):
            styles = []
            is_last = last_ts is not None and str(row.get("ts")) == str(last_ts)
            is_best = best_ts is not None and str(row.get("ts")) == str(best_ts)
            for col in row.index:
                base = ""
                if is_best:
                    base = f"background-color:{c_best};"
                if is_last and not is_best:
                    base = f"background-color:{c_last};"
                # en plus, on met le metric en vert plus soutenu si c’est la best row
                if is_best and col == metric_col:
                    base = f"background-color:{c_cell}; font-weight:bold;"
                styles.append(base)
            return styles

        fmt_cols = {
            "acc": "{:.3f}",
            "bal_acc": "{:.3f}",
            "f1_macro": "{:.3f}",
            "auc_macro": "{:.3f}",
            "ap_macro": "{:.3f}",
        }
        return df.style.format(fmt_cols).apply(_hl, axis=1)

    def _refresh_runs_table(self, highlight_ts: Optional[str] = None) -> None:
        """
        Recharge la table des runs dans un output dédié sans empilement et surligne
        éventuellement la ligne correspondant à highlight_ts ou au dernier run connu.

        Args:
            highlight_ts: Timestamp à surligner explicitement (optionnel).

        Side effects:
            Met à jour l'Output widget self._runs_out avec la nouvelle table.
        """
        df = _load_runs_table(self.reports_dir)
        with self._runs_out:
            clear_output(wait=True)
            if df is None or len(df) == 0:
                display(HTML("<i>Aucun run trouvé</i>"))
                return

            # 1) dernier run demandé ou mémorisé
            last_ts = highlight_ts or getattr(self, "_last_run_ts", None)

            # 2) FALLBACK : si inconnu/non présent, on prend le plus récent par timestamp
            try:
                ts_set = set(df["ts"].astype(str))
                if last_ts is None or str(last_ts) not in ts_set:
                    last_ts = max(ts_set)  # ex: "20250827T162302Z"
            except Exception:
                last_ts = None

            # 3) meilleure ligne (bal_acc ou f1_macro)
            metric_col = (
                "bal_acc"
                if "bal_acc" in df.columns
                else ("f1_macro" if "f1_macro" in df.columns else None)
            )
            best_ts = None
            if metric_col is not None:
                m = pd.to_numeric(df[metric_col], errors="coerce")
                if not m.isna().all():
                    best_ts = df.loc[m.idxmax(), "ts"]

            display(
                self._style_highlight_ts(
                    df,
                    last_ts=last_ts,
                    best_ts=best_ts,
                    metric_col=metric_col or "bal_acc",
                )
            )

    def _parse_exclusions(self, txt: str, selected) -> list[str]:
        """
        Combine les exclusions provenant d’un champ texte et d’une sélection multiple.

        Args:
            txt (str): Chaîne CSV d’exclusions saisie manuellement.
            selected: Valeurs sélectionnées dans la liste interactive (selectmultiple).

        Returns:
            list[str]: Liste triée de classes à exclure du jeu d’entraînement.
        """
        excl = set()
        if selected:
            excl.update(str(s) for s in selected)
        if txt:
            excl.update(s.strip() for s in txt.split(",") if s.strip())
        return sorted(excl)

    def _apply_class_filter(
        self, df: pd.DataFrame, target_col: str, excluded: list[str]
    ):
        """
        Filtre un DataFrame en retirant les lignes dont la classe est exclue.

        Args:
            df (pandas.DataFrame): DataFrame source.
            target_col (str): Nom de la colonne contenant la classe cible.
            excluded (list[str]): Liste des classes à exclure.

        Returns:
            tuple[pandas.DataFrame, int]: Une copie du DataFrame filtré et le
            nombre de lignes supprimées.

        Raises:
            ValueError: Si le filtrage laisse moins de deux classes distinctes.
        """
        if not excluded:
            return df, 0
        before = len(df)
        out = df[~df[target_col].astype(str).isin(excluded)].copy()
        removed = before - len(out)
        ncls = out[target_col].nunique()
        if ncls < 2:
            raise ValueError(
                f"Le filtrage laisse {ncls} classe(s) — il en faut au moins 2."
            )
        return out, removed

    def run_training_session(
        self,
        model_type: str = "XGBoost",
        n_estimators: int = 200,
        prediction_target: str = "main_class",
        save_and_log: bool = True,
        # FS
        use_feature_selection: bool = True,
        selector_model: str = "xgb",
        selector_threshold: str = "median",
        selector_n_estimators: int = 200,
        selector_method: str = "sfrommodel",
        # ---- options avancées ----
        search: str | None = None,
        cv_folds: int = 3,
        scoring: str = "accuracy",
        n_iter: int = 60,
        early_stopping: bool = True,
        early_stopping_rounds: int = 50,  # NEW
        val_size: float = 0.15,
        use_groups: bool = False,
        group_col: str | None = None,
        # split/seed
        test_size: float = 0.21,
        random_state: int = 42,
        # grilles
        param_grid: dict | None = None,
        param_distributions: dict | None = None,
        # poids & calibration
        use_balanced_weights: bool = True,
        calibrate_probs: bool = False,
        calibration_method: str = "sigmoid",
        class_weight_mode: str | None = None,
        class_weight_alpha: float = 1.0,
        weight_col: str | None = None,
        weight_norm: str = "minmax",
        repeated_cv: bool = False,
        cv_repeats: int = 1,
        calibrate_holdout_size: float = 0.0,
        calibrate_cv: int = 3,
        imputer_strategy: str | None = None,
        knn_imputer_k: int = 5,
        scaler_type: str | None = None,
        mi_top_k: int | None = None,
        # artefacts
        fi_n_repeats: int = 10,
        save_confusion_png: bool = False,
        save_curves_roc_pr: bool = True,
        save_calibration: bool = False,
        save_feature_importance: bool = True,
        export_test_predictions: bool = False,
        cm_normalized: bool = False,
        base_params: dict | None = None,
        exp_name: str | None = None,
        notes: str = "",
        # filtres & PCA & sampler
        var_threshold: float | None = None,
        corr_threshold: float | None = None,
        use_pca: bool | None = None,
        pca_components: float | int | None = None,
        sampler: str | None = None,
        # tuning de seuils
        tune_thresholds: bool | None = None,
        threshold_metric: str | None = None,
        # n_jobs pour les learners
        n_jobs: int | None = None,
        exclude_classes: list[str] | None = None,
        # nombre de jobs pour l'estimateur (XGB/RF) et méthode d'arbre XGB
        **kwargs: Any,
    ) -> Optional[SpectralClassifier]:
        """
        Entraîne et évalue un classifieur spectral en journalisant les artefacts.

        Cette méthode constitue le cœur du pipeline d’apprentissage. Elle prépare
        les données d’entraînement et de test (avec option de séparation par
        groupes), applique éventuellement une sélection de variables, instancie et
        entraîne le modèle choisi, effectue une recherche d’hyperparamètres et
        une calibration si demandé, puis évalue le modèle et enregistre les
        différents artefacts (matrice de confusion, courbes ROC/PR, importances,
        rapports JSON, modèle sérialisé, etc.). De nombreuses options permettent
        de contrôler la taille des ensembles, la sélection de features, le
        sampling, la calibration, la recherche d’hyperparamètres et la
        production de sorties.

        Args:
            model_type (str): Type de modèle à entraîner (p. ex. ``"XGBoost"``,
                ``"RandomForest"``, ``"SVM"``, etc.).
            n_estimators (int): Nombre d’estimateurs (arbres) lorsque applicable.
            prediction_target (str): Nom de la colonne cible à prédire dans
                :attr:`features_df`.
            save_and_log (bool): Si ``True``, enregistre le modèle et génère les artefacts.
            use_feature_selection (bool): Active la sélection de variables via un modèle.
            selector_model (str): Modèle utilisé pour la sélection de variables.
            selector_threshold (str): Seuil pour la sélection (``"median"``, valeur
                numérique, etc.).
            selector_n_estimators (int): Nombre d’estimateurs pour le sélecteur.
            selector_method (str): Méthode de sélection (``"sfrommodel"`` ou ``"rfecv"``).
            search (str | None): Mode de recherche d’hyperparamètres (``"grid"``,
                ``"random"`` ou ``None``).
            cv_folds (int): Nombre de folds pour la validation croisée.
            scoring (str): Métrique d’optimisation pour la recherche d’hyperparamètres.
            n_iter (int): Nombre d’itérations pour une recherche aléatoire.
            early_stopping (bool): Active l’early stopping (XGB).
            early_stopping_rounds (int): Nombre de rounds sans gain pour stopper.
            val_size (float): Taille du jeu de validation pour l’early stopping.
            use_groups (bool): Active un split par groupes via ``group_col``.
            group_col (str | None): Nom de la colonne contenant les groupes.
            test_size (float): Proportion dédiée au test si pas de validation croisée.
            random_state (int): Graine de génération aléatoire.
            param_grid (dict | None): Grille pour GridSearchCV.
            param_distributions (dict | None): Distributions pour RandomizedSearchCV.
            use_balanced_weights (bool): Pondération équilibrée des classes.
            calibrate_probs (bool): Calibration des probabilités.
            calibration_method (str): Méthode de calibration (``"sigmoid"`` ou
                ``"isotonic"``).
            class_weight_mode (str | None): Mode de pondération des classes.
            class_weight_alpha (float): Paramètre alpha pour la pondération.
            weight_col (str | None): Colonne des poids d’échantillon.
            weight_norm (str): Méthode de normalisation des poids.
            repeated_cv (bool): Effectuer une validation croisée répétée.
            cv_repeats (int): Nombre de répétitions pour la validation croisée.
            calibrate_holdout_size (float): Taille du holdout pour la calibration.
            calibrate_cv (int): Nombre de folds pour la calibration.
            imputer_strategy (str | None): Stratégie d’imputation des valeurs manquantes.
            knn_imputer_k (int): Nombre de voisins pour l’imputation KNN.
            scaler_type (str | None): Type de mise à l’échelle des features.
            mi_top_k (int | None): Nombre de features à sélectionner via l’information
                mutuelle (0 pour désactiver).
            var_threshold (float | None): Seuil de variance pour filtrer les features.
            corr_threshold (float | None): Seuil de corrélation maximale entre features.
            use_pca (bool | None): Active l’utilisation d’une PCA.
            pca_components (float | int | None): Nombre ou proportion de composantes PCA.
            sampler (str | None): Méthode de sur/échantillonnage (``"smote"``, etc.).
            tune_thresholds (bool | None): Active le tuning de seuils par classe.
            threshold_metric (str | None): Métrique pour le tuning de seuils.
            n_jobs (int | None): Nombre de tâches parallèles pour certains modèles.
            exclude_classes (list[str] | None): Liste de classes à exclure du
                jeu d’entraînement.
            **kwargs: Options supplémentaires transmises au classifieur.

        Returns:
            SpectralClassifier | None: L’objet classifieur entraîné (typiquement
            :class:`SpectralClassifier`), ou ``None`` si l’entraînement n’a pas
            pu être réalisé.

        Raises:
            ValueError: Si la séparation par groupes est activée sans fournir
                ``group_col`` ou si le filtrage de classes laisse moins de deux classes.
            RuntimeError: En cas d’échec critique lors de l’entraînement ou de la journalisation.

        See Also:
            :meth:`_log_and_report` — gestion des artefacts générés.
        """
        if getattr(self, "features_df", None) is None or self.features_df.empty:
            print(
                "ERREUR : Le DataFrame de features est vide. Veuillez d'abord exécuter `process_data`."
            )
            return

        if use_groups:
            if not group_col or group_col not in self.features_df.columns:
                print(
                    f"ERREUR : Group split activé mais colonne absente: {group_col!r}. "
                    "Désactivez 'Group split' ou indiquez une colonne valide."
                )
                return

        df_for_training = self.features_df
        excluded = exclude_classes or []
        if excluded:
            try:
                df_for_training, removed = self._apply_class_filter(
                    df_for_training, prediction_target, excluded
                )
                # Mémorise ce qui s'est passé pour le rapport
                self._last_class_filter = {
                    "mode": "exclude",
                    "classes": list(excluded),
                    "rows_removed": int(removed),
                    "n_classes_after": int(
                        df_for_training[prediction_target].nunique()
                    ),
                }
                print(
                    f"[Class filter] exclusions={excluded}  (lignes retirées: {removed})"
                )
            except ValueError as e:
                print(f"ERREUR filtre de classes: {e}")
                return
        else:
            self._last_class_filter = None

        clf = SpectralClassifier(
            model_type=model_type,
            prediction_target=prediction_target,
            use_feature_selection=use_feature_selection,
            selector_model=selector_model,
            selector_threshold=selector_threshold,
            selector_n_estimators=selector_n_estimators,
        )
        if n_jobs is not None:
            clf.n_jobs = int(n_jobs)

        search = None if search in (None, "None") else search

        result = clf.train_and_evaluate(
            df_for_training,
            n_estimators=n_estimators,
            search=search,
            cv_folds=cv_folds,
            scoring=scoring,
            n_iter=n_iter,
            early_stopping=early_stopping,
            early_stopping_rounds=early_stopping_rounds,
            val_size=val_size,
            use_groups=use_groups,
            group_col=group_col,
            test_size=test_size,
            random_state=random_state,
            param_grid=param_grid,
            param_overrides=base_params,
            param_distributions=param_distributions,
            use_balanced_weights=use_balanced_weights,
            calibrate_probs=calibrate_probs,
            calibration_method=calibration_method,
            class_weight_mode=class_weight_mode,
            class_weight_alpha=class_weight_alpha,
            weight_col=weight_col,
            weight_norm=weight_norm,
            repeated_cv=repeated_cv,
            cv_repeats=cv_repeats,
            calibrate_holdout_size=calibrate_holdout_size,
            calibrate_cv=calibrate_cv,
            imputer_strategy=imputer_strategy,
            knn_imputer_k=knn_imputer_k,
            scaler_type=scaler_type,
            # FS
            use_feature_selection=use_feature_selection,
            selector_model=selector_model,
            selector_threshold=selector_threshold,
            selector_method=selector_method,
            # filtres/PCA/sampler
            var_threshold=var_threshold,
            corr_threshold=corr_threshold,
            use_pca=use_pca,
            pca_components=pca_components,
            sampler=sampler,
            mi_top_k=mi_top_k,
            # tuning de seuils
            tune_thresholds=tune_thresholds,
            threshold_metric=threshold_metric,
        )
        if not result:
            print(
                "\n--- SESSION TERMINÉE SANS ENTRAÎNEMENT (pas assez de données valides) ---"
            )
            return

        trained_clf, feature_cols_before_fs, X_all, y_all, groups_all = result

        # Message sur la sélection de features
        if use_feature_selection and "fs" in trained_clf.model_pipeline.named_steps:
            kept = len(getattr(trained_clf, "selected_features_", []) or [])
            total = len(feature_cols_before_fs)
            msg = (
                f"[Feature selection] activée — {kept}/{total} features conservées."
                if kept
                else "[Feature selection] activée."
            )
            print("\n" + msg)
        else:
            print(
                f"\n[Feature selection] non utilisée. {len(feature_cols_before_fs)} features au total."
            )

        # Sauvegarde + rapport
        processed_files = []
        if "file_path" in self.features_df.columns:
            processed_files = (
                self.features_df["file_path"].dropna().astype(str).unique().tolist()
            )

        if save_and_log:
            # Si on a demandé un split par groupes, valider la colonne
            if use_groups and (
                not group_col or group_col not in self.features_df.columns
            ):
                print(
                    f"ERREUR : Group split activé mais colonne absente: {group_col!r}."
                )
                return

            try:
                # Journalisation & rapport : retourne le chemin du dossier du run
                run_dir = self._log_and_report(
                    trained_clf,
                    feature_cols_before_fs,
                    X_all,
                    y_all,
                    processed_files=processed_files,
                    groups=groups_all,
                    save_confusion_png=save_confusion_png,
                    save_curves_roc_pr=save_curves_roc_pr,
                    save_calibration=save_calibration,
                    save_feature_importance=save_feature_importance,
                    export_test_predictions=export_test_predictions,
                    cm_normalized=cm_normalized,
                    exp_name=exp_name,
                    notes=notes,
                    fi_n_repeats=fi_n_repeats,
                )

                # Enregistre le timestamp du dernier run et rafraîchit la table des runs
                if run_dir:
                    try:
                        ts = Path(run_dir).name
                        self._last_run_ts = ts
                        self._refresh_runs_table(highlight_ts=ts)
                    except Exception:
                        pass

                # mettre à jour trained_spectra.csv via DatasetBuilder
                try:
                    # 1) Préférence : chemins issus des features (col. file_path) utilisés
                    to_log = list(processed_files) if processed_files else []

                    # 2) Robustesse : si un chemin absolu a fuité, rebase en relatif sur raw_data_dir
                    rel: list[str] = []
                    for p in to_log:
                        q = str(p)
                        if os.path.isabs(q):
                            try:
                                q = os.path.relpath(q, start=self.raw_data_dir)
                            except Exception:
                                pass
                        rel.append(q.replace("\\", "/"))

                    # 3) Fallback : si rien côté features, utiliser le lot courant (select_batch)
                    if not rel and getattr(self, "current_batch", None):
                        rel = [str(p).replace("\\", "/") for p in self.current_batch]

                    if rel:
                        self.builder.update_trained_log(rel)  # ← dédup & append safe
                        print(
                            f"  > trained_spectra.csv mis à jour (+{len(rel)} entrées)."
                        )
                    else:
                        print("  > (info) Rien à journaliser (liste vide).")
                except Exception as e:
                    print(f"(warn) Mise à jour trained_spectra.csv impossible: {e}")

            except Exception as e:
                print(f"(avertissement) Échec lors de la génération du rapport : {e}")
        else:
            print("\n--- SESSION D'EXPÉRIMENTATION TERMINÉE (non sauvegardée) ---")

        return trained_clf

    # --------------------- Entrées de haut niveau ---------------------

    def run_full_pipeline(
        self,
        batch_size: int = 500,
        model_type: str = "RandomForest",
        n_estimators: int = 100,
        prediction_target: str = "main_class",
        save_and_log: bool = True,
        enrich_gaia: bool = False,
        **gaia_kwargs: Any,
    ) -> None:
        """
        Exécute l’ensemble du pipeline d’un bout à l’autre, de la sélection au modèle entraîné.

        Cette méthode combine l’appel à :meth:`select_batch`,
        :meth:`generate_and_enrich_catalog`, :meth:`process_data` et
        :meth:`run_training_session` avec les paramètres fournis. Elle est
        pratique pour exécuter un run complet sans passer par l’interface
        interactive.

        Args:
            batch_size (int, optional): Taille du lot de spectres à traiter.
            model_type (str, optional): Type de modèle à entraîner.
            n_estimators (int, optional): Nombre d’estimateurs du modèle final.
            prediction_target (str, optional): Nom de la colonne cible.
            save_and_log (bool, optional): Active la sauvegarde du modèle et des artefacts.
            enrich_gaia (bool, optional): Si ``True``, enrichit le catalogue avec Gaia.
            **gaia_kwargs: Paramètres additionnels pour l’enrichissement Gaia.

        Side Effects:
            Appelle en interne les autres méthodes du pipeline et met à jour l’état interne.
        """
        self.select_batch(batch_size=batch_size)
        if not self.current_batch:
            return

        self.generate_and_enrich_catalog(enrich_gaia=enrich_gaia, **gaia_kwargs)

        self.process_data()
        if self.features_df.empty:
            return

        self.run_training_session(
            model_type, n_estimators, prediction_target, save_and_log
        )

        # Ajout : journaliser les spectres entraînés dans trained_spectra.csv
        # Lorsque save_and_log est activé et qu'un lot a été sélectionné,
        # on enregistre la liste des fichiers relatifs dans le journal via le DatasetBuilder.
        if save_and_log and getattr(self, "current_batch", None):
            try:
                # Utilise update_trained_log pour ne journaliser que les nouveaux spectres
                self.builder.update_trained_log(self.current_batch)
            except Exception as e:
                print(f"(warn) Impossible de mettre à jour trained_spectra.csv : {e}")

    def interactive_training_runner(self) -> None:
        """
        Affiche l’interface interactive d’entraînement sous Jupyter ou VS Code.

        L’interface utilisateur repose sur ipywidgets et est organisée en plusieurs
        onglets permettant de configurer la source des features et la partition
        des données, de choisir le type de modèle et ses paramètres, d’activer
        la sélection de variables et la recherche d’hyperparamètres, de définir
        les poids et la calibration, et de choisir les artefacts à produire.
        Un bouton « Lancer l’entraînement » exécute :meth:`run_training_session`
        avec les paramètres courants, et un explorateur de runs permet de
        consulter les sessions précédentes.

        Notes:
            La configuration courante peut être sauvegardée ou rechargée depuis un
            fichier JSON via des boutons dédiés. Le lancement effectif est
            délégué à une fonction interne ``_on_run``.
        """
        import json as _json
        import ipywidgets as _W

        # Crée une sortie globale dès maintenant pour que les helpers puissent l'utiliser
        out = _W.Output()

        # --- helpers internes ---
        def _parse_json(txt_widget):
            """Parse le contenu d'un champ JSON, retourne None en cas d'échec."""
            try:
                s = (txt_widget.value or "").strip()
                return _json.loads(s) if s else None
            except Exception as e:
                # écrit l'avertissement dans la sortie commune
                with out:
                    print(f"(warn) JSON invalide pour '{txt_widget.description}': {e}")
                return None

        def _grid_size(d):
            """Retourne une estimation du nombre de combinaisons dans un param_grid."""
            try:
                total = 1
                for k, v in (d or {}).items():
                    if isinstance(v, (list, tuple)):
                        total *= max(1, len(v))
                return total
            except Exception:
                return None

        # --- Onglet 1 : Data & Split -------------------------------------------------

        # Choix de la cible : liste fixe des colonnes de classes
        target = _W.Dropdown(
            options=["main_class", "sub_class_top25", "sub_class_bins"],
            value="main_class",
            description="Cible",
        )
        # Force la liste d'options à ces trois cibles ; en cas de modifications du DataFrame, on garde les noms attendus
        try:
            target.options = [
                ("main_class", "main_class"),
                ("sub_class_top25", "sub_class_top25"),
                ("sub_class_bins", "sub_class_bins"),
            ]
            if target.value not in dict(target.options):
                target.value = "main_class"
        except Exception:
            pass
        test_size = _W.FloatSlider(
            value=0.21,
            min=0.05,
            max=0.5,
            step=0.01,
            description="test_size",
            readout_format=".2f",
            layout=_W.Layout(width="300px"),
        )
        seed = _W.IntText(
            value=42,
            description="seed",
            layout=_W.Layout(width="160px"),
        )
        cv_folds = _W.IntSlider(
            value=5,
            min=2,
            max=20,
            step=1,
            description="CV folds",
            layout=_W.Layout(width="300px"),
        )
        rep_cv = _W.Checkbox(value=False, description="Repeated CV")
        rep_cv_repeats = _W.IntSlider(
            value=1,
            min=1,
            max=10,
            step=1,
            description="CV repeats",
            layout=_W.Layout(width="300px"),
        )
        use_groups = _W.Checkbox(value=False, description="Group split")
        group_col = _W.Text(
            value="",
            placeholder="nom de la colonne group",
            description="Col. groupes",
            layout=_W.Layout(width="300px"),
        )

        # --- Bloc "Source des features" -------------------------------------
        feat_src = _W.Dropdown(
            options=[
                ("En mémoire", "mem"),
                ("Dernier CSV", "last"),
                ("Choisir CSV", "pick"),
            ],
            value="mem",
            description="Features",
        )
        feat_files = _W.Dropdown(
            options=[(Path(p).name, p) for p in self._list_feature_files(50)],
            description="fichier",
            layout=_W.Layout(width="420px"),
        )
        feat_refresh = _W.Button(description="🔄", layout=_W.Layout(width="42px"))
        feat_load = _W.Button(description="Charger", icon="upload")
        feat_info = _W.HTML(value="")

        # --- Exclure des classes du target ---
        exclude_txt = _W.Text(
            value="",
            placeholder="ex: B,D,W",
            description="Exclure (CSV)",
            layout=_W.Layout(width="260px"),
        )
        exclude_dd = _W.SelectMultiple(
            options=[],
            value=(),
            description="Exclure (liste)",
            rows=6,
            layout=_W.Layout(width="320px"),
        )
        exclude_refresh = _W.Button(
            description="⟳ classes", layout=_W.Layout(width="110px")
        )

        def _refresh_exclude_choices(*_):
            try:
                df = getattr(self, "features_df", None)
                col = target.value
                if df is None or df.empty or col not in df.columns:
                    exclude_dd.options = []
                    exclude_dd.value = ()
                    return
                classes = sorted(df[col].astype(str).unique().tolist())
                exclude_dd.options = classes

            except Exception:
                exclude_dd.options = []
                exclude_dd.value = ()

        exclude_refresh.on_click(_refresh_exclude_choices)
        target.observe(lambda ch: _refresh_exclude_choices(), "value")

        def _refresh_feat_list(_=None):
            feat_files.options = [
                (Path(p).name, p) for p in self._list_feature_files(50)
            ]

        feat_refresh.on_click(_refresh_feat_list)

        def _do_load(_=None):
            if feat_src.value == "mem":
                if getattr(self, "features_df", None) is None or self.features_df.empty:
                    feat_info.value = "<i>self.features_df est vide.</i>"
                    _refresh_exclude_choices()
                else:
                    n = len(self.features_df)
                    d = self.features_df.select_dtypes(include=["number"]).shape[1]
                    feat_info.value = f"<b>OK</b> — {n:,} lignes, {d} num. features"
            elif feat_src.value == "last":
                df = self.load_features_from_csv(use_last=True)
                feat_info.value = (
                    "<b>Dernier CSV chargé.</b>"
                    if df is not None
                    else "<b>Échec de chargement.</b>"
                )
            else:
                if not feat_files.value:
                    feat_info.value = "<i>Choisis un fichier.</i>"
                else:
                    df = self.load_features_from_csv(path=feat_files.value)
                    feat_info.value = (
                        "<b>CSV chargé.</b>"
                        if df is not None
                        else "<b>Échec de chargement.</b>"
                    )

        feat_load.on_click(_do_load)
        tab_data = _W.VBox(
            [
                _W.HBox([feat_src, feat_files, feat_refresh, feat_load]),
                feat_info,
                _W.HBox([target, test_size, seed]),
                _W.HBox([cv_folds, rep_cv, rep_cv_repeats]),
                _W.HBox([use_groups, group_col]),
                _W.HBox([exclude_txt, exclude_dd, exclude_refresh]),
            ]
        )

        # Auto-charge le dernier CSV au démarrage
        feat_src.value = "last"
        _do_load()

        # ==== onglet 2: Modèle & Pré-traitement ====
        model = _W.Dropdown(
            options=(
                "XGBoost",
                "RandomForest",
                "SVM",
                "ExtraTrees",
                "LogRegOVR",
                "KNN",
                "MLP",
                "LDA",
                "QDA",
                "CatBoost",
                "LightGBM",
                "SoftVoting",
            ),
            value="XGBoost",
            description="Modèle",
        )
        n_estim = _W.IntText(value=400, description="N Estimators")
        imputer = _W.Dropdown(
            options=["median", "mean", "most_frequent", "knn", "none"],
            value="median",
            description="imputer",
        )
        knn_k = _W.IntSlider(value=5, min=2, max=25, step=1, description="knn_k")
        scaler = _W.Dropdown(
            options=["standard", "robust", "minmax", "none"],
            value="standard",
            description="scaler",
        )
        base_params = _W.Textarea(
            value="",
            description="base_params (JSON)",
            layout=_W.Layout(width="100%", height="70px"),
        )
        # n_jobs pour le parallélisme
        n_jobs = _W.IntSlider(value=-1, min=-1, max=16, step=1, description="n_jobs")

        # filtres colinéarité/variance + PCA
        var_threshold = _W.FloatText(description="Var. threshold", value=0.0)
        corr_threshold = _W.FloatSlider(
            description="Corr. max", min=0.90, max=0.999, step=0.001, value=0.98
        )
        use_pca = _W.Checkbox(description="PCA", value=False)
        pca_components = _W.FloatSlider(
            description="PCA n_comp", min=0.5, max=0.999, step=0.001, value=0.99
        )
        # sampler
        sampler = _W.Dropdown(
            description="Sampler (CV)",
            options=["none", "smote", "borderline", "smoteenn", "adasyn"],
            value="none",
        )

        # SVM requiert un scaler ; ajuste si nécessaire et désactive n_estim pour SVM
        def _toggle_svm(change=None):
            # Ajuste le scaler automatiquement pour SVM
            if model.value == "SVM" and scaler.value == "none":
                scaler.value = "standard"
            # Désactive le paramètre n_estimators pour SVM
            n_estim.disabled = model.value == "SVM"

        model.observe(_toggle_svm, "value")
        _toggle_svm()
        tab_model = _W.VBox(
            [
                _W.HBox([model, n_estim]),
                _W.HBox([imputer, knn_k, scaler, n_jobs]),
                base_params,
                var_threshold,
                corr_threshold,
                use_pca,
                pca_components,
                sampler,
            ]
        )

        # ==== onglet 3: Sélection de features ====
        fs_enable = _W.Checkbox(value=True, description="use_feature_selection")
        fs_method = _W.Dropdown(
            options=[
                ("RandomForest", "rf"),
                ("XGBoost", "xgb"),
                ("ExtraTrees", "ext"),
                ("LogReg L1", "l1"),
            ],
            value="xgb",
            description="selector_model",
        )
        fs_thresh = _W.Text(value="median", description="selector_threshold")
        fs_n = _W.IntSlider(
            value=400, min=50, max=1500, step=50, description="selector_n_estimators"
        )
        # FECV vs SelectFromModel
        fs_kind = _W.Dropdown(
            description="Méthode FS",
            options=[("SelectFromModel", "sfrommodel"), ("RFECV", "rfecv")],
            value="sfrommodel",
            tooltip="Choisir RFECV pour un ranking récursif ; sinon SelectFromModel.",
        )
        mi_topk = _W.IntText(value=0, description="MI top-K (0=off)")
        tab_fs = _W.VBox([fs_enable, fs_kind, fs_method, fs_thresh, fs_n, mi_topk])

        # ==== onglet 4: Recherche ====
        search = _W.Dropdown(
            options=[
                ("Aucun", None),
                ("GridSearchCV", "grid"),
                ("RandomizedSearchCV", "random"),
            ],
            value=None,
            description="Search",
        )
        es = _W.Checkbox(value=True, description="Early stopping (XGB)")
        es_rounds = _W.IntSlider(
            value=50, min=10, max=300, step=5, description="ES rounds"
        )
        val_size = _W.FloatSlider(
            value=0.15,
            min=0.05,
            max=0.40,
            step=0.01,
            description="val_size",
            readout_format=".02f",
        )
        scoring = _W.Dropdown(
            options=["accuracy", "balanced_accuracy", "f1_macro", "f1_weighted"],
            value="accuracy",
            description="Scoring",
        )
        n_iter = _W.IntSlider(
            value=80, min=10, max=400, step=10, description="n_iter (random)"
        )
        param_grid = _W.Textarea(
            value="",
            description="param_grid (JSON)",
            layout=_W.Layout(width="100%", height="70px"),
        )
        param_dists = _W.Textarea(
            value="",
            description="param_dists (JSON)",
            layout=_W.Layout(width="100%", height="70px"),
        )

        # Lorsque le modèle change, préremplit base_params, param_grid et param_distributions
        def _on_model_change(change=None):
            mval = model.value
            try:
                base_params.value = _json.dumps(PRESET_BASE.get(mval, {}), indent=2)
            except Exception:
                base_params.value = "{}"
            try:
                param_grid.value = _json.dumps(PRESET_GRID.get(mval, {}), indent=2)
            except Exception:
                param_grid.value = "{}"
            try:
                param_dists.value = _json.dumps(PRESET_DISTS.get(mval, {}), indent=2)
            except Exception:
                param_dists.value = "{}"

        model.observe(_on_model_change, "value")
        _on_model_change()

        grid_hint = _W.HTML(value="")

        def _toggle_search(change=None):
            # Early stopping activé seulement si aucune recherche
            es.disabled = search.value is not None
            es_rounds.disabled = es.disabled
            # Affiche/cacher les champs selon le mode de recherche
            param_grid.layout.display = "block" if search.value == "grid" else "none"
            param_dists.layout.display = "block" if search.value == "random" else "none"
            n_iter.layout.display = "block" if search.value == "random" else "none"
            # Estimation de la taille de la grille
            if search.value == "grid":
                n = _grid_size(_parse_json(param_grid))
                grid_hint.value = f"<i>Grid size estimée: {n}</i>" if n else ""
            else:
                grid_hint.value = ""

        search.observe(_toggle_search, "value")
        _toggle_search()

        def _on_grid_change(_):
            if search.value == "grid":
                n = _grid_size(_parse_json(param_grid))
                grid_hint.value = f"<i>Grid size estimée: {n}</i>" if n else ""

        param_grid.observe(_on_grid_change, "value")

        tab_search = _W.VBox(
            [
                _W.HBox([search, scoring, n_iter]),
                _W.HBox([es, es_rounds, val_size]),
                param_grid,
                param_dists,
                grid_hint,
            ]
        )

        # ==== onglet 5: Poids & Calibration ====
        balanced = _W.Checkbox(value=True, description="balanced_weights")
        cw_mode = _W.Dropdown(
            options=[("None", None), ("Inverse freq", "inv_freq")],
            value=None,
            description="class_weight_mode",
        )
        cw_alpha = _W.FloatSlider(
            value=1.0,
            min=0.1,
            max=2.0,
            step=0.1,
            description="alpha",
            readout_format=".1f",
        )
        wt_col = _W.Text(value="", description="weight_col")
        wt_norm = _W.Dropdown(
            options=["minmax", "log", "none"], value="minmax", description="norm"
        )
        calibrate = _W.Checkbox(value=False, description="calibrate_probs")
        calib_method = _W.Dropdown(
            options=["sigmoid", "isotonic"], value="sigmoid", description="method"
        )
        calib_holdout = _W.FloatSlider(
            value=0.0, min=0.0, max=0.4, step=0.05, description="holdout"
        )
        calib_cv = _W.IntSlider(value=3, min=2, max=10, step=1, description="calib_cv")
        hint = _W.HTML()

        tune_thresholds = _W.Checkbox(
            description="Tuning de seuils par classe", value=False
        )
        threshold_metric = _W.Dropdown(
            description="Métrique seuils",
            options=["f1_macro", "balanced_accuracy"],
            value="f1_macro",
        )

        def _svm_hint(_=None):
            hint.value = (
                "<i>Calibration activera un CalibratedClassifierCV pour SVM.</i>"
                if (model.value == "SVM" and calibrate.value)
                else ""
            )

        model.observe(_svm_hint, "value")
        calibrate.observe(_svm_hint, "value")
        _svm_hint()
        tab_weight = _W.VBox(
            [
                _W.HTML("<b>Poids</b>"),
                _W.HBox([balanced, cw_mode, cw_alpha]),
                _W.HBox([wt_col, wt_norm]),
                _W.HTML("<b>Calibration</b>"),
                _W.HBox([calibrate, calib_method, calib_holdout, calib_cv]),
                _W.HBox([tune_thresholds, threshold_metric]),
                hint,
            ]
        )

        # ==== onglet 6: Sorties ====
        save_log = _W.Checkbox(value=True, description="Sauvegarder & Journaliser")
        save_cm = _W.Checkbox(value=False, description="save_confusion_png")
        norm_cm = _W.Checkbox(value=False, description="normalized")
        save_rocpr = _W.Checkbox(value=True, description="save_curves_roc_pr")
        save_calib = _W.Checkbox(value=False, description="save_calibration")
        save_feat = _W.Checkbox(value=True, description="save_feature_importance")
        export_pred = _W.Checkbox(value=False, description="export_test_predictions")
        fi_nrep = _W.IntSlider(
            value=10, min=3, max=50, step=1, description="FI n_repeats"
        )
        tab_out = _W.VBox(
            [
                _W.HBox([save_log, save_cm, norm_cm]),
                save_rocpr,
                save_calib,
                save_feat,
                fi_nrep,
                export_pred,
            ]
        )
        # --- Préconfigurations des presets et nom d’expérience ----------------
        preset_path = _W.Text(
            value=str(Path(self.reports_dir) / "preset.json"), description="Preset"
        )
        btn_save = _W.Button(description="Sauver", icon="save")
        btn_load = _W.Button(description="Charger", icon="upload")

        def _collect_widgets():
            # Regroupe les widgets utiles dans un dict {nom: widget}
            return {
                # Onglet Data & Split
                "prediction_target": target,
                "test_size": test_size,
                "seed": seed,
                "cv_folds": cv_folds,
                "rep_cv": rep_cv,
                "cv_repeats": rep_cv_repeats,
                "use_groups": use_groups,
                "group_col": group_col,
                # Onglet Modèle & Prep
                "model_type": model,
                "n_estimators": n_estim,
                "imputer_strategy": imputer,
                "knn_imputer_k": knn_k,
                "scaler_type": scaler,
                "base_params": base_params,
                "param_grid": param_grid,
                "param_dists": param_dists,
                # Onglet Recherche HP
                "search": search,
                "scoring": scoring,
                "n_iter": n_iter,
                "val_size": val_size,
            }

        btn_save.on_click(lambda _: _save_preset(preset_path.value, _collect_widgets()))
        btn_load.on_click(lambda _: _load_preset(preset_path.value, _collect_widgets()))

        tab_presets = _W.HBox([preset_path, btn_save, btn_load])

        # ==== Lancer ====
        run_btn = _W.Button(
            description="Lancer l'entraînement",
            button_style="success",
            icon="play",
            layout=_W.Layout(width="240px"),
        )
        summary = _W.Textarea(
            value="",
            description="Résumé",
            layout=_W.Layout(width="100%", height="120px"),
        )
        tab_run = _W.VBox([tab_presets, run_btn, summary, out])

        # --- Templates d’hyperparamètres ---
        template_dropdown = _W.Dropdown(
            options=[
                ("— Template hyperparams —", ""),
                ("XGB · medium", "xgb_medium"),
                ("XGB · wide", "xgb_wide"),
                ("RF  · medium", "rf_medium"),
                ("RF  · deep", "rf_deep"),
                ("SVM · rbf", "svm_rbf"),
                ("SVM · linear", "svm_linear"),
            ],
            value="",
            description="Templates:",
            layout=_W.Layout(width="280px"),
        )

        apply_template_btn = _W.Button(
            description="Appliquer",
            icon="wand-magic-sparkles",
            button_style="",
            layout=_W.Layout(width="120px"),
        )

        # --- Notes run (mémorisées dans le JSON du rapport) ---
        notes_widget = _W.Textarea(
            value="",
            placeholder="Notes sur ce run (idées, dataprep, etc.)",
            description="Notes:",
            layout=_W.Layout(width="520px", height="60px"),
        )

        # --- File d’expériences (batch) ---
        add_to_batch_btn = _W.Button(
            description="Ajouter au batch", icon="plus", layout=_W.Layout(width="160px")
        )
        run_batch_btn = _W.Button(
            description="Lancer le batch",
            icon="play",
            button_style="success",
            layout=_W.Layout(width="160px"),
        )
        clear_batch_btn = _W.Button(
            description="Vider",
            icon="trash",
            button_style="warning",
            layout=_W.Layout(width="110px"),
        )
        batch_progress = _W.IntProgress(
            value=0, min=0, max=1, description="Batch:", layout=_W.Layout(width="520px")
        )
        exp_name_widget = _W.Text(
            value="",
            placeholder="Nom d’expérience (ex: lgbm_bins_v2)",
            description="Exp name:",
            layout=_W.Layout(width="320px"),
        )

        batch_store = []  # mémoire locale

        # --- Estimation coût / nombre de fits ---
        fits_label = _W.HTML("<b>Fits estimés:</b> –")

        tpl_box = _W.HBox([template_dropdown, apply_template_btn, fits_label])
        notes_box = _W.HBox([notes_widget, exp_name_widget])
        batch_box = _W.HBox(
            [add_to_batch_btn, run_batch_btn, clear_batch_btn, batch_progress]
        )

        # ==== onglets globaux ====
        tabs = _W.Tab(
            children=[
                tab_data,
                tab_model,
                tab_fs,
                tab_search,
                tab_weight,
                tab_out,
                tab_run,
            ]
        )
        tabs.set_title(0, "Data & Split")
        tabs.set_title(1, "Modèle & Prep")
        tabs.set_title(2, "Feature Sel.")
        tabs.set_title(3, "Recherche HP")
        tabs.set_title(4, "Poids & Calib.")
        tabs.set_title(5, "Sorties")
        tabs.set_title(6, "Lancer")

        display(tabs, tpl_box, notes_box, batch_box)

        # Templates hyperparams (callback + dictionnaires)
        def _tpl_xgb_medium():
            return {
                "clf__gamma": [0, 0.1, 0.5, 1.0, 2.0, 5.0],
                "clf__reg_lambda": [0.5, 1.0, 1.5, 2.0, 5.0],
                "clf__reg_alpha": [0, 0.1, 0.5, 1.0],
                "clf__learning_rate": [0.03, 0.05, 0.07, 0.1],
                "clf__subsample": [0.7, 0.8, 0.9],
                "clf__colsample_bytree": [0.7, 0.8, 0.9],
                "clf__max_depth": [4, 6, 8],
                "clf__min_child_weight": [1, 3, 5],
            }

        def _tpl_xgb_wide():
            return {
                "clf__gamma": [0, 0.05, 0.1, 0.5, 1.0, 2.0],
                "clf__reg_lambda": [0.0, 0.5, 1.0, 2.0, 5.0, 10.0],
                "clf__reg_alpha": [0.0, 0.1, 0.5, 1.0, 2.0],
                "clf__learning_rate": [0.02, 0.03, 0.05, 0.07, 0.1],
                "clf__subsample": [0.6, 0.7, 0.8, 0.9],
                "clf__colsample_bytree": [0.6, 0.7, 0.8, 0.9],
                "clf__max_depth": [3, 4, 6, 8, 10],
                "clf__min_child_weight": [1, 3, 5, 7],
            }

        def _tpl_rf_medium():
            return {
                "clf__n_estimators": [300, 600, 900],
                "clf__max_depth": [None, 10, 20],
                "clf__max_features": ["sqrt", 0.5, 0.8],
                "clf__min_samples_split": [2, 5, 10],
                "clf__min_samples_leaf": [1, 2, 4],
                "clf__bootstrap": [True],
            }

        def _tpl_rf_deep():
            return {
                "clf__n_estimators": [800, 1200],
                "clf__max_depth": [None, 20, 30],
                "clf__max_features": ["sqrt", 0.5, 0.8],
                "clf__min_samples_split": [2, 5],
                "clf__min_samples_leaf": [1, 2],
                "clf__bootstrap": [True],
            }

        def _tpl_svm_rbf():
            return {
                "clf__kernel": ["rbf"],
                "clf__C": [0.5, 1, 2, 5, 10],
                "clf__gamma": ["scale", 0.01, 0.05, 0.1],
            }

        def _tpl_svm_linear():
            return {
                "clf__kernel": ["linear"],
                "clf__C": [0.5, 1, 2, 5, 10],
            }

        def _apply_template(_):
            tpl = template_dropdown.value
            if not tpl:
                return
            if tpl == "xgb_medium":
                d = _tpl_xgb_medium()
            elif tpl == "xgb_wide":
                d = _tpl_xgb_wide()
            elif tpl == "rf_medium":
                d = _tpl_rf_medium()
            elif tpl == "rf_deep":
                d = _tpl_rf_deep()
            elif tpl == "svm_rbf":
                d = _tpl_svm_rbf()
            elif tpl == "svm_linear":
                d = _tpl_svm_linear()
            else:
                d = {}
            param_dists.value = json.dumps(d, indent=2)
            search.value = "random"
            template_dropdown.value = ""  # reset visuel

        apply_template_btn.on_click(_apply_template)

        # Estimation “nombre de fits” (et petit temps indicatif)
        def _estimate_fits(*args):
            try:
                grid = json.loads(param_grid.value or "{}")
            except Exception:
                grid = {}

            try:
                json.loads(param_dists.value or "{}")
            except Exception:
                pass

            if search.value == "grid":
                combs = 1
                for _, v in grid.items():
                    combs *= max(1, len(v) if isinstance(v, list) else 1)
            elif search.value == "random":
                combs = max(1, int(n_iter.value))
            else:
                combs = 1

            folds = max(2, int(cv_folds.value))
            total = combs * folds
            fits_label.value = f"<b>Fits estimés:</b> {total:,}"

        # observe
        for w in (param_grid, param_dists, cv_folds, n_iter, search):
            w.observe(_estimate_fits, "value")
        _estimate_fits()

        # SVM : proba auto si nécessaire + désactivations contextuelles
        def _model_context_update(change=None):
            # désactiver n_estimators quand SVM
            n_estim.disabled = model.value == "SVM"

            # si SVM et qu'on a besoin de proba → forcer probability=True dans base_params
            need_proba = calibrate.value or save_rocpr.value or export_pred.value
            if model.value == "SVM" and need_proba:
                try:
                    bp = json.loads(base_params.value or "{}")
                except Exception:
                    bp = {}
                if not bp.get("probability", False):
                    bp["probability"] = True
                    base_params.value = json.dumps(bp, indent=2)

        # observe
        for w in (model, calibrate, save_rocpr, export_pred):
            w.observe(_model_context_update, "value")
        _model_context_update()

        # Batch d’expériences (ajout/lecture/exécution sérielle)
        def _current_config_json() -> dict:
            """
            Retourne la configuration courante telle que définie dans l’UI.

            La structure englobe la cible, le modèle, le split, la recherche HP,
            les réglages de pré-traitement/FS/poids/calibration et les options de sortie.

            Returns:
                Dictionnaire sérialisable (prêt à écrire en JSON).
            """
            import json as _json
            from pathlib import Path

            # Helpers localisés
            def _parse_json(txt_widget):
                try:
                    s = (txt_widget.value or "").strip()
                    return _json.loads(s) if s else None
                except Exception as e:
                    with out:
                        print(
                            f"(warn) JSON invalide pour '{txt_widget.description}': {e}"
                        )
                    return None

            def _get_excluded():
                return self._parse_exclusions(exclude_txt.value, exclude_dd.value)

            cfg = {
                # Data & split
                "prediction_target": target.value,
                "test_size": test_size.value,
                "random_state": seed.value,
                "cv_folds": cv_folds.value,
                "repeated_cv": rep_cv.value,
                "cv_repeats": rep_cv_repeats.value,
                "use_groups": use_groups.value,
                "group_col": (group_col.value or None),
                # Model & preprocessing
                "model_type": model.value,
                "n_estimators": n_estim.value,
                "imputer_strategy": None if imputer.value == "none" else imputer.value,
                "knn_imputer_k": knn_k.value,
                "scaler_type": None if scaler.value == "none" else scaler.value,
                "base_params": _parse_json(base_params),
                # parallélisme (utilisé par run_training_session → clf.n_jobs)
                "n_jobs": int(n_jobs.value),
                # Feature selection
                "use_feature_selection": fs_enable.value,
                "selector_model": fs_method.value,
                "selector_threshold": fs_thresh.value,
                "selector_n_estimators": fs_n.value,
                # type de sélection (RFECV vs SelectFromModel)
                "selector_method": fs_kind.value,  # <— BRANCHÉ
                "mi_top_k": (None if (mi_topk.value or 0) <= 0 else int(mi_topk.value)),
                # Search
                "search": search.value,
                "scoring": scoring.value,
                "n_iter": n_iter.value,
                "early_stopping": es.value,
                "early_stopping_rounds": es_rounds.value,
                "val_size": val_size.value,
                "param_grid": _parse_json(param_grid) or {},
                "param_distributions": _parse_json(param_dists) or {},
                # Weights & calibration
                "use_balanced_weights": balanced.value,
                "class_weight_mode": cw_mode.value,
                "class_weight_alpha": cw_alpha.value,
                "weight_col": (wt_col.value or None),
                "weight_norm": wt_norm.value,
                "calibrate_probs": calibrate.value,
                "calibration_method": calib_method.value,
                # utiliser les widgets (pas des constantes)
                "calibrate_holdout_size": float(calib_holdout.value),
                "calibrate_cv": int(calib_cv.value),
                # filtres/ PCA / sampler
                "var_threshold": (
                    None
                    if (var_threshold.value is None or var_threshold.value <= 0)
                    else float(var_threshold.value)
                ),
                "corr_threshold": (
                    None
                    if (corr_threshold.value is None or corr_threshold.value <= 0)
                    else float(corr_threshold.value)
                ),
                "use_pca": bool(use_pca.value),
                "pca_components": (
                    None if not use_pca.value else float(pca_components.value)
                ),
                "sampler": (None if sampler.value == "none" else sampler.value),
                # Tuning de seuils
                "tune_thresholds": bool(tune_thresholds.value),
                "threshold_metric": threshold_metric.value,
                # Outputs
                "save_confusion_png": save_cm.value,
                "cm_normalized": norm_cm.value,
                "save_curves_roc_pr": save_rocpr.value,
                "save_calibration": save_calib.value,
                "save_feature_importance": save_feat.value,
                "export_test_predictions": export_pred.value,
                "fi_n_repeats": int(fi_nrep.value),
                # Notes
                "notes": notes_widget.value,
                "exp_name": (exp_name_widget.value or None),
                "exclude_classes": _get_excluded(),
            }

            # Compte des features numériques (pour info)
            try:
                cfg["n_features_candidate"] = int(
                    self.features_df.select_dtypes(include=["number"]).shape[1]
                )
            except Exception:
                cfg["n_features_candidate"] = None

            # Répartition des classes de la cible (si la colonne existe)
            tcol = cfg.get("prediction_target")
            if tcol in self.features_df.columns:
                try:
                    cfg["class_counts"] = (
                        self.features_df[tcol].value_counts(dropna=False).to_dict()
                    )
                except Exception:
                    cfg["class_counts"] = {}
            else:
                cfg["class_counts"] = {}
                with out:
                    print(
                        f"(warn) Colonne cible absente dans features_df : {tcol!r}. "
                        f"Choisis une des colonnes catégorielles existantes dans le menu « Cible »."
                    )
            # Trace légère de la config
            Path(self.reports_dir, "last_config_used.json").write_text(
                _json.dumps(cfg, indent=2),
                encoding="utf-8",
            )
            return cfg

        def _on_add_to_batch(_):
            cfg = _current_config_json()
            batch_store.append(cfg)
            batch_progress.max = len(batch_store)
            batch_progress.value = min(batch_progress.value, len(batch_store))
            print(f"→ Ajout au batch ({len(batch_store)} config(s)).")

        def _on_clear_batch(_):
            batch_store.clear()
            batch_progress.max = 1
            batch_progress.value = 0
            print("Batch vidé.")

        def _on_run_batch(_):
            if not batch_store:
                print("Batch vide.")
                return
            print(f"⏱ Lancement du batch ({len(batch_store)} runs)…")
            batch_progress.value = 0
            for i, cfg in enumerate(list(batch_store), start=1):
                batch_progress.value = i
                # Ajuste SVM probability si nécessaire (sécurité)
                if cfg["model_type"] == "SVM" and (
                    cfg["calibrate_probs"]
                    or cfg["save_curves_roc_pr"]
                    or cfg["export_test_predictions"]
                ):
                    bp = cfg.get("base_params") or {}
                    bp["probability"] = True
                    cfg["base_params"] = bp

                # Appel direct à la même API que le bouton RUN
                self.run_training_session(**cfg)
                # petit sleep de politesse pour l’UI
                time.sleep(0.1)
            print("Batch terminé.")

        add_to_batch_btn.on_click(_on_add_to_batch)
        clear_batch_btn.on_click(_on_clear_batch)
        run_batch_btn.on_click(_on_run_batch)

        # Rendu du tableau des runs : style + liens cliquables
        def _render_runs_table(df):
            if df is None or df.empty:
                display(HTML("<i>Aucun run trouvé.</i>"))
                return
            df = df.copy()

            # Lien cliquable vers le dossier du run (colonne 'run_dir' attendue dans ton df)
            def _mk_link(p):
                try:
                    return f'<a href="file:///{Path(p).as_posix()}" target="_blank">{Path(p).name}</a>'
                except Exception:
                    return ""

            if "run_dir" in df.columns:
                df["run"] = df["run_dir"].map(_mk_link)

            # colonnes à devant
            wanted = [
                "ts",
                "exp",
                "model",
                "features",
                "acc",
                "bal_acc",
                "f1_macro",
                "auc_macro",
                "ap_macro",
                "run",
            ]
            cols = [c for c in wanted if c in df.columns]

            # style: surligner meilleure bal_acc
            if "bal_acc" in df.columns:
                best = df["bal_acc"].max()

                def _hl(v):
                    return (
                        "background-color:#c3f7c3;font-weight:bold" if v == best else ""
                    )

                styler = (
                    df[cols]
                    .style.map(
                        lambda v: (
                            "background-color:#c3f7c3;font-weight:bold"
                            if (
                                isinstance(v, (int, float)) and v == df["bal_acc"].max()
                            )
                            else ""
                        ),
                        subset=pd.IndexSlice[:, ["bal_acc"]],
                    )
                    .format(precision=3)
                )
            else:
                styler = df[cols].style.format(precision=3)

            display(
                styler.hide(axis="index").set_table_attributes(
                    'class="dataframe table table-striped"'
                )
            )

        # === Explorer de runs =====================================================================

        # Widgets
        runs_refresh_btn = _W.Button(
            description="Rafraîchir",
            icon="rotate-right",
            layout=_W.Layout(width="120px"),
        )
        runs_dropdown = _W.Dropdown(
            options=[("— aucun —", "")],
            value="",
            description="Runs:",
            layout=_W.Layout(width="520px"),
        )
        runs_view_btn = _W.Button(
            description="Voir le run", icon="eye", layout=_W.Layout(width="140px")
        )
        runs_zip_btn = _W.Button(
            description="Zipper le run",
            icon="file-zipper",
            layout=_W.Layout(width="150px"),
        )
        runs_open_btn = _W.Button(
            description="Ouvrir le dossier",
            icon="folder-open",
            layout=_W.Layout(width="170px"),
        )

        runs_box = _W.HBox(
            [
                runs_refresh_btn,
                runs_dropdown,
                runs_view_btn,
                runs_zip_btn,
                runs_open_btn,
            ]
        )

        # Helpers -------------------------------------------------------------------

        def _find_run_dirs() -> list[Path]:
            base = Path(self.reports_dir)
            if not base.exists():
                return []
            # Un run = un dossier de type YYYYMMDDTHHMMSSZ
            return sorted([p for p in base.iterdir() if p.is_dir()], reverse=True)

        def _session_json_path(run_dir: Path) -> Path | None:
            try:
                cand = list(run_dir.glob("session_report_*.json"))
                return cand[0] if cand else None
            except Exception:
                return None

        def _load_session_summary(run_dir: Path) -> dict:
            """
            Charge un résumé de session à partir de son dossier.

            Args:
                run_dir: Dossier de la session (ex: `reports/2025…Z/`).

            Returns:
                Dictionnaire récapitulatif minimal (modèle, scores, notes, etc.).
            """
            out = {
                "run_dir": str(run_dir),
                "timestamp": run_dir.name,
                "model": None,
                "acc": None,
                "bal_acc": None,
                "f1_macro": None,
                "notes": "",
            }
            p = _session_json_path(run_dir)
            if p and p.exists():
                try:
                    js = json.loads(p.read_text(encoding="utf-8"))
                    out["model"] = js.get("model_type") or js.get("model") or None
                    out["acc"] = js.get("accuracy") or js.get("acc") or None
                    out["bal_acc"] = (
                        js.get("balanced_accuracy") or js.get("bal_acc") or None
                    )
                    out["f1_macro"] = js.get("macro_f1") or js.get("f1_macro") or None
                    out["notes"] = js.get("notes", "")
                except Exception:
                    pass
            return out

        def _refresh_runs(_=None):
            """
            Scanne `reports_dir` et reconstruit la liste des sessions dans le
            dropdown, puis rafraîchit la table des runs via un output unique.

            Notes:
                Utilisé par l’onglet “Explorer les runs”.
            """
            dirs = _find_run_dirs()
            options = [("— aucun —", "")]
            for d in dirs:
                s = _load_session_summary(d)
                label = (
                    f'{d.name} · {s["model"] or "?"} · bal_acc={s["bal_acc"]:.3f}'
                    if s["bal_acc"] is not None
                    else d.name
                )
                options.append((label, str(d)))
            runs_dropdown.options = options
            if len(options) > 1:
                runs_dropdown.value = options[1][1]
            else:
                runs_dropdown.value = ""
            # Met à jour la table des runs sans empilement
            self._refresh_runs_table()

        def _open_folder(_=None):
            val = runs_dropdown.value
            if not val:
                print("Pas de run sélectionné.")
                return
            p = Path(val)
            print(f"Dossier: {p}")
            try:
                if os.name == "nt":
                    os.startfile(str(p))
                elif sys.platform == "darwin":
                    os.system(f'open "{p}"')
                else:
                    os.system(f'xdg-open "{p}"')
            except Exception as e:
                print(f"(Info) Impossible d’ouvrir automatiquement: {e}")

        def _zip_run(_=None):
            val = runs_dropdown.value
            if not val:
                print("Pas de run sélectionné.")
                return
            run_dir = Path(val)
            zip_path = run_dir.with_suffix("")
            try:
                archive = shutil.make_archive(
                    str(zip_path), "zip", root_dir=str(run_dir)
                )
                print(f"Archive créée: {archive}")
            except Exception as e:
                print(f"Erreur zip: {e}")

        def _show_run(_=None):
            val = runs_dropdown.value
            if not val:
                print("Pas de run sélectionné.")
                return
            run_dir = Path(val)
            info = _load_session_summary(run_dir)

            # Bandeau résumé
            display(
                Markdown(
                    f"### Run `{run_dir.name}` — **{info.get('model','?')}**  \n"
                    f"- Accuracy: **{info.get('acc','?')}** &nbsp;&nbsp; "
                    f"- Balanced acc: **{info.get('bal_acc','?')}** &nbsp;&nbsp; "
                    f"- Macro F1: **{info.get('f1_macro','?')}**  \n"
                    f"- Notes: _{(info.get('notes') or '').strip() or '—'}_  \n"
                    f"- Dossier: `{run_dir}`"
                )
            )

            # Affiche les figures (on prend les fichiers classiques si présents)
            fig_names = [
                "confusion_matrix_*.png",
                "roc_*.png",
                "pr_*.png",
                "calibration_*.png",
                "feature_importance_*.png",
            ]
            any_img = False
            for pat in fig_names:
                for p in sorted(run_dir.glob(pat)):
                    any_img = True
                    display(Image(filename=str(p), embed=True))
            if not any_img:
                print("(Aucune figure trouvée dans ce run)")

            # Aperçu du CSV de prédictions s'il existe
            preds = list(run_dir.glob("test_predictions_*.csv"))
            if preds:
                try:
                    dfp = pd.read_csv(preds[0])
                    display(Markdown("**Aperçu des prédictions (top 20)**"))
                    display(dfp.head(20))
                except Exception as e:
                    print(f"(Info) Impossible de lire {preds[0].name}: {e}")

        # Events
        runs_refresh_btn.on_click(_refresh_runs)
        runs_view_btn.on_click(_show_run)
        runs_zip_btn.on_click(_zip_run)
        runs_open_btn.on_click(_open_folder)

        # Affichage dans le layout principal (place cette ligne où tu affiches tes autres boxes)
        display(_W.HTML("<hr>"))
        display(_W.HTML("<h3>🔎 Explorer les runs</h3>"))
        display(runs_box)
        # Zone d'affichage unique pour la table des runs
        display(self._runs_out)

        # première population : remplit le dropdown et rafraîchit la table
        _refresh_runs()

        def _on_run(_):
            # UI state
            run_btn.disabled = True
            run_btn.button_style = "warning"
            run_btn.icon = "hourglass"
            run_btn.description = "En cours…"

            with out:
                clear_output(wait=True)
                print("Démarrage de l'entraînement…")

            # 0) Vérifie que des features sont chargées
            if getattr(self, "features_df", None) is None or self.features_df.empty:
                with out:
                    print(
                        "(erreur) Aucun dataset de features chargé. "
                        "Utilise le bloc 'Features' (Dernier CSV ou Choisir CSV) "
                        "ou exécute d'abord les étapes 1–3."
                    )
                # Restore bouton et stop
                run_btn.disabled = False
                run_btn.button_style = "success"
                run_btn.icon = "play"
                run_btn.description = "Lancer l'entraînement"
                return

            try:
                cfg = _current_config_json()

                # Validation rapide de la cible
                tcol = cfg.get("prediction_target")
                if tcol not in self.features_df.columns:
                    possibles = [
                        c
                        for c in self.features_df.columns
                        if str(self.features_df[c].dtype) in ("object", "category")
                    ]
                    with out:
                        print(
                            f"(erreur) La colonne cible {tcol!r} est absente de features_df."
                        )
                        if possibles:
                            print(
                                "Colonnes catégorielles disponibles :",
                                ", ".join(possibles[:12]),
                                "…",
                            )
                    return

                # Petit résumé textuel dans la zone "Résumé"
                try:
                    import json as _json

                    summary.value = _json.dumps(
                        {
                            "model": cfg.get("model_type"),
                            "target": cfg.get("prediction_target"),
                            "cv_folds": cfg.get("cv_folds"),
                            "search": cfg.get("search"),
                            "scoring": cfg.get("scoring"),
                            "n_features": int(
                                self.features_df.select_dtypes(
                                    include=["number"]
                                ).shape[1]
                            ),
                            "n_rows": int(len(self.features_df)),
                        },
                        indent=2,
                    )
                except Exception:
                    pass

                # Lancement effectif
                with out:
                    print("Entraînement…")
                self.run_training_session(**cfg)

                # Rafraîchit le tableau des runs pour voir tout de suite le nouveau dossier
                try:
                    _refresh_runs()
                except Exception:
                    pass

                with out:
                    print("Terminé.")

            finally:
                # Restore button
                run_btn.disabled = False
                run_btn.button_style = "success"
                run_btn.icon = "play"
                run_btn.description = "Lancer l'entraînement"
                try:
                    # Bascule sur l’onglet “Lancer” si l’utilisateur n’y est plus
                    tabs.selected_index = len(tabs.children) - 1
                except Exception:
                    pass

        run_btn.on_click(_on_run)

    def _log_and_report(
        self,
        clf: SpectralClassifier,
        feature_cols: list[str] | None,
        X: pd.DataFrame,
        y: np.ndarray,
        processed_files: list[str],
        groups: np.ndarray | None,
        save_confusion_png: bool = False,
        save_curves_roc_pr: bool = False,
        save_calibration: bool = False,
        save_feature_importance: bool = False,
        export_test_predictions: bool = False,
        cm_normalized: bool = False,
        exp_name: str | None = None,
        notes: str = "",
        fi_n_repeats: int = 10,
    ) -> str | None:
        """
        Sauvegarde le modèle, calcule les métriques et génère les artefacts de session.

        Args:
            clf (SpectralClassifier): Classifieur entraîné.
            feature_cols (list[str] | None): Noms des colonnes de features utilisées
                après sélection.
            X (pandas.DataFrame): Matrice de features pour l’évaluation/test.
            y (numpy.ndarray): Cibles vraies alignées avec ``X``.
            processed_files (list[str]): Liste de fichiers FITS traités durant
                la session.
            groups (numpy.ndarray | None): Tableau des groupes utilisé pour la CV,
                ou ``None`` si non applicable.
            save_confusion_png (bool): Sauvegarder la matrice de confusion au format PNG.
            save_curves_roc_pr (bool): Sauvegarder les courbes ROC et PR multi‑classes.
            save_calibration (bool): Sauvegarder la courbe de calibration.
            save_feature_importance (bool): Sauvegarder un diagramme d’importance des features.
            export_test_predictions (bool): Exporter un CSV des prédictions sur le jeu de test.
            cm_normalized (bool): Normaliser la matrice de confusion par ligne.
            exp_name (str | None): Nom lisible de l’expérience à inclure dans le rapport JSON.
            notes (str): Notes libres à inclure dans le rapport JSON.
            fi_n_repeats (int): Nombre de répétitions pour la permutation d’importance des features.

        Returns:
            str | None: Le chemin du dossier de run créé (p. ex. ``reports/20250827T162302Z/``)
            ou ``None`` en cas d’erreur.

        Side Effects:
            Écrit sur disque : le modèle sérialisé (.pkl), un fichier de métadonnées JSON,
            des figures (matrice de confusion, courbes ROC/PR, calibration), un
            rapport JSON résumant la session et, éventuellement, un CSV des prédictions.
        """
        from sklearn.metrics import (
            balanced_accuracy_score as sk_balanced_accuracy_score,
        )

        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        model_name = f"spectral_classifier_{clf.model_type.lower()}_{ts}.pkl"
        run_dir = os.path.join(self.reports_dir, ts)
        os.makedirs(run_dir, exist_ok=True)
        model_path = os.path.join(self.models_dir, model_name)

        # 7.1 Sauvegarde du modèle
        try:
            joblib.dump(clf, model_path)
            print(f"  > Modèle sauvegardé dans : {model_path}")
        except Exception as e:
            print(f"  (avertissement) Échec sauvegarde modèle : {e}")
            model_path = None

        # 7.2 Métadonnées
        try:
            import sklearn
            import xgboost

            skver = getattr(sklearn, "__version__", None)
            xgver = getattr(xgboost, "__version__", None)
        except Exception:
            skver, xgver = None, None

        meta = {
            "saved_at_utc": ts,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit_learn": skver,
            "xgboost": xgver if clf.model_type == "XGBoost" else None,
            "model_type": clf.model_type,
            "prediction_target": getattr(clf, "prediction_target", None),
            "best_params_": getattr(clf, "best_params_", None),
            "class_labels": (
                clf.class_labels.tolist()
                if hasattr(clf.class_labels, "tolist")
                else list(clf.class_labels)
            ),
            "feature_names_used": (
                list(feature_cols) if feature_cols is not None else None
            ),
            "selected_features_": getattr(clf, "selected_features_", None),
            "trained_on_file": (
                os.path.basename(self.last_features_path)
                if getattr(self, "last_features_path", None)
                else None
            ),
            "n_candidate_features": (
                int(len(feature_cols)) if feature_cols is not None else None
            ),
            "run_dir": run_dir,
            "exp_name": exp_name,
        }
        meta_filename = f"spectral_classifier_{clf.model_type.lower()}_{ts}_meta.json"
        meta_path = os.path.join(self.models_dir, meta_filename)
        try:
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2, default=_json_default)
        except Exception as e:
            print(f"  (avertissement) Échec d’écriture des métadonnées : {e}")

        # Copie le modèle et ses métadonnées dans le dossier de run pour un accès rapide
        try:
            if os.path.exists(model_path):
                shutil.copy2(
                    model_path, os.path.join(run_dir, os.path.basename(model_path))
                )
            if os.path.exists(meta_path):
                shutil.copy2(
                    meta_path, os.path.join(run_dir, os.path.basename(meta_path))
                )
        except Exception as e:
            print(
                f"  (avertissement) Échec de la copie des artefacts dans le dossier de run : {e}"
            )

        # 7.3 Hash MD5 du modèle
        model_hash = "N/A"
        try:
            if os.path.exists(model_path):
                with open(model_path, "rb") as f:
                    model_hash = hashlib.md5(f.read()).hexdigest()
                print(f"  > Hash MD5 du modèle : {model_hash}")
        except Exception as e:
            print(f"  (avertissement) Impossible de calculer le hash : {e}")

        # 7.4 Metrics
        report_dict, cm, accuracy = None, None, None

        try:
            if hasattr(clf, "_split_info") and "te_idx" in clf._split_info:
                te_idx = clf._split_info["te_idx"]
                X_te = X.iloc[te_idx] if hasattr(X, "iloc") else X[te_idx]
                y_te = np.asarray(y)[te_idx]
            else:
                X_te, y_te = X, np.asarray(y)

            y_pred = clf.model_pipeline.predict(X_te)

            if getattr(clf, "label_encoder", None) is not None:
                n_classes = len(clf.class_labels)
                all_labels = np.arange(n_classes)
                y_te_enc = clf.label_encoder.transform(y_te)
                report_dict = classification_report(
                    y_te_enc,
                    y_pred,
                    labels=all_labels,
                    target_names=list(clf.class_labels),
                    zero_division=0,
                    output_dict=True,
                )
                cm = confusion_matrix(y_te_enc, y_pred, labels=all_labels)
            else:
                report_dict = classification_report(
                    y_te,
                    y_pred,
                    labels=list(clf.class_labels),
                    zero_division=0,
                    output_dict=True,
                )
                cm = confusion_matrix(y_te, y_pred, labels=list(clf.class_labels))

            accuracy = float(report_dict.get("accuracy", 0.0))

        except Exception as e:
            print(f"  (avertissement) Échec calcul métriques : {e}")
            report_dict, cm, accuracy = None, None, None

        if getattr(clf, "label_encoder", None) is not None:
            y_te_enc = clf.label_encoder.transform(y_te)
            y_true_for_scores = y_te_enc
        else:
            y_true_for_scores = y_te

        # Assure le même type des deux côtés
        y_pred_arr = np.asarray(y_pred)

        # Valeur par défaut au cas où tout échoue
        bal_acc_val = float("nan")

        try:
            # voie normale
            bal_acc_val = float(
                sk_balanced_accuracy_score(y_true_for_scores, y_pred_arr)
            )
        except Exception:
            # fallback en castant en str si besoin
            try:
                bal_acc_val = float(
                    sk_balanced_accuracy_score(
                        np.asarray(y_true_for_scores).astype(str),
                        y_pred_arr.astype(str),
                    )
                )
            except Exception as e:
                print(f"(warn) balanced_accuracy_score failed in report: {e}")

        # Récupération des probabilités prédictives si disponibles. Certaines
        # implémentations (ex: SVM) nécessitent que `probability=True` soit
        # activé à la construction du modèle. En cas d'échec, proba reste None.
        proba = None
        try:
            if hasattr(clf.model_pipeline, "predict_proba"):
                proba = clf.model_pipeline.predict_proba(X_te)
            else:
                last_est = clf.model_pipeline[-1]
                if hasattr(last_est, "predict_proba"):
                    proba = last_est.predict_proba(X_te)
        except Exception as e:
            print(f"(avertissement) Probabilités indisponibles : {e}")

        # Sauvegarde PNG de la matrice de confusion (après calcul de cm)
        if save_confusion_png and cm is not None:
            try:
                import matplotlib.pyplot as plt
                import seaborn as sns

                # Optionnel : normaliser la matrice sur les lignes si demandé
                cm_plot = cm.astype(float)
                fmt = "d"
                if cm_normalized:
                    # Normalisation par la somme de chaque ligne (évite division par zéro)
                    cm_plot = cm_plot / (cm_plot.sum(axis=1, keepdims=True) + 1e-12)
                    fmt = ".2f"

                fig = plt.figure(figsize=(8, 6))
                sns.heatmap(
                    cm_plot,
                    annot=True,
                    fmt=fmt,
                    cmap="Blues",
                    xticklabels=list(clf.class_labels),
                    yticklabels=list(clf.class_labels),
                )
                plt.xlabel("Prédiction")
                plt.ylabel("Vraie valeur")
                plt.title(f"Matrice de confusion — {clf.model_type}")
                out_png = os.path.join(
                    run_dir,
                    f"confusion_matrix_{clf.model_type.lower()}_{ts}.png",
                )
                fig.tight_layout()
                fig.savefig(out_png, dpi=140)
                plt.close(fig)
                print(f"  > Heatmap sauvegardée : {out_png}")
            except Exception as e:
                print(f"  (avertissement) Échec sauvegarde heatmap : {e}")

        # === Courbes ROC/PR et métriques supplémentaires ===
        # Ces blocs s'exécutent avant la génération du rapport JSON afin
        # d'inclure les résultats dans le dictionnaire session_report. Les
        # variables `roc_auc_results`, `avg_precision_results` et
        # `brier_score_results` sont initialisées à None et mises à jour si
        # l'option correspondante est activée et que les probabilités sont
        # disponibles.
        roc_auc_results = None
        avg_precision_results = None
        brier_score_results = None

        # A) ROC & PR : One-vs-rest
        if save_curves_roc_pr and proba is not None and y_true_for_scores is not None:
            try:
                import matplotlib.pyplot as plt
                from sklearn.preprocessing import label_binarize
                from sklearn.metrics import (
                    roc_curve,
                    auc,
                    precision_recall_curve,
                    average_precision_score,
                )

                classes = list(clf.class_labels)
                n_classes = len(classes)

                # Encodage des labels pour label_binarize : valeurs 0..n-1
                # Si y_true_for_scores contient déjà des entiers, on les utilise
                # sinon, on mappe chaque libellé vers son index.
                if len(classes) > 0:
                    if isinstance(y_true_for_scores[0], (int, np.integer)):
                        encoded_y_true = np.asarray(y_true_for_scores)
                    else:
                        label_map = {lab: idx for idx, lab in enumerate(classes)}
                        encoded_y_true = np.array(
                            [label_map.get(lab, -1) for lab in y_true_for_scores]
                        )
                else:
                    encoded_y_true = np.asarray(y_true_for_scores)

                Y = label_binarize(encoded_y_true, classes=np.arange(n_classes))

                # ROC par classe
                fpr, tpr, roc_auc = {}, {}, {}
                for i in range(n_classes):
                    fpr[i], tpr[i], _ = roc_curve(Y[:, i], proba[:, i])
                    roc_auc[i] = auc(fpr[i], tpr[i])

                # micro/macro ROC
                fpr["micro"], tpr["micro"], _ = roc_curve(Y.ravel(), proba.ravel())
                roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
                all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
                mean_tpr = np.zeros_like(all_fpr)
                for i in range(n_classes):
                    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
                mean_tpr /= n_classes
                roc_auc["macro"] = auc(all_fpr, mean_tpr)

                # Plot ROC curves
                fig = plt.figure(figsize=(8, 6))
                for i, lab in enumerate(classes):
                    plt.plot(
                        fpr[i],
                        tpr[i],
                        lw=1,
                        label=f"{lab} (AUC={roc_auc[i]:.2f})",
                    )
                plt.plot([0, 1], [0, 1], "--", lw=1, color="royalblue")
                plt.plot(
                    all_fpr,
                    mean_tpr,
                    lw=2,
                    label=f"macro (AUC={roc_auc['macro']:.2f})",
                )
                plt.plot(
                    fpr["micro"],
                    tpr["micro"],
                    lw=2,
                    label=f"micro (AUC={roc_auc['micro']:.2f})",
                )
                plt.xlabel("False Positive Rate")
                plt.ylabel("True Positive Rate")
                plt.title(f"ROC — {clf.model_type}")
                plt.legend(fontsize=8, loc="lower right")
                roc_png = os.path.join(
                    run_dir,
                    f"roc_{clf.model_type.lower()}_{ts}.png",
                )
                fig.tight_layout()
                fig.savefig(roc_png, dpi=140)
                plt.close(fig)
                print(f"  > ROC sauvegardée : {roc_png}")

                # Courbes Precision–Recall
                ap = {}
                fig = plt.figure(figsize=(8, 6))
                for i, lab in enumerate(classes):
                    p, r, _ = precision_recall_curve(Y[:, i], proba[:, i])
                    ap[lab] = average_precision_score(Y[:, i], proba[:, i])
                    plt.plot(r, p, lw=1, label=f"{lab} (AP={ap[lab]:.2f})")
                plt.xlabel("Recall")
                plt.ylabel("Precision")
                plt.title(f"Precision–Recall — {clf.model_type}")
                plt.legend(fontsize=8, loc="lower left")
                pr_png = os.path.join(
                    run_dir,
                    f"pr_{clf.model_type.lower()}_{ts}.png",
                )
                fig.tight_layout()
                fig.savefig(pr_png, dpi=140)
                plt.close(fig)
                print(f"  > PR sauvegardée : {pr_png}")

                roc_auc_results = {
                    "micro": float(roc_auc["micro"]),
                    "macro": float(roc_auc["macro"]),
                    **{classes[i]: float(roc_auc[i]) for i in range(n_classes)},
                }
                avg_precision_results = {k: float(v) for k, v in ap.items()}
            except Exception as e:
                print(f"  (avertissement) Courbes ROC/PR ignorées : {e}")

        # B) Calibration curve & Brier score
        if save_calibration and proba is not None and y_true_for_scores is not None:
            try:
                import matplotlib.pyplot as plt
                from sklearn.calibration import calibration_curve
                from sklearn.metrics import brier_score_loss

                classes = list(clf.class_labels)
                n_classes = len(classes)
                # Encodage des labels
                if n_classes > 0:
                    if isinstance(y_true_for_scores[0], (int, np.integer)):
                        encoded_y_true = np.asarray(y_true_for_scores)
                    else:
                        label_map = {lab: idx for idx, lab in enumerate(classes)}
                        encoded_y_true = np.array(
                            [label_map.get(lab, -1) for lab in y_true_for_scores]
                        )
                else:
                    encoded_y_true = np.asarray(y_true_for_scores)
                # Binarisé pour calibration_curve
                from sklearn.preprocessing import label_binarize as _label_binarize

                Y = _label_binarize(encoded_y_true, classes=np.arange(n_classes))

                fig = plt.figure(figsize=(8, 6))
                brier = {}
                for i, lab in enumerate(classes):
                    frac_pos, mean_pred = calibration_curve(
                        Y[:, i], proba[:, i], n_bins=15, strategy="quantile"
                    )
                    plt.plot(
                        mean_pred,
                        frac_pos,
                        marker="o",
                        lw=1,
                        label=lab,
                    )
                    brier[lab] = brier_score_loss(Y[:, i], proba[:, i])
                plt.plot([0, 1], [0, 1], "--", color="grey")
                plt.xlabel("Probabilité prédite")
                plt.ylabel("Fréquence observée")
                plt.title(f"Calibration — {clf.model_type}")
                plt.legend(fontsize=8, loc="best")
                calib_png = os.path.join(
                    run_dir,
                    f"calibration_{clf.model_type.lower()}_{ts}.png",
                )
                fig.tight_layout()
                fig.savefig(calib_png, dpi=140)
                plt.close(fig)
                print(f"  > Calibration sauvegardée : {calib_png}")
                brier_score_results = {k: float(v) for k, v in brier.items()}
            except Exception as e:
                print(f"  (avertissement) Courbe de calibration ignorée : {e}")

        # C) Importances de features
        if save_feature_importance:
            try:
                import matplotlib.pyplot as plt
                from sklearn.inspection import permutation_importance

                names = getattr(clf, "selected_features_", None) or feature_cols
                # Récupère l'estimateur final (dernier élément du pipeline)
                estimator = clf.model_pipeline[-1]
                importances = None
                # Cas 1 : l'estimateur expose feature_importances_
                if hasattr(estimator, "feature_importances_"):
                    importances = estimator.feature_importances_
                # Cas 2 : SVM linéaire → importance ~ poids absolu moyen
                elif (
                    clf.model_type == "SVM"
                    and getattr(estimator, "kernel", "rbf") == "linear"
                    and hasattr(estimator, "coef_")
                ):
                    importances = np.abs(estimator.coef_).mean(axis=0)

                # Cas 3 : fallback par permutation importance
                if importances is None:
                    # Utilise le balanced_accuracy comme scoring par défaut
                    res = permutation_importance(
                        clf.model_pipeline,
                        X_te,
                        y_true_for_scores,
                        n_repeats=int(max(1, fi_n_repeats)),
                        random_state=(
                            meta.get("random_state", 42)
                            if isinstance(meta, dict)
                            else 42
                        ),
                        scoring="balanced_accuracy",
                    )
                    importances = res.importances_mean

                k = min(20, len(names))
                idx = np.argsort(importances)[::-1][:k]
                fig = plt.figure(figsize=(8, 6))
                plt.barh(range(k), importances[idx][::-1])
                plt.yticks(range(k), [names[i] for i in idx][::-1], fontsize=8)
                plt.xlabel("Importance")
                plt.title(f"Top-{k} features — {clf.model_type}")
                fi_png = os.path.join(
                    run_dir,
                    f"feature_importance_{clf.model_type.lower()}_{ts}.png",
                )
                fig.tight_layout()
                fig.savefig(fi_png, dpi=140)
                plt.close(fig)
                print(f"  > Feature importance sauvegardée : {fi_png}")
            except Exception as e:
                print(f"  (avertissement) Feature importance ignorée : {e}")

        # D) Export des prédictions test
        if export_test_predictions:
            try:
                import pandas as pd

                classes = list(clf.class_labels)
                n_samples = len(y_pred)
                # Prépare un mapping label->index
                label_map = {lab: idx for idx, lab in enumerate(classes)}
                # Encodage des prédictions pour indexation
                if isinstance(y_pred[0], (int, np.integer)):
                    y_pred_enc = np.asarray(y_pred)
                else:
                    y_pred_enc = np.array([label_map.get(lab, -1) for lab in y_pred])
                # Encodage des y_true pour export
                if isinstance(y_true_for_scores[0], (int, np.integer)):
                    y_true_enc = np.asarray(y_true_for_scores)
                else:
                    y_true_enc = np.array(
                        [label_map.get(lab, -1) for lab in y_true_for_scores]
                    )

                df_export = pd.DataFrame(
                    {
                        "y_true": [
                            classes[int(t)] if int(t) >= 0 else str(t)
                            for t in y_true_enc
                        ],
                        "y_pred": [
                            classes[int(p)] if int(p) >= 0 else str(p)
                            for p in y_pred_enc
                        ],
                    }
                )
                if proba is not None and proba.shape[1] >= 2:
                    # Top-2 classes par probabilité
                    top2 = np.argsort(proba, axis=1)[:, -2:][:, ::-1]
                    idx = np.arange(len(y_pred_enc))
                    mask = y_pred_enc >= 0
                    # Probabilité de la classe prédite
                    df_export["proba_pred"] = np.nan
                    df_export.loc[mask, "proba_pred"] = proba[
                        idx[mask], y_pred_enc[mask]
                    ]
                    df_export["top1"] = [classes[i] for i in top2[:, 0]]
                    df_export["p_top1"] = proba[np.arange(len(y_pred_enc)), top2[:, 0]]
                    df_export["top2"] = [classes[i] for i in top2[:, 1]]
                    df_export["p_top2"] = proba[np.arange(len(y_pred_enc)), top2[:, 1]]
                csv_path = os.path.join(
                    run_dir,
                    f"test_predictions_{clf.model_type.lower()}_{ts}.csv",
                )
                df_export.to_csv(csv_path, index=False)
                print(f"  > Prédictions test exportées : {csv_path}")
            except Exception as e:
                print(f"  (avertissement) Export prédictions ignoré : {e}")

        # E) Historique d'early stopping pour XGBoost
        if clf.model_type == "XGBoost":
            try:
                booster = clf.model_pipeline[-1].get_booster()
                hist = booster.evals_result()
                xgb_hist_json = os.path.join(run_dir, f"xgb_eval_history_{ts}.json")
                with open(xgb_hist_json, "w", encoding="utf-8") as f:
                    json.dump(hist, f, indent=2)
                print(f"  > Historique XGB sauvegardé : {xgb_hist_json}")
            except Exception:
                pass

        # 7.5 Rapport JSON
        session_report = {
            "session_id": ts,
            "model_type": clf.model_type,
            "model_path": model_path,
            "model_hash_md5": model_hash,
            "total_spectra_processed": int(len(processed_files or [])),
            "training_set_size": (
                int(clf._split_info["n_train"]) if hasattr(clf, "_split_info") else None
            ),
            "test_set_size": (
                int(clf._split_info["n_test"])
                if hasattr(clf, "_split_info")
                else (int(len(X_te)) if "X_te" in locals() else None)
            ),
            "feature_columns": list(feature_cols) if feature_cols is not None else None,
            "selected_features": getattr(clf, "selected_features_", None),
            "class_labels": (
                clf.class_labels.tolist()
                if hasattr(clf.class_labels, "tolist")
                else list(clf.class_labels)
            ),
            "best_model_params": getattr(clf, "best_params_", None),
            "accuracy": accuracy,
            "classification_report": report_dict,
            "confusion_matrix": cm.tolist() if cm is not None else None,
            "exp_name": exp_name,
            "notes": notes,
            "balanced_accuracy": float(bal_acc_val),
        }
        # Si un filtre de classes a été utilisé, on l'enregistre dans le rapport
        try:
            if getattr(self, "_last_class_filter", None):
                session_report["class_filter"] = self._last_class_filter
        except Exception:
            pass

        # Insère les métriques supplémentaires (ROC AUC, Average Precision,
        # Brier score) si disponibles. Les dictionnaires sont convertis en
        # nombres flottants pour assurer la sérialisation JSON.
        if roc_auc_results is not None:
            session_report["roc_auc"] = roc_auc_results
        if avg_precision_results is not None:
            session_report["avg_precision"] = avg_precision_results
        if brier_score_results is not None:
            session_report["brier_score"] = brier_score_results

        # Macro AUC pour compatibilité avec l'ancien comparateur
        try:
            if (
                roc_auc_results
                and isinstance(roc_auc_results, dict)
                and "macro" in roc_auc_results
            ):
                session_report["roc_auc_macro"] = float(roc_auc_results["macro"])
        except Exception:
            pass

        report_filename = f"session_report_{clf.model_type.lower()}_{ts}.json"
        report_path = os.path.join(run_dir, report_filename)
        try:
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(session_report, f, indent=4, default=_json_default)
            print(f"\nRapport de session sauvegardé dans : {report_path}")
        except Exception as e:
            print(
                f"  (avertissement) Échec lors de la génération du rapport de session : {e}"
            )

        # --- AFFICHAGE RÉSUMÉ DANS LA CELLULE ---
        try:
            from sklearn.metrics import balanced_accuracy_score

            y_true_for_scores = (
                y_te_enc
                if "y_te_enc" in locals()
                and clf.model_type in {"MLP", "KNN", "LogRegOVR", "SVM", "SoftVoting"}
                else y_te
            )
            bal_acc = (
                balanced_accuracy_score(y_true_for_scores, y_pred)
                if y_true_for_scores is not None
                else None
            )

            print("\n=== RÉSULTATS (jeu test) ===")
            bal_acc = bal_acc_val if "bal_acc_val" in locals() else None
            if report_dict is not None:
                acc = report_dict.get("accuracy", None)
                macro_f1 = (report_dict.get("macro avg", {}) or {}).get(
                    "f1-score", None
                )
                if acc is not None:
                    print(f"Accuracy          : {acc:.2%}")
                if bal_acc is not None:
                    print(f"Balanced accuracy : {bal_acc:.2%}")
                if macro_f1 is not None:
                    print(f"Macro F1          : {macro_f1:.3f}")

                print("\nPar classe :")
                for cls_name, row in report_dict.items():
                    if cls_name in ("accuracy", "macro avg", "weighted avg"):
                        continue
                    p = row.get("precision")
                    r = row.get("recall")
                    f1 = row.get("f1-score")
                    if p is not None and r is not None and f1 is not None:
                        print(f"  - {cls_name:<6}  P={p:.2f}  R={r:.2f}  F1={f1:.2f}")
            else:
                print("  (pas de rapport de classification disponible)")
        except Exception as e:
            print(f"(avertissement) Affichage métriques ignoré : {e}")

        # --- RÉCAP SELECTION DE FEATURES ---
        try:
            if getattr(clf, "selected_features_", None) is not None:
                kept = len(clf.selected_features_)
                # le bon nom est "feature_cols" (paramètre de la fonction)
                total = len(feature_cols) if feature_cols is not None else None
                msg = f"[FS] Features conservées : {kept}" + (
                    f"/{total}" if total is not None else ""
                )
                print("\n" + msg)
        except Exception as e:
            print(f"(avertissement) Récap FS ignoré : {e}")

        print("\nSESSION DE RECHERCHE TERMINÉE")
        return report_path
