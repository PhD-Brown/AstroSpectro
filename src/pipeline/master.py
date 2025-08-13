"""AstroSpectro — Pipeline maître (orchestrateur).

Ce module expose la classe `MasterPipeline`, qui orchestre l’intégralité du
flux de traitement des spectres :

1) Scan et catalogage des fichiers FITS                           (DatasetBuilder)
2) Enrichissement astrométrique/photométrique via Gaia            (gaia_crossmatcher)
3) Prétraitement + extraction de features                         (ProcessingPipeline + FeatureEngineer)
4) Entraînement / évaluation d’un classifieur                     (SpectralClassifier)
5) Journalisation et génération des artefacts (CSV, modèles, logs)

Entrées / sorties attendues
---------------------------
- `paths`: dict des répertoires clés (p. ex. "RAW_DATA_DIR", "PROCESSED_DIR",
  "CATALOG_DIR", "MODELS_DIR", "LOG_DIR").
- Écrit typiquement :
  * features CSV : `processed/features_YYYYMMDDTHHMMSSZ.csv`
  * modèles      : `data/models/*.pkl`
  * journaux     : `logs/*.jsonl` (selon configuration)

API publique (principales méthodes)
-----------------------------------
- `generate_and_enrich_catalog(enrich_gaia: bool = True, mode: str = "bulk", ...)`
- `process_data(...)` : prétraitements + extraction des features
- `interactive_training_runner()` : entraînement assisté (Jupyter)
- `interactive_peak_tuner()` : visualisation & réglage des pics (Jupyter)
- `_log_and_report(...)` : utilitaire interne pour la journalisation

Remarques
---------
- Les méthodes préfixées par `_` sont internes.
- Les méthodes `interactive_*` utilisent ipywidgets et sont prévues pour notebook.
- Voir `requirements.txt` pour les dépendances (scikit-learn, imbalanced-learn,
  xgboost, astropy, etc.).

Exemple minimal
---------------
>>> from pipeline.master import MasterPipeline
>>> mp = MasterPipeline(paths)
>>> mp.generate_and_enrich_catalog(enrich_gaia=True, mode="bulk")
>>> mp.process_data()
>>> # Dans un notebook : mp.interactive_training_runner()
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import os
import json
import hashlib
from datetime import datetime, timezone

import pandas as pd
import ipywidgets as widgets
from IPython.display import display, clear_output

# --- Imports projet ---
from tools.dataset_builder import DatasetBuilder
from pipeline.processing import ProcessingPipeline
from pipeline.classifier import SpectralClassifier
from tools.generate_catalog_from_fits import generate_catalog_from_fits
from tools.gaia_crossmatcher import enrich_catalog_with_gaia
from utils import md5sum

# --- Imports Scikit-learn ---
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder


class MasterPipeline:
    """
    Orchestrateur « haut niveau » du workflow :
    1) Sélection d’un lot de spectres,
    2) Génération/Enrichissement du catalogue,
    3) Traitement & extraction des features,
    4) Entraînement + évaluation + journalisation.

    Args:
        raw_data_dir: Dossier racine contenant les spectres bruts (FITS).
        catalog_dir: Dossier de sortie des fichiers catalogue CSV.
        processed_dir: Dossier de sortie des datasets de features CSV.
        models_dir: Dossier de sortie des modèles entraînés.
        reports_dir: Dossier de sortie des rapports JSON.

    Attributes:
        builder: Gestionnaire de données pour lister/choisir les spectres.
        current_batch: Chemins relatifs des spectres sélectionnés.
        master_catalog_df: Catalogue courant en mémoire.
        features_df: Dataset de features courant en mémoire.
        master_catalog_path: Chemin du catalogue local temporaire.
        gaia_catalog_path: Chemin du catalogue enrichi Gaia.
        last_features_path: Dernier fichier de features sauvegardé.
    """

    def __init__(
        self,
        raw_data_dir: str,
        catalog_dir: str,
        processed_dir: str,
        models_dir: str,
        reports_dir: str,
    ) -> None:
        self.raw_data_dir = raw_data_dir
        self.catalog_dir = catalog_dir
        self.processed_dir = processed_dir
        self.models_dir = models_dir
        self.reports_dir = reports_dir

        self.builder = DatasetBuilder(
            raw_data_dir=self.raw_data_dir, catalog_dir=self.catalog_dir
        )

        # État
        self.current_batch: List[str] = []
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

    # --------------------- API publique (Notebook) ---------------------

    def select_batch(
        self, batch_size: int = 500, strategy: str = "random"
    ) -> List[str]:
        """
        Étape 1 : sélectionne un **lot de spectres** à traiter.

        Args:
            batch_size: Nombre de spectres à sélectionner.
            strategy: Stratégie de sélection (ex. "random").

        Returns:
            La liste des chemins **relatifs** des spectres sélectionnés.
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
        Étape 2 : génère le **catalogue local** (depuis les FITS du lot) et,
        si demandé, l’**enrichit** avec Gaia.

        Args:
            enrich_gaia: Si True, lance l’enrichissement Gaia.
            **gaia_kwargs: Paramètres passés à `enrich_catalog_with_gaia`.

        Side effects:
            Met à jour `self.master_catalog_df` et sauvegarde le CSV.
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
        Étape 3 : lance le **pipeline de traitement** et l’extraction des **features**.

        Returns:
            Le DataFrame de features si disponible, sinon None.

        Side effects:
            Met à jour `self.features_df` et `self.last_features_path` et
            sauvegarde un fichier `features_YYYYMMDDTHHMMSSZ.csv` dans `processed_dir`.
        """
        print("\n=== ÉTAPE 3 : TRAITEMENT DES DONNÉES ET EXTRACTION DES FEATURES ===")
        if self.master_catalog_df.empty:
            print(
                "  > Erreur : Le catalogue est vide. Lance `generate_and_enrich_catalog` d’abord."
            )
            return None

        pipeline = ProcessingPipeline(self.raw_data_dir, self.master_catalog_df)
        self.features_df = pipeline.run(self.current_batch)

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

    def run_training_session(
        self,
        model_type: str = "XGBoost",
        n_estimators: int = 200,
        prediction_target: str = "main_class",
        save_and_log: bool = True,
        # Sélection de features (optionnelle)
        use_feature_selection: bool = True,
        selector_model: str = "xgb",  # "xgb" ou "rf"
        selector_threshold: str = "median",  # ex. "median", "mean", "0.5*mean", "g*mean"
        selector_n_estimators: int = 200,
    ) -> Optional[SpectralClassifier]:
        """
        Étape 4 : entraîne un **SpectralClassifier** à partir de `self.features_df`.

        - Tune via GridSearchCV,
        - Applique (optionnel) une sélection de features `SelectFromModel`,
        - Évalue, journalise et sauvegarde le modèle.

        Args:
            model_type: "XGBoost" ou "RandomForest".
            n_estimators: Nombre d’arbres du modèle final.
            prediction_target: Cible ("main_class", "sub_class_top25", "sub_class_bins").
            save_and_log: Sauvegarder le modèle et créer un rapport JSON.
            use_feature_selection: Activer la sélection de features amont.
            selector_model: Modèle du sélecteur ("xgb" / "rf").
            selector_threshold: Seuil de sélection (ex. "median").
            selector_n_estimators: Nombre d’arbres du sélecteur.

        Returns:
            Le classifieur entraîné, ou None si l’entraînement n’a pas eu lieu.
        """
        print(
            f"\n=== ÉTAPE 4 : SESSION D'ENTRAÎNEMENT (Modèle: {model_type}, Cible: {prediction_target}) ==="
        )

        if getattr(self, "features_df", None) is None or self.features_df.empty:
            print(
                "ERREUR : Le DataFrame de features est vide. Veuillez d'abord exécuter `process_data`."
            )
            return

        clf = SpectralClassifier(
            model_type=model_type,
            prediction_target=prediction_target,
            use_feature_selection=use_feature_selection,
            selector_model=selector_model,
            selector_threshold=selector_threshold,
            selector_n_estimators=selector_n_estimators,
        )

        result = clf.train_and_evaluate(self.features_df, n_estimators=n_estimators)
        if not result:
            print(
                "\n--- SESSION TERMINÉE SANS ENTRAÎNEMENT (pas assez de données valides) ---"
            )
            return

        trained_clf, feature_cols_before_fs, X_all, y_all = result

        # Message sur la sélection de features
        if (
            use_feature_selection
            and getattr(trained_clf, "selected_features_", None) is not None
        ):
            kept = len(trained_clf.selected_features_)
            total = len(feature_cols_before_fs)
            print(f"\n[Feature selection] {kept}/{total} features conservées.")
        else:
            print(
                f"\n[Feature selection] non utilisée. {len(feature_cols_before_fs)} features au total."
            )

        # Sauvegarde + rapport
        if save_and_log:
            if "file_path" in self.features_df.columns:
                processed_files = (
                    self.features_df["file_path"].dropna().astype(str).unique().tolist()
                )
            else:
                processed_files = []

            try:
                self._log_and_report(
                    trained_clf,
                    feature_cols_before_fs,
                    X_all,
                    y_all,
                    processed_files,
                )
            except Exception as e:
                print(
                    f"\n(avertissement) Échec lors de la génération du rapport de session : {e}"
                )
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
        Exécute **tout le pipeline** de A à Z (sélection → entraînement).

        Args:
            batch_size: Taille du lot de spectres.
            model_type: Modèle final.
            n_estimators: Nombre d’arbres du modèle.
            prediction_target: Cible de prédiction.
            save_and_log: Si True, sauvegarde et journalise.
            enrich_gaia: Si True, enrichit le catalogue avec Gaia.
            **gaia_kwargs: Paramètres pour l’enrichissement Gaia.
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

    def interactive_training_runner(self) -> None:
        """
        Affiche une **UI Jupyter** (ipywidgets) pour lancer l’entraînement
        avec des paramètres choisis.
        """
        # --- 1) Widgets ---
        model_choice = widgets.Dropdown(
            options=["RandomForest", "XGBoost"],
            value="XGBoost",
            description="Modèle:",
        )
        n_estimators_widget = widgets.IntText(
            value=200,
            description="N Estimators:",
            style={"description_width": "initial"},
        )
        target_choice = widgets.Dropdown(
            options=["main_class", "sub_class_top25", "sub_class_bins"],
            value="main_class",
            description="Cible:",
        )
        save_log_checkbox = widgets.Checkbox(
            value=True, description="Sauvegarder & Journaliser"
        )
        run_button = widgets.Button(
            description="Lancer l'entraînement", button_style="success", icon="cogs"
        )
        output_area = widgets.Output()

        # --- 2) Callback ---
        def on_run_button_clicked(_: widgets.Button) -> None:
            with output_area:
                clear_output(wait=True)
                self.run_training_session(
                    model_type=model_choice.value,
                    n_estimators=n_estimators_widget.value,
                    prediction_target=target_choice.value,
                    save_and_log=save_log_checkbox.value,
                )

        # --- 3) Assemblage & affichage ---
        run_button.on_click(on_run_button_clicked)
        controls = widgets.HBox([model_choice, n_estimators_widget, target_choice])
        actions = widgets.HBox([save_log_checkbox, run_button])
        app_layout = widgets.VBox(
            [controls, actions, output_area],
            layout={"border": "1px solid #444", "padding": "10px", "width": "100%"},
        )
        display(app_layout)

    # --------------------- Journalisation & rapport ---------------------

    def _log_and_report(
        self,
        clf: SpectralClassifier,
        feature_cols: List[str],
        X: pd.DataFrame,
        y: pd.Series | List[str] | pd.DataFrame,
        processed_files: List[str],
    ) -> Optional[str]:
        """
        Étapes 6–7 : **journal** + **rapport** de session.

        - Met à jour le journal des spectres utilisés,
        - Sauvegarde le modèle et son MD5,
        - Calcule les métriques cohérentes (re-split),
        - Sauvegarde un rapport **JSON** complet.

        Args:
            clf: Classifieur entraîné.
            feature_cols: Liste des colonnes candidates (avant sélection).
            X: Features complètes (base de split pour les métriques).
            y: Labels correspondants.
            processed_files: Liste des fichiers traités pour la session.

        Returns:
            Chemin du rapport JSON, ou None si rien n’a été généré.
        """

        if clf is None or not processed_files:
            print(
                "\n> Aucun modèle entraîné ou fichiers traités à consigner. Rapport non généré."
            )
            return None

        print("\n--- ÉTAPE 6 : Mise à jour du Journal des Spectres Utilisés ---")
        try:
            self.builder.update_trained_log(processed_files)
        except Exception as e:
            print(f"  (avertissement) impossible de mettre à jour le journal : {e}")

        print("\n--- ÉTAPE 7 : Génération du Rapport de Session ---")

        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

        # 1) Sauvegarde modèle + méta
        model_fname = f"spectral_classifier_{clf.model_type.lower()}_{ts}.pkl"
        model_path = os.path.join(self.models_dir, model_fname)

        extra_info = {}
        if getattr(self, "last_features_path", None) and os.path.exists(
            self.last_features_path
        ):
            try:
                extra_info["trained_on_file_md5"] = md5sum(self.last_features_path)
                extra_info["n_candidate_features"] = len(feature_cols)
            except Exception:
                pass

        clf.save_model(
            model_path,
            trained_on_file=getattr(self, "last_features_path", None),
            extra_info=extra_info,
        )

        # 2) Hash MD5 du fichier modèle
        model_hash = "N/A"
        try:
            with open(model_path, "rb") as f:
                model_hash = hashlib.md5(f.read()).hexdigest()
            print(f"  > Hash MD5 du modèle : {model_hash}")
        except Exception:
            pass

        # 3) Re-split pour des métriques stables
        rnd = getattr(clf, "random_state", 42)

        if clf.model_type == "XGBoost":
            le = LabelEncoder()
            y_enc = le.fit_transform(y)

            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y_enc, test_size=0.25, random_state=rnd, stratify=y_enc
            )

            y_pred = clf.model_pipeline.predict(X_te)

            class_names = (
                list(clf.class_labels)
                if getattr(clf, "class_labels", None) is not None
                else list(le.classes_)
            )
            label_idx = list(range(len(class_names)))

            report_dict = classification_report(
                y_te,
                y_pred,
                labels=label_idx,
                target_names=class_names,
                zero_division=0,
                output_dict=True,
            )
            cm = confusion_matrix(y_te, y_pred, labels=label_idx)

        else:  # RandomForest (labels texte)
            class_names = (
                list(clf.class_labels)
                if getattr(clf, "class_labels", None) is not None
                else sorted(list(set(y)))
            )

            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=0.25, random_state=rnd, stratify=y
            )
            y_pred = clf.model_pipeline.predict(X_te)

            report_dict = classification_report(
                y_te, y_pred, labels=class_names, zero_division=0, output_dict=True
            )
            cm = confusion_matrix(y_te, y_pred, labels=class_names)

        accuracy = float(report_dict.get("accuracy", 0.0))
        print(f"  > Métriques extraites : Accuracy = {accuracy:.4f}")

        # 4) Importances des features du modèle final (post-sélection le cas échéant)
        final_est = clf.model_pipeline.named_steps.get("clf")
        final_importances = None
        try:
            if hasattr(final_est, "feature_importances_"):
                final_feature_names = (
                    clf.selected_features_
                    if getattr(clf, "selected_features_", None)
                    else feature_cols
                )
                final_importances = sorted(
                    [
                        {"feature": f, "importance": float(v)}
                        for f, v in zip(
                            final_feature_names, final_est.feature_importances_
                        )
                    ],
                    key=lambda d: d["importance"],
                    reverse=True,
                )
        except Exception:
            pass

        # 5) Infos de sélection
        selected_features = (
            list(clf.selected_features_)
            if getattr(clf, "selected_features_", None)
            else []
        )
        n_selected = len(selected_features) if selected_features else None

        # 6) Rapport JSON
        session_report: Dict[str, Any] = {
            "session_id": ts,
            "model_type": clf.model_type,
            "model_path": model_path,
            "model_hash_md5": model_hash,
            "random_state": rnd,
            "total_spectra_processed": int(len(processed_files)),
            "training_set_size": int(len(X_tr)),
            "test_set_size": int(len(X_te)),
            "candidate_feature_columns": list(feature_cols),
            "selected_features": selected_features or None,
            "n_selected_features": n_selected,
            "best_model_params": getattr(clf, "best_params_", None),
            "class_labels": class_names,
            "accuracy": accuracy,
            "classification_report": report_dict,
            "confusion_matrix": cm.tolist(),
        }
        if final_importances is not None:
            session_report["feature_importances_final_clf"] = final_importances

        # 7) Sauvegarde du rapport
        report_fname = f"session_report_{clf.model_type.lower()}_{ts}.json"
        report_path = os.path.join(self.reports_dir, report_fname)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(session_report, f, indent=4)
        print(f"\nRapport de session sauvegardé dans : {report_path}")
        print("\nSESSION DE RECHERCHE TERMINÉE")
        return report_path
