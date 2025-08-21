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

from typing import Any, List, Optional

import os
import json
import hashlib
import platform
from datetime import datetime, timezone

import pandas as pd
import numpy as np
import ipywidgets as W
from IPython.display import display, clear_output

# --- Imports projet ---
from tools.dataset_builder import DatasetBuilder
from pipeline.processing import ProcessingPipeline
from pipeline.classifier import SpectralClassifier
from tools.generate_catalog_from_fits import generate_catalog_from_fits
from tools.gaia_crossmatcher import enrich_catalog_with_gaia

# --- Imports Scikit-learn ---
from sklearn.metrics import classification_report, confusion_matrix


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
        # FS
        use_feature_selection: bool = True,
        selector_model: str = "xgb",
        selector_threshold: str = "median",
        selector_n_estimators: int = 200,
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
        # artefacts
        save_confusion_png: bool = False,
        base_params: dict | None = None,
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

        search = None if search in (None, "None") else search

        result = clf.train_and_evaluate(
            self.features_df,
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
                    groups_all,
                    save_confusion_png=save_confusion_png,
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
        # --- Widgets de base ---
        model_choice = W.Dropdown(
            options=["XGBoost", "RandomForest", "SVM"],
            value="XGBoost",
            description="Modèle:",
        )
        n_estimators_widget = W.IntText(value=200, description="N Estimators:")
        target_choice = W.Dropdown(
            options=["main_class", "sub_class_top25", "sub_class_bins"],
            value="main_class",
            description="Cible:",
        )
        save_log_checkbox = W.Checkbox(
            value=True, description="Sauvegarder & Journaliser"
        )

        # --- Widgets "options avancées" ---
        search_mode = W.Dropdown(
            options=[
                ("Aucun", None),
                ("GridSearchCV", "grid"),
                ("RandomizedSearchCV", "random"),
            ],
            value=None,
            description="Search:",
        )
        cv_folds = W.IntSlider(value=3, min=2, max=10, step=1, description="CV folds")
        scoring = W.Dropdown(
            options=["accuracy", "balanced_accuracy", "f1_macro", "f1_weighted"],
            value="accuracy",
            description="Scoring:",
        )
        n_iter_widget = W.IntSlider(
            value=60, min=10, max=200, step=10, description="n_iter (random)"
        )

        early_stopping = W.Checkbox(value=True, description="Early stopping (XGB)")
        val_size = W.FloatSlider(
            value=0.15,
            min=0.05,
            max=0.40,
            step=0.01,
            readout_format=".2f",
            description="val_size",
        )

        # --- Nouveaux widgets généraux ---
        test_size_slider = W.FloatSlider(
            value=0.21,
            min=0.05,
            max=0.40,
            step=0.01,
            readout_format=".02f",
            description="test_size",
        )
        seed_box = W.IntText(value=42, description="seed")

        # Early stopping (XGB) — nb de rounds
        early_rounds = W.IntSlider(
            value=50, min=10, max=300, step=5, description="ES rounds"
        )

        # Grilles / distributions (JSON)
        param_grid_txt = W.Textarea(
            value="",
            placeholder='{"clf__max_depth":[4,6,8], "clf__min_child_weight":[1,3,5]}',
            description="param_grid (JSON)",
            layout=W.Layout(width="48%", height="84px"),
        )
        param_dist_txt = W.Textarea(
            value="",
            placeholder='{"clf__learning_rate":[0.03,0.05,0.1], "clf__subsample":[0.7,0.9]}',
            description="param_dists (JSON)",
            layout=W.Layout(width="48%", height="84px"),
        )
        base_params_txt = W.Textarea(
            value="",
            placeholder='{"max_depth": 6, "learning_rate": 0.05, "subsample": 0.8}',
            description="base_params (JSON)",
            layout=W.Layout(width="48%", height="80px"),
        )
        selector_n_estimators = W.IntSlider(
            value=400,
            min=50,
            max=1500,
            step=50,
            description="selector_n_estimators",
            readout=True,
        )

        # Poids & calibration
        balanced_weights = W.Checkbox(value=True, description="Balanced weights")
        calibrate_probs = W.Checkbox(value=False, description="Calibrate probs")
        calib_method = W.Dropdown(
            options=["sigmoid", "isotonic"], value="sigmoid", description="calib"
        )

        # Artefacts
        save_cm_png = W.Checkbox(value=False, description="Sauver CM .png")

        use_groups = W.Checkbox(value=False, description="Group split")
        group_col_text = W.Text(value="", description="Col. groupes")

        # Overrides de sélection de features (facultatif)
        fs_override = W.Checkbox(value=False, description="Override Feature Selection")
        fs_enabled = W.Checkbox(value=True, description="use_feature_selection")
        fs_model = W.Dropdown(
            options=[("RandomForest", "rf"), ("XGBoost", "xgb")],
            value="rf",
            description="selector_model",
        )
        fs_threshold = W.Text(value="median", description="selector_threshold")

        run_button = W.Button(
            description="Lancer l'entraînement…", button_style="success"
        )
        output_area = W.Output()

        def on_run_button_clicked(_):
            with output_area:
                clear_output(wait=True)

                def _parse_json(ta):
                    import json

                    try:
                        s = (ta.value or "").strip()
                        return json.loads(s) if s else None
                    except Exception as e:
                        print(
                            f"(avertissement) JSON invalide pour '{ta.description}': {e}"
                        )
                        return None

                use_fs_kw = fs_enabled.value if fs_override.value else None
                sel_model_kw = fs_model.value if fs_override.value else None
                sel_thr_kw = fs_threshold.value if fs_override.value else None

                self.run_training_session(
                    model_type=model_choice.value,
                    n_estimators=n_estimators_widget.value,
                    prediction_target=target_choice.value,
                    save_and_log=save_log_checkbox.value,
                    # === options avancées ===
                    search=search_mode.value,  # None | "grid" | "random"
                    cv_folds=cv_folds.value,
                    scoring=scoring.value,
                    n_iter=n_iter_widget.value,
                    early_stopping=early_stopping.value,
                    early_stopping_rounds=early_rounds.value,
                    val_size=val_size.value,
                    use_groups=use_groups.value,
                    group_col=(group_col_text.value or None),
                    selector_n_estimators=selector_n_estimators.value,
                    # split/seed
                    test_size=test_size_slider.value,
                    random_state=seed_box.value,
                    # grilles
                    param_grid=_parse_json(param_grid_txt),
                    param_distributions=_parse_json(param_dist_txt),
                    base_params=_parse_json(base_params_txt),
                    # poids & calibration
                    use_balanced_weights=balanced_weights.value,
                    calibrate_probs=calibrate_probs.value,
                    calibration_method=calib_method.value,
                    # artefacts
                    save_confusion_png=save_cm_png.value,
                    # overrides FS (ou None pour ne pas écraser la config par défaut)
                    use_feature_selection=use_fs_kw,
                    selector_model=sel_model_kw,
                    selector_threshold=sel_thr_kw,
                )

        run_button.on_click(on_run_button_clicked)

        top = W.HBox([model_choice, n_estimators_widget, target_choice])
        row1 = W.HBox([search_mode, cv_folds, scoring, n_iter_widget])
        row2 = W.HBox([early_stopping, val_size, use_groups, group_col_text])
        row2b = W.HBox([test_size_slider, seed_box, early_rounds])
        row3 = W.HBox([fs_override, fs_enabled])  # toggles
        row_sel = W.HBox([fs_model, fs_threshold, selector_n_estimators])  # réglages FS
        rowJSON = W.HBox([param_grid_txt, param_dist_txt, base_params_txt])
        rowXtras = W.HBox(
            [balanced_weights, calibrate_probs, calib_method, save_cm_png]
        )
        row4 = W.HBox([save_log_checkbox, run_button])
        display(
            W.VBox(
                [
                    top,
                    row1,
                    row2,
                    row2b,
                    row3,
                    row_sel,
                    rowJSON,
                    rowXtras,
                    row4,
                    output_area,
                ]
            )
        )

    # --------------------- Journalisation & rapport ---------------------

    def _log_and_report(
        self,
        clf: SpectralClassifier,
        feature_cols: List[str],
        X: pd.DataFrame,
        y: pd.Series | List[str] | pd.DataFrame,
        processed_files: List[str],
        groups=None,
        save_confusion_png: bool = False,
    ) -> Optional[str]:
        """
        Étapes 6 et 7 : met à jour le journal des spectres traités et génère un rapport de session.
        - Sauvegarde le modèle entraîné et calcule un hash MD5
        - Évalue le modèle sur un split de test stratifié (seedé) pour reproductibilité
        - Exporte un rapport JSON (métriques, importances, features retenues, etc.)

        Parameters
        ----------
        clf : SpectralClassifier
            Classifieur entraîné (contient model_pipeline, best_params_, class_labels, etc.)
        feature_cols : list[str]
            Noms des colonnes de features AVANT éventuelle sélection.
        X : np.ndarray
            Features (toutes lignes entraînables, alignées avec y).
        y : np.ndarray | list
            Labels correspondants.
        processed_files : list[str]
            Chemins des spectres traités pour mise à jour du journal.
        """

        # Garde-fous
        if clf is None or X is None or y is None or len(y) == 0:
            print(
                "\n> Aucun modèle entraîné ou données manquantes. Rapport non généré."
            )
            return None

        # 6) Journal des spectres utilisés (optionnel si liste vide)
        try:
            print("\n--- ÉTAPE 6 : Mise à jour du Journal des Spectres Utilisés ---")
            self.builder.update_trained_log(processed_files or [])
        except Exception as e:
            print(f"  (avertissement) Impossible de mettre à jour le journal : {e}")

        # 7) Génération du rapport de session
        print("\n--- ÉTAPE 7 : Génération du Rapport de Session ---")

        # Un seul timestamp pour tous les artefacts
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

        # 7.1 Sauvegarde du modèle
        model_filename = f"spectral_classifier_{clf.model_type.lower()}_{ts}.pkl"
        model_path = os.path.join(self.models_dir, model_filename)
        try:
            clf.save_model(model_path)
        except Exception as e:
            print(f"  (avertissement) Échec de sauvegarde du modèle : {e}")

        # 7.2 Métadonnées (environnement + features)
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
        }
        meta_filename = f"spectral_classifier_{clf.model_type.lower()}_{ts}_meta.json"
        meta_path = os.path.join(self.models_dir, meta_filename)
        try:
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)
        except Exception as e:
            print(f"  (avertissement) Échec d’écriture des métadonnées : {e}")

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
            # réutilise exactement le même test set que pendant l'entraînement
            if hasattr(clf, "_split_info") and "te_idx" in clf._split_info:
                te_idx = clf._split_info["te_idx"]
                X_te = X.iloc[te_idx] if hasattr(X, "iloc") else X[te_idx]
                y_te = np.asarray(y)[te_idx]
            else:
                X_te, y_te = X, np.asarray(y)

            y_pred = clf.model_pipeline.predict(X_te)

            if (
                clf.model_type == "XGBoost"
                and getattr(clf, "label_encoder", None) is not None
            ):
                n_classes = len(clf.class_labels)
                all_labels = np.arange(n_classes)  # 0..K-1
                y_te_enc = clf.label_encoder.transform(y_te)

                report_dict = classification_report(
                    y_te_enc,
                    y_pred,
                    labels=all_labels,
                    target_names=list(clf.class_labels),  # <-- corrigé (underscore)
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

        # Sauvegarde PNG de la matrice de confusion (après calcul de cm)
        if save_confusion_png and cm is not None:
            try:
                import matplotlib.pyplot as plt
                import seaborn as sns

                fig = plt.figure(figsize=(8, 6))
                sns.heatmap(
                    cm,
                    annot=True,
                    fmt="d",
                    xticklabels=list(clf.class_labels),
                    yticklabels=list(clf.class_labels),
                )
                plt.xlabel("Prédiction")
                plt.ylabel("Vraie valeur")
                plt.title(f"Matrice de confusion — {clf.model_type}")
                out_png = os.path.join(
                    self.reports_dir,
                    f"confusion_matrix_{clf.model_type.lower()}_{ts}.png",
                )
                fig.tight_layout()
                fig.savefig(out_png, dpi=140)
                plt.close(fig)
                print(f"  > Heatmap sauvegardée : {out_png}")
            except Exception as e:
                print(f"  (avertissement) Échec sauvegarde heatmap : {e}")

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
        }

        report_filename = f"session_report_{clf.model_type.lower()}_{ts}.json"
        report_path = os.path.join(self.reports_dir, report_filename)
        try:
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(session_report, f, indent=4)
            print(f"\nRapport de session sauvegardé dans : {report_path}")
        except Exception as e:
            print(
                f"  (avertissement) Échec lors de la génération du rapport de session : {e}"
            )

        # --- AFFICHAGE RÉSUMÉ DANS LA CELLULE ---
        try:
            from sklearn.metrics import balanced_accuracy_score

            # y_true / y_pred sont calculés plus haut
            y_true_for_scores = y_te_enc if "y_te_enc" in locals() else y_te
            bal_acc = (
                balanced_accuracy_score(y_true_for_scores, y_pred)
                if y_true_for_scores is not None
                else None
            )

            print("\n=== RÉSULTATS (jeu test) ===")
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
                # ⬇️ le bon nom est "feature_cols" (paramètre de la fonction)
                total = len(feature_cols) if feature_cols is not None else None
                msg = f"[FS] Features conservées : {kept}" + (
                    f"/{total}" if total is not None else ""
                )
                print("\n" + msg)
        except Exception as e:
            print(f"(avertissement) Récap FS ignoré : {e}")

        print("\nSESSION DE RECHERCHE TERMINÉE")
        return report_path
