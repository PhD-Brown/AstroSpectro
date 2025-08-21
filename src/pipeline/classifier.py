"""
AstroSpectro — Outils d’entraînement et d’évaluation du classifieur spectral.

Ce module encapsule le pipeline ML complet (imputation, scaling, SMOTE,
sélection de variables optionnelle, tuning par GridSearchCV), l’entraînement,
l’évaluation et la persistance (pickle + méta-données JSON).

Conventions
-----------
- Les longueurs/largeurs sont en unités natives de scikit-learn / xgboost.
- `feature_names_used` conserve l’ordre des colonnes après préparation (stable).

Entrées / Sorties attendues
---------------------------
- Entrée : DataFrame de features (colonnes numériques + méta), incluant
  au minimum `subclass` pour dériver la cible; autres colonnes sont
  auto-filtrées par `_prepare_features_and_labels`.
- Sortie : modèle entraîné (pipeline sklearn/imb), rapports d’évaluation,
  artefacts persistés (.pkl/.json) via `save_model()`.

API publique (principales méthodes)
-----------------------------------
- clean/filter : `_clean_and_filter_data(df)` → DataFrame prêt à l’entraînement
- features/labels : `_prepare_features_and_labels(df)` → (X, y)
- train : `train_and_evaluate(features_df, ...)` → (self, cols, X, y)
- eval : `evaluate(X_test, y_test)` → affiche rapport + matrice de confusion
- IO : `save_model(path)`, `load_model(path)`

Exemple minimal
---------------
>>> clf = SpectralClassifier(model_type="XGBoost", prediction_target="main_class")
>>> clf.train_and_evaluate(features_df)
>>> clf.save_model("data/models/spectral_classifier_xgboost.pkl")
"""

from __future__ import annotations

import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import os
import sys
import json
import time
import inspect
import sklearn
import xgboost
from typing import Optional, Dict, Any

# --- Imports Scikit-learn ---
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    StratifiedShuffleSplit,
    GroupShuffleSplit,
    StratifiedKFold,
    GridSearchCV,
    RandomizedSearchCV,
    train_test_split,
    GroupKFold,
)

try:
    HAS_SGKF = True
except Exception:
    HAS_SGKF = False
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
)
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.feature_selection import SelectFromModel
from sklearn.calibration import CalibratedClassifierCV


class SpectralClassifier:
    """
    Classifieur spectral encapsulant un pipeline complet (imputation, scaling,
    sur-échantillonnage SMOTE, sélection de variables optionnelle et modèle final).
    Gère aussi le tuning par GridSearchCV, l’évaluation et la persistance du modèle.
    """

    def __init__(
        self,
        model_type="XGBoost",
        prediction_target="main_class",
        use_feature_selection=True,
        selector_threshold="median",
        selector_model="xgb",
        selector_n_estimators=200,
        random_state=42,
    ):
        """
        Initialise la configuration du pipeline de classification.

        Args:
            model_type: Modèle final à entraîner. {"XGBoost", "RandomForest"}.
            prediction_target: Cible de prédiction (ex. "main_class", "sub_class_top25",
                "sub_class_bins").
                NOTE : sub_class_bins fait référence à des classes regroupées en bins donc
                les sous-classes sont classées par ordre de fréquence.
            use_feature_selection: Active la sélection de variables via SelectFromModel.
            selector_threshold: Seuil de sélection (ex. "median", "mean", "0.5*mean").
            selector_model: Estimateur utilisé pour la sélection des features
                {"xgb", "rf"}.
            selector_n_estimators: Nombre d’arbres/boosting rounds pour le sélecteur.
            random_state: Graine de reproductibilité.

        Attributes:
            feature_names_used (list[str]): Noms des variables candidates détectées.
            selected_features_ (list[str] | None): Sous-ensemble retenu après sélection.
            best_estimator_: Pipeline final issu de GridSearchCV (une fois entraîné).
        """
        self.model_type = model_type
        self.prediction_target = prediction_target
        self.use_feature_selection = use_feature_selection
        self.selector_threshold = selector_threshold
        self.selector_model = selector_model
        self.selector_n_estimators = selector_n_estimators
        self.random_state = random_state

        self.feature_names_used = []
        self.selected_features_ = None
        self.best_estimator_ = None

    # ---------------------------------------------------------------------
    # Préparation des données (construction des labels + nettoyage)
    # ---------------------------------------------------------------------

    def _clean_and_filter_data(self, df: pd.DataFrame) -> pd.DataFrame | None:
        """
        Crée la colonne de label selon la stratégie choisie puis nettoie/filtre
        le DataFrame pour l’entraînement.

        Étapes principales :
        1) Construction de `label` selon `prediction_target`.
        2) Suppression des labels invalides (U, n, N, O, OTHER).
        3) Filtrage des classes trop rares (<10 échantillons).

        Args:
            df: DataFrame de départ (catalogue enrichi + features).

        Returns:
            pd.DataFrame | None: DataFrame prêt pour l’entraînement (avec `label`)
            ou `None` si trop peu de données valides.

        Notes:
            Cette méthode ne modifie pas `df` en place (travaille sur une copie).
        """
        if "subclass" not in df.columns:
            print(
                "AVERTISSEMENT: Colonne 'subclass' non trouvée. Impossible de créer des labels."
            )
            return None

        df_copy = df.copy()

        if self.prediction_target == "main_class":
            print(
                "  > Stratégie : Classification des classes principales (A, F, G...)."
            )
            df_copy["label"] = df_copy["subclass"].astype(str).str[0]

        elif self.prediction_target == "sub_class_top25":
            print(
                "  > Stratégie : Classification fine sur les 25 sous-classes les plus fréquentes."
            )
            df_valid = df_copy[
                df_copy["subclass"].notnull() & (df_copy["subclass"] != "Non")
            ].copy()
            # On trouve les 25 sous-classes les plus communes
            top_n_subclasses = (
                df_valid["subclass"].value_counts().nlargest(25).index.tolist()
            )
            df_copy = df_valid[df_valid["subclass"].isin(top_n_subclasses)].copy()
            df_copy["label"] = df_copy["subclass"]
            print(
                f"  > Sélection de {len(top_n_subclasses)} sous-classes pour l'entraînement."
            )
            # Pourquoi: limite la granularité aux classes “fréquentes” pour éviter
            # les splits train/test dégénérés et des métriques non parlantes.
        elif self.prediction_target == "sub_class_bins":
            print(
                "  > Stratégie : Classification fine par 'bacs' (ex: G_early, G_late)."
            )

            def map_to_bin(subclass: str) -> str:
                """
                Regroupe une sous-classe spectrale fine en bin large de type
                « X_early » ou « X_late ».

                Règles:
                    - On ne traite que les formats classiques commençant par une
                    lettre parmi {A,F,G,K,M} suivie d’un chiffre (ex.: "G2", "F5V", "K7III").
                    - Si le chiffre < 5  -> "<TYPE>_early"  (ex.: G0–G4 -> "G_early")
                    - Sinon             -> "<TYPE>_late"   (ex.: G5–G9 -> "G_late")
                    - Pour tout autre format non reconnu -> "OTHER"

                Args:
                    subclass: Libellé de sous-classe d’origine (ex.: "G2", "K7III", "WD").

                Returns:
                    str: Bin large ("G_early", "G_late", …) ou "OTHER" si non reconnu.
                """
                s = str(subclass).strip()
                if s and s[0] in "AFGKM" and len(s) > 1 and s[1].isdigit():
                    main_type = s[0]
                    sub_type_digit = int(s[1])
                    return (
                        f"{main_type}_early"
                        if sub_type_digit < 5
                        else f"{main_type}_late"
                    )
                return "OTHER"

            df_copy["label"] = df_copy["subclass"].apply(map_to_bin)

        else:
            raise ValueError(
                f"prediction_target '{self.prediction_target}' invalide. Choix possibles : 'main_class', 'sub_class_top25', 'sub_class_bins'."
            )

        initial_count = len(df_copy)
        invalid_labels = ["U", "n", "N", "O", "OTHER"]
        df_trainable = df_copy[
            df_copy["label"].notnull() & ~df_copy["label"].isin(invalid_labels)
        ].copy()
        print(
            f"  > {initial_count - len(df_trainable)} lignes avec des labels invalides ou nuls supprimées."
        )

        label_counts = df_trainable["label"].value_counts()
        rare_labels = label_counts[label_counts < 10].index.tolist()
        if rare_labels:
            # Attention: seuil 10 choisi pour la robustesse des splits + SMOTE.
            print(
                f"  > Suppression des classes trop rares (moins de 10 échantillons) : {rare_labels}"
            )
            df_trainable = df_trainable[~df_trainable["label"].isin(rare_labels)]

        if len(df_trainable) < 20:
            print(
                "\nERREUR : Pas assez de données valides après nettoyage pour entraîner un modèle."
            )
            return None

        print(
            f"  > {len(df_trainable)} échantillons finaux avec {len(df_trainable['label'].unique())} classes pour l'entraînement."
        )
        return df_trainable

    # ---------------------------------------------------------------------
    # Sélection dynamique des features & fabrication de X/y
    # ---------------------------------------------------------------------

    def _prepare_features_and_labels(
        self, df_trainable: pd.DataFrame
    ) -> tuple[pd.DataFrame, np.ndarray]:
        """
        Sélectionne dynamiquement toutes les colonnes numériques comme features,
        normalise les NaN/inf, supprime les colonnes vides/constantes et
        renvoie X (features triées) et y (labels).

        Args:
            df_trainable: DataFrame nettoyé contenant au moins la colonne `label`.

        Returns:
            (X, y):
                - X (pd.DataFrame): Matrice de features en float, colonnes triées.
                - y (np.ndarray): Vecteur de labels.

        Side Effects:
            Met à jour `self.feature_names_used` avec l’ordre final des colonnes.
        """
        cols_to_exclude = [
            "file_path",
            "fits_name",
            "obsid",
            "plan_id",
            "mjd",
            "class",
            "subclass",
            "filename_original",
            "author",
            "data_version",
            "date_creation",
            "telescope",
            "obs_date_utc",
            "designation",
            "fiber_type",
            "object_name",
            "catalog_object_type",
            "magnitude_type",
            "heliocentric_correction",
            "radial_velocity_corr",
            "label",
            "main_class",
            "source_id",
            "gaia_ra",
            "gaia_dec",
            "match_dist_arcsec",
            "pipeline_version",
            "processing_notes",
            "download_url",
            "spectrum_hash",
            "flux_unit",
            "wavelength_unit",
        ]

        candidate_cols = [
            c
            for c in df_trainable.columns
            if c not in cols_to_exclude
            and pd.api.types.is_numeric_dtype(df_trainable[c])
        ]

        df_num = df_trainable[candidate_cols].copy()
        df_num.replace({pd.NA: np.nan}, inplace=True)
        df_num.replace([np.inf, -np.inf], np.nan, inplace=True)

        for c in df_num.columns:
            if df_num[c].dtype == "object":
                df_num[c] = pd.to_numeric(df_num[c], errors="coerce")

        df_num = df_num.loc[:, df_num.notna().any(axis=0)]

        const_cols = df_num.nunique(dropna=True)
        const_cols = const_cols[const_cols <= 1].index.tolist()
        if const_cols:
            # Pourquoi: supprimer les colonnes constantes évite du bruit pour la
            # sélection de features et accélère le fit.
            df_num.drop(columns=const_cols, inplace=True)

        df_num = df_num.reindex(sorted(df_num.columns), axis=1)
        self.feature_names_used = list(df_num.columns)

        X = df_num
        y = df_trainable["label"].values

        print(
            "\n--- Préparation pour l'entraînement (Détection Dynamique des Features) ---"
        )
        print(
            f"Features utilisées ({len(self.feature_names_used)}) : {self.feature_names_used}"
        )

        return X, y

    def _build_preprocessor(
        self, X_train: pd.DataFrame, model_type: str
    ) -> ColumnTransformer:
        """
        Préprocesseur minimal et robuste :
        - pour RF/XGB : imputation (médiane) des colonnes numériques
        - pour SVM : imputation + standardisation (important pour SVM)
        """
        num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
        if not num_cols:
            raise ValueError("Aucune colonne numérique disponible pour l'entraînement.")

        if model_type in {"SVM"}:
            return ColumnTransformer(
                [
                    (
                        "num",
                        Pipeline(
                            [
                                ("imp", SimpleImputer(strategy="median")),
                                ("scaler", StandardScaler()),
                            ]
                        ),
                        num_cols,
                    )
                ],
                remainder="drop",
            )
        else:
            return ColumnTransformer(
                [("num", SimpleImputer(strategy="median"), num_cols)],
                remainder="drop",
            )

    def _build_estimator(
        self, model_type: str, n_estimators: int, n_classes: int, random_state: int
    ):
        """
        Estimateur baseline par modèle.
        - RF : rapide, n_jobs=-1
        - XGB : 'hist' pour la vitesse ; encodage de y géré côté train_and_evaluate
        - SVM : RBF + class_weight=balanced
        """
        if model_type == "RandomForest":
            return RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=None,
                n_jobs=-1,
                random_state=random_state,
                class_weight="balanced_subsample",
            )

        if model_type == "XGBoost":
            try:
                import xgboost as xgb
            except Exception as e:
                raise RuntimeError(
                    "XGBoost n'est pas installé dans cet environnement."
                ) from e

            return xgb.XGBClassifier(
                n_estimators=n_estimators,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="multi:softprob",
                eval_metric="mlogloss",
                num_class=n_classes,
                tree_method="hist",  # rapide et CPU-friendly
                n_jobs=-1,
                random_state=random_state,
            )

        if model_type == "SVM":
            return SVC(
                kernel="rbf",
                probability=True,
                class_weight="balanced",
                random_state=random_state,
            )

        raise ValueError(
            f"Modèle inconnu: {model_type} (attendu: RandomForest | XGBoost | SVM)"
        )

    # ---------------------------------------------------------------------
    # Entraînement + tuning + (optionnel) sélection de variables
    # ---------------------------------------------------------------------

    def train_and_evaluate(
        self,
        features_df: pd.DataFrame,
        test_size: float = 0.21,
        n_estimators: int = 300,
        *,
        search: Optional[str] = None,
        cv_folds: int = 3,
        scoring: str = "accuracy",
        use_feature_selection: Optional[bool] = None,
        selector_model: Optional[str] = None,
        selector_threshold: Optional[str] = None,
        early_stopping: bool = False,
        early_stopping_rounds: int = 50,
        val_size: float = 0.15,
        use_groups: bool = False,
        group_col: Optional[str] = None,
        param_grid: Optional[Dict[str, Any]] = None,
        param_distributions: Optional[Dict[str, Any]] = None,
        n_iter: int = 60,
        random_state: int = 42,
        param_overrides: dict | None = None,
        use_balanced_weights: bool = True,
        calibrate_probs: bool = False,
        calibration_method: str = "sigmoid",
    ) -> bool:
        """
        Entraîne et évalue le modèle avec GridSearchCV + pipeline Imputer/Scaler/SMOTE,
        (optionnellement) SelectFromModel, puis affiche un rapport et une matrice
        de confusion.

        Args:
            features_df: DataFrame complet (features + colonnes méta) avant nettoyage.
            test_size: Proportion du jeu de test (stratifié).
            n_estimators: Nombre d’arbres/itérations pour le modèle final.

        Returns:
            (self, cols_for_report, X, y) ou None:
                - self (SpectralClassifier): L’instance entraînée (avec pipeline).
                - cols_for_report (list[str]): Les colonnes utilisées in fine
                (sélectionnées ou toutes les candidates).
                - X (pd.DataFrame): Matrice de features utilisée avant split.
                - y (np.ndarray): Labels correspondants.

        Raises:
            ValueError: Relevée en interne si GridSearchCV échoue (message affiché).

        Notes:
            - XGBoost nécessite un encodage numérique des labels (géré ici).
            - Le nombre de voisins SMOTE est ajusté automatiquement selon la plus
            petite classe du split d’entraînement.
        """
        print(
            f"\n=== ÉTAPE 4 : SESSION D'ENTRAÎNEMENT (Modèle: {self.model_type}, Cible: {self.prediction_target}) ==="
        )

        # 0) Nettoyage + label
        df_trainable = self._clean_and_filter_data(features_df)
        if df_trainable is None:
            print("> Aucune donnée exploitable après nettoyage ; arrêt.")
            return False

        # Groupes (optionnel)
        groups_full = None
        if use_groups and group_col and group_col in df_trainable.columns:
            groups_full = df_trainable[group_col].values

        # 1) X/y + encodage XGB
        X_all, y_all = self._prepare_features_and_labels(df_trainable)
        label_encoder: Optional[LabelEncoder] = None
        y_enc = y_all
        if self.model_type == "XGBoost":
            label_encoder = LabelEncoder().fit(y_all)
            y_enc = label_encoder.transform(y_all)

        # 2) Split train/test (stratifié ou par groupes)
        if use_groups and (groups_full is not None):
            gss = GroupShuffleSplit(
                n_splits=1, test_size=test_size, random_state=random_state
            )
            tr_idx, te_idx = next(gss.split(X_all, y_enc, groups=groups_full))
            groups_train = groups_full[tr_idx]
        else:
            sss = StratifiedShuffleSplit(
                n_splits=1, test_size=test_size, random_state=random_state
            )
            tr_idx, te_idx = next(sss.split(X_all, y_enc))
            groups_train = None

        self._split_info = {
            "tr_idx": tr_idx,
            "te_idx": te_idx,
            "n_train": int(len(tr_idx)),
            "n_test": int(len(te_idx)),
            "test_size": test_size,
            "random_state": random_state,
        }  # pour le rapport

        X_train = X_all.iloc[tr_idx]
        y_train = y_enc[tr_idx]

        # 3) Préprocesseur + estimateur + (optionnel) SelectFromModel
        n_classes = int(np.unique(y_enc).shape[0])
        preproc = self._build_preprocessor(X_train, self.model_type)
        base_est = self._build_estimator(
            self.model_type, n_estimators, n_classes, random_state
        )
        if param_overrides:
            base_est.set_params(**param_overrides)

        fs_enabled = (
            self.use_feature_selection
            if use_feature_selection is None
            else use_feature_selection
        )
        sel_model = (
            (self.selector_model or "rf") if selector_model is None else selector_model
        )
        sel_threshold = (
            self.selector_threshold
            if selector_threshold is None
            else selector_threshold
        )

        steps = [("prep", preproc)]
        if fs_enabled:
            if sel_model == "xgb":
                import xgboost as xgb

                selector_est = xgb.XGBClassifier(
                    n_estimators=self.selector_n_estimators,
                    max_depth=6,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective="multi:softprob",
                    eval_metric="mlogloss",
                    num_class=n_classes,
                    tree_method="hist",
                    n_jobs=-1,
                    random_state=random_state,
                )
            else:
                selector_est = RandomForestClassifier(
                    n_estimators=self.selector_n_estimators,
                    n_jobs=-1,
                    random_state=random_state,
                    class_weight="balanced_subsample",
                )
            steps.append(("fs", SelectFromModel(selector_est, threshold=sel_threshold)))
        steps.append(("clf", base_est))
        pipe = Pipeline(steps)

        # 4) Fit params : weights + early stopping (XGB)
        fit_params: Dict[str, Any] = {}
        if use_balanced_weights:
            sw = compute_sample_weight("balanced", y_train)
            fit_params["clf__sample_weight"] = sw

        # ES uniquement hors recherche (sinon masque FS instable entre folds)
        es_active = (
            (self.model_type == "XGBoost") and early_stopping and (search is None)
        )
        if es_active:
            # split interne pour l'ES
            X_tr_in, X_val_in, y_tr_in, y_val_in = train_test_split(
                X_train,
                y_train,
                test_size=val_size,
                stratify=y_train,
                random_state=random_state,
            )

            # pipeline "prep + fs" clonée et FIT sur TOUT X_train (même data que le fit final)
            pre_fs_steps = steps[
                :-1
            ]  # ['prep', ('fs', ...)] si FS actif, sinon juste 'prep'
            if pre_fs_steps:
                prep_for_es = clone(Pipeline(pre_fs_steps))
                prep_for_es.fit(X_train, y_train)
                X_val_es = prep_for_es.transform(X_val_in)
            else:
                X_val_es = X_val_in  # (cas sans préproc/FS)

            # compat XGB 1.x / 2.x : choisir l'API supportée
            fit_sig = set(inspect.signature(base_est.fit).parameters.keys())
            fit_params["clf__eval_set"] = [(X_val_es, y_val_in)]
            if "callbacks" in fit_sig:
                from xgboost.callback import EarlyStopping

                fit_params["clf__callbacks"] = [
                    EarlyStopping(rounds=int(early_stopping_rounds), save_best=True)
                ]
            elif "early_stopping_rounds" in fit_sig:
                fit_params["clf__early_stopping_rounds"] = int(early_stopping_rounds)
            if "verbose" in fit_sig:
                fit_params["clf__verbose"] = False

        # 5) CV aware of groups
        if use_groups and groups_train is not None:
            try:
                from sklearn.model_selection import StratifiedGroupKFold

                cv = StratifiedGroupKFold(
                    n_splits=cv_folds, shuffle=True, random_state=random_state
                )
            except Exception:
                cv = GroupKFold(n_splits=cv_folds)
        else:
            cv = StratifiedKFold(
                n_splits=cv_folds, shuffle=True, random_state=random_state
            )

        # 6) Fit / Search
        if search == "grid":
            searcher = GridSearchCV(
                pipe,
                param_grid=(param_grid or {}),
                scoring=scoring,
                cv=cv,
                n_jobs=-1,
                refit=True,
                verbose=0,
            )
            searcher.fit(X_train, y_train, groups=groups_train, **fit_params)
            best = searcher.best_estimator_
            self.best_params_ = searcher.best_params_
        elif search == "random":
            searcher = RandomizedSearchCV(
                pipe,
                param_distributions=(param_distributions or {}),
                n_iter=n_iter,
                scoring=scoring,
                cv=cv,
                n_jobs=-1,
                refit=True,
                random_state=random_state,
                verbose=0,
            )
            searcher.fit(X_train, y_train, groups=groups_train, **fit_params)
            best = searcher.best_estimator_
            self.best_params_ = searcher.best_params_
        else:
            pipe.fit(X_train, y_train, **fit_params)
            best = pipe
            self.best_params_ = getattr(best[-1], "get_params", lambda: {})()

        # 7) Calibration (optionnelle)
        if calibrate_probs and hasattr(best, "predict_proba"):
            best = CalibratedClassifierCV(best, method=calibration_method, cv=3)
            best.fit(X_train, y_train)

        # 8) Expose artefacts
        self.model_pipeline = best
        self.class_labels = (
            list(label_encoder.classes_)
            if label_encoder is not None
            else sorted(np.unique(y_all))
        )
        self.label_encoder = label_encoder

        # features sélectionnées (si FS actif)
        self.selected_features_ = None
        if fs_enabled and "fs" in best.named_steps:
            try:
                out_names = best.named_steps["prep"].get_feature_names_out()
                mask = best.named_steps["fs"].get_support()
                sel = [
                    n.split("__", 1)[1] if "__" in n else n
                    for n, keep in zip(out_names, mask)
                    if keep
                ]
                self.selected_features_ = sel
            except Exception:
                pass

        # Retour pour MasterPipeline
        feature_cols_before_fs = list(X_all.columns)
        y_all_out = (
            label_encoder.inverse_transform(y_enc)
            if label_encoder is not None
            else y_all
        )
        return self, feature_cols_before_fs, X_all, y_all_out, groups_full

    # ---------------------------------------------------------------------
    # Évaluation & persistance
    # ---------------------------------------------------------------------

    def evaluate(self, X_test: pd.DataFrame | np.ndarray, y_test: np.ndarray) -> None:
        """
        Évalue le modèle entraîné sur le jeu de test et affiche le rapport
        de classification ainsi que la matrice de confusion.

        Args:
            X_test: Jeu de test transformé comme à l’entraînement (mêmes features).
            y_test: Labels du jeu de test (encodés si le modèle est XGBoost).

        Returns:
            None. Affiche les métriques et la matrice de confusion.
        """
        predictions = self.model_pipeline.predict(X_test)

        # Décodage si XGBoost
        if (
            self.model_type == "XGBoost"
            and getattr(self, "label_encoder", None) is not None
        ):
            # y_test peut être encodé (int) ou déjà texte selon l’appelant
            if np.issubdtype(np.asarray(y_test).dtype, np.integer):
                y_test_dec = self.label_encoder.inverse_transform(y_test)
            else:
                y_test_dec = y_test
            y_pred_dec = self.label_encoder.inverse_transform(predictions)
        else:
            y_test_dec, y_pred_dec = y_test, predictions

        print("\n--- Rapport d'Évaluation ---")
        report = classification_report(y_test_dec, y_pred_dec, zero_division=0)
        cm = confusion_matrix(y_test_dec, y_pred_dec, labels=self.class_labels)
        print(report)
        plt.figure(figsize=(10, 7))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=self.class_labels,
            yticklabels=self.class_labels,
        )
        plt.xlabel("Prédiction")
        plt.ylabel("Vraie Valeur")
        plt.title(f"Matrice de Confusion du Modèle {self.model_type} Optimisé")
        plt.show()

    def save_model(
        self,
        path: str = "model.pkl",
        trained_on_file: str | None = None,
        extra_info: dict | None = None,
    ) -> None:
        """
        Sauvegarde l’objet `SpectralClassifier` (pickle) et un fichier JSON de
        métadonnées à côté du modèle.

        Args:
            path: Chemin du fichier .pkl à créer.
            trained_on_file: Nom/chemin du dataset de features utilisé pour l’entraînement.
            extra_info: Métadonnées supplémentaires à fusionner dans le JSON.

        Returns:
            None. Écrit `<path>` et `<path>_meta.json`.

        Contenu des métadonnées:
            Versions (Python, numpy, scikit-learn, xgboost), type de modèle,
            cible, meilleurs hyperparamètres, labels, features utilisées, etc.
        """
        joblib.dump(self, path)
        meta = {
            "saved_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "xgboost": getattr(xgboost, "__version__", "unknown"),
            "model_type": self.model_type,
            "prediction_target": self.prediction_target,
            "best_params_": getattr(self, "best_params_", None),
            "class_labels": list(getattr(self, "class_labels", [])),
            "feature_names_used": list(getattr(self, "feature_names_used", [])),
            "selected_features_": list(getattr(self, "selected_features_", []) or []),
            "trained_on_file": trained_on_file,
        }
        if isinstance(extra_info, dict):
            meta.update(extra_info)

        meta_path = os.path.splitext(path)[0] + "_meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        print(f"  > Modèle sauvegardé dans : {path}")
        print(f"  > Métadonnées sauvegardées dans : {meta_path}")

    @staticmethod
    def load_model(path: str = "model.pkl") -> "SpectralClassifier":
        """
        Charge un modèle `SpectralClassifier` sauvegardé via `save_model`.

        Args:
            path: Chemin du .pkl à charger.

        Returns:
            SpectralClassifier: L’objet chargé (pipeline + attributs).

        Notes:
            Si un fichier `<path>_meta.json` est présent, son chemin est affiché
            à titre informatif.
        """
        model = joblib.load(path)
        print(f"  > Modèle chargé depuis : {path}")
        meta_path = os.path.splitext(path)[0] + "_meta.json"
        if os.path.exists(meta_path):
            print(f"  > Métadonnées disponibles : {meta_path}")
        return model
