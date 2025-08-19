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
import sklearn
import xgboost

# --- Imports Scikit-learn ---
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    StratifiedKFold,
    GroupShuffleSplit,
)

try:
    from sklearn.model_selection import StratifiedGroupKFold

    HAS_SGKF = True
except Exception:
    HAS_SGKF = False
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.impute import SimpleImputer

# --- Imports Imbalanced-learn ---
from imblearn.pipeline import Pipeline as ImbPipeline


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

    # ---------------------------------------------------------------------
    # Entraînement + tuning + (optionnel) sélection de variables
    # ---------------------------------------------------------------------

    def train_and_evaluate(
        self,
        features_df: pd.DataFrame,
        test_size: float = 0.25,
        n_estimators: int = 100,
    ) -> tuple["SpectralClassifier", list[str], pd.DataFrame, np.ndarray] | None:
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
        import xgboost as xgb
        from xgboost.callback import EarlyStopping
        import inspect

        # 1) Nettoyage / préparation
        df_trainable = self._clean_and_filter_data(features_df)
        if df_trainable is None:
            return None

        # Features/labels
        X_all, y_all = self._prepare_features_and_labels(df_trainable)

        # Groupes (évite fuite d'info entre train/test)
        groups_series = df_trainable.get("plan_id", None)
        groups = (
            None
            if groups_series is None
            else groups_series.astype(str).fillna("NA").to_numpy()
        )

        # Encodage des labels pour XGBoost
        if self.model_type == "XGBoost":
            le = LabelEncoder()
            y_encoded = le.fit_transform(y_all)
            self.class_labels = le.classes_
            y_for_split = y_encoded
        else:
            self.class_labels = sorted(list(set(y_all)))
            y_for_split = y_all

        # 2) Holdout final : GroupShuffleSplit si groupes dispo, sinon classique
        if groups is not None:
            gss = GroupShuffleSplit(
                n_splits=1, test_size=test_size, random_state=self.random_state
            )
            train_idx, test_idx = next(gss.split(X_all, y_for_split, groups))
        else:
            train_idx, test_idx = train_test_split(
                np.arange(len(y_for_split)),
                test_size=test_size,
                random_state=self.random_state,
                stratify=y_for_split,
            )

        X_train, X_test = X_all.iloc[train_idx], X_all.iloc[test_idx]
        y_train, y_test = y_for_split[train_idx], y_for_split[test_idx]
        groups_train = None if groups is None else groups[train_idx]

        # 3) Définition du modèle + grille (SMOTE supprimé → on passe sample_weight)
        if self.model_type == "RandomForest":
            clf_model = RandomForestClassifier(
                n_estimators=n_estimators,
                random_state=self.random_state,
                class_weight="balanced",
            )
            param_grid = {
                "clf__max_depth": [20, None],
                "clf__min_samples_leaf": [1, 3],
            }
        elif self.model_type == "XGBoost":
            clf_model = xgb.XGBClassifier(
                n_estimators=n_estimators,
                random_state=self.random_state,
                eval_metric="mlogloss",
                tree_method="hist",
            )
            param_grid = {
                "clf__max_depth": [6, 8, 10],
                "clf__learning_rate": [0.05, 0.1],
                "clf__min_child_weight": [1, 5],
                "clf__subsample": [0.8, 1.0],
                "clf__colsample_bytree": [0.6, 0.8, 1.0],
                "clf__reg_lambda": [1, 2, 5],
                "clf__gamma": [0, 1],
            }

        # Pipeline = imputer + scaler + (optionnel: selector) + clf
        steps = [
            ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
            ("scaler", StandardScaler()),
        ]
        if self.use_feature_selection:
            # Sélection par modèle (XGB ou RF) avec seuil réglable
            base_selector = (
                xgb.XGBClassifier(
                    n_estimators=getattr(self, "selector_n_estimators", 200),
                    random_state=self.random_state,
                    eval_metric="mlogloss",
                    tree_method="hist",
                )
                if self.selector_model == "xgb"
                else RandomForestClassifier(
                    n_estimators=getattr(self, "selector_n_estimators", 200),
                    random_state=self.random_state,
                    class_weight="balanced",
                )
            )
            steps.append(
                (
                    "selector",
                    SelectFromModel(
                        base_selector,
                        threshold=self.selector_threshold,  # "median", "mean", "0.5*mean", etc.
                    ),
                )
            )
        steps.append(("clf", clf_model))
        pipeline = ImbPipeline(
            steps
        )  # on garde ImbPipeline par compat, mais il n’y a plus SMOTE

        print(
            f"\n--- [Tuning] Recherche des meilleurs hyperparamètres pour {self.model_type} (n_estimators={n_estimators} fixé) ---"
        )

        # CV stratifié groupé si dispo
        if groups_train is not None and HAS_SGKF:
            cv = StratifiedGroupKFold(
                n_splits=3, shuffle=True, random_state=self.random_state
            )
            cv_groups = groups_train
        else:
            cv = StratifiedKFold(
                n_splits=3, shuffle=True, random_state=self.random_state
            )
            cv_groups = None

        # Poids d’échantillons (à la place de SMOTE)
        w_train = compute_sample_weight(class_weight="balanced", y=y_train)

        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=cv,
            n_jobs=-1,
            verbose=2,
            scoring="accuracy",
            error_score="raise",
            refit=True,  # on refitera de toute façon ensuite avec early stopping
        )

        try:
            if cv_groups is not None:
                grid_search.fit(
                    X_train,
                    y_train,
                    groups=cv_groups,
                    **{"clf__sample_weight": w_train},
                )
            else:
                grid_search.fit(X_train, y_train, **{"clf__sample_weight": w_train})
        except ValueError as e:
            print(
                "\nERREUR LORS DU GRIDSEARCHCV. Le processus d'entraînement est arrêté."
            )
            print(f"   Détail de l'erreur : {e}")
            return None

        print(f"\n  > Meilleurs paramètres trouvés : {grid_search.best_params_}")
        print(f"  > Meilleur score de précision (CV) : {grid_search.best_score_:.4f}")

        # 4) Refit final AVEC EARLY STOPPING (XGB uniquement) pour grappiller des points
        best_pipe = grid_search.best_estimator_

        if self.model_type == "XGBoost":
            # Sépare train en (train/val) par groupes si dispo
            if groups_train is not None:
                gss_iv = GroupShuffleSplit(
                    n_splits=1, test_size=0.1, random_state=self.random_state
                )
                tr_idx, val_idx = next(gss_iv.split(X_train, y_train, groups_train))
            else:
                tr_idx, val_idx = train_test_split(
                    np.arange(len(y_train)),
                    test_size=0.1,
                    random_state=self.random_state,
                    stratify=y_train,
                )

            # Fit/transform du préprocesseur (tout sauf 'clf')
            preproc = Pipeline([(n, s) for (n, s) in best_pipe.steps if n != "clf"])
            X_tr_t = preproc.fit_transform(X_train.iloc[tr_idx], y_train[tr_idx])
            # Si la sélection de variables est active, mémoriser les noms retenus
            self.selected_features_ = None
            if self.use_feature_selection and "selector" in preproc.named_steps:
                try:
                    mask = preproc.named_steps["selector"].get_support()
                    self.selected_features_ = [
                        f for f, keep in zip(self.feature_names_used, mask) if keep
                    ]
                    # (optionnel) log rapide
                    print(
                        f"  > Features retenues par le sélecteur ({len(self.selected_features_)}): {self.selected_features_}"
                    )
                except Exception:
                    pass
            X_val_t = preproc.transform(X_train.iloc[val_idx])

            y_tr, y_val = y_train[tr_idx], y_train[val_idx]
            w_tr = compute_sample_weight("balanced", y_tr)

            # Params du meilleur clf
            best_clf_params = best_pipe.named_steps["clf"].get_params()
            # Laisser beaucoup d’itérations, l’early stopping choisira la bonne valeur
            best_clf_params["n_estimators"] = max(
                1000, int(best_clf_params.get("n_estimators", 200) * 5)
            )

            clf_final = xgb.XGBClassifier(**best_clf_params)
            clf_final.set_params(
                random_state=self.random_state,
                eval_metric="mlogloss",
                verbosity=0,
            )

            # --- Refit final XGBoost (avec ES si dispo) ---
            fit_sig = inspect.signature(clf_final.fit)
            fit_kwargs = {}

            # 1) versions récentes : callbacks supportés
            if "callbacks" in fit_sig.parameters:
                from xgboost.callback import EarlyStopping

                fit_kwargs["eval_set"] = [(X_val_t, y_val)]
                fit_kwargs["callbacks"] = [
                    EarlyStopping(rounds=50, save_best=True, maximize=False)
                ]
                # on coupe au besoin le print par itération
                if "verbose" in fit_sig.parameters:
                    fit_kwargs["verbose"] = False

            # 2) versions intermédiaires : early_stopping_rounds
            elif "early_stopping_rounds" in fit_sig.parameters:
                fit_kwargs["eval_set"] = [(X_val_t, y_val)]
                fit_kwargs["early_stopping_rounds"] = 50
                if "eval_metric" in fit_sig.parameters:
                    fit_kwargs["eval_metric"] = "mlogloss"
                if "verbose" in fit_sig.parameters:
                    fit_kwargs["verbose"] = False

            # 3) fallback : PAS d'early-stopping -> ne PAS passer eval_set
            else:
                print(
                    "  > Early stopping indisponible dans cette version de xgboost ; fit sans ES."
                )
                if "verbose" in fit_sig.parameters:
                    fit_kwargs["verbose"] = False

            # appel final
            clf_final.fit(X_tr_t, y_tr, sample_weight=w_tr, **fit_kwargs)

            # Pipeline final = preproc + clf_final
            self.model_pipeline = Pipeline(list(preproc.steps) + [("clf", clf_final)])
        else:
            # RF : on garde le best_estimator_ (refit GridSearch)
            self.model_pipeline = best_pipe

        self.best_params_ = grid_search.best_params_

        print(
            f"\n--- [Évaluation] Performance de {self.model_type} sur le jeu de test ---"
        )
        self.evaluate(X_test, y_test)

        return self, self.feature_names_used, X_all, y_all, groups

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
        print("\n--- Rapport d'Évaluation ---")

        if self.model_type == "XGBoost":
            report = classification_report(
                y_test, predictions, target_names=self.class_labels, zero_division=0
            )
            cm = confusion_matrix(y_test, predictions)
        else:
            report = classification_report(
                y_test, predictions, labels=self.class_labels, zero_division=0
            )
            cm = confusion_matrix(y_test, predictions, labels=self.class_labels)

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
