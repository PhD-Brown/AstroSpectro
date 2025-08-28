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
import inspect
import warnings
import os
import sys
import json
import time
from typing import Optional, Dict, Any, List

import pandas as pd
import numpy as np
import joblib

import seaborn as sns
import matplotlib.pyplot as plt

# --- Imports Scikit-learn ---
import sklearn
from sklearn.base import (
    BaseEstimator,
    TransformerMixin,
    clone,
    ClassifierMixin,
    is_classifier,
)
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    VotingClassifier,
)
from sklearn.model_selection import (
    StratifiedShuffleSplit,
    GroupShuffleSplit,
    StratifiedKFold,
    GridSearchCV,
    RandomizedSearchCV,
    train_test_split,
    GroupKFold,
    RepeatedStratifiedKFold,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
)
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.feature_selection import (
    SelectFromModel,
    mutual_info_classif,
    VarianceThreshold,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.calibration import CalibratedClassifierCV


# imblearn (samplers + pipeline)
try:
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN
    from imblearn.combine import SMOTEENN

    _HAS_IMBLEARN = True
except Exception:
    _HAS_IMBLEARN = False

# Optional learners
try:

    _HAS_XGB = True
except Exception:
    _HAS_XGB = False

try:
    from lightgbm import LGBMClassifier

    _HAS_LGBM = True
except Exception:
    _HAS_LGBM = False

try:
    from catboost import CatBoostClassifier

    _HAS_CATBOOST = True
except Exception:
    _HAS_CATBOOST = False
from sklearn.pipeline import Pipeline as SkPipeline

try:
    _PIPE_TYPES = (SkPipeline, ImbPipeline)
except Exception:
    _PIPE_TYPES = (SkPipeline,)

# ---------- utilitaires ----------


def _ece_score(y_true: np.ndarray, proba: np.ndarray, n_bins: int = 15) -> float:
    """Expected Calibration Error multi-classe (One-vs-max)."""
    # confiance = max proba ; correct = 1 si argmax proba == y_true
    conf = np.max(proba, axis=1)
    pred = np.argmax(proba, axis=1)
    correct = (pred == y_true).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    N = len(y_true)
    for i in range(n_bins):
        m = (conf >= bins[i]) & (
            conf < bins[i + 1] if i < n_bins - 1 else conf <= bins[i + 1]
        )
        if np.any(m):
            acc = correct[m].mean()
            avg_conf = conf[m].mean()
            ece += np.abs(acc - avg_conf) * (m.sum() / N)
    return float(ece)


class _Float64ProbaWrapper(ClassifierMixin, BaseEstimator):
    """Wrap un estimateur de classification et force predict_proba en float64.
    N'expose PAS decision_function si le modèle de base ne l'a pas.
    """

    _estimator_type = "classifier"

    def __init__(self, base):
        self.base = base

    # --- délégation dynamique : n'expose un attribut que si le base l'a ---
    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError
        return getattr(self.base, name)

    # --- API sklearn ---
    def get_params(self, deep=True):
        # pour que clone() fonctionne
        return {"base": self.base}

    def set_params(self, **params):
        if "base" in params:
            self.base = params["base"]
        return self

    def fit(self, X, y=None, **fit_params):
        # on fit le modèle de base tel quel
        self.base.fit(X, y, **fit_params)
        return self

    def predict(self, X):
        return self.base.predict(X)

    def predict_proba(self, X):
        import numpy as np

        proba = self.base.predict_proba(X)
        return np.asarray(proba, dtype=np.float64)

    @property
    def classes_(self):
        # scikit-learn va le lire pendant la calibration
        return getattr(self.base, "classes_", None)

    def __sklearn_is_fitted__(self):
        # utilitaire de compatibilité
        return (
            hasattr(self.base, "__sklearn_is_fitted__")
            and self.base.__sklearn_is_fitted__()
        )


class CollinearityFilter(TransformerMixin, BaseEstimator):
    """Supprime les colonnes très corrélées (> threshold) et faible variance."""

    def __init__(self, var_threshold: float = 0.0, corr_threshold: float = 0.98):
        self.var_threshold = var_threshold
        self.corr_threshold = corr_threshold
        self.keep_columns_: List[str] | None = None

    def fit(self, X: pd.DataFrame, y=None, **fit_params):
        if not isinstance(X, pd.DataFrame):
            # tenter de reconstruire un DataFrame si possible
            X = pd.DataFrame(X)
        df = X.copy()
        # Variance threshold
        if self.var_threshold and self.var_threshold > 0:
            vt = VarianceThreshold(self.var_threshold)
            vt.fit(df)
            df = df.loc[:, vt.get_support()]
        # Remplir les NaN par médiane (sur train)
        df = df.copy()
        for c in df.columns:
            if df[c].isna().any():
                df[c] = df[c].fillna(df[c].median())
        # Corrélation
        if self.corr_threshold and self.corr_threshold < 1.0:
            corr = df.corr(numeric_only=True).abs()
            upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
            to_drop = [
                column
                for column in upper.columns
                if (upper[column] > self.corr_threshold).any()
            ]
            df = df.drop(columns=to_drop, errors="ignore")
        self.keep_columns_ = list(df.columns)
        return self

    def transform(self, X):
        df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        if self.keep_columns_ is None:
            return df.values
        return df.loc[:, [c for c in self.keep_columns_ if c in df.columns]]


class ThresholdTunedClassifier(ClassifierMixin, BaseEstimator):
    """
    Applique un vecteur de seuils par classe sur les probabilités.
    - fit : optionnellement, tune les seuils sur un petit holdout interne
            puis refit sur tout X,y (en forwardant les fit_params).
    - predict : argmax(proba - thr).
    """

    _estimator_type = "classifier"

    def __init__(
        self, base_estimator, tune=False, metric="f1_macro", grid=None, random_state=42
    ):
        self.base_estimator = base_estimator
        self.tune = tune
        self.metric = metric
        self.grid = None if grid is None else np.asarray(grid, dtype=float)
        self.random_state = random_state
        self.thresholds_ = None
        self.classes_ = None

    @staticmethod
    def _fit_with_supported_params(
        estimator, X, y, *, sample_weight=None, fit_params=None
    ):
        """Call estimator.fit with only the kwargs it supports."""
        fit_params = dict(fit_params or {})
        supported = set(inspect.signature(estimator.fit).parameters.keys())

        # Ne garder que les kwargs réellement supportés
        clean = {k: v for k, v in fit_params.items() if k in supported}

        # Ajouter sample_weight seulement si supporté
        if sample_weight is not None and "sample_weight" in supported:
            clean["sample_weight"] = sample_weight
        elif sample_weight is not None:
            warnings.warn(
                f"{estimator.__class__.__name__} ne supporte pas sample_weight ; "
                "les poids seront ignorés pour ce modèle."
            )
        if "sample_weight" in clean and clean["sample_weight"] is not None:
            try:
                clean["sample_weight"] = np.asarray(
                    clean["sample_weight"], dtype=np.float64
                )
            except Exception:
                pass

        return estimator.fit(X, y, **clean)

    def fit(self, X, y, **fit_params):
        # récupère/forwarde les params utiles
        sw = fit_params.pop("sample_weight", None)
        if sw is not None:
            sw = np.asarray(sw, dtype=np.float32)
        eval_set = fit_params.pop("eval_set", None)
        callbacks = fit_params.pop("callbacks", None)
        es_rounds = fit_params.pop("early_stopping_rounds", None)
        verbose = fit_params.pop("verbose", None)
        grid_arr = np.asarray(
            self.grid if self.grid is not None else np.linspace(0.2, 0.8, 13),
            dtype=float,
        )

        if self.tune and hasattr(self.base_estimator, "predict_proba"):
            # split interne pour tuner les seuils
            if sw is None:
                X_tr, X_val, y_tr, y_val = train_test_split(
                    X, y, test_size=0.20, stratify=y, random_state=self.random_state
                )
                sw_tr = None
            else:
                X_tr, X_val, y_tr, y_val, sw_tr, sw_val = train_test_split(
                    X, y, sw, test_size=0.20, stratify=y, random_state=self.random_state
                )

            fit_kwargs = {} if verbose is None else {"verbose": verbose}
            self._fit_with_supported_params(
                self.base_estimator,
                X_tr,
                y_tr,
                sample_weight=sw_tr,
                fit_params=fit_kwargs,
            )
            proba = self.base_estimator.predict_proba(X_val)
            n_classes = proba.shape[1]
            best = np.zeros(n_classes)
            best_score = -1.0

            # grille commune puis affinement par classe
            for t in grid_arr:
                s = self._score_with_thresholds(proba, y_val, np.full(n_classes, t))
                if s > best_score:
                    best_score, best = s, np.full(n_classes, t)
            for k in range(n_classes):
                for t in grid_arr:
                    cand = best.copy()
                    cand[k] = t
                    s = self._score_with_thresholds(proba, y_val, cand)
                    if s > best_score:
                        best_score, best = s, cand
            self.thresholds_ = best

            # 2) refit final sur toutes les données avec les params transmis
            extra = {}
            if eval_set is not None:
                extra["eval_set"] = eval_set
            if callbacks is not None:
                extra["callbacks"] = callbacks
            if es_rounds is not None:
                extra["early_stopping_rounds"] = es_rounds
            if verbose is not None:
                extra["verbose"] = verbose
            final_fit_params = dict(fit_params)
            final_fit_params.update(extra)

            self._fit_with_supported_params(
                self.base_estimator, X, y, sample_weight=sw, fit_params=final_fit_params
            )
        else:
            # pas de tuning → fit direct avec les params transmis
            if eval_set is not None:
                fit_params["eval_set"] = eval_set
            if callbacks is not None:
                fit_params["callbacks"] = callbacks
            if es_rounds is not None:
                fit_params["early_stopping_rounds"] = es_rounds
            if verbose is not None:
                fit_params["verbose"] = verbose
            self._fit_with_supported_params(
                self.base_estimator, X, y, sample_weight=sw, fit_params=fit_params
            )

        self.classes_ = getattr(self.base_estimator, "classes_", None)
        return self

    def _score_with_thresholds(self, proba, y_true, thr):
        scores = proba - thr.reshape(1, -1)
        y_pred = np.argmax(scores, axis=1)
        from sklearn.metrics import f1_score, balanced_accuracy_score

        return (
            balanced_accuracy_score(y_true, y_pred)
            if self.metric == "balanced_accuracy"
            else f1_score(y_true, y_pred, average="macro")
        )

    def predict(self, X):
        if (
            not hasattr(self.base_estimator, "predict_proba")
            or self.thresholds_ is None
        ):
            return self.base_estimator.predict(X)
        proba = self.base_estimator.predict_proba(X)
        idx = np.argmax(proba - self.thresholds_.reshape(1, -1), axis=1)
        return idx if self.classes_ is None else np.array(self.classes_)[idx]

    def predict_proba(self, X):
        return self.base_estimator.predict_proba(X)


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
        # pré-pro
        imputer_strategy: str = "median",
        scaler_type: str = "standard",
        var_threshold: float = 0.0,
        corr_threshold: float = 0.98,
        use_pca: bool = False,
        pca_components: float | int = 0.99,
        # imbalance
        sampler: str | None = None,  # none/smote/borderline/smoteenn/adasyn
        # seuils
        tune_thresholds: bool = False,
        threshold_metric: str = "f1_macro",
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

        # Préférences de pré-traitement par défaut
        self.imputer_strategy = imputer_strategy
        self.scaler_type = scaler_type
        self.knn_imputer_k = 5

        self.var_threshold = var_threshold
        self.corr_threshold = corr_threshold
        self.use_pca = use_pca
        self.pca_components = pca_components

        self.sampler = (sampler or "none").lower()
        self.tune_thresholds = tune_thresholds
        self.threshold_metric = threshold_metric

        self.n_jobs: int = -1

        self.feature_names_used: List[str] = []
        self.selected_features_: List[str] | None = None
        self.best_estimator_ = None
        self.model_pipeline = None
        self.label_encoder: LabelEncoder | None = None
        self.class_labels: List[str] = []

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

        # Sélection de l'imputation en fonction de self.imputer_strategy
        imp_strategy = getattr(self, "imputer_strategy", "median")
        if imp_strategy == "none" or imp_strategy is None:
            imputer = None
        elif imp_strategy == "median":
            imputer = SimpleImputer(strategy="median")
        elif imp_strategy == "mean":
            imputer = SimpleImputer(strategy="mean")
        elif imp_strategy == "most_frequent":
            imputer = SimpleImputer(strategy="most_frequent")
        elif imp_strategy == "knn":
            # KNNImputer plus lent mais parfois plus précis
            n_neighbors = getattr(self, "knn_imputer_k", 5)
            imputer = KNNImputer(n_neighbors=int(n_neighbors))
        else:
            imputer = SimpleImputer(strategy="median")

        # Sélection du scaler en fonction de self.scaler_type
        scaler_type = getattr(self, "scaler_type", "standard")
        if scaler_type == "none" or scaler_type is None:
            scaler = None
        elif scaler_type == "standard":
            scaler = StandardScaler()
        elif scaler_type == "robust":
            scaler = RobustScaler()
        else:
            scaler = StandardScaler()

        # Construction du pipeline de prétraitement pour les colonnes numériques
        steps = []
        if imputer is not None:
            steps.append(("imp", imputer))
        if (model_type == "SVM") or scaler is not None:
            if scaler is None and model_type == "SVM":
                scaler = StandardScaler()
            if scaler is not None:
                steps.append(("scaler", scaler))

        transformer = Pipeline(steps) if steps else "passthrough"

        # Sélection dynamique des colonnes numériques (compatible avec colfilter en amont)
        return ColumnTransformer(
            [("num", transformer, make_column_selector(dtype_include=np.number))],
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

        # RandomForest
        if model_type == "RandomForest":
            return RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=None,
                n_jobs=self.n_jobs if hasattr(self, "n_jobs") else -1,
                random_state=random_state,
                class_weight="balanced_subsample",
            )

        # XGBoost
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
                n_jobs=self.n_jobs if hasattr(self, "n_jobs") else -1,
                random_state=random_state,
            )

        # SVM (RBF)
        if model_type == "SVM":
            return SVC(
                kernel="rbf",
                probability=True,
                class_weight="balanced",
                random_state=random_state,
            )

        # ExtraTrees
        if model_type == "ExtraTrees":
            return ExtraTreesClassifier(
                n_estimators=n_estimators,
                bootstrap=True,
                class_weight="balanced_subsample",
                n_jobs=self.n_jobs if hasattr(self, "n_jobs") else -1,
                random_state=random_state,
            )

        # Logistic Regression One-vs-Rest
        if model_type == "LogRegOVR":
            return LogisticRegression(
                multi_class="ovr",
                class_weight="balanced",
                max_iter=2000,
                solver="lbfgs",
                C=1.0,
                random_state=random_state,
            )

        # K Nearest Neighbors
        if model_type == "KNN":
            return KNeighborsClassifier(
                n_neighbors=15,
                weights="distance",
                n_jobs=self.n_jobs if hasattr(self, "n_jobs") else -1,
            )

        # Multi-Layer Perceptron
        if model_type == "MLP":
            return MLPClassifier(
                hidden_layer_sizes=(256, 128),
                activation="relu",
                alpha=0.0001,
                learning_rate="adaptive",
                early_stopping=True,
                max_iter=300,
            )

        # Naive Bayes Gaussian
        if model_type == "NaiveBayes":
            return GaussianNB()

        # Linear Discriminant Analysis
        if model_type == "LDA":
            return LinearDiscriminantAnalysis(
                solver="svd",
                shrinkage=None,
            )

        # Quadratic Discriminant Analysis
        if model_type == "QDA":
            return QuadraticDiscriminantAnalysis(
                reg_param=0.0,
            )

        # CatBoost
        if model_type == "CatBoost":
            if not _HAS_CATBOOST:
                raise ImportError("catboost n'est pas installé. `pip install catboost`")
            params = dict(
                loss_function="MultiClass",
                iterations=n_estimators,
                learning_rate=0.05,
                depth=6,
                l2_leaf_reg=3.0,
                auto_class_weights="Balanced",
                random_seed=random_state,
                verbose=False,
            )
            # thread count
            if hasattr(self, "n_jobs") and self.n_jobs and self.n_jobs > 0:
                params["thread_count"] = self.n_jobs
            return CatBoostClassifier(**params)

        # LightGBM
        if model_type == "LightGBM":
            if not _HAS_LGBM:
                raise ImportError("lightgbm n'est pas installé. `pip install lightgbm`")
            return LGBMClassifier(
                objective="multiclass",
                class_weight="balanced",
                n_estimators=n_estimators,
                learning_rate=0.05,
                num_leaves=31,
                max_depth=-1,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.0,
                reg_lambda=0.0,
                n_jobs=self.n_jobs if hasattr(self, "n_jobs") else -1,
                random_state=random_state,
            )

        if model_type == "SoftVoting":
            # XGB + LGBM + CatBoost (selon dispos)
            ests = []
            if _HAS_XGB:
                ests.append(
                    (
                        "xgb",
                        self._build_estimator(
                            "XGBoost", n_estimators, n_classes, random_state
                        ),
                    )
                )
            if _HAS_LGBM:
                ests.append(
                    (
                        "lgbm",
                        self._build_estimator(
                            "LightGBM", n_estimators, n_classes, random_state
                        ),
                    )
                )
            if _HAS_CATBOOST:
                ests.append(
                    (
                        "cat",
                        self._build_estimator(
                            "CatBoost", n_estimators, n_classes, random_state
                        ),
                    )
                )
            if not ests:
                raise RuntimeError("Aucun learner dispo pour SoftVoting.")
            return VotingClassifier(estimators=ests, voting="soft", n_jobs=self.n_jobs)

        raise ValueError(
            f"Modèle inconnu: {model_type} (attendu: RandomForest | XGBoost | SVM | ExtraTrees | LogRegOVR | KNN | MLP | NaiveBayes | LDA | QDA | CatBoost | LightGBM)"
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
        cv_folds: int = 5,
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
        n_iter: int = 80,
        random_state: int = 42,
        param_overrides: dict | None = None,
        # poids & calibration
        use_balanced_weights: bool = True,
        class_weight_mode: str | None = None,  # "inv_freq" ou None
        class_weight_alpha: float = 1.0,
        weight_col: str | None = None,
        weight_norm: str = "minmax",
        weight_gamma: float = 1.0,  # **nouveau**
        calibrate_probs: bool = False,
        calibration_method: str = "sigmoid",
        calibrate_cv: int = 3,
        calibrate_holdout_size: float = 0.0,
        # répétitions de CV
        repeated_cv: bool = False,
        cv_repeats: int = 1,
        # filtres/sélection
        imputer_strategy: str | None = None,
        knn_imputer_k: int = 5,
        scaler_type: str | None = None,
        selector_method: Optional[str] = None,  # "rfecv" pour RFECV
        mi_top_k: Optional[int] = None,
        # sampler
        sampler: Optional[str] = None,
        # PCA & filtres
        var_threshold: Optional[float] = None,
        corr_threshold: Optional[float] = None,
        use_pca: Optional[bool] = None,
        pca_components: Optional[float | int] = None,
        # seuils
        tune_thresholds: Optional[bool] = None,
        threshold_metric: Optional[str] = None,
    ) -> tuple[
        "SpectralClassifier", list[str], pd.DataFrame, np.ndarray, np.ndarray | None
    ]:
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

        # Appliquer les réglages du pré-processeur pour cette session
        # On sauvegarde les valeurs existantes pour les restaurer ensuite
        orig_imp = getattr(self, "imputer_strategy", None)
        orig_knn = getattr(self, "knn_imputer_k", None)
        orig_scal = getattr(self, "scaler_type", None)
        if imputer_strategy is not None:
            self.imputer_strategy = imputer_strategy
            self.knn_imputer_k = int(knn_imputer_k)
        if scaler_type is not None:
            self.scaler_type = scaler_type
        if var_threshold is not None:
            self.var_threshold = float(var_threshold)
        if corr_threshold is not None:
            self.corr_threshold = float(corr_threshold)
        if use_pca is not None:
            self.use_pca = bool(use_pca)
        if pca_components is not None:
            self.pca_components = pca_components
        if sampler is not None:
            self.sampler = (sampler or "none").lower()
        if tune_thresholds is not None:
            self.tune_thresholds = bool(tune_thresholds)
        if threshold_metric is not None:
            self.threshold_metric = str(threshold_metric)
        # Groupes (optionnel)
        groups_full = None
        if use_groups and group_col and group_col in df_trainable.columns:
            groups_full = df_trainable[group_col].values

        # 1) X/y + encodage XGB
        X_all, y_all = self._prepare_features_and_labels(df_trainable)
        feature_cols_before_fs = list(self.feature_names_used)
        y_all = y_all.astype(str)

        # Mutual information filter (sélection des top-K features avant tout)
        if mi_top_k is not None and mi_top_k > 0:
            try:
                # On calcule la MI sur un sous-échantillon pour la vitesse
                # S'il y a trop de colonnes catégorielles, mutual_info_classif les gère automatiquement
                mi_scores = mutual_info_classif(
                    X_all.fillna(0), y_all, discrete_features=False
                )
                # Sélection des indices des top-K scores
                idx_sorted = np.argsort(mi_scores)[::-1]
                top_idx = idx_sorted[: int(mi_top_k)]
                top_cols = X_all.columns[top_idx]
                X_all = X_all[top_cols].copy()
                self.feature_names_used = list(X_all.columns)
                print(
                    f"  > Mutual information filter : {len(top_cols)} features conservées (top-{mi_top_k})."
                )
            except Exception as _mi_e:
                # Si la MI échoue (par ex. données non numériques), on ignore la sélection MI
                pass

        # Encodage pour XGBoost + mémorisation de l’encodeur et des labels
        label_encoder: Optional[LabelEncoder] = None
        y_enc = y_all

        if self.model_type in ("XGBoost", "SoftVoting", "MLP"):
            label_encoder = LabelEncoder().fit(y_all)
            y_enc = label_encoder.transform(y_all)
            self.label_encoder = label_encoder
        else:
            y_enc = y_all

        # Sauvegarde pour évaluation / décodage et ordre des classes
        self.label_encoder = label_encoder
        self.class_labels = (
            label_encoder.classes_.tolist()
            if label_encoder is not None
            else sorted(pd.Series(y_all).astype(str).unique().tolist())
        )
        # 2) Split train/test (stratifié ou par groupes)
        if use_groups and (groups_full is not None):
            try:
                from sklearn.model_selection import StratifiedGroupKFold

                cv_split = StratifiedGroupKFold(
                    n_splits=1, shuffle=True, random_state=random_state
                )
                tr_idx, te_idx = next(cv_split.split(X_all, y_enc, groups_full))
                groups_train = groups_full[tr_idx]
            except Exception:
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
        # Choix du modèle pour la sélection guidée
        # Priorité : argument selector_method > argument selector_model > attribut self.selector_model
        if selector_method is not None:
            sel_model = selector_method
        elif selector_model is None:
            sel_model = self.selector_model or "rf"
        else:
            sel_model = selector_model
        sel_threshold = (
            self.selector_threshold
            if selector_threshold is None
            else selector_threshold
        )

        # pipeline (imblearn si sampler actif)
        Pipe = (
            ImbPipeline
            if (_HAS_IMBLEARN and (self.sampler not in ("none", None)))
            else sklearn.pipeline.Pipeline
        )

        steps = [
            ("colfilter", CollinearityFilter(self.var_threshold, self.corr_threshold)),
            ("prep", preproc),
        ]
        if fs_enabled:
            # Modèle pour sélectionner les variables
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
            elif sel_model == "ext":
                selector_est = ExtraTreesClassifier(
                    n_estimators=self.selector_n_estimators,
                    n_jobs=-1,
                    random_state=random_state,
                    class_weight="balanced_subsample",
                )
            elif sel_model == "l1":
                # Sélection via régression logistique L1 (sparse)
                selector_est = LogisticRegression(
                    penalty="l1",
                    solver="liblinear",
                    multi_class="ovr",
                    max_iter=1000,
                    random_state=random_state,
                )
            else:
                # Par défaut : RandomForest
                selector_est = RandomForestClassifier(
                    n_estimators=self.selector_n_estimators,
                    n_jobs=-1,
                    random_state=random_state,
                    class_weight="balanced_subsample",
                )

        fs_kind = (selector_method or "sfrommodel").lower()

        if fs_enabled:
            if fs_kind == "rfecv":
                from sklearn.feature_selection import RFECV

                min_feats = max(5, int(0.05 * X_train.shape[1]))  # garde-fou
                steps.append(
                    (
                        "fs",
                        RFECV(
                            estimator=selector_est,
                            step=1,
                            cv=5,  # simple et robuste; ok aussi sans groupes
                            scoring=scoring,
                            n_jobs=-1,
                            min_features_to_select=min_feats,
                        ),
                    )
                )
            else:
                # chemin par défaut: SelectFromModel
                steps.append(
                    ("fs", SelectFromModel(selector_est, threshold=sel_threshold))
                )

        if self.use_pca:
            steps.append(
                ("pca", PCA(n_components=self.pca_components, svd_solver="full"))
            )

        # sampler (dans les folds)
        sampler_name = (sampler or self.sampler or "none").lower()
        if _HAS_IMBLEARN and sampler_name not in ("none", None):
            if sampler_name == "smote":
                steps.append(("sampler", SMOTE(random_state=random_state)))
            elif sampler_name == "borderline":
                steps.append(("sampler", BorderlineSMOTE(random_state=random_state)))
            elif sampler_name == "smoteenn":
                steps.append(("sampler", SMOTEENN(random_state=random_state)))
            elif sampler_name == "adasyn":
                steps.append(("sampler", ADASYN(random_state=random_state)))
            else:
                print(f"(info) sampler inconnu '{sampler_name}', désactivé.")

        # wrap seuils
        final_est = ThresholdTunedClassifier(
            base_est,
            tune=self.tune_thresholds,
            metric=(self.threshold_metric or "f1_macro"),
            random_state=random_state,
        )

        steps.append(("clf", final_est))
        pipe = Pipe(steps)

        # 4) Fit params : weights + early stopping (XGB)
        fit_params: Dict[str, Any] = {}
        # --- Calcul des poids d'échantillons ---
        # On combine (si applicable) plusieurs sources de poids :
        # 1) Poids de classe avancés (inverse des fréquences)
        # 2) Poids équilibrés standards (balanced)
        # 3) Poids provenant d'une colonne du DataFrame (S/N, etc.)

        sample_weight: Optional[np.ndarray] = None

        # 1) Poids de classe avancés
        if class_weight_mode and str(class_weight_mode).lower() != "none":
            # Fréquence de chaque classe dans y_train
            unique, counts = np.unique(y_train, return_counts=True)
            freq_map = {cls: cnt for cls, cnt in zip(unique, counts)}
            # Poids proportionnels à l'inverse de la fréquence à la puissance alpha
            class_weights = {
                cls: (1.0 / freq_map[cls]) ** float(class_weight_alpha)
                for cls in unique
            }
            sample_weight = np.array([class_weights[cls] for cls in y_train])
        elif use_balanced_weights:
            # Poids balanced fourni par scikit-learn
            sample_weight = compute_sample_weight("balanced", y_train)

        # 2) Poids depuis une colonne du DataFrame (ex: rapport S/N)
        if weight_col and weight_col in df_trainable.columns:
            try:
                # Les indices de X_train correspondent aux index de df_trainable
                weight_vals = df_trainable.loc[X_train.index, weight_col].values.astype(
                    float
                )
                # Normalisation
                w = weight_vals.copy()
                # Remplace les NaN par la médiane
                if np.any(pd.isna(w)):
                    med = np.nanmedian(w)
                    w = np.nan_to_num(w, nan=med)
                if weight_norm == "log":
                    # log1p après déplacement pour éviter log(0)
                    w = np.log1p(w - np.min(w) + 1e-6)
                # Min-max scaling
                mn, mx = np.min(w), np.max(w)
                if mx - mn > 0:
                    w = (w - mn) / (mx - mn)
                else:
                    w = np.ones_like(w)
                # puissance gamma
                try:
                    w = np.power(w, float(weight_gamma))
                except Exception:
                    pass
                if sample_weight is None:
                    sample_weight = w
                else:
                    sample_weight = sample_weight * w
            except Exception:
                # en cas d'erreur, on ignore les poids par colonne
                pass

        # Ajout aux paramètres de fit si présent
        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight, dtype=np.float32)
            fit_params["clf__sample_weight"] = sample_weight

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
                prep_for_es = clone(Pipe(pre_fs_steps))
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

        # 5) Stratégie de validation croisée (CV)
        # Si des groupes sont fournis, on privilégie StratifiedGroupKFold (si dispo).
        # Sinon, en mode répétition activé, on utilise RepeatedStratifiedKFold pour
        # stabiliser les scores. À défaut, on reste sur StratifiedKFold classique.
        if use_groups and groups_train is not None:
            try:
                from sklearn.model_selection import StratifiedGroupKFold

                cv = StratifiedGroupKFold(
                    n_splits=cv_folds, shuffle=True, random_state=random_state
                )
            except Exception:
                cv = GroupKFold(n_splits=cv_folds)
        else:
            if repeated_cv:
                # Validation croisée répétée (k-fold × repeats) pour plus de robustesse
                cv = RepeatedStratifiedKFold(
                    n_splits=cv_folds,
                    n_repeats=cv_repeats,
                    random_state=random_state,
                )
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

        # 7) Si calibration demandée ---
        if calibrate_probs:
            # 1) éventuel split holdout dédié à la calibration
            X_tr, y_tr = X_train, y_train
            X_cal = y_cal = None
            sw_tr = None
            sw_all = fit_params.get("clf__sample_weight", None)
            if calibrate_holdout_size and float(calibrate_holdout_size) > 0:
                if sw_all is not None:
                    X_tr, X_cal, y_tr, y_cal, sw_tr, sw_cal = train_test_split(
                        X_train,
                        y_train,
                        sw_all,
                        test_size=float(calibrate_holdout_size),
                        stratify=y_train,
                        random_state=random_state,
                    )
                else:
                    X_tr, X_cal, y_tr, y_cal = train_test_split(
                        X_train,
                        y_train,
                        test_size=float(calibrate_holdout_size),
                        stratify=y_train,
                        random_state=random_state,
                    )

            # 2) refit sur le sous-ensemble d’entraînement (utile si holdout)
            fit_params_tr = dict(fit_params)
            if sw_tr is not None:
                fit_params_tr["clf__sample_weight"] = sw_tr
            best.fit(X_tr, y_tr, **fit_params_tr)

            # 3) séparer pré-proc/fs et step final
            if isinstance(best, _PIPE_TYPES):
                preproc_fs = best[:-1]  # pipeline déjà FIT sans le dernier step
                final_step = best.steps[-1][1]  # dernier step (peut être ton wrapper)
            else:
                preproc_fs = None
                final_step = best

            # 4) estimateur VRAIMENT calibré = base_estimator s’il y a un wrapper
            est_to_cal = getattr(final_step, "base_estimator", final_step)
            if not is_classifier(est_to_cal):
                raise TypeError(
                    f"Estimator to calibrate is not a classifier: {type(est_to_cal)}"
                )

            if X_cal is not None:
                # ---- mode prefit sur holdout ----
                X_cal_trans = (
                    preproc_fs.transform(X_cal) if preproc_fs is not None else X_cal
                )
                est_for_cal = _Float64ProbaWrapper(est_to_cal)
                cal = CalibratedClassifierCV(
                    estimator=est_for_cal, method=calibration_method, cv="prefit"
                )
                cal.fit(
                    X_cal_trans,
                    y_cal,
                    sample_weight=np.asarray(sw_cal, dtype=np.float32),
                )
                # ré-injecter le calibrateur
                if hasattr(final_step, "base_estimator"):
                    final_step.base_estimator = cal
                else:
                    if isinstance(best, _PIPE_TYPES):
                        best.steps[-1] = (best.steps[-1][0], cal)
                    else:
                        best = cal
            else:
                # ---- calibration par CV (pas de holdout) : on met le calibrateur COMME step final ----
                est_for_cal = _Float64ProbaWrapper(est_to_cal)
                cal = CalibratedClassifierCV(
                    estimator=est_for_cal,
                    method=calibration_method,
                    cv=int(calibrate_cv),
                )
                if hasattr(final_step, "base_estimator"):
                    final_step.base_estimator = cal
                    best.fit(X_train, y_train, **fit_params)  # refit complet avec cal
                else:
                    if isinstance(best, _PIPE_TYPES):
                        best.steps[-1] = (best.steps[-1][0], cal)
                        best.fit(X_train, y_train, **fit_params)
                    else:
                        best = cal
                        best.fit(X_train, y_train, **fit_params)

        # 8) Expose artefacts
        self.model_pipeline = best

        # IMPORTANT : pour introspection, on récupère le pipeline "nu" (sans l'enrobage de calibration)
        pipe_for_fs = getattr(best, "base_estimator", best)

        # 9) Features sélectionnées (si FS actif)
        self.selected_features_ = None
        if (
            fs_enabled
            and hasattr(pipe_for_fs, "named_steps")
            and "fs" in pipe_for_fs.named_steps
        ):
            try:
                # Nommer les colonnes après le préproc
                out_names = pipe_for_fs.named_steps["prep"].get_feature_names_out()
                out_names = [n.split("__", 1)[-1] for n in out_names]
            except Exception:
                out_names = self.feature_names_used
            try:
                mask = pipe_for_fs.named_steps["fs"].get_support()
                self.selected_features_ = [
                    n for n, keep in zip(out_names, mask) if keep
                ]
            except Exception:
                # garde-fou
                self.selected_features_ = out_names

        # --- Rétablit les options de pré-processeur si elles ont été surchargées pour cette session ---
        if imputer_strategy is not None:
            self.imputer_strategy = orig_imp
            self.knn_imputer_k = orig_knn
        if scaler_type is not None:
            self.scaler_type = orig_scal

        return (self, feature_cols_before_fs, X_all, y_all, groups_full)

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
        if self.label_encoder is not None:
            if np.issubdtype(np.asarray(y_test).dtype, np.integer):
                y_test_dec = self.label_encoder.inverse_transform(y_test)
            else:
                y_test_dec = y_test
            y_pred_dec = self.label_encoder.inverse_transform(predictions)
        else:
            y_test_dec, y_pred_dec = y_test, predictions

        proba = getattr(self.model_pipeline, "predict_proba", lambda X: None)(X_test)
        if proba is not None:
            # indices numériques des vraies classes
            if (self.label_encoder is not None) and np.issubdtype(
                np.asarray(y_test).dtype, np.integer
            ):
                y_test_dec = self.label_encoder.inverse_transform(y_test)
                y_pred_dec = self.label_encoder.inverse_transform(predictions)
            else:
                y_test_dec, y_pred_dec = y_test, predictions
                label_to_idx = {lab: i for i, lab in enumerate(self.class_labels)}
                y_idx = np.array([label_to_idx.get(str(v), -1) for v in y_test_dec])
                mask = y_idx >= 0
                proba = proba[mask]
                y_idx = y_idx[mask]
            ece = _ece_score(y_idx, proba)
            top2 = (y_idx == np.argsort(proba, axis=1)[:, -2:]).any(axis=1).mean()
            print(f"ECE={ece:.3f} | Top-2 acc={top2:.3f}")

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
        try:
            import xgboost

            xgb_ver = getattr(xgboost, "__version__", "unknown")
        except Exception:
            xgb_ver = "not-installed"

        joblib.dump(self, path)
        meta = {
            "saved_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "xgboost": xgb_ver,
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
