"""AstroSpectro — Spectral classifier training and evaluation tools.

This module encapsulates the full ML pipeline (imputation, scaling, SMOTE,
optional feature selection, hyperparameter tuning via GridSearchCV), model
training, evaluation, and persistence (pickle + JSON metadata).

Conventions
-----------
- Lengths and widths are in native scikit-learn / xgboost units.
- ``feature_names_used`` preserves column order after preparation (stable).

Inputs / Outputs
----------------
Input :
    A features DataFrame (numeric + meta columns) containing at least
    ``subclass`` to derive the target; other columns are auto-filtered
    by ``_prepare_features_and_labels``.
Output :
    Trained model (sklearn / imblearn pipeline), evaluation reports,
    persisted artifacts (``.pkl`` / ``.json``) via ``save_model()``.

Public API (main methods)
-------------------------
- clean / filter : ``_clean_and_filter_data(df)`` → training-ready DataFrame
- features / labels : ``_prepare_features_and_labels(df)`` → (X, y)
- train : ``train_and_evaluate(features_df, ...)`` → (self, cols, X, y)
- eval : ``evaluate(X_test, y_test)`` → prints report + confusion matrix
- IO : ``save_model(path)``, ``load_model(path)``

Examples
--------
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

# ---------- utilities ----------


def _ece_score(y_true: np.ndarray, proba: np.ndarray, n_bins: int = 15) -> float:
    """
    Compute the Expected Calibration Error (ECE) for a multi-class problem.

    Compare predicted probabilities to observed accuracy within confidence
    bins to estimate the calibration error.

    Parameters
    ----------
    y_true : np.ndarray
        True label vector.
    proba : np.ndarray
        Predicted probability matrix (n_samples x n_classes).
    n_bins : int
        Number of confidence bins used for the computation.

    Returns
    -------
    float
        ECE value (lower is better).
    """
    # confidence = max proba; correct = 1 if argmax proba == y_true
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
    """
    Wrap a classifier to force ``predict_proba`` output to ``float64``.

    This class delegates all attributes and methods to the base estimator and
    ensures that ``predict_proba`` returns a ``float64`` array.  It does not
    expose ``decision_function`` unless the base estimator provides one.
    """

    _estimator_type = "classifier"

    def __init__(self, base):
        self.base = base

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError
        return getattr(self.base, name)

    # --- sklearn API ---
    def get_params(self, deep=True):
        return {"base": self.base}

    def set_params(self, **params):
        if "base" in params:
            self.base = params["base"]
        return self

    def fit(self, X, y=None, **fit_params):
        # fit the base model as-is
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
        return getattr(self.base, "classes_", None)

    def __sklearn_is_fitted__(self):
        # compatibility utility
        return (
            hasattr(self.base, "__sklearn_is_fitted__")
            and self.base.__sklearn_is_fitted__()
        )


class CollinearityFilter(TransformerMixin, BaseEstimator):
    """
    Collinearity and low-variance filter for DataFrames.

    Identify and remove numeric columns whose variance is below a threshold
    or whose absolute correlation with another column exceeds a given
    threshold.  Can be used inside a scikit-learn pipeline to reduce
    feature redundancy.

    Parameters
    ----------
    var_threshold : float
        Minimum variance below which a column is dropped.
        Use ``0.0`` to disable variance filtering.
    corr_threshold : float
        Absolute correlation threshold above which a redundant column is
        dropped.  Must be in [0, 1].

    Attributes
    ----------
    keep_columns_ : list[str] or None
        Columns retained after ``fit`` has been called.
    """

    def __init__(self, var_threshold: float = 0.0, corr_threshold: float = 0.98):
        self.var_threshold = var_threshold
        self.corr_threshold = corr_threshold
        self.keep_columns_: List[str] | None = None

    def fit(self, X: pd.DataFrame, y=None, **fit_params):
        """
        Learn which columns to keep based on variance and correlation.

        Parameters
        ----------
        X : pd.DataFrame
            Input data with numeric columns.
        y : ignored
            Present for sklearn API compatibility.
        **fit_params : dict
            Additional parameters (ignored).

        Returns
        -------
        CollinearityFilter
            The fitted instance.
        """
        if not isinstance(X, pd.DataFrame):
            # attempt to reconstruct a DataFrame if possible
            X = pd.DataFrame(X)
        df = X.copy()
        # Variance threshold
        if self.var_threshold and self.var_threshold > 0:
            vt = VarianceThreshold(self.var_threshold)
            vt.fit(df)
            df = df.loc[:, vt.get_support()]
        # Fill NaN with median (on the training set)
        df = df.copy()
        for c in df.columns:
            if df[c].isna().any():
                df[c] = df[c].fillna(df[c].median())
        # Correlation
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
        """Transform data by keeping only the selected columns.

        Parameters
        ----------
        X : array-like or pd.DataFrame
            Data to transform.

        Returns
        -------
        pd.DataFrame or np.ndarray
            Subset of *X* containing only the columns selected during
            ``fit``.  If no columns have been learned, raw values are
            returned.
        """
        df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        if self.keep_columns_ is None:
            return df.values
        return df.loc[:, [c for c in self.keep_columns_ if c in df.columns]]


class ThresholdTunedClassifier(ClassifierMixin, BaseEstimator):
    """Add per-class threshold tuning to a classification estimator.

    This wrapper applies a class-specific threshold vector to the
    probabilities returned by the base estimator.  During ``fit``, if
    ``tune`` is ``True``, it searches for optimal thresholds on an internal
    validation subset before re-fitting the model on the full training set.
    Prediction is computed via ``argmax(proba - thresholds)``.

    Parameters
    ----------
    base_estimator : estimator
        Base estimator supporting ``predict_proba``.
    tune : bool
        If ``True``, tune thresholds on a validation subset.
    metric : str
        Metric used to optimise thresholds (``'f1_macro'`` or
        ``'balanced_accuracy'``).
    grid : np.ndarray or None
        Threshold grid to explore.  If ``None``, a linear grid is used.
    random_state : int
        Reproducibility seed for the internal split.

    Attributes
    ----------
    thresholds_ : np.ndarray or None
        Per-class thresholds learned during tuning.
    classes_ : np.ndarray or None
        Class label array.
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

        # Keep only kwargs actually supported
        clean = {k: v for k, v in fit_params.items() if k in supported}

        # Add sample_weight only if supported
        if sample_weight is not None and "sample_weight" in supported:
            clean["sample_weight"] = sample_weight
        elif sample_weight is not None:
            warnings.warn(
                f"{estimator.__class__.__name__} ne supporte pas sample_weight ; "
                "weights will be ignored for this model."
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
        # retrieve/forward useful params
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
            # internal split to tune thresholds
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

            # common grid then per-class refinement
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

            # 2) final refit on all data with forwarded params
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
            # no tuning -> direct fit with forwarded params
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
    """Spectral classifier for stellar spectral type classification.

    This class encapsulates the full data-preparation pipeline, feature
    selection, class-imbalance handling, multi-model training,
    hyperparameter search via ``GridSearchCV`` or ``RandomizedSearchCV``,
    evaluation, and persistence.  It supports several models (RandomForest,
    XGBoost, SVM, etc.) and dynamically selects relevant numeric features.

    Parameters
    ----------
    model_type : str
        Final model to train (e.g. ``'XGBoost'``, ``'RandomForest'``).
    prediction_target : str
        Prediction target (e.g. ``'main_class'``, ``'sub_class_top25'``,
        ``'sub_class_bins'``).
    use_feature_selection : bool
        Enable feature selection via ``SelectFromModel``.
    selector_threshold : str
        Threshold for selection (e.g. ``'median'``, ``'mean'``,
        ``'0.5*mean'``).
    selector_model : str
        Estimator used for feature selection (``'xgb'`` or ``'rf'``).
    selector_n_estimators : int
        Number of trees / iterations for the selector estimator.
        random_state : int
        Reproducibility seed for all random generators.

    Attributes
    ----------
    feature_names_used : list[str]
        Candidate feature names detected during preparation.
    selected_features_ : list[str] or None
        Subset retained after selection, or ``None`` if not applied.
    best_estimator_
        Final pipeline from hyperparameter search.
    model_pipeline
        Full pipeline (preprocessor + estimator) after training.
    label_encoder : LabelEncoder or None
        Label encoder used for models like XGBoost.
    class_labels : list[str]
        Class labels in the order used by the encoder.
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
        # preprocessing
        imputer_strategy: str = "median",
        scaler_type: str = "standard",
        var_threshold: float = 0.0,
        corr_threshold: float = 0.98,
        use_pca: bool = False,
        pca_components: float | int = 0.99,
        # imbalance
        sampler: str | None = None,
        # seuils
        tune_thresholds: bool = False,
        threshold_metric: str = "f1_macro",
    ):
        """
        Initialise the classification pipeline configuration.

        Parameters
        ----------
        model_type : str
            Final model to train.  One of ``{"XGBoost", "RandomForest"}``.
        prediction_target : str
            Prediction target (e.g. ``"main_class"``, ``"sub_class_top25"``,
            ``"sub_class_bins"``).

            .. note::

               ``sub_class_bins`` refers to classes grouped into bins;
               sub-classes are ordered by frequency.
        use_feature_selection : bool
            Enable feature selection via ``SelectFromModel``.
        selector_threshold : str
            Selection threshold (e.g. ``"median"``, ``"mean"``,
            ``"0.5*mean"``).
        selector_model : str
            Estimator used for feature selection (``"xgb"`` or ``"rf"``).
        selector_n_estimators : int
            Number of trees / boosting rounds for the selector.
        random_state : int
            Reproducibility seed.

        Attributes
        ----------
        feature_names_used : list[str]
            Candidate feature names detected.
        selected_features_ : list[str] or None
            Subset retained after selection.
        best_estimator_
            Final pipeline from GridSearchCV (once trained).
        """
        self.model_type = model_type
        self.prediction_target = prediction_target
        self.use_feature_selection = use_feature_selection
        self.selector_threshold = selector_threshold
        self.selector_model = selector_model
        self.selector_n_estimators = selector_n_estimators
        self.random_state = random_state

        # Default preprocessing preferences
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
    # Data preparation (label construction + cleaning)
    # ---------------------------------------------------------------------

    def _clean_and_filter_data(self, df: pd.DataFrame) -> pd.DataFrame | None:
        """Build the label column and filter examples.

        Depending on ``prediction_target``, create a ``label`` column from
        ``subclass``, remove rows with invalid labels, and discard classes
        with too few examples.  The original DataFrame is not modified;
        a new object is returned.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing feature columns and a ``subclass`` column
            from which the target is derived.

        Returns
        -------
        pd.DataFrame or None
            Training-ready DataFrame with a ``label`` column, or ``None``
            if cleaning reduces the dataset below a usable size.

        Notes
        -----
        - Invalid labels (``'U'``, ``'n'``, ``'N'``, ``'O'``, ``'OTHER'``)
          are removed.
        - Classes with fewer than 10 examples are dropped to ensure
          minimum support during training.
        """
        if "subclass" not in df.columns:
            print("WARNING: 'subclass' column not found. Cannot create labels.")
            return None

        df_copy = df.copy()

        if self.prediction_target == "main_class":
            print("  > Strategy: Main-class classification (A, F, G...).")
            df_copy["label"] = df_copy["subclass"].astype(str).str[0]

        elif self.prediction_target == "sub_class_top25":
            print(
                "  > Strategy: Fine classification on the 25 most frequent sub-classes."
            )
            df_valid = df_copy[
                df_copy["subclass"].notnull() & (df_copy["subclass"] != "Non")
            ].copy()
            # # Find the 25 most common sub-classes
            top_n_subclasses = (
                df_valid["subclass"].value_counts().nlargest(25).index.tolist()
            )
            df_copy = df_valid[df_valid["subclass"].isin(top_n_subclasses)].copy()
            df_copy["label"] = df_copy["subclass"]
            print(f"  > Selected {len(top_n_subclasses)} sub-classes for training.")
            # Why: limit granularity to "frequent" classes to avoid
            # degenerate train/test splits and uninformative metrics.
        elif self.prediction_target == "sub_class_bins":
            print("  > Strategy: Fine classification by bins (e.g. G_early, G_late).")

            def map_to_bin(subclass: str) -> str:
                """
                Group a fine spectral sub-class into a broad bin of type
                ``X_early`` or ``X_late``.

                Rules
                -----
                - Only classic formats starting with a letter in {A,F,G,K,M}
                  followed by a digit are handled (e.g. ``"G2"``, ``"F5V"``).
                - digit < 5 -> ``"<TYPE>_early"`` (e.g. G0-G4 -> ``"G_early"``).
                - digit >= 5 -> ``"<TYPE>_late"`` (e.g. G5-G9 -> ``"G_late"``).
                - Unrecognised format -> ``"OTHER"``.

                Parameters
                ----------
                subclass : str
                    Original sub-class label (e.g. ``"G2"``, ``"K7III"``).

                Returns
                -------
                str
                    Broad bin (``"G_early"``, ``"G_late"``, ...) or ``"OTHER"``.
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
                f"Invalid prediction_target '{self.prediction_target}'. Valid choices: 'main_class', 'sub_class_top25', 'sub_class_bins'."
            )

        initial_count = len(df_copy)
        invalid_labels = ["U", "n", "N", "O", "OTHER"]
        df_trainable = df_copy[
            df_copy["label"].notnull() & ~df_copy["label"].isin(invalid_labels)
        ].copy()
        print(
            f"  > {initial_count - len(df_trainable)} rows with invalid or null labels removed."
        )

        label_counts = df_trainable["label"].value_counts()
        rare_labels = label_counts[label_counts < 10].index.tolist()
        if rare_labels:
            # Note: threshold 10 chosen for split robustness + SMOTE.
            print(f"  > Dropping rare classes (fewer than 10 samples): {rare_labels}")
            df_trainable = df_trainable[~df_trainable["label"].isin(rare_labels)]

        if len(df_trainable) < 20:
            print("\nERROR: Not enough valid data after cleaning to train a model.")
            return None

        print(
            f"  > {len(df_trainable)} final samples with {len(df_trainable['label'].unique())} classes for training."
        )
        return df_trainable

    # ---------------------------------------------------------------------
    # Dynamic feature selection & X/y construction
    # ---------------------------------------------------------------------

    def _prepare_features_and_labels(
        self, df_trainable: pd.DataFrame
    ) -> tuple[pd.DataFrame, np.ndarray]:
        """
        Dynamically select numeric columns to build X and y.

        Remove non-numeric columns, replace non-numeric / infinite values
        with ``NaN``, drop empty or constant columns, sort columns, and
        return the feature matrix and label vector.  Column order is stored
        in ``feature_names_used``.

        Parameters
        ----------
        df_trainable : pd.DataFrame
            Cleaned DataFrame containing at least a ``label`` column.

        Returns
        -------
        tuple[pd.DataFrame, np.ndarray]
            ``(X, y)`` where *X* is a numeric feature DataFrame and *y*
            is the label array.

        Side Effects
        ------------
        Update ``feature_names_used`` with the final column order.
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
            # Dropping constant columns avoids noise for
            # feature selection and speeds up fitting.
            df_num.drop(columns=const_cols, inplace=True)

        df_num = df_num.reindex(sorted(df_num.columns), axis=1)
        self.feature_names_used = list(df_num.columns)

        X = df_num
        y = df_trainable["label"].values

        print("\n--- Training Preparation (Dynamic Feature Detection) ---")
        print(
            f"Features used ({len(self.feature_names_used)}): {self.feature_names_used}"
        )

        return X, y

    def _build_preprocessor(
        self, X_train: pd.DataFrame, model_type: str
    ) -> ColumnTransformer:
        """
        Build the preprocessor for the pipeline.

        The preprocessor performs missing-value imputation and
        normalisation / standardisation of numeric columns according to
        the chosen model type.  For models that do not require scaling,
        only imputation is applied.

        Parameters
        ----------
        X_train : pd.DataFrame
            Training feature matrix.
        model_type : str
            Final model type (e.g. ``'RandomForest'``, ``'XGBoost'``,
            ``'SVM'``).

        Returns
        -------
        ColumnTransformer
            Transformer applying imputation and scaling steps on numeric
            columns.

        Raises
        ------
        ValueError
            If no numeric columns are available in *X_train*.
        """
        num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
        if not num_cols:
            raise ValueError("No numeric columns available for training.")

        # Select imputation based on self.imputer_strategy
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
            # KNNImputer slower but sometimes more accurate
            n_neighbors = getattr(self, "knn_imputer_k", 5)
            imputer = KNNImputer(n_neighbors=int(n_neighbors))
        else:
            imputer = SimpleImputer(strategy="median")

        # Select scaler based on self.scaler_type
        scaler_type = getattr(self, "scaler_type", "standard")
        if scaler_type == "none" or scaler_type is None:
            scaler = None
        elif scaler_type == "standard":
            scaler = StandardScaler()
        elif scaler_type == "robust":
            scaler = RobustScaler()
        else:
            scaler = StandardScaler()

        # Build the preprocessing pipeline for numeric columns
        steps = []
        if imputer is not None:
            steps.append(("imp", imputer))
        if (model_type == "SVM") or scaler is not None:
            if scaler is None and model_type == "SVM":
                scaler = StandardScaler()
            if scaler is not None:
                steps.append(("scaler", scaler))

        transformer = Pipeline(steps) if steps else "passthrough"

        # Dynamic numeric column selection (compatible with upstream colfilter)
        return ColumnTransformer(
            [("num", transformer, make_column_selector(dtype_include=np.number))],
            remainder="drop",
        )

    def _build_estimator(
        self, model_type: str, n_estimators: int, n_classes: int, random_state: int
    ):
        """
        Build the base estimator according to the model type.

        Parameters
        ----------
        model_type : str
            Model choice (``'RandomForest'``, ``'XGBoost'``, ``'SVM'``,
            ``'ExtraTrees'``, ``'LogRegOVR'``, ``'KNN'``, ``'MLP'``,
            ``'NaiveBayes'``, ``'LDA'``, ``'QDA'``, ``'CatBoost'``,
            ``'LightGBM'``, ``'SoftVoting'``).
        n_estimators : int
            Number of trees or iterations, depending on the model.
        n_classes : int
            Number of target classes.
        random_state : int
            Reproducibility seed.

        Returns
        -------
        estimator
            A scikit-learn-compatible estimator.

        Raises
        ------
        ImportError
            If the chosen model requires a package that is not installed.
            ValueError: Si ``model_type`` n'est pas reconnu.
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
                    "XGBoost is not installed in this environment."
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
                tree_method="hist",  # fast and CPU-friendly
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
                raise ImportError("catboost is not installed. `pip install catboost`")
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
                raise ImportError("lightgbm is not installed. `pip install lightgbm`")
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
            # XGB + LGBM + CatBoost
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
                raise RuntimeError("No learner available for SoftVoting.")
            return VotingClassifier(estimators=ests, voting="soft", n_jobs=self.n_jobs)

        raise ValueError(
            f"Unknown model: {model_type} (expected: RandomForest | XGBoost | SVM | ExtraTrees | LogRegOVR | KNN | MLP | NaiveBayes | LDA | QDA | CatBoost | LightGBM)"
        )

    # ---------------------------------------------------------------------
    # Training + tuning + (optional) feature selection
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
        class_weight_mode: str | None = None,
        class_weight_alpha: float = 1.0,
        weight_col: str | None = None,
        weight_norm: str = "minmax",
        weight_gamma: float = 1.0,  # **nouveau**
        calibrate_probs: bool = False,
        calibration_method: str = "sigmoid",
        calibrate_cv: int = 3,
        calibrate_holdout_size: float = 0.0,
        # CV repetitions
        repeated_cv: bool = False,
        cv_repeats: int = 1,
        # filters / selection
        imputer_strategy: str | None = None,
        knn_imputer_k: int = 5,
        scaler_type: str | None = None,
        selector_method: Optional[str] = None,
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
        """Train the spectral classifier and perform an initial evaluation.

        Execute the full pipeline: data cleaning, ``X`` / ``y``
        preparation, optional feature selection, hyperparameter search via
        ``GridSearchCV`` or ``RandomizedSearchCV``, final model training,
        and evaluation report on the test set.  Many parameters control
        class balancing, imputation, scaling, feature selection,
        dimensionality reduction, oversampling, early stopping, and
        probability calibration.

        Parameters
        ----------
            features_df : pd.DataFrame
            Full DataFrame (features + metadata columns) before cleaning.
            test_size : float, default 0.21
            Proportion of the dataset held out for testing. Defaults to 0.21.
            n_estimators : int, default 300
            Number of trees / iterations for the base estimator.
            search : str or None, default None
            Hyperparameter search type (``'grid'`` or ``'random'``).
            cv_folds : int, default 5
            Number of cross-validation folds. Defaults to 5.
            scoring : str, default 'accuracy'
            Metric optimised by the search. Defaults to 'accuracy'.
            use_feature_selection : bool or None
            Force feature selection on or off.
            selector_model : str or None
            Model used for feature selection.
            selector_threshold : str or None
            Threshold used for feature selection.
            early_stopping : bool, default False
            Enable early stopping for compatible estimators.
            early_stopping_rounds : int, default 50
            Iterations without improvement before stopping.
            val_size : float, default 0.15
            Validation proportion for early stopping.
            use_groups : bool, default False
            Use a group column to stratify the split. Defaults to False.
            group_col : str or None
            Column containing group labels for the split. Defaults to None.
            param_grid : dict or None
            Hyperparameter grid for ``GridSearchCV``.
            param_distributions : dict or None
            Distributions for ``RandomizedSearchCV``.
            n_iter : int, default 80
            Iterations for ``RandomizedSearchCV``.
            random_state : int, default 42
            Random seed. Defaults to 42.
            param_overrides : dict or None
            Extra parameters forced on the estimator.
            use_balanced_weights : bool, default True
            Compute class-balancing sample weights.
            class_weight_mode : str or None
            Weight method (``'inv_freq'`` or ``None``).
            class_weight_alpha : float, default 1.0
            Exponent for inverse-frequency weights. Defaults to 1.0.
            weight_col : str or None
            Column containing custom sample weights.
            weight_norm : str, default 'minmax'
            Weight normalisation method.
            weight_gamma : float, default 1.0
            Exponent applied to the weight distribution. Defaults to 1.0.
            calibrate_probs : bool, default False
            Apply post-hoc probability calibration. Defaults to False.
            calibration_method : str, default 'sigmoid'
            Calibration method (``'sigmoid'`` or ``'isotonic'``).
            calibrate_cv : int, default 3
            Folds for calibration cross-validation. Defaults to 3.
            calibrate_holdout_size : float, default 0.0
            Holdout size for calibration.
            repeated_cv : bool, default False
            Enable repeated cross-validation. Defaults to False.
            cv_repeats : int, default 1
            Number of CV repetitions. Defaults to 1.
            imputer_strategy : str or None
            Imputation strategy overriding the instance value.
            knn_imputer_k : int, default 5
            Neighbours for ``KNNImputer``. Defaults to 5.
            scaler_type : str or None
            Scaler type (``'standard'``, ``'robust'``, etc.).
            selector_method : str or None
            Selection method (e.g. ``'rfecv'``).
            mi_top_k : int or None
            Number of features to keep based on mutual information.
            sampler : str or None
            Oversampling method (``'smote'``, ``'adasyn'``, etc.).
            var_threshold : float or None
            Variance threshold for column dropping. Defaults to None.
            corr_threshold : float or None
            Correlation threshold for column dropping. Defaults to None.
            use_pca : bool or None
            Apply PCA dimensionality reduction. Defaults to None.
            pca_components : float, int, or None
            Number or fraction of PCA components to keep. Defaults to None.
            tune_thresholds : bool or None
            Enable threshold tuning via ``ThresholdTunedClassifier``.
            threshold_metric : str or None
            Metric for threshold optimisation. Defaults to None.

        Returns:
            Tuple[SpectralClassifier, List[str], pd.DataFrame, np.ndarray, np.ndarray | None]:
                - ``self`` : The trained instance (allows chaining).
                - ``cols_for_report`` : Columns actually used for training.
                - ``X`` (pd.DataFrame) : Feature matrix before the split.
                - ``y`` (np.ndarray) : Label vector.
                - ``np.ndarray | None`` : Group array or ``None``.

        Raises:
            ValueError
            If the hyperparameter search fails or a required estimator
            is missing.
        """
        print(
            f"\n=== STEP 4: TRAINING SESSION (Model: {self.model_type}, Target: {self.prediction_target}) ==="
        )

        # 0) Cleaning + label
        df_trainable = self._clean_and_filter_data(features_df)
        if df_trainable is None:
            print("> No usable data after cleaning; stopping.")
            return False

        # Apply preprocessor settings for this session
        # Save existing values to restore them later
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
        # Groups (optional)
        groups_full = None
        if use_groups and group_col and group_col in df_trainable.columns:
            groups_full = df_trainable[group_col].values

        # 1) X/y + XGBoost encoding
        X_all, y_all = self._prepare_features_and_labels(df_trainable)
        feature_cols_before_fs = list(self.feature_names_used)
        y_all = y_all.astype(str)

        # Mutual information filter (select top-K features upfront)
        if mi_top_k is not None and mi_top_k > 0:
            try:
                # Compute MI on a subsample for speed
                # mutual_info_classif handles categorical columns automatically
                mi_scores = mutual_info_classif(
                    X_all.fillna(0), y_all, discrete_features=False
                )
                # Select indices of top-K scores
                idx_sorted = np.argsort(mi_scores)[::-1]
                top_idx = idx_sorted[: int(mi_top_k)]
                top_cols = X_all.columns[top_idx]
                X_all = X_all[top_cols].copy()
                self.feature_names_used = list(X_all.columns)
                print(
                    f"  > Mutual information filter: {len(top_cols)} features retained (top-{mi_top_k})."
                )
            except Exception as _mi_e:
                # If MI fails (e.g. non-numeric data), skip MI selection
                pass

        # Encode for XGBoost + store encoder and labels
        label_encoder: Optional[LabelEncoder] = None
        y_enc = y_all

        if self.model_type in ("XGBoost", "SoftVoting", "MLP"):
            label_encoder = LabelEncoder().fit(y_all)
            y_enc = label_encoder.transform(y_all)
            self.label_encoder = label_encoder
        else:
            y_enc = y_all

        # Store for evaluation / decoding and class order
        self.label_encoder = label_encoder
        self.class_labels = (
            label_encoder.classes_.tolist()
            if label_encoder is not None
            else sorted(pd.Series(y_all).astype(str).unique().tolist())
        )
        # 2) Train/test split (stratified or group-based)
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
        }  # for the report

        X_train = X_all.iloc[tr_idx]
        y_train = y_enc[tr_idx]

        # 3) Preprocessor + estimator + (optional) SelectFromModel
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
        # Choose model for guided selection
        # Priority: selector_method arg > selector_model arg > self.selector_model attr
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

        # pipeline (imblearn if sampler active)
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
            # Model for feature selection
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
                # Selection via L1 logistic regression (sparse)
                selector_est = LogisticRegression(
                    penalty="l1",
                    solver="liblinear",
                    multi_class="ovr",
                    max_iter=1000,
                    random_state=random_state,
                )
            else:
                # Default: RandomForest
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

                min_feats = max(5, int(0.05 * X_train.shape[1]))  # safeguard
                steps.append(
                    (
                        "fs",
                        RFECV(
                            estimator=selector_est,
                            step=1,
                            cv=5,  # simple and robust; works without groups too
                            scoring=scoring,
                            n_jobs=-1,
                            min_features_to_select=min_feats,
                        ),
                    )
                )
            else:
                # default path: SelectFromModel
                steps.append(
                    ("fs", SelectFromModel(selector_est, threshold=sel_threshold))
                )

        if self.use_pca:
            steps.append(
                ("pca", PCA(n_components=self.pca_components, svd_solver="full"))
            )

        # sampler (applied within folds)
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
                print(f"(info) unknown sampler '{sampler_name}', disabled.")

        # threshold wrapper
        final_est = ThresholdTunedClassifier(
            base_est,
            tune=self.tune_thresholds,
            metric=(self.threshold_metric or "f1_macro"),
            random_state=random_state,
        )

        steps.append(("clf", final_est))
        pipe = Pipe(steps)

        # 4) Fit params: weights + early stopping (XGB)
        fit_params: Dict[str, Any] = {}
        # --- Sample weight computation ---
        # Combine (when applicable) several weight sources:
        # 1) Advanced class weights (inverse frequencies)
        # 2) Standard balanced weights
        # 3) Weights from a DataFrame column (S/N, etc.)

        sample_weight: Optional[np.ndarray] = None

        # 1) Advanced class weights
        if class_weight_mode and str(class_weight_mode).lower() != "none":
            # Frequency of each class in y_train
            unique, counts = np.unique(y_train, return_counts=True)
            freq_map = {cls: cnt for cls, cnt in zip(unique, counts)}
            # Weights proportional to inverse frequency raised to alpha
            class_weights = {
                cls: (1.0 / freq_map[cls]) ** float(class_weight_alpha)
                for cls in unique
            }
            sample_weight = np.array([class_weights[cls] for cls in y_train])
        elif use_balanced_weights:
            # Balanced weights provided by scikit-learn
            sample_weight = compute_sample_weight("balanced", y_train)

        # 2) Weights from a DataFrame column (e.g. S/N ratio)
        if weight_col and weight_col in df_trainable.columns:
            try:
                # X_train indices correspond to df_trainable index
                weight_vals = df_trainable.loc[X_train.index, weight_col].values.astype(
                    float
                )
                # Normalisation
                w = weight_vals.copy()
                # Replace NaN with median
                if np.any(pd.isna(w)):
                    med = np.nanmedian(w)
                    w = np.nan_to_num(w, nan=med)
                if weight_norm == "log":
                    # log1p after shifting to avoid log(0)
                    w = np.log1p(w - np.min(w) + 1e-6)
                # Min-max scaling
                mn, mx = np.min(w), np.max(w)
                if mx - mn > 0:
                    w = (w - mn) / (mx - mn)
                else:
                    w = np.ones_like(w)
                # gamma power
                try:
                    w = np.power(w, float(weight_gamma))
                except Exception:
                    pass
                if sample_weight is None:
                    sample_weight = w
                else:
                    sample_weight = sample_weight * w
            except Exception:
                # on error, ignore column-based weights
                pass

        # Add to fit params if present
        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight, dtype=np.float32)
            fit_params["clf__sample_weight"] = sample_weight

        # ES only outside search (otherwise masks unstable FS across folds)
        es_active = (
            (self.model_type == "XGBoost") and early_stopping and (search is None)
        )
        if es_active:
            # internal split for ES
            X_tr_in, X_val_in, y_tr_in, y_val_in = train_test_split(
                X_train,
                y_train,
                test_size=val_size,
                stratify=y_train,
                random_state=random_state,
            )

            # clone "prep + fs" pipeline and FIT on ALL X_train (same data as final fit)
            pre_fs_steps = steps[
                :-1
            ]  # ['prep', ('fs', ...)] si FS actif, sinon juste 'prep'
            if pre_fs_steps:
                prep_for_es = clone(Pipe(pre_fs_steps))
                prep_for_es.fit(X_train, y_train)
                X_val_es = prep_for_es.transform(X_val_in)
            else:
                X_val_es = X_val_in  # (case without preproc/FS)

            # XGB 1.x / 2.x compat: pick the supported API
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

        # 5) Cross-validation strategy (CV)
        # If groups are provided, prefer StratifiedGroupKFold (if available).
        # Otherwise, if repeated mode is on, use RepeatedStratifiedKFold to
        # stabilise scores.  Otherwise, fall back to standard StratifiedKFold.
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
                # Repeated cross-validation (k-fold x repeats) for added robustness
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

        # 7) If calibration requested ---
        if calibrate_probs:
            # 7.1) optional holdout split dedicated to calibration
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

            # 7.2) refit on training subset (useful if holdout)
            fit_params_tr = dict(fit_params)
            if sw_tr is not None:
                fit_params_tr["clf__sample_weight"] = sw_tr
            best.fit(X_tr, y_tr, **fit_params_tr)

            # 7.3) separate preproc/fs from final step
            if isinstance(best, _PIPE_TYPES):
                preproc_fs = best[:-1]
                final_step = best.steps[-1][1]
            else:
                preproc_fs = None
                final_step = best

            # 7.4) actually calibrated estimator = base_estimator if wrapped
            est_to_cal = getattr(final_step, "base_estimator", final_step)
            if not is_classifier(est_to_cal):
                raise TypeError(
                    f"Estimator to calibrate is not a classifier: {type(est_to_cal)}"
                )

            if X_cal is not None:
                # ---- prefit mode on holdout ----
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
                # re-inject the calibrator
                if hasattr(final_step, "base_estimator"):
                    final_step.base_estimator = cal
                else:
                    if isinstance(best, _PIPE_TYPES):
                        best.steps[-1] = (best.steps[-1][0], cal)
                    else:
                        best = cal
            else:
                # ---- CV calibration (no holdout): set calibrator as final step ----
                est_for_cal = _Float64ProbaWrapper(est_to_cal)
                cal = CalibratedClassifierCV(
                    estimator=est_for_cal,
                    method=calibration_method,
                    cv=int(calibrate_cv),
                )
                if hasattr(final_step, "base_estimator"):
                    final_step.base_estimator = cal
                    best.fit(X_train, y_train, **fit_params)
                else:
                    if isinstance(best, _PIPE_TYPES):
                        best.steps[-1] = (best.steps[-1][0], cal)
                        best.fit(X_train, y_train, **fit_params)
                    else:
                        best = cal
                        best.fit(X_train, y_train, **fit_params)

        # 8) Expose artifacts
        self.model_pipeline = best

        # For introspection, retrieve the pipeline without calibration wrapper
        pipe_for_fs = getattr(best, "base_estimator", best)

        # 9) Selected features (if FS active)
        self.selected_features_ = None
        if (
            fs_enabled
            and hasattr(pipe_for_fs, "named_steps")
            and "fs" in pipe_for_fs.named_steps
        ):
            try:
                # Name columns after preprocessing
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
                # safeguard
                self.selected_features_ = out_names

        # --- Restore preprocessor options if they were overridden for this session ---
        if imputer_strategy is not None:
            self.imputer_strategy = orig_imp
            self.knn_imputer_k = orig_knn
        if scaler_type is not None:
            self.scaler_type = orig_scal

        return (self, feature_cols_before_fs, X_all, y_all, groups_full)

    # ---------------------------------------------------------------------
    # Evaluation & persistence
    # ---------------------------------------------------------------------

    def evaluate(self, X_test: pd.DataFrame | np.ndarray, y_test: np.ndarray) -> None:
        """
        Evaluate the trained model on the test set.

        Print the classification report and display the confusion matrix.

        Parameters
        ----------
        X_test : pd.DataFrame or np.ndarray
            Test set transformed like the training set (same features).
        y_test : np.ndarray
            Test labels (encoded if the model is XGBoost).
        """
        predictions = self.model_pipeline.predict(X_test)

        # Decode if XGBoost
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
            # numeric indices of true classes
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

        print("\n--- Evaluation Report ---")
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
        plt.xlabel("Prediction")
        plt.ylabel("True Value")
        plt.title(f"Confusion Matrix — Optimised {self.model_type} Model")
        plt.show()

    def save_model(
        self,
        path: str = "model.pkl",
        trained_on_file: str | None = None,
        extra_info: dict | None = None,
    ) -> None:
        """
        Save the ``SpectralClassifier`` object (pickle) and a JSON metadata
        file alongside the model.

        Parameters
        ----------
        path : str
            Path of the ``.pkl`` file to create.
        trained_on_file : str or None
            Name / path of the features dataset used for training.
        extra_info : dict or None
            Additional metadata to merge into the JSON.

        Notes
        -----
        Writes ``<path>`` and ``<path>_meta.json``.  The metadata contains
        versions (Python, numpy, scikit-learn, xgboost), model type,
        target, best hyperparameters, labels, features used, etc.
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

        print(f"  > Model saved to: {path}")
        print(f"  > Metadata saved to: {meta_path}")

    @staticmethod
    def load_model(path: str = "model.pkl") -> "SpectralClassifier":
        """
        Load a ``SpectralClassifier`` saved via ``save_model``.

        Parameters
        ----------
        path : str
            Path of the ``.pkl`` to load.

        Returns
        -------
        SpectralClassifier
            The loaded object (pipeline + attributes).

        Notes
        -----
        If a ``<path>_meta.json`` file is present, its path is printed
        for reference.
        """
        model = joblib.load(path)
        print(f"  > Model loaded from: {path}")
        meta_path = os.path.splitext(path)[0] + "_meta.json"
        if os.path.exists(meta_path):
            print(f"  > Metadata available: {meta_path}")
        return model
