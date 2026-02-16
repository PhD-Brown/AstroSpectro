"""AstroSpectro -- pipeline orchestrator.

Define the :class:`MasterPipeline` class that orchestrates a complete
processing and training pipeline for astronomical spectra in FITS format.
The workflow comprises batch selection, catalogue construction and
enrichment (optional Gaia cross-match), pre-processing and feature
extraction, spectral classifier training and evaluation, and full
result and artefact logging.

Public methods of :class:`MasterPipeline`:

* :meth:`select_batch` -- select a batch of previously unused spectra.
* :meth:`generate_and_enrich_catalog` -- build the master catalogue and
  optionally enrich it with Gaia.
* :meth:`process_data` -- run the pre-processing pipeline and persist a
  feature CSV.
* :meth:`run_training_session` -- train and evaluate a model while logging
  artefacts (model, reports, figures).
* :meth:`run_full_pipeline` -- chain every step in a single one-shot run.
* :meth:`interactive_training_runner` -- provide an interactive ipywidgets
  interface to configure and launch training runs.

Inputs
------
Root directories for raw data, catalogue, features, models, and reports.
A ready-made feature CSV can also be loaded directly.

Outputs
-------
Feature CSV files, serialised models (with metadata), figures (confusion
matrices, ROC/PR curves, etc.), a session JSON report, and the interface
configuration.

Dependencies
------------
scikit-learn, xgboost, imbalanced-learn, astropy, ipywidgets, pandas,
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

# --- Project imports ---
from tools.dataset_builder import DatasetBuilder
from pipeline.processing import ProcessingPipeline
from pipeline.classifier import SpectralClassifier
from tools.generate_catalog_from_fits import generate_catalog_from_fits
from tools.gaia_crossmatcher import enrich_catalog_with_gaia

# --- Scikit-learn imports ---
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
)

# ================================================================
#  Parameter templates (base_params, grid, distributions)
#  These dicts define default parameters and search grids for each model.
#  They pre-fill the corresponding UI fields when a model is selected.

# ================================================================

# Base presets (starting parameters for each model)
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

# Search grids (GridSearchCV)
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

# Distributions for RandomizedSearchCV
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
# Training dashboard utilities
# -----------------------------------------------------------------------------


def _save_preset(path: str, widgets: dict) -> None:
    """
    Persist widget values to a JSON file.

    Widgets must expose a ``value`` attribute; this function extracts
    those values and serialises them as a JSON dictionary.

    Args:
        path : str
        Path to the output JSON file.
        widgets : dict
        Mapping of widget names to ipywidgets instances.  Only entries
                entries that expose a ``value`` attribute are serialised.

    Side Effects:
        Write a JSON file at *path*.
    """
    payload = {k: w.value for k, w in widgets.items() if hasattr(w, "value")}
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _load_preset(path: str, widgets: dict) -> None:
    """
    Load a preset from a JSON file and update widget values.

    Read a JSON file produced by :func:`_save_preset` and assign the
    stored values to the corresponding widgets.  Keys absent from
    *widgets*, or widgets without a ``value`` attribute, are silently
    skipped.

    Args:
        path : str
        Path to the JSON preset file.
        widgets : dict
        Widget mapping to update; only keys present in this dict are
                modified.

    Side Effects:
        Update the ``value`` property of the matched widgets.
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
    Convert non-serialisable objects to JSON-native types.

    Serve as the ``default`` callback for :func:`json.dump` /
    :func:`json.dumps`.  Handle NumPy scalars and arrays by converting
    them to native Python types.  For third-party objects (e.g.
    them to native Python types.  For third-party objects (e.g.
    scikit-learn estimators), return the class name as a short representation.

    Args:
        o : object
        Object to convert.

    Returns:
        object
            A JSON-serialisable value: numeric, list, class name, or string.

    """
    # numpy -> native Python types
    try:
        import numpy as _np

        if isinstance(o, _np.generic):
            return o.item()
        if hasattr(o, "tolist"):
            return o.tolist()
    except Exception:
        pass
    # sklearn / miscellaneous objects -> short class-name repr
    try:
        name = getattr(o, "__name__", None) or o.__class__.__name__
        return name
    except Exception:
        return str(o)


def _load_runs_table(reports_root: str):
    """
    Load all session reports found under a reports directory.

    Recursively scan *reports_root* for ``session_report_*.json`` files.
    For each report, extract key metrics (accuracy, balanced accuracy,
    macro F1, macro AUC, feature count) and aggregate them into a
    :class:`pandas.DataFrame` sorted by performance.
    :class:`pandas.DataFrame` sorted by performance.


    Args:
        reports_root : str
        Directory containing session JSON reports.


    Returns:
        pandas.DataFrame or None
                Summary DataFrame sorted by descending performance, or ``None``
                if no reports are found.
    """
    rows = []
    for js in Path(reports_root).rglob("session_report_*.json"):
        try:
            meta = json.loads(js.read_text(encoding="utf-8"))
            # Retrieve accuracy and balanced accuracy from varying report structures
            acc = meta.get("accuracy") or (meta.get("test_metrics") or {}).get(
                "accuracy"
            )
            bal = meta.get("balanced_accuracy") or (meta.get("test_metrics") or {}).get(
                "balanced_accuracy"
            )
            # Retrieve the ROC AUC dict (macro/micro + per-class) if present
            roc = meta.get("roc_auc") or {}
            # Retrieve per-class Average Precision if available
            ap = meta.get("avg_precision") or {}
            # Compute macro-AP when possible
            ap_macro = None
            try:
                import numpy as _np

                if ap:
                    ap_macro = float(_np.mean(list(ap.values())))
            except Exception:
                ap_macro = None
            # Extract macro F1 from the classification report
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

        # Sort descending: balanced accuracy then macro F1
        df = pd.DataFrame(rows)
        # Sort while pushing NaN rows to the bottom
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
    """Main orchestrator for the AstroSpectro training pipeline.

    Encapsulate a reproducible workflow from FITS batch selection through
    spectral classifier training and evaluation.  Key steps include
    catalogue construction and enrichment from selected FITS files,
    pre-processing and feature extraction, supervised model training and
    evaluation, and full session artefact logging (figures, reports,
    models).  An interactive ipywidgets interface is provided to drive
    runs from Jupyter or VS Code.

    Attributes
    ----------
    raw_data_dir : str
        Directory containing raw FITS files.
    catalog_dir : str
        Output directory for the intermediate catalogue.
    processed_dir : str
        Output directory for generated feature CSVs.
    models_dir : str
        Output directory for trained, serialised models.
    reports_dir : str
        Session report and figure output directory.
    builder : DatasetBuilder
        Batch selector and log manager.
    current_batch : list[str]
        Relative paths of the current batch.
    master_catalog_df : pandas.DataFrame
        Master catalogue, optionally enriched.
    features_df : pandas.DataFrame
        Latest loaded or generated feature DataFrame.
    master_catalog_path : str
        Path to the temporary master catalogue.
    gaia_catalog_path : str
        Path to the Gaia-enriched catalogue.
    last_features_path : str or None
        Path to the last loaded feature CSV.
    _runs_out : ipywidgets.Output or contextlib.AbstractContextManager
        Output area for the run explorer table.
    """

    def __init__(
        self,
        raw_data_dir: str,
        catalog_dir: str,
        processed_dir: str,
        models_dir: str,
        reports_dir: str,
        use_wandb: bool = False,
    ) -> None:
        """Initialise the orchestrator and prepare internal pipeline state.

        Create working directories if they do not exist, instantiate a
        :class:`DatasetBuilder` pointing at *raw_data_dir* and
        *catalog_dir*, and initialise pipeline attributes (current
        batch, temporary catalogue paths, empty DataFrames).  When
        ipywidgets is available, an output container is created for the
        run explorer.

        Parameters
        ----------
        raw_data_dir : str
            Directory containing raw FITS files (source data).
        catalog_dir : str
            Output directory for the intermediate catalogue.
        processed_dir : str
            Output directory for feature CSVs.
        models_dir : str
            Output directory for serialised models.
        reports_dir : str
            Output directory for JSON reports and figures.

        Raises
        ------
        OSError
            If a required directory cannot be created.
        """
        self.raw_data_dir = raw_data_dir
        self.catalog_dir = catalog_dir
        self.processed_dir = processed_dir
        self.models_dir = models_dir
        self.reports_dir = reports_dir

        self.builder = DatasetBuilder(
            raw_data_dir=self.raw_data_dir, catalog_dir=self.catalog_dir
        )

        # State
        self.current_batch: list[str] = []
        self.master_catalog_df: pd.DataFrame = pd.DataFrame()
        self.features_df: pd.DataFrame = pd.DataFrame()

        # Paths
        self.master_catalog_path = os.path.join(
            self.catalog_dir, "master_catalog_temp.csv"
        )
        self.gaia_catalog_path = os.path.join(
            self.catalog_dir, "master_catalog_gaia.csv"
        )
        self.last_features_path: Optional[str] = None

        # Create directories as needed
        for path in [
            self.catalog_dir,
            self.processed_dir,
            self.models_dir,
            self.reports_dir,
        ]:
            os.makedirs(path, exist_ok=True)

        # Output container for the run explorer (prevent stacking) and last run timestamp
        # Use ipywidgets.Output if available; fall back to nullcontext outside Jupyter
        if W is not None:
            self._runs_out = W.Output(
                layout=W.Layout(
                    border="1px solid #333", max_height="360px", overflow="auto"
                )
            )
        else:
            # Fall-back: no-op context manager
            from contextlib import nullcontext

            self._runs_out = nullcontext()
        self._last_run_ts: Optional[str] = None
        self.use_wandb = use_wandb

    # --------------------- Public API (Notebook) ---------------------

    def select_batch(
        self, batch_size: int = 500, strategy: str = "random"
    ) -> list[str]:
        """
        Select a batch of spectra and store it.

        Query the associated :class:`DatasetBuilder` for a set of FITS
        paths relative to the raw-data directory.  Spectra already
        marked as trained in ``trained_spectra.csv`` are excluded
        automatically to avoid duplicates across sessions.
        les doublons entre sessions.

        Args:
            batch_size : int, optional
            Number of spectra in the batch.
            strategy : str, optional
                Selection strategy (e.g. ``"random"``).  See
                    :class:`DatasetBuilder` for supported values.

        Returns:
            list[str]: Liste des chemins FITS relatifs qui constituent le lot
            courant.

        Side Effects:
            Update :attr:`current_batch` with the selected paths.
        """
        print("\n=== STEP 1: SELECT NEW BATCH ===")
        self.current_batch = self.builder.get_new_training_batch(
            batch_size=batch_size, strategy=strategy
        )
        return self.current_batch

    def generate_and_enrich_catalog(
        self, enrich_gaia: bool = False, **gaia_kwargs: Any
    ) -> None:
        """Build the master catalogue from the current batch and optionally enrich it.

        Read the FITS files of the selected batch, generate an intermediate
        master catalogue, and, if *enrich_gaia* is True, cross-match with
        the Gaia catalogue using the supplied parameters.

        Parameters
        ----------
        enrich_gaia : bool, optional
            Enable Gaia enrichment.
        **gaia_kwargs
            Additional parameters forwarded to
            :func:`enrich_catalog_with_gaia`.

        Side Effects
        ------------
        Update :attr:`master_catalog_df`.  Write
        ``master_catalog_temp.csv`` or ``master_catalog_gaia.csv`` in
        *catalog_dir* depending on the Gaia flag.
        """
        print("\n=== STEP 2: CATALOGUE GENERATION AND ENRICHMENT ===")
        if not self.current_batch:
            print("  > Error: No batch selected. Run `select_batch` first.")
            return

        full_paths = [
            os.path.join(self.raw_data_dir, path) for path in self.current_batch
        ]
        local_df = generate_catalog_from_fits(
            full_paths, self.master_catalog_path, return_df=True
        )
        print(f"  > Local catalogue of {len(local_df)} spectra created.")

        if enrich_gaia:
            enriched_df, stats = enrich_catalog_with_gaia(
                input_catalog_df=local_df,
                output_catalog_path=self.gaia_catalog_path,
                overwrite=True,
                **gaia_kwargs,
            )
            self.master_catalog_df = enriched_df
            print(
                f"  > Gaia: {stats.get('matched', 0)}/{stats.get('total', 0)} objects matched."
            )
        else:
            self.master_catalog_df = local_df

    def process_data(self) -> Optional[pd.DataFrame]:
        """
        Run the pre-processing pipeline and generate a feature CSV.

        Build a :class:`ProcessingPipeline` from the current catalogue,
        execute cleaning, normalisation, and feature extraction, then
        persist the resulting DataFrame as ``features_<timestamp>.csv``
        under *processed_dir*.  Derived target columns are added via
        :meth:`_ensure_derived_targets` and the final DataFrame is
        stored in :attr:`features_df`.
        :attr:`features_df`.

        Returns:
            pandas.DataFrame or None
            Generated feature DataFrame, or ``None`` if the catalogue is empty.
            ``None`` si le catalogue est vide.

        Side Effects:
            Update :attr:`features_df` and :attr:`last_features_path`.
            Write a ``features_*.csv`` file in *processed_dir*.
        """
        print("\n=== STEP 3: DATA PROCESSING AND FEATURE EXTRACTION ===")
        if self.master_catalog_df.empty:
            print(
                "  > Error: Catalogue is empty. Run `generate_and_enrich_catalog` first."
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
                f"\n  > Feature dataset saved to: {os.path.basename(self.last_features_path)}"
            )
            return self.features_df

        return None

    # --- Existing feature utilities ---------------------------------

    def _list_feature_files(self, limit: int = 50) -> list[str]:
        """
        Return a sorted list of available feature CSV files.

        Args:
            limit : int, optional
                Maximum number of files to return.  Sorted by
                    descending modification time.

        Returns:
            list[str]
            Paths of the most recent ``features_*.csv`` files.
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
        """Load an existing feature CSV and reconstruct derived targets.

        If *use_last* is True, locate the most recent
        ``features_*.csv`` in *processed_dir* and load it.  Otherwise
        load the file specified by *path*.  After loading,
        :meth:`_ensure_derived_targets` guarantees the presence of
        derived target columns.

        Parameters
        ----------
        path : str or None, optional
            Explicit CSV path.  Ignored when *use_last* is True.
        use_last : bool, optional
            If True, automatically load the latest ``features_*.csv``
            from *processed_dir*.

        Returns
        -------
        pandas.DataFrame or None
            Loaded feature DataFrame, or ``None`` on failure.

        Side Effects
        ------------
        Update :attr:`features_df` and :attr:`last_features_path`.
        Print a summary of available categorical columns.
        """
        try:
            if use_last:
                cand = self._list_feature_files(limit=1)
                if not cand:
                    print(
                        "No 'features_*.csv' file found in:",
                        self.processed_dir,
                    )
                    return None
                path = cand[0]
            if not path:
                print("Specify a CSV path or pass use_last=True.")
                return None
            df = pd.read_csv(path)
            df = self._ensure_derived_targets(df)

            # Cast targets to categorical (prevent dtype surprises)
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
                f"Features loaded from: {path}  ({len(df):,} rows, {df.select_dtypes(include=['number']).shape[1]} numeric features)"
            )
            # Hint about available target columns
            cat_cols = [
                c for c in df.columns if str(df[c].dtype) in ("category", "object")
            ]
            if cat_cols:
                print(
                    "Categorical columns (potential targets):",
                    ", ".join(cat_cols[:12]),
                    "...",
                )
            return df
        except Exception as e:
            print(f"(error) Unable to load {path}: {e}")
            return None

    def _ensure_derived_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create standardised derived targets if they are missing.

        Add up to three derived target columns:

        * ``main_class`` -- spectral letter (O, B, A, F, G, K, M, ...)
          extracted from ``subclass`` or ``class``.
        * ``sub_class_top25`` -- the 25 most frequent sub-classes in
          ``subclass``; all others are grouped as ``"Other"``.
        * ``sub_class_bins`` -- combination of the spectral letter with a
          numeric bin (``"0-4"`` or ``"5-9"``) derived from the sub-class
          digit.

        Parameters
        ----------
        df : pandas.DataFrame
            Feature DataFrame containing ``class`` and/or ``subclass``.

        Returns:
            pandas.DataFrame
            Copy augmented with the derived columns.
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
                    )  # first spectral-type letter
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

        # ---- sub_class_bins (letter + numeric bin)
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
                # fall back when `subclass` is absent: single bin per letter
                out["sub_class_bins"] = pd.Categorical(
                    out["main_class"].astype(str).str.upper() + "_0-9"
                )

        # Harmonise as categorical dtype
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
        """Return a pandas Styler highlighting the latest and best runs.

        Format numeric metric columns to three decimal places and apply
        background colours to the latest run row, the best-performing
        row, and the top metric cell.

        Parameters
        ----------
        df : pandas.DataFrame
            Runs summary DataFrame.
        last_ts : str or None
            Timestamp of the latest run row to highlight.
        best_ts : str or None
            Timestamp of the best-performing row.
        metric_col : str
            Column used to determine the best row.

        Returns
        -------
        pandas.io.formats.style.Styler
            Ready-to-display styled DataFrame.
        """
        # colours
        c_last = "#fff6bf"  # pale yellow = latest run
        c_best = "#d6f5d6"  # pale green  = best row
        c_cell = "#a3e6a3"  # stronger green = best metric cell

        # locate the best-row index according to metric_col
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
                # en plus, on met le metric en vert plus soutenu si c'est la best row
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
        Reload the runs table in a dedicated output without stacking.
        Optionally highlight the row matching *highlight_ts* or the last known run.

        Args:
            highlight_ts : str, optional
            Timestamp to highlight explicitly.

        Side effects:
            Update the Output widget ``self._runs_out`` with the new table.
        """
        df = _load_runs_table(self.reports_dir)
        with self._runs_out:
            clear_output(wait=True)
            if df is None or len(df) == 0:
                display(HTML("<i>No runs found</i>"))
                return

            # 1) requested or memorised latest run
            last_ts = highlight_ts or getattr(self, "_last_run_ts", None)

            # 2) FALLBACK: if unknown/absent, pick the most recent by timestamp
            try:
                ts_set = set(df["ts"].astype(str))
                if last_ts is None or str(last_ts) not in ts_set:
                    last_ts = max(ts_set)  # ex: "20250827T162302Z"
            except Exception:
                last_ts = None

            # 3) best row (bal_acc or f1_macro)
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
        """Combine exclusions from a text field and a multi-select widget.

        Parameters
        ----------
        txt : str
            Comma-separated exclusion string entered manually.
        selected
            Values chosen in the interactive multi-select widget.

        Returns
        -------
        list[str]
            Sorted list of classes to exclude from training.
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
        """Filter a DataFrame by removing rows whose class is excluded.

        Parameters
        ----------
        df : pandas.DataFrame
            Source DataFrame.
        target_col : str
            Name of the target class column.
        excluded : list[str]
            Classes to exclude.

        Returns
        -------
        tuple[pandas.DataFrame, int]
            Filtered copy of the DataFrame and the number of dropped rows.

        Raises
        ------
        ValueError
            If filtering leaves fewer than two distinct classes.
        """
        if not excluded:
            return df, 0
        before = len(df)
        out = df[~df[target_col].astype(str).isin(excluded)].copy()
        removed = before - len(out)
        ncls = out[target_col].nunique()
        if ncls < 2:
            raise ValueError(
                f"Filtering leaves {ncls} class(es) — at least 2 are required."
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
        # ---- advanced options ----
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
        use_wandb: bool = False,
        # Job count for the estimator (XGB/RF) and XGB tree method
        **kwargs: Any,
    ) -> Optional[SpectralClassifier]:
        """
        Train and evaluate a spectral classifier while logging artefacts.

        Core training method.  Prepare train/test data (with optional
        group-based splitting), apply feature selection, instantiate and
        train the chosen model, optionally run hyper-parameter search and
        probability calibration, evaluate the model, and persist all
        artefacts (confusion matrix, ROC/PR curves, importances, JSON
        report, serialised model, etc.).





        Parameters
        ----------
        model_type : str
            Model identifier (e.g. ``"XGBoost"``, ``"RandomForest"``).
        features_df : pandas.DataFrame or None
            Feature DataFrame; defaults to ``self.features_df``.
        prediction_target : str
            Target column (``"main_class"``, ``"sub_class_top25"``, etc.).
        search : str or None
            Hyper-parameter search type (``"grid"`` or ``"random"``).
        cv_folds : int
            Cross-validation fold count.
        scoring : str
            Metric optimised by the search.
        save_and_log : bool
            Persist the model, report and figures.
        n_estimators : int
            Tree / iteration count for the base estimator.
        test_size : float
            Test-set proportion.
        use_feature_selection : bool or None
            Force feature selection on or off.
        selector_model : str or None
            FS estimator (``"xgb"`` or ``"rf"``).
        selector_threshold : str or None
            FS threshold (e.g. ``"median"``).
        early_stopping : bool
            Enable early stopping.
        early_stopping_rounds : int
            Patience (iterations without improvement).
        val_size : float
            Validation proportion for early stopping.
        param_grid : dict or None
            Grid for ``GridSearchCV``.
        param_distributions : dict or None
            Distributions for ``RandomizedSearchCV``.
        n_iter : int
            Iterations for ``RandomizedSearchCV``.
        random_state : int
            Reproducibility seed.
        save_confusion_png : bool
            Persist confusion matrix as PNG.
        base_params : dict or None
            Base estimator parameters.
        param_overrides : dict or None
            Extra parameters forced on the estimator.
        use_groups : bool
            Enable group-based splitting.
        group_col : str or None
            Column containing group labels.
        use_balanced_weights : bool
            Compute class-balancing sample weights.
        class_weight_mode : str or None
            Weight method (``"inv_freq"`` or ``None``).
        class_weight_alpha : float
            Exponent for inverse-frequency weights.
        weight_col : str or None
            Column containing custom sample weights.
        weight_norm : str
            Weight normalisation method.
        weight_gamma : float
            Exponent for the weight distribution.
        calibrate_probs : bool
            Apply post-hoc probability calibration.
        calibration_method : str
            Calibration method (``"sigmoid"`` or ``"isotonic"``).
        calibrate_cv : int
            Folds for calibration.
        calibrate_holdout_size : float
            Holdout proportion for calibration.
        save_curves_roc_pr : bool
            Persist ROC and PR curves.
        save_calibration : bool
            Persist calibration curve.
        save_feature_importance : bool
            Persist feature importance diagram.
        export_test_predictions : bool
            Export test-set predictions to CSV.
        cm_normalized : bool
            Row-normalise the confusion matrix.
        imputer_strategy : str or None
            Imputation strategy overriding the instance default.
        knn_imputer_k : int
            Neighbours for ``KNNImputer``.
        scaler_type : str or None
            Scaler type.
        selector_method : str or None
            Selection method (e.g. ``"rfecv"``).
        selector_n_estimators : int
            Trees / iterations for the selector.
        mi_top_k : int or None
            Top-K features by mutual information.
        sampler : str or None
            Oversampling method (``"smote"``, ``"adasyn"``, etc.).
        var_threshold : float or None
            Variance threshold for column dropping.
        corr_threshold : float or None
            Correlation threshold for column dropping.
        use_pca : bool or None
            Enable PCA dimensionality reduction.
        pca_components : float, int, or None
            PCA components to keep.
        tune_thresholds : bool or None
            Enable per-class threshold tuning.
        threshold_metric : str or None
            Metric for threshold optimisation.
        exp_name : str or None
            Human-readable experiment name.
        notes : str
            Free-text notes.
        fi_n_repeats : int
            Permutation importance repeat count.
        repeated_cv : bool
            Enable repeated cross-validation.
        cv_repeats : int
            CV repetition count.
        n_jobs : int
            Parallelism level.
        xgb_tree_method : str or None
            XGBoost tree method.
        excluded_classes : list[str] or None
            Classes to exclude before training.

        Returns
        -------
        SpectralClassifier or None
            Trained classifier, or ``None`` if training could not proceed.

        Raises
        ------
        ValueError
            If group splitting is enabled without a valid *group_col*, or
            if class filtering leaves fewer than two classes.
        RuntimeError
            If a critical error occurs during training or logging.

        See Also
        --------
        :meth:`_log_and_report` -- artefact generation.
        """
        if getattr(self, "features_df", None) is None or self.features_df.empty:
            print("ERROR: Feature DataFrame is empty. Run `process_data` first.")
            return

        if use_groups:
            if not group_col or group_col not in self.features_df.columns:
                print(
                    f"ERROR: Group split enabled but column missing: {group_col!r}. "
                    "Disable 'Group split' or specify a valid column."
                )
                return

        df_for_training = self.features_df
        excluded = exclude_classes or []
        if excluded:
            try:
                df_for_training, removed = self._apply_class_filter(
                    df_for_training, prediction_target, excluded
                )
                # Record what happened for the report
                self._last_class_filter = {
                    "mode": "exclude",
                    "classes": list(excluded),
                    "rows_removed": int(removed),
                    "n_classes_after": int(
                        df_for_training[prediction_target].nunique()
                    ),
                }
                print(
                    f"[Class filter] exclusions={excluded}  (rows removed: {removed})"
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
                "\n--- SESSION COMPLETE WITHOUT TRAINING (insufficient valid data) ---"
            )
            return

        trained_clf, feature_cols_before_fs, X_all, y_all, groups_all = result

        # Feature-selection message
        if use_feature_selection and "fs" in trained_clf.model_pipeline.named_steps:
            kept = len(getattr(trained_clf, "selected_features_", []) or [])
            total = len(feature_cols_before_fs)
            msg = (
                f"[Feature selection] enabled — {kept}/{total} features kept."
                if kept
                else "[Feature selection] enabled."
            )
            print("\n" + msg)
        else:
            print(
                f"\n[Feature selection] not used. {len(feature_cols_before_fs)} features total."
            )

        # Sauvegarde + rapport
        processed_files = []
        if "file_path" in self.features_df.columns:
            processed_files = (
                self.features_df["file_path"].dropna().astype(str).unique().tolist()
            )

        if save_and_log:
            # If group splitting was requested, validate the column
            if use_groups and (
                not group_col or group_col not in self.features_df.columns
            ):
                print(f"ERROR: Group split enabled but column missing: {group_col!r}.")
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
                    use_wandb=use_wandb,
                )

                # Record the latest run timestamp and refresh the runs table
                if run_dir:
                    try:
                        ts = Path(run_dir).name
                        self._last_run_ts = ts
                        self._refresh_runs_table(highlight_ts=ts)
                    except Exception:
                        pass

                # Update trained_spectra.csv via DatasetBuilder
                try:
                    # 1) Prefer paths from features (file_path column)
                    to_log = list(processed_files) if processed_files else []

                    # 2) Safety: re-base leaked absolute paths relative to raw_data_dir
                    rel: list[str] = []
                    for p in to_log:
                        q = str(p)
                        if os.path.isabs(q):
                            try:
                                q = os.path.relpath(q, start=self.raw_data_dir)
                            except Exception:
                                pass
                        rel.append(q.replace("\\", "/"))

                    # 3) Fall back to the current batch (select_batch) if features yield nothing
                    if not rel and getattr(self, "current_batch", None):
                        rel = [str(p).replace("\\", "/") for p in self.current_batch]

                    if rel:
                        self.builder.update_trained_log(rel)  # dedup & append safe
                        print(f"  > trained_spectra.csv updated (+{len(rel)} entries).")
                    else:
                        print("  > (info) Nothing to log (empty list).")
                except Exception as e:
                    print(f"(warn) Unable to update trained_spectra.csv: {e}")

            except Exception as e:
                print(f"(warning) Report generation failed: {e}")
        else:
            print("\n--- EXPERIMENTAL SESSION COMPLETE (not saved) ---")

        return trained_clf

    # --------------------- High-level entries ---------------------

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
        Run the entire pipeline end-to-end, from batch selection to a trained model.

        Combine :meth:`select_batch`,
        :meth:`generate_and_enrich_catalog`, :meth:`process_data` et
        :meth:`run_training_session` with the supplied parameters.
        Convenient for a single complete run without the interactive UI.
        interactive.

        Args:
            batch_size : int, optional
            Number of spectra to process.
            model_type : str, optional
            Model to train.
            n_estimators : int, optional
            Estimator count for the final model.
            prediction_target (str, optional): Nom de la colonne cible.
            save_and_log : bool, optional
            Enable model and artefact persistence.
            enrich_gaia (bool, optional): Si ``True``, enrichit le catalogue avec Gaia.
            **gaia_kwargs
            Additional parameters for Gaia enrichment.

        Side Effects:
            Call the other pipeline methods and update internal state.
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

        # Log the trained spectra in trained_spectra.csv
        # When save_and_log is enabled and a batch was selected,
        # record the relative file list via DatasetBuilder.
        if save_and_log and getattr(self, "current_batch", None):
            try:
                # Use update_trained_log to only log new spectra
                self.builder.update_trained_log(self.current_batch)
            except Exception as e:
                print(f"(warn) Unable to update trained_spectra.csv: {e}")

    def interactive_training_runner(self) -> None:
        """
        Display the interactive training interface in Jupyter or VS Code.

        The UI relies on ipywidgets and is organised in tabs covering

        feature source and data splitting, model type and parameters,
        feature selection and hyper-parameter search, class weights and
        calibration, and output artefacts.
        A "Launch training" button invokes :meth:`run_training_session`
        with the current settings, and a run explorer allows browsing
        previous sessions.

        Notes:
            The current configuration can be saved to or reloaded from a
            JSON file via dedicated buttons.  Actual execution is delegated
            to the internal ``_on_run`` callback.
        """
        import json as _json
        import ipywidgets as _W

        # Create a global output area so helpers can use it
        out = _W.Output()

        # --- internal helpers ---
        def _parse_json(txt_widget):
            """Parse JSON widget content; return None on failure."""
            try:
                s = (txt_widget.value or "").strip()
                return _json.loads(s) if s else None
            except Exception as e:
                # Write the warning in the shared output
                with out:
                    print(f"(warn) JSON invalide pour '{txt_widget.description}': {e}")
                return None

        def _grid_size(d):
            """Return an estimate of the number of combinations in a param_grid."""
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
            description="Target",
        )
        # Force the option list to these three targets; keep expected names even if the DataFrame changes
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
            placeholder="group column name",
            description="Group col.",
            layout=_W.Layout(width="300px"),
        )

        # --- Bloc "Source des features" -------------------------------------
        feat_src = _W.Dropdown(
            options=[
                ("In memory", "mem"),
                ("Latest CSV", "last"),
                ("Choose CSV", "pick"),
            ],
            value="mem",
            description="Features",
        )
        feat_files = _W.Dropdown(
            options=[(Path(p).name, p) for p in self._list_feature_files(50)],
            description="file",
            layout=_W.Layout(width="420px"),
        )
        feat_refresh = _W.Button(description="🔄", layout=_W.Layout(width="42px"))
        feat_load = _W.Button(description="Load", icon="upload")
        feat_info = _W.HTML(value="")

        # --- Exclure des classes du target ---
        exclude_txt = _W.Text(
            value="",
            placeholder="ex: B,D,W",
            description="Exclude (CSV)",
            layout=_W.Layout(width="260px"),
        )
        exclude_dd = _W.SelectMultiple(
            options=[],
            value=(),
            description="Exclude (list)",
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
                    feat_info.value = "<i>self.features_df is empty.</i>"
                    _refresh_exclude_choices()
                else:
                    n = len(self.features_df)
                    d = self.features_df.select_dtypes(include=["number"]).shape[1]
                    feat_info.value = f"<b>OK</b> — {n:,} rows, {d} numeric features"
            elif feat_src.value == "last":
                df = self.load_features_from_csv(use_last=True)
                feat_info.value = (
                    "<b>Latest CSV loaded.</b>"
                    if df is not None
                    else "<b>Loading failed.</b>"
                )
            else:
                if not feat_files.value:
                    feat_info.value = "<i>Choose a file.</i>"
                else:
                    df = self.load_features_from_csv(path=feat_files.value)
                    feat_info.value = (
                        "<b>CSV loaded.</b>"
                        if df is not None
                        else "<b>Loading failed.</b>"
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

        # Auto-load the latest CSV on startup
        feat_src.value = "last"
        _do_load()

        # ==== Tab 2: Model & Pre-processing ====
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
            description="Model",
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
        # n_jobs for parallelism
        n_jobs = _W.IntSlider(value=-1, min=-1, max=16, step=1, description="n_jobs")

        # Collinearity / variance filters + PCA
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

        # SVM requires a scaler; adjust if necessary and disable n_estim for SVM
        def _toggle_svm(change=None):
            # Auto-set scaler for SVM
            if model.value == "SVM" and scaler.value == "none":
                scaler.value = "standard"
            # Disable n_estimators for SVM
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

        # ==== Tab 3: Feature Selection ====
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
            description="FS method",
            options=[("SelectFromModel", "sfrommodel"), ("RFECV", "rfecv")],
            value="sfrommodel",
            tooltip="Choose RFECV for recursive ranking; otherwise SelectFromModel.",
        )
        mi_topk = _W.IntText(value=0, description="MI top-K (0=off)")
        tab_fs = _W.VBox([fs_enable, fs_kind, fs_method, fs_thresh, fs_n, mi_topk])

        # ==== Tab 4: Search ====
        search = _W.Dropdown(
            options=[
                ("None", None),
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

        # Pre-fill base_params, param_grid, and param_distributions when the model changes
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
            # Early stopping enabled only when no search is active
            es.disabled = search.value is not None
            es_rounds.disabled = es.disabled
            # Affiche/cacher les champs selon le mode de recherche
            param_grid.layout.display = "block" if search.value == "grid" else "none"
            param_dists.layout.display = "block" if search.value == "random" else "none"
            n_iter.layout.display = "block" if search.value == "random" else "none"
            # Estimation de la taille de la grille
            if search.value == "grid":
                n = _grid_size(_parse_json(param_grid))
                grid_hint.value = f"<i>Estimated grid size: {n}</i>" if n else ""
            else:
                grid_hint.value = ""

        search.observe(_toggle_search, "value")
        _toggle_search()

        def _on_grid_change(_):
            if search.value == "grid":
                n = _grid_size(_parse_json(param_grid))
                grid_hint.value = f"<i>Estimated grid size: {n}</i>" if n else ""

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

        # ==== Tab 5: Weights & Calibration ====
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
            description="Per-class threshold tuning", value=False
        )
        threshold_metric = _W.Dropdown(
            description="Threshold metric",
            options=["f1_macro", "balanced_accuracy"],
            value="f1_macro",
        )

        def _svm_hint(_=None):
            hint.value = (
                "<i>Calibration will wrap SVM in a CalibratedClassifierCV.</i>"
                if (model.value == "SVM" and calibrate.value)
                else ""
            )

        model.observe(_svm_hint, "value")
        calibrate.observe(_svm_hint, "value")
        _svm_hint()
        tab_weight = _W.VBox(
            [
                _W.HTML("<b>Weights</b>"),
                _W.HBox([balanced, cw_mode, cw_alpha]),
                _W.HBox([wt_col, wt_norm]),
                _W.HTML("<b>Calibration</b>"),
                _W.HBox([calibrate, calib_method, calib_holdout, calib_cv]),
                _W.HBox([tune_thresholds, threshold_metric]),
                hint,
            ]
        )

        # ==== Tab 6: Outputs ====
        save_log = _W.Checkbox(value=True, description="Save & Log")
        save_cm = _W.Checkbox(value=False, description="save_confusion_png")
        norm_cm = _W.Checkbox(value=False, description="normalized")
        save_rocpr = _W.Checkbox(value=True, description="save_curves_roc_pr")
        save_calib = _W.Checkbox(value=False, description="save_calibration")
        save_feat = _W.Checkbox(value=True, description="save_feature_importance")
        export_pred = _W.Checkbox(value=False, description="export_test_predictions")
        use_wandb = _W.Checkbox(value=False, description="Log to W&B")
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
                use_wandb,
            ]
        )
        # --- Preset configuration and experiment name ----------------
        preset_path = _W.Text(
            value=str(Path(self.reports_dir) / "preset.json"), description="Preset"
        )
        btn_save = _W.Button(description="Save", icon="save")
        btn_load = _W.Button(description="Load", icon="upload")

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
                # Model & Prep tab
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
            description="Launch training",
            button_style="success",
            icon="play",
            layout=_W.Layout(width="240px"),
        )
        summary = _W.Textarea(
            value="",
            description="Summary",
            layout=_W.Layout(width="100%", height="120px"),
        )
        tab_run = _W.VBox([tab_presets, run_btn, summary, out])

        # --- Hyper-parameter templates ---
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
            description="Apply",
            icon="wand-magic-sparkles",
            button_style="",
            layout=_W.Layout(width="120px"),
        )

        # --- Run notes (stored in the JSON report) ---
        notes_widget = _W.Textarea(
            value="",
            placeholder="Run notes (ideas, data-prep, etc.)",
            description="Notes:",
            layout=_W.Layout(width="520px", height="60px"),
        )

        # --- Experiment queue (batch) ---
        add_to_batch_btn = _W.Button(
            description="Add to batch", icon="plus", layout=_W.Layout(width="160px")
        )
        run_batch_btn = _W.Button(
            description="Run batch",
            icon="play",
            button_style="success",
            layout=_W.Layout(width="160px"),
        )
        clear_batch_btn = _W.Button(
            description="Clear",
            icon="trash",
            button_style="warning",
            layout=_W.Layout(width="110px"),
        )
        batch_progress = _W.IntProgress(
            value=0, min=0, max=1, description="Batch:", layout=_W.Layout(width="520px")
        )
        exp_name_widget = _W.Text(
            value="",
            placeholder="Experiment name (e.g. lgbm_bins_v2)",
            description="Exp name:",
            layout=_W.Layout(width="320px"),
        )

        batch_store = []  # local store

        # --- Estimated cost / number of fits ---
        fits_label = _W.HTML("<b>Estimated fits:</b> –")

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
        tabs.set_title(1, "Model & Prep")
        tabs.set_title(2, "Feature Sel.")
        tabs.set_title(3, "HP Search")
        tabs.set_title(4, "Weights & Calib.")
        tabs.set_title(5, "Outputs")
        tabs.set_title(6, "Run")

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

        # Estimation "nombre de fits" (et petit temps indicatif)
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
            fits_label.value = f"<b>Estimated fits:</b> {total:,}"

        # observe
        for w in (param_grid, param_dists, cv_folds, n_iter, search):
            w.observe(_estimate_fits, "value")
        _estimate_fits()

        # SVM: auto-enable proba if needed + contextual disabling
        def _model_context_update(change=None):
            # Disable n_estimators for SVM
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

        # Experiment batch (add / read / serial execution)
        def _current_config_json() -> dict:
            """
            Return the current UI configuration.

            Cover target, model, split, HP search, pre-processing / FS /
            weights / calibration settings, and output options.

            Returns:
                JSON-serialisable configuration dictionary.
            """
            import json as _json
            from pathlib import Path

            # Localised helpers
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
                # Parallelism (passed to run_training_session → clf.n_jobs)
                "n_jobs": int(n_jobs.value),
                # Feature selection
                "use_feature_selection": fs_enable.value,
                "selector_model": fs_method.value,
                "selector_threshold": fs_thresh.value,
                "selector_n_estimators": fs_n.value,
                # Selection method (RFECV vs SelectFromModel)
                "selector_method": fs_kind.value,  # <-- wired
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
                "use_wandb": use_wandb.value,
                # Notes
                "notes": notes_widget.value,
                "exp_name": (exp_name_widget.value or None),
                "exclude_classes": _get_excluded(),
            }

            # Count numeric features (informational)
            try:
                cfg["n_features_candidate"] = int(
                    self.features_df.select_dtypes(include=["number"]).shape[1]
                )
            except Exception:
                cfg["n_features_candidate"] = None

            # Target class distribution (if column exists)
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
                        f"(warn) Target column missing from features_df: {tcol!r}. "
                        f"Choose one of the existing categorical columns from the Target menu."
                    )
            # Lightweight config trace
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
            print(f"→ Added to batch ({len(batch_store)} config(s)).")

        def _on_clear_batch(_):
            batch_store.clear()
            batch_progress.max = 1
            batch_progress.value = 0
            print("Batch cleared.")

        def _on_run_batch(_):
            if not batch_store:
                print("Batch empty.")
                return
            print(f"⏱ Lancement du batch ({len(batch_store)} runs)...")
            batch_progress.value = 0
            for i, cfg in enumerate(list(batch_store), start=1):
                batch_progress.value = i
                # Ensure SVM probability is enabled (safety)
                if cfg["model_type"] == "SVM" and (
                    cfg["calibrate_probs"]
                    or cfg["save_curves_roc_pr"]
                    or cfg["export_test_predictions"]
                ):
                    bp = cfg.get("base_params") or {}
                    bp["probability"] = True
                    cfg["base_params"] = bp

                # Direct call to the same API as the RUN button
                self.run_training_session(**cfg)
                # Short sleep to keep the UI responsive
                time.sleep(0.1)
            print("Batch complete.")

        add_to_batch_btn.on_click(_on_add_to_batch)
        clear_batch_btn.on_click(_on_clear_batch)
        run_batch_btn.on_click(_on_run_batch)

        # Rendu du tableau des runs : style + liens cliquables
        def _render_runs_table(df):
            if df is None or df.empty:
                display(HTML("<i>No runs found.</i>"))
                return
            df = df.copy()

            # Clickable link to the run directory ('run_dir' column expected)
            def _mk_link(p):
                try:
                    return f'<a href="file:///{Path(p).as_posix()}" target="_blank">{Path(p).name}</a>'
                except Exception:
                    return ""

            if "run_dir" in df.columns:
                df["run"] = df["run_dir"].map(_mk_link)

            # Columns to show first
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

            # Style: highlight best bal_acc
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
            description="Refresh",
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
            description="View run", icon="eye", layout=_W.Layout(width="140px")
        )
        runs_zip_btn = _W.Button(
            description="Zip run",
            icon="file-zipper",
            layout=_W.Layout(width="150px"),
        )
        runs_open_btn = _W.Button(
            description="Open folder",
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
            # A run = a YYYYMMDDTHHMMSSZ-formatted directory
            return sorted([p for p in base.iterdir() if p.is_dir()], reverse=True)

        def _session_json_path(run_dir: Path) -> Path | None:
            try:
                cand = list(run_dir.glob("session_report_*.json"))
                return cand[0] if cand else None
            except Exception:
                return None

        def _load_session_summary(run_dir: Path) -> dict:
            """
            Load a session summary from its directory.

            Args:
                run_dir: Dossier de la session (ex: `reports/2025...Z/`).

            Returns:
                Minimal summary (model, scores, notes, etc.).
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
            """Scan *reports_dir* and rebuild the session dropdown, then refresh the runs table.

            Notes
            -----
            Used by the "Run explorer" tab.
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
            # Update the runs table without stacking
            self._refresh_runs_table()

        def _open_folder(_=None):
            val = runs_dropdown.value
            if not val:
                print("No run selected.")
                return
            p = Path(val)
            print(f"Folder: {p}")
            try:
                if os.name == "nt":
                    os.startfile(str(p))
                elif sys.platform == "darwin":
                    os.system(f'open "{p}"')
                else:
                    os.system(f'xdg-open "{p}"')
            except Exception as e:
                print(f"(Info) Unable to open automatically: {e}")

        def _zip_run(_=None):
            val = runs_dropdown.value
            if not val:
                print("No run selected.")
                return
            run_dir = Path(val)
            zip_path = run_dir.with_suffix("")
            try:
                archive = shutil.make_archive(
                    str(zip_path), "zip", root_dir=str(run_dir)
                )
                print(f"Archive created: {archive}")
            except Exception as e:
                print(f"Zip error: {e}")

        def _show_run(_=None):
            val = runs_dropdown.value
            if not val:
                print("No run selected.")
                return
            run_dir = Path(val)
            info = _load_session_summary(run_dir)

            # Summary banner
            display(
                Markdown(
                    f"### Run `{run_dir.name}` — **{info.get('model','?')}**  \n"
                    f"- Accuracy: **{info.get('acc','?')}** &nbsp;&nbsp; "
                    f"- Balanced acc: **{info.get('bal_acc','?')}** &nbsp;&nbsp; "
                    f"- Macro F1: **{info.get('f1_macro','?')}**  \n"
                    f"- Notes: _{(info.get('notes') or '').strip() or '—'}_  \n"
                    f"- Folder: `{run_dir}`"
                )
            )

            # Display figures (pick standard filenames if present)
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
                print("(No figures found in this run)")

            # Preview prediction CSV if it exists
            preds = list(run_dir.glob("test_predictions_*.csv"))
            if preds:
                try:
                    dfp = pd.read_csv(preds[0])
                    display(Markdown("**Prediction preview (top 20)**"))
                    display(dfp.head(20))
                except Exception as e:
                    print(f"(Info) Unable to read {preds[0].name}: {e}")

        # Events
        runs_refresh_btn.on_click(_refresh_runs)
        runs_view_btn.on_click(_show_run)
        runs_zip_btn.on_click(_zip_run)
        runs_open_btn.on_click(_open_folder)

        # Display in the main layout
        display(_W.HTML("<hr>"))
        display(_W.HTML("<h3>🔎 Run explorer</h3>"))
        display(runs_box)
        # Zone d'affichage unique pour la table des runs
        display(self._runs_out)

        # Initial population: fill dropdown and refresh table
        _refresh_runs()

        def _on_run(_):
            # UI state
            run_btn.disabled = True
            run_btn.button_style = "warning"
            run_btn.icon = "hourglass"
            run_btn.description = "Running..."

            with out:
                clear_output(wait=True)
                print("Starting training...")

            # 0) Verify that features are loaded
            if getattr(self, "features_df", None) is None or self.features_df.empty:
                with out:
                    print(
                        "(error) No feature dataset loaded. "
                        "Use the 'Features' block (Latest CSV or Choose CSV) "
                        "or run steps 1-3 first."
                    )
                # Restore button and stop
                run_btn.disabled = False
                run_btn.button_style = "success"
                run_btn.icon = "play"
                run_btn.description = "Launch training"
                return

            try:
                cfg = _current_config_json()

                # Quick target validation
                tcol = cfg.get("prediction_target")
                if tcol not in self.features_df.columns:
                    possibles = [
                        c
                        for c in self.features_df.columns
                        if str(self.features_df[c].dtype) in ("object", "category")
                    ]
                    with out:
                        print(
                            f"(error) Target column {tcol!r} is missing from features_df."
                        )
                        if possibles:
                            print(
                                "Available categorical columns:",
                                ", ".join(possibles[:12]),
                                "...",
                            )
                    return

                # Brief text summary in the summary area
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

                # Actual launch
                with out:
                    print("Training...")
                self.run_training_session(**cfg)

                # Refresh the runs table to show the new directory immediately
                try:
                    _refresh_runs()
                except Exception:
                    pass

                with out:
                    print("Done.")

            finally:
                # Restore button
                run_btn.disabled = False
                run_btn.button_style = "success"
                run_btn.icon = "play"
                run_btn.description = "Launch training"
                try:
                    # Switch to the Run tab if the user navigated away
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
        use_wandb: bool = False,
    ) -> str | None:
        """Save the model, compute metrics, and generate session artefacts.

        Parameters
        ----------
        clf : SpectralClassifier
            Trained classifier instance.
        feature_cols : list[str] or None
            Feature column names used after selection.
        X : pandas.DataFrame
            Feature matrix for evaluation.
        y : numpy.ndarray
            True labels aligned with *X*.
        processed_files : list[str]
            FITS files processed during the session.
        groups : numpy.ndarray or None
            Group array for cross-validation, or ``None``.
        save_confusion_png : bool
            Save the confusion matrix as a PNG figure.
        save_curves_roc_pr : bool
            Save multi-class ROC and precision--recall curves.
        save_calibration : bool
            Save the calibration curve.
        save_feature_importance : bool
            Save a feature-importance bar chart.
        export_test_predictions : bool
            Export test-set predictions to CSV.
        cm_normalized : bool
            Row-normalise the confusion matrix.
        exp_name : str or None
            Human-readable experiment name for the JSON report.
        notes : str
            Free-text notes for the JSON report.
        fi_n_repeats : int
            Permutation-importance repeat count.
        use_wandb : bool
            Enable Weights & Biases logging for this run.

        Returns
        -------
        str or None
            Path to the created run directory, or ``None`` on error.

        Side Effects
        ------------
        Write to disk the serialised model (``.pkl``), a JSON metadata
        file, a session JSON report, and optionally a predictions CSV.
        """
        from sklearn.metrics import (
            balanced_accuracy_score as sk_balanced_accuracy_score,
        )

        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        model_name = f"spectral_classifier_{clf.model_type.lower()}_{ts}.pkl"
        run_dir = os.path.join(self.reports_dir, ts)
        os.makedirs(run_dir, exist_ok=True)
        model_path = os.path.join(self.models_dir, model_name)

        # 7.1 Save model
        try:
            joblib.dump(clf, model_path)
            print(f"  > Model saved to: {model_path}")
        except Exception as e:
            print(f"  (warning) Model save failed: {e}")
            model_path = None

        # 7.2 Metadata
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
            print(f"  (warning) Metadata write failed: {e}")

        # Copy model and metadata into the run directory for quick access
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
            print(f"  (warning) Failed to copy artefacts into the run directory: {e}")

        # 7.3 Model MD5 hash
        model_hash = "N/A"
        try:
            if os.path.exists(model_path):
                with open(model_path, "rb") as f:
                    model_hash = hashlib.md5(f.read()).hexdigest()
                print(f"  > Model MD5 hash: {model_hash}")
        except Exception as e:
            print(f"  (warning) Unable to compute hash: {e}")

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
            print(f"  (warning) Metric computation failed: {e}")
            report_dict, cm, accuracy = None, None, None

        if getattr(clf, "label_encoder", None) is not None:
            y_te_enc = clf.label_encoder.transform(y_te)
            y_true_for_scores = y_te_enc
        else:
            y_true_for_scores = y_te

        # Ensure both sides have the same dtype
        y_pred_arr = np.asarray(y_pred)

        # Default value in case everything fails
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

        # Retrieve predicted probabilities if available.  Some
        # implementations (e.g. SVM) require `probability=True` at model
        # construction.  On failure, proba remains None.
        proba = None
        try:
            if hasattr(clf.model_pipeline, "predict_proba"):
                proba = clf.model_pipeline.predict_proba(X_te)
            else:
                last_est = clf.model_pipeline[-1]
                if hasattr(last_est, "predict_proba"):
                    proba = last_est.predict_proba(X_te)
        except Exception as e:
            print(f"(warning) Probabilities unavailable: {e}")

        # Save confusion matrix as PNG (after computing cm)
        if save_confusion_png and cm is not None:
            try:
                import matplotlib.pyplot as plt
                import seaborn as sns

                # Optionally row-normalise the matrix
                cm_plot = cm.astype(float)
                fmt = "d"
                if cm_normalized:
                    # Normalise by row sum (avoid division by zero)
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
                plt.xlabel("Predicted")
                plt.ylabel("True label")
                plt.title(f"Confusion Matrix — {clf.model_type}")
                out_png = os.path.join(
                    run_dir,
                    f"confusion_matrix_{clf.model_type.lower()}_{ts}.png",
                )
                fig.tight_layout()
                fig.savefig(out_png, dpi=140)
                plt.close(fig)
                print(f"  > Heatmap saved: {out_png}")
            except Exception as e:
                print(f"  (warning) Heatmap save failed: {e}")

        # === ROC/PR curves and additional metrics ===
        # Run these blocks before JSON report generation so their results
        # can be included in the session_report dict.  The variables
        #  `roc_auc_results`, `avg_precision_results` et
        # are initialised to None and updated when the corresponding
        # option is enabled and probabilities are available.

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

                # Encode labels for label_binarize: values 0..n-1
                # Reuse existing integers; otherwise map each label to its index.

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

                # Per-class ROC
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
                print(f"  > ROC saved: {roc_png}")

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
                print(f"  > PR saved: {pr_png}")

                roc_auc_results = {
                    "micro": float(roc_auc["micro"]),
                    "macro": float(roc_auc["macro"]),
                    **{classes[i]: float(roc_auc[i]) for i in range(n_classes)},
                }
                avg_precision_results = {k: float(v) for k, v in ap.items()}
            except Exception as e:
                print(f"  (warning) ROC/PR curves skipped: {e}")

        # B) Calibration curve & Brier score
        if save_calibration and proba is not None and y_true_for_scores is not None:
            try:
                import matplotlib.pyplot as plt
                from sklearn.calibration import calibration_curve
                from sklearn.metrics import brier_score_loss

                classes = list(clf.class_labels)
                n_classes = len(classes)
                # Encode labels
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
                # Binarise for calibration_curve
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
                plt.xlabel("Predicted probability")
                plt.ylabel("Observed frequency")
                plt.title(f"Calibration — {clf.model_type}")
                plt.legend(fontsize=8, loc="best")
                calib_png = os.path.join(
                    run_dir,
                    f"calibration_{clf.model_type.lower()}_{ts}.png",
                )
                fig.tight_layout()
                fig.savefig(calib_png, dpi=140)
                plt.close(fig)
                print(f"  > Calibration saved: {calib_png}")
                brier_score_results = {k: float(v) for k, v in brier.items()}
            except Exception as e:
                print(f"  (warning) Calibration curve skipped: {e}")

        # C) Feature importances
        if save_feature_importance:
            try:
                import matplotlib.pyplot as plt
                from sklearn.inspection import permutation_importance

                names = getattr(clf, "selected_features_", None) or feature_cols
                # Retrieve the final estimator (last pipeline step)
                estimator = clf.model_pipeline[-1]
                importances = None
                # Case 1: estimator exposes feature_importances_
                if hasattr(estimator, "feature_importances_"):
                    importances = estimator.feature_importances_
                # Case 2: linear SVM → importance ~ mean absolute weight
                elif (
                    clf.model_type == "SVM"
                    and getattr(estimator, "kernel", "rbf") == "linear"
                    and hasattr(estimator, "coef_")
                ):
                    importances = np.abs(estimator.coef_).mean(axis=0)

                # Case 3: fall back to permutation importance
                if importances is None:
                    # Use balanced_accuracy as the default scoring metric
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
                print(f"  > Feature importance saved: {fi_png}")
            except Exception as e:
                print(f"  (warning) Feature importance skipped: {e}")

        # D) Export test predictions
        if export_test_predictions:
            try:
                import pandas as pd

                classes = list(clf.class_labels)
                n_samples = len(y_pred)
                # Build a label->index mapping
                label_map = {lab: idx for idx, lab in enumerate(classes)}
                # Encode predictions for indexing
                if isinstance(y_pred[0], (int, np.integer)):
                    y_pred_enc = np.asarray(y_pred)
                else:
                    y_pred_enc = np.array([label_map.get(lab, -1) for lab in y_pred])
                # Encode y_true for export
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
                    # Top-2 classes by probability
                    top2 = np.argsort(proba, axis=1)[:, -2:][:, ::-1]
                    idx = np.arange(len(y_pred_enc))
                    mask = y_pred_enc >= 0
                    # Probability of the predicted class
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
                print(f"  > Test predictions exported: {csv_path}")
            except Exception as e:
                print(f"  (warning) Prediction export skipped: {e}")

        # E) Early-stopping history for XGBoost
        if clf.model_type == "XGBoost":
            try:
                booster = clf.model_pipeline[-1].get_booster()
                hist = booster.evals_result()
                xgb_hist_json = os.path.join(run_dir, f"xgb_eval_history_{ts}.json")
                with open(xgb_hist_json, "w", encoding="utf-8") as f:
                    json.dump(hist, f, indent=2)
                print(f"  > XGB history saved: {xgb_hist_json}")
            except Exception:
                pass

        # 7.5 JSON report
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
        # Record the class filter in the report if one was applied
        try:
            if getattr(self, "_last_class_filter", None):
                session_report["class_filter"] = self._last_class_filter
        except Exception:
            pass

        # Insert additional metrics (ROC AUC, Average Precision,
        # when available.  Dicts are cast to float for JSON serialisation.
        # Brier score) when available.  Dicts are cast to float for JSON serialisation.
        if roc_auc_results is not None:
            session_report["roc_auc"] = roc_auc_results
        if avg_precision_results is not None:
            session_report["avg_precision"] = avg_precision_results
        if brier_score_results is not None:
            session_report["brier_score"] = brier_score_results

        # Macro AUC for backward compatibility with the legacy comparator
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
            print(f"\nSession report saved to: {report_path}")
        except Exception as e:
            print(f"  (warning) Session report generation failed: {e}")

        # === W&B LOGGING (NON-INVASIVE) ===
        if use_wandb:
            try:
                self._log_to_wandb(
                    session_report=session_report,
                    meta=meta,
                    run_dir=run_dir,
                    ts=ts,
                    model_path=model_path,
                    clf=clf,
                    exp_name=exp_name,
                )
            except Exception as e:
                print(f"⚠️  W&B logging failed (non-critical): {e}")
                print("   Training completed successfully, continuing...")

        # --- CELL SUMMARY DISPLAY ---
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

            print("\n=== RESULTS (test set) ===")
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

                print("\nPer class:")
                for cls_name, row in report_dict.items():
                    if cls_name in ("accuracy", "macro avg", "weighted avg"):
                        continue
                    p = row.get("precision")
                    r = row.get("recall")
                    f1 = row.get("f1-score")
                    if p is not None and r is not None and f1 is not None:
                        print(f"  - {cls_name:<6}  P={p:.2f}  R={r:.2f}  F1={f1:.2f}")
            else:
                print("  (no classification report available)")
        except Exception as e:
            print(f"(warning) Metric display skipped: {e}")

        # --- FEATURE SELECTION RECAP ---
        try:
            if getattr(clf, "selected_features_", None) is not None:
                kept = len(clf.selected_features_)
                # the correct name is "feature_cols" (function parameter)
                total = len(feature_cols) if feature_cols is not None else None
                msg = f"[FS] Features kept: {kept}" + (
                    f"/{total}" if total is not None else ""
                )
                print("\n" + msg)
        except Exception as e:
            print(f"(warning) FS recap skipped: {e}")

        print("\nSEARCH SESSION COMPLETE")
        return report_path

    def _log_to_wandb(
        self,
        session_report: dict,
        meta: dict,
        run_dir: str,
        ts: str,
        model_path: str | None,
        clf: SpectralClassifier,
        exp_name: str | None,
    ) -> None:
        """Log training session to Weights & Biases.

        Reads the in-memory session report and metadata, logs all
        metrics, uploads existing PNG files, and saves the model
        artifact.  This method is NON-INVASIVE: it reuses all
        existing outputs and generates nothing new.

        Parameters
        ----------
        session_report : dict
            Session report dictionary (already saved to disk).
        meta : dict
            Model metadata dictionary.
        run_dir : str
            Path to the run artefact directory.
        ts : str
            Timestamp string for the current run.
        model_path : str or None
            Path to the serialised model file.
        clf : SpectralClassifier
            Trained classifier instance.
        exp_name : str or None
            Experiment name.
        """
        from pipeline.wandb_config import should_use_wandb, init_wandb_run

        if not should_use_wandb(use_wandb=True):
            return

        import wandb

        model_lower = clf.model_type.lower()
        n_selected = len(meta.get("selected_features_") or [])
        n_candidate = max(meta.get("n_candidate_features", 0) or 0, 1)

        # --- Config ---
        run_config = {
            # Model
            "model_type": meta.get("model_type"),
            "prediction_target": meta.get("prediction_target"),
            # Features
            "n_candidate_features": meta.get("n_candidate_features"),
            "n_selected_features": n_selected,
            "feature_selection_ratio": round(n_selected / n_candidate, 4),
            "selected_features": meta.get("selected_features_"),
            # Dataset
            "training_set_size": session_report.get("training_set_size"),
            "test_set_size": session_report.get("test_set_size"),
            "total_spectra": session_report.get("total_spectra_processed"),
            # Classes
            "n_classes": len(session_report.get("class_labels", [])),
            "class_labels": session_report.get("class_labels"),
            # Versions
            "python_version": meta.get("python"),
            "numpy_version": meta.get("numpy"),
            "scikit_learn_version": meta.get("scikit_learn"),
            "xgboost_version": meta.get("xgboost"),
            # Experiment
            "exp_name": meta.get("exp_name", exp_name),
            "timestamp": ts,
            "trained_on_file": meta.get("trained_on_file"),
        }
        best_params = meta.get("best_params_")
        if best_params and isinstance(best_params, dict):
            for key, val in best_params.items():
                run_config[f"hp/{key}"] = val
        class_filter = session_report.get("class_filter")
        if class_filter and isinstance(class_filter, dict):
            run_config["classes_excluded"] = class_filter.get("classes", [])

        # --- Tags ---
        macro_f1 = (
            session_report.get("classification_report", {})
            .get("macro avg", {})
            .get("f1-score", 0)
        )
        tags = [
            clf.model_type,
            meta.get("prediction_target", "main_class"),
            f"acc_{int(session_report.get('accuracy', 0) * 100)}",
            f"f1_{int((macro_f1 or 0) * 100)}",
            f"feat_{n_selected}",
            exp_name if exp_name else "unnamed",
        ]
        tags = [t for t in tags if t and str(t).strip()]

        # --- Init run ---
        run_name = f"{model_lower}-{ts}"
        run = init_wandb_run(name=run_name, config=run_config, tags=tags)

        # --- Notes / description / summary ---
        notes_text = session_report.get("notes", "")
        if notes_text and str(notes_text).strip():
            run.notes = str(notes_text)

        exp_label = exp_name if exp_name else "Unnamed"
        acc_pct = int(session_report.get("accuracy", 0) * 100)
        run.description = (
            f"{exp_label} | {clf.model_type} | {acc_pct}% acc | {n_selected} feat"
        )

        run.summary.update(
            {
                "best_accuracy": session_report.get("accuracy"),
                "best_f1_macro": macro_f1,
                "best_roc_auc": (session_report.get("roc_auc") or {}).get("macro"),
                "n_features_used": n_selected,
            }
        )

        # =============== METRICS ===============
        metrics: dict = {}

        # --- Main ---
        for k in ("accuracy", "balanced_accuracy"):
            if k in session_report:
                metrics[k] = float(session_report[k])

        # --- ROC AUC (global + per-class) ---
        roc = session_report.get("roc_auc")
        if isinstance(roc, dict):
            for key, val in roc.items():
                if isinstance(val, (int, float)):
                    metrics[f"roc_auc/{key}"] = float(val)

        # --- Average Precision (per-class + mean) ---
        avg_prec = session_report.get("avg_precision")
        if isinstance(avg_prec, dict):
            ap_vals = []
            for key, val in avg_prec.items():
                if isinstance(val, (int, float)):
                    metrics[f"avg_precision/{key}"] = float(val)
                    ap_vals.append(float(val))
            if ap_vals:
                metrics["avg_precision/mean"] = float(np.mean(ap_vals))

        # --- Brier Score (per-class + mean) ---
        brier = session_report.get("brier_score")
        if isinstance(brier, dict):
            br_vals = []
            for key, val in brier.items():
                if isinstance(val, (int, float)):
                    metrics[f"brier/{key}"] = float(val)
                    br_vals.append(float(val))
            if br_vals:
                metrics["brier/mean"] = float(np.mean(br_vals))

        # --- Classification Report (F1, Precision, Recall per class + avgs) ---
        cr = session_report.get("classification_report")
        if isinstance(cr, dict):
            for cls_name, row in cr.items():
                if not isinstance(row, dict):
                    continue
                if cls_name == "accuracy":
                    continue
                prefix = cls_name.replace(" ", "_")
                for m in ("f1-score", "precision", "recall", "support"):
                    if m in row:
                        safe_m = m.replace("-", "_")
                        metrics[f"{prefix}/{safe_m}"] = (
                            int(row[m]) if m == "support" else float(row[m])
                        )

        # --- Confusion Matrix derived stats ---
        cm = session_report.get("confusion_matrix")
        labels = session_report.get("class_labels", [])
        if cm and isinstance(cm, list):
            cm_arr = np.array(cm)
            metrics["cm/correct"] = int(np.trace(cm_arr))
            metrics["cm/incorrect"] = int(cm_arr.sum() - np.trace(cm_arr))
            for i, cls in enumerate(labels):
                if i < cm_arr.shape[0]:
                    metrics[f"cm/tp_{cls}"] = int(cm_arr[i, i])
                    metrics[f"cm/fp_{cls}"] = int(cm_arr[:, i].sum() - cm_arr[i, i])
                    metrics[f"cm/fn_{cls}"] = int(cm_arr[i, :].sum() - cm_arr[i, i])

        # --- Dataset / feature counts ---
        metrics["dataset/n_train"] = session_report.get("training_set_size", 0)
        metrics["dataset/n_test"] = session_report.get("test_set_size", 0)
        metrics["dataset/n_spectra"] = session_report.get("total_spectra_processed", 0)
        metrics["features/selected"] = n_selected
        metrics["features/candidate"] = meta.get("n_candidate_features", 0)

        if metrics:
            wandb.log(metrics)
            print(f"  > Logged {len(metrics)} metrics to W&B")

        # =============== IMAGES ===============
        png_map = {
            "confusion_matrix": f"confusion_matrix_{model_lower}_{ts}.png",
            "calibration": f"calibration_{model_lower}_{ts}.png",
            "roc_curves": f"roc_{model_lower}_{ts}.png",
            "pr_curves": f"pr_{model_lower}_{ts}.png",
            "feature_importance": f"feature_importance_{model_lower}_{ts}.png",
        }
        for label, filename in png_map.items():
            img_path = os.path.join(run_dir, filename)
            if os.path.isfile(img_path):
                wandb.log({label: wandb.Image(img_path)})

        # =============== PREDICTIONS TABLE ===============
        try:
            pred_path = os.path.join(
                run_dir, f"test_predictions_{model_lower}_{ts}.csv"
            )
            if os.path.isfile(pred_path):
                df_pred = pd.read_csv(pred_path)
                if len(df_pred) > 1000:
                    df_pred = df_pred.sample(n=1000, random_state=42)
                wandb.log({"predictions_sample": wandb.Table(dataframe=df_pred)})
        except Exception:
            pass

        # =============== CUSTOM CHARTS ===============
        try:
            if isinstance(cr, dict) and labels:
                f1_data = []
                pr_data = []
                for cls in labels:
                    row = cr.get(cls)
                    if isinstance(row, dict):
                        f1_data.append([cls, row.get("f1-score", 0)])
                        pr_data.append(
                            [cls, row.get("precision", 0), row.get("recall", 0)]
                        )
                if f1_data:
                    t = wandb.Table(data=f1_data, columns=["class", "f1_score"])
                    wandb.log(
                        {
                            "f1_by_class": wandb.plot.bar(
                                t, "class", "f1_score", title="F1 Score by Class"
                            )
                        }
                    )
                if pr_data:
                    t = wandb.Table(
                        data=pr_data, columns=["class", "precision", "recall"]
                    )
                    wandb.log(
                        {
                            "precision_vs_recall": wandb.plot.scatter(
                                t,
                                "precision",
                                "recall",
                                title="Precision vs Recall by Class",
                            )
                        }
                    )
        except Exception:
            pass

        # =============== MODEL ARTIFACT ===============
        if model_path and os.path.isfile(model_path):
            artifact = wandb.Artifact(
                name=f"model-{model_lower}-{ts}",
                type="model",
                description=f"{clf.model_type} trained on {ts}",
            )
            artifact.add_file(model_path)
            run.log_artifact(artifact)

        wandb.finish()
        print(f"  > W&B run logged: {run_name}")
