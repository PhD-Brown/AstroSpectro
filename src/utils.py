"""
General-purpose utilities for the AstroSpectro pipeline.

This module provides cross-cutting helper functions used by notebooks and
pipeline scripts, covering project environment setup, file operations,
spectroscopic data sanitization, and model compatibility checks.

Scientific Context
------------------
Several helpers address numerical pitfalls common in spectroscopic data
processing. Inverse-variance arrays from LAMOST FITS files may contain
non-finite or non-positive entries that would trigger warnings or produce
NaN when computing standard deviations. The sanitization functions centralize
robust handling of these edge cases.

Key Components
--------------
- setup_project_env(): Detect project root, build standard paths, update sys.path
- load_env_vars(): Load environment variables from a .env file with prefix filtering
- latest_file(): Retrieve the most recently modified file matching a glob pattern
- sanitize_invvar(): Clean inverse-variance arrays for safe sqrt operations
- make_stddev_uncertainty_from_invvar(): Build an Astropy StdDevUncertainty from invvar
- check_model_compat(): Verify feature consistency between a trained model and a DataFrame

Typical Usage
-------------
>>> paths = setup_project_env()
>>> latest = latest_file(paths["REPORTS_DIR"], "features_*.csv")
>>> env = load_env_vars()  # filters GAIA_* variables by default
>>> print(latest, list(env)[:3])

Notes
-----
- All paths returned by ``setup_project_env`` are **absolute** strings.
- Informational messages are printed by default; disable with ``verbose=False``.
- Directory creation is **optional** and controlled by ``create_missing_dirs``.

Dependencies
------------
astropy.nddata, python-dotenv
"""

from __future__ import annotations

import os
import sys
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from astropy.nddata import StdDevUncertainty
from dotenv import load_dotenv

# Optional NumPy type stubs (no runtime effect)
try:
    import numpy.typing as npt
except Exception:  # pragma: no cover
    npt = None  # type: ignore

__all__ = [
    "setup_project_env",
    "load_env_vars",
    "latest_file",
    "md5sum",
    "sanitize_invvar",
    "make_stddev_uncertainty_from_invvar",
    "check_model_compat",
    "ensure_dir",
    "utc_now_tag",
    "sizeof_fmt",
]

# ---------------------------------------------------------------------
# 1) Project environment
# ---------------------------------------------------------------------


def setup_project_env(
    *,
    create_missing_dirs: bool = True,
    add_to_sys_path: bool = True,
    verbose: bool = True,
) -> Dict[str, str]:
    """
    Detect the project root, build standard directory paths, and optionally update sys.path.

    Walks up from the current working directory until a ``src/`` folder is found,
    then constructs a dictionary of canonical paths used throughout the pipeline.

    Parameters
    ----------
    create_missing_dirs : bool, optional
        If True, create standard directories when they do not exist
        (data/, data/raw/, data/catalog/, data/processed/, data/models/,
        data/reports/, notebooks/, logs/) (default: True).
    add_to_sys_path : bool, optional
        If True, append ``SRC_DIR`` to ``sys.path`` so that pipeline modules
        can be imported directly from notebooks (default: True).
    verbose : bool, optional
        Print informational messages to stdout (default: True).

    Returns
    -------
    dict[str, str]
        Dictionary of **absolute** paths with keys: PROJECT_ROOT, SRC_DIR,
        PIPELINE_DIR, TOOLS_DIR, DATA_DIR, RAW_DATA_DIR, CATALOG_DIR,
        PROCESSED_DIR, MODELS_DIR, REPORTS_DIR, NOTEBOOKS_DIR, LOGS_DIR.

    Raises
    ------
    FileNotFoundError
        If no parent directory containing a ``src/`` folder is found.

    Examples
    --------
    >>> paths = setup_project_env(verbose=False)
    >>> paths["RAW_DATA_DIR"]
    '/home/user/AstroSpectro/data/raw'
    """
    project_root = Path(os.getcwd()).resolve()
    while not (project_root / "src").is_dir():
        parent = project_root.parent
        if parent == project_root:
            raise FileNotFoundError(
                "Cannot locate project root: no 'src' directory found in any parent."
            )
        project_root = parent

    # Build canonical directory paths
    paths: Dict[str, str] = {
        "PROJECT_ROOT": str(project_root),
        "SRC_DIR": str(project_root / "src"),
        "PIPELINE_DIR": str(project_root / "src" / "pipeline"),
        "TOOLS_DIR": str(project_root / "src" / "tools"),
        "DATA_DIR": str(project_root / "data"),
        "RAW_DATA_DIR": str(project_root / "data" / "raw"),
        "CATALOG_DIR": str(project_root / "data" / "catalog"),
        "PROCESSED_DIR": str(project_root / "data" / "processed"),
        "MODELS_DIR": str(project_root / "data" / "models"),
        "REPORTS_DIR": str(project_root / "data" / "reports"),
        "NOTEBOOKS_DIR": str(project_root / "notebooks"),
        "LOGS_DIR": str(project_root / "logs"),
    }

    if create_missing_dirs:
        for key in (
            "DATA_DIR",
            "RAW_DATA_DIR",
            "CATALOG_DIR",
            "PROCESSED_DIR",
            "MODELS_DIR",
            "REPORTS_DIR",
            "NOTEBOOKS_DIR",
            "LOGS_DIR",
        ):
            Path(paths[key]).mkdir(parents=True, exist_ok=True)

    if add_to_sys_path and paths["SRC_DIR"] not in sys.path:
        sys.path.append(paths["SRC_DIR"])

    if verbose:
        print(f"[INFO] Project root detected: {paths['PROJECT_ROOT']}")
        if add_to_sys_path:
            print("[INFO] 'src' directory added to sys.path.")

    return paths


# ---------------------------------------------------------------------
# 2) Environment variables
# ---------------------------------------------------------------------


def load_env_vars(
    env_file_path: str | os.PathLike | None = None,
    *,
    prefix: str = "GAIA_",
) -> Dict[str, str]:
    """
    Load a ``.env`` file and return variables filtered by prefix.

    Parameters
    ----------
    env_file_path : str, os.PathLike, or None, optional
        Explicit path to the ``.env`` file. If None, the file is located
        automatically at ``<PROJECT_ROOT>/.env`` (default: None).
    prefix : str, optional
        Only return variables whose names start with this prefix.
        Use ``prefix=""`` to return all loaded variables (default: ``"GAIA_"``).

    Returns
    -------
    dict[str, str]
        Mapping of environment variable names to their values.

    Raises
    ------
    FileNotFoundError
        If the ``.env`` file does not exist at the resolved path.

    Examples
    --------
    >>> env = load_env_vars(prefix="GAIA_")
    >>> env["GAIA_USER"]
    'my_gaia_username'
    """
    if env_file_path is None:
        # Detect root the same way as setup_project_env()
        root = Path(os.getcwd()).resolve()
        while not (root / "src").is_dir():
            parent = root.parent
            if parent == root:
                raise FileNotFoundError(
                    "Cannot locate project root to find the .env file."
                )
            root = parent
        env_file_path = root / ".env"

    env_path = Path(env_file_path)
    if not env_path.exists():
        raise FileNotFoundError(f".env file not found: {env_path}")

    load_dotenv(env_path)
    print(f"[INFO] Environment variables loaded from {env_path}")

    if prefix:
        return {k: v for k, v in os.environ.items() if k.startswith(prefix)}
    return {k: v for k, v in os.environ.items()}


# ---------------------------------------------------------------------
# 3) File helpers
# ---------------------------------------------------------------------


def latest_file(directory: str | os.PathLike, pattern: str) -> str | None:
    """
    Return the most recently modified file matching a glob pattern.

    Parameters
    ----------
    directory : str or os.PathLike
        Directory to search.
    pattern : str
        Glob pattern to match (e.g., ``'features_*.csv'``).

    Returns
    -------
    str or None
        Absolute path of the most recent matching file, or None if no match.

    Examples
    --------
    >>> latest_file("data/reports", "features_*.csv")
    'data/reports/features_20250101T101500Z.csv'
    """
    p = Path(directory)
    try:
        matches = list(p.glob(pattern))
        if not matches:
            return None
        # Sort by modification time, most recent first
        matches.sort(key=lambda f: f.stat().st_mtime, reverse=True)
        return str(matches[0])
    except Exception:
        # I/O errors should not crash the caller
        return None


def md5sum(path: str | os.PathLike, chunk_size: int = 1 << 20) -> str:
    """
    Compute the MD5 hash of a file.

    Useful for tracing exactly which file was used for training a model.

    Parameters
    ----------
    path : str or os.PathLike
        Path to the file.
    chunk_size : int, optional
        Read block size in bytes (default: 1 MiB).

    Returns
    -------
    str
        Hexadecimal MD5 digest (32 characters).
    """
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_sigma_from_invvar(
    invvar: np.ndarray, *, bad_to_inf: bool = True
) -> np.ndarray:
    """
    Convert an inverse-variance array to robust 1-sigma uncertainties.

    Neutralizes invalid entries (NaN, Inf, non-positive) that would otherwise
    trigger ``sqrt`` warnings in Astropy/specutils.

    Parameters
    ----------
    invvar : np.ndarray
        Inverse-variance per pixel (same shape as the flux array).
    bad_to_inf : bool, optional
        If True (default), invalid pixels receive ``np.inf`` (zero weighting
        in weighted algorithms). Set to False for ``np.nan`` instead.

    Returns
    -------
    sigma : np.ndarray
        1-sigma uncertainties suitable for ``StdDevUncertainty`` (Astropy).

    Notes
    -----
    - For pixels where invvar > 0 and finite: sigma = 1 / sqrt(invvar).
    - For invalid pixels: sigma = +inf (default) or NaN.
    - The output is always ``float64`` and preserves the input shape.

    Examples
    --------
    >>> invvar = np.array([4.0, 0.25, 0.0, np.nan])
    >>> safe_sigma_from_invvar(invvar)
    array([0.5 , 2.  ,  inf,  inf])
    """
    invvar = np.asarray(invvar, dtype=float)

    # Mark non-finite or non-positive entries as invalid
    bad = ~np.isfinite(invvar) | (invvar <= 0.0)
    # Local copy to avoid side effects on the caller's array
    invvar = invvar.copy()
    invvar[bad] = 0.0

    with np.errstate(divide="ignore", invalid="ignore"):
        sigma = np.where(
            invvar > 0.0, 1.0 / np.sqrt(invvar), np.inf if bad_to_inf else np.nan
        )

    return sigma


# ---------------------------------------------------------------------
# 4) Spectroscopic helpers
# ---------------------------------------------------------------------


def sanitize_invvar(
    invvar: "npt.ArrayLike | np.ndarray", min_val: float = 1e-12
) -> np.ndarray:
    """
    Clean an inverse-variance array for safe use in ``sqrt()``.

    Replaces NaN, Inf, and non-positive values with zero, then clips the
    result to ``[min_val, +inf)``.

    Parameters
    ----------
    invvar : array-like
        Input inverse-variance array.
    min_val : float, optional
        Floor value after clipping, preventing strict zero and negatives
        (default: 1e-12).

    Returns
    -------
    np.ndarray
        Cleaned float array, safe for downstream sqrt operations.

    Examples
    --------
    >>> sanitize_invvar(np.array([0.0, -1.0, np.nan, 4.0]))
    array([1.e-12, 1.e-12, 1.e-12, 4.e+00])
    """
    inv = np.asarray(invvar, dtype=float)
    bad = ~np.isfinite(inv) | (inv <= 0)
    if bad.any():
        inv[bad] = 0.0
    return np.clip(inv, min_val, None)


def make_stddev_uncertainty_from_invvar(
    invvar: "npt.ArrayLike | np.ndarray", min_val: float = 1e-12
) -> StdDevUncertainty:
    """
    Build an Astropy ``StdDevUncertainty`` from an inverse-variance array.

    Parameters
    ----------
    invvar : array-like
        Inverse-variance array (sanitized internally via ``sanitize_invvar``).
    min_val : float, optional
        Floor value for ``sanitize_invvar`` (default: 1e-12).

    Returns
    -------
    StdDevUncertainty
        Standard-deviation uncertainty ready for specutils / astropy.

    Examples
    --------
    >>> unc = make_stddev_uncertainty_from_invvar(invvar_array)
    >>> spectrum = Spectrum(flux=flux * u.adu, spectral_axis=wl * u.AA, uncertainty=unc)
    """
    inv = sanitize_invvar(invvar, min_val)
    with np.errstate(invalid="ignore", divide="ignore"):
        sigma = 1.0 / np.sqrt(inv)
    return StdDevUncertainty(sigma)


# ---------------------------------------------------------------------
# 5) Model / feature compatibility
# ---------------------------------------------------------------------


def check_model_compat(clf, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Compare DataFrame columns to the features expected by a trained model.

    Parameters
    ----------
    clf : object
        Trained model exposing a ``feature_names_used`` attribute
        (e.g., ``SpectralClassifier``).
    df : pd.DataFrame
        Feature DataFrame with numeric columns.

    Returns
    -------
    missing : list[str]
        Features expected by the model but absent from ``df``.
    extra : list[str]
        Numeric columns in ``df`` not used by the model.

    Examples
    --------
    >>> missing, extra = check_model_compat(clf, features_df)
    >>> if missing:
    ...     print("Missing columns:", missing)
    """
    expected = set(getattr(clf, "feature_names_used", []))
    present = {c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])}
    missing = sorted(expected - present)
    extra = sorted(present - expected)
    return missing, extra


# ---------------------------------------------------------------------
# 6) Convenience helpers (files, logs, display)
# ---------------------------------------------------------------------


def ensure_dir(path: str | os.PathLike, *, parents: bool = True) -> Path:
    """
    Create a directory if it does not exist (equivalent to ``mkdir -p``).

    Parameters
    ----------
    path : str or os.PathLike
        Directory path to ensure.
    parents : bool, optional
        Create parent directories as needed (default: True).

    Returns
    -------
    Path
        Path object for the directory (guaranteed to exist after the call).
    """
    p = Path(path)
    p.mkdir(parents=parents, exist_ok=True)
    return p


def utc_now_tag(*, with_z: bool = True, ms: bool = False) -> str:
    """
    Generate a UTC timestamp tag for naming files and logs.

    Parameters
    ----------
    with_z : bool, optional
        Append a 'Z' suffix indicating UTC (default: True).
    ms : bool, optional
        Include milliseconds in the tag (default: False).

    Returns
    -------
    str
        Timestamp string, e.g., ``'20250915T152727Z'`` or ``'20250915T152727123Z'``.
    """
    now = datetime.now(timezone.utc)
    if ms:
        s = now.strftime("%Y%m%dT%H%M%S%f")[:-3]  # keep milliseconds
    else:
        s = now.strftime("%Y%m%dT%H%M%S")
    return f"{s}Z" if with_z else s


def sizeof_fmt(num: int | float, suffix: str = "B") -> str:
    """
    Format a byte size as a human-readable string (KiB, MiB, etc.).

    Parameters
    ----------
    num : int or float
        Size in bytes.
    suffix : str, optional
        Unit suffix (default: ``"B"``).

    Returns
    -------
    str
        Formatted size string.

    Examples
    --------
    >>> sizeof_fmt(1536000)
    '1.46 MiB'
    """
    num = float(num)
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
        if abs(num) < 1024.0:
            return f"{num:3.2f} {unit}{suffix}"
        num /= 1024.0
    return f"{num:.2f} Yi{suffix}"
