"""
AstroSpectro — Utilitaires généraux
===================================

Ce module regroupe de petites fonctions transversales utilisées par les
notebooks et les scripts de la pipeline :

1) Initialisation de l’environnement du projet depuis un notebook
   (`setup_project_env`) — détection de la racine, ajout du dossier `src`
   au `sys.path`, construction des chemins standards.
2) Chargement des variables d’environnement depuis un fichier `.env`
   (`load_env_vars`), avec filtrage par préfixe (par défaut: *GAIA_*).
3) Aides de fichiers: récupération du fichier le plus récent (`latest_file`)
   et calcul de checksum (`md5sum`).
4) Aides spectroscopiques: sécurisation d’inverse-variances pour des `sqrt`
   sans warning/erreur (`sanitize_invvar`) et fabrication d’une
   `StdDevUncertainty` cohérente (`make_stddev_uncertainty_from_invvar`).
5) Vérification de compatibilité d’un modèle avec un DataFrame de features
   (`check_model_compat`).

Conventions
-----------
- Tous les chemins retournés par `setup_project_env` sont **absolus** (str).
- Les messages d’information sont imprimés par défaut ; on peut les
  désactiver via `verbose=False`.
- Pas d’effet de bord caché : la création des répertoires standards est
  **optionnelle** et contrôlée par `create_missing_dirs`.

Entrées / Sorties principales
-----------------------------
- `setup_project_env(...) -> dict[str, str]`
- `load_env_vars(...) -> dict[str, str]`
- `latest_file(...) -> str | None`
- `md5sum(...) -> str`
- `sanitize_invvar(...) -> np.ndarray`
- `make_stddev_uncertainty_from_invvar(...) -> StdDevUncertainty`
- `check_model_compat(...) -> tuple[list[str], list[str]]`

Exemple minimal
---------------
>>> paths = setup_project_env()
>>> latest = latest_file(paths["REPORTS_DIR"], "features_*.csv")
>>> env = load_env_vars()          # filtre GAIA_* par défaut
>>> print(latest, list(env)[:3])
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

# Pour hints NumPy (facultatif, ne change rien à l’exécution)
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
# 1) Environnement projet
# ---------------------------------------------------------------------


def setup_project_env(
    *,
    create_missing_dirs: bool = True,
    add_to_sys_path: bool = True,
    verbose: bool = True,
) -> Dict[str, str]:
    """
    Détecte la racine du projet, construit les chemins standards et (optionnel)
    ajoute `src/` au `sys.path`.

    Args:
        create_missing_dirs: Si True, crée les dossiers standards s'ils n'existent pas
            (data/, data/raw/, data/catalog/, data/processed/, data/models/,
            data/reports/, notebooks/, logs/).
        add_to_sys_path: Si True, ajoute `SRC_DIR` à `sys.path` pour
            permettre `import ...` depuis les notebooks.
        verbose: Affiche quelques messages d'information.

    Returns:
        Un dictionnaire de chemins **absolus** :
        PROJECT_ROOT, SRC_DIR, PIPELINE_DIR, TOOLS_DIR, DATA_DIR, RAW_DATA_DIR,
        CATALOG_DIR, PROCESSED_DIR, MODELS_DIR, REPORTS_DIR, NOTEBOOKS_DIR, LOGS_DIR.

    Raises:
        FileNotFoundError: si la racine (contenant `src/`) n’est pas trouvée.

    Notes:
        - La recherche part du répertoire de travail courant (`os.getcwd()`),
          puis remonte jusqu’à trouver un dossier `src/`.
    """
    project_root = Path(os.getcwd()).resolve()
    while not (project_root / "src").is_dir():
        parent = project_root.parent
        if parent == project_root:
            raise FileNotFoundError(
                "Impossible de localiser la racine du projet (dossier 'src' introuvable)."
            )
        project_root = parent

    # Construction des chemins standards
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
        print(f"[INFO] Racine du projet détectée : {paths['PROJECT_ROOT']}")
        if add_to_sys_path:
            print("[INFO] Dossier 'src' ajouté au sys.path.")

    return paths


# ---------------------------------------------------------------------
# 2) Variables d’environnement
# ---------------------------------------------------------------------


def load_env_vars(
    env_file_path: str | os.PathLike | None = None,
    *,
    prefix: str = "GAIA_",
) -> Dict[str, str]:
    """
    Charge un fichier `.env` et retourne les variables filtrées par préfixe.

    Args:
        env_file_path: Chemin explicite du `.env`. Si None, on prend
            `<PROJECT_ROOT>/.env` (en remontant automatiquement la racine).
        prefix: Préfixe pour filtrer les variables retournées (ex. 'GAIA_').
            Utiliser `prefix=""` pour retourner toutes les variables du `.env`.

    Returns:
        Dictionnaire {VAR: valeur} pour les variables dont le nom commence
        par `prefix`.

    Raises:
        FileNotFoundError: si le fichier `.env` n’existe pas.
    """
    if env_file_path is None:
        # Détecte la racine comme dans setup_project_env()
        root = Path(os.getcwd()).resolve()
        while not (root / "src").is_dir():
            parent = root.parent
            if parent == root:
                raise FileNotFoundError(
                    "Impossible de localiser la racine du projet pour trouver le fichier .env."
                )
            root = parent
        env_file_path = root / ".env"

    env_path = Path(env_file_path)
    if not env_path.exists():
        raise FileNotFoundError(f"Fichier .env introuvable : {env_path}")

    load_dotenv(env_path)
    print(f"[INFO] Variables d'environnement chargées depuis {env_path}")

    if prefix:
        return {k: v for k, v in os.environ.items() if k.startswith(prefix)}
    return {k: v for k, v in os.environ.items()}


# ---------------------------------------------------------------------
# 3) Aides fichiers
# ---------------------------------------------------------------------


def latest_file(directory: str | os.PathLike, pattern: str) -> str | None:
    """
    Retourne le fichier le plus récent qui matche un *glob pattern*.

    Args:
        directory: Dossier à parcourir.
        pattern: Motif de recherche type `glob` (ex.: `'features_*.csv'`).

    Returns:
        Chemin (str) du fichier le plus récent, ou `None` si aucun match.

    Exemple:
        >>> latest_file("data/reports", "features_*.csv")
        'data/reports/features_20250101T101500Z.csv'
    """
    p = Path(directory)
    try:
        matches = list(p.glob(pattern))
        if not matches:
            return None
        # Tri par date de modification décroissante
        matches.sort(key=lambda f: f.stat().st_mtime, reverse=True)
        return str(matches[0])
    except Exception:
        # Une erreur d'I/O ne doit pas faire planter l'appelant
        return None


def md5sum(path: str | os.PathLike, chunk_size: int = 1 << 20) -> str:
    """
    Calcule le hash MD5 d’un fichier (utile pour tracer exactement
    quel fichier a servi à l’entraînement d’un modèle, etc.).

    Args:
        path: Chemin du fichier.
        chunk_size: Taille des blocs lus (par défaut 1 MiB).

    Returns:
        Empreinte MD5 hexadécimale (32 caractères).
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
    Convertit une carte d'inverse-variance (invvar) en incertitudes 1-sigma
    robustes, en neutralisant les valeurs invalides/contraintes qui déclenchent
    sinon des avertissements ``sqrt`` côté Astropy.

    Notes
    -----
    - Pour tout pixel avec invvar > 0 et fini → sigma = 1 / sqrt(invvar).
    - Pour tout pixel NaN/Inf ou invvar <= 0 → sigma = +inf (par défaut),
      ce qui revient à *ignorer* ces points dans les algos pondérés.
      Mettre ``bad_to_inf=False`` pour renvoyer NaN à la place (rarement utile).
    - Préserve la forme d'entrée et renvoie un tableau ``float64``.

    Parameters
    ----------
    invvar : np.ndarray
        Tableau d'inverse-variance par pixel (mêmes dimensions que le flux).
    bad_to_inf : bool, optional
        Si True (défaut), les points invalides reçoivent ``np.inf`` (pondération nulle).
        Sinon, ils reçoivent ``np.nan``.

    Returns
    -------
    sigma : np.ndarray
        Incertitudes 1-sigma prêtes pour ``StdDevUncertainty`` (Astropy).

    Examples
    --------
    >>> import numpy as np
    >>> from utils import safe_sigma_from_invvar
    >>> invvar = np.array([4.0, 0.25, 0.0, np.nan])
    >>> safe_sigma_from_invvar(invvar)
    array([0.5 , 2.  ,  inf,  inf])
    """
    invvar = np.asarray(invvar, dtype=float)

    # Marque les entrées non finies ou non strictement positives comme "mauvaises"
    bad = ~np.isfinite(invvar) | (invvar <= 0.0)
    # Copie locale pour éviter les surprises si l'appelant réutilise invvar ensuite
    invvar = invvar.copy()
    invvar[bad] = 0.0

    with np.errstate(divide="ignore", invalid="ignore"):
        sigma = np.where(
            invvar > 0.0, 1.0 / np.sqrt(invvar), np.inf if bad_to_inf else np.nan
        )

    return sigma


# ---------------------------------------------------------------------
# 4) Aides spectroscopiques
# ---------------------------------------------------------------------


def sanitize_invvar(
    invvar: "npt.ArrayLike | np.ndarray", min_val: float = 1e-12
) -> np.ndarray:
    """
    Nettoie un tableau d'inverse-variance pour un usage sûr dans `sqrt()`.

    Opérations réalisées :
      - conversion en float,
      - remplacement des NaN/Inf/valeurs <= 0 par 0,
      - clipping final à `[min_val, +inf)`.

    Args:
        invvar: Tableau inverse-variance d'entrée.
        min_val: Valeur minimale après clipping (évite 0 strict et négatifs).

    Returns:
        `np.ndarray` float nettoyé.

    Notes:
        Ce comportement reproduit ce que tu utilises déjà dans le pipeline,
        en centralisant la logique ici.
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
    Construit une `StdDevUncertainty` (écarts-types) à partir d'une inverse-variance.

    Args:
        invvar: Tableau inverse-variance (sera nettoyé via `sanitize_invvar`).
        min_val: Valeur minimale pour `sanitize_invvar`.

    Returns:
        `StdDevUncertainty` prêt à être passé à `specutils`/`astropy`.
    """
    inv = sanitize_invvar(invvar, min_val)
    with np.errstate(invalid="ignore", divide="ignore"):
        sigma = 1.0 / np.sqrt(inv)
    return StdDevUncertainty(sigma)


# ---------------------------------------------------------------------
# 5) Compatibilité modèle / features
# ---------------------------------------------------------------------


def check_model_compat(clf, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Compare les colonnes numériques présentes dans `df` aux features
    attendues par le modèle (attribut `feature_names_used`).

    Args:
        clf: Modèle entraîné possédant l’attribut `feature_names_used`
             (ex.: `SpectralClassifier`).
        df: DataFrame de features (colonnes numériques).

    Returns:
        `(missing, extra)` :
          - `missing`: features attendues par le modèle mais absentes de `df`,
          - `extra`  : colonnes numériques présentes dans `df` mais non
            utilisées par le modèle.

    Exemple:
        >>> missing, extra = check_model_compat(clf, features_df)
        >>> if missing:
        ...     print("Colonnes manquantes:", missing)
    """
    expected = set(getattr(clf, "feature_names_used", []))
    present = {c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])}
    missing = sorted(expected - present)
    extra = sorted(present - expected)
    return missing, extra


# ---------------------------------------------------------------------
# 6) Petits helpers de confort (fichiers, logs, affichage)
# ---------------------------------------------------------------------


def ensure_dir(path: str | os.PathLike, *, parents: bool = True) -> Path:
    """
    Crée un dossier s'il n'existe pas (équivalent sûr à mkdir -p).

    Args:
        path: Chemin du dossier à garantir.
        parents: Crée les parents si nécessaire.

    Returns:
        Path vers le dossier (toujours existant après l'appel).
    """
    p = Path(path)
    p.mkdir(parents=parents, exist_ok=True)
    return p


def utc_now_tag(*, with_z: bool = True, ms: bool = False) -> str:
    """
    Génère un tag horodaté UTC pour nommer fichiers/logs.

    Args:
        with_z: Ajoute le suffixe 'Z' (UTC) à la fin.
        ms: Inclut les millisecondes.

    Returns:
        Chaîne style 'YYYYMMDDTHHMMSSZ' ou 'YYYYMMDDTHHMMSSmmmZ'.
    """
    now = datetime.now(timezone.utc)
    if ms:
        s = now.strftime("%Y%m%dT%H%M%S%f")[:-3]  # garde millisecondes
    else:
        s = now.strftime("%Y%m%dT%H%M%S")
    return f"{s}Z" if with_z else s


def sizeof_fmt(num: int | float, suffix: str = "B") -> str:
    """
    Formate une taille en octets de façon lisible (KiB, MiB, ...).

    Exemple:
        >>> sizeof_fmt(1536000)  # 1.46 MiB
    """
    num = float(num)
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
        if abs(num) < 1024.0:
            return f"{num:3.2f} {unit}{suffix}"
        num /= 1024.0
    return f"{num:.2f} Yi{suffix}"
