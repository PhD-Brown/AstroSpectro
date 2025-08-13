"""
AstroSpectro — Génération d’un catalogue à partir de headers FITS
=================================================================

Ce module construit un **catalogue CSV** en parcourant une liste de chemins vers
des spectres au format FITS (compressés `.fits.gz` ou non). Les métadonnées
sont lues dans l’en-tête (HDU[0].header) et projetées sur un schéma de
colonnes **stable** attendu par le reste de la pipeline.

Conventions
-----------
- Les chemins d’entrée peuvent être `str` ou `pathlib.Path`.
- Les fichiers peuvent être **compressés** (`.fits.gz`) ou non (`.fits`).
- Le CSV est écrit avec un séparateur `|` (pipe), conforme aux autres scripts.
- Les valeurs manquantes sont remplies avec la sentinelle `"UNKNOWN"`.

Entrées / Sorties
-----------------
Entrée :
    - `fits_paths` : itérable de chemins vers des fichiers FITS (.fits ou .fits.gz)

Sortie :
    - Un fichier CSV sur disque (`output_csv`) avec les colonnes décrites ci-dessous.
    - Optionnel : le `DataFrame` Pandas résultant (`return_df=True`).

Colonnes produites
------------------
['fits_name','obsid','plan_id','mjd','class','subclass',
 'filename_original','author','data_version','date_creation',
 'telescope','longitude_site','latitude_site',
 'obs_date_utc','jd','designation','ra','dec',
 'fiber_id','fiber_type','object_name','catalog_object_type',
 'magnitude_type','magnitude_u','magnitude_g','magnitude_r','magnitude_i','magnitude_z',
 'heliocentric_correction','radial_velocity_corr','seeing',
 'redshift','redshift_error','snr_u','snr_g','snr_r','snr_i','snr_z']

Exemple minimal
---------------
>>> from pathlib import Path
>>> paths = Path("data/raw").glob("**/*.fits.gz")
>>> df = generate_catalog_from_fits(paths, "data/catalog/generated.csv", return_df=True)
>>> df.head()

Notes
-----
- Le module n’ouvre **jamais** l’array de flux ; seule la *header table* est lue,
  ce qui rend l’opération très légère en mémoire.
- Si `tqdm` est installé, une barre de progression s’affiche automatiquement
  quand `verbose=True`.
"""

from __future__ import annotations

import gzip
from pathlib import Path
from typing import Iterable, List, Optional, Union

import pandas as pd
from astropy.io import fits

# --- Schéma des colonnes exportées (ordre stable) -----------------------------

FIELDNAMES: List[str] = [
    "fits_name",
    "obsid",
    "plan_id",
    "mjd",
    "class",
    "subclass",
    # Informations générales
    "filename_original",
    "author",
    "data_version",
    "date_creation",
    # Télescope / site
    "telescope",
    "longitude_site",
    "latitude_site",
    # Observation
    "obs_date_utc",
    "jd",
    # Position / cible
    "designation",
    "ra",
    "dec",
    # Fibre & objet
    "fiber_id",
    "fiber_type",
    "object_name",
    "catalog_object_type",
    # Magnitudes (typiquement 5 bandes)
    "magnitude_type",
    "magnitude_u",
    "magnitude_g",
    "magnitude_r",
    "magnitude_i",
    "magnitude_z",
    # Paramètres de réduction
    "heliocentric_correction",
    "radial_velocity_corr",
    "seeing",
    # Analyse pipeline
    "redshift",
    "redshift_error",
    "snr_u",
    "snr_g",
    "snr_r",
    "snr_i",
    "snr_z",
]

UNKNOWN = "UNKNOWN"


def _open_fits_for_header(path: Path):
    """Retourne un contexte `fits.open(...)` adapté à l’extension du fichier."""
    # Important : on lit *uniquement* la header table ; memmap inutile ici
    if "".join(path.suffixes).lower().endswith(".fits.gz"):
        # Lecture streamée à partir du gzip sans décompresser sur disque
        return fits.open(gzip.open(path, "rb"), memmap=False)
    return fits.open(path, memmap=False)


def _ensure_parent_dir(output_csv: Union[str, Path]) -> None:
    """Crée le dossier parent du CSV si nécessaire."""
    out = Path(output_csv).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)


def _row_from_header(hdr: fits.Header) -> dict:
    """Mappe les clés du header FITS vers le schéma `FIELDNAMES`."""
    # NB: .get(...) renvoie UNKNOWN si la clé n'existe pas — stable et explicite
    return {
        "fits_name": UNKNOWN,  # MAJ faite plus bas (nom du fichier)
        "obsid": hdr.get("OBSID", UNKNOWN),
        "plan_id": hdr.get("PLANID", UNKNOWN),
        "mjd": hdr.get("MJD", UNKNOWN),
        "class": hdr.get("CLASS", UNKNOWN),
        "subclass": hdr.get("SUBCLASS", UNKNOWN),
        # Informations générales
        "filename_original": hdr.get("FILENAME", UNKNOWN),
        "author": hdr.get("AUTHOR", UNKNOWN),
        "data_version": hdr.get("DATA_VRS", UNKNOWN),
        "date_creation": hdr.get("DATE", UNKNOWN),
        # Télescope / site
        "telescope": hdr.get("TELESCOP", UNKNOWN),
        "longitude_site": hdr.get("LONGITUD", UNKNOWN),
        "latitude_site": hdr.get("LATITUDE", UNKNOWN),
        # Observation
        "obs_date_utc": hdr.get("DATE-OBS", UNKNOWN),
        "jd": hdr.get("JD", hdr.get("MJD", UNKNOWN)),
        # Position
        "designation": hdr.get("DESIG", UNKNOWN),
        "ra": hdr.get("RA", UNKNOWN),
        "dec": hdr.get("DEC", UNKNOWN),
        # Fibre & objet
        "fiber_id": hdr.get("FIBERID", UNKNOWN),
        "fiber_type": hdr.get("FIBERTYP", UNKNOWN),
        "object_name": hdr.get("NAME", UNKNOWN),
        "catalog_object_type": hdr.get("OBJTYPE", UNKNOWN),
        # Magnitudes
        "magnitude_type": hdr.get("MAGTYPE", UNKNOWN),
        "magnitude_u": hdr.get("MAG1", UNKNOWN),
        "magnitude_g": hdr.get("MAG2", UNKNOWN),
        "magnitude_r": hdr.get("MAG3", UNKNOWN),
        "magnitude_i": hdr.get("MAG4", UNKNOWN),
        "magnitude_z": hdr.get("MAG5", UNKNOWN),
        # Paramètres de réduction
        "heliocentric_correction": hdr.get("HELIO", UNKNOWN),
        "radial_velocity_corr": hdr.get("VELDISP", UNKNOWN),
        "seeing": hdr.get("SEEING", UNKNOWN),
        # Analyse pipeline
        "redshift": hdr.get("Z", UNKNOWN),
        "redshift_error": hdr.get("Z_ERR", UNKNOWN),
        "snr_u": hdr.get("SNRU", UNKNOWN),
        "snr_g": hdr.get("SNRG", UNKNOWN),
        "snr_r": hdr.get("SNRR", UNKNOWN),
        "snr_i": hdr.get("SNRI", UNKNOWN),
        "snr_z": hdr.get("SNRZ", UNKNOWN),
    }


def generate_catalog_from_fits(
    fits_paths: Iterable[Union[str, Path]],
    output_csv: Union[str, Path],
    *,
    verbose: bool = True,
    return_df: bool = False,
    delimiter: str = "|",
) -> Optional[pd.DataFrame]:
    """
    Génère un **catalogue CSV** à partir d'une liste de fichiers FITS.

    Parameters
    ----------
    fits_paths:
        Itérable de chemins `.fits` ou `.fits.gz` (str/Path).
    output_csv:
        Chemin du fichier CSV à écrire. Le dossier parent est créé si besoin.
    verbose:
        Si True, imprime la progression (et utilise `tqdm` si présent).
    return_df:
        Si True, retourne le DataFrame Pandas créé (utile en notebook).
    delimiter:
        Séparateur du CSV. Par défaut `|` pour rester aligné avec la pipeline.

    Returns
    -------
    pandas.DataFrame or None
        Le DataFrame si `return_df=True`, sinon `None`.

    Notes
    -----
    - Si aucun fichier n'est lisible, un CSV **vide** avec les bonnes colonnes
      est tout de même écrit.
    - Les valeurs absentes dans les headers sont remplacées par `"UNKNOWN"`.
    """
    # Normalisation des chemins en liste concrète (pour tqdm & len)
    files: List[Path] = [Path(p).expanduser() for p in fits_paths]
    _ensure_parent_dir(output_csv)

    if not files:
        if verbose:
            print("  > Aucun fichier FITS fourni : création d’un CSV vide.")
        pd.DataFrame(columns=FIELDNAMES).to_csv(output_csv, sep=delimiter, index=False)
        return pd.DataFrame(columns=FIELDNAMES) if return_df else None

    # Barre de progression optionnelle
    it = files
    if verbose:
        try:
            from tqdm import tqdm  # type: ignore

            it = tqdm(files, desc="Extraction des headers", unit="fichier")
        except Exception:
            # Pas de tqdm : fallback silencieux sur l'itérateur brut
            pass

    rows: List[dict] = []
    for path in it:
        try:
            with _open_fits_for_header(path) as hdul:
                hdr = hdul[0].header
                row = _row_from_header(hdr)
                row["fits_name"] = path.name  # nom de fichier (stable)
                rows.append(row)
                if verbose and "tqdm" not in str(type(it)):
                    print(f"[OK] {path.name} ajouté au catalogue.")
        except Exception as e:
            # On n'arrête pas la génération pour un fichier défaillant
            if verbose:
                print(f"[ERREUR] Lecture impossible pour {path} : {e!s}")

    # Écriture sur disque (une seule passe)
    if not rows:
        if verbose:
            print("  > AVERTISSEMENT : aucun header exploitable — CSV vide écrit.")
        pd.DataFrame(columns=FIELDNAMES).to_csv(output_csv, sep=delimiter, index=False)
        return pd.DataFrame(columns=FIELDNAMES) if return_df else None

    df = pd.DataFrame(rows, columns=FIELDNAMES)
    df.to_csv(output_csv, sep=delimiter, index=False)

    if verbose:
        print(
            f"[OK] Catalogue écrit : {Path(output_csv).resolve()}  ({len(df)} lignes)"
        )

    return df if return_df else None


# --- CLI minimal (optionnel) --------------------------------------------------
if __name__ == "__main__":
    import argparse
    from glob import glob

    parser = argparse.ArgumentParser(
        description="Génère un catalogue CSV à partir de headers FITS."
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Chemins ou motifs glob (ex: data/raw/**/*.fits.gz).",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Chemin du CSV de sortie (sera créé si nécessaire).",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Désactive les messages de progression."
    )
    args = parser.parse_args()

    # Expansion des patterns glob fournis
    expanded: List[str] = []
    for pattern in args.inputs:
        matches = glob(pattern, recursive=True)
        expanded.extend(matches if matches else [pattern])

    generate_catalog_from_fits(
        expanded,
        args.output,
        verbose=not args.quiet,
        return_df=False,
    )
