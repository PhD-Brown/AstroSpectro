"""AstroSpectro — Build a CSV catalog from FITS headers.

This module constructs a **CSV catalog** by iterating over a list of paths to
FITS spectra (compressed ``.fits.gz`` or plain ``.fits``).  Metadata are read
from the primary header (``HDU[0].header``) and projected onto a **stable**
column schema expected by the rest of the pipeline.

Conventions
-----------
- Input paths may be ``str`` or ``pathlib.Path``.
- Files may be **compressed** (``.fits.gz``) or plain (``.fits``).
- The CSV uses a pipe (``|``) delimiter, consistent with other pipeline
  scripts.
- Missing values are filled with the sentinel ``"UNKNOWN"``.

Inputs / Outputs
-----------------
Input :
    ``fits_paths`` — iterable of paths to FITS files (``.fits`` or
    ``.fits.gz``).

Output :
    A CSV file on disk (``output_csv``) with the columns listed below.
    Optionally, the resulting Pandas ``DataFrame`` (``return_df=True``).

Exported columns
----------------
``['fits_name', 'obsid', 'plan_id', 'mjd', 'class', 'subclass',
 'filename_original', 'author', 'data_version', 'date_creation',
 'telescope', 'longitude_site', 'latitude_site',
 'obs_date_utc', 'jd', 'designation', 'ra', 'dec',
 'fiber_id', 'fiber_type', 'object_name', 'catalog_object_type',
 'magnitude_type', 'magnitude_u', 'magnitude_g', 'magnitude_r',
 'magnitude_i', 'magnitude_z',
 'heliocentric_correction', 'radial_velocity_corr', 'seeing',
 'redshift', 'redshift_error', 'snr_u', 'snr_g', 'snr_r',
 'snr_i', 'snr_z']``

Examples
--------
>>> from pathlib import Path
>>> paths = Path("data/raw").glob("**/*.fits.gz")
>>> df = generate_catalog_from_fits(paths, "data/catalog/generated.csv",
...                                 return_df=True)
>>> df.head()

Notes
-----
- The module **never** opens the flux array; only the header table is read,
  making the operation very lightweight in terms of memory.
- If ``tqdm`` is installed, a progress bar is displayed automatically when
  ``verbose=True``.
"""

from __future__ import annotations

import gzip
from pathlib import Path
from typing import Iterable, List, Optional, Union

import pandas as pd
from astropy.io import fits

# --- Exported column schema (stable order) ------------------------------------

FIELDNAMES: List[str] = [
    "fits_name",
    "obsid",
    "plan_id",
    "mjd",
    "class",
    "subclass",
    # General metadata
    "filename_original",
    "author",
    "data_version",
    "date_creation",
    # Telescope / site
    "telescope",
    "longitude_site",
    "latitude_site",
    # Observation
    "obs_date_utc",
    "jd",
    # Position / target
    "designation",
    "ra",
    "dec",
    # Fibre & object
    "fiber_id",
    "fiber_type",
    "object_name",
    "catalog_object_type",
    # Magnitudes (typically 5 bands)
    "magnitude_type",
    "magnitude_u",
    "magnitude_g",
    "magnitude_r",
    "magnitude_i",
    "magnitude_z",
    # Reduction parameters
    "heliocentric_correction",
    "radial_velocity_corr",
    "seeing",
    # Pipeline analysis
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
    """Return a ``fits.open(...)`` context manager suitable for the file extension."""
    # Only the header table is read; memory-mapping is unnecessary here
    if "".join(path.suffixes).lower().endswith(".fits.gz"):
        # Stream from gzip without decompressing to disk
        return fits.open(gzip.open(path, "rb"), memmap=False)
    return fits.open(path, memmap=False)


def _ensure_parent_dir(output_csv: Union[str, Path]) -> None:
    """Create the parent directory of the output CSV if it does not exist."""
    out = Path(output_csv).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)


def _row_from_header(hdr: fits.Header) -> dict:
    """Map FITS header keywords to the ``FIELDNAMES`` schema."""
    # .get(...) returns UNKNOWN when the key is absent — safe and explicit
    return {
        "fits_name": UNKNOWN,  # overwritten below with the file name
        "obsid": hdr.get("OBSID", UNKNOWN),
        "plan_id": hdr.get("PLANID", UNKNOWN),
        "mjd": hdr.get("MJD", UNKNOWN),
        "class": hdr.get("CLASS", UNKNOWN),
        "subclass": hdr.get("SUBCLASS", UNKNOWN),
        # General metadata
        "filename_original": hdr.get("FILENAME", UNKNOWN),
        "author": hdr.get("AUTHOR", UNKNOWN),
        "data_version": hdr.get("DATA_VRS", UNKNOWN),
        "date_creation": hdr.get("DATE", UNKNOWN),
        # Telescope / site
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
        # Fibre & object
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
        # Reduction parameters
        "heliocentric_correction": hdr.get("HELIO", UNKNOWN),
        "radial_velocity_corr": hdr.get("VELDISP", UNKNOWN),
        "seeing": hdr.get("SEEING", UNKNOWN),
        # Pipeline analysis
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
    Generate a **CSV catalog** from a list of FITS files.

    Parameters
    ----------
    fits_paths :
        Iterable of ``.fits`` or ``.fits.gz`` paths (``str`` / ``Path``).
    output_csv :
        Destination CSV path. The parent directory is created if needed.
    verbose :
        If ``True``, print progress messages (and use ``tqdm`` when available).
    return_df :
        If ``True``, return the resulting Pandas ``DataFrame`` (handy in
        notebooks).
    delimiter :
        CSV separator. Defaults to ``|`` for consistency with the rest of
        the pipeline.

    Returns
    -------
    pandas.DataFrame or None
        The ``DataFrame`` when ``return_df=True``, otherwise ``None``.

    Notes
    -----
    - If no file is readable, an **empty** CSV with the correct columns is
      still written.
    - Missing header values are filled with ``"UNKNOWN"``.
    """
    # Materialise paths into a concrete list (needed for tqdm & len)
    files: List[Path] = [Path(p).expanduser() for p in fits_paths]
    _ensure_parent_dir(output_csv)

    if not files:
        if verbose:
            print("  > No FITS files provided: writing an empty CSV.")
        pd.DataFrame(columns=FIELDNAMES).to_csv(output_csv, sep=delimiter, index=False)
        return pd.DataFrame(columns=FIELDNAMES) if return_df else None

    # Optional progress bar
    it = files
    if verbose:
        try:
            from tqdm import tqdm  # type: ignore

            it = tqdm(files, desc="Extracting headers", unit="file")
        except Exception:
            # tqdm unavailable: silent fallback to raw iterator
            pass

    rows: List[dict] = []
    for path in it:
        try:
            with _open_fits_for_header(path) as hdul:
                hdr = hdul[0].header
                row = _row_from_header(hdr)
                row["fits_name"] = path.name  # stable file name
                rows.append(row)
                if verbose and "tqdm" not in str(type(it)):
                    print(f"[OK] {path.name} added to catalog.")
        except Exception as e:
            # Do not abort generation because of a single faulty file
            if verbose:
                print(f"[ERROR] Cannot read {path}: {e!s}")

    # Write to disk (single pass)
    if not rows:
        if verbose:
            print("  > WARNING: no exploitable headers — empty CSV written.")
        pd.DataFrame(columns=FIELDNAMES).to_csv(output_csv, sep=delimiter, index=False)
        return pd.DataFrame(columns=FIELDNAMES) if return_df else None

    df = pd.DataFrame(rows, columns=FIELDNAMES)
    df.to_csv(output_csv, sep=delimiter, index=False)

    if verbose:
        print(f"[OK] Catalog written: {Path(output_csv).resolve()}  ({len(df)} rows)")

    return df if return_df else None


# --- Minimal CLI (optional) ---------------------------------------------------
if __name__ == "__main__":
    import argparse
    from glob import glob

    parser = argparse.ArgumentParser(
        description="Generate a CSV catalog from FITS headers."
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Paths or glob patterns (e.g. data/raw/**/*.fits.gz).",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output CSV path (created if necessary).",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Disable progress messages."
    )
    args = parser.parse_args()

    # Expand the user-supplied glob patterns
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
