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
import glob
import math
import argparse
from pathlib import Path
from typing import List, Any, Iterable, Union, Optional
from astropy.io import fits
import pandas as pd

UNKNOWN = "UNKNOWN"


def _hdr_first(hdr, keys: Iterable[str], default: Any):
    """Return the first existing header value among keys (trim strings)."""
    for k in keys:
        if k in hdr:
            v = hdr.get(k)
            if v is None:
                continue
            if isinstance(v, str):
                v2 = v.strip()
                if v2 == "":
                    continue
                return v2
            return v
    return default


def _to_float(x, default=math.nan):
    """Best-effort float conversion (handles string numbers)."""
    if x is None:
        return default
    if isinstance(x, str):
        x = x.strip()
        if x == "" or x.upper() == "UNKNOWN":
            return default
    try:
        return float(x)
    except Exception:
        return default


def _mag_or_nan(x):
    """
    LAMOST headers sometimes use 99/99.00 as 'missing magnitude'.
    Convert to NaN so the column stays numeric.
    """
    v = _to_float(x, default=math.nan)
    if math.isfinite(v) and v >= 90:
        return math.nan
    return v


# --- Exported column schema (stable order) ------------------------------------

FIELDNAMES = [
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
    "longitude_site",
    "latitude_site",
    "obs_date_utc",
    "jd",
    "designation",
    "ra",
    "dec",
    "fiber_id",
    "fiber_type",
    "object_name",
    "catalog_object_type",
    "magnitude_type",
    "magnitude_u",
    "magnitude_g",
    "magnitude_r",
    "magnitude_i",
    "magnitude_z",
    "heliocentric_correction",
    "radial_velocity_corr",
    "seeing",
    "redshift",
    "redshift_error",
    "snr_u",
    "snr_g",
    "snr_r",
    "snr_i",
    "snr_z",
    # --- extra header metadata (DR5) ---
    "ra_obs",
    "dec_obs",
    "focus_mm",
    "x_value_mm",
    "y_value_mm",
    "objname",
    "tcomment",
    "tsource",
    "tfrom",
    "obs_type",
    "obscomm",
    "magnitude_j",
    "magnitude_h",
    "offset",
    "offset_v",
    "fibermas",
    "scamean",
    "spid",
    "spra",
    "spdec",
    "slit_mod",
    "skychi2",
    "schi2min",
    "schi2max",
    "nstd",
    "fstar",
    "nskies",
    "sflatten",
    "pcaskysb",
    "wfit_type",
    "coeff0",
    "coeff1",
    "crval1",
    "cd1_1",
    "crpix1",
    "dc_flag",
]


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
    return {
        "fits_name": UNKNOWN,  # overwritten later with the file name
        # --- IDs / classes ---
        "obsid": hdr.get("OBSID", UNKNOWN),
        "plan_id": hdr.get("PLANID", UNKNOWN),
        "mjd": hdr.get("MJD", UNKNOWN),
        "class": hdr.get("CLASS", UNKNOWN),
        "subclass": hdr.get("SUBCLASS", UNKNOWN),
        # --- General metadata ---
        "filename_original": hdr.get("FILENAME", UNKNOWN),
        "author": hdr.get("AUTHOR", UNKNOWN),
        "data_version": _hdr_first(hdr, ["DATA_V", "DATA_VRS", "DR"], default=UNKNOWN),
        "date_creation": hdr.get("DATE", UNKNOWN),
        "pipeline_version": hdr.get("VERSPIPE", UNKNOWN),
        # --- Telescope / site ---
        "telescope": hdr.get("TELESCOP", UNKNOWN),
        "longitude_site": _to_float(hdr.get("LONGITUD", None), default=math.nan),
        "latitude_site": _to_float(hdr.get("LATITUDE", None), default=math.nan),
        # --- Observation time ---
        "obs_date_utc": hdr.get("DATE-OBS", UNKNOWN),
        # LAMOST: MJD and LMJD exist often; JD not always present.
        "jd": _hdr_first(hdr, ["JD", "MJD", "LMJD"], default=UNKNOWN),
        # --- Position ---
        "designation": hdr.get("DESIG", UNKNOWN),
        "ra": _to_float(hdr.get("RA", None), default=math.nan),
        "dec": _to_float(hdr.get("DEC", None), default=math.nan),
        # --- Fibre & object ---
        "fiber_id": hdr.get("FIBERID", UNKNOWN),
        "fiber_type": hdr.get("FIBERTYP", UNKNOWN),
        # In DR5 the object name is typically OBJNAME; NAME can be absent.
        "object_name": _hdr_first(hdr, ["OBJNAME", "NAME"], default=UNKNOWN),
        "catalog_object_type": hdr.get("OBJTYPE", UNKNOWN),
        # --- Magnitudes ---
        "magnitude_type": hdr.get("MAGTYPE", UNKNOWN),
        "magnitude_u": _mag_or_nan(hdr.get("MAG1", None)),
        "magnitude_g": _mag_or_nan(hdr.get("MAG2", None)),
        "magnitude_r": _mag_or_nan(hdr.get("MAG3", None)),
        "magnitude_i": _mag_or_nan(hdr.get("MAG4", None)),
        "magnitude_z": _mag_or_nan(hdr.get("MAG5", None)),
        # --- Reduction / analysis ---
        "heliocentric_correction": hdr.get("HELIO", UNKNOWN),
        "radial_velocity_corr": _to_float(hdr.get("HELIO_RV", None), default=math.nan),
        "seeing": _to_float(hdr.get("SEEING", None), default=math.nan),
        "redshift": _to_float(hdr.get("Z", None), default=math.nan),
        "redshift_error": _to_float(hdr.get("Z_ERR", None), default=math.nan),
        "snr_u": _to_float(hdr.get("SNRU", None), default=math.nan),
        "snr_g": _to_float(hdr.get("SNRG", None), default=math.nan),
        "snr_r": _to_float(hdr.get("SNRR", None), default=math.nan),
        "snr_i": _to_float(hdr.get("SNRI", None), default=math.nan),
        "snr_z": _to_float(hdr.get("SNRZ", None), default=math.nan),
        "ra_obs": _to_float(hdr.get("RA_OBS", None), default=math.nan),
        "dec_obs": _to_float(hdr.get("DEC_OBS", None), default=math.nan),
        "focus_mm": _to_float(hdr.get("FOCUS", None), default=math.nan),
        "x_value_mm": _to_float(hdr.get("X_VALUE", None), default=math.nan),
        "y_value_mm": _to_float(hdr.get("Y_VALUE", None), default=math.nan),
        "objname": _hdr_first(hdr, ["OBJNAME", "NAME"], default=UNKNOWN),
        # Targeting / plate info (often an internal field / tile identifier)
        "tcomment": hdr.get("TCOMMENT", UNKNOWN),
        "tsource": hdr.get("TSOURCE", UNKNOWN),
        "tfrom": hdr.get("TFROM", UNKNOWN),
        # Filtering: keep only Science frames if you want
        "obs_type": hdr.get("OBS_TYPE", UNKNOWN),
        "obscomm": hdr.get("OBSCOMM", UNKNOWN),
        # Extra mags (J/H often 99 when missing)
        "magnitude_j": _mag_or_nan(hdr.get("MAG6", None)),
        "magnitude_h": _mag_or_nan(hdr.get("MAG7", None)),
        # Offsets / masks
        "offset": hdr.get("OFFSET", UNKNOWN),
        "offset_v": _to_float(hdr.get("OFFSET_V", None), default=math.nan),
        "fibermas": hdr.get("FIBERMAS", UNKNOWN),
        # Scatter light, spectrograph & reduction QC
        "scamean": _to_float(hdr.get("SCAMEAN", None), default=math.nan),
        "spid": hdr.get("SPID", UNKNOWN),
        "spra": _to_float(hdr.get("SPRA", None), default=math.nan),
        "spdec": _to_float(hdr.get("SPDEC", None), default=math.nan),
        "slit_mod": hdr.get("SLIT_MOD", UNKNOWN),
        "skychi2": _to_float(hdr.get("SKYCHI2", None), default=math.nan),
        "schi2min": _to_float(hdr.get("SCHI2MIN", None), default=math.nan),
        "schi2max": _to_float(hdr.get("SCHI2MAX", None), default=math.nan),
        "nstd": hdr.get("NSTD", UNKNOWN),
        "fstar": hdr.get("FSTAR", UNKNOWN),
        "nskies": hdr.get("NSKIES", UNKNOWN),
        "sflatten": hdr.get("SFLATTEN", UNKNOWN),
        "pcaskysb": hdr.get("PCASKYSB", UNKNOWN),
        # Wavelength solution (often constant across spectra, but ok for traceability)
        "wfit_type": hdr.get("WFITTYPE", UNKNOWN),
        "coeff0": _to_float(hdr.get("COEFF0", None), default=math.nan),
        "coeff1": _to_float(hdr.get("COEFF1", None), default=math.nan),
        "crval1": _to_float(hdr.get("CRVAL1", None), default=math.nan),
        "cd1_1": _to_float(hdr.get("CD1_1", None), default=math.nan),
        "crpix1": _to_float(hdr.get("CRPIX1", None), default=math.nan),
        "dc_flag": hdr.get("DC-FLAG", UNKNOWN),
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
