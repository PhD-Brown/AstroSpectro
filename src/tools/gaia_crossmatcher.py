"""AstroSpectro — Enrich a catalog via Gaia DR3 cross-matching.

This module enriches a source catalog (ICRS RA/Dec positions) with
**Gaia DR3** photometric and astrometric measurements.  Two complementary
strategies are provided:

1. *bulk* (recommended) — Server-side cross-match via ``TAP_UPLOAD``:
   positions are uploaded in batches, joined against
   ``gaiadr3.gaia_source``, and the best match (smallest angular
   distance) is returned.  This is fast and network-efficient.  A
   **retry** logic halves the batch size when the service returns a 500.

2. *cone* — Individual cone searches (**star by star**) via
   ``Gaia.cone_search_async``.  Slower but robust when ``TAP_UPLOAD``
   is temporarily unavailable.

The ``auto`` mode tries *bulk* first, then falls back to *cone*.

Conventions
-----------
- Coordinates in **degrees** (ICRS).
- Search radius given in **arcseconds**.
- An internal ``objid`` field is fabricated if absent, to realign results
  with the rows of the source DataFrame.
- The **RUWE** filter is applied after the cross-match: if
  ``RUWE >= ruwe_max``, the row is **kept** but all Gaia columns are
  set to NaN.  This avoids dropping sources during merges.

Inputs / Outputs
-----------------
Input :
    A ``pandas.DataFrame`` with at least ``ra`` and ``dec`` columns
    (in degrees).  An ``objid`` column is optional.

Output :
    A tuple ``(df_merged, stats)`` where ``df_merged`` is the enriched
    DataFrame and ``stats`` a summary dict.  ``df_merged`` is also
    written to disk at ``output_catalog_path`` (CSV).

Dependencies
------------
astropy (coordinates, table, units), astroquery.gaia, pandas, numpy, tqdm.

Examples
--------
>>> from gaia_crossmatcher import enrich_catalog_with_gaia
>>> df = pd.DataFrame({"ra": [10.1, 231.2], "dec": [-5.3, 19.7]})
>>> out, stats = enrich_catalog_with_gaia(
...     df, "data/catalog/gaia_enriched.csv",
...     search_radius_arcsec=0.5, mode="auto", ruwe_max=1.4
... )
>>> stats["match_rate_pct"]
72.5
"""

from __future__ import annotations

import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astroquery.gaia import Gaia
from tqdm import tqdm

# ----------------------------- Constants ------------------------------------

GAIA_MAIN = "gaiadr3.gaia_source"

# Useful and relatively stable output fields
BASE_FIELDS = [
    "bp_rp",
    "phot_bp_rp_excess_factor",
    "phot_g_mean_flux",
    "phot_bp_mean_flux",
    "phot_rp_mean_flux",
    "teff_gspphot",
    "logg_gspphot",
    "mh_gspphot",
    "ag_gspphot",
    "ebpminrp_gspphot",
    "pmra",
    "pmdec",
    "radial_velocity",
    "distance_gspphot",
    "ruwe",
    "astrometric_excess_noise",
    "phot_variable_flag",
]

# GSP-Phot fields that are sometimes absent/unstable — optional
RISKY_FIELDS = ["radius_gspphot", "mass_gspphot", "age_gspphot"]


# ----------------------------- Utilities -------------------------------------


def _ensure_numeric_radec(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure ``ra`` and ``dec`` are numeric (float64), NaN if invalid.

    Parameters
    ----------
    df : pd.DataFrame
        Input catalog.

    Returns
    -------
    pd.DataFrame
        Copy with ``ra`` / ``dec`` coerced to numeric.
    """
    df = df.copy()
    df["ra"] = pd.to_numeric(df["ra"], errors="coerce")
    df["dec"] = pd.to_numeric(df["dec"], errors="coerce")
    return df


def _validate_catalog(df: pd.DataFrame) -> pd.DataFrame:
    """Validate that ``ra`` / ``dec`` exist and create ``objid`` if absent.

    Raises
    ------
    ValueError
        If ``ra`` or ``dec`` columns are missing.
    """
    if "ra" not in df or "dec" not in df:
        raise ValueError("The input catalog must contain 'ra' and 'dec' columns.")
    if "objid" not in df.columns:
        df = df.copy()
        df["objid"] = df.index.astype(int)
    return df


def _features_from_row(s: pd.Series | dict, include_risky: bool) -> Dict[str, float]:
    """Extract a subset of Gaia fields plus derived colour indices."""
    fields = BASE_FIELDS + (RISKY_FIELDS if include_risky else [])
    out = {k: s.get(k, None) for k in fields}

    # Handy colour indices
    bp = s.get("phot_bp_mean_mag")
    g = s.get("phot_g_mean_mag")
    rp = s.get("phot_rp_mean_mag")
    if bp is not None and g is not None:
        out["bp_g"] = bp - g
    if g is not None and rp is not None:
        out["g_rp"] = g - rp
    return out


# ---------------------------- Mode BULK (TAP) --------------------------------


def _bulk_crossmatch(
    df_src: pd.DataFrame,
    radius_arcsec: float,
    gaia_table: str,
    include_risky: bool,
    debug: bool = False,
) -> pd.DataFrame:
    """Server-side cross-match via ``TAP_UPLOAD`` (fast, batched).

    Strategy
    --------
    - Upload a batch ``src(objid, ra, dec)`` into TAP.
    - Spatial join with ``gaia_table`` via
      ``CONTAINS(..., CIRCLE(...))``.
    - **Retry**: on exception (often a 500), halve the batch size and
      retry; give up on the batch below a size of 50.

    Notes
    -----
    ``Gaia.ROW_LIMIT`` is forced to ``-1`` to prevent truncation.

    Returns
    -------
    pd.DataFrame
        One row per matched ``objid`` (best match).
    """
    # Clean inputs: keep only rows with valid RA/Dec
    df = df_src.loc[
        df_src["ra"].notna() & df_src["dec"].notna(), ["objid", "ra", "dec"]
    ].copy()
    df["ra"] = pd.to_numeric(df["ra"], errors="coerce")
    df["dec"] = pd.to_numeric(df["dec"], errors="coerce")
    df = df.dropna(subset=["ra", "dec"])
    if df.empty:
        return pd.DataFrame(columns=["objid"])

    try:
        Gaia.ROW_LIMIT = -1
    except Exception:
        pass

    radius_deg = float(radius_arcsec) / 3600.0

    gaia_cols = [
        "source_id",
        "ra AS gaia_ra",
        "dec AS gaia_dec",
        "parallax",
        "parallax_error",
        "pmra",
        "pmdec",
        "radial_velocity",
        "ruwe",
        "phot_variable_flag",
        "phot_g_mean_mag",
        "phot_bp_mean_mag",
        "phot_rp_mean_mag",
        "phot_g_mean_flux",
        "phot_bp_mean_flux",
        "phot_rp_mean_flux",
        "bp_rp",
        "phot_bp_rp_excess_factor",
        "astrometric_excess_noise",
    ]
    sel_gaia = ",".join([f"g.{c}" for c in gaia_cols])

    def as_table(chunk: pd.DataFrame) -> Table:
        """Convert a chunk to an astropy Table with explicit dtypes."""
        return Table(
            [
                chunk["objid"].astype("int64").values,
                chunk["ra"].astype("float64").values,
                chunk["dec"].astype("float64").values,
            ],
            names=("objid", "ra", "dec"),
            dtype=("int64", "float64", "float64"),
        )

    def run_cm_on_chunk(chunk_df: pd.DataFrame) -> pd.DataFrame:
        """Run the TAP_UPLOAD query for one batch and return a DataFrame."""
        t = as_table(chunk_df)
        q = f"""
        SELECT
          u.objid,
          {sel_gaia},
          DISTANCE(POINT('ICRS', u.ra, u.dec), POINT('ICRS', g.ra, g.dec)) * 3600.0
          AS match_dist_arcsec
        FROM TAP_UPLOAD.src AS u
        JOIN {gaia_table} AS g
          ON 1 = CONTAINS(
                 POINT('ICRS', u.ra, u.dec),
                 CIRCLE('ICRS', g.ra, g.dec, {radius_deg})
               )
        """
        job = Gaia.launch_job_async(
            q, upload_resource=t, upload_table_name="src", dump_to_file=False
        )
        return job.get_results().to_pandas()

    # Batched loop with retry logic
    all_rows = []
    batch_size = 400
    i = 0
    while i < len(df):
        j = min(i + batch_size, len(df))
        part = df.iloc[i:j]
        try:
            if debug:
                print(f"[bulk] lot {i}:{j} size={len(part)}")
            all_rows.append(run_cm_on_chunk(part))
            i = j
        except Exception as e:
            if debug:
                print(f"[bulk] error on {i}:{j} → retrying smaller ({e})")
            if batch_size <= 50:
                # give up on this batch (will be covered by 'cone' fallback)
                i = j
            else:
                batch_size //= 2

    df_cm = (
        pd.concat(all_rows, ignore_index=True)
        if all_rows
        else pd.DataFrame(columns=["objid"])
    )
    if df_cm.empty:
        return pd.DataFrame(columns=["objid"])

    # Best match per objid (ascending distance)
    df_cm.sort_values(["objid", "match_dist_arcsec"], inplace=True)
    df_best = df_cm.groupby("objid", as_index=False).first()

    # GSP-Phot supplement
    ap_base = [
        "teff_gspphot",
        "logg_gspphot",
        "mh_gspphot",
        "distance_gspphot",
        "ag_gspphot",
        "ebpminrp_gspphot",
    ]
    ap_risky = ["radius_gspphot", "mass_gspphot", "age_gspphot"]
    ap_cols = ap_base + (ap_risky if include_risky else [])
    if ap_cols and not df_best.empty:
        ids_tbl = Table(
            [df_best["source_id"].astype("int64").values],
            names=("source_id",),
            dtype=("int64",),
        )
        sel_ap = ",".join([f"ap.{c}" for c in ap_cols])
        q_ap = f"""
        SELECT i.source_id, {sel_ap}
        FROM TAP_UPLOAD.ids_tmp AS i
        LEFT JOIN gaiadr3.astrophysical_parameters AS ap
               ON ap.source_id = i.source_id
        """
        job_ap = Gaia.launch_job_async(
            q_ap,
            upload_resource=ids_tbl,
            upload_table_name="ids_tmp",
            dump_to_file=False,
        )
        df_ap = job_ap.get_results().to_pandas()
        if not df_ap.empty:
            df_best = df_best.merge(df_ap, on="source_id", how="left")

    # Realign on df_src (order/presence)
    best_map = df_best.set_index("objid")
    aligned = []
    for _, row in df_src.iterrows():
        oid = int(row.objid)
        s = best_map.loc[oid] if oid in best_map.index else None
        if s is None:
            aligned.append({"objid": oid})
        else:
            features = _features_from_row(s, include_risky)
            aligned.append(
                {
                    "objid": oid,
                    "source_id": s.get("source_id"),
                    "gaia_ra": s.get("gaia_ra"),
                    "gaia_dec": s.get("gaia_dec"),
                    "parallax": s.get("parallax"),
                    "parallax_error": s.get("parallax_error"),
                    "phot_g_mean_mag": s.get("phot_g_mean_mag"),
                    "phot_bp_mean_mag": s.get("phot_bp_mean_mag"),
                    "phot_rp_mean_mag": s.get("phot_rp_mean_mag"),
                    "match_dist_arcsec": s.get("match_dist_arcsec"),
                    "ruwe": s.get("ruwe"),
                    **features,
                }
            )
    return pd.DataFrame(aligned)


# ---------------------------- Mode CONE (fallback) ---------------------------


def _cone_crossmatch(
    df_src: pd.DataFrame, radius_arcsec: float, include_risky: bool
) -> pd.DataFrame:
    """Star-by-star cross-match via ``Gaia.cone_search_async`` (slow)."""
    out = []
    radius = radius_arcsec * u.arcsec
    for _, row in tqdm(
        df_src.iterrows(), total=len(df_src), desc="↔ Matching LAMOST ↔ GAIA (cone)"
    ):
        objid = int(row.objid)
        ra = row["ra"]
        dec = row["dec"]
        if pd.isna(ra) or pd.isna(dec):
            out.append({"objid": objid})
            continue
        try:
            coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
            j = Gaia.cone_search_async(coordinate=coord, radius=radius)
            r = j.get_results()
            if len(r) == 0:
                out.append({"objid": objid})
                continue
            r.sort("dist")
            best = r[0]
            s = {k: best.get(k, None) for k in best.colnames}

            # 'dist' is in degrees → convert to arcseconds
            dist_deg = s.get("dist")
            match_arcsec = float(dist_deg) * 3600 if dist_deg is not None else None

            features = _features_from_row(s, include_risky)
            out.append(
                {
                    "objid": objid,
                    "source_id": s.get("source_id"),
                    "gaia_ra": s.get("ra"),
                    "gaia_dec": s.get("dec"),
                    "parallax": s.get("parallax"),
                    "parallax_error": s.get("parallax_error"),
                    "phot_g_mean_mag": s.get("phot_g_mean_mag"),
                    "phot_bp_mean_mag": s.get("phot_bp_mean_mag"),
                    "phot_rp_mean_mag": s.get("phot_rp_mean_mag"),
                    "match_dist_arcsec": match_arcsec,
                    "ruwe": s.get("ruwe"),
                    **features,
                }
            )
        except Exception:
            out.append({"objid": objid})
    return pd.DataFrame(out)


# ------------------------------ Main API -------------------------------------


def enrich_catalog_with_gaia(
    input_catalog_df: pd.DataFrame,
    output_catalog_path: str,
    search_radius_arcsec: float = 0.5,
    ruwe_max: float = 1.4,
    include_risky: bool = False,
    overwrite: bool = False,
    mode: str = "auto",
    gaia_table: str = GAIA_MAIN,
    gaia_user: str | None = None,
    gaia_pass: str | None = None,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Enrich a catalog (RA/Dec) with Gaia DR3 photometry and astrometry.

    Parameters
    ----------
    input_catalog_df : DataFrame
        Input catalog with ``ra``, ``dec`` (degrees).  ``objid`` is optional.
    output_catalog_path : str
        Destination CSV path (overwritten when ``overwrite=True``).
    search_radius_arcsec : float, default=0.5
        Search radius around each source, in **arcseconds**.
    ruwe_max : float, default=1.4
        RUWE threshold.  If available and >= threshold, Gaia columns are
        set to NaN.  Pass ``None`` to disable this filter.
    include_risky : bool, default=False
        Include the less stable GSP-Phot columns (radius/mass/age).
    overwrite : bool, default=False
        Overwrite ``output_catalog_path`` if it already exists.
    mode : {'auto', 'bulk', 'cone'}, default='auto'
        Cross-match strategy.
    gaia_table : str, default=GAIA_MAIN
        Primary Gaia table to join (e.g. ``gaiadr3.gaia_source``).
    gaia_user, gaia_pass : str or None
        Optional credentials for ``Gaia.login`` (ignored on failure).

    Returns
    -------
    (df_merged, stats) : (DataFrame, dict)
        Enriched DataFrame (also saved to disk) and match statistics.
    """
    # (optional) Gaia authentication
    if gaia_user and gaia_pass:
        try:
            Gaia.login(user=gaia_user, password=gaia_pass)
        except Exception:
            pass  # auth failure must not break the enrichment

    # Reuse existing enriched file if available
    if os.path.exists(output_catalog_path) and not overwrite:
        print("  > Existing enriched file found. Loading…")
        return pd.read_csv(output_catalog_path, sep=","), {"skipped": True}

    # Prepare and validate source data
    df_src = _validate_catalog(input_catalog_df.copy())
    df_src = _ensure_numeric_radec(df_src)

    # 1) Gaia cross-match
    df_gaia = pd.DataFrame()
    if mode in ("auto", "bulk"):
        try:
            print("  > Attempting cross-match in 'bulk' mode…")
            df_gaia = _bulk_crossmatch(
                df_src, search_radius_arcsec, gaia_table, include_risky
            )
        except Exception as e:
            print(f"  > 'bulk' mode failed: {e}. Switching to 'cone' mode.")
            if mode == "bulk":
                raise
            df_gaia = pd.DataFrame()

    if df_gaia.empty and mode in ("auto", "cone"):
        print("  > Starting cross-match in 'cone' mode (star by star)…")
        df_gaia = _cone_crossmatch(df_src, search_radius_arcsec, include_risky)

    # 2) RUWE filter (soft nullification)
    ruwe_thr = None if ruwe_max is None else float(np.atleast_1d(ruwe_max)[0])
    if ("ruwe" in df_gaia.columns) and (ruwe_thr is not None):
        ruwe_vals = pd.to_numeric(df_gaia["ruwe"], errors="coerce")
        bad = (ruwe_vals.notna() & (ruwe_vals >= ruwe_thr)).fillna(False)
        cols_to_null = [c for c in df_gaia.columns if c != "objid"]
        df_gaia.loc[bad, cols_to_null] = pd.NA

    # 3) Merge + type post-processing
    df_merged = pd.merge(df_src, df_gaia, on="objid", how="left")
    df_merged.drop(columns=["objid"], inplace=True, errors="ignore")

    numeric_cols = [
        "gaia_ra",
        "gaia_dec",
        "parallax",
        "parallax_error",
        "phot_g_mean_mag",
        "phot_bp_mean_mag",
        "phot_rp_mean_mag",
        "phot_g_mean_flux",
        "phot_bp_mean_flux",
        "phot_rp_mean_flux",
        "bp_rp",
        "phot_bp_rp_excess_factor",
        "astrometric_excess_noise",
        "pmra",
        "pmdec",
        "radial_velocity",
        "ruwe",
        "teff_gspphot",
        "logg_gspphot",
        "mh_gspphot",
        "ag_gspphot",
        "ebpminrp_gspphot",
        "distance_gspphot",
        "match_dist_arcsec",
        "bp_g",
        "g_rp",
        "radius_gspphot",
        "mass_gspphot",
        "age_gspphot",
    ]
    for c in numeric_cols:
        if c in df_merged.columns:
            df_merged[c] = pd.to_numeric(df_merged[c], errors="coerce")

    if "source_id" in df_merged.columns:
        df_merged["source_id"] = pd.to_numeric(
            df_merged["source_id"], errors="coerce"
        ).astype("Int64")
    if "phot_variable_flag" in df_merged.columns:
        df_merged["phot_variable_flag"] = df_merged["phot_variable_flag"].astype(
            "string"
        )

    # 4) Write to disk
    df_merged.to_csv(output_catalog_path, index=False, sep=",")

    # 5) Statistics
    n_total = len(df_src)
    n_matched = (
        int(df_merged["source_id"].notna().sum()) if "source_id" in df_merged else 0
    )
    stats = {
        "matched": n_matched,
        "total": int(n_total),
        "match_rate_pct": round(100 * n_matched / max(1, n_total), 1),
        "mode_used": "bulk" if (not df_gaia.empty and mode != "cone") else "cone",
        "ruwe_max": ruwe_max,
    }

    return df_merged, stats
