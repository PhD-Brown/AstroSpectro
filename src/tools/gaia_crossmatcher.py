"""AstroSpectro — Enrich a catalog via Gaia DR3 cross-matching.

This module enriches a source catalog (ICRS RA/Dec positions) with
**Gaia DR3** photometric and astrometric measurements.  Two complementary
strategies are provided:

1. *bulk* (recommended) — Server-side cross-match via ``TAP_UPLOAD``:
   positions are uploaded in batches, joined against
   ``gaiadr3.gaia_source``, and the best match (smallest angular
   distance) is returned.

   .. note:: **ESA Archive TAP limitation (confirmed April 2026)**
       Expressions like ``COS(RADIANS(u.dec))`` evaluated inside a ``JOIN ON``
       clause against a TAP_UPLOAD table cause a server-side Java NPE:
       ``Cannot invoke "java.util.List.iterator()" because "results" is null``.
       Fix: precompute ``ra_min/ra_max/dec_min/dec_max`` in Python and upload
       them as plain columns; the ADQL JOIN then uses only simple column
       references.  The exact circular filter is applied in Python post-fetch.
       Previously, ``CONTAINS(...CIRCLE(...))`` queries with TAP_UPLOAD JOINs
       timed out (HTTP 408) following the Dec 2025 archive release 3.10.
       Reference: https://www.cosmos.esa.int/web/gaia/news#WorkaroundArchive

   A **retry** logic halves the batch size (with exponential backoff) when
   the service returns an error; batches that fail below 50 sources are
   skipped and logged.

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
- Derived columns added post-match:
    - ``parallax_snr``  : parallax / parallax_error
    - ``M_G``           : absolute G magnitude (requires parallax_snr ≥ 5)

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

.. warning:: astroquery version
    Following the June 2025 Gaia Archive v3.8 update, the TAP job ID format
    changed.  Install the bleeding-edge astroquery to avoid authentication
    or job-tracking issues::

        pip install git+https://github.com/astropy/astroquery.git

Examples
--------
>>> from gaia_crossmatcher import enrich_catalog_with_gaia
>>> df = pd.DataFrame({"ra": [10.1, 231.2], "dec": [-5.3, 19.7]})
>>> out, stats = enrich_catalog_with_gaia(
...     df, "data/catalog/gaia_enriched.csv",
...     search_radius_arcsec=0.5, mode="auto", ruwe_max=1.4
... )
>>> stats["match_rate_pct"]
"""

from __future__ import annotations

import os
import time
import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astroquery.gaia import Gaia
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ----------------------------- Constants ------------------------------------

GAIA_MAIN = "gaiadr3.gaia_source"

# Useful and relatively stable output fields (used by _features_from_row in cone mode)
BASE_FIELDS = [
    "phot_g_mean_mag",
    "phot_bp_mean_mag",
    "phot_rp_mean_mag",
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
    # Enriched GSP-Phot fields (available in astrophysical_parameters)
    "evolstage_gspphot",
    "lum_gspphot",
]

# GSP-Phot fields that are sometimes absent/unstable — optional
RISKY_FIELDS = ["radius_gspphot", "mass_gspphot", "age_gspphot"]

# Polling configuration for async TAP jobs
_POLL_INTERVAL_S = 8  # seconds between status checks
_POLL_MAX_WAIT_S = 300  # abort poll after 5 minutes without completion
_INITIAL_BATCH_SIZE = 200  # conservative: 1000 triggers OOM on ESA TAP (Apr 2026)

# ── CDS XMatch (primary strategy — bypasses ESA TAP_UPLOAD) ──────────────────
# ESA TAP_UPLOAD broken since at least Apr 2026 (Java NPE server-side).
# CDS XMatch is independent infrastructure and handles Gaia DR3 natively.
_VIZIER_GAIA_DR3 = "vizier:I/355/gaiadr3"

# VizieR column names → our internal naming convention.
_XMATCH_COL_MAP = {
    "Source": "source_id",
    "RA_ICRS": "gaia_ra",
    "RAJ2000": "gaia_ra",
    "DE_ICRS": "gaia_dec",
    "DEJ2000": "gaia_dec",
    "Plx": "parallax",
    "e_Plx": "parallax_error",
    "pmRA": "pmra",
    "pmDE": "pmdec",
    "RUWE": "ruwe",
    "Gmag": "phot_g_mean_mag",
    "BPmag": "phot_bp_mean_mag",
    "RPmag": "phot_rp_mean_mag",
    "FG": "phot_g_mean_flux",
    "FBP": "phot_bp_mean_flux",
    "FRP": "phot_rp_mean_flux",
    "BP-RP": "bp_rp",
    "BPERPcor": "phot_bp_rp_excess_factor",
    "VarFlag": "phot_variable_flag",
    "angDist": "match_dist_arcsec",
    # ── GSP-Phot parameters (present in VizieR I/355/gaiadr3 XMatch results) ─
    # These were already returned by XMatch but not mapped — fixed Apr 2026.
    "Teff": "teff_gspphot",  # effective temperature [K]
    "logg": "logg_gspphot",  # surface gravity [log cgs]
    "__M_H_": "mh_gspphot",  # metallicity [M/H] (VizieR encoding)
    "MH": "mh_gspphot",  # alternate VizieR column name
    "Dist": "distance_gspphot",  # GSP-Phot distance [pc]
    "AG": "ag_gspphot",  # G-band extinction [mag]
    "E_BP-RP_": "ebpminrp_gspphot",  # BP-RP reddening [mag]
    "Lum_Flame": "lum_gspphot",  # luminosity (FLAME) [L☉] if present
    "Lum*": "lum_gspphot",  # alternate VizieR name
    # Uncertainty bounds — stored for diagnostics, not used in pipeline
    "b_Teff": "teff_gspphot_lo",
    "B_Teff": "teff_gspphot_hi",
    "b_Dist": "distance_gspphot_lo",
    "B_Dist": "distance_gspphot_hi",
    # Flux errors — not used internally
    "e_FG": None,
    "e_FBP": None,
    "e_FRP": None,
    # Magnitude observation counts — not used
    "o_Gmag": None,
    "o_BPmag": None,
    "o_RPmag": None,
    # Magnitude errors — not used
    "e_Gmag": None,
    "e_BPmag": None,
    "e_RPmag": None,
}

# Batch size for the GSP-Phot `source_id IN (...)` supplementary TAP query
_GSPPHOT_BATCH_SIZE = 500


# ----------------------------- Utilities -------------------------------------


def _ensure_numeric_radec(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure ``ra`` and ``dec`` are numeric (float64), NaN if invalid."""
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


def _poll_job(job, debug: bool = False) -> None:
    """Block until a TAP async job reaches a terminal state.

    Raises
    ------
    RuntimeError
        If the job ends in ERROR or ABORTED, or if the poll timeout is
        exceeded.
    """
    elapsed = 0
    while elapsed < _POLL_MAX_WAIT_S:
        phase = job.get_phase()
        if phase in ("COMPLETED", "ERROR", "ABORTED"):
            if debug:
                logger.info("[TAP] Job %s terminé — statut : %s", job.jobid, phase)
            break
        if debug:
            logger.info(
                "[TAP] Job %s en cours (statut : %s) — attente %ds…",
                job.jobid,
                phase,
                _POLL_INTERVAL_S,
            )
        time.sleep(_POLL_INTERVAL_S)
        elapsed += _POLL_INTERVAL_S
    else:
        raise RuntimeError(
            f"Le job TAP {job.jobid} n'a pas terminé après {_POLL_MAX_WAIT_S}s."
        )

    if job.get_phase() != "COMPLETED":
        raise RuntimeError(
            f"Le job TAP {job.jobid} a échoué — statut final : {job.get_phase()}"
        )


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
    - Spatial JOIN with ``gaia_table`` using a **BETWEEN box** condition
      (replaces the ``CONTAINS(...CIRCLE(...))`` pattern that times out on
      the ESA archive since the Dec 2025 infrastructure upgrade).
    - **Retry**: on exception, halve the batch size with exponential
      backoff; give up on a batch below 50 sources.

    Notes
    -----
    ``Gaia.ROW_LIMIT`` is forced to ``-1`` to prevent truncation.

    Returns
    -------
    pd.DataFrame
        One row per matched ``objid`` (best match by angular distance).
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
    sel_gaia = ", ".join([f"g.{c}" for c in gaia_cols])

    def as_table(chunk: pd.DataFrame) -> Table:
        """Convert a chunk to an astropy Table with precomputed bounding box.

        The bounding box columns (ra_min/ra_max/dec_min/dec_max) are computed
        in Python so that the ADQL JOIN only uses simple column references.
        This avoids a server-side Java NPE that occurs when COS(RADIANS(u.dec))
        is evaluated inside a JOIN ON clause against an uploaded table
        (ESA Gaia TAP limitation, confirmed April 2026).
        """
        ra = chunk["ra"].astype("float64").values
        dec = chunk["dec"].astype("float64").values
        # RA padding scaled by 1/cos(dec) — clamp dec to avoid division by zero
        ra_pad = radius_deg / np.cos(np.radians(np.clip(dec, -89.9, 89.9)))
        return Table(
            [
                chunk["objid"].astype("int64").values,
                ra,
                dec,
                (ra - ra_pad),
                (ra + ra_pad),
                (dec - radius_deg),
                (dec + radius_deg),
            ],
            names=("objid", "ra", "dec", "ra_min", "ra_max", "dec_min", "dec_max"),
            dtype=(
                "int64",
                "float64",
                "float64",
                "float64",
                "float64",
                "float64",
                "float64",
            ),
        )

    def run_cm_on_chunk(chunk_df: pd.DataFrame) -> pd.DataFrame:
        """Run the TAP_UPLOAD query for one batch.

        The JOIN uses pre-uploaded bounding-box columns (ra_min/ra_max/dec_min/
        dec_max) rather than computing COS(RADIANS(u.dec)) inside the ADQL —
        that expression pattern causes a Java NPE on the ESA TAP server
        (``Cannot invoke List.iterator() because results is null``).

        The exact circular filter is applied in Python after fetching results,
        keeping the ADQL as simple as possible for maximum server compatibility.
        """
        t = as_table(chunk_df)

        # Simple column-reference BETWEEN — no per-row expressions in JOIN.
        # Post-fetch Python filter removes box corners outside the true radius.
        q = f"""
        SELECT
          u.objid,
          {sel_gaia},
          DISTANCE(POINT('ICRS', u.ra, u.dec),
                   POINT('ICRS', g.ra, g.dec)) * 3600.0
            AS match_dist_arcsec
        FROM TAP_UPLOAD.src AS u
        JOIN {gaia_table} AS g
          ON  g.dec BETWEEN u.dec_min AND u.dec_max
          AND g.ra  BETWEEN u.ra_min  AND u.ra_max
        """

        if debug:
            logger.info(
                "[TAP] Envoi du job asynchrone (lot de %d sources)…", len(chunk_df)
            )

        job = Gaia.launch_job_async(
            q,
            upload_resource=t,
            upload_table_name="src",
            dump_to_file=False,
        )

        if debug:
            logger.info("[TAP] Job créé — ID : %s", job.jobid)

        _poll_job(job, debug=debug)
        df_raw = job.get_results().to_pandas()

        # Exact circular filter in Python (removes box corners)
        if "match_dist_arcsec" in df_raw.columns and not df_raw.empty:
            df_raw = df_raw[
                pd.to_numeric(df_raw["match_dist_arcsec"], errors="coerce")
                <= radius_arcsec
            ]
        return df_raw

    # --- Batched loop with retry + exponential backoff ---
    all_rows: list[pd.DataFrame] = []
    batch_size = _INITIAL_BATCH_SIZE
    backoff_s = 10
    i = 0

    while i < len(df):
        j = min(i + batch_size, len(df))
        part = df.iloc[i:j]
        try:
            if debug:
                logger.info("[BULK] Lot %d–%d (taille=%d)…", i, j, len(part))
            all_rows.append(run_cm_on_chunk(part))
            i = j
            backoff_s = 10  # reset backoff after success

        except Exception as exc:
            logger.warning("[BULK] Échec lot %d:%d — %s", i, j, exc)

            if batch_size <= 50:
                logger.warning(
                    "[BULK] Lot ignoré (taille minimale atteinte). "
                    "Sources %d–%d perdues.",
                    i,
                    j,
                )
                i = j  # skip and move on
            else:
                batch_size //= 2
                logger.info(
                    "[BULK] Retry avec taille=%d dans %ds…", batch_size, backoff_s
                )
                time.sleep(backoff_s)
                backoff_s = min(backoff_s * 2, 120)

    df_cm = (
        pd.concat(all_rows, ignore_index=True)
        if all_rows
        else pd.DataFrame(columns=["objid"])
    )
    if df_cm.empty:
        return pd.DataFrame(columns=["objid"])

    # Best match per objid (smallest angular distance)
    df_cm.sort_values(["objid", "match_dist_arcsec"], inplace=True)
    df_best = df_cm.groupby("objid", as_index=False).first()

    # ── GSP-Phot supplement (astrophysical_parameters) ────────────────────
    # Fetches GSP-Phot parameters + evolutionary stage + luminosity,
    # which are not in gaia_source but in a separate table.
    ap_base = [
        "teff_gspphot",
        "logg_gspphot",
        "mh_gspphot",
        "distance_gspphot",
        "ag_gspphot",
        "ebpminrp_gspphot",
        "evolstage_gspphot",  # evolutionary stage (MS/RGB/HB/…)
        "lum_gspphot",  # luminosity in solar units
    ]
    ap_risky = ["radius_gspphot", "mass_gspphot", "age_gspphot"]
    ap_cols = ap_base + (ap_risky if include_risky else [])

    if ap_cols and not df_best.empty:
        unique_source_ids = df_best["source_id"].dropna().unique().astype("int64")

        if len(unique_source_ids) > 0:
            ids_tbl = Table(
                [unique_source_ids],
                names=("source_id",),
                dtype=("int64",),
            )
            sel_ap = ", ".join([f"ap.{c}" for c in ap_cols])
            q_ap = f"""
            SELECT i.source_id, {sel_ap}
            FROM TAP_UPLOAD.ids_tmp AS i
            LEFT JOIN gaiadr3.astrophysical_parameters AS ap
                   ON ap.source_id = i.source_id
            """
            try:
                if debug:
                    logger.info(
                        "[TAP] Requête GSP-Phot supplement (%d sources)…",
                        len(unique_source_ids),
                    )
                job_ap = Gaia.launch_job_async(
                    q_ap,
                    upload_resource=ids_tbl,
                    upload_table_name="ids_tmp",
                    dump_to_file=False,
                )
                _poll_job(job_ap, debug=debug)
                df_ap = job_ap.get_results().to_pandas()
                if not df_ap.empty:
                    df_best = df_best.merge(df_ap, on="source_id", how="left")
            except Exception as exc:
                logger.warning(
                    "[TAP] GSP-Phot supplement échoué — colonnes ap absentes. (%s)", exc
                )

    # Realign on df_src (preserve order and presence)
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


# ──── CDS XMatch + GSP-Phot supplement (primary strategy, Apr 2026+) ────────


def _gaia_supplement_by_ids(
    source_ids: "np.ndarray",
    include_risky: bool,
    debug: bool = False,
) -> pd.DataFrame:
    """Fetch GSP-Phot + extra gaia_source columns using ``source_id IN (…)``.

    This does NOT use TAP_UPLOAD — it issues regular async ADQL queries with
    an explicit IN() list, which are unaffected by the ESA TAP_UPLOAD outage.

    Columns fetched from ``gaiadr3.gaia_source``:
        radial_velocity, astrometric_excess_noise

    Columns fetched from ``gaiadr3.astrophysical_parameters``:
        teff_gspphot, logg_gspphot, mh_gspphot, distance_gspphot,
        ag_gspphot, ebpminrp_gspphot, evolstage_gspphot, lum_gspphot
        (+ radius/mass/age if include_risky=True)

    Parameters
    ----------
    source_ids : array-like of int64
        Unique Gaia DR3 source identifiers.
    include_risky : bool
        Whether to fetch radius_gspphot, mass_gspphot, age_gspphot.
    debug : bool
        Emit INFO-level log messages.

    Returns
    -------
    pd.DataFrame
        One row per source_id; NaN where parameters are unavailable.
    """
    ap_base = [
        "teff_gspphot",
        "logg_gspphot",
        "mh_gspphot",
        "distance_gspphot",
        "ag_gspphot",
        "ebpminrp_gspphot",
        "evolstage_gspphot",
        "lum_gspphot",
    ]
    ap_cols = ap_base + (
        ["radius_gspphot", "mass_gspphot", "age_gspphot"] if include_risky else []
    )
    sel_ap = ", ".join(f"ap.{c}" for c in ap_cols)

    ids_list = [int(x) for x in source_ids if not pd.isna(x)]
    if not ids_list:
        return pd.DataFrame(columns=["source_id"])

    all_rows: list[pd.DataFrame] = []
    tap_is_broken = False  # short-circuit: bail after first 500 error

    for i in range(0, len(ids_list), _GSPPHOT_BATCH_SIZE):
        if tap_is_broken:
            break
        batch = ids_list[i : i + _GSPPHOT_BATCH_SIZE]
        ids_str = ", ".join(str(x) for x in batch)

        q = f"""
        SELECT
          gs.source_id,
          gs.radial_velocity,
          gs.astrometric_excess_noise,
          {sel_ap}
        FROM gaiadr3.gaia_source AS gs
        LEFT JOIN gaiadr3.astrophysical_parameters AS ap
               ON gs.source_id = ap.source_id
        WHERE gs.source_id IN ({ids_str})
        """
        try:
            if debug:
                logger.info(
                    "[GSP-Phot] Requête lot %d–%d (%d sources)…",
                    i,
                    i + len(batch),
                    len(batch),
                )
            job = Gaia.launch_job_async(q, dump_to_file=False)
            _poll_job(job, debug=debug)
            all_rows.append(job.get_results().to_pandas())
        except Exception as exc:
            logger.warning("[GSP-Phot] Échec lot %d–%d : %s", i, i + len(batch), exc)
            tap_is_broken = True  # ESA TAP indisponible — skip remaining batches
            logger.warning(
                "[GSP-Phot] ESA TAP indisponible — supplement annulé. "
                "Teff/logg/Dist déjà fournis par CDS XMatch."
            )

    if all_rows:
        return pd.concat(all_rows, ignore_index=True)
    return pd.DataFrame(columns=["source_id"])


def _xmatch_crossmatch(
    df_src: pd.DataFrame,
    radius_arcsec: float,
    include_risky: bool,
    debug: bool = False,
) -> pd.DataFrame:
    """Cross-match via the CDS X-Match service against Gaia DR3.

    Uses ``astroquery.xmatch.XMatch.query`` against the VizieR Gaia DR3
    catalog (``I/355/gaiadr3``).  This bypasses ESA TAP_UPLOAD entirely,
    which has been systematically broken since at least April 2026.

    After the initial match, GSP-Phot parameters and extra gaia_source
    columns (radial_velocity, astrometric_excess_noise) are fetched via a
    regular TAP query on ``source_id IN (…)`` — no upload required.

    Notes
    -----
    - ``angDist`` returned by XMatch is already in arcseconds.
    - VizieR column names are remapped via ``_XMATCH_COL_MAP``.
    - Non-matched sources produce a row with ``objid`` only (all Gaia cols NaN).
    - Best match per ``objid`` is the one with smallest ``angDist``.
    """
    from astroquery.xmatch import XMatch  # lazy import (not always installed)

    df = df_src.loc[
        df_src["ra"].notna() & df_src["dec"].notna(), ["objid", "ra", "dec"]
    ].copy()
    df["ra"] = pd.to_numeric(df["ra"], errors="coerce")
    df["dec"] = pd.to_numeric(df["dec"], errors="coerce")
    df = df.dropna(subset=["ra", "dec"])
    if df.empty:
        return pd.DataFrame(columns=["objid"])

    upload_tbl = Table(
        [
            df["objid"].astype("int64").values,
            df["ra"].astype("float64").values,
            df["dec"].astype("float64").values,
        ],
        names=("objid", "ra", "dec"),
        dtype=("int64", "float64", "float64"),
    )

    if debug:
        logger.info(
            '[XMATCH] Cross-match CDS contre %s (rayon=%.2f")…',
            _VIZIER_GAIA_DR3,
            radius_arcsec,
        )

    matched_tbl = XMatch.query(
        cat1=upload_tbl,
        cat2=_VIZIER_GAIA_DR3,
        max_distance=radius_arcsec * u.arcsec,
        colRA1="ra",
        colDec1="dec",
    )

    if matched_tbl is None or len(matched_tbl) == 0:
        logger.info("[XMATCH] Aucun objet matché.")
        return pd.DataFrame(columns=["objid"])

    matched = matched_tbl.to_pandas()

    # ── Rename VizieR columns to internal names ──────────────────────────────
    rename = {
        k: v
        for k, v in _XMATCH_COL_MAP.items()
        if v is not None and k in matched.columns
    }
    drop = [k for k, v in _XMATCH_COL_MAP.items() if v is None and k in matched.columns]
    matched = matched.rename(columns=rename).drop(columns=drop, errors="ignore")

    # Keep best match per objid (smallest angular distance)
    if "match_dist_arcsec" in matched.columns:
        matched["match_dist_arcsec"] = pd.to_numeric(
            matched["match_dist_arcsec"], errors="coerce"
        )
        matched = (
            matched.sort_values(["objid", "match_dist_arcsec"])
            .groupby("objid", as_index=False)
            .first()
        )

    if debug:
        n_matched = (
            matched["source_id"].notna().sum() if "source_id" in matched.columns else 0
        )
        logger.info("[XMATCH] %d sources matchées (sur %d).", n_matched, len(df))

    # ── GSP-Phot + extra gaia_source columns (no TAP_UPLOAD needed) ──────────
    # Note: Since Apr 2026 ESA TAP is broken even for regular queries.
    # Most GSP-Phot columns (Teff, logg, Dist, AG) are already present in
    # the XMatch result via VizieR I/355/gaiadr3 and mapped above.
    # The supplement only adds mh_gspphot / evolstage / lum if TAP recovers.
    if "source_id" in matched.columns:
        unique_ids = matched["source_id"].dropna().astype("int64").unique()
        if len(unique_ids) > 0:
            try:
                df_supp = _gaia_supplement_by_ids(
                    unique_ids, include_risky, debug=debug
                )
                if not df_supp.empty:
                    existing = [
                        c
                        for c in df_supp.columns
                        if c != "source_id" and c in matched.columns
                    ]
                    df_supp = df_supp.drop(columns=existing, errors="ignore")
                    matched = matched.merge(df_supp, on="source_id", how="left")
                    if debug:
                        logger.info("[XMATCH] Supplement GSP-Phot TAP ajouté.")
            except Exception as exc:
                # ESA TAP broken — XMatch already provides Teff/logg/Dist/AG
                if debug:
                    logger.warning(
                        "[XMATCH] Supplement GSP-Phot TAP ignoré (ESA TAP indisponible) : %s",
                        exc,
                    )

    # ── Realign on df_src (preserve row order; NaN for unmatched) ────────────
    best_map = matched.set_index("objid") if not matched.empty else pd.DataFrame()
    aligned = []
    for _, row in df_src.iterrows():
        oid = int(row.objid)
        if not matched.empty and oid in best_map.index:
            s = best_map.loc[oid]
            if isinstance(s, pd.DataFrame):
                s = s.iloc[0]  # safety: take first if duplicates remain
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
        else:
            aligned.append({"objid": oid})

    return pd.DataFrame(aligned)


# ---------------------------- Mode CONE (fallback) ---------------------------


def _cone_crossmatch(
    df_src: pd.DataFrame, radius_arcsec: float, include_risky: bool
) -> pd.DataFrame:
    """Star-by-star cross-match via ``Gaia.cone_search_async`` (slow fallback)."""
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
        except Exception as exc:
            logger.debug("[CONE] Échec pour objid=%d : %s", objid, exc)
            out.append({"objid": objid})
    return pd.DataFrame(out)


# ------------------------------ Derived columns ------------------------------


def _add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Compute derived astrophysical columns from cross-matched Gaia data.

    Columns added
    -------------
    parallax_snr : float
        Signal-to-noise ratio of the parallax measurement
        (``parallax / parallax_error``).  NaN when either value is
        missing or ``parallax_error`` is zero.

    M_G : float
        Absolute G-band magnitude computed from the parallax distance
        modulus: ``M_G = G + 5*log10(parallax_mas/1000) + 5``.
        Only populated where ``parallax > 0`` and ``parallax_snr >= 5``
        (quality threshold consistent with Gaia DR3 best practices).
        NaN otherwise.

    Parameters
    ----------
    df : pd.DataFrame
        Merged catalog (output of the cross-match + RUWE filter).

    Returns
    -------
    pd.DataFrame
        Same DataFrame with new columns appended in-place.
    """
    # parallax_snr
    if "parallax" in df.columns and "parallax_error" in df.columns:
        plx = pd.to_numeric(df["parallax"], errors="coerce")
        plx_err = pd.to_numeric(df["parallax_error"], errors="coerce")
        df["parallax_snr"] = np.where(
            plx_err.notna() & (plx_err != 0),
            plx / plx_err,
            np.nan,
        )
    else:
        df["parallax_snr"] = np.nan

    # M_G — absolute magnitude via parallax distance modulus
    if "phot_g_mean_mag" in df.columns and "parallax" in df.columns:
        g_mag = pd.to_numeric(df["phot_g_mean_mag"], errors="coerce")
        plx = pd.to_numeric(df["parallax"], errors="coerce")
        snr = df.get("parallax_snr", pd.Series(np.nan, index=df.index))
        snr = pd.to_numeric(snr, errors="coerce")

        good = plx.notna() & (plx > 0) & snr.notna() & (snr >= 5.0)
        df["M_G"] = np.where(
            good,
            g_mag + 5.0 * np.log10(plx / 1000.0) + 5.0,
            np.nan,
        )
    else:
        df["M_G"] = np.nan

    return df


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
    verbose: bool = True,
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
    mode : {'auto', 'xmatch', 'bulk', 'cone'}, default='auto'
        Cross-match strategy.  ``auto`` tries xmatch → bulk → cone in order.
        ``xmatch`` uses the CDS X-Match service (recommended; no TAP_UPLOAD).
        ``bulk`` uses ESA TAP_UPLOAD (broken since Apr 2026; legacy fallback).
        ``cone`` does individual star-by-star queries (slow but always works).
    gaia_table : str, default=GAIA_MAIN
        Primary Gaia table to join (e.g. ``gaiadr3.gaia_source``).
    gaia_user, gaia_pass : str or None
        Optional credentials for ``Gaia.login`` (ignored on failure).
    verbose : bool, default=True
        Enable detailed logging.

    Returns
    -------
    (df_merged, stats) : (DataFrame, dict)
        Enriched DataFrame (also saved to disk) and match statistics.
        ``stats`` keys: ``matched``, ``total``, ``match_rate_pct``,
        ``mode_used``, ``ruwe_max``, ``n_with_M_G``, ``n_with_teff``.
    """
    if verbose:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s  %(levelname)-8s %(message)s",
            datefmt="%H:%M:%S",
        )

    # (optional) Gaia authentication
    if gaia_user and gaia_pass:
        try:
            Gaia.login(user=gaia_user, password=gaia_pass)
        except Exception as exc:
            logger.warning("Gaia login échoué (non bloquant) : %s", exc)

    # Reuse existing enriched file if available
    if os.path.exists(output_catalog_path) and not overwrite:
        logger.info("Fichier enrichi existant trouvé — chargement direct.")
        return pd.read_csv(output_catalog_path, sep=","), {"skipped": True}

    # Prepare and validate source data
    df_src = _validate_catalog(input_catalog_df.copy())
    df_src = _ensure_numeric_radec(df_src)

    # 1) Gaia cross-match
    mode_used = "none"
    df_gaia = pd.DataFrame()

    # xmatch (CDS) — primary strategy, bypasses broken ESA TAP_UPLOAD
    if mode in ("auto", "xmatch"):
        try:
            logger.info("Cross-match en mode 'xmatch' (CDS X-Match + TAP GSP-Phot)…")
            df_gaia = _xmatch_crossmatch(
                df_src, search_radius_arcsec, include_risky, debug=verbose
            )
            mode_used = "xmatch"
        except Exception as exc:
            logger.warning("Mode 'xmatch' échoué : %s", exc)
            if mode == "xmatch":
                raise
            df_gaia = pd.DataFrame()

    # bulk (TAP_UPLOAD) — legacy fallback; may fail if ESA TAP_UPLOAD is broken
    if df_gaia.empty and mode in ("auto", "bulk"):
        try:
            logger.info("Cross-match en mode 'bulk' (TAP_UPLOAD + BETWEEN box)…")
            df_gaia = _bulk_crossmatch(
                df_src, search_radius_arcsec, gaia_table, include_risky, debug=verbose
            )
            mode_used = "bulk"
        except Exception as exc:
            logger.warning("Mode 'bulk' échoué : %s", exc)
            if mode == "bulk":
                raise
            df_gaia = pd.DataFrame()

    # cone — individual queries, always works but very slow for large catalogs
    if df_gaia.empty and mode in ("auto", "cone"):
        logger.info("Cross-match en mode 'cone' (étoile par étoile)…")
        df_gaia = _cone_crossmatch(df_src, search_radius_arcsec, include_risky)
        mode_used = "cone"

    # 2) RUWE filter (soft nullification — keeps row, sets Gaia cols to NaN)
    ruwe_thr = None if ruwe_max is None else float(np.atleast_1d(ruwe_max)[0])
    if ("ruwe" in df_gaia.columns) and (ruwe_thr is not None):
        ruwe_vals = pd.to_numeric(df_gaia["ruwe"], errors="coerce")
        bad = (ruwe_vals.notna() & (ruwe_vals >= ruwe_thr)).fillna(False)
        cols_to_null = [c for c in df_gaia.columns if c != "objid"]
        df_gaia.loc[bad, cols_to_null] = pd.NA
        n_ruwe_nulled = int(bad.sum())
        logger.info("RUWE ≥ %.1f : %d sources nullifiées.", ruwe_thr, n_ruwe_nulled)

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
        "lum_gspphot",
        "teff_gspphot_lo",
        "teff_gspphot_hi",
        "distance_gspphot_lo",
        "distance_gspphot_hi",
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
    if "evolstage_gspphot" in df_merged.columns:
        df_merged["evolstage_gspphot"] = df_merged["evolstage_gspphot"].astype("string")

    # 4) Derived columns (parallax_snr, M_G)
    df_merged = _add_derived_columns(df_merged)

    # 5) Write to disk
    df_merged.to_csv(output_catalog_path, index=False, sep=",")
    logger.info("Catalogue enrichi sauvegardé → %s", output_catalog_path)

    # 6) Statistics
    n_total = len(df_src)
    n_matched = (
        int(df_merged["source_id"].notna().sum()) if "source_id" in df_merged else 0
    )
    n_M_G = int(df_merged["M_G"].notna().sum()) if "M_G" in df_merged else 0
    n_teff = (
        int(df_merged["teff_gspphot"].notna().sum())
        if "teff_gspphot" in df_merged
        else 0
    )

    stats = {
        "matched": n_matched,
        "total": int(n_total),
        "match_rate_pct": round(100 * n_matched / max(1, n_total), 1),
        "mode_used": mode_used,
        "ruwe_max": ruwe_max,
        "n_with_M_G": n_M_G,
        "n_with_teff": n_teff,
    }

    logger.info(
        "Statistiques : %d/%d matchés (%.1f%%) | %d avec M_G | %d avec T_eff",
        n_matched,
        n_total,
        stats["match_rate_pct"],
        n_M_G,
        n_teff,
    )

    return df_merged, stats
