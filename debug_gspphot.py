"""
debug_gspphot.py — Test toutes les stratégies disponibles pour récupérer
teff_gspphot, logg_gspphot, mh_gspphot, distance_gspphot depuis Gaia DR3.

Usage:
    python debug_gspphot.py

Stratégies testées (par ordre de priorité) :
    A. ESA TAP async  — WHERE source_id IN (...) [méthode actuelle, cassée?]
    B. ESA TAP sync   — launch_job() au lieu de launch_job_async()
    C. astroquery.vizier — accès direct via VizieR I/355/gaiadr3
    D. CDS XMatch sur I/355/astrophysical_parameters
    E. astroquery.gaia get_individual_spectra (dernière chance)
"""

import time
import warnings
import traceback

import pandas as pd
from astropy import units as u
from astropy.table import Table

warnings.filterwarnings("ignore")

# ── Vrais source_ids tirés du run 500 spectres ──────────────────────────────
# (issus de features_20260430T153406Z.csv)
TEST_IDS = [
    295790862918453888,  # STAR M0
    3808717766199279616,  # STAR G0
    872129294553809536,  # STAR M2
    4049757358668495872,  # ajout si dispo
]
# Coordonnées correspondantes (pour le XMatch test)
TEST_COORDS = [
    (24.189794, 27.413969),  # ra, dec  → M0
    (163.709407, 2.304877),  # G0
    (114.04205, 26.980255),  # M2
]

GSP_COLS = [
    "teff_gspphot",
    "logg_gspphot",
    "mh_gspphot",
    "distance_gspphot",
    "ag_gspphot",
    "ebpminrp_gspphot",
    "evolstage_gspphot",
    "lum_gspphot",
]

SEP = "─" * 70


def ok(msg):
    print(f"  ✓  {msg}")


def fail(msg):
    print(f"  ✗  {msg}")


def info(msg):
    print(f"     {msg}")


# ============================================================================
# A. ESA TAP async — approach actuelle (gaia_crossmatcher._gaia_supplement)
# ============================================================================
def test_A_tap_async():
    print(f"\n{SEP}")
    print("A. ESA TAP ASYNC — launch_job_async() + WHERE source_id IN (...)")
    print(SEP)
    from astroquery.gaia import Gaia

    ids_str = ", ".join(str(x) for x in TEST_IDS[:3])
    sel = ", ".join(f"ap.{c}" for c in GSP_COLS)
    q = f"""
    SELECT gs.source_id, gs.radial_velocity, gs.astrometric_excess_noise,
           {sel}
    FROM gaiadr3.gaia_source AS gs
    LEFT JOIN gaiadr3.astrophysical_parameters AS ap
           ON gs.source_id = ap.source_id
    WHERE gs.source_id IN ({ids_str})
    """
    try:
        t0 = time.time()
        job = Gaia.launch_job_async(q, dump_to_file=False, verbose=False)
        r = job.get_results().to_pandas()
        elapsed = time.time() - t0
        if r.empty:
            fail(f"Requête OK mais résultat vide ({elapsed:.1f}s)")
            return None
        ok(f"{len(r)} lignes en {elapsed:.1f}s")
        teff_ok = r["teff_gspphot"].notna().sum()
        info(f"teff_gspphot non-null : {teff_ok}/{len(r)}")
        info(r[["source_id", "teff_gspphot", "logg_gspphot", "mh_gspphot"]].to_string())
        return r
    except Exception as e:
        fail(str(e)[:120])
        return None


# ============================================================================
# B. ESA TAP sync — launch_job() (différent code path côté serveur)
# ============================================================================
def test_B_tap_sync():
    print(f"\n{SEP}")
    print("B. ESA TAP SYNC — launch_job() (synchrone, timeout=120s)")
    print(SEP)
    from astroquery.gaia import Gaia

    ids_str = ", ".join(str(x) for x in TEST_IDS[:3])
    sel = ", ".join(f"ap.{c}" for c in GSP_COLS)
    q = f"""
    SELECT gs.source_id, {sel}
    FROM gaiadr3.gaia_source AS gs
    LEFT JOIN gaiadr3.astrophysical_parameters AS ap
           ON gs.source_id = ap.source_id
    WHERE gs.source_id IN ({ids_str})
    """
    try:
        t0 = time.time()
        job = Gaia.launch_job(q, dump_to_file=False, verbose=False)
        r = job.get_results().to_pandas()
        elapsed = time.time() - t0
        if r.empty:
            fail(f"Requête OK mais résultat vide ({elapsed:.1f}s)")
            return None
        ok(f"{len(r)} lignes en {elapsed:.1f}s")
        info(r[["source_id", "teff_gspphot", "logg_gspphot", "mh_gspphot"]].to_string())
        return r
    except Exception as e:
        fail(str(e)[:120])
        return None


# ============================================================================
# C. astroquery.vizier — I/355/gaiadr3 par source_id
# ============================================================================
def test_C_vizier_by_sourceid():
    print(f"\n{SEP}")
    print("C. astroquery.Vizier — catalogue I/355/gaiadr3 par source_id (Sourcecolumn)")
    print(SEP)
    from astroquery.vizier import Vizier

    # VizieR Gaia DR3 — colonnes GSP-Phot disponibles dans la table principale
    v = Vizier(
        catalog="I/355/gaiadr3",
        columns=["Source", "Teff", "logg", "__M_H_", "Dist", "A0", "E_BP-RP_", "Lum*"],
        row_limit=20,
    )
    # On cherche par ra/dec (VizieR ne supporte pas les requêtes par source_id directement)
    results = []
    for ra, dec in TEST_COORDS:
        try:
            from astropy.coordinates import SkyCoord

            coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
            r = v.query_region(coord, radius=2 * u.arcsec)
            if r and len(r) > 0 and len(r[0]) > 0:
                df = r[0].to_pandas()
                results.append(df.iloc[0])
        except Exception as e:
            fail(f"({ra:.3f}, {dec:.3f}) → {str(e)[:80]}")

    if results:
        df_out = pd.DataFrame(results)
        ok(f"{len(df_out)} objets récupérés")
        info(df_out.to_string())
        return df_out
    else:
        fail("Aucun résultat")
        return None


# ============================================================================
# D. CDS XMatch → I/355/gaiadr3 (récupère les colonnes GSP-Phot VizieR)
# ============================================================================
def test_D_cds_xmatch_with_gspphot():
    print(f"\n{SEP}")
    print("D. CDS XMatch — vizier:I/355/gaiadr3 (inclut colonnes Teff VizieR?)")
    print(SEP)
    from astroquery.xmatch import XMatch

    upload = Table(
        [[1, 2, 3], [c[0] for c in TEST_COORDS], [c[1] for c in TEST_COORDS]],
        names=("objid", "ra", "dec"),
        dtype=("int64", "float64", "float64"),
    )
    try:
        t0 = time.time()
        r = XMatch.query(
            cat1=upload,
            cat2="vizier:I/355/gaiadr3",
            max_distance=2 * u.arcsec,
            colRA1="ra",
            colDec1="dec",
        )
        elapsed = time.time() - t0
        if r is None or len(r) == 0:
            fail(f"Aucun match ({elapsed:.1f}s)")
            return None
        df = r.to_pandas()
        ok(f"{len(df)} lignes en {elapsed:.1f}s")
        info(f"Colonnes disponibles ({len(df.columns)}) :")
        # Affiche toutes les colonnes qui ressemblent à GSP-Phot
        gsp_like = [
            c
            for c in df.columns
            if any(
                k in c.lower()
                for k in [
                    "teff",
                    "tef",
                    "logg",
                    "m_h",
                    "dist",
                    "lum",
                    "ag",
                    "a0",
                    "ebp",
                ]
            )
        ]
        if gsp_like:
            ok(f"Colonnes GSP-Phot trouvées : {gsp_like}")
            info(df[["objid"] + gsp_like].to_string())
        else:
            fail("Pas de colonnes GSP-Phot dans les résultats XMatch")
            info(f"Toutes les colonnes : {list(df.columns)}")
        return df
    except Exception as e:
        fail(str(e)[:120])
        traceback.print_exc()
        return None


# ============================================================================
# E. VizieR — table astrophysical_parameters (I/355/astrophysical_parameters?)
# ============================================================================
def test_E_vizier_astrophysical_params():
    print(f"\n{SEP}")
    print(
        "E. astroquery.Vizier — sous-table I/355/astrophysical_parameters (si disponible)"
    )
    print(SEP)
    from astroquery.vizier import Vizier
    from astropy.coordinates import SkyCoord

    # D'abord, lister les sous-tables disponibles dans I/355
    try:
        cats = Vizier.find_catalogs("Gaia DR3 astrophysical")
        info(f"Catalogues trouvés : {list(cats.keys())[:10]}")
    except Exception as e:
        info(f"find_catalogs échoué : {e}")

    # Essai direct sur gaiadr3/astrophysical_parameters via VizieR
    for cat_id in ["I/355/h5astroph", "I/355/astroph", "I/355/gaiafpr"]:
        try:
            v = Vizier(catalog=cat_id, row_limit=5)
            coord = SkyCoord(ra=24.19 * u.deg, dec=27.41 * u.deg, frame="icrs")
            r = v.query_region(coord, radius=5 * u.arcsec)
            if r and len(r) > 0:
                ok(
                    f"Table '{cat_id}' accessible ! Colonnes : {list(r[0].colnames)[:15]}"
                )
                return r[0].to_pandas()
            else:
                fail(f"'{cat_id}' → aucun résultat")
        except Exception as e:
            fail(f"'{cat_id}' → {str(e)[:80]}")

    return None


# ============================================================================
# RÉSUMÉ
# ============================================================================
def print_summary(results: dict):
    print(f"\n{'═' * 70}")
    print("RÉSUMÉ")
    print("═" * 70)
    for name, df in results.items():
        if df is not None and not (isinstance(df, pd.DataFrame) and df.empty):
            teff_col = next(
                (
                    c
                    for c in (df.columns if hasattr(df, "columns") else [])
                    if "teff" in c.lower() or c == "Teff"
                ),
                None,
            )
            n_teff = int(df[teff_col].notna().sum()) if teff_col else 0
            print(f"  ✓  {name:35s} → teff non-null : {n_teff}")
        else:
            print(f"  ✗  {name:35s} → ÉCHEC")

    # Recommandation
    working = [k for k, v in results.items() if v is not None]
    if working:
        print(f"\n→ Stratégie recommandée : {working[0]}")
    else:
        print(
            "\n→ Toutes les stratégies ont échoué — ESA/VizieR probablement en maintenance."
        )


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("  DEBUG GSP-Phot — Gaia DR3 teff_gspphot recovery test")
    print(f"  Source IDs testés : {TEST_IDS[:3]}")
    print("=" * 70)

    results = {}
    results["A. TAP async (launch_job_async)"] = test_A_tap_async()
    results["B. TAP sync  (launch_job)"] = test_B_tap_sync()
    results["C. VizieR par coord (I/355/gaiadr3)"] = test_C_vizier_by_sourceid()
    results["D. CDS XMatch (I/355/gaiadr3)"] = test_D_cds_xmatch_with_gspphot()
    results["E. VizieR astrophysical_params"] = test_E_vizier_astrophysical_params()

    print_summary(results)
