import os
import gzip
import csv
from astropy.io import fits

def generate_catalog_from_fits(fits_paths, output_csv, verbose=True):
    """
    Génère un catalogue CSV enrichi à partir des headers FITS des spectres fournis.
    """

    # Définir toutes les colonnes que tu souhaites inclure
    fieldnames = [
        'fits_name', 'obsid', 'plan_id', 'mjd', 'class', 'subclass',
        'filename_original', 'author', 'data_version', 'date_creation',
        'telescope', 'longitude_site', 'latitude_site',
        'obs_date_utc', 'jd',
        'designation', 'ra', 'dec',
        'fiber_id', 'fiber_type', 'object_name', 'catalog_object_type',
        'magnitude_type', 'magnitude_u', 'magnitude_g', 'magnitude_r', 'magnitude_i', 'magnitude_z',
        'heliocentric_correction', 'radial_velocity_corr', 'seeing',
        'redshift', 'redshift_error', 'snr_u', 'snr_g', 'snr_r', 'snr_i', 'snr_z'
    ]

    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter='|')
        writer.writeheader()

        for path in fits_paths:
            try:
                with gzip.open(path, 'rb') as f:
                    with fits.open(f, memmap=False) as hdul:
                        hdr = hdul[0].header

                        entry = {
                            'fits_name': os.path.basename(path),
                            'obsid': hdr.get('OBSID', 'UNKNOWN'),
                            'plan_id': hdr.get('PLANID', 'UNKNOWN'),
                            'mjd': hdr.get('MJD', 'UNKNOWN'),
                            'class': hdr.get('CLASS', 'UNKNOWN'),
                            'subclass': hdr.get('SUBCLASS', 'UNKNOWN'),

                            # Informations Générales
                            'filename_original': hdr.get('FILENAME', 'UNKNOWN'),
                            'author': hdr.get('AUTHOR', 'UNKNOWN'),
                            'data_version': hdr.get('DATA_VRS', 'UNKNOWN'),
                            'date_creation': hdr.get('DATE', 'UNKNOWN'),

                            # Télescope
                            'telescope': hdr.get('TELESCOP', 'UNKNOWN'),
                            'longitude_site': hdr.get('LONGITUD', 'UNKNOWN'),
                            'latitude_site': hdr.get('LATITUDE', 'UNKNOWN'),

                            # Observation
                            'obs_date_utc': hdr.get('DATE-OBS', 'UNKNOWN'),
                            'jd': hdr.get('JD', hdr.get('MJD', 'UNKNOWN')),

                            # Position
                            'designation': hdr.get('DESIG', 'UNKNOWN'),
                            'ra': hdr.get('RA', 'UNKNOWN'),
                            'dec': hdr.get('DEC', 'UNKNOWN'),

                            # Fibre & objet
                            'fiber_id': hdr.get('FIBERID', 'UNKNOWN'),
                            'fiber_type': hdr.get('FIBERTYP', 'UNKNOWN'),
                            'object_name': hdr.get('NAME', 'UNKNOWN'),
                            'catalog_object_type': hdr.get('OBJTYPE', 'UNKNOWN'),

                            # Magnitudes
                            'magnitude_type': hdr.get('MAGTYPE', 'UNKNOWN'),
                            'magnitude_u': hdr.get('MAGU', 'UNKNOWN'),
                            'magnitude_g': hdr.get('MAGG', 'UNKNOWN'),
                            'magnitude_r': hdr.get('MAGR', 'UNKNOWN'),
                            'magnitude_i': hdr.get('MAGI', 'UNKNOWN'),
                            'magnitude_z': hdr.get('MAGZ', 'UNKNOWN'),

                            # Paramètres de réduction
                            'heliocentric_correction': hdr.get('HELIO', 'UNKNOWN'),
                            'radial_velocity_corr': hdr.get('VELDISP', 'UNKNOWN'),
                            'seeing': hdr.get('SEEING', 'UNKNOWN'),

                            # Analyse pipeline
                            'redshift': hdr.get('Z', 'UNKNOWN'),
                            'redshift_error': hdr.get('Z_ERR', 'UNKNOWN'),
                            'snr_u': hdr.get('SNRU', 'UNKNOWN'),
                            'snr_g': hdr.get('SNRG', 'UNKNOWN'),
                            'snr_r': hdr.get('SNRR', 'UNKNOWN'),
                            'snr_i': hdr.get('SNRI', 'UNKNOWN'),
                            'snr_z': hdr.get('SNRZ', 'UNKNOWN'),
                        }

                        writer.writerow(entry)
                        if verbose:
                            print(f"[OK] {entry['fits_name']} ajouté au catalogue.")
            except Exception as e:
                print(f"[ERREUR] Impossible de lire {path} : {e}")
