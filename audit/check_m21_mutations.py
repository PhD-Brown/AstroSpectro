"""Replay named review mutations in disposable copies, never production files.

Run from the repository with its test interpreter::

    python audit/check_m21_mutations.py --output test-results/m21-mutations.json

These are reconstructions from Claude's published descriptions, not his
unpublished 31-mutant script. Each candidate runs the complete green loader
suite with the repository's offline pytest configuration. A survivor is a
coverage observation, not automatically a production defect. The basename
mutation is expected to survive until the separate #53 correction.
"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET


MUTATIONS = [
    (
        "spectral_classes_ignored",
        "if classes is not None:\n            # Filtre de classes",
        "if False:\n            # Filtre de classes",
    ),
    (
        "spectral_snr_ignored",
        'if snr_min > 0 and "snr_r" in catalog.columns:',
        "if False:",
    ),
    (
        "sampling_head",
        "catalog.sample(n_spectra, random_state=self.random_state)",
        "catalog.head(n_spectra)",
    ),
    ("median_native", "med = np.nanmedian(flux_interp)", "med = np.nanmedian(flux)"),
    (
        "threshold_11",
        "valid = nan_frac_per_row < 0.20",
        "valid = nan_frac_per_row < 0.11",
    ),
    (
        "threshold_30",
        "valid = nan_frac_per_row < 0.20",
        "valid = nan_frac_per_row < 0.30",
    ),
    (
        "threshold_inclusive",
        "valid = nan_frac_per_row < 0.20",
        "valid = nan_frac_per_row <= 0.20",
    ),
    ("index_keep_last", "if p.name not in self._fits_index:", "if True:"),
    (
        "rejection_log_debug",
        'logger.info(\n                "Spectres rejetés',
        'logger.debug(\n                "Spectres rejetés',
    ),
    (
        "features_snr_strict",
        "df[df[snr_band] >= snr_min]",
        "df[df[snr_band] > snr_min]",
    ),
    (
        "spectral_snr_strict",
        'catalog[catalog["snr_r"] >= snr_min]',
        'catalog[catalog["snr_r"] > snr_min]',
    ),
    ("inner_join", 'how="left",', 'how="inner",'),
    (
        "column_nan_inclusive",
        "if nan_frac > nan_threshold:",
        "if nan_frac >= nan_threshold:",
    ),
    ("balance_with_replacement", "replace=False", "replace=True"),
    (
        "meta_index_not_reset",
        "meta = df[meta_cols].copy().reset_index(drop=True)",
        "meta = df[meta_cols].copy()",
    ),
    ("dropna_all_columns", "df.dropna(subset=feat_cols)", "df.dropna()"),
    (
        "inverse_identity",
        "return self.scaler_.inverse_transform(X_scaled)",
        "return X_scaled",
    ),
    (
        "wavelength_one_pixel_shift",
        'np.arange(len(flux)) * header["COEFF1"]',
        '(np.arange(len(flux)) + 1) * header["COEFF1"]',
    ),
    ("nearest_interpolation", 'kind="linear",', 'kind="nearest",'),
    ("extrapolate", "fill_value=np.nan,", 'fill_value="extrapolate",'),
    ("nonzero_median", "if med > 0:", "if med != 0:"),
    ("wrong_flux_row", "flux = data[0].astype(float)", "flux = data[1].astype(float)"),
    ("nan_fill_removed", "X = np.nan_to_num(X, nan=0.0)", "X = X.copy()"),
    ("catalog_mask_shifted", "catalog.iloc[valid]", "catalog.iloc[np.roll(valid, 1)]"),
    ("cache_ignored", "if self._fits_index is not None:", "if False:"),
    (
        "spectro_prefix_ignored",
        "if spectro_only and c.startswith(_NON_SPECTRO_PREFIXES):",
        "if False:",
    ),
    (
        "scaler_no_std",
        "self.scaler_ = StandardScaler()",
        "self.scaler_ = StandardScaler(with_std=False)",
    ),
    ("feature_order_reversed", "return feat_cols", "return feat_cols[::-1]"),
    (
        "read_warning_debug",
        'logger.warning("Erreur lecture %s : %s", path.name, exc)',
        'logger.debug("Erreur lecture %s : %s", path.name, exc)',
    ),
    (
        "boolean_not_numeric",
        "if not pd.api.types.is_numeric_dtype(df[c]):",
        "if not pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c]):",
    ),
]


def main():
    """Validate the baseline, then report detected and surviving mutations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    relative_paths = [
        "src/dimred/data_loader.py",
        "tests/conftest.py",
        "tests/test_dimred_data_loader.py",
        "pyproject.toml",
    ]
    report = {
        "input_sha256": {
            p: hashlib.sha256((root / p).read_bytes()).hexdigest()
            for p in relative_paths
        },
        "cases": [],
    }
    with tempfile.TemporaryDirectory(prefix="astrospectro-m21-mutation-") as temporary:
        sandbox = Path(temporary)
        for relative in relative_paths:
            target = sandbox / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(root / relative, target)
        source_path = sandbox / relative_paths[0]
        source = source_path.read_text(encoding="utf-8")
        for name, before, after in [("baseline", None, None)] + MUTATIONS:
            if before is not None and source.count(before) != 1:
                raise RuntimeError(f"{name}: expected exactly one replacement site")
            source_path.write_text(
                source if before is None else source.replace(before, after),
                encoding="utf-8",
            )
            # Avoid stale bytecode across same-length edits and fast repetitions.
            for cache in (sandbox / "src").rglob("__pycache__"):
                resolved_cache = cache.resolve()
                resolved_cache.relative_to(sandbox.resolve())
                shutil.rmtree(resolved_cache)
            junit = sandbox / "result.xml"
            command = [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "tests/test_dimred_data_loader.py",
                "--tb=no",
                f"--junitxml={junit}",
            ]
            result = subprocess.run(
                command,
                cwd=sandbox,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=120,
            )
            if result.returncode not in (0, 1) or not junit.exists():
                raise RuntimeError(
                    f"{name}: invalid test execution\n{result.stdout}\n{result.stderr}"
                )
            suite = ET.parse(junit).getroot().find("testsuite")
            if int(suite.get("errors", 0)) or int(suite.get("skipped", 0)):
                raise RuntimeError(
                    f"{name}: collection/fixture errors or unexpected skips"
                )
            failures = [
                c.get("name")
                for c in suite.findall("testcase")
                if c.find("failure") is not None
            ]
            if name == "baseline" and result.returncode:
                raise RuntimeError(f"Unmodified baseline failed: {failures}")
            status = (
                "PASS" if name == "baseline" else "KILLED" if failures else "SURVIVED"
            )
            report["cases"].append(
                {
                    "name": name,
                    "status": status,
                    "tests": int(suite.get("tests")),
                    "failures": failures,
                }
            )
            print(f"{name}: {status} ({len(failures)} failing cases)", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Report: {args.output}")


if __name__ == "__main__":
    main()
