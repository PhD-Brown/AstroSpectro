"""Replay M3.1 validation or targeted source-copy counter-experiments.

The default report goes to ignored test-results; publishing a frozen report
requires an explicit --output. No source in src/ is ever written.
"""

import argparse
import ast
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
BASE = "b4018074269011e5fbec232df068b8d3326a4e8d"
SOURCE = ROOT / "src/tools/visualizer.py"
GREEN = "tests/test_tools_visualizer.py"
RED = "audit/proofs_m31_visualizer.py"


def git(*args):
    return subprocess.check_output(
        ["git", *args], cwd=ROOT, text=True, encoding="utf-8"
    ).strip()


def hashes():
    paths = [
        SOURCE,
        ROOT / GREEN,
        ROOT / RED,
        ROOT / "tests/m31_visualizer_contracts.py",
        Path(__file__),
    ]
    paths += [
        ROOT / p
        for p in [
            "src/pipeline/preprocessor.py",
            "src/pipeline/peak_detector.py",
            "src/pipeline/feature_engineering.py",
            "src/pipeline/classifier.py",
            "src/tools/dataset_builder.py",
            "src/tools/generate_catalog_from_fits.py",
            "src/utils.py",
        ]
    ]
    return {
        p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in paths
    }


def inventory(source):
    tree = ast.parse(source)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
    return {
        "imports": [
            ast.unparse(n)
            for n in ast.walk(tree)
            if isinstance(n, (ast.Import, ast.ImportFrom))
        ],
        "methods": [
            {
                "name": n.name,
                "line": n.lineno,
                "end_line": n.end_lineno,
                "signature": ast.unparse(n.args),
                "self_calls": sorted(
                    {
                        call.func.attr
                        for call in ast.walk(n)
                        if isinstance(call, ast.Call)
                        and isinstance(call.func, ast.Attribute)
                        and isinstance(call.func.value, ast.Name)
                        and call.func.value.id == "self"
                    }
                ),
                "handlers": [
                    ast.unparse(h.type) if h.type else "bare"
                    for h in ast.walk(n)
                    if isinstance(h, ast.ExceptHandler)
                ],
                "io_and_effect_calls": sorted(
                    {
                        ast.unparse(call.func)
                        for call in ast.walk(n)
                        if isinstance(call, ast.Call)
                        and any(
                            key in ast.unparse(call.func)
                            for key in [
                                "open",
                                "read",
                                "save",
                                "to_csv",
                                "to_latex",
                                "makedirs",
                                "seed",
                                "style",
                                "display",
                                "load",
                                "log_analyses",
                            ]
                        )
                    }
                ),
            }
            for n in cls.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        ],
    }


def run(command, *, env=None):
    started = time.monotonic()
    result = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return {
        "command": command,
        "exit_code": result.returncode,
        "seconds": round(time.monotonic() - started, 2),
        "output": result.stdout,
    }


def pytest_run(selection, xml, *, candidate=None):
    env = os.environ.copy()
    env.update(
        PYTHONPATH=str(ROOT / "src"),
        MPLBACKEND="Agg",
        PYTHONHASHSEED="0",
        PYTHONIOENCODING="utf-8",
    )
    env.pop("M31_VISUALIZER_SOURCE", None)
    if candidate:
        env["M31_VISUALIZER_SOURCE"] = str(candidate)
    result = run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "--tb=short",
            "--disable-warnings",
            f"--junitxml={xml}",
            *selection,
        ],
        env=env,
    )
    if xml.exists():
        tree = ET.parse(xml)
        cases = tree.findall(".//testcase")
        failures = tree.findall(".//failure")
        errors = tree.findall(".//error")
        skipped = tree.findall(".//skipped")
        result["counts"] = {
            "collected": len(cases),
            "passed": len(cases) - len(failures) - len(errors) - len(skipped),
            "failed": len(failures),
            "errors": len(errors),
            "skipped": len(skipped),
        }
        result["failures"] = [
            {
                "name": c.get("name"),
                # Pytest's xunit2 writer omits the XML type attribute.
                "type": c.find("failure").get("type")
                or c.find("failure").get("message", "").split(":", 1)[0],
                "message": c.find("failure").get("message"),
            }
            for c in cases
            if c.find("failure") is not None
        ]
    return result


def replace(source, old, new):
    assert old in source, f"Candidate site absent: {old[:60]}"
    return source.replace(old, new)


def local_style(source):
    lines = source.splitlines(keepends=True)
    cls = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef))
    method = next(
        n for n in cls.body if getattr(n, "name", "") == "_plot_spectrum_analysis"
    )
    start = method.body[1].lineno - 1  # leave docstring outside context
    end = method.end_lineno
    lines[start:end] = ["        with plt.rc_context():\n"] + [
        "    " + line if line.strip() else line for line in lines[start:end]
    ]
    return "".join(lines)


def adversarial(report):
    source = SOURCE.read_text(encoding="utf-8")
    # Each mutant is paired with its passing original selection; failures
    # must be assertion failures, without setup/import/runtime errors.
    mutants = [
        (
            "M01_wavelength_pair",
            "finite_gaussian",
            "peak_wl = wavelength[peak_idx]",
            "peak_wl = wavelength[peak_idx] + 1.0",
        ),
        (
            "M02_normalization",
            "comparison_scale_invariance",
            "y = flux + (i * offset if offset else 0.0)",
            "y = flux * 2 + (i * offset if offset else 0.0)",
        ),
        (
            "M03_population_counts",
            "subclass_raw_counts",
            "values = counts\n",
            "values = counts * 2\n",
        ),
        (
            "M04_shap_signed_mean",
            "binary_shap_mean",
            "vals = np.abs(vals)",
            "vals = vals",
        ),
        (
            "M05_column_alignment",
            "prepare_flat",
            "X = X.reindex(columns=used_cols)",
            "X = X.reindex(columns=used_cols[::-1])",
        ),
        (
            "M06_feature_labels",
            "model_inspector_selected",
            "names_sorted = np.array(feature_names_for_plot)[order]",
            "names_sorted = np.array(feature_names_for_plot)[order[::-1]]",
        ),
        (
            "M07_zero_denominator",
            "zero_statistics",
            "zero_count / len(df)",
            "zero_count / (len(df) + 1)",
        ),
        (
            "M08_declination",
            "sky_artists",
            "y = np.radians(dec_deg)",
            "y = np.radians(-dec_deg)",
        ),
    ]
    remedies = [
        (
            "A01_circular_RA",
            "ra_wrap_centroid",
            'ra=("ra", "mean")',
            'ra=("ra", lambda s: np.degrees(np.arctan2(np.sin(np.radians(s)).mean(), np.cos(np.radians(s)).mean())) % 360)',
        ),
        (
            "A02_tick_labels",
            "sky_ra_tick",
            "((L % 360) / 15)",
            "(((180 - L) % 360) / 15)",
        ),
        (
            "A03_total_population",
            "truncated_percentages",
            "total = counts.sum()",
            "total = len(df)",
        ),
        (
            "A04_disclosed_conditional",
            "truncated_percentages",
            'ax.set_title("Spectral Sub-class Distribution")',
            'ax.set_title("Top 3 classes: 12 of 15 examples; 3 excluded; conditional frequency")',
        ),
        (
            "A05_shap_axes",
            "multiclass_shap",
            "vals = np.mean(np.abs(vals), axis=1)",
            "vals = np.mean(np.abs(vals), axis=2)",
        ),
        (
            "A06_selector_support",
            "selected_feature_names_derive",
            "or used_cols[: Xt.shape[1]]",
            "or list(np.asarray(used_cols)[sel.get_support()])",
        ),
        (
            "A07_uncertainty_scale",
            "normalized_flux_uncertainties",
            "sigma = safe_sigma_from_invvar(invvar)  # helper utils.py",
            "sigma = safe_sigma_from_invvar(invvar) / np.median(flux)  # normalized sigma",
        ),
        (
            "A08_actual_prep",
            "actual_classifier_prep",
            "Xt = X.values",
            'Xt = pipe.named_steps["prep"].transform(X) if "prep" in pipe.named_steps else X.values',
        ),
        (
            "A09_no_duplicate_SHAP",
            "computed_once",
            '        # 4) SHAP values (uniform API: direct call)\n        with warnings.catch_warnings():\n            warnings.filterwarnings("ignore", category=RuntimeWarning)\n            sv = explainer(Xt)\n',
            "",
        ),
        (
            "A10_missing_stays_nan",
            "missing_measurement",
            "df[c] = 0.0",
            "df[c] = np.nan",
        ),
        (
            "A12_absorption_profile",
            "absorption_fwhm_export",
            "gaussian_fwhm(spec_full, regions=region)",
            "gaussian_fwhm(Spectrum(spectral_axis=wavelength*u.AA, flux=(1-flux_norm)*u.adu), regions=region)",
        ),
        (
            "A13_fresh_cache",
            "added_backdated_fits",
            "if os.path.exists(catalog_path) and not force_regenerate:",
            "if os.path.exists(catalog_path) and not force_regenerate and False:",
        ),
        (
            "A14_relative_identity",
            "duplicate_basename_legend",
            "label=os.path.basename(rel_path)",
            "label=rel_path",
        ),
        (
            "A15_explicit_catalog_conflict",
            "duplicate_conflicting_catalog",
            'labels = pd.Series(df_cat["subclass"].values, index=key_series).to_dict()',
            'if key_series.duplicated().any():\n                print("Conflicting catalogue identities")\n                return {}\n            labels = pd.Series(df_cat["subclass"].values, index=key_series).to_dict()',
        ),
    ]
    results = []
    with tempfile.TemporaryDirectory(prefix="m31-candidates-") as temp:
        folder = Path(temp)
        for name, expression, old, new in mutants:
            reference = pytest_run(
                [GREEN, "-k", expression], folder / f"{name}-base.xml"
            )
            candidate = folder / f"{name}.py"
            candidate.write_text(replace(source, old, new), encoding="utf-8")
            compile(candidate.read_text(encoding="utf-8"), str(candidate), "exec")
            outcome = pytest_run(
                [GREEN, "-k", expression], folder / f"{name}.xml", candidate=candidate
            )
            counts = outcome.get("counts", {})
            killed = (
                reference["exit_code"] == 0
                and reference["counts"]["passed"] > 0
                and counts.get("failed", 0) > 0
                and counts.get("errors") == 0
                and all(
                    f["type"] == "AssertionError" for f in outcome.get("failures", [])
                )
            )
            results.append(
                {
                    "id": name,
                    "status": "KILLED" if killed else "INCONCLUSIVE",
                    "site": old,
                    "replacement": new,
                    "baseline": reference,
                    "result": outcome,
                }
            )
            print(name, results[-1]["status"], flush=True)
        for name, expression, old, new in remedies:
            candidate = folder / f"{name}.py"
            candidate.write_text(replace(source, old, new), encoding="utf-8")
            outcome = pytest_run(
                [RED, "-k", expression], folder / f"{name}.xml", candidate=candidate
            )
            admitted = outcome["exit_code"] == 0 and outcome["counts"]["passed"] > 0
            results.append(
                {
                    "id": name,
                    "status": "ADMITTED" if admitted else "INCONCLUSIVE",
                    "site": old,
                    "replacement": new,
                    "result": outcome,
                }
            )
            print(name, results[-1]["status"], flush=True)
        candidate = folder / "A11_local_style.py"
        candidate.write_text(local_style(source), encoding="utf-8")
        outcome = pytest_run(
            [RED, "-k", "restores_global_matplotlib"],
            folder / "style.xml",
            candidate=candidate,
        )
        results.append(
            {
                "id": "A11_local_style",
                "status": "ADMITTED" if outcome["exit_code"] == 0 else "INCONCLUSIVE",
                "result": outcome,
            }
        )
        print(results[-1]["id"], results[-1]["status"], flush=True)
        candidate = folder / "A16_pixel_mapping.py"
        pixel = replace(
            source,
            "peak_wl = wavelength[peak_idx]",
            "peak_idx = np.flatnonzero(np.isfinite(wavelength) & np.isfinite(flux_norm))[peak_idx]\n            peak_wl = wavelength[peak_idx]",
        )
        pixel = replace(
            pixel,
            "peak_wavelengths = wavelength[peak_indices]",
            "peak_indices = np.flatnonzero(np.isfinite(wavelength) & np.isfinite(flux_norm))[peak_indices]\n            peak_wavelengths = wavelength[peak_indices]",
        )
        candidate.write_text(pixel, encoding="utf-8")
        outcome = pytest_run(
            [RED, "-k", "nonfinite_prefix"], folder / "pixels.xml", candidate=candidate
        )
        results.append(
            {
                "id": "A16_pixel_mapping",
                "status": "ADMITTED" if outcome["exit_code"] == 0 else "INCONCLUSIVE",
                "result": outcome,
            }
        )
        print(results[-1]["id"], results[-1]["status"], flush=True)
    report["adversarial"] = results


def validation(report):
    folder = ROOT / "test-results/m31"
    folder.mkdir(parents=True, exist_ok=True)
    checks = [
        ("targeted", [GREEN]),
        ("importlib", [GREEN, "--import-mode=importlib"]),
        ("integration", ["tests", "-m", "integration and not slow and not network"]),
        ("full", ["tests"]),
        ("red", [RED]),
    ]
    report["checks"] = {}
    for name, selection in checks:
        result = pytest_run(selection, folder / f"{name}.xml")
        report["checks"][name] = result
        print(name, result.get("counts"), result["exit_code"], flush=True)
    # Reverse only M3.1 nodes; avoid requiring a random-order pytest plugin.
    collect = run([sys.executable, "-m", "pytest", "--collect-only", "-q", GREEN])
    nodes = [
        line for line in collect["output"].splitlines() if line.startswith(GREEN + "::")
    ]
    report["checks"]["reverse"] = pytest_run(
        list(reversed(nodes)), folder / "reverse.xml"
    )
    for name, args in [
        (
            "black",
            [
                "-m",
                "black",
                "--check",
                "src",
                "tests",
                "scripts",
                "app.py",
                "audit/check_m31_visualizer.py",
                RED,
            ],
        ),
        (
            "ruff",
            [
                "-m",
                "ruff",
                "check",
                "src",
                "tests",
                "scripts",
                "app.py",
                "audit/check_m31_visualizer.py",
                RED,
            ],
        ),
        ("pip_check", ["-m", "pip", "check"]),
    ]:
        report["checks"][name] = run([sys.executable, *args])
        print(name, report["checks"][name]["exit_code"], flush=True)
    report["checks"]["diff_check"] = run(["git", "diff", "--check"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation", action="store_true")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "test-results/m31-replay.json"
    )
    args = parser.parse_args()
    before = hashes()
    report = {
        "base": BASE,
        "input_head": git("rev-parse", "HEAD"),
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "versions": {
            n: importlib.metadata.version(n)
            for n in [
                "numpy",
                "pandas",
                "astropy",
                "specutils",
                "scipy",
                "matplotlib",
                "plotly",
                "shap",
                "scikit-learn",
                "pytest",
                "black",
                "ruff",
            ]
        },
        "input_hashes": before,
        "source_blob": git("hash-object", "src/tools/visualizer.py"),
        "base_blob": git("rev-parse", f"{BASE}:src/tools/visualizer.py"),
        "inventory": inventory(SOURCE.read_text(encoding="utf-8")),
        "protected_diff": git(
            "diff",
            BASE,
            "--",
            "src",
            "app.py",
            "notebooks",
            "pyproject.toml",
            "requirements",
            ".github",
            ".pre-commit-config.yaml",
        ),
    }
    if args.validation:
        validation(report)
    else:
        adversarial(report)
    report["hashes_after"] = hashes()
    report["inputs_unchanged"] = before == report["hashes_after"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    assert report["inputs_unchanged"] and not report["protected_diff"]
    print(args.output, flush=True)


if __name__ == "__main__":
    main()
