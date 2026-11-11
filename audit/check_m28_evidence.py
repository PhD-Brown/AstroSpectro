"""Replay M2.8 semantic mutants and admissible controls in disposable trees.

Only reached source edits and assertion failures with unchanged collection are
KILLED. Environment/import/setup failures are INCONCLUSIVE. Exact candidate
patches, traces, pytest output, JUnit and effective LF hashes are retained.
"""

import argparse
import ast
import difflib
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
SOURCE = "src/dimred/dimred_visualizer.py"
GREEN = "tests/test_dimred_visualizer.py"
PROOF = "tests/proofs_dimred_visualizer.py"

# name, method, unique old site, replacement, pytest selection
MUTANTS = [
    (
        "M01_color_row_roll",
        "plot_embedding",
        "mask = y == cls",
        "mask = np.roll(y == cls, 1)",
        "embedding_exact_observation",
    ),
    (
        "M02_variance_units",
        "plot_variance_explained",
        "ratio * 100",
        "ratio",
        "real_pca_variance",
    ),
    (
        "M03_best_is_max",
        "plot_training_history",
        "np.argmin(val_loss)",
        "np.argmax(val_loss)",
        "training_history_minimum",
    ),
    (
        "M04_family_median",
        "plot_ae_error_by_family",
        '.agg(["mean", "std", "count"])',
        '.agg(["median", "std", "count"]).rename(columns={"median": "mean"})',
        "error_family_mean",
    ),
    (
        "M05_ae_as_pca",
        "plot_ae_vs_pca",
        'row["mse_mean"], color=',
        'pca_df["mse_mean"].iloc[0], color=',
        "ae_pca_comparison",
    ),
    (
        "M06_identity_reconstruction",
        "plot_ae_reconstruction",
        "recon = X_recon[si]",
        "recon = X[si]",
        "reconstruction_drawn_rows",
    ),
    (
        "M07_hr_color_roll",
        "plot_hr_diagram_embedding",
        "c=z_coord[valid]",
        "c=np.roll(z_coord[valid], 1)",
        "hr_finite_triplets",
    ),
    (
        "M08_cluster_colors",
        "plot_hdbscan_clusters",
        "c=[color_map[cid]]",
        "c=[color_map[cluster_ids[-1]]]",
        "real_analyzer_state",
    ),
    (
        "M09_tsne_as_umap",
        "plot_trustworthiness",
        "X_sub, Zt_sub, n_neighbors=k",
        "X_sub, Zu_sub, n_neighbors=k",
        "real_sklearn_trustworthiness",
    ),
    (
        "M10_confidence_constant",
        "plot_xgboost_umap",
        "c=confidence",
        "c=np.ones(len(confidence))",
        "xgboost_classes_confidence",
    ),
    (
        "M11_synthesis_ae_as_pca",
        "plot_synthesis_grid",
        '("Autoencodeur", Z_ae,',
        '("Autoencodeur", scores_pca,',
        "synthesis_preserves",
    ),
    (
        "M12_stability_median",
        "plot_stability",
        "dists[1:].mean()",
        "np.median(dists[1:])",
        "stability_producer",
    ),
    (
        "M13_profile_sign_erased",
        "plot_hdbscan_feature_profiles",
        "vals = profile[top5].values",
        "vals = np.abs(profile[top5].values)",
        "profiles_signed_top",
    ),
    (
        "M14_density_wrong_population",
        "plot_ae_error_distribution",
        "density=True",
        "density=False",
        "histogram_densities",
    ),
    (
        "M15_kde_wrong_population",
        "plot_latent_kde",
        'star_mask = y == "STAR"',
        "star_mask = np.ones(len(y), dtype=bool)",
        "real_kde_reproducible",
    ),
    (
        "M16_export_blank_figure",
        "_save",
        'fig.savefig(path, bbox_inches="tight", dpi=self.dpi)',
        'plt.figure().savefig(path, bbox_inches="tight", dpi=self.dpi)',
        "real_export_readable",
    ),
]

CONTROLS = [
    (
        "C14_renderable_mathtext",
        "plot_eigenspectra",
        "                        nom,",
        "                        nom.replace(chr(92)*2, chr(92)),",
        "default_eigenspectrum",
    ),
    (
        "C01_equivalent_percent",
        "plot_variance_explained",
        "ratio * 100",
        "100 * ratio",
        "real_pca_variance",
    ),
    (
        "C02_alternative_export",
        "_save",
        'fig.savefig(path, bbox_inches="tight", dpi=self.dpi)',
        'fig.canvas.print_figure(path, bbox_inches="tight", dpi=self.dpi)',
        "real_export_readable",
    ),
    (
        "C03_alternative_palette",
        "plot_loadings_bar",
        '"#DD8452" if v < 0 else "#4C72B0"',
        '"#CC5500" if v < 0 else "#0055CC"',
        "signed_loadings",
    ),
    (
        "C04_explicit_alignment_refusal",
        "plot_ae_recon_by_spectral_type",
        "RECON_FEATURES = [",
        'if not (len(X) == len(X_recon) == len(spectral_types)):\n            raise RuntimeError("Observation alignment: incompatible lengths")\n        RECON_FEATURES = [',
        "reconstruction_cardinality",
    ),
    (
        "C05_gravity_units",
        "plot_hr_teff_regions",
        "G_HI - T_PAD * 2",
        "G_HI - G_PAD * 2",
        "hr_region_vertices",
    ),
    (
        "C06_count_omitted",
        "plot_hr_teff_regions",
        "43 019 étoiles",
        "population fournie",
        "hr_population_title",
    ),
    (
        "C07_small_errors_retained",
        "plot_ae_error_distribution_logscale",
        "max(vals.min(), 1e-4)",
        "max(vals.min(), np.finfo(float).tiny)",
        "log_histogram_keeps",
    ),
    (
        "C08_scalar_axes_supported",
        "plot_ae_embedding",
        "ax = axes[0]",
        "axes = np.atleast_1d(axes)\n        ax = axes[0]",
        "no_physical_columns and plot_ae_embedding",
    ),
    (
        "C09_zoom_scalar_axes_supported",
        "plot_latent_zoomed",
        "ax = axes[0]",
        "axes = np.atleast_1d(axes)\n        ax = axes[0]",
        "no_physical_columns and plot_latent_zoomed",
    ),
    (
        "C10_profile_single_slot",
        "plot_hdbscan_feature_profiles",
        "axes_flat = axes.flatten() if n_show > 1 else [axes]",
        "axes_flat = np.asarray(axes).ravel()",
        "cluster_profiles_unavailable and 1",
    ),
    (
        "C11_display_whole_trust_range",
        "plot_trustworthiness",
        "ax.set_ylim(0.75, 1.01)",
        "ax.set_ylim(0, 1.01)",
        "trustworthiness_axis_contains",
    ),
    (
        "C12_hide_unused_reconstructions",
        "plot_ae_reconstruction",
        "for i, (ax, si) in enumerate(zip(axes, idx)):",
        "for unused in axes[len(idx):]:\n            unused.set_visible(False)\n        for i, (ax, si) in enumerate(zip(axes, idx)):",
        "more_reconstructions or reconstruction_drawn_rows",
    ),
]

FALSE_REMEDIES = [
    (
        "F01_always_refuse_alignment",
        "plot_ae_recon_by_spectral_type",
        "RECON_FEATURES = [",
        'raise RuntimeError("Observation alignment unavailable")\n        RECON_FEATURES = [',
        "reconstruction_cardinality or spectral_type_means",
    ),
    (
        "F02_empty_hr",
        "plot_hr_teff_regions",
        "for region in REGIONS:",
        "REGIONS = []\n        for region in REGIONS:",
        "hr_region_vertices",
    ),
]

PLUGIN = """import json, os, sys
seen=set()
def trace(frame, event, arg):
    if frame.f_code.co_filename.replace("\\\\", "/").endswith("src/dimred/dimred_visualizer.py"):
        if event == "line": seen.add(frame.f_lineno)
        return trace
    return None
def pytest_sessionstart(session): sys.settrace(trace)
def pytest_sessionfinish(session, exitstatus):
    sys.settrace(None)
    with open(os.environ["M28_TRACE"], "w") as f: json.dump(sorted(seen), f)
"""


def digest(path):
    return hashlib.sha256(
        path.read_text(encoding="utf-8").replace("\r\n", "\n").encode()
    ).hexdigest()


def edit(source, method, old, new):
    tree = ast.parse(source)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
    node = next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method
    )
    lines = source.splitlines(keepends=True)
    body = "".join(lines[node.lineno - 1 : node.end_lineno])
    if body.count(old) != 1:
        raise ValueError(
            f"{method}: expected one edit site, got {body.count(old)}: {old}"
        )
    changed_line = node.lineno + body[: body.index(old)].count("\n")
    statements = [
        n
        for n in ast.walk(node)
        if isinstance(n, ast.stmt) and n.lineno <= changed_line <= n.end_lineno
    ]
    statement_line = min(statements, key=lambda n: n.end_lineno - n.lineno).lineno
    changed = (
        "".join(lines[: node.lineno - 1])
        + body.replace(old, new)
        + "".join(lines[node.end_lineno :])
    )
    ast.parse(changed)
    return changed, {
        statement_line,
        *range(changed_line, changed_line + new.count("\n") + 1),
    }


def run(tree, tests, selection, reverse=False):
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        *tests,
        "-p",
        "m28_trace",
        "-k",
        selection,
        "--tb=short",
        "--junitxml=result.xml",
    ]
    if reverse:
        with (tree / "m28_trace.py").open("a", encoding="utf-8") as stream:
            stream.write(
                "\ndef pytest_collection_modifyitems(items): items.reverse()\n"
            )
    env = dict(
        os.environ,
        MPLBACKEND="Agg",
        PYTHONHASHSEED="7" if reverse else "0",
        M28_TRACE=str(tree / "trace.json"),
        PYTHONPATH=str(tree) + os.pathsep + str(tree / "src"),
        PYTHONIOENCODING="utf-8",
    )
    result = subprocess.run(
        command,
        cwd=tree,
        env=env,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        timeout=180,
    )
    cases = []
    if (tree / "result.xml").exists():
        for case in ET.parse(tree / "result.xml").iter("testcase"):
            child = next(iter(case), None)
            cases.append(
                {
                    "id": case.attrib["classname"] + "::" + case.attrib["name"],
                    "status": "PASS" if child is None else child.tag.upper(),
                    "message": "" if child is None else (child.text or ""),
                }
            )
    seen = (
        json.loads((tree / "trace.json").read_text())
        if (tree / "trace.json").exists()
        else []
    )
    return {
        "command": command,
        "returncode": result.returncode,
        "cases": cases,
        "executed_lines": seen,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["mutations", "proofs", "reverse", "baseline"],
        default="mutations",
    )
    parser.add_argument("--output", default="test-results/m28-replay.json")
    parser.add_argument("--proofs-ref")
    parser.add_argument("--only", nargs="+")
    args = parser.parse_args()
    protected = [SOURCE, GREEN, "tests/conftest.py"]
    before = {p: digest(ROOT / p) for p in protected}
    source = (ROOT / SOURCE).read_text(encoding="utf-8")
    payload = {
        "active_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "python": sys.version,
        "tested_lf_sha256": before,
        "mode": args.mode,
        "runs": [],
    }
    plans = MUTANTS if args.mode == "mutations" else []
    plans += CONTROLS + FALSE_REMEDIES if args.mode == "proofs" else []
    if args.mode == "proofs":
        start = source.index("        zz_flat = ZZ.ravel()")
        stop = source.index("        params = [", start)
        old_levels = source[start:stop]
        mass_levels = """        center = Z_stars.mean(axis=0)
        scale = Z_stars.std(axis=0)
        qx = np.linspace(center[0]-8*scale[0], center[0]+8*scale[0], 401)
        qy = np.linspace(center[1]-8*scale[1], center[1]+8*scale[1], 401)
        a, b = np.meshgrid(qx, qy)
        density = kde(np.vstack([a.ravel(), b.ravel()]))
        ordered = np.sort(density)[::-1]
        mass = np.cumsum(ordered) / ordered.sum()
        levels = sorted([float(ordered[np.searchsorted(mass, p)]) for p in [.68, .95, .99]])

"""
        plans += [
            (
                "C13_integrated_kde_mass",
                "plot_latent_kde",
                old_levels,
                mass_levels,
                "kde_labeled_regions or real_kde_reproducible",
            ),
            (
                "F03_normal_thresholds_for_every_kde",
                "plot_latent_kde",
                old_levels,
                "        levels = [(1-p)/(2*np.pi) for p in [.99, .95, .68]]\n\n",
                "kde_labeled_regions",
            ),
        ]
    if args.only:
        plans = [plan for plan in plans if plan[0] in args.only]
    with tempfile.TemporaryDirectory(prefix="m28-synthetic-") as directory:
        tree = Path(directory)
        (tree / "tests").mkdir()
        shutil.copytree(
            ROOT / "src/dimred",
            tree / "src/dimred",
            ignore=shutil.ignore_patterns("__pycache__"),
        )
        for p in [GREEN, "tests/conftest.py", "pyproject.toml"]:
            shutil.copy2(ROOT / p, tree / p)
        if args.proofs_ref:
            for p in [PROOF, GREEN, "tests/conftest.py"]:
                content = subprocess.check_output(
                    ["git", "show", f"{args.proofs_ref}:{p}"], cwd=ROOT
                )
                (tree / p).write_bytes(content)
            payload["proofs_ref"] = args.proofs_ref
        elif (ROOT / PROOF).exists():
            shutil.copy2(ROOT / PROOF, tree / PROOF)
        (tree / "m28_trace.py").write_text(PLUGIN, encoding="utf-8")
        payload["executed_files_lf_sha256"] = {
            p: digest(tree / p)
            for p in [SOURCE, GREEN, "tests/conftest.py", PROOF]
            if (tree / p).exists()
        }
        tests = [GREEN] if args.mode in ["mutations", "reverse"] else [PROOF]
        baseline = run(tree, tests, "", reverse=args.mode == "reverse")
        payload["baseline"] = baseline
        for name, method, old, new, selection in plans:
            tests = [GREEN, PROOF] if args.mode == "proofs" else [GREEN]
            (tree / SOURCE).write_text(source, encoding="utf-8")
            original = run(tree, tests, selection)
            changed, sites = edit(source, method, old, new)
            (tree / SOURCE).write_text(changed, encoding="utf-8")
            actual = run(tree, tests, selection)
            same = sorted(c["id"] for c in original["cases"]) == sorted(
                c["id"] for c in actual["cases"]
            )
            failures = [c for c in actual["cases"] if c["status"] == "FAILURE"]
            assertions = failures and all(
                "AssertionError" in c["message"] or "E   assert" in c["message"]
                for c in failures
            )
            reached = bool(sites.intersection(actual["executed_lines"]))
            clean = actual["cases"] and all(
                c["status"] in ["PASS", "FAILURE"] for c in actual["cases"]
            )
            classification = "INCONCLUSIVE"
            if same and reached and clean:
                classification = (
                    "KILLED"
                    if assertions
                    else (
                        "PASS" if actual["returncode"] == 0 else "NON_ASSERTION_FAILURE"
                    )
                )
            payload["runs"].append(
                {
                    "name": name,
                    "method": method,
                    "selection": selection,
                    "changed_lines": sorted(sites),
                    "reached": reached,
                    "same_collection": same,
                    "classification": classification,
                    "candidate_lf_sha256": hashlib.sha256(changed.encode()).hexdigest(),
                    "patch": "".join(
                        difflib.unified_diff(
                            source.splitlines(True), changed.splitlines(True)
                        )
                    ),
                    "original": original,
                    "candidate": actual,
                }
            )
            print(name, classification, flush=True)
        payload["protected_unchanged"] = before == {
            p: digest(ROOT / p) for p in protected
        }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    print(output)


if __name__ == "__main__":
    main()
