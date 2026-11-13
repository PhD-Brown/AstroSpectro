"""Reconstruct Claude review candidates without changing production in the worktree.

Each plan has a selected original-source baseline, exact patches and executed
statement anchors. Wrong candidates count as KILLED only on assertion failures.
Admissible controls and joint satisfiability are reported separately.
"""

import argparse
import ast
import difflib
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE = "src/dimred/dimred_visualizer.py"
GREEN = "tests/test_dimred_visualizer.py"
PROOF = "tests/proofs_dimred_visualizer.py"
EXTRAS = [
    "tests/conftest.py",
    "tests/visualizer_review_contracts.py",
    "tests/test_visualizer_review_contracts.py",
]
spec = importlib.util.spec_from_file_location(
    "m28_original_runner", ROOT / "audit/check_m28_evidence.py"
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def replace(source, method, old, new):
    return runner.edit(source, method, old, new)[0]


def kde_levels(source, new):
    start = source.index("        zz_flat = ZZ.ravel()")
    stop = source.index("        params = [", start)
    source = replace(source, "plot_latent_kde", source[start:stop], new)
    # Probability solvers are independent of this display-domain completion.
    # Original percentile bounds cut the nominal 99% contour into open arcs.
    return replace(
        source,
        "plot_latent_kde",
        "margin = (p99 - p1) * 0.15",
        "margin = (p99 - p1) * 0.8",
    )


C13 = """        center = Z_stars.mean(axis=0)
        scale = Z_stars.std(axis=0)
        qx = np.linspace(center[0]-8*scale[0], center[0]+8*scale[0], 401)
        qy = np.linspace(center[1]-8*scale[1], center[1]+8*scale[1], 401)
        a, b = np.meshgrid(qx, qy)
        density = kde(np.vstack([a.ravel(), b.ravel()]))
        ordered = np.sort(density)[::-1]
        mass = np.cumsum(ordered)/ordered.sum()
        levels = sorted([float(ordered[np.searchsorted(mass,p)]) for p in [.68,.95,.99]])

"""
K4 = """        draws = kde.resample(200000, seed=np.random.default_rng(7))
        density = kde.evaluate(draws)
        levels = sorted(float(np.quantile(density,1-p)) for p in [.99,.95,.68])

"""
K5 = """        sigma = np.sqrt(np.diag(kde.covariance))
        qx = np.linspace(kde.dataset[0].min()-4*sigma[0],kde.dataset[0].max()+4*sigma[0],500)
        qy = np.linspace(kde.dataset[1].min()-4*sigma[1],kde.dataset[1].max()+4*sigma[1],500)
        a,b = np.meshgrid(qx,qy)
        density = kde(np.vstack([a.ravel(),b.ravel()]))
        ordered = np.sort(density)[::-1]
        mass = np.cumsum(ordered)/ordered.sum()
        levels = sorted([float(ordered[np.searchsorted(mass,p)]) for p in [.68,.95,.99]])

"""


def scalar_patch(method, old, new):
    return lambda source: replace(source, method, old, new)


def categorical(source, legend):
    for method, variable in [
        ("plot_hdbscan_clusters", "sm"),
        ("plot_hdbscan_hr", "sm2"),
    ]:
        if legend:
            node = next(
                n
                for n in ast.walk(ast.parse(source))
                if isinstance(n, ast.FunctionDef) and n.name == method
            )
            statement = next(
                n
                for n in node.body
                if isinstance(n, ast.Expr) and "plt.colorbar(" in ast.unparse(n)
            )
            old = "\n".join(
                source.splitlines()[statement.lineno - 1 : statement.end_lineno]
            )
            new = '        ax.legend(handles=[mpatches.Patch(color=color_map[c],label=f"C{c}") for c in cluster_ids])\n        for annotation in list(ax.texts):\n            annotation.remove()'
            source = replace(source, method, old, new)
        else:
            source = replace(
                source,
                method,
                "cmap=cmap_c, norm=mcolors.Normalize(vmin=0, vmax=n - 1)",
                "cmap=mcolors.ListedColormap([color_map[c] for c in cluster_ids] or [(0.8,0.8,0.8,1)]), norm=mcolors.BoundaryNorm(([cluster_ids[0]-.5]+[(a+b)/2 for a,b in zip(cluster_ids[:-1],cluster_ids[1:])]+[cluster_ids[-1]+.5]) if cluster_ids else [-.5,.5], max(1,len(cluster_ids)))",
            )
    return source


def exports(source):
    old = """        if save_path is not None:
            fig.savefig(save_path, bbox_inches="tight", dpi=self.dpi)
            logger.info("Figure sauvegardée : %s", save_path)"""
    for method in ["plot_training_history", "plot_ae_vs_pca"]:
        source = replace(
            source, method, old, f'        self._save(fig,save_path,"{method}")'
        )
    return source


def joint(source):
    # Original C01-C14 controls except C13 (constructed below).
    for _, method, old, new, _ in runner.CONTROLS:
        source = replace(source, method, old, new)
    source = kde_levels(source, C13)
    source = categorical(source, False)
    source = replace(
        source,
        "plot_xgboost_umap",
        "confidence is not None and not np.all(confidence == 1.0)",
        "confidence is not None",
    )
    source = replace(
        source,
        "plot_hdbscan_feature_profiles",
        "axes_flat[n_show:]",
        "axes_flat[len(top_cids):]",
    )
    source = replace(
        source,
        "plot_variance_explained",
        'f"{n_thresh} PCs\\n→ {cumvar[idx]*100:.1f}%"',
        'f"{n_thresh} PCs\\n→ {pca_analyzer.explained_variance_ratio[:n_thresh].sum()*100:.1f}%"',
    )
    source = replace(
        source,
        "plot_ae_error_distribution",
        "zip(classes, palette)",
        "zip(classes, palette*((len(classes)+len(palette)-1)//len(palette)))",
    )
    source = exports(source)
    source = disclose_histogram(source)
    return source


def disclose_histogram(source):
    source = replace(
        source,
        "plot_ae_error_distribution_logscale",
        "                ax.hist(\n                    vals,",
        '                missing = int(((vals<bins[0])|(vals>bins[-1])).sum())\n                if missing:\n                    ax.text(.03,.05,f"{cls}: excluded {missing} of {len(vals)} outside support {bins[0]:.6f} to {bins[-1]:.6f}",transform=ax.transAxes)\n                ax.hist(\n                    vals,',
    )
    return source


def plans():
    p = []

    def add(name, selection, transform, admissible=False):
        p.append((name, selection, transform, admissible))

    add("C13", "kde_labeled_regions", lambda s: kde_levels(s, C13), True)
    add("K4", "kde_labeled_regions", lambda s: kde_levels(s, K4), True)
    add("K5", "kde_labeled_regions", lambda s: kde_levels(s, K5), True)
    add(
        "K1B",
        "kde_labeled_regions",
        lambda s: kde_levels(
            s,
            "        peak=float(ZZ.max())\n        levels=sorted((1-p)*peak for p in [.99,.95,.68])\n\n",
        ),
    )
    for name, old, new in [
        ("K6", '["99%", "95%", "68%"]', '["68%", "95%", "99%"]'),
        (
            "K7",
            'gaussian_kde(Z_stars[idx].T, bw_method="scott")',
            'gaussian_kde(Z_ae.T, bw_method="scott")',
        ),
        (
            "K8",
            "                    ZZ,\n                    levels=levels,",
            "                    ZZ.T,\n                    levels=levels,",
        ),
    ]:
        add(
            name,
            "kde_labeled_regions",
            lambda s, old=old, new=new: replace(
                kde_levels(s, C13), "plot_latent_kde", old, new
            ),
        )
    add(
        "K9",
        "kde_labeled_regions",
        lambda s: replace(
            kde_levels(s, C13),
            "plot_latent_kde",
            "                    ZZ,\n                    levels=levels,",
            "                    np.ma.masked_where(XX > 3, ZZ),\n                    levels=levels,",
        ),
    )
    add("AD05", "cluster_colorbar", lambda s: categorical(s, True), True)
    add("AC05", "cluster_colorbar", lambda s: categorical(s, False), True)
    add(
        "FD02",
        "hr_region_vertices",
        scalar_patch(
            "plot_hr_teff_regions",
            "ax.set_ylim(G_HI + G_PAD * 5, G_LO - G_PAD * 5)",
            '_ys = [v[1] for r in REGIONS for v in r["poly"]]\n        ax.set_ylim(max(G_HI+G_PAD*5,max(_ys)),min(G_LO-G_PAD*5,min(_ys)))',
        ),
    )
    add(
        "FD09",
        "more_reconstructions",
        scalar_patch(
            "plot_ae_reconstruction",
            "for i, (ax, si) in enumerate(zip(axes, idx)):",
            "for unused in list(axes)[len(idx):]:\n            unused.plot([],[])\n        for i, (ax, si) in enumerate(zip(axes, idx)):",
        ),
    )

    def fd11(s):
        s = replace(
            s,
            "plot_ae_error_distribution_logscale",
            "max(vals.min(), 1e-4)",
            "vals.min()",
        )
        return replace(
            s,
            "plot_ae_error_distribution_logscale",
            "                ax.hist(\n                    vals,",
            '                ax.hist(\n                    vals[vals>=1e-4] if xscale=="log" else vals,',
        )

    add("FD11", "log_histogram_keeps", fd11)
    add(
        "FD12",
        "fifth_class",
        scalar_patch(
            "plot_ae_error_distribution",
            '        ax.set_xlabel("MSE de reconstruction")',
            '        for cls in classes[len(palette):]:\n            ax.plot([],[],label=f"{cls} (n={(y==cls).sum()})")\n        ax.set_xlabel("MSE de reconstruction")',
        ),
    )
    add(
        "FD13",
        "default_output_directory and plot_training_history",
        scalar_patch(
            "plot_training_history",
            "        if save_path is not None:",
            '        if self.output_dir is not None:\n            plt.figure().savefig(self.output_dir/"empty.png")\n        if save_path is not None:',
        ),
    )
    add(
        "FD03",
        "hr_population_title",
        scalar_patch("plot_hr_teff_regions", "43 019 étoiles", "N = 43 019"),
    )
    add(
        "FD10",
        "truncated_variance_annotation",
        scalar_patch(
            "plot_variance_explained",
            'f"{n_thresh} PCs\\n→ {cumvar[idx]*100:.1f}%"',
            'f"→ {cumvar[idx]*100:.1f}% avec {n_thresh} PCs"',
        ),
    )
    for name, method, old, new, selection in [
        (
            "N01",
            "plot_xgboost_umap",
            'label=f"{cls} (n={mask.sum()})"',
            'label=f"{classes_pred[(classes_pred.index(cls)+1)%len(classes_pred)]} (n={mask.sum()})"',
            "xgboost_classes_confidence",
        ),
        (
            "N02",
            "plot_hdbscan_clusters",
            'label=f"{cls} (n={mask.sum()})"',
            'label=f"{np.unique(y)[(list(np.unique(y)).index(cls)+1)%len(np.unique(y))]} (n={mask.sum()})"',
            "real_analyzer_state",
        ),
        (
            "N04",
            "plot_synthesis_grid",
            'f"{method_name}\\n{ylabel}"',
            'f"{METHODS[-1-row][0]}\\n{ylabel}"',
            "synthesis_preserves",
        ),
        (
            "N10",
            "plot_trustworthiness",
            "Zt_sub = Z_tsne[idx]",
            "Zt_sub = Z_tsne[np.roll(idx,1)]",
            "real_sklearn_trustworthiness",
        ),
    ]:
        add(name, selection, scalar_patch(method, old, new))

    def n05(s):
        s = replace(s, "plot_latent_interpolation", "label=label_a", "label=(label_b)")
        return replace(s, "plot_latent_interpolation", "label=label_b", "label=label_a")

    add("N05", "interpolation_actual", n05)

    def n06(s):
        s = replace(s, "plot_loadings_family_bar", 'label="PC1"', 'label=("PC2")')
        return replace(s, "plot_loadings_family_bar", 'label="PC2"', 'label="PC1"')

    add("N06", "family_weights_bar", n06)
    add(
        "N03",
        "real_export_readable",
        scalar_patch(
            "_save",
            'fig.savefig(path, bbox_inches="tight", dpi=self.dpi)',
            'import pickle\n        clone=pickle.loads(pickle.dumps(fig))\n        for axis in clone.axes:\n            for collection in list(axis.collections):\n                collection.remove()\n        clone.savefig(path,bbox_inches="tight",dpi=self.dpi)\n        plt.close(clone)',
        ),
    )
    add(
        "A_TRUST",
        "real_sklearn_trustworthiness",
        scalar_patch(
            "plot_trustworthiness",
            "idx = rng.choice(len(X_ref), size=n, replace=False)",
            "idx = np.sort(rng.permutation(len(X_ref))[:n])",
        ),
        True,
    )
    add(
        "AD14",
        "trustworthiness_axis_contains",
        scalar_patch(
            "plot_trustworthiness",
            "ax.set_ylim(0.75, 1.01)",
            'ax.set_ylim(0.75,1.01)\n        ax.text(.1,.1,f"T(k) < 0.75 hors champ : min = {min(trust_t):.3f}",transform=ax.transAxes)',
        ),
        True,
    )
    add("A_EXPORT", "default_output_directory or real_export_readable", exports, True)
    add(
        "A_METHODS",
        "synthesis_preserves",
        scalar_patch("plot_synthesis_grid", '("Autoencodeur", Z_ae,', '("AE", Z_ae,'),
        True,
    )
    add(
        "A_HIST",
        "linear_histogram_truncation",
        scalar_patch(
            "plot_ae_error_distribution_logscale",
            "star_p99 * 1.5",
            "float(np.max(mse_per_sample))",
        ),
        True,
    )
    add("AD_HIST", "linear_histogram_truncation", disclose_histogram, True)
    add("AD_LOG", "log_histogram_keeps", disclose_histogram, True)
    add("JOINT", "", joint, True)
    return p


def changed_anchors(before, after):
    statements = [n for n in ast.walk(ast.parse(after)) if isinstance(n, ast.stmt)]
    groups = []
    for op, a, b, c, d in difflib.SequenceMatcher(
        None, before.splitlines(), after.splitlines()
    ).get_opcodes():
        if op == "equal":
            continue
        anchors = set()
        for line in range(c + 1, max(c + 2, d + 1)):
            candidates = [n for n in statements if n.lineno <= line <= n.end_lineno]
            if candidates:
                anchors.add(
                    min(candidates, key=lambda n: n.end_lineno - n.lineno).lineno
                )
        groups.append(sorted(anchors))
    return groups


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="+")
    parser.add_argument("--proofs-ref")
    parser.add_argument("--output", default="test-results/m28-review.json")
    args = parser.parse_args()
    source = (ROOT / SOURCE).read_text(encoding="utf-8")
    protected = {
        p: runner.digest(ROOT / p)
        for p in [SOURCE, GREEN, PROOF, *EXTRAS]
        if (ROOT / p).exists()
    }
    if args.proofs_ref:
        archived_source = (
            subprocess.check_output(
                ["git", "show", f"{args.proofs_ref}:{SOURCE}"], cwd=ROOT
            )
            .decode("utf-8")
            .replace("\r\n", "\n")
        )
        if archived_source != source:
            raise RuntimeError("Production differs from the chosen proof archive")
    ledger = {
        "active_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "python": sys.version,
        "working_hashes": protected,
        "proofs_ref": args.proofs_ref,
        "runner_lf_sha256": runner.digest(Path(__file__)),
        "runs": [],
    }
    with tempfile.TemporaryDirectory(prefix="m28-review-synthetic-") as directory:
        tree = Path(directory)
        shutil.copytree(
            ROOT / "src/dimred",
            tree / "src/dimred",
            ignore=shutil.ignore_patterns("__pycache__"),
        )
        (tree / "tests").mkdir()
        for p in [GREEN, PROOF, *EXTRAS, "pyproject.toml"]:
            if args.proofs_ref and p.startswith("tests/"):
                data = subprocess.check_output(
                    ["git", "show", f"{args.proofs_ref}:{p}"], cwd=ROOT
                )
                (tree / p).write_bytes(data)
            else:
                shutil.copy2(ROOT / p, tree / p)
        ledger["executed_hashes"] = {
            p: runner.digest(tree / p) for p in [SOURCE, GREEN, PROOF, *EXTRAS]
        }
        (tree / "m28_trace.py").write_text(runner.PLUGIN, encoding="utf-8")
        tests = [GREEN, EXTRAS[-1], PROOF]
        for name, selection, transform, admissible in plans():
            if args.only and name not in args.only:
                continue
            (tree / SOURCE).write_text(source, encoding="utf-8")
            baseline_source = source if admissible else joint(source)
            (tree / SOURCE).write_text(baseline_source, encoding="utf-8")
            baseline = runner.run(tree, tests, selection)
            candidate = transform(source)
            ast.parse(candidate)
            anchors = changed_anchors(source, candidate)
            (tree / SOURCE).write_text(candidate, encoding="utf-8")
            actual = runner.run(tree, tests, selection)
            same = sorted(c["id"] for c in baseline["cases"]) == sorted(
                c["id"] for c in actual["cases"]
            )
            reached = all(
                set(a).intersection(actual["executed_lines"]) for a in anchors
            )
            failures = [c for c in actual["cases"] if c["status"] == "FAILURE"]
            assertions = bool(failures) and all(
                "AssertionError" in c["message"] or "E   assert" in c["message"]
                for c in failures
            )
            clean = bool(actual["cases"]) and all(
                c["status"] in ["PASS", "FAILURE"] for c in actual["cases"]
            )
            result = "INCONCLUSIVE"
            if (
                same
                and reached
                and clean
                and (admissible or baseline["returncode"] == 0)
            ):
                result = (
                    ("PASS" if admissible else "SURVIVED")
                    if actual["returncode"] == 0
                    else "KILLED" if assertions else "NON_ASSERTION_FAILURE"
                )
            ledger["runs"].append(
                dict(
                    name=name,
                    selection=selection,
                    admissible=admissible,
                    classification=result,
                    reached_all_hunks=reached,
                    same_collection=same,
                    changed_anchors=anchors,
                    patch="".join(
                        difflib.unified_diff(
                            source.splitlines(True), candidate.splitlines(True)
                        )
                    ),
                    candidate_sha256=hashlib.sha256(candidate.encode()).hexdigest(),
                    baseline_kind=(
                        "original production"
                        if admissible
                        else "admissible joint control"
                    ),
                    baseline_source_sha256=hashlib.sha256(
                        baseline_source.encode()
                    ).hexdigest(),
                    baseline=baseline,
                    candidate=actual,
                )
            )
            output = ROOT / args.output
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(
                json.dumps(ledger, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
            )
            print(name, result, flush=True)
    ledger["protected_unchanged"] = protected == {
        p: runner.digest(ROOT / p) for p in protected
    }
    output.write_text(
        json.dumps(ledger, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    assert ledger["protected_unchanged"], "Protected worktree inputs changed"


if __name__ == "__main__":
    main()
