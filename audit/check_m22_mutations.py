"""Replay PCA mutations in temporary copies with offline pytest and JUnit checks.

Run: python audit/check_m22_mutations.py --output test-results/m22-mutations.json
Only named source replacements are made, never in the working tree. A surviving
mutant is an assertion/contract observation, not automatically a production bug.
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
    ("fit_standardizes", "self._pca.fit(X)", "self._pca.fit(X / X.std(axis=0))"),
    ("seed_ignored", "random_state=self.random_state,", "random_state=0,"),
    ("solver_ignored", "svd_solver=self.svd_solver,", 'svd_solver="full",'),
    (
        "variance_renormalized",
        "return self._pca.explained_variance_ratio_",
        "return self._pca.explained_variance_ratio_ / self._pca.explained_variance_ratio_.sum()",
    ),
    (
        "covariance_population_denominator",
        "self._X_fit = X",
        "self._pca.explained_variance_ *= (len(X)-1)/len(X)\n        self._X_fit = X",
    ),
    (
        "threshold_right_search",
        "np.searchsorted(cumvar, threshold)",
        'np.searchsorted(cumvar, threshold, side="right")',
    ),
    (
        "top_signed_order",
        "np.argsort(np.abs(vec))[::-1][:n_top]",
        "np.argsort(vec)[::-1][:n_top]",
    ),
    (
        "top_loses_sign",
        '"loading": vec[idx_sorted],',
        '"loading": np.abs(vec[idx_sorted]),',
    ),
    (
        "family_absolute_weights",
        "w = loadings[pc_idx] ** 2",
        "w = np.abs(loadings[pc_idx])",
    ),
    ("family_no_normalization", "w /= w.sum()", "w = w.copy()"),
    (
        "family_reverse_priority",
        "for family, patterns in self._FAMILIES.items():",
        "for family, patterns in reversed(list(self._FAMILIES.items())):",
    ),
    (
        "correlation_default_pearson",
        'method: str = "spearman",',
        'method: str = "pearson",',
    ),
    ("correlation_thirty_excluded", "if valid.sum() < 30:", "if valid.sum() <= 30:"),
    (
        "correlation_ignores_score_mask",
        "np.isfinite(param_vals) & np.isfinite(pc_scores)",
        "np.isfinite(param_vals)",
    ),
    ("correlation_ignores_explicit_scores", "if scores is None:", "if True:"),
    (
        "reconstruction_omits_mean",
        "X_recon = self._pca.inverse_transform(scores_trunc)",
        "X_recon = self._pca.inverse_transform(scores_trunc) - self._pca.mean_",
    ),
    (
        "reconstruction_sum_not_mean",
        "mse = np.mean((X - X_recon) ** 2, axis=1)",
        "mse = np.sum((X - X_recon) ** 2, axis=1)",
    ),
    ("curve_sample_std", '"mse_std": mse.std(),', '"mse_std": mse.std(ddof=1),'),
    ("curve_cap_fortynine", "min(n_max + 1, 51)", "min(n_max + 1, 50)"),
    ("class_wrong_mask", "mask = y == cls", "mask = np.roll(y == cls, 1)"),
    (
        "class_sample_std",
        '"std": scores[mask, i].std(),',
        '"std": scores[mask, i].std(ddof=1),',
    ),
    (
        "save_reverses_names",
        '{"pca": self._pca, "feature_names": self.feature_names_}',
        '{"pca": self._pca, "feature_names": self.feature_names_[::-1]}',
    ),
    # Known failure transaction: successful-fit contracts cannot detect this.
    (
        "failed_fit_guard_still_inadequate",
        "if self._pca is None:",
        "if self._pca is None or not hasattr(self._pca, 'fit'):",
    ),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("test-results/m22-mutations.json")
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    paths = [
        "src/dimred/pca_analyzer.py",
        "src/dimred/data_loader.py",
        "src/dimred/dimred_visualizer.py",
        "tests/conftest.py",
        "tests/test_dimred_pca_analyzer.py",
        "pyproject.toml",
    ]
    hashes = {p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in paths}
    report = {"inputs_sha256": hashes, "python": sys.version, "cases": []}
    with tempfile.TemporaryDirectory(prefix="astrospectro-m22-mutation-") as temporary:
        sandbox = Path(temporary)
        for relative in paths:
            destination = sandbox / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(root / relative, destination)
        source_path = sandbox / paths[0]
        source = source_path.read_text(encoding="utf-8")
        for name, before, after in [("baseline", None, None)] + MUTATIONS:
            if before is not None and source.count(before) != 1:
                raise RuntimeError(f"{name}: expected one replacement site")
            changed = source if before is None else source.replace(before, after)
            compile(changed, str(source_path), "exec")
            source_path.write_text(changed, encoding="utf-8")
            # Clear only caches within this disposable, resolved sandbox.
            for cache in (sandbox / "src").rglob("__pycache__"):
                cache.resolve().relative_to(sandbox.resolve())
                shutil.rmtree(cache)
            junit = sandbox / "result.xml"
            if junit.exists():
                junit.unlink()
            command = [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "tests/test_dimred_pca_analyzer.py",
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
                timeout=60,
            )
            if result.returncode not in (0, 1) or not junit.exists():
                raise RuntimeError(
                    f"{name}: invalid execution\n{result.stdout}\n{result.stderr}"
                )
            suite = ET.parse(junit).getroot().find("testsuite")
            if int(suite.get("errors", 0)) or int(suite.get("skipped", 0)):
                raise RuntimeError(f"{name}: collection/fixture error or skip")
            tests = int(suite.get("tests"))
            if report["cases"] and tests != report["cases"][0]["tests"]:
                raise RuntimeError(f"{name}: changed collected case count")
            failures = [
                case.get("name")
                for case in suite.findall("testcase")
                if case.find("failure") is not None
            ]
            if name == "baseline" and failures:
                raise RuntimeError(f"Baseline failed: {failures}")
            status = (
                "PASS" if name == "baseline" else "KILLED" if failures else "SURVIVED"
            )
            report["cases"].append(
                {"name": name, "status": status, "tests": tests, "failures": failures}
            )
            print(f"{name}: {status} ({len(failures)} failing cases)", flush=True)
    assert hashes == {
        p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in paths
    }, "Working-tree input changed"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Report: {args.output}")


if __name__ == "__main__":
    main()
