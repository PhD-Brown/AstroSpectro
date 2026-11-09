"""Replay M2.5 semantic mutants in temporary copies, never edit production.

Run with the optional CPU Python: python audit/check_m25_mutations.py.
Each candidate gets fresh source/tests and a JUnit report. Collection failures,
errors, missing tests and optional skips are inconclusive, never KILLED.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET


# name, expected category, exact source expression, replacement
CANDIDATES = [
    (
        "C05_clip_after_step",
        "incorrect",
        "nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=1.0)\n                optimizer.step()",
        "optimizer.step()\n                nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=1.0)",
    ),
    (
        "C06_sixty_neighbours",
        "incorrect",
        "np.argsort(dists)[:50]",
        "np.argsort(dists)[:60]",
    ),
    (
        "C07_twenty_neighbours",
        "incorrect",
        "np.argsort(dists)[:50]",
        "np.argsort(dists)[:20]",
    ),
    ("C08_inclusive_gravity", "incorrect", "logg_v < g_hi", "logg_v <= g_hi"),
    (
        "C01b_snapshot_without_buffers",
        "incorrect",
        "best_state = {k: v.clone() for k, v in self._model.state_dict().items()}",
        "best_state = {k: v.clone() for k, v in self._model.named_parameters()}",
    ),
    (
        "C02_patience_not_reset",
        "incorrect",
        "                patience_counter = 0",
        "                pass  # missing improvement reset",
    ),
    ("C03_relu_identity", "incorrect", "nn.ReLU(inplace=True),", "nn.Identity(),"),
    (
        "C04_relu_before_bn",
        "incorrect",
        "nn.BatchNorm1d(h),\n                nn.ReLU(inplace=True),",
        "nn.ReLU(inplace=True),\n                nn.BatchNorm1d(h),",
    ),
    (
        "split_overlap",
        "incorrect",
        "train_idx = np.setdiff1d(np.arange(N), val_idx)",
        "train_idx = np.arange(N)",
    ),
    (
        "split_size",
        "incorrect",
        "n_val = max(1, int(N * val_fraction))",
        "n_val = max(1, int(N * val_fraction) + 1)",
    ),
    (
        "split_seed",
        "incorrect",
        "rng = np.random.default_rng(42)",
        "rng = np.random.default_rng(43)",
    ),
    (
        "validation_train_mode",
        "incorrect",
        "self._model.eval()\n            with torch.no_grad():",
        "self._model.train()\n            with torch.no_grad():",
    ),
    (
        "inference_dropout",
        "incorrect",
        "self._model.eval()\n        with torch.no_grad():",
        "self._model.train()\n        with torch.no_grad():",
    ),
    (
        "no_best_restore",
        "incorrect",
        "self._model.load_state_dict(best_state)",
        "pass  # lost best-state restoration",
    ),
    (
        "best_state_alias",
        "incorrect",
        "v.clone() for k, v in self._model.state_dict().items()",
        "v for k, v in self._model.state_dict().items()",
    ),
    (
        "patience_off_by_one",
        "incorrect",
        "patience_counter >= early_stopping_patience",
        "patience_counter > early_stopping_patience",
    ),
    (
        "train_history",
        "incorrect",
        'history["train_loss"].append(train_loss)',
        'history["train_loss"].append(val_loss)',
    ),
    (
        "lr_history",
        "incorrect",
        'history["lr_history"].append(current_lr)',
        'history["lr_history"].append(lr)',
    ),
    (
        "no_gradient_reset",
        "incorrect",
        "optimizer.zero_grad()",
        "pass  # gradients accumulate",
    ),
    (
        "no_clipping",
        "incorrect",
        "nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=1.0)",
        "pass  # no clipping",
    ),
    (
        "ignored_weight_decay",
        "incorrect",
        "self._model.parameters(), lr=lr, weight_decay=weight_decay",
        "self._model.parameters(), lr=lr, weight_decay=0.",
    ),
    (
        "mse_mae",
        "incorrect",
        "np.mean((X - X_recon) ** 2, axis=1)",
        "np.mean(np.abs(X - X_recon), axis=1)",
    ),
    (
        "mse_sum_features",
        "incorrect",
        "np.mean((X - X_recon) ** 2, axis=1)",
        "np.sum((X - X_recon) ** 2, axis=1)",
    ),
    (
        "mse_global_median",
        "incorrect",
        "float(mse_per.mean())",
        "float(np.median(mse_per))",
    ),
    ("summary_ddof", "incorrect", "mse.std()", "mse.std(ddof=1)"),
    (
        "summary_median",
        "incorrect",
        '"mse_mean": mse.mean()',
        '"mse_mean": np.median(mse)',
    ),
    (
        "summary_quantile",
        "incorrect",
        "np.percentile(mse, 95)",
        "np.percentile(mse, 90)",
    ),
    (
        "wrong_latent_dim",
        "incorrect",
        "nn.Linear(prev, latent_dim)",
        "nn.Linear(prev, latent_dim + 1)",
    ),
    (
        "reversed_features",
        "incorrect",
        "X_t = torch.FloatTensor(X).to(self.device)",
        "X_t = torch.FloatTensor(X[:, ::-1].copy()).to(self.device)",
    ),
    (
        "reversed_rows",
        "incorrect",
        "Z = self._model.encode(X_t).cpu().numpy()",
        "Z = self._model.encode(X_t).cpu().numpy()[::-1]",
    ),
    (
        "checkpoint_weights_missing",
        "incorrect",
        '"model_state_dict": self._model.state_dict()',
        '"model_state_dict": {}',
    ),
    (
        "checkpoint_architecture",
        "incorrect",
        'latent_dim=checkpoint["latent_dim"]',
        'latent_dim=checkpoint["latent_dim"] + 1',
    ),
    (
        "interpolation_reversed",
        "incorrect",
        "(1 - a) * z_a + a * z_b",
        "a * z_a + (1 - a) * z_b",
    ),
    (
        "arithmetic_sign",
        "incorrect",
        'z["K_geante"] - z["K_naine"] + z["G_naine"]',
        'z["K_geante"] + z["K_naine"] + z["G_naine"]',
    ),
    (
        "arithmetic_reference",
        "incorrect",
        'z["K_naine"] - z["G_geante"]',
        'z["K_naine"] - z["G_naine"]',
    ),
    (
        "arithmetic_population",
        "incorrect",
        '"K_naine": select_pop((3800, 5200)',
        '"K_naine": select_pop((5200, 6200)',
    ),
    (
        "candidate_noise_votes",
        "incorrect",
        "int(np.bincount(nn_cl_valid).argmax())",
        "int(np.unique(nn_cl, return_counts=True)[0][np.unique(nn_cl, return_counts=True)[1].argmax()])",
    ),
    (
        "candidate_wrong_distance_axis",
        "incorrect",
        "np.linalg.norm(Z_train - z, axis=1)",
        "np.abs((Z_train - z)[:, 1])",
    ),
    (
        "candidate_no_scaler",
        "incorrect",
        "x_scaled = loader_ref.scaler_.transform(x)",
        "x_scaled = x.copy()",
    ),
    (
        "candidate_all_neighbours",
        "incorrect",
        "np.argsort(dists)[:50]",
        "np.argsort(dists)",
    ),
    (
        "pca_median",
        "incorrect",
        "pca_analyzer.reconstruction_error(X, n_components=n).mean()",
        "np.median(pca_analyzer.reconstruction_error(X, n_components=n))",
    ),
    (
        "mse_equivalent",
        "equivalent",
        "np.mean((X - X_recon) ** 2, axis=1)",
        'np.einsum("ij,ij->i", X - X_recon, X - X_recon) / X.shape[1]',
    ),
    (
        "architecture_equivalent",
        "equivalent",
        "return self.net(x)",
        "return self.net.forward(x)",
    ),
    (
        "extra_checkpoint_metadata",
        "admissible",
        '"fit_time": self.fit_time_,',
        '"fit_time": self.fit_time_, "audit_provenance": "synthetic",',
    ),
    (
        "cautious_interpolation_prose",
        "admissible",
        "physiquement cohérent",
        "géométriquement continu, sans validation physique",
    ),
    (
        "cautious_candidate_verdict",
        "admissible",
        "dans le top 5% des erreurs",
        "seuil heuristique non calibré",
    ),
    (
        "explicit_shape_rejection",
        "admissible",
        "self._check_fitted()\n        self._model.eval()",
        'self._check_fitted()\n        if np.asarray(X if "X" in locals() else Z).ndim != 2:\n            raise ValueError("Expected a 2D input")\n        self._model.eval()',
    ),
]


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="audit/m25_mutation_results.json")
    parser.add_argument(
        "--only", default="", help="Substring filter, for investigation"
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    relative_paths = [
        *[
            str(path.relative_to(root)).replace("\\", "/")
            for path in sorted((root / "src/dimred").glob("*.py"))
        ],
        "tests/conftest.py",
        "tests/test_dimred_autoencoder.py",
        "tests/test_dimred_autoencoder_real.py",
        "pyproject.toml",
        "audit/check_m25_mutations.py",
    ]
    before = {name: digest(root / name) for name in relative_paths}
    source_path = "src/dimred/autoencoder.py"
    source = (root / source_path).read_text(encoding="utf-8")
    candidates = [("baseline", "baseline", "", ""), *CANDIDATES]
    if args.only:
        candidates = [
            c
            for c in candidates
            if c[0] == "baseline" or any(part in c[0] for part in args.only.split(","))
        ]
    records = []
    baseline_count = None
    for name, category, old, new in candidates:
        with tempfile.TemporaryDirectory(prefix="astrospectro-m25-") as folder:
            directory = Path(folder).resolve()
            for rel in relative_paths:
                target = directory / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(root / rel, target)
            count = source.count(old) if old else 0
            if old and count == 0:
                raise ValueError(f"Missing source anchor: {name}")
            mutated = source.replace(old, new) if old else source
            if name == "C01b_snapshot_without_buffers":
                mutated = mutated.replace(
                    "self._model.load_state_dict(best_state)",
                    "self._model.load_state_dict(best_state, strict=False)",
                )
            compile(mutated, "autoencoder.py", "exec")
            (directory / source_path).write_text(mutated, encoding="utf-8")
            env = {
                **os.environ,
                "MPLBACKEND": "Agg",
                "PYTHONHASHSEED": "7",
                "PYTHONDONTWRITEBYTECODE": "1",
            }
            command = [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "-m",
                "slow or not slow",
                "tests/test_dimred_autoencoder.py",
                "tests/test_dimred_autoencoder_real.py",
                "-k",
                "not fresh_process and not independent_loader and not png_export",
                "--junitxml=results.xml",
            ]
            result = subprocess.run(
                command,
                cwd=directory,
                env=env,
                text=True,
                capture_output=True,
                timeout=120,
            )
            report = directory / "results.xml"
            cases = list(ET.parse(report).iter("testcase")) if report.exists() else []
            errors = [c.attrib["name"] for c in cases if c.find("error") is not None]
            failures = [
                {
                    "name": c.attrib["name"],
                    "message": c.find("failure").get("message", ""),
                }
                for c in cases
                if c.find("failure") is not None
            ]
            skips = sum(c.find("skipped") is not None for c in cases)
            if name == "baseline":
                baseline_count = len(cases)
            conclusive = (
                len(cases) == baseline_count
                and not errors
                and not skips
                and result.returncode in [0, 1]
            )
            status = (
                ("KILLED" if failures else "SURVIVED")
                if category == "incorrect"
                else ("PASS" if not failures else "FAIL")
            )
            if not conclusive:
                status = "INCONCLUSIVE"
            record = dict(
                name=name,
                category=category,
                status=status,
                replacements=count,
                tests=len(cases),
                failed=failures,
                errors=errors,
                skipped=skips,
                exit_code=result.returncode,
            )
            records.append(record)
            print(name, status, len(failures), flush=True)
            if name == "baseline" and status != "PASS":
                raise RuntimeError(result.stdout + result.stderr)
    assert before == {name: digest(root / name) for name in relative_paths}
    import torch

    payload = dict(
        schema=1,
        git_head_at_run=git(root, "rev-parse", "HEAD"),
        working_tree_at_run=git(root, "status", "--short"),
        inputs_sha256=before,
        source_blob=git(root, "rev-parse", "HEAD:src/dimred/autoencoder.py"),
        python=sys.version,
        platform=platform.platform(),
        torch=torch.__version__,
        cuda=torch.version.cuda,
        command=sys.argv,
        cases=records,
    )
    (root / args.output).write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    return int(
        any(r["status"] in ["INCONCLUSIVE", "FAIL", "SURVIVED"] for r in records)
    )


if __name__ == "__main__":
    raise SystemExit(main())
