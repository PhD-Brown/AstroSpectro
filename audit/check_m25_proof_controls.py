"""Compatibility controls for red oracles, in synthetic temporary source copies."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET


def main():
    root = Path(__file__).resolve().parents[1]
    source_path = root / "src/dimred/autoencoder.py"
    source = source_path.read_text(encoding="utf-8")
    anchor = "        # Construction du modèle\n"
    guard = "        if batch_size <= 0:\n            raise ValueError('batch_size must be positive')\n\n"
    controls = [
        (
            "preserve_completed_state",
            source.replace(anchor, guard + anchor),
            "failed_fit",
        ),
        (
            "invalidate_failed_state",
            source.replace(
                anchor,
                "        if batch_size <= 0:\n            self._model = None\n            self.history_ = self.fit_time_ = None\n            raise ValueError('batch_size must be positive')\n\n"
                + anchor,
            ),
            "failed_fit",
        ),
        (
            "reject_singleton_batch",
            source.replace(
                anchor,
                "        n_train = len(X) - max(1, int(len(X) * val_fraction))\n        if n_train == 1 or (batch_size > 0 and n_train % batch_size == 1):\n            raise ValueError('Singleton training batch unsupported')\n\n"
                + anchor,
            ),
            "valid_small_training",
        ),
        (
            "reject_nonfinite",
            source.replace(
                anchor,
                "        if not np.isfinite(X).all():\n            raise ValueError('Training features must be finite')\n\n"
                + anchor,
            ),
            "nonfinite_training",
        ),
        (
            "honor_empty_hidden",
            source.replace(
                "hidden_dims or [256, 128, 64]",
                "[256, 128, 64] if hidden_dims is None else hidden_dims",
            ),
            "explicit_empty_hidden",
        ),
        (
            "weighted_epoch_mse",
            source.replace(
                "train_losses.append(loss.item())",
                "train_losses.append((loss.item(), len(X_batch)))",
            ).replace(
                "np.mean(train_losses)",
                "sum(v * n for v, n in train_losses) / sum(n for _, n in train_losses)",
            ),
            "epoch_mse",
        ),
        (
            "conditional_class_bases",
            source.replace(
                "(nn.Module):", "(nn.Module if _TORCH_AVAILABLE else object):"
            ),
            "absent_torch",
        ),
    ]
    paths = [
        *list((root / "src/dimred").glob("*.py")),
        root / "tests/conftest.py",
        root / "tests/proofs_dimred_autoencoder.py",
        root / "pyproject.toml",
    ]
    before = {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in paths
    }
    results = []
    for name, changed, selector in controls:
        assert changed != source
        compile(changed, "autoencoder.py", "exec")
        with tempfile.TemporaryDirectory(prefix="astrospectro-m25-control-") as temp:
            folder = Path(temp)
            for path in paths:
                destination = folder / path.relative_to(root)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, destination)
            (folder / "src/dimred/autoencoder.py").write_text(changed, encoding="utf-8")
            command = [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "-m",
                "slow or not slow",
                "tests/proofs_dimred_autoencoder.py",
                "-k",
                selector,
                "--junitxml=results.xml",
            ]
            run = subprocess.run(
                command,
                cwd=folder,
                env={**os.environ, "MPLBACKEND": "Agg"},
                capture_output=True,
                text=True,
                timeout=120,
            )
            cases = list(ET.parse(folder / "results.xml").iter("testcase"))
            status = (
                "PASS"
                if run.returncode == 0
                and cases
                and all(
                    c.find("skipped") is None and c.find("error") is None for c in cases
                )
                else "FAIL"
            )
            results.append(
                dict(
                    name=name,
                    status=status,
                    tests=len(cases),
                    output=run.stdout + run.stderr,
                )
            )
            print(name, status, flush=True)
    assert before == {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in paths
    }
    payload = dict(
        git_head_at_run=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        inputs_sha256=before,
        cases=results,
    )
    (root / "audit/m25_proof_control_results.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    return int(any(row["status"] != "PASS" for row in results))


if __name__ == "__main__":
    raise SystemExit(main())
