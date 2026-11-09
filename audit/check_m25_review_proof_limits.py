"""Demonstrate R8 limits without changing the immutable historical archive.

All patches and the stronger observation test exist only in temporary copies.
A PASS means the documented limitation/remedy was observed, not that a proposed
production policy has received scientific approval.
"""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

ARCHIVE = "59eb774eb618cdf639cde3c0358da319ee4bcffa"
ANCHOR = "        # Construction du modèle\n"

OBSERVATION = """
import numpy as np
import pytest
pytestmark = pytest.mark.slow

def test_review_observation(autoencoder_real_module, monkeypatch):
    m = autoencoder_real_module
    if KIND == 'batch':
        real_adam = m.optim.Adam
        steps = []
        class ObservedAdam(real_adam):
            def step(self, *args, **kwargs):
                steps.append(True)
                return super().step(*args, **kwargs)
        monkeypatch.setattr(m.optim, 'Adam', ObservedAdam)
        ae = m.SpectralAutoencoder(2, 1, [2], device='cpu')
        try:
            history = ae.fit(np.arange(4.).reshape(2, 2), epochs=1, batch_size=10, verbose=False)
        except ValueError as error:
            accepted = 'batch' in str(error).lower()
            print('Explicit refusal:', str(error))
        else:
            accepted = bool(steps) and all(np.isfinite(values).all() for values in history.values())
            print('Actual optimizer steps:', len(steps), 'history:', history)
        assert accepted == EXPECTED
    elif KIND == 'state':
        ae = m.SpectralAutoencoder(2, 1, [2], device='cpu')
        X = np.arange(16.).reshape(8, 2)
        ae.fit(X, epochs=1, batch_size=20, verbose=False)
        old_history, old_time = ae.history_, ae.fit_time_
        old_state = {k: v.clone() for k, v in ae._model.state_dict().items()}
        with pytest.raises(ValueError):
            ae.fit(X, epochs=1, batch_size=0, verbose=False)
        if ae._model is None:
            accepted = ae.history_ is None and ae.fit_time_ is None
        else:
            current = ae._model.state_dict()
            accepted = set(current) == set(old_state) and all(m.torch.equal(current[k], v) for k, v in old_state.items()) and ae.history_ == old_history and ae.fit_time_ == old_time
        print('Model/history/time present:', ae._model is not None, ae.history_ is not None, ae.fit_time_ is not None)
        assert accepted == EXPECTED
    else:
        print('Nonfinite imputation satisfies the historical oracle; scientific admissibility remains OPEN.')
"""


def main():
    root = Path(__file__).resolve().parents[1]
    started_at = datetime.now(timezone.utc).isoformat()

    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args])

    head = git("rev-parse", "HEAD").decode().strip()
    tree_status = git("status", "--short").decode().strip()
    paths = git("ls-tree", "-r", "--name-only", ARCHIVE).decode().splitlines()
    selected = [p for p in paths if p.startswith("src/dimred/") and p.endswith(".py")]
    selected += [
        "tests/conftest.py",
        "tests/proofs_dimred_autoencoder.py",
        "pyproject.toml",
    ]
    content = {path: git("show", f"{ARCHIVE}:{path}") for path in selected}
    original = content["src/dimred/autoencoder.py"].decode("utf-8")
    drop_last = original.replace("drop_last=False", "drop_last=True")
    safe_batch = original.replace(
        "drop_last=False", "drop_last=(len(X_train) % batch_size == 1)"
    ).replace(
        ANCHOR,
        "        if len(X) - max(1, int(len(X) * val_fraction)) < 2:\n            raise ValueError('Training batch requires at least two rows')\n\n"
        + ANCHOR,
    )
    partial = original.replace(
        ANCHOR,
        "        if batch_size <= 0:\n            self._model = None\n            raise ValueError('batch_size must be positive')\n\n"
        + ANCHOR,
    )
    complete = original.replace(
        ANCHOR,
        "        if batch_size <= 0:\n            self._model = self.history_ = self.fit_time_ = None\n            raise ValueError('batch_size must be positive')\n\n"
        + ANCHOR,
    )
    preserve = original.replace(
        ANCHOR,
        "        if batch_size <= 0:\n            raise ValueError('batch_size must be positive')\n\n"
        + ANCHOR,
    )
    impute = original.replace(
        ANCHOR,
        "        X = np.nan_to_num(X, nan=0., posinf=0., neginf=0.)\n\n" + ANCHOR,
    )
    trials = [
        ("drop_all_singleton_train", drop_last, "batch", False, "valid_small_training"),
        ("safe_batch_or_refusal", safe_batch, "batch", True, "valid_small_training"),
        ("invalidate_model_only", partial, "state", False, "failed_fit"),
        ("invalidate_complete_state", complete, "state", True, "failed_fit"),
        ("preserve_complete_state", preserve, "state", True, "failed_fit"),
        ("silent_nonfinite_imputation", impute, "finite", None, "nonfinite_training"),
    ]
    records = []
    for name, changed, kind, expected, selector in trials:
        assert changed != original
        compile(changed, "autoencoder.py", "exec")
        with tempfile.TemporaryDirectory(prefix="astrospectro-m25-r8-") as temporary:
            directory = Path(temporary)
            for path, data in content.items():
                destination = directory / path
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(data)
            (directory / "src/dimred/autoencoder.py").write_text(
                changed, encoding="utf-8"
            )
            (directory / "tests/probe_review_limits.py").write_text(
                f"KIND = {kind!r}\nEXPECTED = {expected!r}\n" + OBSERVATION,
                encoding="utf-8",
            )
            command = [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "-m",
                "slow or not slow",
                "tests/proofs_dimred_autoencoder.py",
                "tests/probe_review_limits.py",
                "-k",
                f"{selector} or review_observation",
                "--junitxml=results.xml",
                "-o",
                "junit_logging=all",
            ]
            result = subprocess.run(
                command,
                cwd=directory,
                env={**os.environ, "MPLBACKEND": "Agg", "PYTHONHASHSEED": "7"},
                text=True,
                capture_output=True,
                timeout=120,
            )
            cases = list(ET.parse(directory / "results.xml").iter("testcase"))
            historical = [
                c
                for c in cases
                if "proofs_dimred_autoencoder" in c.get("classname", "")
            ]
            clean = all(
                not any(
                    c.find(tag) is not None for tag in ["failure", "error", "skipped"]
                )
                for c in cases
            )
            success = (
                result.returncode == 0
                and bool(historical)
                and len(cases) == len(historical) + 1
                and clean
            )
            records.append(
                dict(
                    name=name,
                    status="PASS" if success else "FAIL",
                    historical_cases=len(historical),
                    tests=len(cases),
                    stronger_criterion_accepts=expected,
                    patched_source_sha256=hashlib.sha256(changed.encode()).hexdigest(),
                    observations=[c.findtext("system-out", "") for c in cases],
                    command=command,
                    output=result.stdout + result.stderr,
                )
            )
            print(name, records[-1]["status"], flush=True)
    import torch

    payload = dict(
        schema=1,
        archive_sha=ARCHIVE,
        git_head_at_run=head,
        working_tree_at_start=tree_status,
        started_at_utc=started_at,
        archive_inputs_sha256={
            p: hashlib.sha256(b).hexdigest() for p, b in content.items()
        },
        checker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        python=sys.version,
        torch=torch.__version__,
        cases=records,
    )
    (root / "audit/m25_review_proof_limit_results.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    return int(any(r["status"] != "PASS" for r in records))


if __name__ == "__main__":
    raise SystemExit(main())
