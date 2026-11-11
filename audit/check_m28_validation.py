"""Record actual M2.8 validation commands, versions and protected-path checks."""

import argparse
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
BASE = "39dad874610745ccd456d26972e90f6fb3a11405"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="test-results/m28-validation.json")
    parser.add_argument("--tests", action="store_true")
    args = parser.parse_args()
    files = [
        "src/dimred/dimred_visualizer.py",
        "tests/conftest.py",
        "tests/test_dimred_visualizer.py",
    ]
    helpers = [
        "audit/check_m28_evidence.py",
        "audit/check_m28_boundaries.py",
        "audit/check_m28_validation.py",
    ]
    result = {
        "active_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "python": sys.version,
        "executable": sys.executable,
        "versions": {
            name: version(name)
            for name in [
                "numpy",
                "pandas",
                "scipy",
                "scikit-learn",
                "matplotlib",
                "pytest",
                "black",
                "ruff",
                "pillow",
            ]
        },
        "effective_lf_sha256": {
            p: hashlib.sha256(
                (ROOT / p).read_text(encoding="utf-8").replace("\r\n", "\n").encode()
            ).hexdigest()
            for p in files
        },
        "commands": [],
    }
    commands = []
    if args.tests:
        commands += [
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "tests/test_dimred_visualizer.py",
                "-W",
                "error",
                "--junitxml=test-results/m28-targeted-validation.xml",
            ],
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "--junitxml=test-results/m28-complete-validation.xml",
            ],
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "--import-mode=importlib",
                "--junitxml=test-results/m28-importlib-validation.xml",
            ],
        ]
    commands += [
        [
            sys.executable,
            "-m",
            "black",
            "--check",
            "src",
            "tests",
            "scripts",
            "app.py",
            *helpers,
        ],
        [
            sys.executable,
            "-m",
            "ruff",
            "check",
            "src",
            "tests",
            "scripts",
            "app.py",
            *helpers,
        ],
        [sys.executable, "-m", "pip", "check"],
        ["git", "diff", "--check"],
        [
            "git",
            "diff",
            "--exit-code",
            BASE,
            "--",
            "src",
            "notebooks",
            "data",
            "models",
            "requirements",
            "pyproject.toml",
            ".github",
            "app.py",
            "AGENTS.md",
        ],
    ]
    env = dict(
        os.environ, MPLBACKEND="Agg", PYTHONHASHSEED="0", PYTHONIOENCODING="utf-8"
    )
    for command in commands:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=240,
        )
        result["commands"].append(
            {
                "command": command,
                "exit_code": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
            }
        )
        print(" ".join(command[1:]), completed.returncode, flush=True)
    result["all_commands_passed"] = all(c["exit_code"] == 0 for c in result["commands"])
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    if not result["all_commands_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
