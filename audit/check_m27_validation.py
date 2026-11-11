"""Capture actual validation commands, versions and protected-file boundaries."""

import hashlib
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
BASE = "1e2266385eab287d3a1c54c2e919e012293fa44b"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path, default=ROOT / "audit/m27_validation_results.json"
    )
    parser.add_argument("--review", action="store_true")
    args = parser.parse_args()
    output = args.output
    protected = [
        "src/dimred/run_reporter.py",
        "tests/test_dimred_run_reporter.py",
        "tests/conftest.py",
    ]
    if args.review:
        protected += [
            "tests/reporter_review_contracts.py",
            "tests/test_reporter_review_contracts.py",
        ]

    def hashes():
        return {
            p: hashlib.sha256(
                (ROOT / p).read_text(encoding="utf-8").encode()
            ).hexdigest()
            for p in protected
        }

    before = hashes()
    result = {
        "schema": 1,
        "head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "base": BASE,
        "python": sys.version,
        "executable": sys.executable,
        "tested_lf_sha256": before,
        "versions": {
            p: importlib.metadata.version(p)
            for p in [
                "numpy",
                "pandas",
                "scipy",
                "scikit-learn",
                "joblib",
                "pytest",
                "pytest-socket",
                "black",
                "ruff",
            ]
        },
        "checks": [],
    }
    commands = [
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/test_dimred_run_reporter.py",
            "-W",
            "error",
            "--junitxml=test-results/m27-final-targeted.xml",
        ],
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "--junitxml=test-results/m27-final-fast.xml",
        ],
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "--import-mode=importlib",
            "--junitxml=test-results/m27-final-importlib.xml",
        ],
        [
            sys.executable,
            "-m",
            "black",
            "--check",
            "src",
            "tests",
            "scripts",
            "app.py",
            "audit/check_m27_mutations.py",
            "audit/check_m27_boundaries.py",
            "audit/check_m27_validation.py",
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
            "audit/check_m27_mutations.py",
            "audit/check_m27_boundaries.py",
            "audit/check_m27_validation.py",
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
            "requirements",
            "pyproject.toml",
            ".github",
            "scripts",
            "app.py",
            "AGENTS.md",
        ],
        ["git", "status", "--short", "--branch"],
    ]
    if args.review:
        commands[0].insert(5, "tests/test_reporter_review_contracts.py")
        for command in commands:
            if "black" in command or "ruff" in command:
                command.extend(
                    ["audit/check_m27_review.py", "audit/check_m27_reference.py"]
                )
    for command in commands:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            env=dict(os.environ, PYTHONIOENCODING="utf-8", MPLBACKEND="Agg"),
            timeout=300,
        )
        check = {
            "command": command,
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
        result["checks"].append(check)
        output.write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(" ".join(command), "EXIT", completed.returncode, flush=True)
        print(completed.stdout[-1500:], flush=True)
    result["protected_unchanged"] = before == hashes()
    output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    assert result["protected_unchanged"]
    assert all(c["returncode"] == 0 for c in result["checks"])


if __name__ == "__main__":
    main()
