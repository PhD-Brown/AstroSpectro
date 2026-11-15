"""Replay M3.1 validation or targeted source-copy counter-experiments.

The default report goes to ignored test-results; publishing a frozen report
requires an explicit --output. No source in src/ is ever written.
"""

import argparse
import ast
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed

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
        ROOT / "audit/m31_review_candidates.py",
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


def pytest_run(selection, xml, *, candidate=None, trace_dir=None):
    env = os.environ.copy()
    env.update(
        PYTHONPATH=str(ROOT / "src"),
        MPLBACKEND="Agg",
        PYTHONHASHSEED="0",
        PYTHONIOENCODING="utf-8",
    )
    env.pop("M31_VISUALIZER_SOURCE", None)
    env.pop("M31_TRACE_DIR", None)
    if candidate:
        env["M31_VISUALIZER_SOURCE"] = str(candidate)
    if trace_dir:
        env["M31_TRACE_DIR"] = str(trace_dir)
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
                or (
                    "AssertionError"
                    if c.find("failure").get("message", "").startswith("assert ")
                    else c.find("failure").get("message", "").split(":", 1)[0]
                ),
                "message": c.find("failure").get("message"),
            }
            for c in cases
            if c.find("failure") is not None
        ]
        result["outcomes"] = {
            c.get("name"): {
                "status": (
                    "ERROR"
                    if c.find("error") is not None
                    else (
                        "FAIL"
                        if c.find("failure") is not None
                        else "SKIP" if c.find("skipped") is not None else "PASS"
                    )
                ),
                "message": (
                    c.find("failure").get("message", "")
                    if c.find("failure") is not None
                    else ""
                )[:700],
            }
            for c in cases
        }
    if trace_dir:
        traces = [
            json.loads(p.read_text(encoding="utf-8")) for p in trace_dir.glob("*.json")
        ]
        result["reached_lines"] = sorted({n for t in traces for n in t["lines"]})
        result["trace_nodes"] = [t["node"] for t in traces]
    return result


def replace(source, old, new, *, expected_count=1):
    assert (
        source.count(old) == expected_count
    ), f"Candidate occurrence mismatch: {old[:60]}"
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


def adversarial(report, only=None):
    """Cross every remedy with all green and red contracts; trace mutants."""
    spec = importlib.util.spec_from_file_location(
        "m31_candidates", ROOT / "audit/m31_review_candidates.py"
    )
    builders = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builders)
    source = SOURCE.read_text(encoding="utf-8")
    historical = json.loads(
        (ROOT / "audit/m31_adversarial.json").read_text(encoding="utf-8")
    )
    items = builders.candidates(source, historical, local_style)
    if only:
        wanted = set(only.split(","))
        items = [item for item in items if item["id"] in wanted]
        assert {item["id"] for item in items} == wanted
    report["candidate_selection"] = only or "all"

    def compact(result):
        result["output"] = "\n".join(result["output"].splitlines()[-6:])
        return result

    with tempfile.TemporaryDirectory(prefix="m31-review-") as temp:
        folder = Path(temp)
        baseline_green = pytest_run([GREEN], folder / "baseline-green.xml")
        baseline_red = pytest_run([RED], folder / "baseline-red.xml")
        assert baseline_green["exit_code"] == 0
        report["baseline_green"] = compact(baseline_green)
        report["baseline_red"] = compact(baseline_red)

        def evaluate(item):
            case_dir = folder / item["id"]
            case_dir.mkdir()
            path = case_dir / "visualizer.py"
            path.write_text(item["source"], encoding="utf-8")
            record = {k: v for k, v in item.items() if k not in {"source", "reference"}}
            record["candidate_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            record["changed_methods"] = [
                method["name"]
                for method in inventory(item["source"])["methods"]
                if any(
                    method["line"] <= line <= method["end_line"]
                    for line in item["changed_lines"]
                )
            ]
            if item["kind"].startswith("mutant"):
                selection = GREEN if item["kind"] == "mutant_green" else RED
                reference = None
                if item["reference"] is not None:
                    reference = case_dir / "reference.py"
                    reference.write_text(item["reference"], encoding="utf-8")
                record["reference_sha256"] = hashlib.sha256(
                    reference.read_bytes() if reference else SOURCE.read_bytes()
                ).hexdigest()
                record["reference_kind"] = (
                    "admissible physical/statistical reference copy"
                    if reference
                    else "unchanged production"
                )
                baseline = pytest_run(
                    [selection, "-k", item["target"]],
                    case_dir / "reference.xml",
                    candidate=reference,
                )
                outcome = pytest_run(
                    [selection, "-k", item["target"]],
                    case_dir / "mutation.xml",
                    candidate=path,
                    trace_dir=case_dir / "trace",
                )
                reached = set(outcome.get("reached_lines", []))
                # Mutants only alter executable expressions, not formatting.
                hit = bool(item["changed_lines"]) and set(
                    item["changed_lines"]
                ).issubset(reached)
                assertion = outcome.get("counts", {}).get("failed", 0) > 0 and all(
                    f["type"] == "AssertionError" for f in outcome.get("failures", [])
                )
                same = baseline.get("counts", {}).get("collected") == outcome.get(
                    "counts", {}
                ).get("collected")
                killed = (
                    baseline["exit_code"] == 0
                    and baseline["counts"]["passed"] > 0
                    and assertion
                    and same
                    and hit
                    and outcome["counts"]["errors"] == 0
                )
                record.update(
                    status=(
                        "KILLED"
                        if killed
                        else (
                            "SURVIVED"
                            if outcome["exit_code"] == 0 and hit
                            else "INCONCLUSIVE"
                        )
                    ),
                    baseline=compact(baseline),
                    result=compact(outcome),
                    mutation_reached=hit,
                    reached_changed_lines=sorted(reached & set(item["changed_lines"])),
                )
            else:
                green = pytest_run([GREEN], case_dir / "green.xml", candidate=path)
                red = pytest_run([RED], case_dir / "red.xml", candidate=path)
                # Ask pytest to resolve -k semantics; never infer a targeted
                # PASS from an empty or mismatched selection.
                target = pytest_run(
                    [RED, "-k", item["target"]], case_dir / "target.xml", candidate=path
                )
                target_ok = (
                    target["exit_code"] == 0
                    and target.get("counts", {}).get("passed", 0) > 0
                )
                green_ok = green["exit_code"] == 0
                previous = baseline_red.get("outcomes", {})
                regressions = [
                    name
                    for name, state in red.get("outcomes", {}).items()
                    if previous.get(name, {}).get("status") == "PASS"
                    and state["status"] != "PASS"
                ]
                incidental = red.get("counts", {}).get("errors", 0) > 0 or any(
                    f["type"] != "AssertionError" for f in red.get("failures", [])
                )
                status = (
                    "ADMITTED_BOUNDED"
                    if target_ok and green_ok and not regressions and not incidental
                    else (
                        "REGRESSION"
                        if target_ok and (not green_ok or regressions)
                        else (
                            "INCONCLUSIVE"
                            if incidental or target.get("counts", {}).get("errors", 0)
                            else "TARGET_REJECTED"
                        )
                    )
                )
                record.update(
                    status=status,
                    target=compact(target),
                    green=compact(green),
                    red=compact(red),
                    other_regressions=regressions,
                    red_passed=[
                        name
                        for name, state in red.get("outcomes", {}).items()
                        if state["status"] == "PASS"
                    ],
                )
            print(record["id"], record["status"], flush=True)
            return record

        with ThreadPoolExecutor(max_workers=2) as pool:
            pending = [pool.submit(evaluate, item) for item in items]
            results = [future.result() for future in as_completed(pending)]
        report["adversarial"] = sorted(results, key=lambda item: item["id"])
        report["summary"] = {
            status: sum(item["status"] == status for item in results)
            for status in sorted({item["status"] for item in results})
        }


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
                "audit/m31_review_candidates.py",
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
                "audit/m31_review_candidates.py",
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
        "--only", help="Comma-separated candidate IDs for an explicit partial replay"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "test-results/m31-review-replay.json"
    )
    args = parser.parse_args()
    if args.output.resolve() in {
        (ROOT / "audit/m31_experiments.json").resolve(),
        (ROOT / "audit/m31_adversarial.json").resolve(),
    }:
        parser.error("Historical manifests are immutable; choose a review output path")
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
        adversarial(report, args.only)
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
