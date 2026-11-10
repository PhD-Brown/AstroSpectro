"""Replay bounded semantic reporter mutations in disposable synthetic trees."""

import argparse
import ast
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
SOURCE = "src/dimred/run_reporter.py"
TEST = "tests/test_dimred_run_reporter.py"

# Each old site must occur exactly once. Controls are outside the mutant count.
MUTANTS = [
    (
        "M01_mean_to_median",
        "float(np.nanmean(arr[:, 0]))",
        "float(np.nanmedian(arr[:, 0]))",
    ),
    (
        "M02_sample_std",
        "float(np.nanstd(arr[:, 0]))",
        "float(np.nanstd(arr[:, 0], ddof=1))",
    ),
    (
        "M03_wrong_axis",
        'out[f"axis_{i+1}_mean"] = float(np.nanmean(vals))',
        'out[f"axis_{i+1}_mean"] = float(np.nanmean(arr[0]))',
    ),
    (
        "M04_wrong_percentile",
        "float(np.percentile(vals, 5))",
        "float(np.percentile(vals, 25))",
    ),
    ("M05_vector_median", "float(np.mean(vals))", "float(np.median(vals))"),
    ("M06_include_noise", "if c != -1}", "if True}"),
    ("M07_pca_rows", '"scores": scores,', '"scores": np.roll(scores, 1, axis=0),'),
    (
        "M08_pca_labels",
        '"scores_95pct": scores[:, :n_keep],\n        "y": y,',
        '"scores_95pct": scores[:, :n_keep],\n        "y": np.roll(y, 1),',
    ),
    (
        "M09_pca_components",
        "n_keep = min(int(pca.n_components_for_variance(0.95)), max_pcs_features)",
        "n_keep = min(1, max_pcs_features)",
    ),
    (
        "M10_latest_path",
        'path_latest = run_dir / f"{stem_latest}.joblib"',
        'path_latest = run_dir / f"wrong_{stem_latest}.joblib"',
    ),
    ("M11_galaxy_alias", '"GLAXAXY": "GALAXY"', '"GLAXAXY": "STAR"'),
    (
        "M12_confusion_classes",
        'pd.Series(yp, name="pred")',
        'pd.Series(np.roll(yp, 1), name="pred")',
    ),
    (
        "M13_best_is_last",
        "best_val_loss = float(min(val_loss))",
        "best_val_loss = float(val_loss[-1])",
    ),
    (
        "M14_wrong_pca_baseline",
        "pca.reconstruction_error(X, n_components=2).mean()",
        "pca.reconstruction_error(X, n_components=1).mean()",
    ),
    (
        "M15_latent_rows",
        '"Z_ae": Z_ae_current,',
        '"Z_ae": np.roll(Z_ae_current, 1, axis=0),',
    ),
    (
        "M16_wrong_model_path",
        '"ae_model_path": str(ae_model_path) if ae_model_path is not None else None,',
        '"ae_model_path": str(FIGURES_DIR),',
    ),
    (
        "M17_wrong_spearman_axis",
        '"latent_axis_1": float(r1) if pd.notna(r1) else None,',
        '"latent_axis_1": float(r2) if pd.notna(r2) else None,',
    ),
    (
        "M18_timestamp_metadata",
        '"timestamp_utc": timestamp,',
        '"timestamp_utc": "19990101T000000Z",',
    ),
]
CONTROLS = [
    ("C01_equivalent_mean", "float(np.mean(vals))", "float(np.sum(vals) / vals.size)"),
    (
        "C02_dict_order",
        "indent=2, ensure_ascii=False, default=json_default",
        "indent=2, ensure_ascii=False, default=json_default, sort_keys=True",
    ),
    (
        "C03_text_spacing",
        'text_content = "\\n".join(text_lines)',
        'text_content = "\\n\\n".join(text_lines)',
    ),
    (
        "C04_strict_json_refusal",
        "indent=2, ensure_ascii=False, default=json_default",
        "indent=2, ensure_ascii=False, default=json_default, allow_nan=False",
    ),
    (
        "C05_alternative_float_dtype",
        "arr = np.asarray(Z)\n    if arr.ndim != 2 or",
        "arr = np.asarray(Z, dtype=np.float32)\n    if arr.ndim != 2 or",
    ),
    (
        "C06_explicit_type_refusal",
        'raise TypeError(f"Type non sérialisable : {type(obj)!r}")',
        'raise TypeError(f"Unsupported JSON value: {type(obj)!r}")',
    ),
]


def digest(path):
    return hashlib.sha256(path.read_text(encoding="utf-8").encode()).hexdigest()


def execute(tree, name, reverse=False, proofs=False):
    # A trace limited to this source proves that the changed site was executed.
    plugin = tree / "m27_trace_plugin.py"
    plugin.write_text(
        "import json,sys\nfrom pathlib import Path\n"
        "lines=set()\n"
        "def trace(frame,event,arg):\n"
        "    if frame.f_code.co_filename.endswith('dimred/run_reporter.py') or frame.f_code.co_filename.endswith('dimred\\\\run_reporter.py'):\n"
        "        if event=='line': lines.add(frame.f_lineno)\n"
        "        return trace\n"
        "    return None\n"
        "def pytest_sessionstart(session): sys.settrace(trace)\n"
        "def pytest_sessionfinish(session,exitstatus):\n"
        "    sys.settrace(None)\n"
        "    Path('executed.json').write_text(json.dumps(sorted(lines)))\n"
        + (
            "def pytest_collection_modifyitems(items): items.reverse()\n"
            if reverse
            else ""
        ),
        encoding="utf-8",
    )
    target = "tests/proofs_dimred_run_reporter.py" if proofs else TEST
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        target,
        "--tb=short",
        "-p",
        "m27_trace_plugin",
        "--junitxml=trial.xml",
    ]
    if not proofs:
        command += ["-W", "error"]
    env = dict(os.environ, PYTHONHASHSEED="7", PYTHONIOENCODING="utf-8")
    completed = subprocess.run(
        command,
        cwd=tree,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=120,
    )
    xml = ET.parse(tree / "trial.xml").getroot()
    cases = []
    for case in xml.iter("testcase"):
        failure, error, skipped = (
            case.find("failure"),
            case.find("error"),
            case.find("skipped"),
        )
        cases.append(
            {
                "name": case.attrib["name"],
                "status": (
                    "ERROR"
                    if error is not None
                    else (
                        "FAIL"
                        if failure is not None
                        else "SKIP" if skipped is not None else "PASS"
                    )
                ),
                "message": (
                    failure.attrib.get("message", "") if failure is not None else ""
                ),
                "detail": failure.text if failure is not None else None,
            }
        )
    return {
        "name": name,
        "command": command,
        "returncode": completed.returncode,
        "counts": {
            s: sum(c["status"] == s for c in cases)
            for s in ["PASS", "FAIL", "ERROR", "SKIP"]
        },
        "cases": cases,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "executed_lines": json.loads((tree / "executed.json").read_text()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path, default=ROOT / "audit/m27_mutation_results.json"
    )
    parser.add_argument("--reverse", action="store_true")
    parser.add_argument("--proofs", action="store_true")
    parser.add_argument(
        "--proofs-ref", default="51fe4d1ff3645aea28cda59f0260acc86fa0c834"
    )
    parser.add_argument("--only", nargs="*")
    args = parser.parse_args()
    protected = {p: digest(ROOT / p) for p in [SOURCE, TEST, "tests/conftest.py"]}
    source = (ROOT / SOURCE).read_text(encoding="utf-8")
    ledger = {
        "schema": 1,
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "tested_lf_sha256": protected,
        "python": sys.version,
        "trials": [],
    }
    with tempfile.TemporaryDirectory(prefix="m27-synthetic-replay-") as folder:
        tree = Path(folder)
        paths = [
            SOURCE,
            TEST,
            "src/dimred/pca_analyzer.py",
            "tests/conftest.py",
            "pyproject.toml",
        ]
        if args.proofs:
            paths.append("tests/proofs_dimred_run_reporter.py")
        for path in paths:
            (tree / path).parent.mkdir(parents=True, exist_ok=True)
            if args.proofs and path in [
                "tests/proofs_dimred_run_reporter.py",
                "tests/conftest.py",
            ]:
                content = subprocess.check_output(
                    ["git", "show", f"{args.proofs_ref}:{path}"], cwd=ROOT
                )
                (tree / path).write_bytes(content)
                ledger.setdefault("archived_inputs", {})[path] = {
                    "ref": args.proofs_ref,
                    "lf_sha256": digest(tree / path),
                }
            else:
                shutil.copyfile(ROOT / path, tree / path)
        trials = (
            [("baseline", None, None)]
            if args.reverse or args.proofs
            else [("baseline", None, None)] + MUTANTS + CONTROLS
        )
        if args.only:
            trials = [
                trial
                for trial in trials
                if trial[0] == "baseline" or trial[0] in args.only
            ]
        baseline_names = None
        for name, old, new in trials:
            line = None
            patched = source
            if old is not None:
                assert source.count(old) == 1, (name, source.count(old))
                line = source[: source.index(old)].count("\n") + 1
                patched = source.replace(old, new)
                ast.parse(patched)
            (tree / SOURCE).write_text(patched, encoding="utf-8")
            # Avoid equal-size and sub-second timestamp bytecode cache collisions.
            for cache in tree.rglob("__pycache__"):
                shutil.rmtree(cache)
            result = execute(tree, name, args.reverse, args.proofs)
            names = sorted(c["name"] for c in result["cases"])
            if baseline_names is None:
                baseline_names = names
            result.update(
                old=old,
                new=new,
                site_line=line,
                site_executed=line in result["executed_lines"] if line else None,
                collection_matches=names == baseline_names,
                active_source_lf_sha256=digest(tree / SOURCE),
            )
            if args.proofs:
                result["outcome"] = "RED_PROOF_REPLAY"
            elif name == "baseline" or name.startswith("C"):
                result["outcome"] = (
                    "PASS" if result["returncode"] == 0 else "CONTROL_FAILED"
                )
            elif (
                result["counts"]["FAIL"]
                and not result["counts"]["ERROR"]
                and result["site_executed"]
                and result["collection_matches"]
            ):
                failures = [c for c in result["cases"] if c["status"] == "FAIL"]
                result["outcome"] = (
                    "KILLED"
                    if all(
                        "AssertionError" in (c["detail"] or "")
                        or "\nE   assert " in (c["detail"] or "")
                        for c in failures
                    )
                    else "NON_ASSERTION_FAILURE"
                )
            else:
                result["outcome"] = (
                    "SURVIVED" if result["returncode"] == 0 else "INCONCLUSIVE"
                )
            ledger["trials"].append(result)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(ledger, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            print(
                name,
                result["outcome"],
                result["counts"],
                "site",
                result["site_executed"],
                flush=True,
            )
    assert protected == {
        p: digest(ROOT / p) for p in protected
    }, "Protected input changed"
    if args.proofs:
        assert all(t["counts"]["ERROR"] == 0 for t in ledger["trials"])
    else:
        assert all(t["outcome"] in ["PASS", "KILLED"] for t in ledger["trials"])


if __name__ == "__main__":
    main()
