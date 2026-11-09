"""Replay scoped review candidates in temporary copies; never edit production.

Expected red baselines, incomplete remedies and admissible controls are distinct.
Each record declares its exact pytest selection, assertion diagnostics and scope.
"""

import argparse
import ast
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

from check_m26_mutations import TRACE_PLUGIN


TARGET = "src/dimred/xgboost_bridge.py"
PROOF = "tests/proofs_dimred_xgboost_bridge_review.py"
GREEN = [
    "tests/test_dimred_xgboost_bridge.py",
    "tests/test_dimred_xgboost_bridge_real.py",
]
ALIGN = "    if len(df_aligned) != len(Z_umap):"
LENGTH_GUARD = '    if not (len(df_aligned) == len(Z_umap) == len(y) == len(cluster_labels)):\n        raise ValueError("Observation alignment length mismatch")\n'
IDENTITY_GUARD = '    if "id_echo" in df_aligned and not np.array_equal(df_aligned["id_echo"].to_numpy(), Z_umap[:, 0]):\n        raise ValueError("Observation identity mismatch")\n'
SCHEMA = "X_pred_df = df_aligned[needed].fillna(0)"
MAXIMUM = "confidence = proba.max(axis=1)"
PRESERVE_NAN = [
    ("df_aligned[f] = 0.0", "df_aligned[f] = np.nan"),
    (SCHEMA, "X_pred_df = df_aligned[needed].copy()"),
]
PROBA_LINE = "        proba = clf.model_pipeline.predict_proba(X_pred_df)"
PROBA_GUARD = """
        if (np.ndim(proba) != 2 or proba.shape[0] != len(X_pred_df)
            or proba.shape[1] == 0 or not np.isfinite(proba).all()
            or (proba < 0).any() or (proba > 1).any()
            or not np.allclose(proba.sum(axis=1), 1)):
            raise ValueError("Probability distribution invalid shape or values")"""


def replace(source, changes):
    for old, new in changes:
        assert source.count(old) == 1, (old, source.count(old))
        source = source.replace(old, new, 1)
    ast.parse(source)
    return source


def candidates(phase):
    if phase == "r1":
        selection = [PROOF, "-k", "loader_population"]
        return [
            (
                "baseline",
                [],
                selection,
                3,
                1,
                "red baseline: true loader, unchanged bridge",
            ),
            (
                "length_refusal_only",
                [(ALIGN, LENGTH_GUARD + ALIGN)],
                selection,
                2,
                2,
                "incomplete remedy must still fail equal-size default/NB01 cases",
            ),
            (
                "synthetic_identity_refusal",
                [(ALIGN, IDENTITY_GUARD + ALIGN)],
                selection,
                0,
                4,
                "admissible control only: fixture embedding column0 explicitly stores IDs",
            ),
            (
                "synthetic_shared_population",
                [
                    (
                        "df_aligned = df_f.dropna(subset=pca_feat_cols).reset_index(drop=True)",
                        'df_aligned = df_raw.set_index("obsid").loc[Z_umap[:, 0].astype(int)].reset_index()',
                    )
                ],
                selection,
                0,
                4,
                "admissible synthetic population transport, not proposed production architecture",
            ),
        ]
    if phase == "r2":
        selection = [PROOF, "-k", "fitted_train_median or complete_finite"]
        mean_fill = """X_pred_df = df_aligned[needed].copy()
    prep = clf.model_pipeline.named_steps["named"].named_transformers_["numeric"]
    for index, name in enumerate(needed):
        X_pred_df[name] = X_pred_df[name].fillna(prep.named_steps["scaler"].mean_[index])"""
        reject = """X_pred_df = df_aligned[needed].copy()
    if X_pred_df.isna().any().any():
        raise ValueError("Input preprocessing incompatible with missing values")"""
        return [
            (
                "r2_baseline",
                [],
                selection,
                2,
                1,
                "red booster-matrix oracle, not labels alone",
            ),
            (
                "inference_batch_median",
                [
                    (
                        SCHEMA,
                        "X_pred_df = df_aligned[needed].fillna(df_aligned[needed].median(numeric_only=True))",
                    )
                ],
                selection,
                2,
                1,
                "forbidden inference-fitted statistics detected",
            ),
            (
                "trained_mean_not_saved_median",
                [(SCHEMA, mean_fill)],
                selection,
                2,
                1,
                "wrong mean gives post-scale zero; beta/gamma oracle still rejects",
            ),
            (
                "invented_seven",
                [(SCHEMA, "X_pred_df = df_aligned[needed].fillna(7)")],
                selection,
                2,
                1,
                "changed placeholder still violates trained transformation",
            ),
            (
                "preserve_present_missing_values",
                [(SCHEMA, "X_pred_df = df_aligned[needed].copy()")],
                selection,
                0,
                3,
                "present NaNs pass to trained pipeline; absent-column policy separately exercised by full green control",
            ),
            (
                "explicit_preprocessing_refusal",
                [(SCHEMA, reject)],
                selection,
                0,
                3,
                "diagnosed refusal accepted; finite positive case still succeeds",
            ),
        ]
    if phase == "r3":
        selection = [
            PROOF,
            "-k",
            "incompatible_vectors or undefined_probabilities or tensor_probabilities or valid_vectors",
        ]
        crop = [
            ("        y_aligned = y\n", "        y_aligned = y[:len(df_aligned)]\n"),
            (
                "        cl_aligned = cluster_labels\n",
                "        cl_aligned = cluster_labels[:len(df_aligned)]\n",
            ),
        ]
        return [
            (
                "r3_baseline",
                [],
                selection,
                6,
                1,
                "six red cases plus valid-output positive control",
            ),
            (
                "invented_nan_confidence",
                [(MAXIMUM, "confidence = np.nan_to_num(proba, nan=0.0).max(axis=1)")],
                selection,
                6,
                1,
                "Gaussian oracle no longer accepts invented confidence",
            ),
            (
                "silent_vector_crop",
                crop,
                selection,
                6,
                1,
                "both too-long vectors still fail on original values, not merely length",
            ),
            (
                "tensor_flattened",
                [(MAXIMUM, "confidence = proba.reshape(len(proba), -1).max(axis=1)")],
                selection,
                6,
                1,
                "tensor oracle requires unavailable/refusal, not scalar shape alone",
            ),
            (
                "explicit_vector_refusal",
                [(ALIGN, LENGTH_GUARD + ALIGN)],
                selection,
                2,
                5,
                "all four incompatible input vectors refused; output defects remain",
            ),
            (
                "probability_unavailable",
                [(PROBA_LINE, PROBA_LINE + PROBA_GUARD)],
                selection,
                4,
                3,
                "invalid distributions unavailable; input-vector defects remain",
            ),
            (
                "both_boundary_guards",
                [(ALIGN, LENGTH_GUARD + ALIGN), (PROBA_LINE, PROBA_LINE + PROBA_GUARD)],
                selection,
                0,
                7,
                "admissible conditional validation, no blanket refusal",
            ),
        ]
    if phase == "all_proofs":
        changes = [(SCHEMA, "X_pred_df = df_aligned[needed].copy()")] + [
            (ALIGN, LENGTH_GUARD + IDENTITY_GUARD + ALIGN),
            (PROBA_LINE, PROBA_LINE + PROBA_GUARD),
        ]
        return [
            (
                "all_proof_baseline",
                [],
                [PROOF],
                13,
                3,
                "all supplementary proofs on unchanged source",
            ),
            (
                "combined_admissible_control",
                changes,
                [PROOF],
                0,
                16,
                "synthetic identity guard, trained preprocessing and output validation; not a production patch",
            ),
        ]
    if phase == "green":
        reject_missing = [
            (
                "        for f in missing:",
                '        raise ValueError("Feature schema missing required columns")\n        for f in missing:',
            )
        ]
        strings = """X_pred_df = df_aligned[needed].fillna(0)
    if any(pd.api.types.is_string_dtype(X_pred_df[c]) for c in X_pred_df.columns):
        raise ValueError("Feature schema contains unsupported non-numeric data")"""
        crop = [
            ("        y_aligned = y\n", "        y_aligned = y[:len(df_aligned)]\n"),
            (
                "        cl_aligned = cluster_labels\n",
                "        cl_aligned = cluster_labels[:len(df_aligned)]\n",
            ),
        ]
        rows = (
            SCHEMA
            + """
    if len(X_pred_df) >= 4:
        X_pred_df = X_pred_df.iloc[[3, 1, 2, 0, *range(4, len(X_pred_df))]].reset_index(drop=True)"""
        )
        source = (Path(__file__).resolve().parents[1] / TARGET).read_text(
            encoding="utf-8"
        )
        start = source.index("    y_pred_enc = clf.model_pipeline.predict(X_pred_df)")
        end = source.index(
            "\n    try:\n        # Si le modèle expose predict_proba", start
        )
        prediction = source[start:end]
        structured = (
            "    try:\n"
            + "\n".join("    " + line for line in prediction.splitlines())
            + """
    except ValueError as exc:
        if not any(token in str(exc).lower() for token in
                   ("unsupported features", "invalid class codes", "infinity", "too large")):
            raise
        logger.error("Prediction features incompatible: %s", exc)
        return _empty_result()
"""
        )
        discovery = 'latest_file(paths["MODELS_DIR"], "spectral_classifier*.pkl")'
        return [
            ("green_baseline", [], GREEN, 0, 71, "entire revised M2.6 suite"),
            (
                "complete_nan_passthrough",
                PRESERVE_NAN,
                GREEN,
                0,
                71,
                "admissible complete #89 policy",
            ),
            (
                "missing_schema_refusal",
                reject_missing,
                GREEN,
                0,
                71,
                "admissible explicit schema refusal",
            ),
            (
                "float64_representation",
                [(SCHEMA, SCHEMA + '.astype("float64")')],
                GREEN,
                0,
                71,
                "lossless fixture values and real pipeline parity",
            ),
            (
                "float32_representation",
                [(SCHEMA, SCHEMA + '.astype("float32")')],
                GREEN,
                0,
                71,
                "fixture-compatible representation; actual precision checked",
            ),
            (
                "string_refusal",
                [(SCHEMA, strings)],
                GREEN,
                0,
                71,
                "explicit non-numeric refusal permitted",
            ),
            (
                "diagnosed_postload_refusal",
                [(prediction, structured)],
                GREEN,
                0,
                71,
                "R6 propagated or structured error both admissible",
            ),
            (
                "predicted_class_probability",
                [
                    (
                        MAXIMUM,
                        "confidence = proba[np.arange(len(y_pred_enc)), y_pred_enc]",
                    )
                ],
                GREEN,
                0,
                71,
                "R6 alternative confidence semantics permitted consistently",
            ),
            (
                "inference_batch_median",
                [
                    (
                        SCHEMA,
                        "X_pred_df = df_aligned[needed].fillna(df_aligned[needed].median(numeric_only=True))",
                    )
                ],
                GREEN,
                0,
                71,
                "M04 still survives GREEN, separately detected by red R2 proofs",
            ),
            (
                "invented_nan_confidence",
                [(MAXIMUM, "confidence = np.nan_to_num(proba, nan=0.0).max(axis=1)")],
                GREEN,
                0,
                71,
                "M08 still survives GREEN, rejected by new Gaussian proof",
            ),
            (
                "silent_vector_crop",
                crop,
                GREEN,
                0,
                71,
                "M09 still survives GREEN, rejected by new cardinality proofs",
            ),
            (
                "fillna_seven",
                [(SCHEMA, "X_pred_df = df_aligned[needed].fillna(7)")],
                GREEN,
                0,
                71,
                "full-suite survivor; never approve historical zero just to kill this",
            ),
            (
                "broad_model_family",
                [(discovery, 'latest_file(paths["MODELS_DIR"], "*.pkl")')],
                GREEN,
                2,
                69,
                "M10 now detected by both artifact-family discovery cases",
            ),
            (
                "rows_swapped_same_class",
                [(SCHEMA, rows)],
                GREEN,
                None,
                None,
                "values/identity assertions must detect row permutation",
            ),
            (
                "r7_real_baseline",
                [],
                [GREEN[1]],
                0,
                5,
                "real pipeline selection, includes passive booster-input observations",
            ),
            (
                "r7_real_same_class_swap",
                [(SCHEMA, rows)],
                [GREEN[1]],
                3,
                2,
                "R7 now detected by actual transformed inputs despite identical labels/confidences",
            ),
        ]
    raise ValueError(phase)


def git(root, *args):
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


def fingerprint(path):
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def execute(directory, selection):
    for file in (directory / "src/dimred").glob("__pycache__/xgboost_bridge.*.pyc"):
        file.unlink()
    for name in ("trace.json", "result.xml"):
        (directory / name).unlink(missing_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        *selection,
        "-p",
        "m26_trace",
        "--tb=short",
        "--junitxml=result.xml",
        "-W",
        "error",
    ]
    env = os.environ.copy()
    env.update(PYTHONIOENCODING="utf-8", PYTHONHASHSEED="7", MPLBACKEND="Agg")
    result = subprocess.run(
        cmd,
        cwd=directory,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    cases = []
    if (directory / "result.xml").exists():
        for case in ET.parse(directory / "result.xml").iter("testcase"):
            failure, error, skipped = (
                case.find("failure"),
                case.find("error"),
                case.find("skipped"),
            )
            detail = error if error is not None else failure
            cases.append(
                {
                    "name": case.attrib["name"],
                    "outcome": (
                        "ERROR"
                        if error is not None
                        else (
                            "FAIL"
                            if failure is not None
                            else "SKIP" if skipped is not None else "PASS"
                        )
                    ),
                    "detail": detail.text if detail is not None else None,
                }
            )
    return {
        "command": cmd,
        "returncode": result.returncode,
        "cases": cases,
        "executed_bridge_lines": (
            json.loads((directory / "trace.json").read_text())
            if (directory / "trace.json").exists()
            else []
        ),
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", choices=["r1", "r2", "r3", "all_proofs", "green"], required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--only",
        nargs="+",
        help="Replay selected records with their unchanged-source positive baselines.",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    paths = [
        TARGET,
        "src/dimred/data_loader.py",
        "src/utils.py",
        "src/pipeline/classifier.py",
        "tests/conftest.py",
        *GREEN,
        PROOF,
        "pyproject.toml",
        "audit/check_m26_mutations.py",
        "audit/check_m26_review_mutations.py",
    ]
    inputs = {
        name: {
            "sha256_lf": fingerprint(root / name),
            "git_blob_at_head": git(root, "rev-parse", f"HEAD:{name}"),
        }
        for name in paths
    }
    head, dirty = git(root, "rev-parse", "HEAD"), git(root, "status", "--porcelain")
    source = (root / TARGET).read_text(encoding="utf-8")
    records = []
    collections = {}
    with tempfile.TemporaryDirectory(prefix="astrospectro-m26-review-") as temporary:
        directory = Path(temporary)
        for name in paths:
            if name.startswith("audit/"):
                continue
            destination = directory / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes((root / name).read_bytes())
        (directory / "m26_trace.py").write_text(TRACE_PLUGIN, encoding="utf-8")
        planned = candidates(args.phase)
        if args.only:
            assert set(args.only).issubset({c[0] for c in planned})
            scopes = {tuple(c[2]) for c in planned if c[0] in args.only}
            planned = [
                c
                for c in planned
                if c[0] in args.only or (not c[1] and tuple(c[2]) in scopes)
            ]
        for name, changes, selection, want_fail, want_pass, meaning in planned:
            mutated = replace(source, changes)
            (directory / TARGET).write_text(mutated, encoding="utf-8")
            outcome = execute(directory, selection)
            counts = {
                key: sum(c["outcome"] == key for c in outcome["cases"])
                for key in ("FAIL", "PASS", "ERROR", "SKIP")
            }
            assertion_only = all(
                re.search(r"(?m)^E\s+(?:AssertionError|assert\s)", c["detail"] or "")
                for c in outcome["cases"]
                if c["outcome"] == "FAIL"
            )
            case_names = [c["name"] for c in outcome["cases"]]
            if not changes:
                collections[tuple(selection)] = case_names
            collection_matches = case_names == collections[tuple(selection)]
            changed_sites = []
            for old, new in changes:
                position = mutated.index(new)
                begin = mutated[:position].count("\n") + 1
                possible = set(range(begin, begin + new.count("\n") + 1))
                changed_sites.append(
                    sorted(possible & set(outcome["executed_bridge_lines"]))
                )
            matched = (
                counts["ERROR"] == counts["SKIP"] == 0
                and (
                    counts["FAIL"] > 0
                    if want_fail is None
                    else counts["FAIL"] == want_fail
                )
                and (want_pass is None or counts["PASS"] == want_pass)
                and collection_matches
                and all(changed_sites)
            )
            matched &= (
                outcome["returncode"] == (1 if counts["FAIL"] else 0) and assertion_only
            )
            records.append(
                {
                    "name": name,
                    "scope": selection,
                    "patches": changes,
                    "meaning": meaning,
                    "expected_fail": want_fail,
                    "expected_pass": want_pass,
                    "counts": counts,
                    "assertion_failures_only": assertion_only,
                    "collection_matches_baseline": collection_matches,
                    "changed_sites_executed": changed_sites,
                    "expectation_met": matched,
                    **outcome,
                }
            )
            print(name, counts, "EXPECTED" if matched else "UNEXPECTED", flush=True)
    assert all(
        fingerprint(root / name) == item["sha256_lf"] for name, item in inputs.items()
    )
    assert git(root, "rev-parse", "HEAD") == head
    payload = {
        "phase": args.phase,
        "selected_records": args.only,
        "git_head_at_run": head,
        "working_tree_at_run": dirty,
        "python": sys.version,
        "versions": {
            p: importlib.metadata.version(p)
            for p in ["numpy", "pandas", "scikit-learn", "xgboost", "pytest"]
        },
        "inputs": inputs,
        "results": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return 0 if all(r["expectation_met"] for r in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
