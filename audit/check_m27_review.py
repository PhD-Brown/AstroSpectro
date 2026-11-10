"""Replay Claude's false repairs and admissible alternatives in temporary trees."""

import argparse
import ast
import difflib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[1]
SOURCE = "src/dimred/run_reporter.py"
PROOF = "tests/proofs_dimred_run_reporter_review.py"
spec = importlib.util.spec_from_file_location(
    "m27_original_replay", ROOT / "audit/check_m27_mutations.py"
)
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)


def replace(source, old, new, count=1):
    assert source.count(old) == count, (old, source.count(old))
    return source.replace(old, new)


COLLISION = """
_review_joblib = save_joblib_pair
_review_json = save_json_txt_pair
def save_joblib_pair(obj, run_dir, stem_run, stem_latest, legacy_path=None):
    import joblib
    if (run_dir / (stem_run + '.joblib')).exists():
        {before}
        raise RuntimeError('Run identifier collision, nothing written')
    return _review_joblib(obj, run_dir, stem_run, stem_latest, legacy_path)
def save_json_txt_pair(report, run_dir, stem_run, stem_latest, text_lines):
    if (run_dir / (stem_run + '.json')).exists():
        {json_before}
        raise RuntimeError('Run identifier collision, nothing written')
    return _review_json(report, run_dir, stem_run, stem_latest, text_lines)
"""
ROLLBACK = """
_review_embeddings = save_umap_tsne_run
def save_umap_tsne_run(**kwargs):
    root = Path(kwargs['paths']['REPORTS_DIR'])
    before = {{str(p.relative_to(root)):p.read_bytes() for p in root.rglob('*') if p.is_file(){filter}}}
    try:
        return _review_embeddings(**kwargs)
    except Exception as exc:
        for p in root.rglob('*'):
            if p.is_file() and str(p.relative_to(root)) not in before{removal}:
                p.unlink()
        for relative, data in before.items():
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        raise RuntimeError('Publication aborted with retained cause') from exc
"""


def candidate(source, name):
    if name == "F9unique":
        return (
            source
            + """
_review_joblib = save_joblib_pair
_review_json = save_json_txt_pair
def _unique_stem(run_dir, stem, suffix):
    candidate, counter = stem, 1
    while (run_dir / (candidate + suffix)).exists():
        counter += 1
        candidate = stem + '_' + str(counter)
    return candidate
def save_joblib_pair(obj, run_dir, stem_run, stem_latest, legacy_path=None):
    return _review_joblib(obj, run_dir, _unique_stem(run_dir, stem_run, '.joblib'), stem_latest, legacy_path)
def save_json_txt_pair(report, run_dir, stem_run, stem_latest, text_lines):
    return _review_json(report, run_dir, _unique_stem(run_dir, stem_run, '.json'), stem_latest, text_lines)
"""
        )
    if name == "F10marker":
        return (
            source
            + """
_review_embeddings = save_umap_tsne_run
def save_umap_tsne_run(**kwargs):
    try:
        return _review_embeddings(**kwargs)
    except Exception as exc:
        root = Path(kwargs['paths']['REPORTS_DIR'])
        aliases = ['phy3500_embeddings_latest.joblib', 'phy3500_embeddings.joblib', 'phy3500_umap_tsne_run_latest.json', 'phy3500_umap_tsne_run_latest.txt']
        marker = {'status': 'incomplete', 'reason': str(exc), 'artifacts': aliases + [str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()]}
        (root / 'publication_status.json').write_text(json.dumps(marker), encoding='utf-8')
        raise RuntimeError('Publication explicitly unavailable') from exc
"""
        )
    if name == "F11log":
        source = replace(
            source,
            "except Exception:\n        # La sauvegarde reste possible même si cette métrique échoue.\n        mse_ae_global = None",
            'except Exception as exc:\n        logger.warning("AE reconstruction metric unavailable: %s", exc)\n        mse_ae_global = None',
        )
        return replace(
            source,
            "        except Exception:\n            pass\n\n    # ── Corrélations",
            '        except Exception as exc:\n            logger.warning("PCA reconstruction metric unavailable: %s", exc)\n\n    # ── Corrélations',
        )
    if name == "F1":
        return (
            source
            + """
_review_json = save_json_txt_pair
def save_json_txt_pair(report, *args, **kwargs):
    def clean(value):
        if isinstance(value, (float, np.floating)) and not np.isfinite(value): return 0.
        if isinstance(value, np.ndarray): return clean(value.tolist())
        if isinstance(value, dict): return {key:clean(v) for key,v in value.items()}
        if isinstance(value, (list, tuple)): return [clean(v) for v in value]
        return value
    return _review_json(clean(report), *args, **kwargs)
"""
        )
    if name in ["F2", "F9"]:
        return source + COLLISION.format(
            before=(
                "joblib.dump(obj, run_dir / (stem_latest + '.joblib'))\n        if legacy_path is not None: joblib.dump(obj, legacy_path)"
                if name == "F2"
                else "pass"
            ),
            json_before=(
                "_review_json(report, run_dir, 'temporary_wrong_run', stem_latest, text_lines)"
                if name == "F2"
                else "pass"
            ),
        )
    if name in ["F3", "F10"]:
        return source + ROLLBACK.format(
            filter=" and 'latest' in p.name" if name == "F3" else "",
            removal=" and 'latest' in p.name" if name == "F3" else "",
        )
    field = '"model_path": str(path_xgb) if path_xgb is not None else None,'
    if name in ["F4", "F4b", "F4c"]:
        new = {
            "F4": '"model_path": str(Path(path_xgb).with_suffix(".pkl")) if path_xgb else None,',
            "F4b": '"model_path": None, "model_artifact": str(path_xgb),',
            "F4c": '"model_path": None,',
        }[name]
        return replace(source, field, new)
    if name == "F5":
        return replace(
            source,
            '"metrics": {',
            '"diagnostics": {"error": None},\n        "metrics": {',
        )
    if name == "F5c":
        return (
            source
            + """
_review_ae = save_autoencoder_run
def save_autoencoder_run(**kwargs):
    logger.warning('PCA reconstruction_error baseline requested')
    return _review_ae(**kwargs)
"""
        )
    if name == "F8":
        return replace(
            source,
            "indent=2, ensure_ascii=False, default=json_default",
            "indent=2, ensure_ascii=False, default=json_default, allow_nan=False",
        )
    if name == "F11":
        source = replace(
            source,
            "except Exception:\n        # La sauvegarde reste possible même si cette métrique échoue.\n        mse_ae_global = None",
            'except Exception as exc:\n        raise RuntimeError("AE reconstruction metric failed") from exc',
        )
        return replace(
            source,
            "        except Exception:\n            pass\n\n    # ── Corrélations",
            '        except Exception as exc:\n            raise RuntimeError("PCA reconstruction metric failed") from exc\n\n    # ── Corrélations',
        )
    if name in ["F6", "F6c"]:
        old = "valid = np.isfinite(vals)"
        source = replace(
            source,
            old,
            "valid = np.isfinite(vals) & np.isfinite(Z_ae_current).all(axis=1)",
        )
        if name == "F6":
            source = replace(
                source,
                'pd.Series(vals[valid]), method="spearman"',
                'pd.Series(vals[:int(valid.sum())]), method="spearman"',
                count=2,
            )
        return source
    if name == "F6refuse":
        return (
            source
            + """
_review_ae = save_autoencoder_run
def save_autoencoder_run(**kwargs):
    Z = kwargs.get('Z_ae')
    if Z is not None and not np.isfinite(Z).all():
        raise RuntimeError('Nonfinite latent coordinate pairs are unavailable')
    return _review_ae(**kwargs)
"""
        )
    if name in ["F7", "F7c"]:
        expression = "Z_ae_current[valid, 1]"
        if name == "F7":
            return replace(
                source,
                expression,
                "Z_ae_current[valid, min(1, Z_ae_current.shape[1] - 1)]",
            )
        return replace(
            source,
            '"latent_axis_2": float(r2) if pd.notna(r2) else None,',
            '"latent_axis_2": float(r2) if Z_ae_current.shape[1] >= 2 and pd.notna(r2) else None,',
        ).replace(expression, "Z_ae_current[valid, min(1, Z_ae_current.shape[1] - 1)]")
    if name == "F7refuse":
        return (
            source
            + """
_review_ae = save_autoencoder_run
def save_autoencoder_run(**kwargs):
    Z = kwargs.get('Z_ae')
    if Z is not None and Z.shape[1] < 2:
        raise RuntimeError('Latent dimension requires at least two axes')
    return _review_ae(**kwargs)
"""
        )
    if name == "R1safe":
        # Preflight JSON before any target is opened; explicit refusal is a
        # control, not a selected production remediation policy.
        source = replace(
            source,
            "out = out.where(pd.notna(out), None)",
            "out = out.astype(object).where(pd.notna(out), None)",
        )
        return (
            source
            + """
_review_json = save_json_txt_pair
def save_json_txt_pair(report, *args, **kwargs):
    def reject_missing(value):
        if value is pd.NaT: raise ValueError('Missing NaT cannot be published')
        if isinstance(value, dict):
            for child in value.values(): reject_missing(child)
        elif isinstance(value, (list, tuple, np.ndarray)):
            for child in value: reject_missing(child)
    reject_missing(report)
    json.dumps(report, default=json_default, allow_nan=False)
    return _review_json(report, *args, **kwargs)
"""
        )
    raise ValueError(name)


TRIALS = [
    ("F1", "r1_writer", False),
    ("F8", "r1_writer or r6_", False),
    ("F2", "r2_", False),
    ("F9", "r2_ or positive_", True),
    ("F9unique", "r2_ or positive_", True),
    ("F3", "r3_", False),
    ("F10", "r3_ or positive_", True),
    ("F10marker", "r3_ or positive_", True),
    ("F4", "r4_", False),
    ("F4b", "r4_", False),
    ("F4c", "r4_ or positive_", True),
    ("F5", "r5_", False),
    ("F5c", "r5_", False),
    ("F11", "r5_ or positive_", True),
    ("F11log", "r5_ or positive_", True),
    ("F6", "r6_", False),
    ("F6c", "r6_ or positive_", True),
    ("F6refuse", "r6_ or positive_", True),
    ("F7", "r7_", False),
    ("F7c", "r7_ or positive_", True),
    ("F7refuse", "r7_ or positive_", True),
    ("R1safe", "r1_writer or positive_", True),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "audit/m27_review_proof_controls_results.json",
    )
    parser.add_argument("--only", nargs="*")
    parser.add_argument("--proofs-ref")
    args = parser.parse_args()
    source = (ROOT / SOURCE).read_text(encoding="utf-8")
    paths = [
        SOURCE,
        "src/dimred/pca_analyzer.py",
        "tests/conftest.py",
        "tests/reporter_review_contracts.py",
        PROOF,
        "pyproject.toml",
    ]
    protected = {p: replay.digest(ROOT / p) for p in paths if (ROOT / p).exists()}
    result = {
        "schema": 1,
        "head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "proofs_ref": args.proofs_ref,
        "tested_lf_sha256": protected,
        "trials": [],
    }
    trials = TRIALS if not args.only else [t for t in TRIALS if t[0] in args.only]
    with tempfile.TemporaryDirectory(prefix="m27-review-synthetic-") as folder:
        tree = Path(folder)
        for path in paths:
            (tree / path).parent.mkdir(parents=True, exist_ok=True)
            if args.proofs_ref and path in [
                PROOF,
                "tests/conftest.py",
                "tests/reporter_review_contracts.py",
            ]:
                (tree / path).write_bytes(
                    subprocess.check_output(
                        ["git", "show", f"{args.proofs_ref}:{path}"], cwd=ROOT
                    )
                )
            else:
                shutil.copyfile(ROOT / path, tree / path)
        result["effective_lf_sha256"] = {
            path: replay.digest(tree / path) for path in paths
        }
        for name, selection, admissible in [("baseline", "", False)] + trials:
            patched = source if name == "baseline" else candidate(source, name)
            ast.parse(patched)
            (tree / SOURCE).write_text(patched, encoding="utf-8")
            for cache in list(tree.rglob("__pycache__")):
                shutil.rmtree(cache)

            # Separate original baseline for every exact selected population.
            def trial(label):
                return replay.execute(
                    tree, label, proofs=True, target=PROOF, selection=selection or None
                )

            if name != "baseline":
                (tree / SOURCE).write_text(source, encoding="utf-8")
                baseline = trial(name + "_baseline")
                for cache in list(tree.rglob("__pycache__")):
                    shutil.rmtree(cache)
                (tree / SOURCE).write_text(patched, encoding="utf-8")
            else:
                baseline = None
            actual = trial(name)
            differences, changed_lines = [], []
            original_lines, patched_lines = source.splitlines(), patched.splitlines()
            for operation, i1, i2, j1, j2 in difflib.SequenceMatcher(
                a=original_lines, b=patched_lines
            ).get_opcodes():
                if operation != "equal":
                    differences.append(
                        {
                            "original_start": i1 + 1,
                            "old": original_lines[i1:i2],
                            "active_start": j1 + 1,
                            "new": patched_lines[j1:j2],
                        }
                    )
                    changed_lines.extend(
                        n + 1
                        for n in range(j1, j2)
                        if patched_lines[n].strip()
                        and not patched_lines[n].lstrip().startswith(("#", "def "))
                    )
            actual.update(
                admissible=admissible,
                selection=selection,
                baseline=baseline,
                patches=differences,
                changed_site_executed=bool(
                    set(changed_lines) & set(actual["executed_lines"])
                ),
                active_source_lf_sha256=replay.digest(tree / SOURCE),
            )
            if baseline:
                actual["collection_matches"] = sorted(
                    c["name"] for c in baseline["cases"]
                ) == sorted(c["name"] for c in actual["cases"])
                failures = [c for c in actual["cases"] if c["status"] == "FAIL"]
                actual["assertion_failures_only"] = all(
                    "AssertionError" in (c["detail"] or "")
                    or "\nE   assert " in (c["detail"] or "")
                    or "Failed:" in (c["detail"] or "")
                    for c in failures
                )
                actual["outcome"] = (
                    "ACCEPTED"
                    if admissible and actual["returncode"] == 0
                    else (
                        "REJECTED"
                        if not admissible
                        and failures
                        and actual["assertion_failures_only"]
                        and not actual["counts"]["ERROR"]
                        else "UNEXPECTED"
                    )
                )
            else:
                actual["outcome"] = "RED_BASELINE"
            result["trials"].append(actual)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            print(name, actual["outcome"], actual["counts"], flush=True)
    assert protected == {p: replay.digest(ROOT / p) for p in protected}
    assert all(
        t["changed_site_executed"] for t in result["trials"] if t["name"] != "baseline"
    )
    assert all(
        t["outcome"] != "UNEXPECTED" and t.get("collection_matches", True)
        for t in result["trials"]
    )


if __name__ == "__main__":
    main()
