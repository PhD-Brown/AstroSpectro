"""Replay M2.9 on temporary copies; never mutate the checked-out production."""

import argparse
import ast
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
from facade_probe import (  # noqa: E402
    EXPORTS,
    TORCH_STATEMENTS,
    assert_success,
    run_probe,
    probe_record,
    make_hijack,
)


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def hashes():
    files = [
        *sorted((ROOT / "src/dimred").glob("*.py")),
        ROOT / "tests/test_dimred_facade.py",
        ROOT / "tests/facade_probe.py",
        ROOT / "audit/proofs_m29_facade.py",
        Path(__file__),
    ]
    return {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in files
    }


def record(result):
    return dict(
        argv=result.args,
        returncode=result.returncode,
        stdout=result.stdout,
        stderr=result.stderr,
    )


def replace(path, old, new, count=1):
    text = path.read_text(encoding="utf-8")
    assert text.count(old) == count, (path, old, text.count(old), count)
    path.write_text(text.replace(old, new), encoding="utf-8")


def copy_source(destination):
    for source in (ROOT / "src").rglob("*.py"):
        target = destination / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    for name in ("pyproject.toml", "README.md"):
        shutil.copy2(ROOT / name, destination / name)


def consumers():
    rows = []
    paths = git("ls-files", "*.py", "*.ipynb", "*.md").splitlines()
    for name in paths:
        if name.startswith("audit/"):
            continue
        path = ROOT / name
        if path.suffix == ".ipynb":
            parts = [
                (f"cell {i}", "".join(c["source"]))
                for i, c in enumerate(
                    json.loads(path.read_text(encoding="utf-8"))["cells"]
                )
                if c["cell_type"] == "code"
            ]
        else:
            parts = [("file", path.read_text(encoding="utf-8"))]
        for part, source in parts:
            for number, line in enumerate(source.splitlines(), 1):
                if re.search(r"\b(?:from dimred|import dimred)\b", line):
                    rows.append(
                        dict(path=name, part=part, line=number, text=line.strip())
                    )
    return rows


def review_campaign(scratch, original, result):
    """Replay Claude's counterexamples and admissible remedies on private copies."""
    result["review_mutations"], result["review_alternatives"] = [], []

    def trial(
        name,
        mode,
        *,
        content=None,
        blocked=(),
        operation=None,
        change=None,
        startup=None,
        oracle=None,
    ):
        work = scratch / name
        copy_source(work)
        if content is not None:
            (work / "src/dimred/__init__.py").write_text(content, encoding="utf-8")
        if change:
            change(work)
        probe = run_probe(
            work / "cwd",
            mode,
            root=work,
            blocked=blocked,
            operation=operation,
            startup_dir=startup,
        )
        state = probe_record(probe, "M29_PROVENANCE")
        row = dict(
            name=name, mode=mode, operation=operation, provenance=state, **record(probe)
        )
        if oracle:
            error = state.get("exception", {})
            row.update(
                oracle=oracle,
                verdict=(
                    "KILLED"
                    if (
                        probe.returncode == 1
                        and error.get("type") == "AssertionError"
                        and oracle in error.get("message", "")
                    )
                    else "INCONCLUSIVE"
                ),
            )
            result["review_mutations"].append(row)
        else:
            result["review_alternatives"].append(row)
            assert_success(probe)
        return probe

    def conditional(work):
        replace(
            work / "src/dimred/autoencoder.py",
            "(nn.Module):",
            "(nn.Module if _TORCH_AVAILABLE else object):",
            count=3,
        )

    # Exact C1 variants remove BOTH eager AE import sites.
    for label, fallback in (
        (
            "C1a_None",
            "SpectralAutoencoder = tester_candidat = latent_arithmetic = None",
        ),
        ("C1b_swallow", "pass"),
    ):
        content = original.replace(
            "from .autoencoder import SpectralAutoencoder",
            "try:\n    from .autoencoder import SpectralAutoencoder, tester_candidat, latent_arithmetic\n"
            f"except Exception:\n    {fallback}",
        )
        content = content.replace(
            "from .autoencoder import tester_candidat, latent_arithmetic", ""
        )
        for i, statement in enumerate(TORCH_STATEMENTS):
            oracle = (
                "AE export SpectralAutoencoder must be callable"
                if label.startswith("C1a")
                else (
                    "AE import attribution"
                    if statement == "from dimred import SpectralAutoencoder"
                    else (
                        "public entry unusable"
                        if statement == "from dimred import *"
                        else "AE access SpectralAutoencoder"
                    )
                )
            )
            trial(
                f"{label}_{i}",
                "torch_boundary",
                content=content,
                blocked=("torch",),
                operation=statement,
                oracle=oracle,
            )
    for name, assignment, oracle in (
        (
            "C1_helper_None",
            "tester_candidat = None",
            "AE export tester_candidat must be callable",
        ),
        (
            "C1_helper_noncallable",
            "latent_arithmetic = object()",
            "AE export latent_arithmetic must be callable",
        ),
        (
            "C1_helper_empty_result",
            "tester_candidat = lambda **kwargs: None",
            "AE helper tester_candidat result",
        ),
    ):
        trial(
            name,
            "torch_boundary",
            content=original + "\n" + assignment + "\n",
            blocked=("torch",),
            operation="import dimred",
            change=conditional,
            oracle=oracle,
        )
    bad_lazy = original.replace("from .pca_analyzer import PCAAnalyzer", "") + (
        "\nimport sys as _sys\ndef __getattr__(name):\n"
        "    if name == 'PCAAnalyzer':\n        return _sys.modules[__name__ + '.pca_analyzer'].PCAAnalyzer\n"
        "    raise AttributeError(name)\n"
    )
    trial(
        "C2_preloaded_only",
        "real",
        content=bad_lazy,
        oracle="fresh public attribute PCAAnalyzer",
    )
    fake = scratch / "C3_fake.py"
    fake.write_text(
        "class PCAAnalyzer: pass\nraise RuntimeError('fake partial import')\n",
        encoding="utf-8",
    )
    for mode, operation in (
        ("entry", "import dimred"),
        ("entry", "from dimred import PCAAnalyzer"),
        ("wiring", None),
    ):
        startup = scratch / f"C3_startup_{len(result['review_mutations'])}"
        make_hijack(startup, fake)
        trial(
            f"C3_{mode}_{len(result['review_mutations'])}",
            mode,
            operation=operation,
            startup=startup,
            oracle="facade provenance mismatch",
        )
    name = "C3_partial_code"
    startup = scratch / "C3_partial_startup"
    make_hijack(startup, fake, scratch / name / "src/dimred/__init__.py")
    trial(
        name,
        "entry",
        operation="import dimred",
        startup=startup,
        oracle="facade code provenance mismatch",
    )
    for suffix in (
        ".npy",
        ".npz",
        ".parquet",
        ".feather",
        ".h5",
        ".hdf5",
        ".pth",
        ".ckpt",
        ".safetensors",
    ):
        data = scratch / ("C4_data" + suffix)
        data.write_bytes(b"synthetic read must be intercepted")
        content = original + (
            f"\ntry:\n    open({str(data)!r}, 'rb').read()\nexcept RuntimeError:\n    pass\n"
        )
        trial(
            "C4_read_" + suffix[1:],
            "wiring",
            content=content,
            oracle="caught forbidden attempts",
        )
    for name, call in (
        ("gethostbyname", "socket.gethostbyname('localhost')"),
        ("gethostbyaddr", "socket.gethostbyaddr('127.0.0.1')"),
        ("bind", "socket.socket().bind(('127.0.0.1', 0))"),
    ):
        content = (
            original
            + f"\nimport socket\ntry:\n    {call}\nexcept RuntimeError:\n    pass\n"
        )
        trial(
            "C4_" + name, "wiring", content=content, oracle="caught forbidden attempts"
        )

    def partial_style(work):
        path = work / "src/dimred/dimred_visualizer.py"
        replace(path, '            "figure.dpi": 150,\n', "")
        replace(path, '            "axes.grid": True,\n', "")

    c5 = trial(
        "C5_partial_style",
        "style",
        change=partial_style,
        oracle="import changed caller Matplotlib configuration",
    )
    result["C5_style"] = probe_record(c5, "M29_STYLE")

    # Valid lazy facade + existing AE-source remedy: both boundaries must work.
    lazy = (
        "import importlib as _importlib\n"
        f"_exports = {EXPORTS!r}\n__all__ = list(reversed(_exports))\n"
        '__version__ = "0.3.0"\n__author__ = "AstroSpectro — PHY-3500"\n'
        "def __getattr__(name):\n"
        "    if name not in _exports:\n        raise AttributeError(name)\n"
        "    source, producer = _exports[name]\n"
        "    return getattr(_importlib.import_module('.' + source, __name__), producer)\n"
    )
    for i, statement in enumerate(TORCH_STATEMENTS):
        trial(
            f"A04_A05_boundary_{i}",
            "torch_boundary",
            content=lazy,
            change=conditional,
            operation=statement,
            blocked=("torch",),
        )
    explicit = original + (
        "\nclass SpectralAutoencoder:\n"
        "    def __init__(self, *args, **kwargs):\n        raise ImportError('PyTorch required; install torch')\n"
        "def _ae_refusal(*args, **kwargs):\n    raise ImportError('Torch extension unavailable')\n"
        "tester_candidat = latent_arithmetic = _ae_refusal\n"
    )
    trial(
        "A07_explicit_use_refusal",
        "torch_boundary",
        content=explicit,
        change=conditional,
        operation="from dimred import *",
        blocked=("torch",),
    )
    lazy_refusal = lazy.replace(
        "    source, producer = _exports[name]\n",
        "    source, producer = _exports[name]\n"
        "    if source == 'autoencoder':\n        raise ImportError('PyTorch required for AE exports')\n",
    )
    for i, statement in enumerate(
        (
            "import dimred",
            "from dimred import SpectralAutoencoder",
            "from dimred import *",
        )
    ):
        trial(
            f"A08_lazy_access_refusal_{i}",
            "torch_boundary",
            content=lazy_refusal,
            operation=statement,
            blocked=("torch",),
        )

    def local_style(work):
        path = work / "src/dimred/dimred_visualizer.py"
        text = path.read_text(encoding="utf-8")
        start, end = text.index("try:\n"), text.index("\n# Palette")
        path.write_text(
            text[:start]
            + "with plt.rc_context():\n"
            + "\n".join("    " + line for line in text[start:end].splitlines())
            + text[end:],
            encoding="utf-8",
        )

    trial("A09_rc_context", "style", change=local_style)
    observed = run_probe(
        scratch / "style_failed", "style_failed_import", blocked=("torch",)
    )
    assert_success(observed)
    result["failed_import_style"] = dict(
        measurement=probe_record(observed, "M29_STYLE"), **record(observed)
    )
    guards = run_probe(scratch / "guard_controls", "guard_control")
    assert_success(guards)
    result["guard_controls"] = dict(
        coverage=probe_record(guards, "M29_GUARDS"), **record(guards)
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "test-results/m29-replay.json",
        help="Explicit result destination; default is ignored by Git.",
    )
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    reference = ROOT / "audit/m29_experiments.json"
    reference_hash = hashlib.sha256(reference.read_bytes()).hexdigest()
    before = hashes()
    result = dict(
        head=git("rev-parse", "HEAD"),
        branch=git("branch", "--show-current"),
        status=git("status", "--short"),
        executable=sys.executable,
        python=sys.version,
        input_hashes=before,
        consumers=consumers(),
        historical_reference="43fbdac2425e9b36d253ef08c700859a87638f69:audit/m29_experiments.json",
        reference_manifest_sha256_before=reference_hash,
    )
    result["versions"] = {}
    for package in (
        "numpy",
        "pandas",
        "scikit-learn",
        "matplotlib",
        "torch",
        "umap-learn",
        "hdbscan",
        "plotly",
        "wandb",
        "pytest",
        "setuptools",
        "pip",
    ):
        try:
            result["versions"][package] = version(package)
        except PackageNotFoundError:
            result["versions"][package] = None
    result["source_blobs"] = {
        p: git("rev-parse", "HEAD:" + p) for p in before if p.startswith("src/")
    }
    tree = ast.parse((ROOT / "src/dimred/__init__.py").read_text(encoding="utf-8"))
    result["direct_import_order"] = [
        dict(
            module=n.module, names=[(a.name, a.asname) for a in n.names], line=n.lineno
        )
        for n in tree.body
        if isinstance(n, ast.ImportFrom)
    ]
    with tempfile.TemporaryDirectory(prefix="m29-audit-") as directory:
        scratch = Path(directory)
        result["baseline"] = record(run_probe(scratch / "baseline", "wiring"))
        assert_success(type("Result", (), result["baseline"])())
        trials = [
            (
                "M01_missing_export",
                "remove",
                '    "PCAAnalyzer",\n',
                "",
                "public export surface",
            ),
            (
                "M02_duplicate_export",
                "append",
                "",
                '\n__all__.append("PCAAnalyzer")\n',
                "duplicate export",
            ),
            (
                "M03_wrong_class",
                "append",
                "",
                "\nPCAAnalyzer = DimRedDataLoader\n",
                "identity PCAAnalyzer",
            ),
            (
                "M04_wrong_alias",
                "append",
                "",
                "\nxgboost_predict = save_pca_run\n",
                "identity xgboost_predict",
            ),
            (
                "M05_missing_attribute",
                "append",
                "",
                "\ndel PCAAnalyzer\n",
                "fresh public attribute PCAAnalyzer",
            ),
            (
                "M06_incomplete_placeholder",
                "append",
                "",
                "\nSpectralAutoencoder = object()\n",
                "callable SpectralAutoencoder",
            ),
            (
                "M07_path_side_effect",
                "append",
                "",
                '\nimport sys\nsys.path.append("m29_unwanted")\n',
                "import changed sys.path",
            ),
            (
                "M08_logging_side_effect",
                "append",
                "",
                "\nimport logging\nlogging.basicConfig(level=logging.DEBUG)\n",
                "import changed root logging",
            ),
            (
                "M09_masked_core_error",
                "replace",
                "from .pca_analyzer import PCAAnalyzer",
                "try:\n    from .pca_analyzer import PCAAnalyzer\nexcept ImportError:\n    PCAAnalyzer = object()",
                "core producer failure was hidden",
            ),
        ]
        result["mutations"] = []
        original = (ROOT / "src/dimred/__init__.py").read_text(encoding="utf-8")
        for name, kind, old, new, oracle in trials:
            work = scratch / name
            copy_source(work)
            facade = work / "src/dimred/__init__.py"
            if kind == "append":
                facade.write_text(original + new, encoding="utf-8")
            else:
                replace(facade, old, new)
            probe = run_probe(
                work / "cwd",
                "wiring_failure" if name.startswith("M09") else "wiring",
                root=work,
            )
            # A kill requires the designated assertion, never a collection/fixture error.
            killed = (
                probe.returncode == 1 and "AssertionError: " + oracle in probe.stderr
            )
            result["mutations"].append(
                dict(
                    name=name,
                    oracle=oracle,
                    verdict="KILLED" if killed else "INCONCLUSIVE",
                    **record(probe),
                )
            )
        # This import-boundary mutation must execute before any producer is cached.
        work = scratch / "M10_eager_umap"
        copy_source(work)
        (work / "src/dimred/__init__.py").write_text(
            "import umap  # M29 deliberate eager optional import\n" + original,
            encoding="utf-8",
        )
        probe = run_probe(work / "cwd", "real", root=work, blocked=("umap",))
        killed = (
            probe.returncode == 1
            and "ModuleNotFoundError: M29 blocked umap" in probe.stderr
            and "M29 deliberate eager optional import" in probe.stderr
        )
        result["mutations"].append(
            dict(
                name="M10_eager_umap",
                verdict="KILLED" if killed else "INCONCLUSIVE",
                oracle="required fresh facade import unavailable at modified import boundary",
                category="expected import-boundary failure, not incidental runtime kill",
                **record(probe),
            )
        )
        alternatives = {
            "A01_reordered_exports": original + "\n__all__.reverse()\n",
            "A02_equivalent_alias": original.replace(
                "from .xgboost_bridge import load_and_predict as xgboost_predict",
                "from . import xgboost_bridge as _bridge\nxgboost_predict = _bridge.load_and_predict",
            ),
            "A03_internal_attribute": original + "\n_internal_cache = {}\n",
            "A04_lazy_facade": (
                "import importlib as _importlib\n"
                f"_exports = {EXPORTS!r}\n__all__ = list(_exports)\n"
                '__version__ = "0.3.0"\n__author__ = "AstroSpectro — PHY-3500"\n'
                "def __getattr__(name):\n"
                "    if name not in _exports:\n        raise AttributeError(name)\n"
                "    source, producer = _exports[name]\n"
                "    return getattr(_importlib.import_module('.' + source, __name__), producer)\n"
            ),
        }
        result["alternatives"] = []
        for name, content in alternatives.items():
            work = scratch / name
            copy_source(work)
            (work / "src/dimred/__init__.py").write_text(content, encoding="utf-8")
            for mode in ("wiring", "real"):
                probe = run_probe(work / mode, mode, root=work)
                result["alternatives"].append(
                    dict(name=name, mode=mode, **record(probe))
                )
                assert_success(probe)
            probe = run_probe(work / "failure", "wiring_failure", root=work)
            assert probe.returncode == 1 and (
                "ImportError: M29 producer ABI failure PCA" in probe.stderr
            ), (probe.stdout + probe.stderr)
            result["alternatives"].append(
                dict(name=name, mode="deferred core error", **record(probe))
            )
        # Candidate remedies are controls of proof admissibility, not proposed fixes.
        work = scratch / "A05_conditional_torch_bases"
        copy_source(work)
        replace(
            work / "src/dimred/autoencoder.py",
            "(nn.Module):",
            "(nn.Module if _TORCH_AVAILABLE else object):",
            count=3,
        )
        probe = run_probe(work / "cwd", "real", root=work, blocked=("torch",))
        assert_success(probe)
        result["alternatives"].append(
            dict(name="A05_conditional_torch_bases", **record(probe))
        )
        work = scratch / "A06_explicit_style_application"
        copy_source(work)
        visualizer = work / "src/dimred/dimred_visualizer.py"
        content = visualizer.read_text(encoding="utf-8")
        start, end = content.index("try:\n"), content.index("\n# Palette")
        content = (
            content[:start]
            + "def _apply_style():\n"
            + "\n".join("    " + line for line in content[start:end].splitlines())
            + content[end:]
        )
        visualizer.write_text(content, encoding="utf-8")
        probe = run_probe(work / "cwd", "style", root=work)
        assert_success(probe)
        result["alternatives"].append(
            dict(name="A06_explicit_style_application", **record(probe))
        )
        result["red_observations"] = []
        statements = TORCH_STATEMENTS
        for i, statement in enumerate(statements):
            probe = run_probe(
                scratch / f"red{i}",
                "torch_boundary",
                blocked=("torch",),
                operation=statement,
            )
            result["red_observations"].append(
                dict(
                    operation=statement,
                    provenance=probe_record(probe, "M29_PROVENANCE"),
                    **record(probe),
                )
            )
            state = result["red_observations"][-1]["provenance"]
            error = state["exception"]
            assert "torch" in state["blocked_hits"]
            assert error["type"] == "NameError" and any(
                Path(frame["file"]).name == "autoencoder.py" and frame["line"] == 87
                for frame in error["frames"]
            ), error
        style = run_probe(scratch / "red_style", "style")
        measurement = probe_record(style, "M29_STYLE")
        assert (
            style.returncode == 1
            and measurement["changed"]
            and measurement["import_failure"] is None
        )
        result["red_observations"].append(
            dict(
                operation="preserve caller style",
                measurement=measurement,
                **record(style),
            )
        )
        review_campaign(scratch, original, result)
        # Build/install only a temporary source copy. No dependency resolution/network.
        build = scratch / "build"
        copy_source(build)
        wheel_dir, installed = scratch / "wheels", scratch / "installed"
        commands = [
            [
                sys.executable,
                "-m",
                "pip",
                "wheel",
                "--no-index",
                "--no-deps",
                "--no-build-isolation",
                "--wheel-dir",
                str(wheel_dir),
                str(build),
            ],
        ]
        result["packaging"] = []
        for command in commands:
            built = subprocess.run(
                command, cwd=scratch, capture_output=True, text=True, timeout=120
            )
            result["packaging"].append(dict(command=command, **record(built)))
            assert built.returncode == 0, built.stdout + built.stderr
        wheel = next(wheel_dir.glob("*.whl"))
        with zipfile.ZipFile(wheel) as archive:
            members = archive.namelist()
            assert "dimred/__init__.py" in members
            result["additional_astrospectro_dimred_namespace"] = any(
                n.startswith("astrospectro/dimred/") for n in members
            )  # Inventory, not a prohibition of an additive namespace.
            result["wheel_members"] = members
        command = [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-index",
            "--no-deps",
            "--target",
            str(installed),
            str(wheel),
        ]
        built = subprocess.run(
            command, cwd=scratch, capture_output=True, text=True, timeout=120
        )
        result["packaging"].append(dict(command=command, **record(built)))
        assert built.returncode == 0, built.stdout + built.stderr
        code = (
            "import sys, importlib.metadata, pathlib, runpy\n"
            f"sys.path.insert(0, {str(installed)!r})\n"
            f"helper = runpy.run_path({str(ROOT / 'tests/facade_probe.py')!r})\n"
            "violations = helper['guarded_imports'](pathlib.Path.cwd())\n"
            "import dimred\n"
            f"assert pathlib.Path(dimred.__file__).resolve() == pathlib.Path({str(installed)!r}) / 'dimred/__init__.py'\n"
            "helper['check_exports'](dimred)\n"
            "assert not violations\n"
            "print('M29_INSTALLED_OK', importlib.metadata.version('astrospectro'), dimred.__version__)\n"
        )
        command = [sys.executable, "-I", "-B", "-c", code]
        env = dict(
            __import__("os").environ,
            MPLBACKEND="Agg",
            MPLCONFIGDIR=str(scratch / "mpl"),
        )
        built = subprocess.run(
            command,
            cwd=scratch,
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=90,
        )
        result["packaging"].append(dict(command=command, **record(built)))
        assert built.returncode == 0, built.stdout + built.stderr
    result["end_hashes"] = hashes()
    result["end_head"] = git("rev-parse", "HEAD")
    assert result["end_hashes"] == before and result["end_head"] == result["head"]
    result["reference_overwrite_explicit"] = (
        args.output.resolve() == reference.resolve()
    )
    result["reference_preserved_during_replay"] = (
        hashlib.sha256(reference.read_bytes()).hexdigest() == reference_hash
    )
    assert result["reference_preserved_during_replay"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            dict(
                mutations=[(m["name"], m["verdict"]) for m in result["mutations"]],
                alternatives=len(result["alternatives"]),
                red_returns=[r["returncode"] for r in result["red_observations"]],
                packaging="PASS",
                unchanged_inputs=True,
                review_mutations=[
                    (m["name"], m["verdict"]) for m in result["review_mutations"]
                ],
                review_alternatives=len(result["review_alternatives"]),
                output=str(args.output),
            )
        )
    )
    assert all(m["verdict"] == "KILLED" for m in result["mutations"])
    assert all(m["verdict"] == "KILLED" for m in result["review_mutations"])


if __name__ == "__main__":
    main()
