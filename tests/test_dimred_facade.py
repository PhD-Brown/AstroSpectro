"""Public facade contracts; actual Torch imports are separate slow integrations."""

import ast
import importlib.util
import json
from pathlib import Path
import tomllib

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "m29_probe", ROOT / "tests/facade_probe.py"
)
PROBE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROBE)
EXPORTS, OPTIONAL = PROBE.EXPORTS, PROBE.OPTIONAL
assert_success, run_probe = PROBE.assert_success, PROBE.run_probe


@pytest.mark.unit
def test_declared_exports_match_independent_public_inventory():
    tree = ast.parse((ROOT / "src/dimred/__init__.py").read_text(encoding="utf-8"))
    declared = next(
        ast.literal_eval(n.value)
        for n in tree.body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "__all__" for t in n.targets)
    )
    assert len(declared) == len(set(declared)) == 14
    assert set(declared) == set(EXPORTS)


@pytest.mark.unit
def test_public_producers_exist_in_source():
    for source, name in EXPORTS.values():
        tree = ast.parse((ROOT / f"src/dimred/{source}.py").read_text(encoding="utf-8"))
        assert name in {
            node.name
            for node in tree.body
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        }


@pytest.mark.integration
def test_wiring_against_controlled_siblings_in_fresh_process(tmp_path):
    result = run_probe(tmp_path, "wiring")
    assert_success(result)
    state = PROBE.probe_record(result, "M29_PROVENANCE")
    assert Path(state["loaded"]).resolve() == ROOT / "src/dimred/__init__.py"
    assert state["executed"][0]["code"] == str(ROOT / "src/dimred/__init__.py")


@pytest.mark.integration
def test_side_effect_guard_has_positive_controls(tmp_path):
    assert_success(run_probe(tmp_path, "guard_control"))


@pytest.mark.integration
@pytest.mark.parametrize("source", ["pca_analyzer", "embedding", "autoencoder"])
def test_core_producer_failure_is_not_silently_masked(tmp_path, source):
    result = run_probe(tmp_path, "wiring_failure", operation=source)
    assert result.returncode == 1
    label = "PCA" if source == "pca_analyzer" else source
    error = PROBE.probe_record(result, "M29_PROVENANCE")["exception"]
    assert error["type"] == "ImportError"
    assert error["message"] == f"M29 producer ABI failure {label}"
    assert "M29_OK" not in result.stdout


@pytest.mark.integration
@pytest.mark.parametrize(
    "source",
    [
        "data_loader",
        "pca_analyzer",
        "embedding",
        "dimred_visualizer",
        "hdbscan_analyzer",
        "run_reporter",
        "xgboost_bridge",
    ],
)
def test_independent_sources_load_without_extensions(tmp_path, source):
    # Explicitly NOT `import dimred.source`: no parent facade is executed.
    assert_success(run_probe(tmp_path, "isolated", blocked=OPTIONAL, operation=source))


@pytest.mark.integration
@pytest.mark.parametrize(
    "operation,dependency,install",
    [
        ("umap", "umap", "umap-learn"),
        ("hdbscan", "hdbscan", "hdbscan"),
        ("hdbscan_presentation", "hdbscan", "hdbscan"),
        ("hdbscan_sensitivity", "hdbscan", "hdbscan"),
        ("plotly_umap", "plotly", "plotly"),
        ("plotly_tsne", "plotly", "plotly"),
    ],
)
def test_missing_extension_has_attributable_diagnostic(
    tmp_path, operation, dependency, install
):
    result = run_probe(tmp_path, "optional", blocked=(dependency,), operation=operation)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "ImportError:" in result.stderr or "ModuleNotFoundError:" in result.stderr
    diagnostic = result.stderr.strip().splitlines()[-1].lower()
    assert dependency in diagnostic or install in diagnostic
    assert "M29 blocked " + dependency in result.stderr
    assert "NameError" not in result.stderr


@pytest.mark.integration
@pytest.mark.parametrize(
    "dependency,statement",
    [
        ("numpy", "from dimred import PCAAnalyzer"),
        ("pandas", "from dimred import DimRedDataLoader"),
        ("scipy", "from dimred import PCAAnalyzer"),
        ("sklearn", "from dimred import PCAAnalyzer"),
        ("matplotlib", "from dimred import DimRedVisualizer"),
    ],
)
def test_missing_required_dependency_is_attributable(tmp_path, dependency, statement):
    result = run_probe(tmp_path, "entry", blocked=(dependency,), operation=statement)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "M29 blocked " + dependency in result.stderr
    assert "ImportError" in result.stderr or "ModuleNotFoundError" in result.stderr
    assert "NameError" not in result.stderr


@pytest.mark.unit
def test_notebook_imports_reference_existing_public_objects():
    found = []
    for path in (ROOT / "notebooks/dimred").glob("*.ipynb"):
        notebook = json.loads(path.read_text(encoding="utf-8"))
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] != "code":
                continue
            source = "".join(cell["source"])
            for node in PROBE.notebook_imports(source):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name.split(".")[0] == "dimred":
                            assert (
                                ROOT / "src" / alias.name.replace(".", "/")
                            ).is_dir()
                    found.append(
                        (path.name, index, "import", [a.name for a in node.names])
                    )
                    continue
                names = [a.name for a in node.names]
                if node.module == "dimred":
                    assert set(names) <= set(EXPORTS), (path, index, names)
                else:
                    tree = ast.parse(
                        (
                            ROOT / "src" / (node.module.replace(".", "/") + ".py")
                        ).read_text(encoding="utf-8")
                    )
                    producers = {
                        n.name
                        for n in tree.body
                        if isinstance(n, (ast.ClassDef, ast.FunctionDef))
                    }
                    producers.update(
                        target.id
                        for n in tree.body
                        if isinstance(n, ast.Assign)
                        for target in n.targets
                        if isinstance(target, ast.Name)
                    )
                    assert set(names) <= producers, (path, index, names)
                found.append((path.name, index, node.module, names))
    assert len({name for name, *_ in found}) == 3
    assert len(found) == 15


@pytest.mark.unit
def test_notebook_parser_handles_aliases_inline_nested_and_magics():
    nodes = PROBE.notebook_imports(
        "%matplotlib inline\n!echo ignored\nimport dimred as dr\n"
        "x = 1; from dimred import PCAAnalyzer as P\n"
        "if True:\n    from dimred.a.b import (C, D)\n"
    )
    assert len(nodes) == 3
    assert isinstance(nodes[0], ast.Import) and nodes[0].names[0].asname == "dr"
    assert {n.module for n in nodes if isinstance(n, ast.ImportFrom)} == {
        "dimred",
        "dimred.a.b",
    }


@pytest.mark.unit
@pytest.mark.parametrize(
    "kind,message,accepted",
    [
        (ImportError, "PyTorch required; install torch", True),
        (
            ImportError,
            "cannot import name 'SpectralAutoencoder' from '/venv-torch/dimred'",
            False,
        ),
        (TypeError, "torch is missing", False),
        (NameError, "nn is undefined", False),
        (AttributeError, "torch", False),
        (ImportError, "unrelated optional backend", False),
        (ImportError, "PyTorch warning\nunrelated final diagnostic", False),
        (ImportError, "Optional backend\nPyTorch required", True),
    ],
)
def test_torch_refusal_uses_final_typed_diagnostic(kind, message, accepted):
    assert PROBE.torch_refusal(kind(message)) is accepted


@pytest.mark.integration
def test_style_unavailable_is_not_a_style_measurement(tmp_path):
    result = run_probe(tmp_path, "style", blocked=("torch",))
    state = PROBE.probe_record(result, "M29_PROVENANCE")
    assert result.returncode == 1
    assert state["exception"]["type"] == "RuntimeError"
    assert "M29_STYLE_UNAVAILABLE" in state["exception"]["message"]
    assert "M29_STYLE " not in result.stdout


@pytest.mark.integration
@pytest.mark.parametrize(
    "mode,statement,partial",
    [
        ("entry", "import dimred", False),
        ("entry", "from dimred import PCAAnalyzer", False),
        ("wiring", None, False),
        ("entry", "import dimred", True),
    ],
)
def test_private_pth_finder_cannot_replace_checkout_facade(
    tmp_path, mode, statement, partial
):
    fake = tmp_path / "fake.py"
    fake.write_text(
        "class PCAAnalyzer: pass\nraise RuntimeError('fake partial import')\n"
    )
    startup = tmp_path / "startup"
    PROBE.make_hijack(
        startup, fake, ROOT / "src/dimred/__init__.py" if partial else None
    )
    result = run_probe(tmp_path / "cwd", mode, operation=statement, startup_dir=startup)
    state = PROBE.probe_record(result, "M29_PROVENANCE")
    assert result.returncode == 1 and "provenance mismatch" in result.stderr
    assert "m29_hijack.Finder" in state["meta_path"]
    if partial:
        assert state["executed"][0]["code"] == str(fake)
    else:
        assert state["resolved"] == str(fake)


@pytest.mark.unit
def test_packaging_units_are_distinct_and_dimred_is_discovered():
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert config["project"]["name"] == "astrospectro"
    # Do not require the dimred subproject and distribution versions to match.
    from setuptools import find_packages

    assert "dimred" in find_packages(str(ROOT / "src"))
    # An additional future namespace is admissible; the legacy dimred API stays.


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.parametrize(
    "entry,blocked",
    [
        (None, ()),
        (None, ("umap", "hdbscan", "plotly", "wandb")),
        (None, ("xgboost", "lightgbm", "catboost")),
        ("from dimred import PCAAnalyzer", ()),
        ("from dimred import *", ()),
        ("from dimred.pca_analyzer import PCAAnalyzer", ()),
        ("from dimred.autoencoder import SpectralAutoencoder", ()),
        (
            "from dimred.embedding import compare_embeddings\n"
            "from dimred.data_loader import SpectralMatrixLoader\n"
            "from dimred.dimred_visualizer import CLASS_COLORS, CLASS_MARKERS\n"
            "from dimred.xgboost_bridge import load_and_predict",
            (),
        ),
    ],
)
def test_real_facade_identity_star_repeat_and_unrelated_absences(
    tmp_path, entry, blocked
):
    if importlib.util.find_spec("torch") is None:
        pytest.skip("actual facade needs real Torch until #21 is corrected")
    assert_success(run_probe(tmp_path, "real", blocked=blocked, operation=entry))


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.parametrize(
    "operation,dependency,install",
    [
        ("umap", "umap", "umap-learn"),
        ("hdbscan", "hdbscan", "hdbscan"),
        ("hdbscan_presentation", "hdbscan", "hdbscan"),
        ("hdbscan_sensitivity", "hdbscan", "hdbscan"),
        ("plotly_umap", "plotly", "plotly"),
        ("plotly_tsne", "plotly", "plotly"),
    ],
)
def test_real_facade_defers_optional_failure_until_use(
    tmp_path, operation, dependency, install
):
    if importlib.util.find_spec("torch") is None:
        pytest.skip("actual facade needs real Torch until #21 is corrected")
    result = run_probe(
        tmp_path, "optional_real", blocked=(dependency,), operation=operation
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert "ImportError:" in result.stderr or "ModuleNotFoundError:" in result.stderr
    diagnostic = result.stderr.strip().splitlines()[-1].lower()
    assert dependency in diagnostic or install in diagnostic
    assert "M29 blocked " + dependency in result.stderr
    assert "NameError" not in result.stderr
