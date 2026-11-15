"""Public facade contracts; actual Torch imports are separate slow integrations."""

import ast
import importlib.util
import json
from pathlib import Path
import re
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
    assert_success(run_probe(tmp_path, "wiring"))


@pytest.mark.integration
def test_side_effect_guard_has_positive_controls(tmp_path):
    assert_success(run_probe(tmp_path, "guard_control"))


@pytest.mark.integration
def test_core_producer_failure_is_not_silently_masked(tmp_path):
    result = run_probe(tmp_path, "wiring_failure")
    assert result.returncode == 1
    assert "ImportError: M29 producer ABI failure PCA" in result.stderr
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
        ("plotly_umap", "plotly", "plotly"),
        ("plotly_tsne", "plotly", "plotly"),
    ],
)
def test_missing_extension_has_attributable_diagnostic(
    tmp_path, operation, dependency, install
):
    result = run_probe(tmp_path, "optional", blocked=(dependency,), operation=operation)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "ImportError:" in result.stderr
    assert f"pip install {install}" in result.stderr
    assert "M29 blocked " + dependency in result.stderr
    assert "NameError" not in result.stderr


@pytest.mark.unit
def test_notebook_imports_reference_existing_public_objects():
    found = []
    for path in ROOT.rglob("*.ipynb"):
        if any(part.startswith(".") for part in path.relative_to(ROOT).parts):
            continue
        notebook = json.loads(path.read_text(encoding="utf-8"))
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] != "code":
                continue
            # Read import statements only: notebook magic is not Python syntax.
            source = "".join(cell["source"])
            for match in re.finditer(
                r"(?m)^[ \t]*from (dimred(?:\.\w+)?) import ([^\n]+)", source
            ):
                statement = match[0].strip()
                # Parenthesized multiline imports need their complete statement.
                if "(" in statement:
                    start = match.start()
                    statement = source[start : source.index(")", start) + 1].strip()
                node = ast.parse(statement).body[0]
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
    assert len({name for name, *_ in found}) >= 3


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
    assert "ImportError:" in result.stderr
    assert f"pip install {install}" in result.stderr
    assert "M29 blocked " + dependency in result.stderr
    assert "NameError" not in result.stderr
