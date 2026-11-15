"""Fresh-process probes for M2.9; controlled wiring is explicitly synthetic."""

import importlib
import importlib.abc
import importlib.util
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import types


# Independent public contract from the documented PHY-3500 API, not the AST.
EXPORTS = {
    "DimRedDataLoader": ("data_loader", "DimRedDataLoader"),
    "PCAAnalyzer": ("pca_analyzer", "PCAAnalyzer"),
    "EmbeddingEngine": ("embedding", "EmbeddingEngine"),
    "DimRedVisualizer": ("dimred_visualizer", "DimRedVisualizer"),
    "SpectralAutoencoder": ("autoencoder", "SpectralAutoencoder"),
    "save_pca_run": ("run_reporter", "save_pca_run"),
    "save_umap_tsne_run": ("run_reporter", "save_umap_tsne_run"),
    "save_autoencoder_run": ("run_reporter", "save_autoencoder_run"),
    "HDBSCANAnalyzer": ("hdbscan_analyzer", "HDBSCANAnalyzer"),
    "compute_feature_profiles": ("hdbscan_analyzer", "compute_feature_profiles"),
    "compute_sensitivity": ("hdbscan_analyzer", "compute_sensitivity"),
    "xgboost_predict": ("xgboost_bridge", "load_and_predict"),
    "tester_candidat": ("autoencoder", "tester_candidat"),
    "latent_arithmetic": ("autoencoder", "latent_arithmetic"),
}
OPTIONAL = ("torch", "umap", "hdbscan", "plotly", "wandb")


def run_probe(tmp_path, mode, *, root=None, blocked=(), operation=None, real=False):
    """Run a probe with -I, explicit source path, alien cwd and private caches."""
    root = Path(root or Path(__file__).resolve().parents[1])
    tmp_path.mkdir(parents=True, exist_ok=True)
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTHON")}
    env.update(
        MPLBACKEND="Agg",
        MPLCONFIGDIR=str(tmp_path / "mpl"),
        XDG_CACHE_HOME=str(tmp_path / "cache"),
        CUDA_VISIBLE_DEVICES="-1",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
    )
    args = dict(
        root=str(root), mode=mode, blocked=list(blocked), operation=operation, real=real
    )
    # -I ignores PYTHONPATH; the helper is executed by explicit absolute path.
    command = [
        sys.executable,
        "-I",
        "-B",
        str(Path(__file__).resolve()),
        json.dumps(args),
    ]
    return subprocess.run(
        command,
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=90,
    )


def assert_success(result):
    assert result.returncode == 0, result.stdout + result.stderr
    assert "M29_OK" in result.stdout, result.stdout + result.stderr


class BlockImports(importlib.abc.MetaPathFinder):
    """Deny exact dependency roots and descendants, recording actual attempts."""

    def __init__(self, roots):
        self.roots = set(roots)
        self.hits = []

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in self.roots:
            self.hits.append(fullname)
            raise ModuleNotFoundError(f"M29 blocked {fullname}", name=fullname)


def check_exports(package):
    """Assert object identity, star surface, metadata and repeated import identity."""
    assert len(package.__all__) == len(set(package.__all__)), "duplicate export"
    assert set(package.__all__) == set(EXPORTS), "public export surface"
    assert package.__version__ == "0.3.0"
    assert package.__author__ == "AstroSpectro \u2014 PHY-3500"
    for name, (source, producer) in EXPORTS.items():
        module = importlib.import_module(f"dimred.{source}")
        assert hasattr(package, name), f"missing public attribute {name}"
        assert getattr(package, name) is getattr(module, producer), f"identity {name}"
        assert callable(getattr(package, name)), f"callable {name}"
    namespace = {}
    exec("from dimred import *", namespace)
    assert set(namespace) - {"__builtins__"} == set(EXPORTS), "star surface"
    for name in EXPORTS:
        assert namespace[name] is getattr(package, name)
    for _ in range(3):
        assert importlib.import_module("dimred") is package
        for name in EXPORTS:
            namespace = {}
            exec(f"from dimred import {name}", namespace)
            assert namespace[name] is getattr(package, name)


def isolated(root, source):
    """Load a source file without executing its parent package (explicit bypass)."""
    spec = importlib.util.spec_from_file_location(
        f"m29_isolated_{source}", root / "src/dimred" / f"{source}.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert "dimred" not in sys.modules, "isolated source unexpectedly loaded facade"
    return module


def guarded_imports(scratch):
    """Reject network and writes outside scratch; retain attempts even if caught."""
    violations = []

    def audit(event, args):
        forbidden = event in {"socket.connect", "socket.getaddrinfo", "socket.sendto"}
        paths = []
        if event == "open":
            path, mode, flags = args
            if isinstance(path, (str, bytes)) and Path(os.fsdecode(path)).suffix in {
                ".fits",
                ".csv",
                ".pkl",
                ".joblib",
                ".pt",
            }:
                forbidden = True  # No catalog/model is needed to import the API.
            if isinstance(path, (str, bytes)) and (
                (mode and any(c in mode for c in "wax+"))
                or flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC)
            ):
                paths = [path]
        if event in {"os.mkdir", "os.remove", "os.rmdir", "os.rename"}:
            paths = list(args[:2] if event == "os.rename" else args[:1])
        for path in paths:
            if os.fsdecode(path).lower() == os.devnull.lower():
                continue  # OS null device is used by platform detection.
            resolved = Path(os.fsdecode(path)).resolve()
            # Matplotlib's cache is legitimate; caller outputs are not import work.
            if not any(
                resolved.is_relative_to(scratch / directory)
                for directory in ("mpl", "cache")
            ):
                forbidden = True
        if forbidden:
            violations.append(event)
            raise RuntimeError(f"M29 forbidden side effect: {event}")

    sys.addaudithook(audit)
    return violations


def optional_operation(module, operation):
    import numpy as np
    import pandas as pd

    X = np.arange(24.0).reshape(8, 3)
    if operation == "umap":
        module.EmbeddingEngine(method="umap").fit_transform(X)
    elif operation == "hdbscan":
        module.HDBSCANAnalyzer(min_cluster_size=2).fit(X[:, :2])
    elif operation == "hdbscan_presentation":
        module.HDBSCANAnalyzer().fit_presentation(X[:, :2], min_cluster_size=2)
    elif operation == "hdbscan_sensitivity":
        module.compute_sensitivity(X[:, :2], min_samples=2, min_sizes=[2])
    elif operation == "plotly_umap":
        module.EmbeddingEngine().compute_umap3d(
            X,
            pd.DataFrame(index=range(8)),
            np.repeat("STAR", 8),
            np.zeros(8, dtype=int),
            Path.cwd(),
        )
    elif operation == "plotly_tsne":
        module.EmbeddingEngine(method="tsne").compute_tsne3d(
            X, pd.DataFrame(index=range(8)), np.repeat("STAR", 8), Path.cwd()
        )
    else:
        raise AssertionError(operation)


def main(args):
    sys.stdout.reconfigure(encoding="utf-8")
    root = Path(args["root"]).resolve()
    sys.path.insert(0, str(root / "src"))
    assert not any(n == "dimred" or n.startswith("dimred.") for n in sys.modules)
    assert not any(n.split(".")[0] in args["blocked"] for n in sys.modules)
    blocker = BlockImports(args["blocked"])
    sys.meta_path.insert(0, blocker)
    # Positive controls run BEFORE the operation; clear recorded control hits.
    for dependency in args["blocked"]:
        for name in (dependency, dependency + ".m29_child"):
            try:
                importlib.import_module(name)
            except ModuleNotFoundError as error:
                assert error.name == dependency
            else:
                raise AssertionError(f"blocker failed for {name}")
        # Exercise the descendant rule directly as the parent is unavailable.
        try:
            blocker.find_spec(dependency + ".m29_child")
        except ModuleNotFoundError as error:
            assert error.name == dependency + ".m29_child"
        else:
            raise AssertionError("descendant blocker failed")
    blocker.hits.clear()
    violations = guarded_imports(Path.cwd().resolve())
    before_path = list(sys.path)
    root_logger = logging.getLogger()
    before_logging = (root_logger.level, tuple(root_logger.handlers))
    mode, operation = args["mode"], args["operation"]
    if mode in {"wiring", "wiring_failure"}:
        # This checks the real __init__ against controlled sibling producers.
        # It cannot establish real dependency availability or component behavior.
        for source, producer in EXPORTS.values():
            fullname = f"dimred.{source}"
            module = sys.modules.setdefault(fullname, types.ModuleType(fullname))
            setattr(module, producer, type(producer, (), {"__module__": fullname}))
        if mode == "wiring_failure":
            # Fail while executing the producer, rather than simulate absence.
            # The public import must not silently replace a broken core PCA API.
            del sys.modules["dimred.pca_analyzer"]

            class BrokenProducer(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, path=None, target=None):
                    if fullname == "dimred.pca_analyzer":
                        raise ImportError("M29 producer ABI failure PCA")

            sys.meta_path.insert(0, BrokenProducer())
        package = importlib.import_module("dimred")
        if mode == "wiring_failure":
            getattr(package, "PCAAnalyzer")  # A legitimate lazy facade may defer it.
            raise AssertionError("core producer failure was hidden")
        check_exports(package)
    elif mode in {"real", "optional_real", "style"}:
        if mode == "style":
            import matplotlib as mpl

            mpl.rcParams["axes.grid"] = False
            mpl.rcParams["figure.dpi"] = 91
            before = {k: mpl.rcParams[k] for k in ("axes.grid", "figure.dpi")}
        if mode == "real" and operation:
            exec(operation, {})
        package = importlib.import_module("dimred")
        assert Path(package.__file__).resolve() == root / "src/dimred/__init__.py"
        check_exports(package)
        if mode == "style":
            after = {k: mpl.rcParams[k] for k in before}
            print(json.dumps(dict(before=before, after=after)))
            assert before == after, "import changed caller Matplotlib configuration"
        elif mode == "optional_real":
            optional_operation(package, operation)
        else:
            # Inspect eager loads without declaring module presence a defect.
            print(json.dumps({d: d in sys.modules for d in OPTIONAL}))
            if "torch" in sys.modules:
                assert not sys.modules["torch"].cuda.is_initialized(), "GPU initialized"
    elif mode in {"isolated", "optional"}:
        source = (
            operation
            if mode == "isolated"
            else (
                "hdbscan_analyzer" if operation.startswith("hdbscan") else "embedding"
            )
        )
        module = isolated(root, source)
        if mode == "optional":
            optional_operation(module, operation)
        elif operation == "pca_analyzer":
            import numpy as np

            pca = module.PCAAnalyzer(n_components=1)
            pca.fit(np.array([[-2.0, 0.0], [-1.0, 0.0], [1.0, 0.0], [2.0, 0.0]]))
            assert pca is not None  # Producer's numerical contracts belong to M2.2.
    elif mode == "entry":
        exec(operation, {})
    elif mode == "guard_control":
        import socket

        for action in (
            lambda: socket.getaddrinfo("m29.invalid", 443),
            lambda: (root / "m29_forbidden_write").write_text("forbidden"),
        ):
            try:
                action()
            except RuntimeError as error:
                assert "M29 forbidden" in str(error)
            else:
                raise AssertionError("guard did not intercept forbidden action")
        assert len(violations) == 2
        violations.clear()
    else:
        raise AssertionError(mode)
    assert sys.path == before_path, "import changed sys.path"
    assert (
        root_logger.level,
        tuple(root_logger.handlers),
    ) == before_logging, "import changed root logging"
    assert not violations, f"caught forbidden attempts: {violations}"
    print("M29_OK", json.dumps(dict(hits=blocker.hits)))


if __name__ == "__main__":
    main(json.loads(sys.argv[1]))
