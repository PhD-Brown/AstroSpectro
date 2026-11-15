"""Fresh-process probes for M2.9; controlled wiring is explicitly synthetic."""

import ast
import copy
import importlib
import importlib.abc
import importlib.util
import json
import logging
import os
from pathlib import Path
import re
import subprocess
import sys
import traceback
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
AE_EXPORTS = ("SpectralAutoencoder", "tester_candidat", "latent_arithmetic")
TORCH_STATEMENTS = (
    "import dimred",
    "from dimred import PCAAnalyzer",
    "from dimred import SpectralAutoencoder",
    "from dimred.pca_analyzer import PCAAnalyzer",
    "from dimred.embedding import EmbeddingEngine",
    "from dimred.data_loader import DimRedDataLoader",
    "from dimred import *",
)
SCIENTIFIC_SUFFIXES = {
    ".fits",
    ".csv",
    ".pkl",
    ".joblib",
    ".pt",
    ".npy",
    ".npz",
    ".parquet",
    ".feather",
    ".h5",
    ".hdf5",
    ".pth",
    ".ckpt",
    ".safetensors",
}
NETWORK_EVENTS = {
    "socket.connect",
    "socket.getaddrinfo",
    "socket.sendto",
    "socket.gethostbyname",
    "socket.gethostbyaddr",
    "socket.bind",
}


def run_probe(
    tmp_path,
    mode,
    *,
    root=None,
    blocked=(),
    operation=None,
    real=False,
    startup_dir=None,
):
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
    if startup_dir is not None:
        # Process a private .pth before the helper starts, without editing a venv.
        bootstrap = (
            "import site, sys, runpy; "
            f"site.addsitedir({str(startup_dir)!r}); "
            f"sys.argv = {command[3:]!r}; "
            f"runpy.run_path({str(Path(__file__).resolve())!r}, run_name='__main__')"
        )
        command = [sys.executable, "-I", "-B", "-c", bootstrap]
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


def probe_record(result, prefix):
    """Read one structured child record; missing telemetry is a harness failure."""
    records = [
        json.loads(line[len(prefix) + 1 :])
        for line in result.stdout.splitlines()
        if line.startswith(prefix + " ")
    ]
    assert len(records) == 1, result.stdout + result.stderr
    return records[0]


def assert_torch_contract(result):
    """Shared desired-behavior oracle for red proofs and admissibility controls."""
    state = probe_record(result, "M29_PROVENANCE")
    assert state["executed"], "proof did not execute the checkout facade"
    if result.returncode:
        # Classify the unchanged defect before reporting it; never bless this error.
        # A successful lazy refusal need not attempt an import of the dependency.
        assert (
            "torch" in state["blocked_hits"]
        ), "failed operation did not hit the Torch blocker"
        error = state["exception"]
        assert error["type"] == "NameError" and any(
            Path(frame["file"]).name == "autoencoder.py" and frame["line"] == 87
            for frame in error["frames"]
        ), f"Unexpected #21 proof failure cause: {error}"
    assert_success(result)


def make_hijack(startup_dir, facade, expected=None):
    """Create a private .pth finder counterexample, without changing site-packages."""
    startup_dir.mkdir(parents=True, exist_ok=True)
    code = (
        "import sys, importlib.abc, importlib.util\n"
        "class Finder(importlib.abc.MetaPathFinder):\n"
        "    def find_spec(self, fullname, path=None, target=None):\n"
        "        if fullname == 'dimred':\n"
        f"            spec = importlib.util.spec_from_file_location(fullname, {str(facade)!r}, submodule_search_locations=[])\n"
    )
    if expected is not None:
        code += f"            spec.origin = {str(expected)!r}\n"
    code += "            return spec\nsys.meta_path.insert(0, Finder())\n"
    (startup_dir / "m29_hijack.py").write_text(code, encoding="utf-8")
    (startup_dir / "m29_hijack.pth").write_text("import m29_hijack\n", encoding="utf-8")


def torch_refusal(error):
    """Attribute only the final ImportError diagnostic, never preceding warnings."""
    diagnostic = str(error).splitlines()[-1] if str(error) else ""
    return (
        isinstance(error, ImportError)
        and "cannot import name" not in diagnostic.lower()
        and re.search(r"\b(?:torch|pytorch)\b", diagnostic, re.IGNORECASE) is not None
    )


def notebook_imports(source):
    """Parse static imports after masking standalone Jupyter commands, without execution."""
    lines = source.splitlines(keepends=True)
    # These notebooks use standalone magics, not Python inside cell-magic strings.
    normalized = "".join(
        (
            re.match(r"\s*", line)[0].replace("\n", "") + "pass\n"
            if line.lstrip().startswith(("%", "!"))
            else line
        )
        for line in lines
    )
    tree = ast.parse(normalized)  # Parse failure is explicit, not an empty inventory.
    return [
        node
        for node in ast.walk(tree)
        if (
            isinstance(node, ast.ImportFrom)
            and node.module
            and node.module.split(".")[0] == "dimred"
        )
        or (
            isinstance(node, ast.Import)
            and any(a.name.split(".")[0] == "dimred" for a in node.names)
        )
    ]


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
    # First pass: no voluntary producer imports. A lazy facade resolves itself.
    public = {}
    for name in EXPORTS:
        try:
            public[name] = getattr(package, name)
        except Exception as error:
            raise AssertionError(
                f"fresh public attribute {name}: {type(error).__name__}"
            ) from error
        assert callable(public[name]), f"callable {name}"
    namespace = {}
    exec("from dimred import *", namespace)
    assert set(namespace) - {"__builtins__"} == set(EXPORTS), "star surface"
    for name in EXPORTS:
        assert namespace[name] is public[name]
        named = {}
        exec(f"from dimred import {name}", named)
        assert named[name] is public[name], f"fresh named identity {name}"
    # Second pass: independent producer identity, after exercising the facade.
    for name, (source, producer) in EXPORTS.items():
        module = importlib.import_module(f"dimred.{source}")
        assert public[name] is getattr(module, producer), f"identity {name}"
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
        forbidden = event in NETWORK_EVENTS
        paths = []
        if event == "open":
            path, mode, flags = args
            if (
                isinstance(path, (str, bytes))
                and Path(os.fsdecode(path)).suffix.lower() in SCIENTIFIC_SUFFIXES
            ):
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


def check_torch_boundary(package):
    """Exercise independent constructors and AE exports without requiring Torch."""
    import numpy as np
    import pandas as pd

    for name in ("PCAAnalyzer", "DimRedDataLoader", "EmbeddingEngine"):
        assert callable(getattr(package, name, None)), f"independent export {name}"
    package.PCAAnalyzer(n_components=1)
    package.DimRedDataLoader("not-read-features.csv", "not-read-catalog.csv")
    package.EmbeddingEngine(method="tsne")
    outcomes = {}
    for name in AE_EXPORTS:
        try:
            value = getattr(package, name)
        except ImportError as error:
            assert torch_refusal(error), f"AE attribution {name}: {error}"
            outcomes[name] = "explicit Torch refusal at access"
            continue
        except Exception as error:
            raise AssertionError(f"AE access {name}: {type(error).__name__}") from error
        assert value is not None and callable(
            value
        ), f"AE export {name} must be callable"
        try:
            if name == "SpectralAutoencoder":
                model = value(input_dim=4, latent_dim=2, hidden_dims=[4], device="cpu")
                assert all(
                    callable(getattr(model, method, None))
                    for method in ("fit", "encode", "decode", "reconstruct")
                ), "AE instance protocol"
            else:
                import matplotlib.pyplot as plt

                class NumericModel:
                    def encode(self, X):
                        return np.zeros((len(X), 2))

                    def reconstruct(self, X):
                        return np.asarray(X).copy()

                figures = set(plt.get_fignums())
                try:
                    if name == "tester_candidat":
                        output = value(
                            x_raw=np.arange(4.0),
                            ae_model=NumericModel(),
                            loader_ref=types.SimpleNamespace(scaler_=None),
                            feature_names_ref=list("abcd"),
                            Z_train=np.zeros((4, 2)),
                        )
                        assert isinstance(output, dict) and {
                            "z",
                            "mse",
                            "x_recon",
                        } <= set(output), "AE helper tester_candidat result"
                    else:
                        output = value(
                            Z_ae=np.zeros((4, 2)),
                            meta=pd.DataFrame(
                                {
                                    "teff_gspphot": [12000.0] * 4,
                                    "logg_gspphot": [4.0] * 4,
                                }
                            ),
                            y=np.repeat("STAR", 4),
                        )
                        assert isinstance(output, dict) and {
                            "populations",
                            "tests",
                            "fig",
                        } <= set(output), "AE helper latent_arithmetic result"
                finally:
                    for number in set(plt.get_fignums()) - figures:
                        plt.close(number)
        except ImportError as error:
            assert torch_refusal(error), f"AE attribution {name}: {error}"
            outcomes[name] = "explicit Torch refusal at use"
        except (TypeError, NameError, AttributeError) as error:
            raise AssertionError(f"AE use {name}: {type(error).__name__}") from error
        else:
            outcomes[name] = "usable synthetic invocation"
    print("M29_AE", json.dumps(outcomes))


def rc_snapshot(mpl):
    """Copy caller state; backend selection is environmental, not plotting style."""
    return {
        key: copy.deepcopy(value)
        for key, value in mpl.rcParams.items()
        if key != "backend"
    }


def execute_probe(args):
    root = Path(args["root"]).resolve()
    blocker = BlockImports(args["blocked"])
    args["_blocker"] = blocker
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
            source = operation or "pca_analyzer"
            export = next(
                name for name, (module, _) in EXPORTS.items() if module == source
            )
            label = "PCA" if source == "pca_analyzer" else source
            del sys.modules[f"dimred.{source}"]

            class BrokenProducer(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, path=None, target=None):
                    if fullname == f"dimred.{source}":
                        raise ImportError(f"M29 producer ABI failure {label}")

            sys.meta_path.insert(0, BrokenProducer())
        package = importlib.import_module("dimred")
        if mode == "wiring_failure":
            getattr(package, export)  # A legitimate lazy facade may defer it.
            raise AssertionError("core producer failure was hidden")
        check_exports(package)
    elif mode in {"real", "optional_real", "style", "style_failed_import"}:
        if mode in {"style", "style_failed_import"}:
            if mode == "style":
                try:
                    import torch  # Deliberate style precondition, separate from API probes.
                except ImportError as error:
                    raise RuntimeError(
                        "M29_STYLE_UNAVAILABLE: real Torch required"
                    ) from error
                print("M29_STYLE_ENV", json.dumps({"torch": torch.__version__}))
            import matplotlib as mpl
            import matplotlib.pyplot as plt

            plt.get_backend()  # Resolve backend before the snapshot (backend is excluded).
            mpl.rcParams["axes.grid"] = False
            mpl.rcParams["figure.dpi"] = 91
            before = rc_snapshot(mpl)
        if mode == "real" and operation:
            exec(operation, {})
        failure = None
        try:
            package = importlib.import_module("dimred")
        except Exception as error:
            if mode != "style_failed_import":
                raise
            failure = {"type": type(error).__name__, "message": str(error)}
        if mode in {"style", "style_failed_import"}:
            after = rc_snapshot(mpl)
            changed = sorted(key for key in before if before[key] != after[key])
            print(
                "M29_STYLE",
                json.dumps(
                    dict(
                        changed=changed,
                        values={
                            key: {
                                "before": repr(before[key]),
                                "after": repr(after[key]),
                            }
                            for key in changed
                        },
                        import_failure=failure,
                        excluded=["backend"],
                    )
                ),
            )
            if mode == "style":
                assert not changed, "import changed caller Matplotlib configuration"
        if mode == "style_failed_import":
            print("M29_OK measured failed-import state, not a style-contract PASS")
            return
        assert Path(package.__file__).resolve() == root / "src/dimred/__init__.py"
        if mode != "style":
            check_exports(package)
        if mode == "optional_real":
            optional_operation(package, operation)
        else:
            # Inspect eager loads without declaring module presence a defect.
            print(json.dumps({d: d in sys.modules for d in OPTIONAL}))
            if "torch" in sys.modules:
                # CPU wheel + masked devices: observational, not a falsifiable GPU guard.
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
    elif mode in {"entry", "torch_boundary"}:
        if mode == "torch_boundary":
            try:
                exec(operation, {})
            except ImportError as error:
                assert operation in (
                    "from dimred import SpectralAutoencoder",
                    "from dimred import *",
                ) and torch_refusal(error), f"AE import attribution: {error}"
            except (TypeError, AttributeError, KeyError) as error:
                raise AssertionError(
                    f"public entry unusable: {type(error).__name__}"
                ) from error
            package = importlib.import_module("dimred")
            check_torch_boundary(package)
        else:
            exec(operation, {})
    elif mode == "guard_control":
        import socket

        actions = [
            ("socket.getaddrinfo", lambda: socket.getaddrinfo("m29.invalid", 443)),
            ("socket.gethostbyname", lambda: socket.gethostbyname("localhost")),
            ("socket.gethostbyaddr", lambda: socket.gethostbyaddr("127.0.0.1")),
            ("socket.bind", lambda: socket.socket().bind(("127.0.0.1", 0))),
            ("open", lambda: (root / "m29_forbidden_write").write_text("forbidden")),
        ]
        actions.extend(
            (
                "open",
                lambda suffix=suffix: open(Path.cwd() / ("not-read" + suffix), "rb"),
            )
            for suffix in sorted(SCIENTIFIC_SUFFIXES)
        )
        controls = []
        for event, action in actions:
            count = len(violations)
            try:
                action()
            except RuntimeError as error:
                assert "M29 forbidden" in str(error)
            else:
                raise AssertionError("guard did not intercept forbidden action")
            assert violations[count:] == [event], (event, violations[count:])
            controls.append(event)
        # Legitimate writes/reads in private caches remain possible.
        cache = Path.cwd() / "cache"
        cache.mkdir(exist_ok=True)
        (cache / "legitimate.json").write_text("{}")
        assert (cache / "legitimate.json").read_text() == "{}"
        print(
            "M29_GUARDS",
            json.dumps(dict(controls=controls, scope="Python audit events only")),
        )
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


def main(args):
    """Trace the actual facade even if Python removes its partial failed import."""
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    root = Path(args["root"]).resolve()
    sys.path.insert(0, str(root / "src"))
    assert not any(n == "dimred" or n.startswith("dimred.") for n in sys.modules)
    assert not any(n.split(".")[0] in args["blocked"] for n in sys.modules)
    expected = root / "src/dimred/__init__.py"
    state = dict(
        expected=str(expected),
        executed=[],
        loaded=None,
        meta_path=[
            f"{getattr(f, '__module__', type(f).__module__)}.{getattr(f, '__qualname__', type(f).__qualname__)}"
            for f in sys.meta_path
        ],
    )
    facade_mode = args["mode"] not in {"guard_control", "isolated", "optional"}

    def observe(event, values):
        if event == "exec" and facade_mode:
            module = sys.modules.get("dimred")
            if module is not None and not state["executed"]:
                path = getattr(module, "__file__", None)
                filename = values[0].co_filename
                state["executed"].append(dict(file=path, code=filename))
                assert (
                    path and Path(path).resolve() == expected
                ), "facade provenance mismatch"
                assert (
                    Path(filename).resolve() == expected
                ), "facade code provenance mismatch"

    sys.addaudithook(observe)
    try:
        if facade_mode:
            spec = importlib.util.find_spec(
                "dimred"
            )  # Resolve, do not execute producers.
            state["resolved"] = None if spec is None else spec.origin
            assert (
                spec and spec.origin and Path(spec.origin).resolve() == expected
            ), "facade provenance mismatch"
        execute_probe(args)
    except BaseException as error:
        state["exception"] = dict(
            type=type(error).__name__,
            message=str(error),
            final=traceback.format_exception_only(type(error), error)[-1].strip(),
            frames=[
                dict(file=f.filename, line=f.lineno, name=f.name)
                for f in traceback.extract_tb(error.__traceback__)
            ],
        )
        raise
    finally:
        state["blocked_hits"] = (
            list(args["_blocker"].hits) if "_blocker" in args else []
        )
        module = sys.modules.get("dimred")
        if module is not None:
            state["loaded"] = getattr(module, "__file__", None)
        print("M29_PROVENANCE", json.dumps(state))
        if facade_mode and module is not None:
            assert (
                state["loaded"] and Path(state["loaded"]).resolve() == expected
            ), "facade provenance mismatch"


if __name__ == "__main__":
    main(json.loads(sys.argv[1]))
