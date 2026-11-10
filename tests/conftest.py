"""Synthetic CSV/FITS fixtures for the isolated dimred loader audit."""

import importlib.util
from itertools import count
from pathlib import Path
import sys
import types
import re
import math

from astropy.io import fits
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def reporter_module():
    """Load the actual reporter independently of the optional facade (#21)."""
    path = Path(__file__).resolve().parents[1] / "src/dimred/run_reporter.py"
    spec = importlib.util.spec_from_file_location("m27_reporter", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def reporter_contracts():
    path = Path(__file__).resolve().parent / "reporter_review_contracts.py"
    spec = importlib.util.spec_from_file_location("m27_review_contracts", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def reporter_case(tmp_path, monkeypatch, reporter_module):
    """Synthetic populations with row markers and deterministic method boundaries."""
    monkeypatch.setattr(reporter_module, "make_timestamp", lambda: "20261109T170304Z")
    reports = tmp_path / "reports"
    figures = tmp_path / "figures"
    figures.mkdir()
    # Distinct values make row permutations and method substitutions observable.
    X = np.array(
        [[3.0, 0, 0], [-3, 0, 0], [0, 2, 0], [0, -2, 0], [0, 0, 1], [0, 0, -1]]
    )
    y = np.array(["STAR", "QSO", "STAR", "GALAXY", "STAR", "QSO"])
    meta = pd.DataFrame({"obsid": [91, 7, 300, 18, 2, 700]}, index=[8, 3, 9, 1, 0, 6])
    scores = np.array(
        [
            [11.0, 2, 31, 4],
            [19, 7, 8, 43],
            [3, 17, 5, 67],
            [23, 6, 41, 9],
            [5, 29, 13, 2],
            [37, 1, 7, 59],
        ]
    )

    class PCA:
        cumulative_variance = np.array([0.80, 0.90, 0.96, 1.0])

        def n_components_for_variance(self, threshold):
            return next(
                i
                for i, total in enumerate(self.cumulative_variance, 1)
                if total >= threshold
            )

        def variance_summary(self):
            return pd.DataFrame({"PC": [1, 2, 3, 4], "ratio": [0.8, 0.1, 0.06, 0.04]})

        def top_features_per_pc(self, pc_idx=0, n_top=10):
            return pd.DataFrame(
                {"feature": ["gamma", "alpha"], "loading": [1 + pc_idx, -2 - pc_idx]}
            )

    class AE:
        history_ = {
            "train_loss": [9.0, 3.0, 5.0, 2.0],
            "val_loss": [8.0, 1.0, 4.0, 6.0],
        }
        fit_time_ = 12.345

        def __init__(self):
            self.encoded = []

        def encode(self, values):
            self.encoded.append(np.array(values, copy=True))
            return values[:, :2] * [2.0, 3.0]

        def reconstruction_mse(self, values):
            # A fixed linear projection discards the third coordinate.
            residual = values[:, 2]
            return float(residual @ residual / values.size)

    common = {"paths": {"REPORTS_DIR": reports}, "FIGURES_DIR": figures}
    features = tmp_path / "features_synthetic.csv"
    catalog = tmp_path / "catalog_synthetic.csv"
    names = ["gamma", "alpha", "beta"]
    pca_args = dict(
        common,
        pca=PCA(),
        pca_spec=PCA(),
        scores=scores,
        scores_spec=scores[:, [2, 0]],
        X=X,
        X_spec=X[:, ::-1],
        y=y,
        y_spec=y[::-1],
        meta=meta,
        meta_spec=meta.iloc[::-1],
        feature_names=names,
        FEATURES_PATH=features,
        CATALOG_PATH=catalog,
    )
    embedding_args = dict(
        common,
        Z_umap=scores[:, :2],
        Z_tsne=None,
        y=y,
        meta=meta,
        scores_95=scores[:, :3],
    )
    ae = AE()
    ae_args = dict(
        common,
        ae=ae,
        X=X,
        y=y,
        meta=meta,
        feature_names=names,
        FEATURES_PATH=features,
        CATALOG_PATH=catalog,
    )
    return types.SimpleNamespace(
        root=tmp_path,
        reports=reports,
        figures=figures,
        X=X,
        y=y,
        meta=meta,
        scores=scores,
        ae=ae,
        pca_args=pca_args,
        embedding_args=embedding_args,
        ae_args=ae_args,
    )


@pytest.fixture
def bridge_module(monkeypatch):
    """Execute the actual bridge source while restoring its sys.path changes."""
    monkeypatch.setattr(sys, "path", sys.path.copy())
    path = Path(__file__).resolve().parents[1] / "src/dimred/xgboost_bridge.py"
    spec = importlib.util.spec_from_file_location("m26_bridge", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def bridge_case(tmp_path, monkeypatch, bridge_module):
    """Provide real temporary CSV/discovery and a recording classifier boundary."""
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("m26_utils", root / "src/utils.py")
    utilities = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(utilities)
    monkeypatch.setitem(sys.modules, "utils", utilities)

    class Spy:
        def __init__(self):
            self.predict_inputs = []
            self.proba_inputs = []
            self.predict_fn = lambda X: X["feature_alpha"].to_numpy(dtype=int) % 3
            self.proba_fn = lambda X: np.array(
                [[0.1, 0.2, 0.7], [0.55, 0.25, 0.2], [0.1, 0.8, 0.1]]
            )[X["feature_alpha"].to_numpy(dtype=int) % 3]

        def predict(self, X):
            self.predict_inputs.append(X.copy(deep=True))
            return self.predict_fn(X)

        def predict_proba(self, X):
            self.proba_inputs.append(X.copy(deep=True))
            return self.proba_fn(X)

    spy = Spy()
    encoder = types.SimpleNamespace(
        classes_=np.array(["G", "M", "F"]),
        inverse_transform=lambda codes: np.array(["G", "M", "F"])[codes],
    )
    classifier = types.SimpleNamespace(
        feature_names_used=["feature_gamma", "feature_alpha", "feature_beta"],
        class_labels=["G", "M", "F"],
        label_encoder=encoder,
        model_pipeline=spy,
    )
    loaded = []

    def load(path):
        loaded.append(str(path))
        return classifier

    package = types.ModuleType("pipeline")
    package.__path__ = []
    module = types.ModuleType("pipeline.classifier")
    module.SpectralClassifier = types.SimpleNamespace(load_model=load)
    monkeypatch.setitem(sys.modules, "pipeline", package)
    monkeypatch.setitem(sys.modules, "pipeline.classifier", module)
    models = tmp_path / "models"
    processed = tmp_path / "processed"
    models.mkdir()
    processed.mkdir()
    model = models / "spectral_classifier_fixture.pkl"
    model.write_text("not deserialized: spy boundary", encoding="utf-8")
    frame = pd.DataFrame(
        {
            "obsid": [104, 19, 807, 33, 250],
            "feature_alpha": [3, 10, 8, 23, 4],
            "feature_beta": [101, 31, 89, 17, 53],
            "feature_gamma": [3, 8, 1, 6, 4],
            "snr_r": [10.0, 11.0, 20.0, 40.0, 50.0],
        }
    )
    csv = processed / "features_fixture.csv"
    frame.to_csv(csv, index=False)
    args = {
        "paths": {
            "MODELS_DIR": models,
            "PROCESSED_DIR": processed,
            "SRC_DIR": root / "src",
        },
        "features_stem": "features_fixture",
        "Z_umap": np.column_stack((frame["obsid"], [-2, 8, -9, 3, 7])),
        "y": np.array(["true104", "true19", "true807", "true33", "true250"]),
        "cluster_labels": np.array([7, -1, 2, 9, 1]),
    }

    def run(**overrides):
        return bridge_module.load_and_predict(**(args | overrides))

    return types.SimpleNamespace(
        module=bridge_module,
        classifier=classifier,
        spy=spy,
        args=args,
        frame=frame,
        csv=csv,
        model=model,
        loaded=loaded,
        run=run,
        import_module=module,
        utils=utilities,
    )


@pytest.fixture
def bridge_review_contracts(caplog):
    """Accept explicit domain refusals without treating unrelated errors as evidence."""
    # Separate domain and cause checks do not prescribe diagnostic word order.
    # Generic "invalid column" is insufficient evidence of a data/schema refusal.
    patterns = {
        "identity": (
            r"\b(?:observations?|population|identity|alignment|provenance)\b",
            r"mismatch|incompatib|unverified|unknown|ambig|cannot|cardinal|length",
        ),
        "schema": (
            r"\b(?:schema|features?|columns?)\b",
            r"missing|incompatib|unsupported|non[ -]?numeric|absent",
        ),
        "preprocessing": (
            r"\b(?:input|features?|columns?|schema|preprocessing)\b",
            r"\bnan\b|missing|non[ -]?finite|incompatib",
        ),
        "probability": (
            r"\b(?:probability|probabilities|distribution)\b",
            r"invalid|undefined|non[ -]?finite|shape|rank|dimension|unavailable",
        ),
    }

    def relevant(message, domain):
        # These programming faults cannot establish a scientific refusal even
        # when a wrapper also mentions a domain. This is a bounded M2.6 helper.
        if re.search(
            r"off[ -]by[ -]one|insertion loop|column index|buffer overflow",
            message,
            re.I,
        ):
            return False
        return all(re.search(pattern, message, re.I) for pattern in patterns[domain])

    def invoke(case, domain, **overrides):
        caplog.clear()
        try:
            result = case.run(**overrides)
        except ValueError as exc:
            assert relevant(
                str(exc), domain
            ), f"Unrelated ValueError is not a {domain} validation refusal: {exc}"
            return None
        if result["_xgb_ok"]:
            return result
        assert result.get("has_proba") is False
        for key in ("y_pred", "confidence"):
            assert result.get(key) is None or np.asarray(result[key]).size == 0
        messages = " ".join(
            record.getMessage() for record in caplog.records if record.levelno >= 40
        )
        assert relevant(
            messages, domain
        ), f"A structured {domain} refusal needs a relevant error diagnostic: {messages}"
        return None

    def unavailable(result, count):
        if result is None:
            return
        assert result["_xgb_ok"] is True
        assert (
            result["has_proba"] is False
        ), "Invalid probabilities cannot become invented confidence."
        for key in ("y_pred", "y_aligned", "cl_aligned", "Z_umap_aligned"):
            assert len(result[key]) == count
        if result.get("confidence") is not None:
            assert np.asarray(result["confidence"]).shape == (count,)

    return types.SimpleNamespace(invoke=invoke, unavailable=unavailable)


@pytest.fixture
def bridge_loader_population(bridge_case, dimred_loader_module):
    """Build populations with the actual loader, independent of bridge filtering."""

    def make(scenario):
        c = bridge_case
        ids = np.array([311, 42, 9001, 77, 5, 640, 128, 3, 2048, 999])
        df = pd.DataFrame(
            {
                "obsid": ids,
                "id_echo": ids,
                "snr_r": 20.0,
                "class": [
                    "STAR",
                    "GALAXY",
                    "QSO",
                    "STAR",
                    "GALAXY",
                    "QSO",
                    "STAR",
                    "GALAXY",
                    "QSO",
                    "STAR",
                ],
                "feature_alpha": [3, 10, 8, 23, 4, 6, 12, 17, 26, 40],
                "feature_beta": [101, 31, 89, 17, 53, 8, 97, 13, 79, 41],
                "feature_gamma": np.arange(10) + 1.0,
            }
        )
        options = {"scale": False}
        removed = []
        if scenario == "default":
            df["radial_velocity_corr"] = np.arange(10, dtype=float)
            df.loc[2, "radial_velocity_corr"] = np.nan
            df["label"] = np.arange(10, dtype=float)
            df.loc[6, "label"] = np.nan
            removed = [2]
        elif scenario in {"nb01", "unequal"}:
            df["seeing"] = np.arange(10, dtype=float) + 1.0
            df.loc[2, "seeing"] = np.nan
            if scenario == "nb01":
                df.loc[7, "class"] = "Unknown"
                removed = [7]
            options.update(spectro_only=True, classes=["STAR", "GALAXY", "QSO"])
        else:
            assert scenario == "shared"
        df.to_csv(c.csv, index=False)
        catalog = c.csv.with_name("catalog_review.csv")
        pd.DataFrame({"obsid": ids}).to_csv(catalog, index=False)
        loader = dimred_loader_module.DimRedDataLoader(c.csv, catalog)
        X, y, meta = loader.load(**options)
        population = loader._df_merged.copy(deep=True)
        expected_ids = np.delete(ids, removed)
        np.testing.assert_array_equal(population.obsid.to_numpy(), expected_ids)
        # Check actual X against retained raw values, not only its row count.
        np.testing.assert_array_equal(
            X, population[loader.get_feature_names()].to_numpy(float)
        )
        np.testing.assert_array_equal(y, population["class"].to_numpy())
        np.testing.assert_array_equal(meta["class"], y)
        c.classifier.feature_names_used = ["feature_gamma", "id_echo", "feature_beta"]
        c.spy.predict_fn = lambda frame: frame["id_echo"].to_numpy(int) % 3
        c.spy.proba_fn = lambda frame: np.tile([0.2, 0.3, 0.5], (len(frame), 1))
        c.args.update(
            Z_umap=np.column_stack((expected_ids, np.arange(len(expected_ids)))),
            y=y,
            cluster_labels=expected_ids + 10000,
        )
        c.population, c.loader, c.raw = population, loader, df
        return c

    return make


@pytest.fixture
def bridge_real_bundle(tmp_path, monkeypatch, bridge_module):
    """Fit/save only a tiny trusted CPU model; use real classifier and joblib IO."""
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from xgboost import XGBClassifier

    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "src"))
    package = types.ModuleType("pipeline")
    package.__path__ = [str(root / "src/pipeline")]
    monkeypatch.setitem(sys.modules, "pipeline", package)
    classifier_spec = importlib.util.spec_from_file_location(
        "pipeline.classifier", root / "src/pipeline/classifier.py"
    )
    classifier_module = importlib.util.module_from_spec(classifier_spec)
    monkeypatch.setitem(sys.modules, "pipeline.classifier", classifier_module)
    classifier_spec.loader.exec_module(classifier_module)
    assert (
        Path(classifier_module.__file__).resolve()
        == root / "src/pipeline/classifier.py"
    )
    util_spec = importlib.util.spec_from_file_location(
        "m26_real_utils", root / "src/utils.py"
    )
    utilities = importlib.util.module_from_spec(util_spec)
    util_spec.loader.exec_module(utilities)
    monkeypatch.setitem(sys.modules, "utils", utilities)

    train = pd.DataFrame(
        {
            "feature_alpha": np.repeat([-4.0, 1.0, 6.0], 10)
            + np.tile(np.arange(10) / 20, 3),
            "feature_beta": np.tile([2.0, 7.0, -3.0, 5.0, 9.0], 6),
            "feature_gamma": np.arange(30) % 7 + 2.0,
        }
    )
    names = ["feature_gamma", "feature_alpha", "feature_beta"]
    encoder = LabelEncoder().fit(["M", "F", "G"])
    target = encoder.transform(np.repeat(["M", "F", "G"], 10))
    pipeline = Pipeline(
        [
            (
                "named",
                ColumnTransformer(
                    [
                        (
                            "numeric",
                            Pipeline(
                                [
                                    ("imputer", SimpleImputer(strategy="median")),
                                    ("scaler", StandardScaler()),
                                ]
                            ),
                            names,
                        )
                    ],
                    remainder="drop",
                ),
            ),
            (
                "xgb",
                XGBClassifier(
                    n_estimators=12,
                    max_depth=2,
                    min_child_weight=0,
                    n_jobs=1,
                    random_state=1729,
                    tree_method="hist",
                    device="cpu",
                    learning_rate=0.4,
                ),
            ),
        ]
    ).fit(train[names], target)
    clf = classifier_module.SpectralClassifier()
    clf.model_pipeline = pipeline
    clf.feature_names_used = names
    clf.class_labels = encoder.classes_.tolist()
    clf.label_encoder = encoder
    models, processed = tmp_path / "models", tmp_path / "processed"
    models.mkdir()
    processed.mkdir()
    model = models / "spectral_classifier_synthetic.pkl"
    clf.save_model(str(model), trained_on_file="synthetic_training_only.csv")
    query = train.iloc[[1, 14, 23, 8, 19]].reset_index(drop=True).copy()
    query["obsid"] = [104, 19, 807, 33, 250]
    csv = processed / "features_synthetic.csv"
    query.to_csv(csv, index=False)
    args = {
        "paths": {
            "SRC_DIR": root / "src",
            "MODELS_DIR": models,
            "PROCESSED_DIR": processed,
        },
        "features_stem": "features_synthetic",
        "Z_umap": np.column_stack((query.obsid, [-2, 8, -9, 3, 7])),
        "y": np.array(["M", "F", "G", "M", "F"]),
        "cluster_labels": np.array([7, -1, 2, 9, 1]),
    }
    return types.SimpleNamespace(
        module=bridge_module,
        classifier=clf,
        classifier_module=classifier_module,
        train=train,
        query=query,
        csv=csv,
        model=model,
        args=args,
        run=lambda **kw: bridge_module.load_and_predict(**(args | kw)),
    )


@pytest.fixture
def bridge_preprocessing_oracle(bridge_real_bundle, monkeypatch):
    """Observe actual booster input and independently compute train-only transforms."""
    c = bridge_real_bundle
    names = c.classifier.feature_names_used
    medians, means, scales = [], [], []
    for name in names:
        values = sorted(float(v) for v in c.train[name])
        medians.append((values[14] + values[15]) / 2)
        mean = math.fsum(values) / 30
        means.append(mean)
        scales.append(math.sqrt(math.fsum((v - mean) ** 2 for v in values) / 30))
    medians, means, scales = map(np.asarray, (medians, means, scales))
    np.testing.assert_allclose(medians, [5.0, 1.225, 5.0], rtol=0, atol=2e-15)
    np.testing.assert_allclose(means, [29 / 6, 1.225, 4.0], rtol=0, atol=2e-15)
    loaded_models, booster_inputs = [], []

    def trained_state(clf):
        numeric = clf.model_pipeline.named_steps["named"].named_transformers_["numeric"]
        np.testing.assert_allclose(
            numeric.named_steps["imputer"].statistics_, medians, rtol=0, atol=2e-14
        )
        np.testing.assert_allclose(
            numeric.named_steps["scaler"].mean_, means, rtol=0, atol=2e-14
        )
        np.testing.assert_allclose(
            numeric.named_steps["scaler"].scale_, scales, rtol=0, atol=2e-14
        )
        return numeric

    numeric = trained_state(c.classifier)
    original_load = c.classifier_module.SpectralClassifier.load_model

    def load(path):
        loaded = original_load(path)
        trained_state(loaded)
        loaded_models.append(loaded)
        return loaded

    monkeypatch.setattr(
        c.classifier_module.SpectralClassifier, "load_model", staticmethod(load)
    )
    booster_type = type(c.classifier.model_pipeline.named_steps["xgb"])
    original_predict = booster_type.predict

    def predict(estimator, X, *args, **kwargs):
        booster_inputs.append(np.array(X, copy=True))
        return original_predict(estimator, X, *args, **kwargs)

    monkeypatch.setattr(booster_type, "predict", predict)

    def forbid_fit(*args, **kwargs):
        raise AssertionError("Inference must not refit learned preprocessing.")

    monkeypatch.setattr(type(numeric.named_steps["imputer"]), "fit", forbid_fit)
    monkeypatch.setattr(type(numeric.named_steps["scaler"]), "fit", forbid_fit)

    def expected(query):
        values = query[names].to_numpy(dtype=float, copy=True)
        for column in range(len(names)):
            values[np.isnan(values[:, column]), column] = medians[column]
        return (values - means) / scales

    def compare(actual, expected_values):
        assert np.asarray(actual).shape == expected_values.shape
        # Cap at eight float32 epsilons independently of the produced dtype.
        # Float64 arithmetic over30 training rows retains the stricter2e-14 floor.
        precision = min(
            np.finfo(np.asarray(actual).dtype).eps, np.finfo(np.float32).eps
        )
        np.testing.assert_allclose(
            actual, expected_values, rtol=8 * precision, atol=max(2e-14, 8 * precision)
        )
        for loaded in loaded_models:
            trained_state(loaded)

    return types.SimpleNamespace(
        case=c,
        expected=expected,
        compare=compare,
        inputs=booster_inputs,
        loaded_models=loaded_models,
        medians=medians,
        means=means,
        scales=scales,
    )


@pytest.fixture
def autoencoder_numeric_module(monkeypatch):
    """Execute source with inert Torch names; never execute neural operations.

    This bypasses known import defect #21 for NumPy/plotting contracts only.
    Fresh subprocess proofs separately test the actual dependency boundary.
    """
    path = Path(__file__).resolve().parents[1] / "src/dimred/autoencoder.py"
    names = ["torch", "torch.nn", "torch.optim", "torch.utils", "torch.utils.data"]
    modules = {name: types.ModuleType(name) for name in names}
    modules["torch.nn"].Module = object
    modules["torch.utils.data"].DataLoader = object
    modules["torch.utils.data"].TensorDataset = object
    with monkeypatch.context() as context:
        for name, module in modules.items():
            context.setitem(sys.modules, name, module)
        spec = importlib.util.spec_from_file_location("m25_numeric", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module


@pytest.fixture
def autoencoder_real_module():
    """Load the actual module using the installed CPU PyTorch, if available."""
    torch = pytest.importorskip("torch")
    previous_threads = torch.get_num_threads()
    previous_rng = torch.get_rng_state()
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.set_num_threads(1)
    torch.manual_seed(1729)
    torch.use_deterministic_algorithms(True)
    path = Path(__file__).resolve().parents[1] / "src/dimred/autoencoder.py"
    spec = importlib.util.spec_from_file_location("m25_real", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    try:
        yield module
    finally:
        torch.set_rng_state(previous_rng)
        torch.use_deterministic_algorithms(
            previous_deterministic, warn_only=previous_warn_only
        )
        torch.set_num_threads(previous_threads)


@pytest.fixture
def autoencoder_plotting(monkeypatch):
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    monkeypatch.setattr(plt, "show", lambda: None)
    # Known layout/glyph warnings are presentation limitations, not math oracles.
    monkeypatch.setattr(plt, "tight_layout", lambda: None)
    yield plt
    plt.close("all")


@pytest.fixture
def embedding_module():
    # Isolate the optional facade (#21), but execute the real embedding source.
    path = Path(__file__).resolve().parents[1] / "src/dimred/embedding.py"
    spec = importlib.util.spec_from_file_location("dimred_embedding_audit", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def embedding_backends(monkeypatch):
    """Capture estimator boundaries without claiming algorithm validation."""
    import sys
    import types
    import sklearn.manifold

    instances = []

    class Estimator:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.input = None
            self.output = None
            instances.append(self)

        def fit_transform(self, X):
            self.input = np.array(X, copy=True)
            self.n_features_in_ = X.shape[1]
            self.output = np.column_stack(
                [X[:, j % X.shape[1]] for j in range(self.kwargs["n_components"])]
            )
            return self.output

        def transform(self, X):
            if X.shape[1] != self.n_features_in_:
                raise ValueError("Feature count differs from fitted input")
            return np.column_stack(
                [X[:, j % X.shape[1]] for j in range(self.kwargs["n_components"])]
            )

    monkeypatch.setitem(sys.modules, "umap", types.SimpleNamespace(UMAP=Estimator))
    monkeypatch.setattr(sklearn.manifold, "TSNE", Estimator)
    return instances


@pytest.fixture
def embedding_figures(monkeypatch):
    """Capture exact Plotly tables and export requests, without large HTML."""
    import plotly.express as px

    calls = []

    class Figure:
        def __init__(self, data_frame, **kwargs):
            self.table = data_frame.copy(deep=True)
            self.kwargs = kwargs
            self.paths = []
            calls.append(self)

        def update_traces(self, **kwargs):
            return self

        def update_layout(self, **kwargs):
            return self

        def write_html(self, path):
            self.paths.append(Path(path))

    monkeypatch.setattr(px, "scatter_3d", Figure)
    return calls


@pytest.fixture(scope="session")
def dimred_pca_class():
    # Exercise the real source; the optional-dependency facade is tracked in #21.
    path = Path(__file__).resolve().parents[1] / "src/dimred/pca_analyzer.py"
    spec = importlib.util.spec_from_file_location("dimred_pca_audit", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.PCAAnalyzer


@pytest.fixture
def pca_axes():
    # Six paired observations: covariance eigenvalues 18/5, 8/5, 2/5.
    # Rational rotation mixes features without imposing sklearn's axis signs.
    basis = np.array([[0.6, 0.8, 0.0], [-0.8, 0.6, 0.0], [0.0, 0.0, 1.0]])
    coordinates = np.array(
        [
            [3.0, 0, 0],
            [-3.0, 0, 0],
            [0, 2.0, 0],
            [0, -2.0, 0],
            [0, 0, 1.0],
            [0, 0, -1.0],
        ]
    )
    mean = np.array([10.0, -7.0, 4.0])
    return mean + coordinates @ basis, basis, coordinates, mean


@pytest.fixture(scope="session")
def dimred_loader_module():
    # Load the actual source without executing dimred's optional Torch facade
    # (known issue #21). This does not validate `import dimred`.
    path = Path(__file__).resolve().parents[1] / "src/dimred/data_loader.py"
    spec = importlib.util.spec_from_file_location("dimred_loader_audit", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def dimred_feature_loader(tmp_path, dimred_loader_module):
    instances = count()

    def make(features=None, catalog=None, **kwargs):
        if features is None:
            features = {
                "obsid": [30, 10, 20, 40],
                "class": ["STAR", "QSO", "STAR", "GALAXY"],
                "snr_r": [30.0, 10.0, 9.0, 40.0],
                "feature_b": [3.0, 1.0, 2.0, 4.0],
                "feature_a": [9.0, 1.0, 4.0, 16.0],
            }
        if catalog is None:
            catalog = {
                "obsid": [10, 40, 30, 20],
                "teff_gspphot": [1000.0, 4000.0, 3000.0, 2000.0],
            }
        case_dir = tmp_path / f"dimred-features-{next(instances)}"
        case_dir.mkdir()
        feature_path = case_dir / "features.csv"
        catalog_path = case_dir / "catalog.csv"
        pd.DataFrame(features).to_csv(feature_path, index=False)
        pd.DataFrame(catalog).to_csv(catalog_path, index=False)
        return dimred_loader_module.DimRedDataLoader(
            feature_path, catalog_path, **kwargs
        )

    return make


@pytest.fixture
def dimred_fits_writer(tmp_path):
    def write(
        name="plan/spectrum.fits",
        flux=None,
        coeff0=None,
        coeff1=0.001,
        data=None,
        header=None,
    ):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if flux is None:
            flux = np.arange(1.0, 11.0)
        if coeff0 is None:
            coeff0 = np.log10(4000.0)
        wavelength = 10 ** (coeff0 + np.arange(len(flux)) * coeff1)
        if data is None:
            data = np.vstack([flux, np.ones(len(flux))])
        hdu = fits.PrimaryHDU(data=data)
        hdu.header["COEFF0"] = coeff0
        hdu.header["COEFF1"] = coeff1
        if header:
            hdu.header.update(header)
        hdu.writeto(path, overwrite=True)
        return path, wavelength

    return write


@pytest.fixture
def dimred_spectral_loader(tmp_path, dimred_loader_module):
    instances = count()

    def make(paths, grid, catalog=None, **kwargs):
        if catalog is None:
            catalog = {
                "fits_name": [p.name for p in paths],
                "class": ["STAR"] * len(paths),
                "snr_r": [20.0] * len(paths),
                "teff_gspphot": list(range(1000, 1000 + len(paths))),
            }
        case_dir = tmp_path / f"dimred-spectra-{next(instances)}"
        case_dir.mkdir()
        catalog_path = case_dir / "spectral_catalog.csv"
        pd.DataFrame(catalog).to_csv(catalog_path, index=False)
        return dimred_loader_module.SpectralMatrixLoader(
            tmp_path, catalog_path, wl_grid=np.asarray(grid), n_jobs=1, **kwargs
        )

    return make


@pytest.fixture
def dimred_guarded_parallel(monkeypatch):
    """Keep real loky workers offline without disabling their IPC machinery."""
    import joblib

    real_parallel = joblib.Parallel

    def deny_worker_network():
        import socket

        def blocked(*args, **kwargs):
            raise AssertionError("Network forbidden in spectral audit worker")

        socket.socket.connect = blocked
        socket.socket.connect_ex = blocked
        socket.create_connection = blocked

    def guarded_parallel(*args, **kwargs):
        return real_parallel(*args, initializer=deny_worker_network, **kwargs)

    def worker_has_network_guard():
        import socket

        return socket.create_connection.__name__ == "blocked"

    monkeypatch.setattr(joblib, "Parallel", guarded_parallel)
    assert all(
        guarded_parallel(n_jobs=2, backend="loky")(
            joblib.delayed(worker_has_network_guard)() for _ in range(2)
        )
    )
    return guarded_parallel


@pytest.fixture
def hdbscan_module():
    path = Path(__file__).resolve().parents[1] / "src/dimred/hdbscan_analyzer.py"
    spec = importlib.util.spec_from_file_location("m24_hdbscan", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def hdbscan_backend(monkeypatch):
    """Queue labels/errors at the backend boundary, without validating HDBSCAN."""
    instances, outcomes = [], []

    class Backend:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            instances.append(self)

        def fit_predict(self, Z):
            self.input = np.array(Z, copy=True)
            outcome = outcomes.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            self.labels_ = np.array(outcome, dtype=int)
            return self.labels_

    monkeypatch.setitem(sys.modules, "hdbscan", types.SimpleNamespace(HDBSCAN=Backend))
    return instances, outcomes


@pytest.fixture
def hdbscan_csv(tmp_path, monkeypatch, dimred_loader_module):
    # Bypass only the known optional facade (#21); use real loader constants.
    package = types.ModuleType("dimred")
    package.__path__ = [str(Path(__file__).resolve().parents[1] / "src/dimred")]
    monkeypatch.setitem(sys.modules, "dimred", package)
    monkeypatch.setitem(sys.modules, "dimred.data_loader", dimred_loader_module)
    counter = 0

    def write(data):
        nonlocal counter
        stem = f"m24_{counter}"
        counter += 1
        path = tmp_path / f"{stem}.csv"
        pd.DataFrame(data).to_csv(path, index=False)
        return {"PROCESSED_DIR": tmp_path}, stem, path

    return write
