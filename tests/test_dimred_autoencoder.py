"""Torch-free numerical contracts; inert import names do not validate a network."""

from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler


def test_independent_loader_and_pca_source_import_without_torch():
    root = Path(__file__).resolve().parents[1]
    code = """
import importlib.abc, importlib.util, pathlib, sys
class BlockTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] == 'torch':
            raise ModuleNotFoundError('Torch deliberately absent')
sys.meta_path.insert(0, BlockTorch())
for name in ['data_loader', 'pca_analyzer']:
    path = pathlib.Path(sys.argv[1]) / 'src' / 'dimred' / (name + '.py')
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
assert 'torch' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", code, str(root)],
        capture_output=True,
        text=True,
        timeout=45,
    )
    assert result.returncode == 0, result.stderr


def test_check_torch_diagnostic_when_boundary_is_reachable(autoencoder_numeric_module):
    module = autoencoder_numeric_module
    module._TORCH_AVAILABLE = False
    with pytest.raises(ImportError, match="[Tt]orch"):
        module._check_torch()
    with pytest.raises(ImportError, match="pip install torch"):
        module.SpectralAutoencoder(2)


@pytest.mark.parametrize("method", ["encode", "decode", "reconstruct", "save"])
def test_unfitted_operations_refuse(autoencoder_numeric_module, method, tmp_path):
    ae = autoencoder_numeric_module.SpectralAutoencoder.__new__(
        autoencoder_numeric_module.SpectralAutoencoder
    )
    ae._model = None
    value = str(tmp_path / "unfitted.pt") if method == "save" else np.ones((2, 2))
    with pytest.raises(RuntimeError, match="fit"):
        getattr(ae, method)(value)


@pytest.mark.parametrize("exact", [False, True])
def test_mse_manual_asymmetric_residuals(autoencoder_numeric_module, exact):
    ae = autoencoder_numeric_module.SpectralAutoencoder.__new__(
        autoencoder_numeric_module.SpectralAutoencoder
    )
    X = np.array([[1.0, 5.0, -2.0], [4.0, -3.0, 7.0], [8.0, 0.0, -1.0]])
    residual = (
        np.zeros_like(X)
        if exact
        else np.array([[1.0, 2.0, 2.0], [0.0, 3.0, 0.0], [2.0, 6.0, 3.0]])
    )
    original = X.copy()
    ae.reconstruct = lambda values: values - residual
    expected = [0.0, 0.0, 0.0] if exact else [3.0, 3.0, 49 / 3]
    np.testing.assert_allclose(
        ae.reconstruction_mse(X, True), expected, rtol=0, atol=1e-14
    )
    assert ae.reconstruction_mse(X) == pytest.approx(sum(expected) / 3)
    np.testing.assert_array_equal(X, original)


def test_summary_independent_moments_quantile_and_noncontiguous_classes(
    autoencoder_numeric_module,
):
    ae = autoencoder_numeric_module.SpectralAutoencoder.__new__(
        autoencoder_numeric_module.SpectralAutoencoder
    )
    # Residuals squared: 0, 1, 4, 16. Mean=21/4 != median=5/2.
    X = np.array([[0.0], [1.0], [2.0], [4.0]])
    ae.reconstruct = np.zeros_like
    row = ae.reconstruction_summary(X).iloc[0]
    assert row["classe"] == "all"
    assert row["n"] == 4
    assert row["mse_mean"] == 21 / 4
    assert row["mse_std"] == pytest.approx(np.sqrt(651) / 4)  # population ddof=0
    assert row["mse_median"] == 5 / 2
    assert row["mse_q95"] == pytest.approx(14.2)  # linear rank 2.85: 4+.85*12
    table = ae.reconstruction_summary(X, np.array([2, 9, 2, 2])).set_index("classe")
    assert set(table.index) == {2, 9}
    np.testing.assert_allclose(table.loc[2], [3, 20 / 3, np.sqrt(416 / 9), 4, 14.8])
    np.testing.assert_allclose(table.loc[9], [1, 1, 0, 1, 1])


@pytest.mark.integration
def test_pca_comparison_analytic_axes(autoencoder_numeric_module, dimred_pca_class):
    X = np.array(
        [
            [3.0, 0.0, 0.0],
            [-3.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, -2.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ]
    )
    pca = dimred_pca_class(n_components=3, svd_solver="full")
    pca.fit(X)
    ae = autoencoder_numeric_module.SpectralAutoencoder.__new__(
        autoencoder_numeric_module.SpectralAutoencoder
    )
    ae.latent_dim = 2
    ae.reconstruct = np.zeros_like
    result = ae.compare_with_pca(X, pca, [3, 1, 4, 2])
    assert list(result["n_components"]) == [2, 3, 1, 2]
    assert list(result["method"])[1:] == ["PCA"] * 3
    np.testing.assert_allclose(
        result["mse_mean"], [14 / 9, 0, 5 / 9, 1 / 9], atol=1e-15, rtol=0
    )


def test_pca_forwarding_same_rows_default_and_custom_order(autoencoder_numeric_module):
    ae = autoencoder_numeric_module.SpectralAutoencoder.__new__(
        autoencoder_numeric_module.SpectralAutoencoder
    )
    ae.latent_dim = 7
    X = np.arange(15.0).reshape(5, 3)
    ae.reconstruction_mse = lambda values: 11.0
    calls = []

    def error(values, n_components):
        np.testing.assert_array_equal(values, X)
        calls.append(n_components)
        return np.array([1.0, 1.0, 7.0])

    pca = SimpleNamespace(
        sklearn_pca=SimpleNamespace(n_components_=5), reconstruction_error=error
    )
    result = ae.compare_with_pca(X, pca)
    assert calls == [1, 2, 3, 5]
    np.testing.assert_allclose(result["mse_mean"], [11.0, 3.0, 3.0, 3.0, 3.0])


@pytest.mark.parametrize("steps", [1, 2, 5])
def test_interpolation_affine_oracle(autoencoder_numeric_module, steps):
    ae = autoencoder_numeric_module.SpectralAutoencoder.__new__(
        autoencoder_numeric_module.SpectralAutoencoder
    )
    ae.encode = lambda X: np.asarray(X) @ np.array([[2.0, 1.0], [-1.0, 3.0]]) + [
        1.0,
        -2.0,
    ]
    ae.decode = lambda Z: np.asarray(Z) @ np.array([[3.0, -1.0], [2.0, 4.0]]) + [
        -2.0,
        5.0,
    ]
    start, end = np.array([1.0, 2.0]), np.array([-3.0, 4.0])
    Z, reconstructed = ae.latent_interpolation(start, end[None, :], steps)
    # Encoded endpoints [1,5], [-9,7].
    fractions = np.arange(steps) / (steps - 1) if steps > 1 else np.array([0.0])
    expected = np.column_stack([1 - 10 * fractions, 5 + 2 * fractions])
    np.testing.assert_allclose(Z, expected, atol=1e-14, rtol=0)
    np.testing.assert_allclose(
        reconstructed,
        np.column_stack(
            [
                3 * expected[:, 0] + 2 * expected[:, 1] - 2,
                -expected[:, 0] + 4 * expected[:, 1] + 5,
            ]
        ),
    )


@pytest.mark.parametrize(
    "cluster_labels,expected",
    [
        (None, None),
        ([-1, -1, -1, -1, -1], -1),
        ([9, 9, 2, -1, 9], 9),
        ([9, -1, -1, -1, -1], 9),
    ],
)
@pytest.mark.parametrize("scaled", [False, True])
def test_candidate_scaling_reconstruction_vote_and_feature_order(
    autoencoder_numeric_module, autoencoder_plotting, cluster_labels, expected, scaled
):
    names = ["feature_Mg_b_eq_width", "feature_Hα_eq_width"]
    raw = np.array([5.0, -2.0])
    scaler = StandardScaler().fit([[1.0, -4.0], [5.0, 0.0]]) if scaled else None
    transformed = np.array([[1.0, 0.0]]) if scaled else raw[None, :]
    seen = []

    def encode(X):
        seen.append(X.copy())
        return np.array([[2.0, -1.0]])

    ae = SimpleNamespace(encode=encode, reconstruct=lambda X: X - [1.0, 3.0])
    result = autoencoder_numeric_module.tester_candidat(
        raw,
        ae,
        SimpleNamespace(scaler_=scaler),
        names,
        np.array([[2.0, -1.0], [3.0, -1.0], [2.0, 1.0], [6.0, -1.0], [2.0, 8.0]]),
        cluster_labels_train=(
            None if cluster_labels is None else np.array(cluster_labels)
        ),
    )
    assert result["cluster"] == expected
    assert result["mse"] == 5.0
    np.testing.assert_array_equal(seen[0], transformed)
    np.testing.assert_array_equal(result["x_scaled"], transformed[0])
    np.testing.assert_array_equal(result["x_recon"], transformed[0] - [1.0, 3.0])
    np.testing.assert_array_equal(result["z"], [2.0, -1.0])
    np.testing.assert_array_equal(raw, [5.0, -2.0])
    # Bars follow declared feature names, while the diagnostic lists Halpha first.
    bars = autoencoder_plotting.gcf().axes[1].patches
    np.testing.assert_array_equal(
        [p.get_width() for p in bars[:2]], transformed[0, [1, 0]]
    )


def test_candidate_nearest_fifty_excludes_distant_majority(
    autoencoder_numeric_module, autoencoder_plotting
):
    # Rows scrambled; near fifty have unique winner 7, far majority have label 2.
    distances = np.arange(120.0)
    order = np.random.default_rng(11).permutation(120)
    Z = np.column_stack([distances, np.zeros(120)])[order]
    labels = np.where(distances < 50, 7, 2)[order]
    ae = SimpleNamespace(encode=lambda X: np.zeros((1, 2)), reconstruct=np.zeros_like)
    result = autoencoder_numeric_module.tester_candidat(
        np.array([1.0, 2.0]), ae, object(), ["a", "b"], Z, labels
    )
    assert result["cluster"] == 7


@pytest.mark.parametrize("missing", ["teff_gspphot", "logg_gspphot"])
def test_arithmetic_missing_required_parameters(autoencoder_numeric_module, missing):
    meta = pd.DataFrame({"teff_gspphot": [4500.0], "logg_gspphot": [4.5]}).drop(
        columns=missing
    )
    with pytest.raises(ValueError, match="teff_gspphot.*logg_gspphot"):
        autoencoder_numeric_module.latent_arithmetic(
            np.ones((1, 2)), meta, np.array(["STAR"])
        )


@pytest.mark.parametrize("count", [1, 23, 207])
def test_latent_arithmetic_independent_barycentres_and_distances(
    autoencoder_numeric_module, autoencoder_plotting, count
):
    centres = np.array(
        [[1.0, 2.0], [4.0, -1.0], [8.0, 5.0], [-2.0, 7.0], [3.0, 9.0], [6.0, 12.0]]
    )
    temperatures = [3800.0, 5200.0, 6200.0, 4000.0, 5500.0, 7000.0]
    gravities = [4.0, 4.5, 5.0, 0.5, 2.0, 2.9]
    Z = np.repeat(centres, count, axis=0)
    # Vary rows so RNG selection/cap is discriminating beyond constant clusters.
    offsets = np.column_stack([np.arange(count) / 100, -(np.arange(count) ** 2) / 1000])
    Z += np.tile(offsets, (6, 1))
    meta = pd.DataFrame(
        {
            "teff_gspphot": np.repeat(temperatures, count),
            "logg_gspphot": np.repeat(gravities, count),
        }
    )
    y = np.repeat("STAR", 6 * count)
    # Excluded non-star, missing and boundary-outside rows must not contaminate.
    Z = np.vstack([Z, [[1e6, 1e6]] * 3])
    meta = pd.concat(
        [
            meta,
            pd.DataFrame(
                {
                    "teff_gspphot": [4500.0, np.nan, 7500.0],
                    "logg_gspphot": [4.5, 4.5, 5.5],
                }
            ),
        ],
        ignore_index=True,
    )
    y = np.r_[y, ["QSO", "STAR", "STAR"]]
    rng = np.random.default_rng(42)
    expected = np.array(
        [
            Z[
                rng.choice(
                    np.arange(i * count, (i + 1) * count),
                    min(200, count),
                    replace=False,
                )
            ].mean(axis=0)
            for i in range(6)
        ]
    )
    result = autoencoder_numeric_module.latent_arithmetic(Z, meta, y)
    for key, value in zip(
        ["K_naine", "G_naine", "F_naine", "K_geante", "G_geante", "F_geante"], expected
    ):
        np.testing.assert_allclose(result["populations"][key], value, atol=1e-12)
    predictions = [
        expected[3] - expected[0] + expected[1],
        expected[4] - expected[1] + expected[0],
    ]
    reference = np.sqrt(sum((expected[0] - expected[4]) ** 2))
    for row, predicted, target in zip(
        result["tests"], predictions, [expected[4], expected[3]]
    ):
        distance = np.sqrt(sum((predicted - target) ** 2))
        assert row["distance"] == pytest.approx(distance)
        assert row["ratio_pct"] == pytest.approx(100 * distance / reference)
    again = autoencoder_numeric_module.latent_arithmetic(Z, meta, y)
    assert result["populations"] == again["populations"]
    assert result["tests"] == again["tests"]
    assert result["fig"] is not None


def test_arithmetic_absent_populations(
    autoencoder_numeric_module, autoencoder_plotting
):
    result = autoencoder_numeric_module.latent_arithmetic(
        np.array([[1.0, 2.0]]),
        pd.DataFrame({"teff_gspphot": [4500.0], "logg_gspphot": [4.5]}),
        np.array(["QSO"]),
    )
    assert all(value is None for value in result["populations"].values())
    assert result["tests"] == []


@pytest.mark.integration
def test_candidate_temperature_mask_and_png_export(
    autoencoder_numeric_module, autoencoder_plotting, tmp_path
):
    import warnings

    Z = np.array([[1.0, 7.0], [9.0, -3.0], [-2.0, 4.0]])
    temperature = np.array([4200.0, np.nan, 6100.0])
    ae = SimpleNamespace(
        encode=lambda X: np.array([[2.0, 1.0]]), reconstruct=np.zeros_like
    )
    destination = tmp_path / "candidate.png"
    with warnings.catch_warnings():
        # Font coverage is platform-specific; this checks export/data, not glyph QA.
        warnings.filterwarnings(
            "ignore", message="Glyph .* missing from font", category=UserWarning
        )
        result = autoencoder_numeric_module.tester_candidat(
            np.array([1.0, 2.0]),
            ae,
            object(),
            ["a", "b"],
            Z,
            teff_train=temperature,
            save_path=destination,
        )
    assert destination.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    collection = autoencoder_plotting.gcf().axes[0].collections[0]
    np.testing.assert_array_equal(collection.get_offsets(), [[1.0, 7.0], [-2.0, 4.0]])
    np.testing.assert_array_equal(collection.get_array(), [4200.0, 6100.0])
    assert result["mse"] == 2.5


@pytest.mark.integration
def test_arithmetic_png_export_without_required_populations(
    autoencoder_numeric_module, autoencoder_plotting, tmp_path
):
    destination = tmp_path / "arithmetic.png"
    result = autoencoder_numeric_module.latent_arithmetic(
        np.array([[1.0, 2.0]]),
        pd.DataFrame({"teff_gspphot": [4500.0], "logg_gspphot": [4.5]}),
        np.array(["QSO"]),
        save_path=destination,
    )
    assert destination.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert result["fig"] is autoencoder_plotting.gcf()
    assert result["tests"] == []
