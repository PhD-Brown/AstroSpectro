"""Explicit red proofs and observations; never default-discovered or xfailed."""

from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


@pytest.mark.parametrize("operation", ["module", "package", "pca", "invoke"])
def test_absent_torch_optional_boundary_has_domain_diagnostic(operation):
    root = Path(__file__).resolve().parents[1]
    code = """
import importlib.abc, importlib.util, pathlib, sys
class BlockTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] == 'torch':
            raise ModuleNotFoundError('Torch deliberately absent')
sys.meta_path.insert(0, BlockTorch())
sys.path.insert(0, str(pathlib.Path(sys.argv[1]) / 'src'))
operation = sys.argv[2]
try:
    if operation in ['module', 'invoke']:
        spec = importlib.util.spec_from_file_location('isolated_autoencoder', pathlib.Path(sys.argv[1]) / 'src/dimred/autoencoder.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if operation == 'invoke':
            module.SpectralAutoencoder(2)
    elif operation == 'pca':
        from dimred import PCAAnalyzer
    else:
        import dimred
except ImportError as error:
    # Explicit optional-dependency refusal is admissible; NameError is not.
    assert 'torch' in str(error).lower()
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", code, str(root), operation],
        capture_output=True,
        text=True,
        timeout=45,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.slow
@pytest.mark.parametrize("N,batch,fraction", [(6, 4, 0.1), (2, 10, 0.1)])
def test_valid_small_training_does_not_crash_inside_batchnorm(
    autoencoder_real_module, N, batch, fraction
):
    m = autoencoder_real_module
    ae = m.SpectralAutoencoder(2, 1, [2], device="cpu")
    # Accept explicit domain validation, or defined successful training.
    try:
        history = ae.fit(
            np.arange(N * 2, dtype=float).reshape(N, 2),
            epochs=1,
            batch_size=batch,
            val_fraction=fraction,
            verbose=False,
        )
    except ValueError as error:
        assert "batch" in str(
            error
        ).lower() and "more than 1 value per channel" not in str(error)
    else:
        assert np.isfinite(history["val_loss"]).all()


@pytest.mark.slow
def test_failed_fit_cannot_expose_random_model_with_stale_history(
    autoencoder_real_module,
):
    m = autoencoder_real_module
    ae = m.SpectralAutoencoder(2, 1, [2], device="cpu")
    X = np.arange(16.0, dtype=float).reshape(8, 2)
    ae.fit(X, epochs=1, batch_size=20, verbose=False)
    previous = ae.encode(X)
    try:
        ae.fit(X, epochs=1, batch_size=0, verbose=False)
    except ValueError:
        pass
    else:
        pytest.fail("Invalid batch size unexpectedly accepted")
    # Either preserve the old completed model or invalidate the usable state.
    try:
        output = ae.encode(X)
    except (RuntimeError, ValueError):
        return
    np.testing.assert_array_equal(output, previous)


@pytest.mark.slow
@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_nonfinite_training_cannot_silently_report_success(
    autoencoder_real_module, value
):
    m = autoencoder_real_module
    ae = m.SpectralAutoencoder(2, 1, [2], device="cpu")
    X = np.full((8, 2), value)
    try:
        history = ae.fit(X, epochs=1, batch_size=20, verbose=False)
    except (ValueError, RuntimeError, FloatingPointError):
        return
    assert all(np.isfinite(values).all() for values in history.values()), history


@pytest.mark.slow
def test_epoch_mse_weights_observations_not_batches(
    autoencoder_real_module, monkeypatch
):
    m = autoencoder_real_module

    class ZeroNetwork(m.nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.level = m.nn.Parameter(m.torch.tensor(0.0))

        def forward(self, X):
            return m.torch.ones_like(X) * self.level, X

    monkeypatch.setattr(m, "_AutoencoderNet", ZeroNetwork)
    X = np.array([[0.0], [1.0], [3.0], [6.0], [10.0], [15.0], [21.0], [28.0]])
    validation = np.random.default_rng(42).choice(8, 1, replace=False)
    train = np.delete(X, validation, axis=0)
    expected = sum(float(row[0]) ** 2 for row in train) / len(train)
    ae = m.SpectralAutoencoder(1, 1, [1], device="cpu")
    history = ae.fit(X, epochs=1, batch_size=4, lr=0.0, weight_decay=0.0, verbose=False)
    assert history["train_loss"][0] == pytest.approx(expected, abs=1e-5)


@pytest.mark.slow
def test_explicit_empty_hidden_configuration_is_not_replaced(autoencoder_real_module):
    m = autoencoder_real_module
    # Empty hidden list defines a linear AE; explicit rejection also admissible.
    try:
        ae = m.SpectralAutoencoder(2, 1, hidden_dims=[], device="cpu")
        ae.fit(np.ones((8, 2)), epochs=1, batch_size=20, verbose=False)
    except ValueError:
        return
    assert (
        sum(isinstance(layer, m.nn.Linear) for layer in ae._model.encoder.modules())
        == 1
    )


@pytest.mark.slow
@pytest.mark.parametrize(
    "N,fraction", [(0, 0.1), (1, 0.1), (8, 1.0), (8, 2.0), (8, -0.5)]
)
def test_observe_split_boundaries(autoencoder_real_module, N, fraction):
    m = autoencoder_real_module
    ae = m.SpectralAutoencoder(2, 1, [2], device="cpu")
    try:
        history = ae.fit(
            np.ones((N, 2)),
            epochs=1,
            batch_size=20,
            val_fraction=fraction,
            verbose=False,
        )
    except (ValueError, RuntimeError) as error:
        print(type(error).__name__, str(error))
    else:
        print(history)


@pytest.mark.slow
@pytest.mark.parametrize("epochs", [0, -1])
def test_observe_zero_negative_epochs(autoencoder_real_module, epochs):
    m = autoencoder_real_module
    ae = m.SpectralAutoencoder(2, 1, [2], device="cpu")
    print(ae.fit(np.ones((8, 2)), epochs=epochs, batch_size=20, verbose=False))
    print(ae.encode(np.ones((1, 2))))


@pytest.mark.parametrize("steps", [0, -1])
def test_observe_interpolation_empty_or_invalid_steps(
    autoencoder_numeric_module, steps
):
    ae = autoencoder_numeric_module.SpectralAutoencoder.__new__(
        autoencoder_numeric_module.SpectralAutoencoder
    )
    ae.encode = np.asarray
    ae.decode = np.asarray
    try:
        print(ae.latent_interpolation(np.ones(2), np.zeros(2), steps))
    except ValueError as error:
        print(type(error).__name__, str(error))


@pytest.mark.slow
def test_observe_duplicate_objects_cross_validation_boundary(
    autoencoder_real_module, monkeypatch
):
    m = autoencoder_real_module
    seen = []
    original = m.DataLoader

    def capture(dataset, **kwargs):
        seen.append(dataset.tensors[0].numpy().copy())
        return original(dataset, **kwargs)

    monkeypatch.setattr(m, "DataLoader", capture)
    X = np.repeat(np.arange(10.0), 2).reshape(20, 1)
    ae = m.SpectralAutoencoder(1, 1, [2], device="cpu")
    ae.fit(X, epochs=1, batch_size=40, val_fraction=0.25, verbose=False)
    val_idx = np.random.default_rng(42).choice(20, 5, replace=False)
    overlap = set(X[val_idx, 0]) & set(seen[0][:, 0])
    print("Synthetic repeated object IDs crossing train/validation:", sorted(overlap))
    assert overlap  # observation of absent grouping API, not validated policy


@pytest.mark.parametrize(
    "kind",
    ["nan_label", "wrong_label_length", "empty_summary", "interpolation_multiple_rows"],
)
def test_observe_numeric_boundary_policies(autoencoder_numeric_module, kind):
    ae = autoencoder_numeric_module.SpectralAutoencoder.__new__(
        autoencoder_numeric_module.SpectralAutoencoder
    )
    ae.reconstruct = np.zeros_like
    ae.encode = np.asarray
    ae.decode = np.asarray
    try:
        if kind == "nan_label":
            print(
                ae.reconstruction_summary(np.ones((3, 2)), np.array([1.0, np.nan, 1.0]))
            )
        elif kind == "wrong_label_length":
            print(ae.reconstruction_summary(np.ones((3, 2)), np.array([1.0, 2.0])))
        elif kind == "empty_summary":
            print(ae.reconstruction_summary(np.empty((0, 2))))
        else:
            print(ae.latent_interpolation(np.ones((2, 2)), np.zeros((2, 2)), 3))
    except (IndexError, ValueError, RuntimeError) as error:
        print(type(error).__name__, str(error))
