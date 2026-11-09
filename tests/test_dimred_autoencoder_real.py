"""Explicit optional CPU suite: pytest -m slow tests/test_dimred_autoencoder_real.py."""

import numpy as np
import pytest

pytestmark = [pytest.mark.slow, pytest.mark.integration]


def test_fresh_process_import_with_real_torch(autoencoder_real_module):
    from pathlib import Path
    import subprocess
    import sys

    root = Path(__file__).resolve().parents[1]
    code = """
import importlib.util, pathlib, sys, torch
root = pathlib.Path(sys.argv[1])
spec = importlib.util.spec_from_file_location('isolated_ae', root / 'src/dimred/autoencoder.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
assert module._TORCH_AVAILABLE
assert str(module.SpectralAutoencoder(2, device='cpu').device) == 'cpu'
sys.path.insert(0, str(root / 'src'))
from dimred import SpectralAutoencoder, PCAAnalyzer
assert SpectralAutoencoder is not None and PCAAnalyzer is not None
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", code, str(root)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "input_dim,hidden,latent", [(3, [4, 2], 1), (1, [2], 3), (2, [], 1)]
)
def test_architecture_parameter_count_and_gradients(
    autoencoder_real_module, input_dim, hidden, latent
):
    m = autoencoder_real_module
    net = m._AutoencoderNet(input_dim, latent, hidden, dropout=0.2)

    def count_path(dimensions):
        return sum(a * b + b for a, b in zip(dimensions, dimensions[1:])) + 2 * sum(
            dimensions[1:-1]
        )

    expected = count_path([input_dim, *hidden, latent]) + count_path(
        [latent, *hidden[::-1], input_dim]
    )
    assert sum(p.numel() for p in net.parameters()) == expected
    for block, sizes in [
        (net.encoder, [input_dim, *hidden, latent]),
        (net.decoder, [latent, *hidden[::-1], input_dim]),
    ]:
        linear = [layer for layer in block.modules() if isinstance(layer, m.nn.Linear)]
        assert [(layer.in_features, layer.out_features) for layer in linear] == list(
            zip(sizes, sizes[1:])
        )
        assert isinstance(block.net[-1], m.nn.Linear)
        assert sum(
            isinstance(layer, m.nn.BatchNorm1d) for layer in block.modules()
        ) == len(hidden)
        assert all(
            layer.p == 0.2
            for layer in block.modules()
            if isinstance(layer, m.nn.Dropout)
        )
    X = m.torch.arange(4 * input_dim, dtype=m.torch.float32).reshape(4, input_dim) / 7
    out, z = net(X)
    assert out.shape == X.shape
    assert z.shape == (4, latent)
    out.square().sum().backward()
    assert all(
        p.grad is not None and m.torch.isfinite(p.grad).all() for p in net.parameters()
    )
    # Analytic final-bias gradient: d(sum(output))/d(bias) = batch size.
    net.eval()
    parameter = net.decoder.net[-1].bias
    net.zero_grad()
    net(X)[0].sum().backward()
    assert m.torch.allclose(parameter.grad, m.torch.full_like(parameter, 4.0))


def test_hidden_nonlinearity_has_manual_batchnorm_relu_oracle(autoencoder_real_module):
    m = autoencoder_real_module
    net = m._AutoencoderNet(2, 1, [2], dropout=0.5)
    # Configure by numerical role; the oracle does not assert container order.
    affine = [
        layer for layer in net.encoder.modules() if isinstance(layer, m.nn.Linear)
    ]
    bn = next(
        layer for layer in net.encoder.modules() if isinstance(layer, m.nn.BatchNorm1d)
    )
    with m.torch.no_grad():
        affine[0].weight.copy_(m.torch.tensor([[1.0, -2.0], [3.0, 1.0]]))
        affine[0].bias.copy_(m.torch.tensor([0.5, -1.0]))
        bn.running_mean.copy_(m.torch.tensor([1.0, -1.0]))
        bn.running_var.fill_(4.0 - bn.eps)
        bn.weight.copy_(m.torch.tensor([2.0, 0.5]))
        bn.bias.copy_(m.torch.tensor([0.0, 1.0]))
        affine[1].weight.copy_(m.torch.tensor([[1.0, -1.0]]))
        affine[1].bias.fill_(0.25)
    net.eval()
    X = m.torch.tensor([[1.0, 2.0], [-1.0, 0.0]])
    # Affine: [-2.5,4],[-.5,-4]. BN denominator sqrt(var+eps)=2:
    # [-3.5,2.25],[-1.5,.25]. ReLU gives [0,2.25],[0,.25].
    # Final dot with [1,-1], plus .25, gives -2 and 0.
    expected = np.array([[-2.0], [0.0]])
    for _ in range(2):  # Dropout must be inactive in eval mode.
        actual = net.encode(X).detach().numpy()
        np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-6)


def analytic_model(m):
    ae = m.SpectralAutoencoder(2, latent_dim=1, hidden_dims=[1], device="cpu")
    # Deliberately linear test network: independently calculable, not training.
    ae._model = m._AutoencoderNet(2, 1, [], dropout=0.0)
    with m.torch.no_grad():
        ae._model.encoder.net[0].weight.copy_(m.torch.tensor([[2.0, -1.0]]))
        ae._model.encoder.net[0].bias.copy_(m.torch.tensor([0.5]))
        ae._model.decoder.net[0].weight.copy_(m.torch.tensor([[3.0], [-2.0]]))
        ae._model.decoder.net[0].bias.copy_(m.torch.tensor([1.0, 4.0]))
    return ae


def test_inference_values_no_grad_eval_cpu_and_input_preservation(
    autoencoder_real_module,
):
    m = autoencoder_real_module
    ae = analytic_model(m)
    X = np.array([[1.0, 3.0], [-2.0, 4.0], [5.0, -1.0]])
    original = X.copy()
    observed = []
    handle = ae._model.encoder.register_forward_hook(
        lambda net, args, out: observed.append(
            (net.training, m.torch.is_grad_enabled(), out.requires_grad)
        )
    )
    ae._model.train()
    Z = ae.encode(X)
    expected_z = np.array([[-0.5], [-7.5], [11.5]])
    expected_x = np.array([[-0.5, 5.0], [-21.5, 19.0], [35.5, -19.0]])
    np.testing.assert_array_equal(Z, expected_z)
    np.testing.assert_array_equal(ae.decode(Z), expected_x)
    np.testing.assert_array_equal(ae.reconstruct(X), expected_x)
    np.testing.assert_array_equal(ae.encode(X), Z)
    np.testing.assert_array_equal(X, original)
    assert all(value == (False, False, False) for value in observed)
    assert isinstance(Z, np.ndarray) and Z.dtype == np.float32
    handle.remove()


def test_real_dropout_and_batchnorm_inference_stability(autoencoder_real_module):
    m = autoencoder_real_module
    ae = m.SpectralAutoencoder(3, 2, [4], dropout=0.8, device="cpu")
    ae._model = m._AutoencoderNet(3, 2, [4], dropout=0.8)
    X = np.arange(18.0, dtype=np.float32).reshape(6, 3)
    ae._model.train()
    ae._model(m.torch.tensor(X))  # update BN buffers once
    buffers = {k: v.clone() for k, v in ae._model.named_buffers()}
    first = ae.encode(X)
    second = ae.encode(X)
    np.testing.assert_array_equal(first, second)
    for key, value in ae._model.named_buffers():
        assert m.torch.equal(value, buffers[key])
    np.testing.assert_array_equal(ae.reconstruct(X), ae.decode(first))
    assert all(not part.training for part in ae._model.modules())


@pytest.mark.parametrize("fraction", [0.1, 0.25, 0.75])
def test_split_disjoint_complete_fixed_and_loader_targets(
    autoencoder_real_module, monkeypatch, fraction
):
    m = autoencoder_real_module
    captures = []
    real_loader = m.DataLoader

    def loader(dataset, **kwargs):
        captures.append((dataset, kwargs))
        return real_loader(dataset, **kwargs)

    monkeypatch.setattr(m, "DataLoader", loader)
    X = np.column_stack([np.arange(20.0), np.arange(20.0) ** 2])
    for seed in [3, 19]:
        m.torch.manual_seed(seed)
        ae = m.SpectralAutoencoder(2, 1, [2], dropout=0.0, device="cpu")
        validation = []
        original = m._AutoencoderNet.forward

        def forward(net, x):
            if not net.training:
                validation.append(x.detach().numpy().copy())
            return original(net, x)

        with monkeypatch.context() as ctx:
            ctx.setattr(m._AutoencoderNet, "forward", forward)
            ae.fit(X, epochs=1, batch_size=40, val_fraction=fraction, verbose=False)
        dataset, options = captures[-1]
        train, target = [t.numpy() for t in dataset.tensors]
        val = validation[0]
        assert len(val) == max(1, int(20 * fraction))
        expected_indices = np.random.default_rng(42).choice(
            20, max(1, int(20 * fraction)), replace=False
        )
        np.testing.assert_array_equal(val, X[expected_indices].astype(np.float32))
        assert len(train) + len(val) == len(X)
        assert set(train[:, 0]).isdisjoint(val[:, 0])
        assert set(train[:, 0]) | set(val[:, 0]) == set(range(20))
        np.testing.assert_array_equal(train, target)
        assert options["shuffle"] is True
        assert options["batch_size"] == 40
        if seed == 3:
            previous = val.copy()
        else:
            np.testing.assert_array_equal(val, previous)


def controlled_network(m, monkeypatch):
    class Network(m.nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.level = m.nn.Parameter(m.torch.tensor(0.0))

        def forward(self, X):
            return m.torch.ones_like(X) * self.level, X[:, :1]

        def encode(self, X):
            return X[:, :1] * self.level

        def decode(self, Z):
            return Z * self.level

    monkeypatch.setattr(m, "_AutoencoderNet", Network)
    return Network


@pytest.mark.parametrize("patience,expected_epochs", [(2, 4), (0, 6), (1, 3)])
def test_early_stopping_restores_actual_best_weights(
    autoencoder_real_module, monkeypatch, patience, expected_epochs
):
    m = autoencoder_real_module
    controlled_network(m, monkeypatch)
    validation_losses = iter([5.0, 2.0, 3.0, 4.0, 6.0, 7.0])
    actual_criterion = m.nn.MSELoss
    observed = []

    class Criterion(m.nn.Module):
        def forward(self, output, target):
            if m.torch.is_grad_enabled():
                return actual_criterion()(output, target)
            observed.append(float(output[0, 0]))
            return m.torch.tensor(next(validation_losses))

    class Optimizer:
        def __init__(self, parameters, lr, weight_decay):
            self.parameters = list(parameters)
            self.param_groups = [{"lr": lr}]

        def zero_grad(self):
            for p in self.parameters:
                p.grad = None

        def step(self):
            with m.torch.no_grad():
                for p in self.parameters:
                    p.add_(1.0)

    monkeypatch.setattr(m.nn, "MSELoss", Criterion)
    monkeypatch.setattr(m.optim, "Adam", Optimizer)
    ae = m.SpectralAutoencoder(1, 1, [1], device="cpu")
    history = ae.fit(
        np.ones((8, 1)),
        epochs=6,
        batch_size=10,
        val_fraction=0.25,
        lr_scheduler=False,
        early_stopping_patience=patience,
        verbose=False,
    )
    assert len(history["val_loss"]) == expected_epochs
    assert history["val_loss"] == [5.0, 2.0, 3.0, 4.0, 6.0, 7.0][:expected_epochs]
    assert observed == list(np.arange(1.0, expected_epochs + 1))
    assert ae._model.level.item() == 2.0  # epoch 2, not final epoch
    assert history["train_loss"] == [1.0, 0.0, 1.0, 4.0, 9.0, 16.0][:expected_epochs]


def test_real_adam_clipping_gradient_reset_and_scheduler_history(
    autoencoder_real_module, monkeypatch
):
    m = autoencoder_real_module
    controlled_network(m, monkeypatch)
    original_adam, original_clip = m.optim.Adam, m.nn.utils.clip_grad_norm_
    zero_calls, clips, optimizers, scheduler_values = [], [], [], []

    class Adam(original_adam):
        def __init__(self, params, **kwargs):
            super().__init__(params, **kwargs)
            optimizers.append(self)
            assert kwargs == {"lr": 0.1, "weight_decay": 0.02}

        def zero_grad(self, *args, **kwargs):
            zero_calls.append(True)
            return super().zero_grad(*args, **kwargs)

    def clip(params, max_norm):
        params = list(params)
        before = [p.grad.clone() for p in params]
        result = original_clip(params, max_norm)
        clips.append((before, [p.grad.clone() for p in params], max_norm))
        return result

    class Scheduler:
        def __init__(self, optimizer, **kwargs):
            self.optimizer = optimizer
            assert kwargs == dict(mode="min", factor=0.5, patience=10, min_lr=1e-6)

        def step(self, loss):
            scheduler_values.append(loss)
            self.optimizer.param_groups[0]["lr"] *= 0.5

    monkeypatch.setattr(m.optim, "Adam", Adam)
    monkeypatch.setattr(m.nn.utils, "clip_grad_norm_", clip)
    monkeypatch.setattr(m.optim.lr_scheduler, "ReduceLROnPlateau", Scheduler)
    ae = m.SpectralAutoencoder(1, 1, [1], device="cpu")
    history = ae.fit(
        np.full((8, 1), 2.0),
        epochs=3,
        lr=0.1,
        weight_decay=0.02,
        batch_size=10,
        verbose=False,
    )
    assert history["lr_history"] == [0.1, 0.05, 0.025]
    assert scheduler_values == history["val_loss"]
    assert len(zero_calls) == len(clips) == 3
    assert clips[0][0][0].item() == -4.0
    assert all(c[2] == 1.0 and abs(c[1][0].item()) <= 1.0 for c in clips)
    assert all(0 < v < 4 for v in history["val_loss"])


def test_one_real_adam_step_has_analytic_update(autoencoder_real_module, monkeypatch):
    m = autoencoder_real_module
    controlled_network(m, monkeypatch)
    ae = m.SpectralAutoencoder(1, 1, [1], device="cpu")
    history = ae.fit(
        np.full((8, 1), 2.0),
        epochs=1,
        lr=0.1,
        weight_decay=0.0,
        batch_size=10,
        lr_scheduler=False,
        verbose=False,
    )
    # Initial gradient -4 is clipped to -1. Adam's bias-corrected first update
    # is +.1/(1+eps), independent of beta1/beta2 to float32 precision.
    assert ae._model.level.item() == pytest.approx(0.1, abs=2e-7)
    assert history["train_loss"] == [4.0]
    assert history["val_loss"][0] == pytest.approx(1.9**2, abs=1e-6)


@pytest.mark.parametrize(
    "kwargs",
    [dict(dropout=-0.1), dict(dropout=1.1), dict(hidden_dims=[-2]), dict(input_dim=-1)],
)
def test_invalid_architecture_refused_when_built(autoencoder_real_module, kwargs):
    m = autoencoder_real_module
    options = dict(input_dim=2, latent_dim=1, hidden_dims=[2], device="cpu")
    options.update(kwargs)
    with pytest.raises((ValueError, RuntimeError, TypeError)):
        ae = m.SpectralAutoencoder(**options)
        ae.fit(np.ones((8, 2)), epochs=1, batch_size=20, verbose=False)


@pytest.mark.parametrize(
    "operation,value", [("encode", np.ones((2, 3))), ("decode", np.ones((2, 2)))]
)
def test_wrong_inference_shapes_refused(autoencoder_real_module, operation, value):
    ae = analytic_model(autoencoder_real_module)
    with pytest.raises((ValueError, RuntimeError)):
        getattr(ae, operation)(value)


def test_real_scheduler_reduces_on_plateau_and_disabled_is_unused(
    autoencoder_real_module, monkeypatch
):
    m = autoencoder_real_module
    controlled_network(m, monkeypatch)

    class Loss(m.nn.Module):
        def forward(self, output, target):
            return output.sum() * 0 + 1

    monkeypatch.setattr(m.nn, "MSELoss", Loss)
    ae = m.SpectralAutoencoder(1, 1, [1], device="cpu")
    history = ae.fit(
        np.ones((8, 1)),
        epochs=14,
        lr=0.01,
        batch_size=10,
        early_stopping_patience=0,
        verbose=False,
    )
    assert history["lr_history"][:12] == [0.01] * 12
    assert history["lr_history"][12:] == [0.005] * 2

    def forbidden(*args, **kwargs):
        raise AssertionError("Disabled scheduler instantiated")

    monkeypatch.setattr(m.optim.lr_scheduler, "ReduceLROnPlateau", forbidden)
    assert (
        len(
            ae.fit(
                np.ones((8, 1)),
                epochs=2,
                batch_size=10,
                lr_scheduler=False,
                verbose=False,
            )["train_loss"]
        )
        == 2
    )


def test_external_seed_controls_training_and_second_fit_resets(autoencoder_real_module):
    m = autoencoder_real_module
    X = np.random.default_rng(17).normal(size=(20, 3)).astype(np.float32)
    ae = m.SpectralAutoencoder(3, 2, [4], dropout=0.3, device="cpu")
    m.torch.manual_seed(97)
    history1 = ae.fit(X, epochs=3, batch_size=6, val_fraction=0.1, verbose=False)
    output1 = ae.encode(X)
    model1 = ae._model
    m.torch.manual_seed(97)
    history2 = ae.fit(X, epochs=3, batch_size=6, val_fraction=0.1, verbose=False)
    assert ae._model is not model1
    assert history1 == history2
    np.testing.assert_array_equal(output1, ae.encode(X))
    m.torch.manual_seed(98)
    ae.fit(X, epochs=3, batch_size=6, val_fraction=0.1, verbose=False)
    assert not np.array_equal(output1, ae.encode(X))


def test_checkpoint_full_roundtrip_weights_buffers_outputs_and_metadata(
    autoencoder_real_module, tmp_path
):
    m = autoencoder_real_module
    X = np.random.default_rng(9).normal(size=(12, 3)).astype(np.float32)
    ae = m.SpectralAutoencoder(3, 2, [4, 2], dropout=0.2, device="cpu")
    ae.fit(X, epochs=2, batch_size=20, verbose=False)
    path = tmp_path / "trusted.pt"
    ae.save(str(path))
    loaded = m.SpectralAutoencoder.load(str(path), device="cpu")
    assert (
        loaded.input_dim,
        loaded.latent_dim,
        loaded.hidden_dims,
        loaded.dropout,
    ) == (3, 2, [4, 2], 0.2)
    assert loaded.history_ == ae.history_
    assert loaded.fit_time_ == ae.fit_time_
    for name, weight in ae._model.state_dict().items():
        assert m.torch.equal(loaded._model.state_dict()[name], weight)
    np.testing.assert_array_equal(loaded.encode(X), ae.encode(X))
    np.testing.assert_array_equal(loaded.reconstruct(X), ae.reconstruct(X))
    assert not loaded._model.training
    assert str(loaded.device) == "cpu"


@pytest.mark.parametrize(
    "kind", ["missing", "corrupt", "incomplete", "wrong_weights", "wrong_dimensions"]
)
def test_checkpoint_invalid_local_payload_refused(
    autoencoder_real_module, tmp_path, kind
):
    m = autoencoder_real_module
    path = tmp_path / "controlled.pt"
    if kind == "corrupt":
        path.write_bytes(b"not a checkpoint")
    elif kind != "missing":
        ae = m.SpectralAutoencoder(2, 1, [2], device="cpu")
        ae._model = m._AutoencoderNet(2, 1, [2])
        ae.save(str(path))
        payload = m.torch.load(path, weights_only=False)
        if kind == "incomplete":
            del payload["input_dim"]
        elif kind == "wrong_dimensions":
            payload["input_dim"] = 3
        else:
            payload["model_state_dict"] = {}
        m.torch.save(payload, path)
    # Backend-specific diagnostic classes are not fixed as public policy.
    import pickle

    with pytest.raises(
        (FileNotFoundError, RuntimeError, ValueError, KeyError, pickle.UnpicklingError)
    ):
        m.SpectralAutoencoder.load(str(path), device="cpu")


def test_checkpoint_optional_metadata_and_extra_fields(
    autoencoder_real_module, tmp_path
):
    m = autoencoder_real_module
    ae = m.SpectralAutoencoder(2, 1, [2], device="cpu")
    ae._model = m._AutoencoderNet(2, 1, [2])
    path = tmp_path / "trusted.pt"
    ae.save(str(path))
    payload = m.torch.load(path, weights_only=False)
    del payload["history"], payload["fit_time"]
    payload["additional_metadata"] = {"provenance": "synthetic"}
    m.torch.save(payload, path)
    loaded = m.SpectralAutoencoder.load(str(path), device="cpu")
    assert loaded.history_ is None and loaded.fit_time_ is None
    np.testing.assert_array_equal(
        ae.encode(np.ones((2, 2))), loaded.encode(np.ones((2, 2)))
    )
