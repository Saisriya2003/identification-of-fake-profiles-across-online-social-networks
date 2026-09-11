"""Unit tests for the from-scratch neural network.

The gradient check is the important one: it proves that ``_backward`` computes
the same gradients as a numerical finite-difference estimate, i.e. that the
hand-written backpropagation is correct.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

from app.ann import NeuralNetwork


def _toy_data(n: int = 256, seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, size=(n, 3))
    y = ((x[:, 0] + x[:, 1]) > 1.0).astype(np.float64)
    return x, y


def test_sigmoid_is_bounded_and_centered() -> None:
    z = np.array([-1000.0, -2.0, 0.0, 2.0, 1000.0])
    s = NeuralNetwork.sigmoid(z)
    # Inputs are clipped to ±60, so extremes saturate to exactly 0/1 without NaN.
    assert np.all(np.isfinite(s))
    assert np.all((s >= 0.0) & (s <= 1.0))
    assert 0.0 < s[1] < 0.5 < s[3] < 1.0
    assert s[2] == pytest.approx(0.5)
    assert s[3] == pytest.approx(1.0 - s[1])


def test_forward_output_shape_and_range() -> None:
    net = NeuralNetwork([3, 5, 1], seed=1)
    x, _ = _toy_data(16)
    out = net.forward(x)
    assert out.shape == (16, 1)
    assert np.all((out > 0.0) & (out < 1.0))
    assert net.predict_proba(x[0]).shape == (1,)


def test_backprop_matches_numerical_gradient() -> None:
    """(W_before - W_after) / lr must equal the finite-difference gradient."""
    lr = 1e-3
    net = NeuralNetwork([3, 4, 1], learning_rate=lr, seed=3)
    x, y = _toy_data(8)

    reference = copy.deepcopy(net)
    stepped = copy.deepcopy(net)
    stepped.forward(x)
    stepped._backward(y)  # noqa: SLF001 — testing the private step on purpose

    eps = 1e-6
    checked = 0
    for layer in range(len(reference.weights)):
        analytical = (reference.weights[layer] - stepped.weights[layer]) / lr
        rows, cols = reference.weights[layer].shape
        for i in range(rows):
            for j in range(cols):
                probe = copy.deepcopy(reference)
                probe.weights[layer][i, j] += eps
                loss_plus = probe.binary_cross_entropy(y, probe.forward(x))
                probe.weights[layer][i, j] -= 2 * eps
                loss_minus = probe.binary_cross_entropy(y, probe.forward(x))
                numerical = (loss_plus - loss_minus) / (2 * eps)
                assert analytical[i, j] == pytest.approx(numerical, rel=1e-4, abs=1e-7)
                checked += 1
        # biases too
        analytical_b = (reference.biases[layer] - stepped.biases[layer]) / lr
        for j in range(reference.biases[layer].shape[1]):
            probe = copy.deepcopy(reference)
            probe.biases[layer][0, j] += eps
            loss_plus = probe.binary_cross_entropy(y, probe.forward(x))
            probe.biases[layer][0, j] -= 2 * eps
            loss_minus = probe.binary_cross_entropy(y, probe.forward(x))
            numerical = (loss_plus - loss_minus) / (2 * eps)
            assert analytical_b[0, j] == pytest.approx(numerical, rel=1e-4, abs=1e-7)
            checked += 1
    assert checked == 3 * 4 + 4 + 4 * 1 + 1


def test_training_reduces_loss_and_learns_separable_data() -> None:
    x, y = _toy_data(400)
    net = NeuralNetwork([3, 6, 1], learning_rate=0.6, seed=11)
    history = net.train(x, y, epochs=150, batch_size=32, seed=11, verbose=False)
    assert len(history) == 150
    assert history[-1] < history[0] * 0.5
    accuracy = float(np.mean(net.predict(x) == y.astype(int)))
    assert accuracy > 0.9


def test_save_and_load_roundtrip(tmp_path) -> None:
    x, y = _toy_data(64)
    net = NeuralNetwork([3, 4, 2, 1], learning_rate=0.3, seed=5)
    net.train(x, y, epochs=5, batch_size=16, verbose=False)
    path = tmp_path / "weights.npz"
    net.save(path)

    loaded = NeuralNetwork.load(path)
    assert loaded.layers == [3, 4, 2, 1]
    assert loaded.learning_rate == pytest.approx(0.3)
    np.testing.assert_allclose(loaded.predict_proba(x), net.predict_proba(x))


def test_feature_contributions_have_one_signed_weight_per_input() -> None:
    net = NeuralNetwork([3, 4, 1], seed=2)
    assert net.effective_input_weights().shape == (3,)
    contrib = net.feature_contributions(np.array([0.5, 0.0, 1.0]))
    assert contrib.shape == (3,)
    assert contrib[1] == 0.0  # zero input contributes nothing


def test_rejects_degenerate_architecture() -> None:
    with pytest.raises(ValueError):
        NeuralNetwork([14])
