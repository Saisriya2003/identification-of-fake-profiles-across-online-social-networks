"""From-scratch multilayer perceptron: sigmoid, BCE, mini-batch backprop."""

from __future__ import annotations

from pathlib import Path

import numpy as np


class NeuralNetwork:
    """Fully connected ANN with sigmoid units on every layer.

    ``layers`` is a list of widths, e.g. ``[14, 16, 8, 1]``. Weights are
    Xavier-normal initialized. Training is mini-batch gradient descent on
    binary cross-entropy — no sklearn/autograd.
    """

    def __init__(
        self,
        layers: list[int],
        learning_rate: float = 0.18,
        seed: int = 2025,
    ) -> None:
        if len(layers) < 2:
            raise ValueError("Need at least an input and output width")
        self.layers = [int(n) for n in layers]
        self.learning_rate = float(learning_rate)
        self.weights: list[np.ndarray] = []
        self.biases: list[np.ndarray] = []
        self._activations: list[np.ndarray] = []
        rng = np.random.default_rng(seed)
        for fan_in, fan_out in zip(self.layers[:-1], self.layers[1:]):
            scale = np.sqrt(1.0 / fan_in)
            self.weights.append(rng.normal(0.0, scale, size=(fan_in, fan_out)))
            self.biases.append(np.zeros((1, fan_out), dtype=np.float64))

    @staticmethod
    def sigmoid(z: np.ndarray) -> np.ndarray:
        z = np.clip(z, -60.0, 60.0)
        return 1.0 / (1.0 + np.exp(-z))

    @staticmethod
    def sigmoid_prime_from_activation(a: np.ndarray) -> np.ndarray:
        return a * (1.0 - a)

    @staticmethod
    def binary_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        eps = 1e-9
        y = y_true.reshape(-1, 1).astype(np.float64)
        p = np.clip(y_pred.reshape(-1, 1), eps, 1.0 - eps)
        return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))

    def forward(self, x: np.ndarray) -> np.ndarray:
        a = np.asarray(x, dtype=np.float64)
        if a.ndim == 1:
            a = a.reshape(1, -1)
        self._activations = [a]
        for weight, bias in zip(self.weights, self.biases):
            z = a @ weight + bias
            a = self.sigmoid(z)
            self._activations.append(a)
        return a

    def _backward(self, y: np.ndarray) -> None:
        y = y.reshape(-1, 1).astype(np.float64)
        batch = max(y.shape[0], 1)
        pred = self._activations[-1]
        # dL/dZ for mean BCE + sigmoid: (a - y) / batch
        delta = (pred - y) / batch
        for i in range(len(self.weights) - 1, -1, -1):
            prev = self._activations[i]
            grad_w = prev.T @ delta
            grad_b = np.sum(delta, axis=0, keepdims=True)
            if i > 0:
                delta = (delta @ self.weights[i].T) * self.sigmoid_prime_from_activation(
                    self._activations[i]
                )
            self.weights[i] -= self.learning_rate * grad_w
            self.biases[i] -= self.learning_rate * grad_b

    def train(
        self,
        x: np.ndarray,
        y: np.ndarray,
        epochs: int = 180,
        batch_size: int = 64,
        seed: int = 2025,
        verbose: bool = True,
    ) -> list[float]:
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).reshape(-1, 1)
        n = x.shape[0]
        rng = np.random.default_rng(seed)
        history: list[float] = []
        for epoch in range(1, epochs + 1):
            order = rng.permutation(n)
            xs, ys = x[order], y[order]
            for start in range(0, n, batch_size):
                xb = xs[start : start + batch_size]
                yb = ys[start : start + batch_size]
                self.forward(xb)
                self._backward(yb)
            pred = self.forward(x)
            loss = self.binary_cross_entropy(y, pred)
            history.append(loss)
            if verbose and (epoch == 1 or epoch % 20 == 0 or epoch == epochs):
                print(f"epoch {epoch:3d}/{epochs}  loss={loss:.5f}")
        return history

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x).reshape(-1)

    def predict(self, x: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(x) >= threshold).astype(int)

    def effective_input_weights(self) -> np.ndarray:
        """Collapse W1 @ W2 @ ... so each input has one signed weight."""
        effective = self.weights[0].copy()
        for weight in self.weights[1:]:
            effective = effective @ weight
        return effective.reshape(-1)

    def feature_contributions(self, x_norm: np.ndarray) -> np.ndarray:
        """Input × first-layer path weights (collapsed through later layers)."""
        vec = np.asarray(x_norm, dtype=np.float64).reshape(-1)
        return vec * self.effective_input_weights()

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, np.ndarray] = {
            "layers": np.array(self.layers, dtype=np.int32),
            "learning_rate": np.array([self.learning_rate]),
        }
        for i, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            payload[f"W{i}"] = weight
            payload[f"b{i}"] = bias
        np.savez(path, **payload)

    @classmethod
    def load(cls, path: str | Path) -> NeuralNetwork:
        data = np.load(path)
        layers = [int(n) for n in data["layers"].tolist()]
        lr = float(data["learning_rate"][0]) if "learning_rate" in data.files else 0.18
        net = cls(layers, learning_rate=lr, seed=0)
        depth = len(layers) - 1
        net.weights = [np.array(data[f"W{i}"], dtype=np.float64) for i in range(depth)]
        net.biases = [np.array(data[f"b{i}"], dtype=np.float64) for i in range(depth)]
        return net
