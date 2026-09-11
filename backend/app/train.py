"""Train the from-scratch ANN and persist weights, scaler, and metrics."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from app.ann import NeuralNetwork
from app.features import FEATURE_NAMES, fit_scaler, normalize_matrix
from app.generate_data import DATA_PATH, generate_and_save

BACKEND_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = BACKEND_ROOT / "models"
WEIGHTS_PATH = MODELS_DIR / "ann_weights.npz"
METRICS_PATH = MODELS_DIR / "metrics.json"
SCALER_PATH = MODELS_DIR / "scaler.json"

ARCHITECTURE = [len(FEATURE_NAMES), 16, 8, 1]


def _stratified_split(
    x: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    seed: int = 2025,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_idx: list[int] = []
    test_idx: list[int] = []
    for label in (0, 1):
        idx = np.flatnonzero(y == label)
        rng.shuffle(idx)
        cut = max(1, int(round(len(idx) * test_size)))
        test_idx.extend(idx[:cut].tolist())
        train_idx.extend(idx[cut:].tolist())
    train_idx_arr = np.array(train_idx)
    test_idx_arr = np.array(test_idx)
    rng.shuffle(train_idx_arr)
    rng.shuffle(test_idx_arr)
    return x[train_idx_arr], x[test_idx_arr], y[train_idx_arr], y[test_idx_arr]


def train(
    epochs: int = 180,
    batch_size: int = 64,
    learning_rate: float = 0.18,
    seed: int = 2025,
) -> dict:
    if not DATA_PATH.exists():
        generate_and_save()

    frame = pd.read_csv(DATA_PATH)
    x = frame[FEATURE_NAMES].to_numpy(dtype=np.float64)
    # Positive class = fake (the class trust & safety wants to catch).
    y = (frame["label"].astype(str) == "fake").to_numpy(dtype=np.int64)

    x_train, x_test, y_train, y_test = _stratified_split(x, y, test_size=0.2, seed=seed)
    scaler = fit_scaler(x_train)
    x_train_n = normalize_matrix(x_train, scaler)
    x_test_n = normalize_matrix(x_test, scaler)

    model = NeuralNetwork(ARCHITECTURE, learning_rate=learning_rate, seed=seed)
    print(f"architecture {ARCHITECTURE}  train={len(y_train)}  test={len(y_test)}")
    loss_history = model.train(
        x_train_n,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        seed=seed,
        verbose=True,
    )

    y_prob = model.predict_proba(x_test_n)
    y_hat = (y_prob >= 0.5).astype(int)

    accuracy = float(accuracy_score(y_test, y_hat))
    precision = float(precision_score(y_test, y_hat, zero_division=0))
    recall = float(recall_score(y_test, y_hat, zero_division=0))
    f1 = float(f1_score(y_test, y_hat, zero_division=0))
    tn, fp, fn, tp = (int(v) for v in confusion_matrix(y_test, y_hat, labels=[0, 1]).ravel())

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}  (fake class)")
    print(f"Recall   : {recall:.4f}  (fake class)")
    print(f"F1       : {f1:.4f}")
    print(f"Confusion: tn={tn} fp={fp} fn={fn} tp={tp}")

    metrics = {
        "architecture": ARCHITECTURE,
        "activation": "sigmoid",
        "loss": "binary_cross_entropy",
        "optimizer": "mini-batch gradient descent",
        "learning_rate": learning_rate,
        "epochs": epochs,
        "batch_size": batch_size,
        "n_features": len(FEATURE_NAMES),
        "features": FEATURE_NAMES,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "positive_class": "fake",
        "accuracy": round(accuracy, 6),
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "loss_history": [round(float(v), 6) for v in loss_history],
    }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.save(WEIGHTS_PATH)
    SCALER_PATH.write_text(json.dumps(scaler, indent=2), encoding="utf-8")
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"saved {WEIGHTS_PATH}")
    print(f"saved {SCALER_PATH}")
    print(f"saved {METRICS_PATH}")
    return metrics


if __name__ == "__main__":
    train()
