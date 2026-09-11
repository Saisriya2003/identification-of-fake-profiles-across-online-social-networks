from __future__ import annotations

import numpy as np

from app.features import (
    FEATURE_NAMES,
    compute_profile_completeness,
    fit_scaler,
    normalize,
    normalize_matrix,
    username_digit_ratio,
    vector_from_mapping,
)


def test_feature_schema_is_stable() -> None:
    assert len(FEATURE_NAMES) == 14
    assert FEATURE_NAMES[0] == "follower_count"
    assert "profile_completeness" in FEATURE_NAMES


def test_username_digit_ratio() -> None:
    assert username_digit_ratio("") == 0.0
    assert username_digit_ratio("maya.nair") == 0.0
    assert username_digit_ratio("user1234") == 0.5
    assert username_digit_ratio("12345") == 1.0


def test_completeness_is_bounded_and_monotone() -> None:
    thin = compute_profile_completeness(0, 0, 0, 0, 2, 3)
    full = compute_profile_completeness(1, 1, 120, 200, 9, 900)
    assert 0.0 <= thin <= 1.0
    assert 0.0 <= full <= 1.0
    assert full > thin
    assert full == 1.0


def test_normalize_uses_training_bounds_and_clips() -> None:
    matrix = np.array([[0.0, 10.0], [100.0, 10.0], [50.0, 10.0]])
    scaler = {"features": ["a", "b"], "min": [0.0, 10.0], "max": [100.0, 10.0]}
    fitted = fit_scaler(matrix)
    assert fitted["min"] == scaler["min"] and fitted["max"] == scaler["max"]

    scaled = normalize(np.array([50.0, 10.0]), scaler)
    assert scaled.tolist() == [0.5, 0.0]  # zero-span feature maps to 0, no division error

    clipped = normalize(np.array([-40.0, 999.0]), scaler)
    assert clipped.min() >= 0.0 and clipped.max() <= 1.0

    rows = normalize_matrix(matrix, scaler)
    assert rows.shape == matrix.shape


def test_vector_from_mapping_orders_and_defaults() -> None:
    vec = vector_from_mapping({"follower_count": "12", "posts_per_week": 3.5, "bogus": 1})
    assert vec.shape == (14,)
    assert vec[0] == 12.0
    assert vec[FEATURE_NAMES.index("posts_per_week")] == 3.5
    assert vec[1] == 0.0
    assert vector_from_mapping({"follower_count": "not-a-number"})[0] == 0.0
