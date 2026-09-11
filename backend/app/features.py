"""Feature schema, completeness score, and 0–1 min-max scaling."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

FEATURE_NAMES: list[str] = [
    "follower_count",
    "following_count",
    "friends_count",
    "posts_count",
    "account_age_days",
    "has_profile_photo",
    "has_cover_photo",
    "bio_length",
    "username_digit_ratio",
    "username_length",
    "friend_request_mutuals",
    "profile_completeness",
    "avg_likes_per_post",
    "posts_per_week",
]

FEATURE_LABELS: dict[str, str] = {
    "follower_count": "Followers",
    "following_count": "Following",
    "friends_count": "Friends",
    "posts_count": "Posts",
    "account_age_days": "Account age (days)",
    "has_profile_photo": "Profile photo",
    "has_cover_photo": "Cover photo",
    "bio_length": "Bio length",
    "username_digit_ratio": "Username digit ratio",
    "username_length": "Username length",
    "friend_request_mutuals": "Mutual connections",
    "profile_completeness": "Profile completeness",
    "avg_likes_per_post": "Avg. likes per post",
    "posts_per_week": "Posts per week",
}

# Sensible UI / clip bounds (scaler mins/maxes come from training data).
FEATURE_BOUNDS: dict[str, tuple[float, float]] = {
    "follower_count": (0, 80_000),
    "following_count": (0, 12_000),
    "friends_count": (0, 6_000),
    "posts_count": (0, 8_000),
    "account_age_days": (1, 5_200),
    "has_profile_photo": (0, 1),
    "has_cover_photo": (0, 1),
    "bio_length": (0, 320),
    "username_digit_ratio": (0, 1),
    "username_length": (1, 32),
    "friend_request_mutuals": (0, 250),
    "profile_completeness": (0, 1),
    "avg_likes_per_post": (0, 8_000),
    "posts_per_week": (0, 40),
}


def username_digit_ratio(username: str) -> float:
    if not username:
        return 0.0
    digits = sum(ch.isdigit() for ch in username)
    return digits / len(username)


def compute_profile_completeness(
    has_profile_photo: float,
    has_cover_photo: float,
    bio_length: float,
    posts_count: float,
    username_length: float,
    account_age_days: float,
) -> float:
    """Aggregate 0–1 score used both at data gen and inference."""
    parts = [
        float(has_profile_photo),
        float(has_cover_photo),
        min(max(bio_length, 0.0) / 80.0, 1.0),
        min(max(posts_count, 0.0) / 40.0, 1.0),
        1.0 if 4 <= username_length <= 18 else 0.35,
        min(max(account_age_days, 0.0) / 365.0, 1.0),
    ]
    return float(np.mean(parts))


def vector_from_mapping(payload: dict[str, Any]) -> np.ndarray:
    values = []
    for name in FEATURE_NAMES:
        raw = payload.get(name, 0)
        try:
            values.append(float(raw))
        except (TypeError, ValueError):
            values.append(0.0)
    return np.asarray(values, dtype=np.float64)


def fit_scaler(matrix: np.ndarray) -> dict[str, Any]:
    mins = matrix.min(axis=0)
    maxs = matrix.max(axis=0)
    return {
        "features": FEATURE_NAMES,
        "min": [float(v) for v in mins],
        "max": [float(v) for v in maxs],
    }


def normalize(vector: np.ndarray, scaler: dict[str, Any]) -> np.ndarray:
    lo = np.asarray(scaler["min"], dtype=np.float64)
    hi = np.asarray(scaler["max"], dtype=np.float64)
    span = np.where(hi - lo == 0.0, 1.0, hi - lo)
    scaled = (np.asarray(vector, dtype=np.float64).reshape(-1) - lo) / span
    return np.clip(scaled, 0.0, 1.0)


def normalize_matrix(matrix: np.ndarray, scaler: dict[str, Any]) -> np.ndarray:
    return np.vstack([normalize(row, scaler) for row in matrix])


def load_scaler(path: str | Path) -> dict[str, Any]:
    import json

    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)
