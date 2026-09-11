"""Identification of Fake Profiles Across Online Social Networks — API that scores inbound social profiles for authenticity."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field

from app.ann import NeuralNetwork
from app.features import (
    FEATURE_LABELS,
    FEATURE_NAMES,
    compute_profile_completeness,
    normalize,
    username_digit_ratio,
    vector_from_mapping,
)

BACKEND_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = BACKEND_ROOT / "data" / "profiles.csv"
WEIGHTS_PATH = BACKEND_ROOT / "models" / "ann_weights.npz"
METRICS_PATH = BACKEND_ROOT / "models" / "metrics.json"
SCALER_PATH = BACKEND_ROOT / "models" / "scaler.json"

app = FastAPI(
    title="Identification of Fake Profiles Across Online Social Networks",
    description="Fake profile identification across online social networks.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictBody(BaseModel):
    model_config = ConfigDict(extra="ignore")

    follower_count: float = 0
    following_count: float = 0
    friends_count: float = 0
    posts_count: float = 0
    account_age_days: float = 1
    has_profile_photo: float = 0
    has_cover_photo: float = 0
    bio_length: float = 0
    username_digit_ratio: float | None = None
    username_length: float | None = None
    friend_request_mutuals: float = 0
    profile_completeness: float | None = None
    avg_likes_per_post: float = 0
    posts_per_week: float = 0
    username: str | None = None
    platform: str | None = None


class Contribution(BaseModel):
    feature: str
    label: str
    value: float
    contribution: float
    direction: Literal["fake", "authentic"]


class PredictResponse(BaseModel):
    score: float = Field(description="Authenticity in [0, 1]")
    p_fake: float
    label: Literal["authentic", "fake"]
    verdict: Literal["Likely authentic", "Suspicious", "Likely fake"]
    confidence: float
    explanation: list[Contribution]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _artifacts() -> tuple[NeuralNetwork, dict[str, Any], dict[str, Any]]:
    if not (WEIGHTS_PATH.exists() and SCALER_PATH.exists() and METRICS_PATH.exists()):
        raise HTTPException(
            status_code=503,
            detail="Model artifacts missing. Run `python -m app.train` from backend/.",
        )
    return (
        NeuralNetwork.load(WEIGHTS_PATH),
        _load_json(SCALER_PATH),
        _load_json(METRICS_PATH),
    )


def _load_profiles() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise HTTPException(status_code=503, detail="Profile dataset missing.")
    return pd.read_csv(DATA_PATH)


def _verdict(score: float) -> tuple[str, str]:
    if score >= 0.65:
        return "authentic", "Likely authentic"
    if score <= 0.35:
        return "fake", "Likely fake"
    return "fake" if score < 0.5 else "authentic", "Suspicious"


def _prepare_features(body: PredictBody) -> dict[str, float]:
    data = body.model_dump()
    username = data.get("username") or ""
    if body.username_length is None:
        data["username_length"] = float(len(username)) if username else 0.0
    if body.username_digit_ratio is None:
        data["username_digit_ratio"] = username_digit_ratio(username) if username else 0.0
    if body.profile_completeness is None:
        data["profile_completeness"] = compute_profile_completeness(
            float(data["has_profile_photo"]),
            float(data["has_cover_photo"]),
            float(data["bio_length"]),
            float(data["posts_count"]),
            float(data["username_length"]),
            float(data["account_age_days"]),
        )
    return {name: float(data[name]) for name in FEATURE_NAMES}


def _row_to_public(row: pd.Series) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "username": str(row["username"]),
        "platform": str(row["platform"]),
        "label": str(row["label"]),
    }
    for name in FEATURE_NAMES:
        value = row[name]
        payload[name] = float(value) if pd.notna(value) else 0.0
    return payload


@app.get("/api/health")
def health() -> dict[str, Any]:
    ready = WEIGHTS_PATH.exists() and SCALER_PATH.exists()
    return {
        "status": "ok" if ready else "degraded",
        "service": "fake-profile-identification",
        "model_ready": ready,
    }


@app.get("/api/metrics")
def metrics() -> dict[str, Any]:
    _, _, stored = _artifacts()
    stored = dict(stored)
    stored["feature_labels"] = FEATURE_LABELS
    return stored


@app.get("/api/platforms")
def platforms() -> dict[str, Any]:
    frame = _load_profiles()
    counts = {str(k): int(v) for k, v in frame["platform"].value_counts().items()}
    labels = {str(k): int(v) for k, v in frame["label"].value_counts().items()}
    return {
        "by_platform": counts,
        "by_label": labels,
        "total": int(len(frame)),
    }


@app.get("/api/samples")
def samples(
    label: Literal["fake", "real"] | None = Query(default=None),
    limit: int = Query(default=8, ge=1, le=24),
) -> dict[str, Any]:
    frame = _load_profiles()
    if label is not None:
        frame = frame[frame["label"] == label]
    if frame.empty:
        return {"samples": [], "label": label}
    drawn = frame.sample(n=min(limit, len(frame)), random_state=None)
    return {"samples": [_row_to_public(row) for _, row in drawn.iterrows()], "label": label}


@app.post("/api/predict", response_model=PredictResponse)
def predict(body: PredictBody) -> PredictResponse:
    model, scaler, _ = _artifacts()
    raw = _prepare_features(body)
    vector = vector_from_mapping(raw)
    scaled = normalize(vector, scaler)
    p_fake = float(model.predict_proba(scaled.reshape(1, -1))[0])
    score = float(max(0.0, min(1.0, 1.0 - p_fake)))
    label, verdict = _verdict(score)
    confidence = float(max(score, 1.0 - score))

    contrib = model.feature_contributions(scaled)
    # Model predicts P(fake); positive contribution pushes toward fake.
    ranked = sorted(
        zip(FEATURE_NAMES, vector.tolist(), contrib.tolist()),
        key=lambda item: abs(item[2]),
        reverse=True,
    )
    explanation = []
    for name, value, weight in ranked[:6]:
        explanation.append(
            Contribution(
                feature=name,
                label=FEATURE_LABELS[name],
                value=round(float(value), 4),
                contribution=round(float(weight), 6),
                direction="fake" if weight > 0 else "authentic",
            )
        )

    return PredictResponse(
        score=round(score, 4),
        p_fake=round(p_fake, 4),
        label=label,
        verdict=verdict,
        confidence=round(confidence, 4),
        explanation=explanation,
    )
