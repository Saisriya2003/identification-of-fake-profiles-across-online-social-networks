"""API tests against the committed model artifacts (no training required)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app

AUTHENTIC = {
    "username": "maya.nair",
    "platform": "facebook",
    "follower_count": 842,
    "following_count": 361,
    "friends_count": 478,
    "posts_count": 216,
    "account_age_days": 1480,
    "has_profile_photo": 1,
    "has_cover_photo": 1,
    "bio_length": 96,
    "friend_request_mutuals": 18,
    "avg_likes_per_post": 34,
    "posts_per_week": 1.1,
}

SUSPICIOUS = {
    "username": "promo_deal84721",
    "platform": "instagram",
    "follower_count": 38,
    "following_count": 4200,
    "friends_count": 12,
    "posts_count": 3,
    "account_age_days": 11,
    "has_profile_photo": 0,
    "has_cover_photo": 0,
    "bio_length": 8,
    "friend_request_mutuals": 0,
    "avg_likes_per_post": 1,
    "posts_per_week": 7.5,
}


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


def test_health_reports_model_ready(client: TestClient) -> None:
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["service"] == "fake-profile-identification"
    assert body["model_ready"] is True


def test_metrics_expose_training_artifacts(client: TestClient) -> None:
    body = client.get("/api/metrics").json()
    assert body["architecture"] == [14, 16, 8, 1]
    assert body["activation"] == "sigmoid"
    assert body["loss"] == "binary_cross_entropy"
    assert body["accuracy"] > 0.9
    assert set(body["confusion_matrix"]) == {"tn", "fp", "fn", "tp"}
    assert len(body["loss_history"]) == body["epochs"]
    assert body["feature_labels"]["follower_count"] == "Followers"


def test_predict_authentic_profile(client: TestClient) -> None:
    body = client.post("/api/predict", json=AUTHENTIC).json()
    assert body["label"] == "authentic"
    assert body["verdict"] == "Likely authentic"
    assert 0.0 <= body["score"] <= 1.0
    assert body["p_fake"] == pytest.approx(1.0 - body["score"], abs=1e-3)
    assert body["confidence"] >= 0.5
    assert len(body["explanation"]) == 6
    assert {item["direction"] for item in body["explanation"]} <= {"fake", "authentic"}


def test_predict_suspicious_profile(client: TestClient) -> None:
    body = client.post("/api/predict", json=SUSPICIOUS).json()
    assert body["label"] == "fake"
    assert body["verdict"] == "Likely fake"
    assert body["p_fake"] > 0.65


def test_predict_derives_username_stats_and_ignores_extra_keys(client: TestClient) -> None:
    payload = {**SUSPICIOUS, "label": "fake", "unexpected": "ignored"}
    payload.pop("username_length", None)
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 200
    features = {item["feature"]: item for item in response.json()["explanation"]}
    if "username_digit_ratio" in features:
        assert features["username_digit_ratio"]["value"] > 0.0


def test_samples_filter_and_validate(client: TestClient) -> None:
    body = client.get("/api/samples", params={"label": "fake", "limit": 5}).json()
    assert len(body["samples"]) == 5
    assert all(row["label"] == "fake" for row in body["samples"])
    assert all(set(row) >= {"username", "platform", "follower_count"} for row in body["samples"])
    assert client.get("/api/samples", params={"limit": 0}).status_code == 422
    assert client.get("/api/samples", params={"label": "unknown"}).status_code == 422


def test_platform_counts_are_consistent(client: TestClient) -> None:
    body = client.get("/api/platforms").json()
    assert body["total"] == sum(body["by_platform"].values()) == sum(body["by_label"].values())
    assert set(body["by_platform"]) == {"facebook", "instagram", "twitter"}
