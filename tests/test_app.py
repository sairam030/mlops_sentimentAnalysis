"""
test_app.py — Test FastAPI endpoints using httpx TestClient.
"""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


# ─── Fixtures ────────────────────────────────────────────
@pytest.fixture
def mock_model_loader():
    """Patch model_loader so no real model is needed."""
    with patch("serving.app.model_loader") as mock_loader:
        mock_loader.is_loaded = True
        mock_loader.status.return_value = {
            "loaded": True,
            "model_stage": "staging",
            "model_version": "1",
            "source": "local://test",
            "model_info": {
                "model_type": "DistilBERT",
                "accuracy": 0.855,
                "f1_macro": 0.808,
            },
        }
        mock_loader.predictor = MagicMock()
        mock_loader.predictor.predict.return_value = [
            {
                "text": "This is great!",
                "prediction": "positive",
                "confidence": 0.9912,
                "all_scores": {"positive": 0.9912, "negative": 0.0050, "neutral": 0.0038},
            }
        ]
        yield mock_loader


@pytest.fixture
def client(mock_model_loader):
    """TestClient with mocked model."""
    from serving.app import app

    with TestClient(app) as c:
        yield c


# ─── Root page ───────────────────────────────────────────
def test_root_returns_html(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "Sentiment Analysis" in r.text


# ─── Health endpoint ─────────────────────────────────────
def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert "timestamp" in data


# ─── Predict endpoint ────────────────────────────────────
def test_predict_single(client, mock_model_loader):
    r = client.post("/predict", json={"texts": ["This is great!"]})
    assert r.status_code == 200
    data = r.json()
    assert len(data["predictions"]) == 1
    assert data["predictions"][0]["prediction"] == "positive"
    assert data["predictions"][0]["confidence"] > 0.9
    assert "latency_ms" in data


def test_predict_multiple(client, mock_model_loader):
    texts = ["Good video", "Bad video", "Okay video"]
    mock_model_loader.predictor.predict.return_value = [
        {"text": t, "prediction": "positive", "confidence": 0.9, "all_scores": {"positive": 0.9, "negative": 0.05, "neutral": 0.05}}
        for t in texts
    ]
    r = client.post("/predict", json={"texts": texts})
    assert r.status_code == 200
    assert len(r.json()["predictions"]) == 3


def test_predict_empty_list(client):
    r = client.post("/predict", json={"texts": []})
    assert r.status_code == 422  # validation error


def test_predict_not_loaded(client, mock_model_loader):
    mock_model_loader.is_loaded = False
    r = client.post("/predict", json={"texts": ["hello"]})
    assert r.status_code == 503


def test_predict_get_method_not_allowed(client):
    r = client.get("/predict")
    assert r.status_code == 405


# ─── Metrics endpoint ────────────────────────────────────
def test_metrics(client):
    # Make a prediction first to populate metrics
    client.post("/predict", json={"texts": ["test"]})
    r = client.get("/metrics")
    assert r.status_code == 200
    data = r.json()
    assert "total_requests" in data
    assert "total_predictions" in data
    assert "avg_latency_ms" in data
    assert data["total_requests"] >= 1


# ─── Reload endpoint ─────────────────────────────────────
def test_reload(client, mock_model_loader):
    # load() is called once during startup (lifespan) + once by /reload
    initial_calls = mock_model_loader.load.call_count
    r = client.post("/reload")
    assert r.status_code == 200
    assert r.json()["status"] == "reloaded"
    assert mock_model_loader.load.call_count == initial_calls + 1


# ─── Request validation ──────────────────────────────────
def test_predict_invalid_body(client):
    r = client.post("/predict", json={"wrong_field": "hello"})
    assert r.status_code == 422


def test_predict_no_body(client):
    r = client.post("/predict")
    assert r.status_code == 422
