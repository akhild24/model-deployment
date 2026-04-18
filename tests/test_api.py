from __future__ import annotations

import os
import sys

from fastapi.testclient import TestClient

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import main


def test_health_endpoint(monkeypatch):
    monkeypatch.setattr(main, "load_model", lambda: None)
    with TestClient(main.app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_embed_endpoint_batch(monkeypatch):
    monkeypatch.setattr(main, "load_model", lambda: None)
    monkeypatch.setattr(main, "get_embedding", lambda texts: [[0.1, 0.2], [0.3, 0.4]])

    with TestClient(main.app) as client:
        response = client.post("/embed", json={"texts": ["hello", "world"]})

    assert response.status_code == 200
    data = response.json()
    assert data["embeddings"] == [[0.1, 0.2], [0.3, 0.4]]


def test_similarity_endpoint(monkeypatch):
    monkeypatch.setattr(main, "load_model", lambda: None)
    monkeypatch.setattr(main, "get_embedding", lambda text: [[0.5, 0.6]])
    monkeypatch.setattr(main, "cosine_similarity", lambda a, b: 0.75)

    with TestClient(main.app) as client:
        response = client.post(
            "/similarity",
            json={"text1": "FastAPI is fast.", "text2": "FastAPI is a web framework."},
        )

    assert response.status_code == 200
    assert response.json() == {"similarity": 0.75}
