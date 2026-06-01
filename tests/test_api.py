import numpy as np
from fastapi.testclient import TestClient

from app.api import app

client = TestClient(app)
DIM = 8


def _unit_vector(axis: int) -> list[float]:
    vector = np.zeros(DIM, dtype=np.float32)
    vector[axis % DIM] = 1.0
    return vector.tolist()


def test_optimize_rejects_empty_chunks() -> None:
    response = client.post("/v1/optimize", json={"chunks": []})

    assert response.status_code == 422


def test_optimize_with_precomputed_embeddings() -> None:
    payload = {
        "chunks": [
            {
                "id": "1",
                "text": "JWT authentication uses bearer tokens.",
                "embedding": _unit_vector(0),
                "score": 0.9,
            },
            {
                "id": "2",
                "text": "JWT authentication uses bearer tokens.",
                "embedding": _unit_vector(0),
                "score": 0.85,
            },
            {
                "id": "3",
                "text": "Annual leave policy allows twenty days.",
                "embedding": _unit_vector(1),
                "score": 0.7,
            },
        ],
        "namespace": "docs",
        "target_k": 2,
        "token_budget": 500,
        "compress": False,
    }

    response = client.post("/v1/optimize", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["stats"]["exact_duplicate_count"] == 1
    assert body["stats"]["output_count"] == 2
    assert "cluster_sizes" not in body["stats"]
    assert len(body["chunks"]) == 2
    assert all("embedding" not in chunk for chunk in body["chunks"])


def test_optimize_without_token_budget_does_not_skip_large_chunks() -> None:
    payload = {
        "chunks": [
            {
                "id": "1",
                "text": "alpha " * 1600,
                "embedding": _unit_vector(0),
                "score": 0.9,
            },
            {
                "id": "2",
                "text": "beta " * 1600,
                "embedding": _unit_vector(1),
                "score": 0.8,
            },
        ],
        "namespace": "docs",
        "target_k": 2,
        "compress": False,
    }

    response = client.post("/v1/optimize", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["stats"]["output_count"] == 2
    assert body["stats"]["budget_skipped_count"] == 0
