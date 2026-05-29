import numpy as np

from app.core.clustering import cluster_chunks, select_representatives


def _chunk(chunk_id: str, axis: int, score: float, dim: int = 8) -> dict:
    embedding = np.zeros(dim, dtype=np.float32)
    embedding[axis % dim] = 1.0
    return {
        "id": chunk_id,
        "text": f"text-{chunk_id}",
        "embedding": embedding.tolist(),
        "score": score,
    }


def test_clustering_groups_similar_vectors() -> None:
    chunks = [
        _chunk("1", 0, 0.9),
        _chunk("2", 0, 0.8),
        _chunk("3", 1, 0.7),
    ]

    clusters = cluster_chunks(chunks, dedup_threshold=0.15)

    assert len(clusters) == 2
    assert sorted(len(cluster["chunks"]) for cluster in clusters) == [1, 2]


def test_auto_representative_prefers_score() -> None:
    clusters = [
        {
            "id": 0,
            "chunks": [
                _chunk("low", 0, 0.2),
                _chunk("high", 0, 0.95),
            ],
        }
    ]

    representatives = select_representatives(clusters, representative_strategy="auto")

    assert representatives[0]["id"] == "high"
