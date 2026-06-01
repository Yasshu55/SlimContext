import numpy as np

from app.core.semantic_dedup import remove_semantic_duplicate_chunks


def _chunk(chunk_id: str, axis: int, score: float, dim: int = 8) -> dict:
    embedding = np.zeros(dim, dtype=np.float32)
    embedding[axis % dim] = 1.0
    return {
        "id": chunk_id,
        "text": f"text-{chunk_id}",
        "embedding": embedding.tolist(),
        "score": score,
    }


def test_semantic_dedup_removes_near_identical_vectors() -> None:
    duplicate = _chunk("high", 0, 0.95)
    near_duplicate = {
        **duplicate,
        "id": "near",
        "score": 0.8,
    }
    chunks = [
        duplicate,
        near_duplicate,
        _chunk("other", 1, 0.7),
    ]

    unique, removed = remove_semantic_duplicate_chunks(chunks, threshold=0.15)

    assert removed == 1
    assert {chunk["id"] for chunk in unique} == {"high", "other"}
