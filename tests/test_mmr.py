import numpy as np

from app.core.mmr import enforce_token_budget, select_mmr


def _chunk(chunk_id: str, axis: int, score: float, text: str, dim: int = 8) -> dict:
    embedding = np.zeros(dim, dtype=np.float32)
    embedding[axis % dim] = 1.0
    return {
        "id": chunk_id,
        "text": text,
        "embedding": embedding.tolist(),
        "score": score,
    }


def test_mmr_limits_target_k() -> None:
    chunks = [
        _chunk("1", 0, 0.9, "a"),
        _chunk("2", 0, 0.8, "b"),
        _chunk("3", 1, 0.7, "c"),
    ]

    selected = select_mmr(chunks, target_k=2, mmr_lambda=0.5)

    assert len(selected) == 2


def test_token_budget_skips_oversized_chunks() -> None:
    oversized = {"id": "1", "text": "x" * 200, "score": 1.0}
    small = {"id": "2", "text": "short", "score": 0.9}

    result, skipped = enforce_token_budget(
        [oversized, small],
        token_budget=10,
        token_counter=lambda item: len(item["text"]),
    )

    assert len(result) == 1
    assert result[0]["id"] == "2"
    assert skipped == 1


def test_token_budget_skips_chunk_that_would_exceed_budget() -> None:
    chunks = [
        {"id": "1", "text": "aaaaa", "score": 1.0},
        {"id": "2", "text": "bbbbbbbbb", "score": 0.9},
    ]

    result, skipped = enforce_token_budget(
        chunks,
        token_budget=10,
        token_counter=lambda item: len(item["text"]),
    )

    assert [chunk["id"] for chunk in result] == ["1"]
    assert skipped == 1
