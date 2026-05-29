from app.api import _ensure_embeddings, _has_embedding


class _FakeEmbeddingModel:
    def encode(self, texts: list[str]):
        return [[float(index), 0.0, 0.0, 0.0] for index, _ in enumerate(texts)]


def test_has_embedding_rejects_empty_vector() -> None:
    assert _has_embedding({"embedding": []}) is False
    assert _has_embedding({"embedding": [0.1, 0.2]}) is True
    assert _has_embedding({}) is False


def test_ensure_embeddings_reembeds_all_when_any_missing(monkeypatch) -> None:
    import app.api as api_module

    monkeypatch.setattr(
        api_module,
        "_get_embedding_model",
        lambda _name: _FakeEmbeddingModel(),
    )

    chunks = [
        {"id": "1", "text": "first", "embedding": [9.0, 9.0, 9.0, 9.0]},
        {"id": "2", "text": "second", "embedding": None},
    ]

    _ensure_embeddings(chunks, "test-model")

    assert chunks[0]["embedding"] == [0.0, 0.0, 0.0, 0.0]
    assert chunks[1]["embedding"] == [1.0, 0.0, 0.0, 0.0]


def test_ensure_embeddings_leaves_all_provided_unchanged() -> None:
    chunks = [
        {"id": "1", "text": "first", "embedding": [1.0, 0.0]},
        {"id": "2", "text": "second", "embedding": [0.0, 1.0]},
    ]

    _ensure_embeddings(chunks, "unused-model")

    assert chunks[0]["embedding"] == [1.0, 0.0]
    assert chunks[1]["embedding"] == [0.0, 1.0]
