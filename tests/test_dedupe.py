from app.core.dedupe import remove_exact_duplicate_chunks


def test_namespace_scoped_exact_dedupe() -> None:
    chunks = [
        {"id": "1", "text": "Same text", "score": 0.9},
        {"id": "2", "text": "Same text", "score": 0.8},
        {"id": "3", "text": "Different text", "score": 0.7},
    ]

    result = remove_exact_duplicate_chunks(chunks, namespace="team-a")

    assert len(result) == 2
    assert result[0]["id"] == "1"
    assert result[1]["id"] == "3"


def test_different_namespaces_do_not_share_hashes() -> None:
    chunk = {"id": "1", "text": "Shared text", "score": 1.0}

    first = remove_exact_duplicate_chunks([chunk, dict(chunk, id="2")], namespace="a")
    second = remove_exact_duplicate_chunks([chunk, dict(chunk, id="2")], namespace="b")

    assert len(first) == 1
    assert len(second) == 1
