"""Shared embedding vector helpers."""

import numpy as np

Chunk = dict


def embedding_matrix(chunks: list[Chunk]) -> np.ndarray:
    embeddings = []

    for chunk in chunks:
        embedding = chunk.get("embedding")
        if embedding is None:
            raise ValueError(f"Chunk {chunk.get('id')} is missing an embedding.")
        embeddings.append(embedding)

    matrix = np.asarray(embeddings, dtype=np.float32)

    if matrix.ndim != 2:
        raise ValueError("Chunk embeddings must be a 2D array-like structure.")

    return matrix


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return embeddings / norms


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left_norm = np.linalg.norm(left)
    right_norm = np.linalg.norm(right)

    if left_norm == 0 or right_norm == 0:
        return 0.0

    return float(np.dot(left, right) / (left_norm * right_norm))


def validate_embedding_dimensions(
    chunks: list[Chunk],
    query_embedding: list[float] | np.ndarray | None = None,
) -> None:
    if not chunks:
        return

    matrix = embedding_matrix(chunks)
    expected_dim = matrix.shape[1]

    for chunk in chunks:
        embedding = chunk.get("embedding")
        if embedding is None:
            continue
        if len(embedding) != expected_dim:
            raise ValueError(
                f"Chunk {chunk.get('id')} embedding dimension {len(embedding)} "
                f"does not match expected dimension {expected_dim}."
            )

    if query_embedding is None:
        return

    query = np.asarray(query_embedding, dtype=np.float32)
    if query.ndim != 1:
        raise ValueError("query_embedding must be a 1D array-like structure.")
    if query.shape[0] != expected_dim:
        raise ValueError(
            f"query_embedding dimension {query.shape[0]} "
            f"does not match chunk embedding dimension {expected_dim}."
        )
