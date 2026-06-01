"""Lexical similarity helpers shared by dedup, clustering, and MMR."""

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

from app.core.dedupe import normalize_text

Chunk = dict


def text_similarity(left: str, right: str) -> float:
    left_tokens = set(normalize_text(left).split())
    right_tokens = set(normalize_text(right).split())

    if not left_tokens or not right_tokens:
        return 0.0

    union = left_tokens | right_tokens
    if not union:
        return 0.0

    return len(left_tokens & right_tokens) / len(union)


def tfidf_similarity_matrix(texts: list[str]) -> np.ndarray:
    if len(texts) < 2:
        return np.ones((len(texts), len(texts)), dtype=np.float32)

    matrix = TfidfVectorizer(stop_words="english").fit_transform(texts).toarray()
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normalized = matrix / norms
    return (normalized @ normalized.T).astype(np.float32)


def median_embedding_distance(embeddings: np.ndarray) -> float:
    if embeddings.shape[0] < 2:
        return 1.0

    distances: list[float] = []
    for left in range(embeddings.shape[0]):
        for right in range(left + 1, embeddings.shape[0]):
            distances.append(1.0 - float(embeddings[left] @ embeddings[right]))

    return float(np.median(distances))
