"""Embedding- and lexical-aware near-duplicate removal before topical clustering."""

import numpy as np

from app.core.dedupe import normalize_text
from app.core.intent import intent_key
from app.core.text_similarity import text_similarity, tfidf_similarity_matrix
from app.core.vectors import embedding_matrix, normalize_embeddings

Chunk = dict

EMBEDDING_NEAR_COPY_SIMILARITY = 0.999
TFIDF_PARAPHRASE_SIMILARITY = 0.13
TFIDF_STRONG_PARAPHRASE_SIMILARITY = 0.25
MIN_TOKENS_FOR_TFIDF_ONLY_DEDUP = 12


def remove_semantic_duplicate_chunks(
    chunks: list[Chunk],
    threshold: float = 0.001,
) -> tuple[list[Chunk], int]:
    """Drop lower-scored redundant chunks using embedding and lexical signals."""
    if not chunks:
        return [], 0

    if threshold <= 0:
        return list(chunks), 0

    embeddings = normalize_embeddings(embedding_matrix(chunks))
    threshold = _effective_semantic_threshold(embeddings, threshold)
    texts = [chunk.get("text", "") for chunk in chunks]
    tfidf_matrix = tfidf_similarity_matrix(texts)

    ranked_indexes = sorted(
        range(len(chunks)),
        key=lambda index: chunks[index].get("score", 0.0),
        reverse=True,
    )

    kept_indexes: list[int] = []
    kept_embeddings: list[np.ndarray] = []

    for index in ranked_indexes:
        embedding = embeddings[index]
        is_duplicate = any(
            _is_semantic_duplicate(
                chunks[index],
                chunks[kept_index],
                embedding,
                kept_embeddings[kept_slot],
                tfidf_matrix[index, kept_index],
                threshold,
            )
            for kept_slot, kept_index in enumerate(kept_indexes)
        )

        if is_duplicate:
            continue

        kept_indexes.append(index)
        kept_embeddings.append(embedding)

    pairwise_unique = [chunks[index] for index in sorted(kept_indexes)]
    intent_unique, intent_removed = _collapse_intent_duplicates(pairwise_unique)

    removed_count = len(chunks) - len(intent_unique)
    return intent_unique, removed_count


def _collapse_intent_duplicates(chunks: list[Chunk]) -> tuple[list[Chunk], int]:
    """Keep the highest-scored chunk for each topical intent label."""
    if not chunks:
        return [], 0

    best_by_intent: dict[str, Chunk] = {}
    general_chunks: list[Chunk] = []

    for chunk in sorted(chunks, key=lambda item: item.get("score", 0.0), reverse=True):
        label = intent_key(chunk.get("text", ""))
        if label == "general":
            general_chunks.append(chunk)
            continue

        if label not in best_by_intent:
            best_by_intent[label] = chunk

    kept = list(best_by_intent.values()) + general_chunks
    kept.sort(key=lambda chunk: chunk.get("score", 0.0), reverse=True)
    return kept, len(chunks) - len(kept)


def _effective_semantic_threshold(embeddings: np.ndarray, threshold: float) -> float:
    """Tighten threshold when vectors sit in a collapsed cone (mock / low-dim embeddings)."""
    if embeddings.shape[0] < 2:
        return threshold

    distances: list[float] = []
    for left in range(embeddings.shape[0]):
        for right in range(left + 1, embeddings.shape[0]):
            distances.append(1.0 - float(embeddings[left] @ embeddings[right]))

    median_distance = float(np.median(distances))
    if median_distance < 0.02:
        adaptive = max(0.00005, median_distance / 5.0)
        return min(threshold, adaptive)

    return threshold


def _is_semantic_duplicate(
    candidate: Chunk,
    kept: Chunk,
    candidate_embedding: np.ndarray,
    kept_embedding: np.ndarray,
    tfidf_similarity: float,
    threshold: float,
) -> bool:
    embedding_similarity = float(candidate_embedding @ kept_embedding)
    distance = 1.0 - embedding_similarity
    text_overlap = text_similarity(candidate.get("text", ""), kept.get("text", ""))

    if (
        tfidf_similarity >= TFIDF_STRONG_PARAPHRASE_SIMILARITY
        and _token_count(candidate) >= MIN_TOKENS_FOR_TFIDF_ONLY_DEDUP
        and _token_count(kept) >= MIN_TOKENS_FOR_TFIDF_ONLY_DEDUP
    ):
        return True

    if (
        embedding_similarity >= EMBEDDING_NEAR_COPY_SIMILARITY
        and tfidf_similarity >= TFIDF_PARAPHRASE_SIMILARITY
    ):
        return True

    if distance < threshold and text_overlap >= 0.55:
        return True

    if intent_key(candidate.get("text", "")) == intent_key(kept.get("text", "")):
        intent_label = intent_key(candidate.get("text", ""))
        if intent_label != "general" and (
            tfidf_similarity >= TFIDF_PARAPHRASE_SIMILARITY
            or (embedding_similarity >= EMBEDDING_NEAR_COPY_SIMILARITY and tfidf_similarity >= 0.08)
        ):
            return True

    return False


def _token_count(chunk: Chunk) -> int:
    return len(normalize_text(chunk.get("text", "")).split())
