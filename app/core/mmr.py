import numpy as np

from app.core.text_similarity import text_similarity
from app.core.vectors import embedding_matrix, normalize_embeddings

Chunk = dict


def select_mmr(
    chunks: list[Chunk],
    target_k: int = 8,
    mmr_lambda: float = 0.5,
    query_embedding: list[float] | np.ndarray | None = None,
    return_trace: bool = False,
) -> list[Chunk] | tuple[list[Chunk], list[dict]]:
    if not chunks or target_k <= 0:
        return ([], []) if return_trace else []

    if not 0.0 <= mmr_lambda <= 1.0:
        raise ValueError("mmr_lambda must be between 0.0 and 1.0.")

    embeddings = normalize_embeddings(embedding_matrix(chunks))
    relevance = _relevance_scores(chunks, embeddings, query_embedding)

    selected_indexes: list[int] = []
    remaining_indexes = set(range(len(chunks)))
    trace = []

    while remaining_indexes and len(selected_indexes) < target_k:
        best_index = -1
        best_mmr_score = -float("inf")
        best_diversity_penalty = 0.0

        for index in remaining_indexes:
            diversity_penalty = _max_similarity_to_selected(
                embeddings[index],
                embeddings,
                selected_indexes,
                chunks[index],
                chunks,
            )
            mmr_score = (
                mmr_lambda * relevance[index]
                - (1.0 - mmr_lambda) * diversity_penalty
            )

            if mmr_score > best_mmr_score:
                best_index = index
                best_mmr_score = mmr_score
                best_diversity_penalty = diversity_penalty

        selected_indexes.append(best_index)
        remaining_indexes.remove(best_index)

        trace.append(
            {
                "chunk_id": chunks[best_index].get("id"),
                "mmr_score": float(best_mmr_score),
                "relevance": float(relevance[best_index]),
                "diversity_penalty": float(best_diversity_penalty),
            }
        )

    selected = [chunks[index] for index in selected_indexes]
    return (selected, trace) if return_trace else selected


def enforce_token_budget(
    chunks: list[Chunk],
    token_budget: int,
    token_counter=None,
) -> tuple[list[Chunk], int]:
    """Pack whole chunks into the budget; skip any chunk that does not fit."""
    if token_budget <= 0:
        return [], len(chunks)

    if not chunks:
        return [], 0

    counter = token_counter or _token_count
    selected: list[Chunk] = []
    used_tokens = 0
    skipped = 0

    for chunk in chunks:
        token_count = counter(chunk)

        if token_count > token_budget:
            skipped += 1
            continue

        if used_tokens + token_count > token_budget:
            skipped += 1
            continue

        selected.append(chunk)
        used_tokens += token_count

    return selected, skipped


def _normalize_scores(scores: np.ndarray) -> np.ndarray:
    if scores.size == 0 or scores.max() == scores.min():
        return scores

    return (scores - scores.min()) / (scores.max() - scores.min())


def _relevance_scores(
    chunks: list[Chunk],
    embeddings: np.ndarray,
    query_embedding: list[float] | np.ndarray | None,
) -> np.ndarray:
    if query_embedding is not None:
        query = np.asarray(query_embedding, dtype=np.float32)

        if query.ndim != 1:
            raise ValueError("query_embedding must be a 1D array-like structure.")

        if query.shape[0] != embeddings.shape[1]:
            raise ValueError("query_embedding dimension must match chunk embeddings.")

        query_norm = np.linalg.norm(query)
        if query_norm == 0:
            return np.zeros(len(chunks), dtype=np.float32)

        query = query / query_norm
        return embeddings @ query

    scores = np.asarray([chunk.get("score", 0.0) for chunk in chunks], dtype=np.float32)
    return _normalize_scores(scores)


def _max_similarity_to_selected(
    embedding: np.ndarray,
    embeddings: np.ndarray,
    selected_indexes: list[int],
    chunk: Chunk,
    chunks: list[Chunk],
) -> float:
    if not selected_indexes:
        return 0.0

    selected_embeddings = embeddings[selected_indexes]
    embedding_similarity = float(np.max(selected_embeddings @ embedding))

    chunk_text = chunk.get("text", "")
    lexical_similarities = [
        text_similarity(chunk_text, chunks[selected_index].get("text", ""))
        for selected_index in selected_indexes
    ]
    lexical_similarity = max(lexical_similarities) if lexical_similarities else 0.0

    return max(embedding_similarity, lexical_similarity)


def _token_count(chunk: Chunk) -> int:
    if "token_count" in chunk:
        return int(chunk["token_count"])

    return len(chunk.get("text", "").split())
