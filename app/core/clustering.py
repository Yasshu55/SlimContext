from collections import defaultdict

from sklearn.cluster import AgglomerativeClustering
import numpy as np

from app.core.intent import intent_key
from app.core.text_similarity import median_embedding_distance, text_similarity, tfidf_similarity_matrix
from app.core.vectors import (
    cosine_similarity,
    embedding_matrix,
    normalize_embeddings,
)

Chunk = dict
Cluster = dict


def cluster_chunks(
    chunks: list[Chunk],
    dedup_threshold: float = 0.15,
) -> list[Cluster]:
    if not chunks:
        return []

    if len(chunks) == 1:
        return [{"id": 0, "chunks": chunks}]

    if dedup_threshold <= 0:
        raise ValueError("dedup_threshold must be greater than 0.")

    intent_clusters = _cluster_by_intent(chunks)
    if intent_clusters is not None:
        return intent_clusters

    return _cluster_by_hybrid_distance(chunks, dedup_threshold)


def _cluster_by_intent(chunks: list[Chunk]) -> list[Cluster] | None:
    grouped: dict[str, list[Chunk]] = defaultdict(list)

    for chunk in chunks:
        grouped[intent_key(chunk.get("text", ""))].append(chunk)

    topical_labels = [label for label in grouped if label != "general"]
    if len(topical_labels) < 2:
        return None

    clusters = [
        {"id": cluster_id, "chunks": grouped[label]}
        for cluster_id, label in enumerate(sorted(grouped))
    ]
    return clusters


def _cluster_by_hybrid_distance(chunks: list[Chunk], dedup_threshold: float) -> list[Cluster]:
    embeddings = normalize_embeddings(embedding_matrix(chunks))
    texts = [chunk.get("text", "") for chunk in chunks]
    tfidf_matrix = tfidf_similarity_matrix(texts)
    distance_matrix = _hybrid_distance_matrix(embeddings, tfidf_matrix, texts)

    model = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=dedup_threshold,
        metric="precomputed",
        linkage="complete",
    )
    labels = model.fit_predict(distance_matrix)

    clusters_by_label: dict[int, list[Chunk]] = {}
    for label, chunk in zip(labels, chunks):
        clusters_by_label.setdefault(int(label), []).append(chunk)

    clusters = [
        {"id": cluster_id, "chunks": cluster_chunks}
        for cluster_id, cluster_chunks in clusters_by_label.items()
    ]

    clusters.sort(key=lambda cluster: cluster["id"])
    return clusters


def _hybrid_distance_matrix(
    embeddings: np.ndarray,
    tfidf_matrix: np.ndarray,
    texts: list[str],
) -> np.ndarray:
    count = len(texts)
    matrix = np.zeros((count, count), dtype=np.float32)
    collapsed = median_embedding_distance(embeddings) < 0.02
    text_weight = 0.85 if collapsed else 0.45

    for left in range(count):
        for right in range(left + 1, count):
            embedding_distance = 1.0 - float(embeddings[left] @ embeddings[right])
            lexical_distance = 1.0 - max(
                float(tfidf_matrix[left, right]),
                text_similarity(texts[left], texts[right]),
            )
            distance = (
                (1.0 - text_weight) * embedding_distance + text_weight * lexical_distance
            )
            matrix[left, right] = distance
            matrix[right, left] = distance

    return matrix


def select_representatives(
    clusters: list[Cluster],
    representative_strategy: str = "auto",
    query_embedding: list[float] | np.ndarray | None = None,
) -> list[Chunk]:
    representatives = []

    for cluster in clusters:
        cluster_chunks = cluster["chunks"]

        if representative_strategy == "auto":
            if any(chunk.get("score", 0.0) > 0 for chunk in cluster_chunks):
                representative = _select_score_representative(cluster_chunks)
            else:
                representative = _select_centroid_representative(cluster_chunks)
        elif representative_strategy == "score":
            representative = _select_score_representative(cluster_chunks)
        elif representative_strategy == "centroid":
            representative = _select_centroid_representative(cluster_chunks)
        elif representative_strategy == "query_closest":
            representative = _select_query_closest_representative(
                cluster_chunks,
                query_embedding,
            )
        elif representative_strategy == "longest":
            representative = _select_longest_representative(cluster_chunks)
        else:
            raise ValueError(
                "representative_strategy must be one of: "
                "auto, score, centroid, query_closest, longest"
            )

        representatives.append(representative)

    return representatives


def select_cluster_candidates(
    clusters: list[Cluster],
    max_per_cluster: int = 2,
    representative_strategy: str = "auto",
    query_embedding: list[float] | np.ndarray | None = None,
) -> list[Chunk]:
    """Return up to ``max_per_cluster`` chunks per cluster for downstream MMR."""
    if max_per_cluster <= 0:
        raise ValueError("max_per_cluster must be greater than 0.")

    candidates: list[Chunk] = []

    for cluster in clusters:
        cluster_chunks = cluster["chunks"]
        if len(cluster_chunks) <= max_per_cluster:
            candidates.extend(cluster_chunks)
            continue

        ranked = sorted(
            cluster_chunks,
            key=lambda chunk: chunk.get("score", 0.0),
            reverse=True,
        )
        if representative_strategy in {"auto", "score"}:
            candidates.extend(ranked[:max_per_cluster])
            continue

        representative = select_representatives(
            [{"id": cluster["id"], "chunks": cluster_chunks}],
            representative_strategy=representative_strategy,
            query_embedding=query_embedding,
        )[0]
        extras = [chunk for chunk in ranked if chunk["id"] != representative["id"]]
        candidates.append(representative)
        candidates.extend(extras[: max_per_cluster - 1])

    return candidates


def _select_score_representative(chunks: list[Chunk]) -> Chunk:
    return max(chunks, key=lambda chunk: chunk.get("score", 0.0))


def _select_centroid_representative(chunks: list[Chunk]) -> Chunk:
    embeddings = normalize_embeddings(embedding_matrix(chunks))
    centroid = embeddings.mean(axis=0)
    best_index = int(
        np.argmax([cosine_similarity(embedding, centroid) for embedding in embeddings])
    )
    return chunks[best_index]


def _select_query_closest_representative(
    chunks: list[Chunk],
    query_embedding: list[float] | np.ndarray | None,
) -> Chunk:
    if query_embedding is None:
        raise ValueError("query_embedding is required for query_closest strategy.")

    embeddings = normalize_embeddings(embedding_matrix(chunks))
    query = np.asarray(query_embedding, dtype=np.float32)

    if query.ndim != 1:
        raise ValueError("query_embedding must be a 1D array-like structure.")

    if query.shape[0] != embeddings.shape[1]:
        raise ValueError("query_embedding dimension must match chunk embeddings.")

    best_index = int(
        np.argmax([cosine_similarity(embedding, query) for embedding in embeddings])
    )
    return chunks[best_index]


def _select_longest_representative(chunks: list[Chunk]) -> Chunk:
    return max(chunks, key=lambda chunk: len(chunk.get("text", "")))
