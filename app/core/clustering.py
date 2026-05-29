from sklearn.cluster import AgglomerativeClustering
import numpy as np

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

    embeddings = normalize_embeddings(embedding_matrix(chunks))

    if dedup_threshold <= 0:
        raise ValueError("dedup_threshold must be greater than 0.")

    model = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=dedup_threshold,
        metric="cosine",
        linkage="average",
    )
    labels = model.fit_predict(embeddings)

    clusters_by_label: dict[int, list[Chunk]] = {}
    for label, chunk in zip(labels, chunks):
        clusters_by_label.setdefault(int(label), []).append(chunk)

    clusters = [
        {"id": cluster_id, "chunks": cluster_chunks}
        for cluster_id, cluster_chunks in clusters_by_label.items()
    ]

    clusters.sort(key=lambda cluster: cluster["id"])
    return clusters


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

