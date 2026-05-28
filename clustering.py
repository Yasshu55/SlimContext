from sklearn.cluster import AgglomerativeClustering
import numpy as np


Chunk = dict
Cluster = dict


def _embedding_matrix(chunks: list[Chunk]) -> np.ndarray:
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


def _normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return embeddings / norms


def _cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left_norm = np.linalg.norm(left)
    right_norm = np.linalg.norm(right)

    if left_norm == 0 or right_norm == 0:
        return 0.0

    return float(np.dot(left, right) / (left_norm * right_norm))


def cluster_chunks(
    chunks: list[Chunk],
    target_k: int = 8,
    dedup_threshold: float = 0.15,
) -> list[Cluster]:
    if not chunks:
        return []

    if len(chunks) == 1:
        return [{"id": 0, "chunks": chunks}]

    embeddings = _normalize_embeddings(_embedding_matrix(chunks))

    distance_threshold = dedup_threshold if dedup_threshold > 0 else None
    n_clusters = None if distance_threshold is not None else min(target_k, len(chunks))

    model = AgglomerativeClustering(
        n_clusters=n_clusters,
        distance_threshold=distance_threshold,
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

    if target_k > 0 and len(clusters) > target_k:
        clusters = _merge_to_target_k(clusters, target_k)

    return clusters


def _merge_to_target_k(clusters: list[Cluster], target_k: int) -> list[Cluster]:
    sorted_clusters = sorted(
        clusters,
        key=lambda cluster: max(chunk.get("score", 0.0) for chunk in cluster["chunks"]),
        reverse=True,
    )

    kept_clusters = sorted_clusters[:target_k]
    overflow_chunks = [
        chunk
        for cluster in sorted_clusters[target_k:]
        for chunk in cluster["chunks"]
    ]

    if overflow_chunks:
        kept_clusters[-1]["chunks"].extend(overflow_chunks)

    for index, cluster in enumerate(kept_clusters):
        cluster["id"] = index

    return kept_clusters


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
    embeddings = _normalize_embeddings(_embedding_matrix(chunks))
    centroid = embeddings.mean(axis=0)
    best_index = int(
        np.argmax([_cosine_similarity(embedding, centroid) for embedding in embeddings])
    )
    return chunks[best_index]


def _select_query_closest_representative(
    chunks: list[Chunk],
    query_embedding: list[float] | np.ndarray | None,
) -> Chunk:
    if query_embedding is None:
        raise ValueError("query_embedding is required for query_closest strategy.")

    embeddings = _normalize_embeddings(_embedding_matrix(chunks))
    query = np.asarray(query_embedding, dtype=np.float32)

    if query.ndim != 1:
        raise ValueError("query_embedding must be a 1D array-like structure.")

    if query.shape[0] != embeddings.shape[1]:
        raise ValueError("query_embedding dimension must match chunk embeddings.")

    best_index = int(
        np.argmax([_cosine_similarity(embedding, query) for embedding in embeddings])
    )
    return chunks[best_index]


def _select_longest_representative(chunks: list[Chunk]) -> Chunk:
    return max(chunks, key=lambda chunk: len(chunk.get("text", "")))

