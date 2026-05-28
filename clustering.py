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
) -> list[Chunk]:
    representatives = []

    for cluster in clusters:
        cluster_chunks = cluster["chunks"]

        if representative_strategy == "score":
            representative = max(
                cluster_chunks,
                key=lambda chunk: chunk.get("score", 0.0),
            )
        elif representative_strategy == "centroid":
            representative = _select_centroid_representative(cluster_chunks)
        elif representative_strategy == "auto":
            representative = _select_auto_representative(cluster_chunks)
        else:
            raise ValueError(
                "representative_strategy must be one of: auto, score, centroid"
            )

        representatives.append(representative)

    return representatives


def _select_centroid_representative(chunks: list[Chunk]) -> Chunk:
    embeddings = _normalize_embeddings(_embedding_matrix(chunks))
    centroid = embeddings.mean(axis=0)
    best_index = int(
        np.argmax([_cosine_similarity(embedding, centroid) for embedding in embeddings])
    )
    return chunks[best_index]


def _select_auto_representative(chunks: list[Chunk]) -> Chunk:
    embeddings = _normalize_embeddings(_embedding_matrix(chunks))
    centroid = embeddings.mean(axis=0)

    scores = np.asarray([chunk.get("score", 0.0) for chunk in chunks], dtype=np.float32)
    centroid_similarities = np.asarray(
        [_cosine_similarity(embedding, centroid) for embedding in embeddings],
        dtype=np.float32,
    )

    if scores.max() > scores.min():
        scores = (scores - scores.min()) / (scores.max() - scores.min())

    if centroid_similarities.max() > centroid_similarities.min():
        centroid_similarities = (
            (centroid_similarities - centroid_similarities.min())
            / (centroid_similarities.max() - centroid_similarities.min())
        )

    combined_scores = (0.6 * scores) + (0.4 * centroid_similarities)
    return chunks[int(np.argmax(combined_scores))]

