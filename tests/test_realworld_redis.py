"""Regression: mock Redis RAG set must not collapse or over-select cache paraphrases.

This test uses synthetic (non-BGE) embeddings that are intentionally spaced apart so
agglomerative clustering can form multiple clusters.
"""

from fastapi.testclient import TestClient
import numpy as np

from app.api import app

client = TestClient(app)


def _unit_vector(axis: int, dim: int = 8) -> list[float]:
    vector = np.zeros(dim, dtype=np.float32)
    vector[axis % dim] = 1.0
    return vector.tolist()

REDIS_CHUNKS = [
    {
        "id": "redis_core_1",
        "text": "Redis improves application performance by storing frequently accessed data in memory, reducing the need to repeatedly query slower databases. Because Redis operations are memory-based, read latency is typically sub-millisecond.",
        "score": 0.962,
        "embedding": _unit_vector(0),
    },
    {
        "id": "redis_core_duplicate",
        "text": "Redis improves application performance by storing frequently accessed data in memory, reducing the need to repeatedly query slower databases. Because Redis operations are memory-based, read latency is typically sub-millisecond.",
        "score": 0.959,
        "embedding": _unit_vector(0),
    },
    {
        "id": "redis_cache_paraphrase",
        "text": "Applications commonly use Redis as an in-memory cache layer to avoid repeated database hits and decrease response times for hot data.",
        "score": 0.944,
        "embedding": _unit_vector(0),
    },
    {
        "id": "redis_session_store",
        "text": "Redis is often used for distributed session storage because it provides fast reads and writes across multiple application instances.",
        "score": 0.901,
        "embedding": _unit_vector(1),
    },
    {
        "id": "redis_rate_limiting",
        "text": "High-scale APIs frequently use Redis for rate limiting because atomic increment operations are extremely fast and efficient.",
        "score": 0.882,
        "embedding": _unit_vector(2),
    },
    {
        "id": "redis_pubsub",
        "text": "Redis Pub/Sub enables lightweight real-time messaging between distributed services without requiring a heavyweight message broker.",
        "score": 0.854,
        "embedding": _unit_vector(3),
    },
    {
        "id": "redis_persistence",
        "text": "Although Redis is primarily in-memory, it supports persistence using RDB snapshots and AOF logs to recover data after restarts.",
        "score": 0.821,
        "embedding": _unit_vector(4),
    },
    {
        "id": "redis_scaling",
        "text": "Redis Cluster allows horizontal scaling by partitioning keys across multiple nodes, improving throughput for distributed workloads.",
        "score": 0.847,
        "embedding": _unit_vector(5),
    },
    {
        "id": "redis_distributed_locking",
        "text": "Distributed systems often implement locking with Redis using algorithms such as Redlock to coordinate access to shared resources.",
        "score": 0.819,
        "embedding": _unit_vector(6),
    },
    {
        "id": "semantic_overlap_1",
        "text": "Caching hot data inside Redis minimizes expensive database queries and improves throughput for read-heavy applications.",
        "score": 0.913,
        "embedding": _unit_vector(0),
    },
    {
        "id": "verbose_chunk_1",
        "text": "Modern distributed systems frequently place Redis between the application layer and the primary relational database. By caching high-frequency queries inside memory, Redis dramatically lowers average response times and reduces pressure on backend systems. In large-scale deployments, Redis may also be used for distributed locking, session persistence, pub/sub communication, rate limiting, and queue management. Organizations operating microservice architectures commonly deploy Redis clusters to improve scalability and fault tolerance across services.",
        "score": 0.936,
        "embedding": _unit_vector(0),
    },
    {
        "id": "irrelevant_postgres",
        "text": "PostgreSQL supports advanced indexing techniques such as GIN and BRIN indexes for analytical and transactional workloads.",
        "score": 0.601,
        "embedding": _unit_vector(7),
    },
]

CACHE_VARIANT_IDS = {
    "redis_cache_paraphrase",
    "semantic_overlap_1",
    "verbose_chunk_1",
}


def test_redis_realworld_preserves_diverse_context() -> None:
    payload = {
        "query": "How does Redis improve application performance in distributed systems?",
        "query_embedding": _unit_vector(0),
        "namespace": "slimcontext-realworld-test",
        "target_k": 8,
        "dedup_threshold": 0.15,
        "semantic_dedup_threshold": 0.001,
        "mmr_lambda": 0.75,
        "representative_strategy": "auto",
        "compress": True,
        "token_budget": 1800,
        "chunks": REDIS_CHUNKS,
    }

    response = client.post("/v1/optimize", json=payload)

    assert response.status_code == 200
    body = response.json()
    selected_ids = {chunk["id"] for chunk in body["chunks"]}

    assert body["stats"]["output_count"] >= 6
    assert body["stats"]["exact_duplicate_count"] == 1
    assert body["stats"]["semantic_duplicate_count"] >= 2
    assert body["stats"]["cluster_count"] >= 1
    assert "redis_core_1" in selected_ids
    assert selected_ids.isdisjoint(CACHE_VARIANT_IDS)
    assert {
        "redis_pubsub",
        "redis_scaling",
        "redis_persistence",
        "redis_rate_limiting",
        "redis_session_store",
        "redis_distributed_locking",
    } <= selected_ids
