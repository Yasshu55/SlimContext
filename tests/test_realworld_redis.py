"""Regression: mock Redis RAG set must not collapse or over-select cache paraphrases."""

from fastapi.testclient import TestClient

from app.api import app

client = TestClient(app)

REDIS_CHUNKS = [
    {
        "id": "redis_core_1",
        "text": "Redis improves application performance by storing frequently accessed data in memory, reducing the need to repeatedly query slower databases. Because Redis operations are memory-based, read latency is typically sub-millisecond.",
        "score": 0.962,
        "embedding": [0.801, 0.631, 0.714, 0.897, 0.521, 0.418, 0.601, 0.852],
    },
    {
        "id": "redis_core_duplicate",
        "text": "Redis improves application performance by storing frequently accessed data in memory, reducing the need to repeatedly query slower databases. Because Redis operations are memory-based, read latency is typically sub-millisecond.",
        "score": 0.959,
        "embedding": [0.801, 0.631, 0.714, 0.897, 0.521, 0.418, 0.601, 0.852],
    },
    {
        "id": "redis_cache_paraphrase",
        "text": "Applications commonly use Redis as an in-memory cache layer to avoid repeated database hits and decrease response times for hot data.",
        "score": 0.944,
        "embedding": [0.792, 0.624, 0.708, 0.884, 0.537, 0.409, 0.613, 0.841],
    },
    {
        "id": "redis_session_store",
        "text": "Redis is often used for distributed session storage because it provides fast reads and writes across multiple application instances.",
        "score": 0.901,
        "embedding": [0.773, 0.618, 0.699, 0.851, 0.541, 0.427, 0.592, 0.824],
    },
    {
        "id": "redis_rate_limiting",
        "text": "High-scale APIs frequently use Redis for rate limiting because atomic increment operations are extremely fast and efficient.",
        "score": 0.882,
        "embedding": [0.741, 0.603, 0.684, 0.834, 0.554, 0.442, 0.581, 0.801],
    },
    {
        "id": "redis_pubsub",
        "text": "Redis Pub/Sub enables lightweight real-time messaging between distributed services without requiring a heavyweight message broker.",
        "score": 0.854,
        "embedding": [0.728, 0.588, 0.673, 0.812, 0.561, 0.451, 0.566, 0.792],
    },
    {
        "id": "redis_persistence",
        "text": "Although Redis is primarily in-memory, it supports persistence using RDB snapshots and AOF logs to recover data after restarts.",
        "score": 0.821,
        "embedding": [0.711, 0.574, 0.652, 0.804, 0.582, 0.467, 0.551, 0.773],
    },
    {
        "id": "redis_scaling",
        "text": "Redis Cluster allows horizontal scaling by partitioning keys across multiple nodes, improving throughput for distributed workloads.",
        "score": 0.847,
        "embedding": [0.732, 0.596, 0.681, 0.825, 0.563, 0.448, 0.573, 0.794],
    },
    {
        "id": "redis_distributed_locking",
        "text": "Distributed systems often implement locking with Redis using algorithms such as Redlock to coordinate access to shared resources.",
        "score": 0.819,
        "embedding": [0.709, 0.573, 0.649, 0.798, 0.564, 0.459, 0.552, 0.771],
    },
    {
        "id": "semantic_overlap_1",
        "text": "Caching hot data inside Redis minimizes expensive database queries and improves throughput for read-heavy applications.",
        "score": 0.913,
        "embedding": [0.774, 0.616, 0.702, 0.863, 0.538, 0.425, 0.597, 0.831],
    },
    {
        "id": "verbose_chunk_1",
        "text": "Modern distributed systems frequently place Redis between the application layer and the primary relational database. By caching high-frequency queries inside memory, Redis dramatically lowers average response times and reduces pressure on backend systems. In large-scale deployments, Redis may also be used for distributed locking, session persistence, pub/sub communication, rate limiting, and queue management. Organizations operating microservice architectures commonly deploy Redis clusters to improve scalability and fault tolerance across services.",
        "score": 0.936,
        "embedding": [0.786, 0.627, 0.711, 0.891, 0.526, 0.413, 0.609, 0.846],
    },
    {
        "id": "irrelevant_postgres",
        "text": "PostgreSQL supports advanced indexing techniques such as GIN and BRIN indexes for analytical and transactional workloads.",
        "score": 0.601,
        "embedding": [0.532, 0.471, 0.588, 0.694, 0.431, 0.517, 0.491, 0.611],
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
        "query_embedding": [0.812, 0.644, 0.721, 0.903, 0.533, 0.402, 0.618, 0.844],
        "namespace": "slimcontext-realworld-test",
        "target_k": 8,
        "dedup_threshold": 0.12,
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
    assert body["stats"]["cluster_count"] >= 4
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
