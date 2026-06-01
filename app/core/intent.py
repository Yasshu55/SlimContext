"""Lightweight topical intent labels for clustering and redundancy collapse."""

from app.core.dedupe import normalize_text

INTENT_PATTERNS: list[tuple[str, tuple[str, ...]]] = [
    ("caching", ("cache", "caching", "hot data", "in-memory", "latency", "database hit", "read-heavy")),
    ("session", ("session", "session storage")),
    ("messaging", ("pub/sub", "pubsub", "messaging", "message broker")),
    ("rate_limiting", ("rate limit", "rate limiting", "increment operation")),
    ("persistence", ("persistence", "rdb", "aof", "snapshot")),
    ("scaling", ("redis cluster", "partition", "horizontal scaling", "horizontal scale")),
    ("locking", ("distributed lock", "redlock", "locking")),
    ("queueing", ("queue", "job processing", "push and pop")),
    ("failover", ("sentinel", "failover", "high availability")),
    ("eviction", ("eviction", "lru", "lfu")),
]


def intent_key(text: str) -> str:
    normalized = normalize_text(text)

    for label, phrases in INTENT_PATTERNS:
        if any(phrase in normalized for phrase in phrases):
            return label

    return "general"
