import time
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from app.core.clustering import cluster_chunks, select_cluster_candidates, select_representatives
from app.core.compression import compress_chunks
from app.core.dedupe import remove_exact_duplicate_chunks
from app.core.mmr import enforce_token_budget, select_mmr
from app.core.semantic_dedup import remove_semantic_duplicate_chunks
from app.core.vectors import validate_embedding_dimensions

Embedding = list[float]
RepresentativeStrategy = Literal[
    "auto",
    "score",
    "centroid",
    "query_closest",
    "longest",
]

DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
MAX_CHUNKS_PER_REQUEST = 500

app = FastAPI(title="SlimContext", version="0.1.0")
_embedding_models: dict[str, Any] = {}
_token_encoder = None


class ChunkIn(BaseModel):
    id: str
    text: str
    embedding: Embedding | None = None
    score: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChunkOut(BaseModel):
    id: str
    text: str
    score: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class OptimizeRequest(BaseModel):
    chunks: list[ChunkIn] = Field(min_length=1, max_length=MAX_CHUNKS_PER_REQUEST)
    query: str = ""
    namespace: str = "default"
    token_budget: int | None = Field(default=None, ge=1)
    target_k: int = Field(default=8, ge=1)
    dedup_threshold: float = Field(
        default=0.15,
        gt=0,
        le=2,
        description="Cosine distance threshold for topical clustering (stats and optional pre-MMR grouping).",
    )
    semantic_dedup_threshold: float = Field(
        default=0.001,
        gt=0,
        le=2,
        description="Tight cosine distance threshold for near-duplicate removal before MMR.",
    )
    cluster_threshold: float | None = Field(
        default=None,
        gt=0,
        le=2,
        description="Optional override for topical clustering; defaults to dedup_threshold.",
    )
    max_per_cluster: int = Field(
        default=1,
        ge=1,
        description="When >1, pass up to this many scored chunks per cluster into MMR instead of all chunks.",
    )
    mmr_lambda: float = Field(default=0.5, ge=0, le=1)
    representative_strategy: RepresentativeStrategy = "auto"
    compress: bool = True
    query_embedding: Embedding | None = None
    embedding_model: str = DEFAULT_EMBEDDING_MODEL

    @field_validator("namespace")
    @classmethod
    def validate_namespace(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("namespace must be a non-empty string.")
        return cleaned


class OptimizeStats(BaseModel):
    input_count: int
    output_count: int
    exact_duplicate_count: int
    semantic_duplicate_count: int = 0
    cluster_count: int
    input_tokens: int
    output_tokens: int
    reduction_pct: float
    latency_ms: int
    budget_skipped_count: int = 0


class OptimizeResponse(BaseModel):
    chunks: list[ChunkOut]
    stats: OptimizeStats


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/optimize", response_model=OptimizeResponse)
def optimize(request: OptimizeRequest) -> OptimizeResponse:
    started_at = time.perf_counter()

    try:
        chunks = [chunk.model_dump() for chunk in request.chunks]
        input_tokens = sum(_count_text_tokens(chunk["text"]) for chunk in chunks)

        exact_unique_chunks = remove_exact_duplicate_chunks(chunks, request.namespace)
        _ensure_embeddings(exact_unique_chunks, request.embedding_model)

        query_embedding = _query_embedding_for_request(request)
        validate_embedding_dimensions(exact_unique_chunks, query_embedding)

        semantic_unique_chunks, semantic_duplicate_count = remove_semantic_duplicate_chunks(
            exact_unique_chunks,
            threshold=request.semantic_dedup_threshold,
        )

        cluster_distance = (
            request.cluster_threshold
            if request.cluster_threshold is not None
            else request.dedup_threshold
        )
        clusters = cluster_chunks(
            semantic_unique_chunks,
            dedup_threshold=cluster_distance,
        )

        if request.max_per_cluster > 1:
            mmr_candidates = select_cluster_candidates(
                clusters,
                max_per_cluster=request.max_per_cluster,
                representative_strategy=request.representative_strategy,
                query_embedding=query_embedding,
            )
        else:
            representatives = select_representatives(
                clusters,
                representative_strategy=request.representative_strategy,
                query_embedding=query_embedding,
            )
            mmr_candidates = (
                representatives
                if len(representatives) > 1
                else semantic_unique_chunks
            )

        mmr_chunks = select_mmr(
            mmr_candidates,
            target_k=request.target_k,
            mmr_lambda=request.mmr_lambda,
            query_embedding=query_embedding,
        )

        final_candidates = compress_chunks(mmr_chunks) if request.compress else mmr_chunks
        if request.token_budget is None:
            final_chunks = final_candidates
            budget_skipped_count = 0
        else:
            final_chunks, budget_skipped_count = enforce_token_budget(
                final_candidates,
                request.token_budget,
                token_counter=lambda chunk: _count_text_tokens(chunk.get("text", "")),
            )

        output_tokens = sum(_count_text_tokens(chunk["text"]) for chunk in final_chunks)
        latency_ms = int((time.perf_counter() - started_at) * 1000)

        return OptimizeResponse(
            chunks=[ChunkOut(**chunk) for chunk in final_chunks],
            stats=OptimizeStats(
                input_count=len(chunks),
                output_count=len(final_chunks),
                exact_duplicate_count=len(chunks) - len(exact_unique_chunks),
                semantic_duplicate_count=semantic_duplicate_count,
                cluster_count=len(clusters),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                reduction_pct=_reduction_pct(input_tokens, output_tokens),
                latency_ms=latency_ms,
                budget_skipped_count=budget_skipped_count,
            ),
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


def _needs_query_embedding(request: OptimizeRequest) -> bool:
    if request.query_embedding is not None:
        return False

    if not request.query.strip():
        return False

    return True


def _has_embedding(chunk: dict) -> bool:
    embedding = chunk.get("embedding")
    return embedding is not None and len(embedding) > 0


def _ensure_embeddings(chunks: list[dict], embedding_model: str) -> None:
    """Embed all chunks when any embedding is missing (Distill-style, single vector space)."""
    if not chunks:
        return

    if all(_has_embedding(chunk) for chunk in chunks):
        return

    model = _get_embedding_model(embedding_model)
    texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(texts)

    for chunk, embedding in zip(chunks, embeddings):
        chunk["embedding"] = _embedding_to_list(embedding)


def _embedding_to_list(embedding: Any) -> list[float]:
    if hasattr(embedding, "tolist"):
        return embedding.tolist()
    return list(embedding)


def _query_embedding_for_request(request: OptimizeRequest) -> Embedding | None:
    if request.query_embedding is not None:
        return request.query_embedding

    if not _needs_query_embedding(request):
        return None

    model = _get_embedding_model(request.embedding_model)
    return model.encode([request.query.strip()])[0].tolist()


def _get_embedding_model(model_name: str):
    if model_name not in _embedding_models:
        from sentence_transformers import SentenceTransformer

        _embedding_models[model_name] = SentenceTransformer(model_name)

    return _embedding_models[model_name]


def _count_text_tokens(text: str) -> int:
    global _token_encoder

    if _token_encoder is None:
        try:
            import tiktoken

            _token_encoder = tiktoken.get_encoding("cl100k_base")
        except Exception:
            _token_encoder = False

    if _token_encoder:
        return len(_token_encoder.encode(text))

    return len(text.split())


def _reduction_pct(input_tokens: int, output_tokens: int) -> float:
    if input_tokens <= 0:
        return 0.0

    return round(((input_tokens - output_tokens) / input_tokens) * 100, 2)
