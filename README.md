# SlimContext

Post-retrieval context optimizer: hash dedup → embed → semantic dedup → topical clustering → representative selection → MMR → compress → token budget.

SlimContext sits between your retriever and your LLM. You send it the chunks your vector DB, BM25, or hybrid search returned. It hands back only the ones worth sending — deduplicated, clustered, diversity-ranked, and trimmed to a token budget.

## Run

Activate the environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install packages:

```powershell
python -m pip install -r requirements.txt
```

Run the API:

```powershell
uvicorn app.api:app --reload
```

## API

```http
POST /v1/optimize
```

```json
{
  "chunks": [{ "id": "1", "text": "...", "embedding": null, "score": 0.91 }],
  "query": "How does JWT auth work?",
  "namespace": "hr-docs",
  "token_budget": 1500,
  "target_k": 8,
  "dedup_threshold": 0.15,
  "semantic_dedup_threshold": 0.001,
  "mmr_lambda": 0.5,
  "representative_strategy": "auto",
  "compress": true
}
```

`embedding` is optional. If any chunk is missing one, SlimContext embeds all chunks server-side using `embedding_model` (default: `BAAI/bge-small-en-v1.5`). If every chunk already has an embedding, client vectors are used as-is and embedding generation is skipped entirely.

`token_budget` is optional. When omitted or `null`, no token cap is applied after MMR.

## Benchmark

Two paths were compared: send all retrieved chunks to the LLM as-is, or run them through SlimContext first (`target_k=1`, `dedup_threshold=0.15`, `mmr_lambda=0.8`, `compress=false`, caller-supplied embeddings). Answer correctness was checked with GPT-5.5 using only the provided context.

| Metric | Without SlimContext | With SlimContext | Change |
|---|---:|---:|---|
| Chunks sent to LLM | 5 | 1 | −4 (80%) |
| Input tokens | 319 | 134 | −185 |
| Token reduction | — | **57.99%** | — |
| Exact duplicates removed | — | 1 | — |
| Irrelevant chunks removed | — | 3 | — |
| Clusters formed | — | 3 | — |
| Budget-skipped chunks | — | 0 | — |
| Answer correct | ✓ | ✓ | Same |
| Optimize latency | — | **2 ms** | — |

Token cost proxy at $0.005 / 1K input tokens:

| Metric | Without SlimContext | With SlimContext |
|---|---:|---:|
| Cost per request | $0.00160 | $0.00067 |
| Cost per 1M requests | $1,595 | $670 |
| Savings per 1M requests | — | **$925 (57.99%)** |

```
tokens_saved = input_tokens − output_tokens          → 319 − 134 = 185
reduction_pct  = (tokens_saved / input_tokens) × 100 → 57.99%
cost_per_req   = tokens × $0.000005
```

With pre-supplied embeddings, SlimContext only ran dedup → cluster → MMR → budget enforcement — no server-side embedding work.

### Latency

| Scenario | Observed / typical |
|---|---|
| Embeddings provided by caller | **2 ms** (observed) |
| Server-side embedding, warm model | 200–600 ms |
| First request, model loading from disk | 1–5 s |

### Reproduce pipeline metrics

Run the optimizer pipeline locally without an LLM API:

```powershell
python benchmarks/run_dirty_eval.py --target-k 1 --dedup-threshold 0.15 --mmr-lambda 0.8
```

Outputs go to `benchmarks/results/manual_eval/` (`summary.json`, `summary.csv`, spot-check prompt).

## Representative Selection

`auto` is the recommended default. It selects by `score` when any chunk in the cluster has a retrieval score greater than `0` — the normal case for RAG chunks ranked by a vector DB or reranker. Falls back to `centroid` when no usable score is present.

| Strategy | When to use |
|---|---|
| `auto` | Default. Score-based when scores exist, centroid otherwise |
| `score` | Chunks already ranked by vector DB, hybrid search, or reranker |
| `centroid` | No meaningful scores — logs, tool dumps, mixed-source content |
| `query_closest` | Q&A flows where proximity to the query matters most |
| `longest` | Summarization or context packing where information density is preferred |

## Threshold Tuning

Use two thresholds — they solve different problems:

| Parameter | Purpose | Typical range |
|---|---|---|
| `semantic_dedup_threshold` | Remove near-duplicate embeddings before MMR (also requires high text overlap) | `0.001`–`0.02` |
| `dedup_threshold` / `cluster_threshold` | Topical clustering for stats / optional `max_per_cluster` | `0.10`–`0.35` (looser) |

`semantic_dedup_threshold` should stay tight so paraphrases and distinct use cases are not merged. `dedup_threshold` controls how aggressively chunks are grouped for `cluster_count` and optional per-cluster sampling (`max_per_cluster` > 1).

With mock or low-dimensional embeddings, vectors often sit in a tight cone — rely on MMR plus the built-in text diversity penalty rather than expecting clustering alone to separate topics.

## MMR Selection

MMR (maximal marginal relevance) selects chunks with:

```
score = lambda × relevance − (1 − lambda) × diversity_penalty
```

Relevance uses `query_embedding` when provided, otherwise normalized chunk `score`. The diversity penalty is the max cosine similarity between the candidate and anything already selected.

`mmr_lambda` controls the tradeoff. `1.0` = pure relevance, `0.0` = pure diversity, `0.5` = balanced default.

`target_k` sets the chunk count limit. Token budget is enforced after MMR — chunks are packed in order until the budget is full. Chunks that would push the total over are skipped (`stats.budget_skipped_count`).

## Tests

```powershell
python -m pytest tests/ -q
```

Fast checks use fake embeddings — no model download required.

| File | What it verifies |
|------|------------------|
| `test_dedupe.py` | Exact duplicate removal and namespace isolation |
| `test_semantic_dedup.py` | Embedding + text gated near-duplicate removal |
| `test_realworld_redis.py` | Diverse Redis RAG selection (no single-chunk collapse) |
| `test_clustering.py` | Similar vectors cluster together |
| `test_mmr.py` | MMR `target_k` and token budget skipping |
| `test_embeddings.py` | Re-embed-all when any embedding is missing |
| `test_api.py` | Full HTTP endpoint with precomputed vectors |
| `test_benchmark_metrics.py` | Token reduction, cost, and summary math |
| `test_benchmark_rag.py` | Benchmark prompt construction smoke test |

## Project Structure

```
app/
  api.py                 FastAPI app and /v1/optimize endpoint
  core/
    clustering.py        Agglomerative clustering and per-cluster candidate selection
    semantic_dedup.py    Tight embedding near-duplicate removal
    compression.py       Lightweight prune and structured placeholder compression
    dedupe.py            Exact hash dedupe and MinHash near-duplicate helpers
    mmr.py               MMR ranking and token-budget packing
  data/
    chunks.py            Sample policy chunks for local testing
  scripts/
    demo_dedupe.py       Local dedupe demo
    embedding_generator.py
benchmarks/
  build_dirty_squad_dataset.py
  run_dirty_eval.py
  benchmark_rag.py
  metrics.py
  data/
    dirty_test_set.json
```
