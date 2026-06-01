# SlimContext

**Post-retrieval context optimizer for RAG — deterministic, fast, no LLM.**

SlimContext is a standalone Python service for the **retrieve → optimize → generate** step in RAG. It takes over-fetched chunks from your vector DB, BM25, or hybrid retriever and returns a smaller set that maximizes **information per token**: deduplicated, topically grouped, diversity-ranked, and trimmed to a budget.

**No LLM calls.** Same input always yields the same output. Typical latency is single-digit to low hundreds of milliseconds depending on whether embeddings are supplied by the caller.

---

## The problem

Production RAG usually **over-fetches** on purpose: retrieve 20–50 chunks, then hope the model figures it out. In practice:

- **30–40% of retrieved context is semantically redundant** (same idea, different wording).
- **High-scoring chunks cluster on one theme** (e.g. five “Redis is a cache” passages) while other useful topics never reach the LLM.
- **Token cost and latency scale with raw chunk count**, not with useful facts.

Fetching fewer results from the vector DB hurts **recall**. The better pattern is:

> **Over-fetch for recall → optimize for precision and diversity → send to the LLM.**

SlimContext is that optimization step, exposed as a small HTTP API.

---

## Where it fits

```
Vector DB / BM25 / hybrid search
        ↓  (over-fetch: many chunks + scores + embeddings)
   SlimContext                    ← you are here
        ↓  (dedupe · cluster · MMR · budget)
        LLM
```

---

## Architecture

### End-to-end pipeline

```
┌─────────────┐   ┌──────────────────┐   ┌─────────────────┐   ┌──────────────┐
│ Exact dedup │ → │ Semantic dedup   │ → │ Topical cluster │ → │ Representative│
│ (hash)      │   │ (paraphrase +    │   │ (intent / hybrid│   │ (1 per cluster)│
│             │   │  intent collapse)│   │  text+embedding)│   │               │
└─────────────┘   └──────────────────┘   └─────────────────┘   └───────┬──────┘
                                                                         ↓
┌─────────────┐   ┌──────────────────┐   ┌─────────────────────────────────┐
│ Compression │ ← │ Token budget     │ ← │ MMR selection (relevance +       │
│ (optional)  │   │ (pack in order)  │   │  diversity across topics)        │
└─────────────┘   └──────────────────┘   └─────────────────────────────────┘
```

Canonical RAG optimization path:

```
Query → Over-fetch (N) → Cluster → Select → MMR (k) → LLM
```

SlimContext adds an explicit **semantic dedup** stage before clustering so paraphrases do not consume `target_k` slots.

### What each layer does (and why)

| Layer | What it does | Why it exists |
|-------|----------------|---------------|
| **1. Exact dedup** | SHA-256 hash of normalized text, scoped by `namespace`. | Cheap, perfect removal of copy-paste duplicates from multi-source retrieval. |
| **2. Embed** (optional) | If any chunk lacks an embedding, encode all texts with `BAAI/bge-small-en-v1.5` (single vector space). | Clustering and MMR need vectors; callers with precomputed embeddings skip this entirely. |
| **3. Semantic dedup** | Two passes on chunks still ranked by retrieval `score`: **(a) Pairwise paraphrase removal** — drop a lower-scored chunk when embeddings are nearly identical *and* TF-IDF overlap suggests the same claim (not merely the same topic). **(b) Intent collapse** — assign a lightweight topic label from keywords (e.g. `caching`, `session`, `messaging`, `rate_limiting`) and keep only the top-scored chunk per label. | Example: five retrieved passages all explain “Redis caches hot data in RAM” (`redis_core_1`, `redis_cache_paraphrase`, `semantic_overlap_1`, …). After this stage you keep **one** caching passage (highest score), plus separate winners for session storage, pub/sub, persistence, etc. That is different from topical clustering (layer 4), which groups what is left; semantic dedup answers “have we already said this?” |
| **4. Topical clustering** | Group remaining chunks by intent labels when possible; otherwise agglomerative clustering on a **hybrid** distance (text-weighted when embeddings are collapsed). | Produces meaningful `cluster_count` and one representative per *idea*, not one blob per embedding cone. |
| **5. Representative selection** | Pick the best chunk per cluster (`auto` = highest retrieval `score`, or centroid / query-closest / longest). | Reduces each topic to its strongest evidence before diversity ranking. |
| **6. MMR** | Maximal Marginal Relevance: `λ × relevance − (1−λ) × diversity_penalty`. Diversity uses embedding similarity **and** lexical overlap vs. already-selected chunks. | Maximizes **marginal information gain** under `target_k`, not raw cosine score alone. |
| **7. Compression** | Light filler removal; structured text truncated with a placeholder. | Cuts noise without an LLM summarizer. |
| **8. Token budget** | Pack whole chunks in MMR order until `token_budget` is full; skip chunks that do not fit. | Hard cap for model context windows and cost control. |

**Design principle:** optimize for **useful coverage under a token budget**, not minimum redundancy alone. A dedup engine returns one chunk; a context optimizer returns one chunk *per distinct intent* (up to `target_k`).

---

## Why no LLM?

| Approach | Deterministic | Typical latency | Cost |
|----------|---------------|-----------------|------|
| LLM compression / rerank | No | ~500ms+ | Per-token API |
| **SlimContext** | **Yes** | **~2ms** (precomputed embeddings) to **~600ms** (server-side embed) | Compute only |

Algorithms only: cosine distance, TF-IDF, agglomerative clustering, MMR. Auditable, testable, safe to run on every request.

---

## Quick start

```powershell
cd SlimContext
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.api:app --reload
```

Health check: `GET http://127.0.0.1:8000/health`

---

## API

### `POST /v1/optimize`

**Request** (minimal):

```json
{
  "query": "How does Redis improve performance in distributed systems?",
  "query_embedding": [0.81, 0.64, 0.72, 0.90],
  "namespace": "docs",
  "chunks": [
    {
      "id": "chunk-1",
      "text": "Redis stores hot data in memory...",
      "embedding": [0.80, 0.63, 0.71, 0.89],
      "score": 0.95
    }
  ],
  "target_k": 8,
  "token_budget": 1800,
  "semantic_dedup_threshold": 0.001,
  "dedup_threshold": 0.15,
  "mmr_lambda": 0.5,
  "representative_strategy": "auto",
  "compress": true
}
```

**Response:**

```json
{
  "chunks": [{ "id": "chunk-1", "text": "...", "score": 0.95, "metadata": {} }],
  "stats": {
    "input_count": 21,
    "output_count": 7,
    "exact_duplicate_count": 1,
    "semantic_duplicate_count": 4,
    "cluster_count": 7,
    "input_tokens": 573,
    "output_tokens": 210,
    "reduction_pct": 63.35,
    "latency_ms": 12,
    "budget_skipped_count": 0
  }
}
```

### Parameters

| Field | Default | Role |
|-------|---------|------|
| `chunks` | required | Retrieved passages (`id`, `text`, optional `embedding`, `score`, `metadata`). |
| `namespace` | `"default"` | Isolates exact-dedup hashes across tenants / indexes. |
| `query` | `""` | Used to embed a query vector when `query_embedding` is omitted. |
| `query_embedding` | optional | Query vector for MMR relevance; preferred when the retriever already has it. |
| `target_k` | `8` | Max chunks after MMR. |
| `token_budget` | optional | Max tokens after MMR; `null` = no cap. |
| `semantic_dedup_threshold` | `0.001` | Tight cosine distance for paraphrase detection (see tuning). |
| `dedup_threshold` | `0.15` | Topical clustering distance (looser than semantic dedup). |
| `cluster_threshold` | optional | Overrides `dedup_threshold` for clustering only. |
| `mmr_lambda` | `0.5` | `1.0` = relevance only, `0.0` = diversity only. |
| `representative_strategy` | `auto` | `auto` \| `score` \| `centroid` \| `query_closest` \| `longest` |
| `max_per_cluster` | `1` | If `>1`, send up to N chunks per cluster into MMR before final selection. |
| `compress` | `true` | Apply lightweight compression to output text. |
| `embedding_model` | `BAAI/bge-small-en-v1.5` | Used only when any chunk is missing an embedding. |

### Representative strategies

After topical clustering, several chunks may still belong to the same group (e.g. three passages tagged `caching` that survived semantic dedup because wording differed enough). **Representative selection** picks **one chunk per cluster** to send forward to MMR. That keeps MMR focused on *topics*, not on picking among siblings in the same cluster.

Set via `representative_strategy` (default: `auto`).

| Strategy | How the winner is chosen | Best for |
|----------|--------------------------|----------|
| **`auto`** | If any chunk in the cluster has `score > 0`, use **`score`**; otherwise use **`centroid`**. | Most RAG pipelines (vector DB or reranker already provides scores). |
| **`score`** | Highest `score` in the cluster. | Hybrid / BM25 / vector search where `score` reflects retriever confidence. |
| **`centroid`** | Chunk whose embedding is closest to the cluster’s average embedding. | Tool output, logs, or scraped text with no retrieval score — picks the most “typical” passage, not the longest or highest arbitrary score. |
| **`query_closest`** | Chunk whose embedding has highest cosine similarity to `query_embedding`. Requires `query_embedding` (or `query` so the server can embed it). | Q&A when the best evidence is “closest to what the user asked,” not highest retriever score (e.g. a lower-ranked chunk that directly answers the question). |
| **`longest`** | Chunk with the most characters in `text`. | Summarization or context packing when length proxies for detail (use carefully — long ≠ relevant). |

**Example.** One cluster contains:

- `redis_core_1` — score `0.96`, defines in-memory caching  
- `redis_cache_paraphrase` — score `0.94`, shorter paraphrase  

With `score` or `auto`, `redis_core_1` becomes the representative. With `query_closest`, the winner depends on which embedding aligns better with the query vector.

**Interaction with `max_per_cluster`.** Default `max_per_cluster=1`: only representatives go to MMR. If `max_per_cluster > 1`, up to N highest-scored chunks *per cluster* are passed to MMR instead of a single representative — useful when a cluster is broad and you want MMR to trim within it.

---

## Tuning

Use **two thresholds** — they answer different questions:

| Parameter | Question it answers | Typical range |
|-----------|---------------------|---------------|
| `semantic_dedup_threshold` | “Are these the same information?” | `0.0005` – `0.02` (keep tight) |
| `dedup_threshold` | “Which chunks belong to the same topic?” | `0.10` – `0.35` (looser) |

**MMR:**

```
mmr_score = λ × relevance − (1 − λ) × diversity_penalty
```

- Relevance: cosine(query, chunk) if `query_embedding` set, else normalized retrieval `score`.
- Diversity penalty: max(embedding similarity, lexical similarity) vs. already-selected chunks.

**Practical defaults for prose RAG:** `semantic_dedup_threshold=0.001`, `dedup_threshold=0.15`, `mmr_lambda=0.5–0.75`, `target_k=6–10`.

**Note:** Low-dimensional or hand-made test embeddings often sit in a tight cone; SlimContext compensates with text/intent signals. Use real embeddings (e.g. bge-small) for production tuning.

---

## Benchmarks

**The latencies below are from the offline benchmark script, which does not use caller-supplied embeddings** (it builds 64-dim hash vectors per chunk). On `POST /v1/optimize` with embeddings already attached to every chunk, the same pipeline is typically **under ~10 ms per request** — often faster — because no embedding model runs.

Pipeline-only evaluation on `benchmarks/data/dirty_test_set.json` (100 RAG-style cases, 5 noisy chunks each including one exact duplicate and one `truth` chunk). **No LLM calls** — metrics are word counts, chunk counts, whether the `truth` chunk survived, and wall-clock time. Embeddings are **deterministic 64-dim hash vectors** generated inside the script (not BGE).

From the project root (`SlimContext/`):

```powershell
$env:PYTHONPATH = (Get-Location)
python benchmarks/run_dirty_eval.py --target-k 1 --dedup-threshold 0.15 --mmr-lambda 0.8
```

```bash
PYTHONPATH=. python benchmarks/run_dirty_eval.py --target-k 1 --dedup-threshold 0.15 --mmr-lambda 0.8
```

Outputs: `benchmarks/results/manual_eval/summary.json`, `summary.csv`, `spot_check_case_1.md`.

### Results (measured May 2026)

**Config:** `--target-k 1 --dedup-threshold 0.15 --semantic-dedup-threshold 0.001` (default) `--mmr-lambda 0.8` `--token-budget 1500`

| Metric | Value |
|--------|------:|
| Cases evaluated | 100 |
| Total words (raw retrieved context) | 26,685 |
| Total words (after SlimContext) | 7,102 |
| Word reduction | **73.39%** |
| Truth chunk retained (`id: truth`) | **48%** of cases |
| Total runtime | 2,594.67 ms |
| Avg latency per case | **25.95 ms** |

**Same dataset with more output slots:** `--target-k 8 --mmr-lambda 0.75`

| Metric | Value |
|--------|------:|
| Total words (after SlimContext) | 10,149 |
| Word reduction | **61.97%** |
| Truth chunk retained | **48%** of cases |
| Avg latency per case | **9.53 ms** |

**Example — case 1** (Massachusetts compulsory education query):

| Metric | Before | After |
|--------|-------:|------:|
| Chunks | 5 | 1 |
| Words | 206 | 59 |
| Word reduction | — | 71.36% |
| Exact duplicates removed | — | 1 |
| Clusters formed | — | 4 |
| Truth retained | — | no (`exact_duplicate` kept over `truth`, same text, higher pipeline order) |

Interpretation: high **reduction** is expected with `target_k=1`. **Truth retention** depends on scores, clustering, and MMR — it is not an answer-quality benchmark. For production, use real retriever embeddings and tune `target_k`, `mmr_lambda`, and thresholds on your own data.

---

## Tests

```powershell
python -m pytest tests/ -q
```

Unit tests use synthetic embeddings — no model download required.

| Test | Covers |
|------|--------|
| `test_dedupe.py` | Exact duplicate removal, namespace isolation |
| `test_semantic_dedup.py` | Paraphrase / near-duplicate removal |
| `test_realworld_redis.py` | Multi-topic RAG set (no single-chunk collapse) |
| `test_clustering.py` | Vector clustering |
| `test_mmr.py` | `target_k`, token budget |
| `test_api.py` | HTTP `/v1/optimize` |
| `test_embeddings.py` | Re-embed when any vector is missing |

---

## Project layout

```
app/
  api.py                  # FastAPI — POST /v1/optimize
  core/
    dedupe.py             # Exact hash dedup
    semantic_dedup.py     # Paraphrase + intent collapse
    clustering.py         # Topical clusters + representatives
    mmr.py                # MMR + token budget packing
    compression.py        # Deterministic text pruning
    text_similarity.py    # Lexical / TF-IDF helpers
    intent.py             # Lightweight topic labels
    vectors.py            # Embedding utilities
benchmarks/
  run_dirty_eval.py       # Pipeline metrics without an LLM
  data/dirty_test_set.json
tests/
```

---

## Credits

Pipeline concepts (over-fetch, semantic dedup, clustering, MMR) draw on ideas explored in [Distill](https://github.com/Siddhant-K-code/distill) and the [Agentic Engineering Guide — context engineering stack](https://agents.siddhantkhare.com/05-context-engineering-stack/). SlimContext is an independent project with its own codebase and API.

---

## License

MIT — see [LICENSE](LICENSE) in this repository.
