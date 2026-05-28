# SlimContext


## Run

Activate the environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install packages:

```powershell
python -m pip install -r requirements.txt
```

Run the project:

```powershell
python main.py
```

## Representative Selection

`auto` is the recommended default. It selects by `score` when any chunk in the cluster has a retrieval score greater than `0`, because RAG chunks usually arrive already ranked by vector search, hybrid search, or reranking. If no usable score is present, it falls back to `centroid`.

`score` selects the chunk with the highest retrieval score in each cluster. Use this as the default RAG behavior when chunks are already ranked by a vector database, hybrid search system, or reranker.

`centroid` selects the chunk closest to the average embedding of the cluster. Use this for generic chunks with no meaningful retrieval scores, such as logs, tool dumps, mixed-source content, or offline clustering jobs.

`query_closest` selects the chunk whose embedding is closest to the query embedding. Use this for Q&A flows where the best answer to the user query matters more than choosing the most typical chunk in the cluster.

`longest` selects the chunk with the most text. Use this when you want maximum information density, such as summarization preparation or context packing where longer chunks are preferred.

## Threshold Tuning

Clustering uses `dedup_threshold` as a cosine distance threshold. Start with `0.15` for prose and `0.10` for code, then expose the value to users so they can tune it for their corpus. Lower values make clustering stricter; higher values merge more chunks together.
