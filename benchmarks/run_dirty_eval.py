from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from app.core.clustering import (
    cluster_chunks,
    select_representatives,
    select_top_k_by_score,
)
from app.core.compression import compress_chunks
from app.core.dedupe import remove_exact_duplicate_chunks
from app.core.mmr import enforce_token_budget, select_mmr
from app.core.semantic_dedup import remove_semantic_duplicate_chunks


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate SlimContext on dirty chunks without an LLM API key."
    )
    parser.add_argument("--input", default="benchmarks/data/dirty_test_set.json")
    parser.add_argument("--output-dir", default="benchmarks/results/manual_eval")
    parser.add_argument("--target-k", type=int, default=1)
    parser.add_argument("--token-budget", type=int, default=1500)
    parser.add_argument("--dedup-threshold", type=float, default=0.15)
    parser.add_argument("--semantic-dedup-threshold", type=float, default=0.001)
    parser.add_argument("--mmr-lambda", type=float, default=0.8)
    parser.add_argument(
        "--enable-mmr",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()

    cases = json.loads(Path(args.input).read_text(encoding="utf-8"))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    started_at = time.perf_counter()
    results = [
        run_case(
            case=case,
            target_k=args.target_k,
            token_budget=args.token_budget,
            dedup_threshold=args.dedup_threshold,
            semantic_dedup_threshold=args.semantic_dedup_threshold,
            mmr_lambda=args.mmr_lambda,
            enable_mmr=args.enable_mmr,
        )
        for case in cases
    ]
    elapsed_ms = (time.perf_counter() - started_at) * 1000

    summary = summarize(results, elapsed_ms)
    (output_dir / "summary.json").write_text(
        json.dumps({"summary": summary, "cases": results}, indent=2),
        encoding="utf-8",
    )
    write_csv(output_dir / "summary.csv", results)
    write_spot_check_prompt(output_dir / "spot_check_case_1.md", cases[0], results[0])

    print("=================== BENCHMARK RESULTS ===================")
    print(f"Total Questions Evaluated : {summary['case_count']}")
    print(f"Total Words Before System : {summary['total_before_words']} words")
    print(f"Total Words After System  : {summary['total_after_words']} words")
    print(f"Data Reduction Rate       : {summary['word_reduction_pct']:.2f}% fewer words")
    print(f"Truth Chunk Retention     : {summary['truth_retention_pct']:.2f}%")
    print(f"Total System Runtime      : {elapsed_ms:.2f} ms")
    print(f"Per Query Runtime         : {summary['avg_latency_ms']:.2f} ms")
    print(f"Outputs                   : {output_dir}")
    print("=========================================================")


def run_case(
    case: dict,
    target_k: int,
    token_budget: int,
    dedup_threshold: float,
    semantic_dedup_threshold: float,
    mmr_lambda: float,
    enable_mmr: bool,
) -> dict:
    started_at = time.perf_counter()
    dirty_chunks = [_with_embedding(chunk) for chunk in case["dirty_chunks"]]
    before_words = _word_count(chunk["text"] for chunk in dirty_chunks)

    unique_chunks = remove_exact_duplicate_chunks(
        dirty_chunks,
        namespace=f"manual-eval-{case['id']}",
    )
    semantic_unique_chunks, semantic_duplicate_count = remove_semantic_duplicate_chunks(
        unique_chunks,
        threshold=semantic_dedup_threshold,
    )
    clusters = cluster_chunks(semantic_unique_chunks, dedup_threshold=dedup_threshold)
    representatives = select_representatives(clusters, representative_strategy="auto")
    if len(representatives) <= target_k:
        selected = representatives
    elif enable_mmr:
        selected = select_mmr(
            representatives,
            target_k=target_k,
            mmr_lambda=mmr_lambda,
        )
    else:
        selected = select_top_k_by_score(representatives, target_k)
    compressed = compress_chunks(selected)
    optimized, budget_skipped_count = enforce_token_budget(
        compressed,
        token_budget=token_budget,
        token_counter=lambda chunk: len(chunk["text"].split()),
    )

    after_words = _word_count(chunk["text"] for chunk in optimized)
    optimized_ids = [chunk["id"] for chunk in optimized]

    return {
        "id": case["id"],
        "query": case["query"],
        "ground_truth_answer": case.get("ground_truth_answer", ""),
        "before_words": before_words,
        "after_words": after_words,
        "reduction_pct": _reduction_pct(before_words, after_words),
        "input_count": len(dirty_chunks),
        "output_count": len(optimized),
        "exact_duplicate_count": len(dirty_chunks) - len(unique_chunks),
        "semantic_duplicate_count": semantic_duplicate_count,
        "cluster_count": len(clusters),
        "budget_skipped_count": budget_skipped_count,
        "truth_retained": "truth" in optimized_ids,
        "optimized_chunks": [
            {
                "id": chunk["id"],
                "text": chunk["text"],
                "score": chunk.get("score", 0.0),
            }
            for chunk in optimized
        ],
        "latency_ms": round((time.perf_counter() - started_at) * 1000, 2),
    }


def summarize(results: list[dict], elapsed_ms: float) -> dict:
    total_before = sum(result["before_words"] for result in results)
    total_after = sum(result["after_words"] for result in results)
    retained_count = sum(1 for result in results if result["truth_retained"])

    return {
        "case_count": len(results),
        "total_before_words": total_before,
        "total_after_words": total_after,
        "word_reduction_pct": _reduction_pct(total_before, total_after),
        "truth_retention_pct": round((retained_count / len(results)) * 100, 2)
        if results
        else 0.0,
        "total_latency_ms": round(elapsed_ms, 2),
        "avg_latency_ms": round(elapsed_ms / len(results), 2) if results else 0.0,
    }


def write_csv(path: Path, results: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "id",
                "before_words",
                "after_words",
                "reduction_pct",
                "input_count",
                "output_count",
                "exact_duplicate_count",
                "cluster_count",
                "budget_skipped_count",
                "truth_retained",
                "latency_ms",
            ],
        )
        writer.writeheader()
        for result in results:
            writer.writerow({field: result[field] for field in writer.fieldnames})


def write_spot_check_prompt(path: Path, case: dict, result: dict) -> None:
    dirty_context = "\n\n".join(
        f"[{chunk['id']}] {chunk['text']}" for chunk in case["dirty_chunks"]
    )
    clean_context = "\n\n".join(
        f"[{chunk['id']}] {chunk['text']}" for chunk in result["optimized_chunks"]
    )

    path.write_text(
        "\n".join(
            [
                "# SlimContext Spot Check Case 1",
                "",
                "## Bloated Prompt",
                "",
                f"Context:\n{dirty_context}",
                "",
                f"Question: {case['query']}",
                "",
                "## Optimized Prompt",
                "",
                f"Context:\n{clean_context}",
                "",
                f"Question: {case['query']}",
                "",
                f"Expected answer: {case.get('ground_truth_answer', '')}",
            ]
        ),
        encoding="utf-8",
    )


def _with_embedding(chunk: dict) -> dict:
    enriched = dict(chunk)
    enriched["embedding"] = _text_embedding(enriched["text"])
    return enriched


def _text_embedding(text: str, dimensions: int = 64) -> list[float]:
    vector = np.zeros(dimensions, dtype=np.float32)

    for token in text.lower().split():
        vector[hash(token) % dimensions] += 1.0

    norm = np.linalg.norm(vector)
    if norm > 0:
        vector = vector / norm

    return vector.tolist()


def _word_count(texts) -> int:
    return sum(len(text.split()) for text in texts)


def _reduction_pct(before: int, after: int) -> float:
    if before <= 0:
        return 0.0

    return round(((before - after) / before) * 100, 2)


if __name__ == "__main__":
    main()

