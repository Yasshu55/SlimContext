from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Protocol

import numpy as np

from app.api import ChunkIn, OptimizeRequest, optimize
from benchmarks.metrics import BenchmarkCase, CaseResult, load_jsonl_dataset, summarize_results


class AnswerClient(Protocol):
    def answer(self, model: str, question: str, context: list[str]) -> tuple[str, float] | str:
        ...


class Judge(Protocol):
    def score(
        self,
        question: str,
        expected_answer: str,
        actual_answer: str,
        context: list[str],
    ) -> float:
        ...


class EchoAnswerClient:
    def answer(self, model: str, question: str, context: list[str]) -> tuple[str, float]:
        started_at = time.perf_counter()
        answer = context[0] if context else ""
        return answer, (time.perf_counter() - started_at) * 1000


class ExactSubstringJudge:
    def score(
        self,
        question: str,
        expected_answer: str,
        actual_answer: str,
        context: list[str],
    ) -> float:
        if not expected_answer:
            return 0.0

        haystack = " ".join([actual_answer, *context]).lower()
        return 10.0 if expected_answer.lower() in haystack else 1.0


def build_answer_prompt(question: str, context: list[str]) -> str:
    numbered_context = "\n".join(
        f"[{index}] {chunk}" for index, chunk in enumerate(context, start=1)
    )
    return (
        "Answer the question using only the provided context.\n\n"
        f"Context:\n{numbered_context}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )


def embed_case(
    case: BenchmarkCase,
    embedding_model_name: str,
) -> tuple[list[ChunkIn], list[float]]:
    chunks = [
        ChunkIn(
            id=f"{case.id}-chunk-{index}",
            text=text,
            embedding=_text_embedding(text),
            score=max(0.1, 1.0 - index * 0.05),
        )
        for index, text in enumerate(case.chunks)
    ]
    return chunks, _text_embedding(case.question)


def run_case(
    case: BenchmarkCase,
    answer_client: AnswerClient,
    judge: Judge,
    model: str,
    embedding_model_name: str,
    target_k: int,
    token_budget: int,
) -> CaseResult:
    baseline_context = case.chunks
    baseline_answer, baseline_latency_ms = _answer_with_latency(
        answer_client,
        model,
        case.question,
        baseline_context,
    )

    embedded_chunks, query_embedding = embed_case(case, embedding_model_name)
    optimize_started_at = time.perf_counter()
    optimized = optimize(
        OptimizeRequest(
            chunks=embedded_chunks,
            query=case.question,
            query_embedding=query_embedding,
            namespace="benchmark",
            target_k=target_k,
            token_budget=token_budget,
            compress=True,
        )
    )
    optimize_latency_ms = (time.perf_counter() - optimize_started_at) * 1000

    slim_context = [chunk.text for chunk in optimized.chunks]
    slim_answer, slim_answer_latency_ms = _answer_with_latency(
        answer_client,
        model,
        case.question,
        slim_context,
    )

    return CaseResult(
        id=case.id,
        baseline_input_tokens=sum(_count_tokens(chunk) for chunk in baseline_context),
        slimcontext_input_tokens=sum(_count_tokens(chunk) for chunk in slim_context),
        baseline_quality=judge.score(
            case.question,
            case.expected_answer,
            baseline_answer,
            baseline_context,
        ),
        slimcontext_quality=judge.score(
            case.question,
            case.expected_answer,
            slim_answer,
            slim_context,
        ),
        baseline_latency_ms=baseline_latency_ms,
        slimcontext_total_latency_ms=optimize_latency_ms + slim_answer_latency_ms,
        optimize_latency_ms=optimize_latency_ms,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a local SlimContext RAG benchmark.")
    parser.add_argument("--dataset", required=True, help="Path to a JSONL dataset.")
    parser.add_argument("--output", default="benchmarks/results/summary.json")
    parser.add_argument("--model", default="local-echo")
    parser.add_argument("--embedding-model", default="deterministic-hash")
    parser.add_argument("--target-k", type=int, default=8)
    parser.add_argument("--token-budget", type=int, default=1500)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    cases = load_jsonl_dataset(args.dataset)
    if args.limit:
        cases = cases[: args.limit]

    results = [
        run_case(
            case=case,
            answer_client=EchoAnswerClient(),
            judge=ExactSubstringJudge(),
            model=args.model,
            embedding_model_name=args.embedding_model,
            target_k=args.target_k,
            token_budget=args.token_budget,
        )
        for case in cases
    ]

    summary = summarize_results(results)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def _answer_with_latency(
    answer_client: AnswerClient,
    model: str,
    question: str,
    context: list[str],
) -> tuple[str, float]:
    started_at = time.perf_counter()
    result = answer_client.answer(model, question, context)
    fallback_latency_ms = (time.perf_counter() - started_at) * 1000

    if isinstance(result, tuple):
        return result

    return result, fallback_latency_ms


def _text_embedding(text: str, dimensions: int = 64) -> list[float]:
    vector = np.zeros(dimensions, dtype=np.float32)

    for token in text.lower().split():
        vector[hash(token) % dimensions] += 1.0

    norm = np.linalg.norm(vector)
    if norm > 0:
        vector = vector / norm

    return vector.tolist()


def _count_tokens(text: str) -> int:
    return len(text.split())


if __name__ == "__main__":
    main()

