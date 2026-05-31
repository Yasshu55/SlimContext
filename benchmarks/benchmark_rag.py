from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from functools import lru_cache
from pathlib import Path
from typing import Protocol

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.api import ChunkIn, OptimizeRequest, _count_text_tokens, optimize
from benchmarks.metrics import (
    BenchmarkCase,
    CaseResult,
    load_jsonl_dataset,
    render_chart,
    summarize_results,
    write_summary_outputs,
)


class AnswerClient(Protocol):
    def answer(self, model: str, question: str, context: list[str]) -> tuple[str, float]:
        ...


class Judge(Protocol):
    def score(self, question: str, expected_answer: str, actual_answer: str, context: list[str]) -> float:
        ...


class OllamaAnswerClient:
    def __init__(self, base_url: str, temperature: float = 0.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature

    def answer(self, model: str, question: str, context: list[str]) -> tuple[str, float]:
        prompt = build_answer_prompt(question, context)
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": self.temperature},
        }
        started_at = time.perf_counter()
        request = urllib.request.Request(
            f"{self.base_url}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as error:
            raise RuntimeError(f"Could not call Ollama at {self.base_url}. Is Ollama running?") from error

        latency_ms = (time.perf_counter() - started_at) * 1000
        return str(body.get("response", "")).strip(), latency_ms


class DeepEvalOllamaJudge:
    def __init__(self, model: str, base_url: str) -> None:
        try:
            from deepeval.metrics import GEval
            from deepeval.models.base_model import DeepEvalBaseLLM
            from deepeval.test_case import LLMTestCase, SingleTurnParams
        except ImportError as error:
            raise RuntimeError("Install deepeval to use the LLM judge.") from error

        class OllamaJudgeModel(DeepEvalBaseLLM):
            def __init__(self, model_name: str, ollama_url: str) -> None:
                self.model_name = model_name
                self.ollama_url = ollama_url.rstrip("/")

            def load_model(self):
                return None

            def generate(self, prompt: str) -> str:
                payload = {"model": self.model_name, "prompt": prompt, "stream": False}
                request = urllib.request.Request(
                    f"{self.ollama_url}/api/generate",
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(request, timeout=180) as response:
                    body = json.loads(response.read().decode("utf-8"))
                return str(body.get("response", "")).strip()

            async def a_generate(self, prompt: str) -> str:
                return self.generate(prompt)

            def get_model_name(self) -> str:
                return f"ollama/{self.model_name}"

        self._test_case_cls = LLMTestCase
        self.metric = GEval(
            name="SlimContext answer quality",
            criteria=(
                "Score whether the actual answer is relevant to the question, factually supported "
                "by the retrieval context, and covers the expected answer. Penalize hallucinations. "
                "Return a calibrated quality score."
            ),
            evaluation_params=[
                SingleTurnParams.INPUT,
                SingleTurnParams.ACTUAL_OUTPUT,
                SingleTurnParams.EXPECTED_OUTPUT,
                SingleTurnParams.RETRIEVAL_CONTEXT,
            ],
            model=OllamaJudgeModel(model, base_url),
        )

    def score(self, question: str, expected_answer: str, actual_answer: str, context: list[str]) -> float:
        test_case = self._test_case_cls(
            input=question,
            actual_output=actual_answer,
            expected_output=expected_answer,
            retrieval_context=context,
        )
        self.metric.measure(test_case)
        score = float(self.metric.score or 0.0)
        return round(score * 10 if score <= 1 else score, 2)


def build_answer_prompt(question: str, context: list[str]) -> str:
    joined_context = "\n\n".join(f"[{index}] {chunk}" for index, chunk in enumerate(context, start=1))
    return (
        "Answer the question using only the provided context. "
        "If the context is insufficient, say what is missing.\n\n"
        f"Question: {question}\n\nContext:\n{joined_context}\n\nAnswer:"
    )


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
    baseline_answer, baseline_latency_ms = answer_client.answer(model, case.question, baseline_context)

    chunks, query_embedding = embed_case(case, embedding_model_name)
    optimize_started_at = time.perf_counter()
    optimized = optimize(
        OptimizeRequest(
            chunks=chunks,
            query=case.question,
            query_embedding=query_embedding,
            token_budget=token_budget,
            target_k=target_k,
            namespace="benchmark",
            compress=True,
        )
    )
    optimize_latency_ms = (time.perf_counter() - optimize_started_at) * 1000
    optimized_context = [chunk.text for chunk in optimized.chunks]
    slim_answer, slim_answer_latency_ms = answer_client.answer(model, case.question, optimized_context)

    return CaseResult(
        id=case.id,
        baseline_input_tokens=sum(_count_text_tokens(chunk) for chunk in baseline_context),
        slimcontext_input_tokens=sum(_count_text_tokens(chunk) for chunk in optimized_context),
        baseline_quality=judge.score(case.question, case.expected_answer, baseline_answer, baseline_context),
        slimcontext_quality=judge.score(case.question, case.expected_answer, slim_answer, optimized_context),
        baseline_latency_ms=baseline_latency_ms,
        slimcontext_total_latency_ms=optimize_latency_ms + slim_answer_latency_ms,
        optimize_latency_ms=optimize_latency_ms,
    )


def embed_case(case: BenchmarkCase, embedding_model_name: str) -> tuple[list[ChunkIn], list[float]]:
    model = get_embedding_model(embedding_model_name)
    embeddings = model.encode(case.chunks)
    query_embedding = model.encode([case.question])[0].tolist()
    chunks = [
        ChunkIn(id=f"{case.id}-chunk-{index}", text=text, embedding=embedding.tolist(), score=1.0)
        for index, (text, embedding) in enumerate(zip(case.chunks, embeddings), start=1)
    ]
    return chunks, query_embedding


@lru_cache(maxsize=2)
def get_embedding_model(embedding_model_name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(embedding_model_name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark SlimContext token, cost, quality, and latency impact.")
    parser.add_argument("--dataset", default="benchmarks/data/wikipedia_eval.jsonl")
    parser.add_argument("--model", default="qwen2.5:7b")
    parser.add_argument("--judge-model", default="qwen2.5:7b")
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--embedding-model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--target-k", type=int, default=8)
    parser.add_argument("--token-budget", type=int, default=1500)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-dir", default="benchmarks/results")
    parser.add_argument("--skip-chart", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases = load_jsonl_dataset(args.dataset)
    if args.limit is not None:
        cases = cases[: args.limit]

    answer_client = OllamaAnswerClient(args.ollama_url)
    judge = DeepEvalOllamaJudge(args.judge_model, args.ollama_url)
    results = [
        run_case(
            case=case,
            answer_client=answer_client,
            judge=judge,
            model=args.model,
            embedding_model_name=args.embedding_model,
            target_k=args.target_k,
            token_budget=args.token_budget,
        )
        for case in cases
    ]

    summary = summarize_results(results)
    write_summary_outputs(summary, results, args.output_dir)
    if not args.skip_chart:
        render_chart(summary, Path(args.output_dir) / "benchmark_chart.png")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
