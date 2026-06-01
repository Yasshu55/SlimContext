from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any


COST_PER_1K_INPUT_TOKENS = 0.000005


@dataclass(frozen=True)
class BenchmarkCase:
    id: str
    question: str
    chunks: list[str]
    expected_answer: str = ""
    source_title: str = ""


@dataclass(frozen=True)
class CaseResult:
    id: str
    baseline_input_tokens: int
    slimcontext_input_tokens: int
    baseline_quality: float
    slimcontext_quality: float
    baseline_latency_ms: float
    slimcontext_total_latency_ms: float
    optimize_latency_ms: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def estimate_cost_per_1m_calls(avg_input_tokens: float) -> float:
    return round(avg_input_tokens * 1_000_000 / 1000 * COST_PER_1K_INPUT_TOKENS, 2)


def token_reduction_pct(before_tokens: int | float, after_tokens: int | float) -> float:
    if before_tokens <= 0:
        return 0.0

    return round(((before_tokens - after_tokens) / before_tokens) * 100, 2)


def load_jsonl_dataset(path: str | Path) -> list[BenchmarkCase]:
    cases: list[BenchmarkCase] = []

    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue

            raw = json.loads(line)
            chunks = raw.get("chunks", [])
            if not isinstance(chunks, list):
                raise ValueError(f"line {line_number}: chunks must be a list")

            cases.append(
                BenchmarkCase(
                    id=str(raw.get("id", f"case-{line_number}")),
                    question=str(raw.get("question", "")),
                    expected_answer=str(raw.get("expected_answer", "")),
                    source_title=str(raw.get("source_title", "")),
                    chunks=[str(chunk) for chunk in chunks],
                )
            )

    return cases


def summarize_results(results: list[CaseResult]) -> dict[str, Any]:
    if not results:
        return {
            "case_count": 0,
            "avg_input_tokens_without": 0,
            "avg_input_tokens_with": 0,
            "token_reduction_pct": 0.0,
            "avg_quality_without": 0.0,
            "avg_quality_with": 0.0,
            "pipeline_latency_added_ms": 0.0,
            "cost_per_1m_calls_without": 0.0,
            "cost_per_1m_calls_with": 0.0,
            "cases": [],
        }

    avg_without = mean(result.baseline_input_tokens for result in results)
    avg_with = mean(result.slimcontext_input_tokens for result in results)
    avg_baseline_latency = mean(result.baseline_latency_ms for result in results)
    avg_slimcontext_latency = mean(
        result.slimcontext_total_latency_ms for result in results
    )

    return {
        "case_count": len(results),
        "avg_input_tokens_without": round(avg_without, 2),
        "avg_input_tokens_with": round(avg_with, 2),
        "token_reduction_pct": token_reduction_pct(avg_without, avg_with),
        "avg_quality_without": round(
            mean(result.baseline_quality for result in results),
            2,
        ),
        "avg_quality_with": round(
            mean(result.slimcontext_quality for result in results),
            2,
        ),
        "pipeline_latency_added_ms": round(
            avg_slimcontext_latency - avg_baseline_latency,
            2,
        ),
        "avg_optimize_latency_ms": round(
            mean(result.optimize_latency_ms for result in results),
            2,
        ),
        "cost_per_1m_calls_without": estimate_cost_per_1m_calls(avg_without),
        "cost_per_1m_calls_with": estimate_cost_per_1m_calls(avg_with),
        "cases": [result.to_dict() for result in results],
    }

