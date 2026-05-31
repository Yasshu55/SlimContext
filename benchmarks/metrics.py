from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Iterable


TOKEN_PROXY_USD_PER_1K = 0.000005


@dataclass(frozen=True)
class BenchmarkCase:
    id: str
    question: str
    expected_answer: str
    chunks: list[str]
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


def load_jsonl_dataset(path: str | Path) -> list[BenchmarkCase]:
    cases: list[BenchmarkCase] = []

    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            chunks = payload.get("chunks")
            if not isinstance(chunks, list) or not all(isinstance(item, str) for item in chunks):
                raise ValueError(f"Line {line_number}: chunks must be a list of strings.")

            cases.append(
                BenchmarkCase(
                    id=str(payload["id"]),
                    question=str(payload["question"]),
                    expected_answer=str(payload.get("expected_answer", "")),
                    chunks=chunks,
                    source_title=str(payload.get("source_title", "")),
                )
            )

    if not cases:
        raise ValueError(f"No benchmark cases found in {path}.")

    return cases


def estimate_cost_per_1m_calls(avg_input_tokens: float, usd_per_1k: float = TOKEN_PROXY_USD_PER_1K) -> float:
    return round((avg_input_tokens * 1_000_000 / 1000) * usd_per_1k, 2)


def token_reduction_pct(without_tokens: float, with_tokens: float) -> float:
    if without_tokens <= 0:
        return 0.0
    return round(((without_tokens - with_tokens) / without_tokens) * 100, 2)


def summarize_results(results: Iterable[CaseResult]) -> dict:
    rows = list(results)
    if not rows:
        raise ValueError("Cannot summarize an empty benchmark result set.")

    avg_without_tokens = mean(row.baseline_input_tokens for row in rows)
    avg_with_tokens = mean(row.slimcontext_input_tokens for row in rows)
    avg_baseline_latency = mean(row.baseline_latency_ms for row in rows)
    avg_with_latency = mean(row.slimcontext_total_latency_ms for row in rows)

    return {
        "case_count": len(rows),
        "avg_input_tokens_without": round(avg_without_tokens, 2),
        "avg_input_tokens_with": round(avg_with_tokens, 2),
        "token_reduction_pct": token_reduction_pct(avg_without_tokens, avg_with_tokens),
        "avg_quality_without": round(mean(row.baseline_quality for row in rows), 2),
        "avg_quality_with": round(mean(row.slimcontext_quality for row in rows), 2),
        "cost_per_1m_without_usd": estimate_cost_per_1m_calls(avg_without_tokens),
        "cost_per_1m_with_usd": estimate_cost_per_1m_calls(avg_with_tokens),
        "pipeline_latency_added_ms": round(avg_with_latency - avg_baseline_latency, 2),
        "avg_optimize_latency_ms": round(mean(row.optimize_latency_ms for row in rows), 2),
        "avg_baseline_latency_ms": round(avg_baseline_latency, 2),
        "avg_slimcontext_total_latency_ms": round(avg_with_latency, 2),
    }


def write_summary_outputs(summary: dict, results: list[CaseResult], output_dir: str | Path) -> None:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    (destination / "summary.json").write_text(
        json.dumps({"summary": summary, "cases": [row.__dict__ for row in results]}, indent=2),
        encoding="utf-8",
    )

    with (destination / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "without_slimcontext", "with_slimcontext"])
        writer.writerow(["Avg input tokens", summary["avg_input_tokens_without"], summary["avg_input_tokens_with"]])
        writer.writerow(["Token reduction", "-", f'{summary["token_reduction_pct"]}%'])
        writer.writerow(
            [
                "Answer quality (LLM judge 1-10)",
                summary["avg_quality_without"],
                summary["avg_quality_with"],
            ]
        )
        writer.writerow(
            [
                "Cost per 1M calls (Ollama Qwen token proxy)",
                f'${summary["cost_per_1m_without_usd"]:.2f}',
                f'${summary["cost_per_1m_with_usd"]:.2f}',
            ]
        )
        writer.writerow(["Pipeline latency added", "0ms", f'{summary["pipeline_latency_added_ms"]}ms'])


def render_chart(summary: dict, output_path: str | Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("Install matplotlib to render benchmark_chart.png.") from error

    labels = ["Input tokens", "Quality", "Cost"]
    without = [
        summary["avg_input_tokens_without"],
        summary["avg_quality_without"],
        summary["cost_per_1m_without_usd"],
    ]
    with_slim = [
        summary["avg_input_tokens_with"],
        summary["avg_quality_with"],
        summary["cost_per_1m_with_usd"],
    ]

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    colors = ["#6b7280", "#2563eb"]

    for axis, label, baseline, optimized in zip(axes, labels, without, with_slim):
        axis.bar(["Without", "With"], [baseline, optimized], color=colors)
        axis.set_title(label)
        axis.grid(axis="y", alpha=0.25)
        for index, value in enumerate([baseline, optimized]):
            axis.text(index, value, f"{value:.2f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle("SlimContext benchmark")
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
