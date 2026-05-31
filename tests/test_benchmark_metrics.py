import json

import pytest

from benchmarks.metrics import (
    CaseResult,
    estimate_cost_per_1m_calls,
    load_jsonl_dataset,
    summarize_results,
    token_reduction_pct,
)


def test_cost_proxy_matches_readme_example() -> None:
    assert estimate_cost_per_1m_calls(8400) == 42.0
    assert estimate_cost_per_1m_calls(1380) == 6.9


def test_token_reduction_pct() -> None:
    assert token_reduction_pct(8400, 1380) == 83.57
    assert token_reduction_pct(0, 100) == 0.0


def test_load_jsonl_dataset(tmp_path) -> None:
    dataset_path = tmp_path / "dataset.jsonl"
    dataset_path.write_text(
        json.dumps(
            {
                "id": "case-1",
                "question": "What is SlimContext?",
                "expected_answer": "A context optimizer.",
                "source_title": "SlimContext",
                "chunks": ["SlimContext optimizes context."],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    cases = load_jsonl_dataset(dataset_path)

    assert len(cases) == 1
    assert cases[0].id == "case-1"
    assert cases[0].chunks == ["SlimContext optimizes context."]


def test_load_jsonl_dataset_rejects_bad_chunks(tmp_path) -> None:
    dataset_path = tmp_path / "dataset.jsonl"
    dataset_path.write_text(
        json.dumps({"id": "case-1", "question": "Q", "chunks": "not-a-list"}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="chunks must be a list"):
        load_jsonl_dataset(dataset_path)


def test_summarize_results() -> None:
    summary = summarize_results(
        [
            CaseResult(
                id="a",
                baseline_input_tokens=100,
                slimcontext_input_tokens=40,
                baseline_quality=7.0,
                slimcontext_quality=8.0,
                baseline_latency_ms=100,
                slimcontext_total_latency_ms=125,
                optimize_latency_ms=15,
            ),
            CaseResult(
                id="b",
                baseline_input_tokens=300,
                slimcontext_input_tokens=60,
                baseline_quality=6.0,
                slimcontext_quality=7.0,
                baseline_latency_ms=120,
                slimcontext_total_latency_ms=140,
                optimize_latency_ms=10,
            ),
        ]
    )

    assert summary["avg_input_tokens_without"] == 200
    assert summary["avg_input_tokens_with"] == 50
    assert summary["token_reduction_pct"] == 75.0
    assert summary["avg_quality_without"] == 6.5
    assert summary["avg_quality_with"] == 7.5
    assert summary["pipeline_latency_added_ms"] == 22.5
