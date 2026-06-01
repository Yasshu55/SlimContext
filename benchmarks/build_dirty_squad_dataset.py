from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


NOISE_POOL = [
    "2026-05-31 17:00:00 [ERROR] HTTP 500 Internal Server Error in core.clustering",
    "2026-05-31 17:01:15 [WARN] HikariPool-1 - Connection leak detected.",
    "The quick brown fox jumps over the lazy dog checking system properties.",
    "Going to the gym regularly can dramatically improve physical endurance levels.",
    "Low-level design components like Singletons or Strategies keep code modular.",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a dirty 100-case SQuAD-style JSON dataset."
    )
    parser.add_argument("--output", default="benchmarks/data/dirty_test_set.json")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cases = build_cases(count=args.count, seed=args.seed)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(cases, indent=2), encoding="utf-8")
    print(f"Saved {len(cases)} messy test cases to {output_path}")


def build_cases(count: int, seed: int) -> list[dict]:
    random.seed(seed)

    samples = _load_squad_samples(count=count, seed=seed)
    test_cases = []

    for index, item in enumerate(samples, start=1):
        true_context = item["context"]
        chunks = [
            {"id": "truth", "text": true_context, "score": 0.95},
            {"id": "exact_duplicate", "text": true_context, "score": 0.95},
        ]

        for noise_index, noise in enumerate(random.sample(NOISE_POOL, 3)):
            chunks.append(
                {
                    "id": f"noise_{noise_index}",
                    "text": noise,
                    "score": round(random.uniform(0.1, 0.4), 3),
                }
            )

        random.shuffle(chunks)
        test_cases.append(
            {
                "id": index,
                "query": item["question"],
                "ground_truth_answer": item["answer"],
                "dirty_chunks": chunks,
            }
        )

    return test_cases


def _load_squad_samples(count: int, seed: int) -> list[dict]:
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise SystemExit(
            "The Hugging Face datasets package is required for SQuAD generation. "
            "Install it with: python -m pip install datasets"
        ) from error

    try:
        squad = load_dataset("squad", split="validation")
    except Exception:
        squad = load_dataset("rajpurkar/squad", split="validation")
    samples = squad.shuffle(seed=seed).select(range(count))

    return [
        {
            "question": item["question"],
            "context": item["context"],
            "answer": item["answers"]["text"][0] if item["answers"]["text"] else "",
        }
        for item in samples
    ]


if __name__ == "__main__":
    main()
