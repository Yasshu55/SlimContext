from __future__ import annotations

import argparse
import json
import re
import textwrap
import urllib.parse
import urllib.request
from pathlib import Path


DEFAULT_TITLES = [
    "Artificial intelligence",
    "Machine learning",
    "Natural language processing",
    "Information retrieval",
    "Search engine",
    "Knowledge graph",
    "Data compression",
    "Vector space model",
    "Question answering",
    "Transformer (deep learning architecture)",
    "Large language model",
    "Retrieval-augmented generation",
    "Document classification",
    "Cluster analysis",
    "Cosine similarity",
    "PageRank",
    "Database index",
    "Distributed computing",
    "Computer security",
    "Cloud computing",
]


def fetch_wikipedia_extract(title: str) -> str:
    params = urllib.parse.urlencode(
        {
            "action": "query",
            "format": "json",
            "prop": "extracts",
            "explaintext": "1",
            "redirects": "1",
            "titles": title,
        }
    )
    request = urllib.request.Request(
        f"https://en.wikipedia.org/w/api.php?{params}",
        headers={"User-Agent": "SlimContextBenchmark/0.1 (https://github.com/)"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))

    pages = payload["query"]["pages"]
    page = next(iter(pages.values()))
    extract = page.get("extract", "").strip()
    if not extract:
        raise ValueError(f"No extract returned for Wikipedia title: {title}")
    return extract


def chunk_text(text: str, max_chars: int) -> list[str]:
    paragraphs = [part.strip() for part in re.split(r"\n{2,}", text) if part.strip()]
    chunks: list[str] = []

    for paragraph in paragraphs:
        if len(paragraph) <= max_chars:
            chunks.append(paragraph)
            continue
        chunks.extend(textwrap.wrap(paragraph, width=max_chars, break_long_words=False))

    return chunks


def build_questions(title: str, chunks: list[str], questions_per_page: int) -> list[dict]:
    cases = []
    usable_chunks = [chunk for chunk in chunks if len(chunk.split()) >= 30]

    for index, chunk in enumerate(usable_chunks[:questions_per_page], start=1):
        question = (
            f"According to the Wikipedia article, what should a reader know about "
            f"{title} from source passage {index}?"
        )
        cases.append(
            {
                "id": f"{slugify(title)}-{index}",
                "source_title": title,
                "question": question,
                "expected_answer": first_sentences(chunk, sentence_count=2),
                "chunks": chunks,
            }
        )

    return cases


def first_sentences(text: str, sentence_count: int) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return " ".join(sentences[:sentence_count]).strip()


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a reproducible Wikipedia JSONL benchmark dataset.")
    parser.add_argument("--pages", type=int, default=20)
    parser.add_argument("--questions-per-page", type=int, default=3)
    parser.add_argument("--chunk-chars", type=int, default=900)
    parser.add_argument("--output", default="benchmarks/data/wikipedia_eval.jsonl")
    parser.add_argument("--titles", nargs="*", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    titles = (args.titles or DEFAULT_TITLES)[: args.pages]
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    case_count = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for title in titles:
            extract = fetch_wikipedia_extract(title)
            chunks = chunk_text(extract, args.chunk_chars)
            for case in build_questions(title, chunks, args.questions_per_page):
                handle.write(json.dumps(case, ensure_ascii=False) + "\n")
                case_count += 1

    print(f"Wrote {case_count} benchmark cases to {output_path}")


if __name__ == "__main__":
    main()
