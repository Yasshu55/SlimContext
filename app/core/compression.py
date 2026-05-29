import json
import re


Chunk = dict


FILLER_PATTERNS = [
    r"\bplease note that\b",
    r"\bit is important to note that\b",
    r"\bin order to\b",
    r"\bas a reminder\b",
]


def compress_chunks(chunks: list[Chunk]) -> list[Chunk]:
    return [_compress_chunk(chunk) for chunk in chunks]


def _compress_chunk(chunk: Chunk) -> Chunk:
    compressed = dict(chunk)
    text = compressed.get("text", "")

    if _looks_structured(text):
        compressed["text"] = _placeholder_structured_text(text)
    else:
        compressed["text"] = _prune_text(text)

    return compressed


def _prune_text(text: str) -> str:
    pruned = text

    for pattern in FILLER_PATTERNS:
        pruned = re.sub(pattern, "", pruned, flags=re.IGNORECASE)

    return re.sub(r"\s+", " ", pruned).strip()


def _looks_structured(text: str) -> bool:
    stripped = text.strip()

    if not stripped:
        return False

    if stripped.startswith(("{", "[")):
        try:
            json.loads(stripped)
            return True
        except json.JSONDecodeError:
            pass

    return (
        stripped.startswith("<")
        or "|" in stripped
        or stripped.count("\t") >= 2
        or stripped.count(",") >= 6
    )


def _placeholder_structured_text(text: str) -> str:
    stripped = text.strip()

    if len(stripped) <= 400:
        return stripped

    return f"{stripped[:220]} ... [structured content truncated] ... {stripped[-120:]}"
