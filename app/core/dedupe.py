import hashlib
import re

from datasketch import MinHash


Chunk = dict

NEAR_DUPLICATE_THRESHOLD = 0.8
SHINGLE_SIZE = 4
MINHASH_PERMUTATIONS = 128


def normalize_text(text: str) -> str:
    text = re.sub(r"[^\w\s]", " ", text.casefold())
    return re.sub(r"\s+", " ", text).strip()


def chunk_hash(text: str) -> str:
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


def remove_exact_duplicate_texts(items: list[str]) -> list[str]:
    seen_hashes: set[str] = set()
    unique_chunks: list[str] = []

    for chunk in items:
        digest = chunk_hash(chunk)

        if digest in seen_hashes:
            continue

        seen_hashes.add(digest)
        unique_chunks.append(chunk)

    return unique_chunks


def remove_exact_duplicate_chunks(chunks: list[Chunk], namespace: str) -> list[Chunk]:
    seen_hashes: set[str] = set()
    unique_chunks: list[Chunk] = []

    for chunk in chunks:
        digest = f"{namespace}:{chunk_hash(chunk['text'])}"

        if digest in seen_hashes:
            continue

        seen_hashes.add(digest)
        unique_chunks.append(chunk)

    return unique_chunks


def shingles(text: str, size: int = SHINGLE_SIZE) -> set[str]:
    words = normalize_text(text).split()

    if not words:
        return set()

    if len(words) <= size:
        return {" ".join(words)}

    return {" ".join(words[index : index + size]) for index in range(len(words) - size + 1)}


def minhash_for(text: str) -> MinHash:
    signature = MinHash(num_perm=MINHASH_PERMUTATIONS)

    for shingle in shingles(text):
        signature.update(shingle.encode("utf-8"))

    return signature


def remove_near_duplicate_texts(
    items: list[str], threshold: float = NEAR_DUPLICATE_THRESHOLD
) -> list[str]:
    kept_chunks: list[str] = []
    kept_signatures: list[MinHash] = []

    for chunk in items:
        signature = minhash_for(chunk)

        best_similarity = 0.0

        for kept_signature in kept_signatures:
            similarity = signature.jaccard(kept_signature)
            if similarity > best_similarity:
                best_similarity = similarity

        if best_similarity >= threshold:
            continue

        kept_chunks.append(chunk)
        kept_signatures.append(signature)

    return kept_chunks
