from chunks import chunks
import hashlib
import re

from datasketch import MinHash


NEAR_DUPLICATE_THRESHOLD = 0.8
SHINGLE_SIZE = 4
MINHASH_PERMUTATIONS = 128


def normalize_text(text: str) -> str:
    text = re.sub(r"[^\w\s]", " ", text.casefold())
    return re.sub(r"\s+", " ", text).strip()


def chunk_hash(text: str) -> str:
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


def remove_exact_duplicates(items: list[str]) -> list[str]:
    seen_hashes: dict[str, int] = {}
    unique_chunks: list[str] = []

    for index, chunk in enumerate(items):
        digest = chunk_hash(chunk)

        if digest in seen_hashes:
            continue

        seen_hashes[digest] = index
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


def remove_near_duplicates(
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


def main() -> None:
    print(f"Loaded {len(chunks)} chunks.")

    exact_unique_chunks = remove_exact_duplicates(chunks)
    deduplicated_chunks = remove_near_duplicates(exact_unique_chunks)

    print(f"After exact hash deduplication: {len(exact_unique_chunks)} chunks.")
    print(f"After near-duplicate removal: {len(deduplicated_chunks)} chunks.")

    print("\nFinal chunks:")
    for index, chunk in enumerate(deduplicated_chunks):
        print(f"{index + 1}. {chunk}")


if __name__ == "__main__":
    main()
