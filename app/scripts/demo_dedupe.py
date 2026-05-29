from app.core.dedupe import (
    remove_exact_duplicate_texts,
    remove_near_duplicate_texts,
)
from app.data.chunks import chunks


def main() -> None:
    print(f"Loaded {len(chunks)} chunks.")

    exact_unique_chunks = remove_exact_duplicate_texts(chunks)
    deduplicated_chunks = remove_near_duplicate_texts(exact_unique_chunks)

    print(f"After exact hash deduplication: {len(exact_unique_chunks)} chunks.")
    print(f"After near-duplicate removal: {len(deduplicated_chunks)} chunks.")

    print("\nFinal chunks:")
    for index, chunk in enumerate(deduplicated_chunks):
        print(f"{index + 1}. {chunk}")


if __name__ == "__main__":
    main()
