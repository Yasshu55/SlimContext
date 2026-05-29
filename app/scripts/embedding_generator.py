from sentence_transformers import SentenceTransformer

from app.data.chunks import chunks


def main() -> None:
    model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    embeddings = model.encode(chunks)

    print(f"Generated embeddings for {len(chunks)} chunks.")
    print(f"Embedding matrix shape: {embeddings.shape}")
    print(embeddings)


if __name__ == "__main__":
    main()
