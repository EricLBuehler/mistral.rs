from mistralrs import EmbeddingArchitecture, EmbeddingRequest, Runner, Which


def main() -> None:
    runner = Runner(
        which=Which.Embedding(
            model_id="Qwen/Qwen3-Embedding-0.6B",
            arch=EmbeddingArchitecture.Qwen3Embedding,
        ),
    )

    request = EmbeddingRequest(
        input=[
            "Graphene conductivity",
            "Explain superconductors in simple terms.",
        ],
        truncate_sequence=True,
    )

    embeddings = runner.send_embedding_request(request)

    for index, embedding in enumerate(embeddings):
        print(f"Embedding {index}: {len(embedding)} dimensions")


if __name__ == "__main__":
    main()
