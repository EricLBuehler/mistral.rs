from mistralrs import EmbeddingRequest, Runner, Which, EmbeddingArchitecture


def main() -> None:
    runner = Runner(
        which=Which.Embedding(
            model_id="google/embeddinggemma-300m",
            arch=EmbeddingArchitecture.EmbeddingGemma,
        ),
    )

    request = EmbeddingRequest(
        input=[
            "task: search result | query: What is graphene?",
            "task: search result | query: Explain superconductors in simple terms.",
        ],
        truncate_sequence=True,
    )

    embeddings = runner.send_embedding_request(request)

    for index, embedding in enumerate(embeddings):
        print(f"Embedding {index}: {len(embedding)} dimensions")


if __name__ == "__main__":
    main()
