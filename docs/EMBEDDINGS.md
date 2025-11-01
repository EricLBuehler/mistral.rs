# Embeddings Overview

Mistral.rs can load embedding models alongside chat, vision, diffusion, and speech workloads. Embedding models
produce dense vector representations that you can use for similarity search, clustering, reranking, and other
semantic tasks.

## Supported models

| Model | Notes | Documentation |
| --- | --- | --- |
| EmbeddingGemma | Google’s multilingual embedding model. | [EMBEDDINGGEMMA.md](EMBEDDINGGEMMA.md) |

> Have another embedding model you would like supported? Open an issue with the model ID and configuration.

## Usage overview

1. **Choose a model** from the table above.
2. **Load it through one of our APIs:**
   - CLI/HTTP
   - Python
   - Rust

Detailed examples for each model live in their dedicated documentation pages.
