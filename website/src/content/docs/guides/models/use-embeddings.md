---
title: Use embedding models
description: Generate dense vector representations of text with EmbeddingGemma, Qwen3-Embedding, and others.
sidebar:
  order: 4
---

Embedding models map text to dense vectors for semantic search, reranking, clustering, and downstream retrieval. mistral.rs serves embeddings through the standard OpenAI `POST /v1/embeddings` endpoint, so any tool that already targets that endpoint (LangChain, LlamaIndex, vector stores) works unchanged.

## Loading an embedding model

Two regularly tested options:

- `google/embeddinggemma-300m`: small, fast, 768-dim. Good general-purpose default.
- `Qwen/Qwen3-Embedding-0.6B`: larger, higher-quality, higher cost.

```bash
mistralrs serve -m google/embeddinggemma-300m
```

Embedding models run fast on most hardware. A CPU-only install handles thousands of embeddings per minute on a small model.

## Requesting an embedding

```bash
curl http://localhost:1234/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "input": "The cat sat on the mat."
  }'
```

The response includes the vector in `embedding`:

```json
{
  "object": "list",
  "data": [{
    "object": "embedding",
    "index": 0,
    "embedding": [0.123, -0.456, 0.789, ...]
  }],
  "model": "default",
  "usage": {"prompt_tokens": 7, "total_tokens": 7}
}
```

## Batching

Pass a list of strings to embed many at once:

```bash
curl http://localhost:1234/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "input": [
      "The cat sat on the mat.",
      "A dog chased a squirrel.",
      "A raven croaked from the fencepost."
    ]
  }'
```

The `data` array has one entry per input in input order.

## Normalization

mistral.rs returns vectors as the model produces them. Normalize on the client when cosine similarity is computed as dot product.

To normalize in Python:

```python
import numpy as np

v = np.array(response["data"][0]["embedding"])
v_normalized = v / np.linalg.norm(v)
```

Many vector stores (FAISS, pgvector) handle normalization internally.

## Using the vectors

Standard pipeline:

1. Embed a corpus offline and store vectors in a vector database (FAISS, Qdrant, pgvector, Pinecone).
2. At query time, embed the query with the same model.
3. Search the store for nearest neighbors.
4. Optionally rerank top results with a reranker.
5. Feed retrieved documents as language model context.

mistral.rs handles steps 1, 2, and (with a reranker) 4. The rest is the vector store and application logic. The [web search guide](/mistral.rs/guides/agents/web-search/) covers using embeddings to rerank search results within an agent.
