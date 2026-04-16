---
title: Use embedding models
description: Generate dense vector representations of text with EmbeddingGemma, Qwen3-Embedding, and others.
sidebar:
  order: 4
---

Embedding models turn text into dense vectors. The vectors are useful for semantic search, reranking, clustering, and as inputs to downstream retrieval systems. mistral.rs supports embedding models through the standard OpenAI `POST /v1/embeddings` endpoint, so any tool that already talks to that endpoint (LangChain, LlamaIndex, custom vector stores) works unchanged.

## Loading an embedding model

Two embedding models we regularly test:

- `google/embeddinggemma-300m`: a small, fast model that produces 768-dimensional vectors. Good all-around choice.
- `Qwen/Qwen3-Embedding-0.6B`: a larger model that produces higher-quality embeddings, at higher cost.

```bash
mistralrs serve -m google/embeddinggemma-300m
```

Dedicated embedding models run fast on almost any hardware. A CPU-only install will serve thousands of embeddings per minute for a small model.

## Requesting an embedding

```bash
curl http://localhost:1234/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "input": "The cat sat on the mat."
  }'
```

The response includes the vector in the `embedding` field:

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

You can pass a list of strings to embed many at once. Batching is much faster than making separate requests because the model processes all inputs in parallel:

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

The response's `data` array has one entry per input, in the same order as the inputs.

## Normalization

Embeddings are usually used with cosine similarity, which is equivalent to dot product when vectors are L2-normalized. Some downstream systems expect normalized vectors; some normalize on their own. mistralrs returns vectors in whatever form the model produces them (usually unnormalized).

To normalize in Python:

```python
import numpy as np

v = np.array(response["data"][0]["embedding"])
v_normalized = v / np.linalg.norm(v)
```

If you are using the OpenAI client and the vectors are going into a FAISS index or pgvector table, the receiving system usually does the right thing without manual normalization.

## Reranking

Some embedding models are trained specifically for reranking (given a query and a list of candidates, produce relevance scores). Those use the same embedding endpoint with a slightly different request shape:

```json
{
  "model": "default",
  "input": ["query"],
  "instruction": "Retrieve relevant passages for the query."
}
```

Not every embedding model supports instructions. Check the model card for which ones do.

## Sizing vectors

Embedding dimensions trade off storage cost and retrieval quality. The default dimension of each model is what its card documents. Some models also support Matryoshka-style truncation, where you can use a prefix of the vector for a cheap-and-fast index at the cost of a small quality drop.

EmbeddingGemma supports truncation down to 512, 256, 128, or 64 dimensions. If your index storage is the bottleneck, this is an easy win. Pass `dimensions` in the request:

```json
{
  "model": "default",
  "input": "...",
  "dimensions": 256
}
```

Qwen3-Embedding does not support truncation; its output is always the full dimensionality.

## What to do with the vectors

The usual pipeline is:

1. Embed a corpus offline and store the vectors in a vector database (FAISS, Qdrant, pgvector, Pinecone, or similar).
2. At query time, embed the query with the same model.
3. Search the vector store for nearest neighbors of the query vector.
4. Optionally rerank the top results with a reranking model.
5. Feed the retrieved documents as context into a language model.

mistralrs plays the "embed" role in steps 1 and 2 (and step 4 if you are reranking with an embedding model). The rest is handled by your vector store and application logic. The [web search guide](/mistral.rs/guides/agents/web-search/) covers how to use embeddings specifically for reranking search results within an agent.
