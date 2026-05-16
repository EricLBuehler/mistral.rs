---
title: Use embedding models
description: Generate dense vector representations of text with EmbeddingGemma, Qwen3-Embedding, and others.
sidebar:
  order: 5
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

Use `Qwen/Qwen3-Embedding-0.6B` the same way:

```bash
mistralrs serve -m Qwen/Qwen3-Embedding-0.6B
```

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

## EmbeddingGemma prompts

EmbeddingGemma works best when the input is prefixed for the task:

| Use case | Prompt form |
|---|---|
| Retrieval query | `task: search result \| query: <query>` |
| Retrieval document | `title: <title or none> \| text: <document>` |
| Question answering | `task: question answering \| query: <question>` |
| Fact verification | `task: fact checking \| query: <claim>` |
| Classification | `task: classification \| query: <text>` |
| Clustering | `task: clustering \| query: <text>` |
| Semantic similarity | `task: sentence similarity \| query: <text>` |
| Code retrieval | `task: code retrieval \| query: <query>` |

Example:

```bash
curl http://localhost:1234/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "input": [
      "task: search result | query: What is graphene?",
      "title: none | text: Graphene is a single layer of carbon atoms."
    ]
  }'
```

Qwen3-Embedding does not require these prefixes, but task-specific prefixes can still help keep a retrieval system consistent.

## Python SDK

EmbeddingGemma:

```python
from mistralrs import EmbeddingArchitecture, EmbeddingRequest, Runner, Which

runner = Runner(
    which=Which.Embedding(
        model_id="google/embeddinggemma-300m",
        arch=EmbeddingArchitecture.EmbeddingGemma,
    )
)

embeddings = runner.send_embedding_request(
    EmbeddingRequest(
        input=[
            "task: search result | query: What is graphene?",
            "task: search result | query: What is an apple?",
        ],
        truncate_sequence=True,
    )
)
print(len(embeddings), len(embeddings[0]))
```

Qwen3-Embedding:

```python
from mistralrs import EmbeddingArchitecture, EmbeddingRequest, Runner, Which

runner = Runner(
    which=Which.Embedding(
        model_id="Qwen/Qwen3-Embedding-0.6B",
        arch=EmbeddingArchitecture.Qwen3Embedding,
    )
)

embeddings = runner.send_embedding_request(
    EmbeddingRequest(
        input=["Graphene conductivity", "Explain superconductors in simple terms."],
        truncate_sequence=True,
    )
)
print(len(embeddings), len(embeddings[0]))
```

## Using the vectors

Standard pipeline:

1. Embed a corpus offline and store vectors in a vector database (FAISS, Qdrant, pgvector, Pinecone).
2. At query time, embed the query with the same model.
3. Search the store for nearest neighbors.
4. Optionally rerank top results with a reranker.
5. Feed retrieved documents as language model context.

mistral.rs handles steps 1, 2, and (with a reranker) 4. The rest is the vector store and application logic. The [web search guide](/mistral.rs/guides/agents/web-search/) covers using embeddings to rerank search results within an agent.
