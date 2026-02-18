# EmbeddingGemma

EmbeddingGemma was the first embedding model supported by mistral.rs. This guide walks through serving the
model via the OpenAI-compatible HTTP server, running it from Python, and embedding text directly in Rust.

For a catalog of available embedding models and general usage tips, see [EMBEDDINGS.md](EMBEDDINGS.md).

## Prompt instructions

EmbeddingGemma can generate optimized embeddings for various use cases-such as document retrieval, question answering, and fact verification-or for specific input types, either, a query or a document-using prompts that are prepended to the input strings. 

- Query prompts follow the form `task: {task description} | query: ` where the task description varies by the use case, with the default task description being search result. 
- Document-style prompts follow the form `title: {title | "none"} | text: ` where the title is either none (the default) or the actual title of the document. Note that providing a title, if available, will improve model performance for document prompts but may require manual formatting.

| **Use Case (task type enum)** | **Descriptions** | **Recommended Prompt** |
|-------------------------------|------------------|-------------------------|
| **Retrieval (Query)** | Used to generate embeddings that are optimized for document search or information retrieval. | `task: search result \| query: {content}` |
| **Retrieval (Document)** | Used to generate embeddings that are optimized for document search or information retrieval (document side). | `title: {title \| "none"} \| text: {content}` |
| **Question Answering** | Used to generate embeddings that are optimized for answering natural language questions. | `task: question answering \| query: {content}` |
| **Fact Verification** | Used to generate embeddings that are optimized for verifying factual correctness. | `task: fact checking \| query: {content}` |
| **Classification** | Used to generate embeddings that are optimized to classify texts according to preset labels. | `task: classification \| query: {content}` |
| **Clustering** | Used to generate embeddings that are optimized to cluster texts based on their similarities. | `task: clustering \| query: {content}` |
| **Semantic Similarity** | Used to generate embeddings that are optimized to assess text similarity. This is not intended for retrieval use cases. | `task: sentence similarity \| query: {content}` |
| **Code Retrieval** | Used to retrieve a code block based on a natural language query, such as *sort an array* or *reverse a linked list*. Embeddings of code blocks are computed using `retrieval_document`. | `task: code retrieval \| query: {content}` |


## HTTP server

Launch the server in embedding mode to expose an OpenAI-compatible `/v1/embeddings` endpoint:

```bash
mistralrs serve -p 1234 -m google/embeddinggemma-300m
```

Once running, call the endpoint with an OpenAI client or raw `curl`:

```bash
curl http://localhost:1234/v1/embeddings \
  -H "Authorization: Bearer EMPTY" \
  -H "Content-Type: application/json" \
  -d '{"model": "default", "input": ["task: search result | query: What is graphene?", "task: search result | query: What is an apple?"]}'
```

An example with the OpenAI client can be found [here](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/server/embedding.py).

By default the server registers the model as `default`. To expose it under a custom name or alongside chat
models, run in multi-model mode and assign an identifier in the selector configuration:

```json
{
  "embed-gemma": {
    "Embedding": {
      "model_id": "google/embeddinggemma-300m",
      "arch": "embeddinggemma"
    }
  }
}
```

See [docs/HTTP.md](HTTP.md#post-v1embeddings) for the full request schema and response layout.

## Python SDK

Instantiate `Runner` with the `Which.Embedding` selector and request EmbeddingGemma explicitly. The helper method
`send_embedding_request` returns batched embeddings as Python lists.

```python
from mistralrs import EmbeddingArchitecture, EmbeddingRequest, Runner, Which

runner = Runner(
    which=Which.Embedding(
        model_id="google/embeddinggemma-300m",
        arch=EmbeddingArchitecture.EmbeddingGemma,
    )
)

request = EmbeddingRequest(
    input=["task: search result | query: What is graphene?", "task: search result | query: What is an apple?"],
    truncate_sequence=True,
)

embeddings = runner.send_embedding_request(request)
print(len(embeddings), len(embeddings[0]))
```

Refer to [this example](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/python/embedding_gemma.py) for a complete runnable script.

## Rust SDK

Use the `EmbeddingModelBuilder` helper from the `mistralrs` crate to create the model and submit an
`EmbeddingRequest`:

```rust
use anyhow::Result;
use mistralrs::{EmbeddingModelBuilder, EmbeddingRequest};

#[tokio::main]
async fn main() -> Result<()> {
    let model = EmbeddingModelBuilder::new("google/embeddinggemma-300m")
        .with_logging()
        .build()
        .await?;

    let embeddings = model
        .generate_embeddings(
            EmbeddingRequest::builder()
                .add_prompt("task: search result | query: What is graphene?")
        )
        .await?;

    println!("Returned {} vectors", embeddings.len());
    Ok(())
}
```

This example lives [here](https://github.com/EricLBuehler/mistral.rs/blob/master/mistralrs/examples/getting_started/embedding/main.rs), and can be run with:

```bash
cargo run --package mistralrs --example embedding_gemma
```
