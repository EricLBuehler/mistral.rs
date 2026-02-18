# Qwen3 Embedding

The Qwen3 Embedding model series is the latest proprietary model of the Qwen family, specifically designed for text embedding and ranking tasks. 

For a catalog of all embedding backends, see [EMBEDDINGS.md](EMBEDDINGS.md).

## HTTP server

Serve the model with the OpenAI-compatible endpoint enabled:

```bash
mistralrs serve -p 1234 -m Qwen/Qwen3-Embedding-0.6B
```

Call the endpoint via `curl` or the OpenAI SDK:

```bash
curl http://localhost:1234/v1/embeddings \
  -H "Authorization: Bearer EMPTY" \
  -H "Content-Type: application/json" \
  -d '{"model": "default", "input": ["Graphene conductivity", "Explain superconductors in simple terms."]}'
```

An example with the OpenAI client can be found [here](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/server/embedding.py).

To expose the model alongside chat models, register it in your selector configuration using the
`qwen3embedding` architecture tag:

```json
{
  "embed-qwen3": {
    "Embedding": {
      "model_id": "Qwen/Qwen3-Embedding-0.6B",
      "arch": "qwen3embedding"
    }
  }
}
```

See [docs/HTTP.md](HTTP.md#post-v1embeddings) for the full request schema.

## Python SDK

Instantiate `Runner` with the embedding selector and request Qwen3 explicitly. The output mirrors the
OpenAI embeddings array shape:

```python
from mistralrs import EmbeddingArchitecture, EmbeddingRequest, Runner, Which

runner = Runner(
    which=Which.Embedding(
        model_id="Qwen/Qwen3-Embedding-0.6B",
        arch=EmbeddingArchitecture.Qwen3Embedding,
    )
)

request = EmbeddingRequest(
    input=["Graphene conductivity", "Explain superconductors in simple terms."],
    truncate_sequence=True,
)

embeddings = runner.send_embedding_request(request)
print(len(embeddings), len(embeddings[0]))
```

A ready-to-run version can be found at [`examples/python/qwen3_embedding.py`](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/python/qwen3_embedding.py).

## Rust SDK

Use the `EmbeddingModelBuilder` helper just like with EmbeddingGemma. The example below mirrors the
repository sample:

```rust
use anyhow::Result;
use mistralrs::{EmbeddingModelBuilder, EmbeddingRequest};

#[tokio::main]
async fn main() -> Result<()> {
    let model = EmbeddingModelBuilder::new("Qwen/Qwen3-Embedding-0.6B")
        .with_logging()
        .build()
        .await?;

    let embeddings = model
        .generate_embeddings(
            EmbeddingRequest::builder()
                .add_prompt("What is graphene?")
                .add_prompt("Explain superconductors in simple terms.")
        )
        .await?;

    println!("Returned {} vectors", embeddings.len());
    Ok(())
}
```

You can find the full example at [`mistralrs/examples/advanced/embeddings/main.rs`](https://github.com/EricLBuehler/mistral.rs/blob/master/mistralrs/examples/advanced/embeddings/main.rs).
