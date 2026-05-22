---
title: Gemma 4 MTP
description: Use Gemma 4 assistant checkpoints for MTP speculative decoding.
sidebar:
  order: 9
---

Gemma 4 assistant checkpoints are Multi-Token Prediction drafters for Gemma 4 target models. The assistant proposes several future tokens, and the target verifies the proposal before tokens are emitted. See the [`google/gemma-4-E4B-it-assistant`](https://huggingface.co/google/gemma-4-E4B-it-assistant) model card for the upstream checkpoint.

Gemma 4 MTP currently requires PagedAttention.

## CLI

Use the published assistant checkpoint:

```bash
mistralrs run -m google/gemma-4-E4B-it --quant 8 \
  --mtp-model google/gemma-4-E4B-it-assistant \
  --mtp-n-predict 6
```

Or use a downloaded checkout:

```bash
mistralrs run -m google/gemma-4-E4B-it --quant 8 \
  --mtp-model ./gemma-4-E4B-it-assistant \
  --mtp-n-predict 6
```

`--mtp-n-predict` controls how many assistant tokens are proposed per step. If it is omitted, mistral.rs reads `num_assistant_tokens` from the assistant `generation_config.json` and falls back to 6.

## Python

```python
from mistralrs import Runner, Which

runner = Runner(
    which=Which.MultimodalPlain(model_id="google/gemma-4-E4B-it"),
    in_situ_quant="8",
    mtp_model="google/gemma-4-E4B-it-assistant",
    mtp_n_predict=6,
)
```

## Rust

```rust
use mistralrs::{ModelBuilder, MtpConfig};

let model = ModelBuilder::new("google/gemma-4-E4B-it")
    .with_mtp_config(MtpConfig {
        model: "google/gemma-4-E4B-it-assistant".to_string(),
        n_predict: Some(6),
    })
    .build()
    .await?;
```

For concise builder code, use:

```rust
let model = mistralrs::ModelBuilder::new("google/gemma-4-E4B-it")
    .with_mtp_model("google/gemma-4-E4B-it-assistant", Some(6))
    .build()
    .await?;
```

## Compatibility

The target must run with PagedAttention. Non-paged KV-cache MTP is intentionally disabled for now.

The target and assistant configs must match where required by the implementation, including vocabulary size and target hidden size. If they do not match, loading fails before generation starts.

MTP supports batched generation and constrained decoding.
