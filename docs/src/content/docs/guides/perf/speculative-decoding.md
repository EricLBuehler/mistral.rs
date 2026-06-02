---
title: Use speculative decoding
description: Use MTP assistants to draft multiple tokens per target verification pass.
sidebar:
  order: 9
---

Speculative decoding lets a smaller assistant propose future tokens while the target model verifies them in parallel. mistral.rs exposes this through the generic MTP API.

## Support Matrix

| Mode | Target models | Assistant model | Status | Guide |
|---|---|---|---|---|
| MTP | Gemma 4 | Gemma 4 assistant checkpoints | Supported with PagedAttention | [Gemma 4 MTP](/mistral.rs/guides/perf/gemma4-mtp/) |

Legacy target/draft speculative decoding has been removed. New speculative decoding features should use the MTP proposer/target path.

## CLI

Use `--mtp-model` with an assistant model id or path:

```bash
mistralrs run -m <target-model> \
  --mtp-model <assistant-model-or-path> \
  --mtp-n-predict 6
```

`--mtp-n-predict` controls how many assistant tokens are proposed per step. If it is omitted, mistral.rs reads `num_assistant_tokens` from the assistant `generation_config.json` and falls back to 6.

## Python

`Runner` accepts `mtp_model` and `mtp_n_predict`:

```python
from mistralrs import Runner, Which

runner = Runner(
    which=Which.Plain(model_id="<target-model>"),
    mtp_model="<assistant-model-or-path>",
    mtp_n_predict=6,
)
```

## Rust

Builders that load text, multimodal, or auto-detected models accept an MTP config:

```rust
use mistralrs::{ModelBuilder, MtpConfig};

let model = ModelBuilder::new("<target-model>")
    .with_mtp_config(MtpConfig {
        model: "<assistant-model-or-path>".to_string(),
        n_predict: Some(6),
    })
    .build()
    .await?;
```

`with_mtp_model("<assistant-model-or-path>", Some(6))` is equivalent for common cases.

## Notes

MTP remains exact because accepted output is verified by the target model before it is emitted. Throughput gain depends on how many proposed tokens the target accepts and on the cost of the target verification pass.

Gemma 4 MTP requires PagedAttention. Non-paged KV-cache MTP is disabled while that path is developed separately.

MTP supports batched generation and constrained decoding.
