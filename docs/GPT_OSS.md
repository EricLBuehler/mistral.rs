# GPT-OSS

GPT-OSS is a Mixture of Experts (MoE) language model with specialized attention mechanisms and efficient quantization. Key features include:

- MXFP4 quantized MoE experts for efficient inference
- Per-head attention sinks for improved attention patterns
- YARN RoPE scaling for extended context
- Hybrid cache supporting both full and sliding window attention

## Quick Start

```bash
mistralrs run -m openai/gpt-oss-20b
```

> Note: GPT-OSS MoE experts are pre-quantized in MXFP4 format. ISQ can be applied to attention layers only.

> Note: PagedAttention is not supported for GPT-OSS due to custom attention with sinks.

## HTTP API
You can find a more detailed example [here](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/server/gpt_oss.py).

```bash
mistralrs serve -p 1234 -m openai/gpt-oss-20b
```

```py
from openai import OpenAI

client = OpenAI(api_key="foobar", base_url="http://localhost:1234/v1/")

completion = client.chat.completions.create(
    model="default",
    messages=[
        {"role": "user", "content": "Tell me a story about the Rust type system."}
    ],
    max_tokens=256,
)
print(completion.choices[0].message.content)
```

## Python SDK
You can find a more detailed example [here](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/python/gpt_oss.py).

```py
from mistralrs import Runner, Which, ChatCompletionRequest, Architecture

runner = Runner(
    which=Which.Plain(
        model_id="openai/gpt-oss-20b",
        arch=Architecture.GptOss,
    ),
)

res = runner.send_chat_completion_request(
    ChatCompletionRequest(
        model="default",
        messages=[
            {"role": "user", "content": "Tell me a story about the Rust type system."}
        ],
        max_tokens=256,
        presence_penalty=1.0,
        top_p=0.1,
        temperature=0.1,
    )
)
print(res.choices[0].message.content)
print(res.usage)
```

## Rust SDK
You can find a more detailed example [here](https://github.com/EricLBuehler/mistral.rs/blob/master/mistralrs/examples/models/text_models/main.rs).

```rust
use anyhow::Result;
use mistralrs::{TextMessageRole, TextMessages, TextModelBuilder};

#[tokio::main]
async fn main() -> Result<()> {
    let model = TextModelBuilder::new("openai/gpt-oss-20b")
        .with_logging()
        .build()
        .await?;

    let messages = TextMessages::new()
        .add_message(TextMessageRole::User, "Hello!");

    let response = model.send_chat_request(messages).await?;
    println!("{}", response.choices[0].message.content.as_ref().unwrap());

    Ok(())
}
```

## Technical Details

### MXFP4 Quantization
GPT-OSS MoE experts use MXFP4 (4-bit microscaling floating point) quantization for compact and efficient storage:
- `gate_up_proj`: Packed experts with MXFP4 weights
- `down_proj`: Packed experts with MXFP4 weights
- Scales stored at 1 byte per 32 elements

### Attention with Sinks
The model uses per-head attention sinks that are added to attention logits before softmax, helping to regularize attention patterns. This custom attention mechanism is incompatible with PagedAttention.

### ISQ Support
In-situ quantization (ISQ) can be applied to attention projection layers:
- `q_proj`, `k_proj`, `v_proj`, `o_proj`
- `lm_head`

MoE expert layers are already MXFP4 quantized and excluded from ISQ.
