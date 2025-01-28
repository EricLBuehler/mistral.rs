# PagedAttention in mistral.rs

Mistral.rs supports PagedAttention ([paper here](https://arxiv.org/abs/2309.06180)) to accelerate both normal inference and batched inference on:
- CUDA (Unix-like platforms such as WSL, Linux)
- Metal

Our PagedAttention implementation has 2 inputs: GPU KV cache memory size, and block size. This enables you to have fine-tuned control over the available context length, by configuring the available memory for KV cache. When using a CUDA device, PagedAttention is actiated by default but can be disabled with `no_paged_attn` for Python or `no-paged-attn` for the CLI tools.

> Note: The default block size if not specified is 32.

> Note: if OOM occurs (this can be caused by a variety of factors including adapter activation, re-ISQ, and others), it is likely because the PagedAttention KV cache has already been allocated. To counter this, either set the KV cache memory to a lower amount or usage percentage (recommended) or disable paged attention entirely for a dynamically allocated cache.

> Note: Paged Attention is not enabled on Windows platforms, only Unix-based platforms.

> Note: In the CLI and Python API, Paged Attention is disabled by default for Metal. It can be enabled with the `--paged-attn`/`paged_attn` flags.

**There are more features being added to this:**
- GGML model support 
- Adapter model support
- Speculative decoding
- Prefix caching

**Supported models:**
- Normal models
- GGUF models
- Vision models

> Note: the prefix cacher will be disabled when using PagedAttention regardless of settings. This functionality will be added soon!

## FlashAttention V2/V3 + PagedAttention in mistral.rs

If mistral.rs is compiled with [FlashAttention](FLASH_ATTENTION.md) and PagedAttention is enabled, then FlashAttention will be used in tandem to accelerate
the prefill phase.

## Using the CLI

Add the `--pa-gpu-mem`/`--pa-gpu-mem-usage` and `--pa-blk-size` parameters before the model kind selector. The GPU memory is in MBs and the block size means the number of tokens per block. These parameters may be passed on any supported model type.

```
cargo run --release --features cuda -- -i --pa-gpu-mem 8192 --pa-blk-size 32 --isq Q4K plain -m microsoft/Phi-3-mini-128k-instruct -a phi3
```

```
cargo run --release --features cuda -- -i --pa-gpu-mem-usage .95 --pa-blk-size 32 gguf -t mistralai/Mistral-7B-Instruct-v0.1 -m TheBloke/Mistral-7B-Instruct-v0.1-GGUF -f mistral-7b-instruct-v0.1.Q4_K_M.gguf
```

## Using the Rust API
You can find this example [here](../mistralrs/examples/paged_attn/main.rs).

```rust
use anyhow::Result;
use mistralrs::{
    IsqType, MemoryGpuConfig, PagedAttentionMetaBuilder, TextMessageRole, TextMessages,
    TextModelBuilder,
};

#[tokio::main]
async fn main() -> Result<()> {
    let model = TextModelBuilder::new("microsoft/Phi-3.5-mini-instruct")
        .with_isq(IsqType::Q8_0)
        .with_logging()
        .with_paged_attn(|| {
            PagedAttentionMetaBuilder::default()
                .with_block_size(32)
                .with_gpu_memory(MemoryGpuConfig::ContextSize(1024))
                .build()
        })?
        .build()
        .await?;

    let messages = TextMessages::new()
        .add_message(
            TextMessageRole::System,
            "You are an AI agent with a specialty in programming.",
        )
        .add_message(
            TextMessageRole::User,
            "Hello! How are you? Please write generic binary search function in Rust.",
        );

    let response = model.send_chat_request(messages).await?;

    println!("{}", response.choices[0].message.content.as_ref().unwrap());
    dbg!(
        response.usage.avg_prompt_tok_per_sec,
        response.usage.avg_compl_tok_per_sec
    );

    Ok(())
}
```

## Using the Python API
```py
from mistralrs import Runner, Which, ChatCompletionRequest, Architecture

runner = Runner(
    which=Which.Plain(
        model_id="mistralai/Mistral-7B-Instruct-v0.1",
        arch=Architecture.Mistral,
    ),
    pa_gpu_mem = 4096,
    pa_blk_size = 32,
)

res = runner.send_chat_completion_request(
    ChatCompletionRequest(
        model="mistral",
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