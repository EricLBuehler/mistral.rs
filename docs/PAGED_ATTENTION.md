# PagedAttention in mistral.rs

Mistral.rs supports PagedAttention ([paper here](https://arxiv.org/abs/2309.06180)) to accelerate both normal inference and batched inference on:
- CUDA (Unix-like platforms such as WSL, Linux)
- Metal

Our PagedAttention implementation has 2 inputs: GPU KV cache memory size, and block size. This enables you to have fine-tuned control over the available context length, by configuring the available memory for KV cache. When using a CUDA device, PagedAttention is actiated by default but can be disabled with `no_paged_attn` for Python or `no-paged-attn` for the CLI tools.

## KV Cache Quantization

PagedAttention now supports KV cache quantization to reduce memory usage and potentially improve performance. The KV cache can be quantized to FP8 (F8E4M3 format) instead of using the model's native dtype, significantly reducing memory requirements while maintaining model quality.

**Available cache types:**
- `auto` (default): Uses the model's native dtype for KV cache
- `f8e4m3`: Quantizes KV cache to 8-bit floating point (E4M3 format)

When using FP8 quantization, the memory usage for KV cache is approximately halved compared to FP16, allowing for longer context lengths with the same GPU memory allocation.

> Note: The default block size if not specified is 32.

> Note: if OOM occurs (this can be caused by a variety of factors including adapter activation, re-ISQ, and others), it is likely because the PagedAttention KV cache has already been allocated. To counter this, either set the KV cache memory to a lower amount or usage percentage (recommended) or disable paged attention entirely for a dynamically allocated cache.

> Note: Paged Attention is not enabled on Windows platforms, only Unix-based platforms.

> Note: In the CLI and Python API, Paged Attention is disabled by default for Metal. It can be enabled with the `--paged-attn`/`paged_attn` flags.

**There are more features being added to this:**
- GGML model support
- Adapter model support
- Speculative decoding

**Prefix caching is now supported with PagedAttention.** PagedAttention can leverage the prefix cacher to cache KV prefix states across iterations for faster multi-turn inference.

**Supported models:**
- Normal models
- GGUF models
- Vision models

> Note: Prefix caching is supported when using PagedAttention. Configure the number of sequences to cache on the device with:
> - CLI: `--prefix-cache-n <N>` (default 16)
> - Python API: `prefix_cache_n=<N>` (default 16)
> - Rust API: `.with_prefix_cache_n(Some(N))` (default 16)

## FlashAttention V2/V3 + PagedAttention in mistral.rs

If mistral.rs is compiled with [FlashAttention](FLASH_ATTENTION.md) and PagedAttention is enabled, then FlashAttention will be used in tandem to accelerate
the prefill phase.

## Using the CLI

Add the `--pa-gpu-mem`/`--pa-gpu-mem-usage` and `--pa-blk-size` parameters before the model kind selector. The GPU memory is in MBs and the block size means the number of tokens per block. These parameters may be passed on any supported model type.

To enable KV cache quantization, use the `--pa-cache-type` parameter with either `auto` (default) or `f8e4m3`.

```
cargo run --release --features cuda -- -i --pa-gpu-mem 8192 --pa-blk-size 32 --isq 4 plain -m microsoft/Phi-3-mini-128k-instruct
```

```
cargo run --release --features cuda -- -i --pa-gpu-mem-usage .95 --pa-blk-size 32 gguf -t mistralai/Mistral-7B-Instruct-v0.1 -m TheBloke/Mistral-7B-Instruct-v0.1-GGUF -f mistral-7b-instruct-v0.1.Q4_K_M.gguf
```

Example with FP8 KV cache quantization:
```
cargo run --release --features metal -- -i --pa-gpu-mem 4096 --pa-blk-size 32 --pa-cache-type f8e4m3 plain -m microsoft/Phi-3-mini-128k-instruct
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

Example with FP8 KV cache quantization:
```rust
use anyhow::Result;
use mistralrs::{
    IsqType, MemoryGpuConfig, PagedAttentionMetaBuilder, PagedCacheType, 
    TextMessageRole, TextMessages, TextModelBuilder,
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
                .with_cache_type(PagedCacheType::F8E4M3)
                .build()
        })?
        .build()
        .await?;

    // ... rest of the code remains the same
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

Example with FP8 KV cache quantization:
```py
from mistralrs import Runner, Which, ChatCompletionRequest, Architecture, PagedCacheType

runner = Runner(
    which=Which.Plain(
        model_id="mistralai/Mistral-7B-Instruct-v0.1",
        arch=Architecture.Mistral,
    ),
    pa_gpu_mem = 4096,
    pa_blk_size = 32,
    pa_cache_type = PagedCacheType.F8E4M3,
)

# ... rest of the code remains the same
```