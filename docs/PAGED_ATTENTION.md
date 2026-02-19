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

> Note: In the CLI and Python SDK, Paged Attention is disabled by default for Metal. It can be enabled with the `--paged-attn`/`paged_attn` flags.

**There are more features being added to this:**
- GGML model support
- Adapter model support
- Speculative decoding

**Prefix caching is now supported with PagedAttention.** PagedAttention can leverage the prefix cacher to cache KV prefix states across iterations for faster multi-turn inference.

## Block-Level Prefix Caching

Prefix caching is a technique to reuse computed KV cache blocks across requests that share common prefixes (like system prompts). This can significantly speed up inference when multiple requests use the same prefix.

### How It Works

1. **Block Hashing**: Each block of tokens is assigned a unique hash based on its contents and the hash of its parent block:
   ```
   hash(block) = hash(parent_hash, block_tokens)
   ```
   This creates a hash chain that uniquely identifies any prefix sequence.

2. **Cache Lookup**: When allocating blocks for a new request, the scheduler checks if any full blocks match existing cached blocks by comparing hashes.

3. **Block Reuse**: Matched blocks are reused directly - their pre-computed KV cache values are used without recomputation. Only the non-matching suffix tokens need to be processed.

4. **LRU Eviction**: When memory is needed, least recently used cached blocks are evicted first.

### Benefits

- **Multi-turn conversations**: System prompts and conversation history are cached and reused
- **Batched requests**: Multiple requests with shared prefixes (e.g., same system prompt) benefit from caching
- **Reduced TTFT**: Time-to-first-token is reduced by skipping prefix computation

### How It's Enabled

Prefix caching is **enabled by default** when using PagedAttention and controlled by the same `prefix_cache_n` setting that controls the sequence-level prefix cacher:

- **CLI**: `--prefix-cache-n <N>` (default 16). Set to 0 to disable prefix caching.
- **Python SDK**: `prefix_cache_n=<N>` (default 16). Set to `None` or `0` to disable.
- **Rust SDK**: `.with_prefix_cache_n(Some(N))` (default 16). Pass `None` to disable.

**Important:** The two prefix caching systems are mutually exclusive:
- **PagedAttention** uses block-level prefix caching (handled by `PrefixCacher` in `BlockEngine`)
- **Non-PagedAttention** uses sequence-level prefix caching (handled by `PrefixCacheManagerV2`)

The `prefix_cache_n` setting controls both systems, but only one is active depending on whether PagedAttention is enabled. You'll see one of these log messages at startup indicating which system is active:
- `Prefix caching enabled (block-level, PagedAttention).`
- `Prefix caching enabled (sequence-level, non-paged attention).`

### Implementation Details

The prefix cache operates at the block level (not token level) for efficiency:

1. **Full blocks only**: Only complete blocks (block_size tokens) are cached. Partial blocks at the end of a sequence are not cached.

2. **Hash chain**: The hash for each block depends on all preceding blocks, ensuring the entire prefix matches.

3. **Copy-on-Write**: Cached blocks use reference counting. When a cached block needs modification, it's copied first (CoW).

4. **Memory management**: The cache uses LRU eviction when allocating new blocks. Evicted blocks are returned to the free pool.

### Performance Considerations

- Block size affects cache granularity: larger blocks = fewer cache entries but coarser matching
- Cache hit rate improves with more repeated prefixes
- Memory overhead is minimal (just hash-to-block mappings)

**Supported models:**
- Normal models
- GGUF models
- Vision models

> Note: Prefix caching is supported when using PagedAttention. Configure the number of sequences to cache on the device with:
> - CLI: `--prefix-cache-n <N>` (default 16)
> - Python SDK: `prefix_cache_n=<N>` (default 16)
> - Rust SDK: `.with_prefix_cache_n(Some(N))` (default 16)

## Metal Memory Behavior

On Metal (macOS Apple Silicon), the GPU and CPU share the same physical RAM (unified memory). Unlike CUDA GPUs with dedicated VRAM where unused memory would otherwise be wasted, allocating large KV caches on Metal wires physical RAM away from the OS and CPU, which can cause system-wide memory pressure and thrashing.

To avoid this, mistral.rs automatically caps the PagedAttention KV cache on Metal to `max_seq_len * max_batch_size` tokens — just enough for the configured context length. On CUDA, the full available memory is used for maximum request concurrency (following the vLLM approach).

You can override this behavior on any platform with `--pa-memory-mb` to set an explicit KV cache budget in megabytes.

## FlashAttention V2/V3 + PagedAttention in mistral.rs

If mistral.rs is compiled with [FlashAttention](FLASH_ATTENTION.md) and PagedAttention is enabled, then FlashAttention will be used in tandem to accelerate
the prefill phase.

## Using the CLI

Add the `--pa-gpu-mem`/`--pa-gpu-mem-usage` and `--pa-blk-size` parameters before the model kind selector. The GPU memory is in MBs and the block size means the number of tokens per block. These parameters may be passed on any supported model type.

To enable KV cache quantization, use the `--pa-cache-type` parameter with either `auto` (default) or `f8e4m3`.

```
mistralrs run --pa-memory-mb 8192 --pa-block-size 32 --isq 4 -m microsoft/Phi-3-mini-128k-instruct
```

```
mistralrs run --pa-memory-fraction 0.95 --pa-block-size 32 --format gguf -t mistralai/Mistral-7B-Instruct-v0.1 -m TheBloke/Mistral-7B-Instruct-v0.1-GGUF -f mistral-7b-instruct-v0.1.Q4_K_M.gguf
```

Example with FP8 KV cache quantization:
```
mistralrs run --paged-attn on --pa-memory-mb 4096 --pa-block-size 32 --pa-cache-type f8e4m3 -m microsoft/Phi-3-mini-128k-instruct
```

## Using the Rust SDK
You can find this example [here](https://github.com/EricLBuehler/mistral.rs/blob/master/mistralrs/examples/advanced/paged_attn/main.rs).

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
        .with_paged_attn(
            PagedAttentionMetaBuilder::default()
                .with_block_size(32)
                .with_gpu_memory(MemoryGpuConfig::ContextSize(1024))
                .build()?,
        )
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
        .with_paged_attn(
            PagedAttentionMetaBuilder::default()
                .with_block_size(32)
                .with_gpu_memory(MemoryGpuConfig::ContextSize(1024))
                .with_cache_type(PagedCacheType::F8E4M3)
                .build()?,
        )
        .build()
        .await?;

    // ... rest of the code remains the same
}
```

## Using the Python SDK
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