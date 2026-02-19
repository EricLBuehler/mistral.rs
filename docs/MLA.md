# Multi-head Latent Attention (MLA) in mistral.rs

Multi-head Latent Attention (MLA) is an efficient attention mechanism that reduces KV cache memory usage by compressing key-value states into a low-rank latent space. This technique was introduced in DeepSeek V2 and is also used in DeepSeek V3 and GLM-4.7-Flash models.

## How It Works

MLA compresses the key-value cache by:
1. Projecting KV states into a compact latent representation (`kv_lora_rank` dimensions)
2. Storing only the compressed latent vectors and rotary position embeddings in the KV cache
3. Reconstructing full KV states on-the-fly during attention computation

This results in significant memory savings compared to standard multi-head attention, enabling longer context lengths with the same GPU memory.

## Supported Models

MLA is automatically enabled for the following model architectures when using [PagedAttention](PAGED_ATTENTION.md) on CUDA:

| Model | Architecture | MLA Dimensions |
|-------|--------------|----------------|
| [DeepSeek V2](DEEPSEEKV2.md) | `deepseekv2` | kv_lora_rank varies |
| [DeepSeek V3](DEEPSEEKV3.md) | `deepseekv3` | kv_lora_rank=512, kpe_head_dim=64 |
| [GLM-4.7-Flash](GLM4_MOE_LITE.md) | `glm4moelite` | kv_lora_rank=512, kpe_head_dim=64 |

## Requirements

MLA decode optimization requires:
- **CUDA** on Unix-like platforms (Linux, WSL)
- **PagedAttention** enabled
- Compatible model architecture (see table above)

When these conditions are met, MLA is automatically used during the decode phase for optimal performance.

## Performance Benefits

MLA provides two key optimizations:

1. **Reduced KV Cache Memory**: The compressed latent representation uses significantly less memory than full key-value states, allowing for:
   - Longer context lengths
   - Larger batch sizes
   - More efficient memory utilization

2. **Optimized Decode Kernels**: Custom FlashInfer-based MLA kernels accelerate single-token generation by:
   - Operating directly on compressed latent states
   - Avoiding repeated KV decompression
   - Leveraging efficient memory access patterns

## Disabling MLA

If you encounter issues or want to compare performance, you can disable MLA by setting the environment variable:

```bash
MISTRALRS_NO_MLA=1 mistralrs ...
```

When disabled, the model falls back to standard PagedAttention with full KV cache storage.

## Technical Details

### KV Cache Layout

When MLA is enabled, PagedAttention uses a specialized cache layout:
- **Key cache**: Stores compressed latent vectors (`kv_lora_rank` dimensions) + rotary position embeddings (`kpe_head_dim` dimensions)
- **Value cache**: Shares the same block structure for efficient memory management

### Decode Path

During single-token generation (decode phase):
1. Query is projected to latent space
2. Attention is computed directly on compressed KV states using FlashInfer MLA kernels
3. Output is projected back from latent space

### Prefill Path

During prompt processing (prefill phase):
1. Full KV states are computed for the current chunk
2. Compressed latents are stored in the PagedAttention cache
3. For prefix-cached sequences, latents are retrieved and decompressed as needed

## See Also

- [PagedAttention](PAGED_ATTENTION.md) - Required for MLA optimization
- [FlashAttention](FLASH_ATTENTION.md) - Accelerates prefill phase
- [DeepSeek V2](DEEPSEEKV2.md) - Model documentation
- [DeepSeek V3](DEEPSEEKV3.md) - Model documentation
- [GLM-4.7-Flash](GLM4_MOE_LITE.md) - Model documentation
