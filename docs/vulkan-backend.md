# Vulkan backend

mistral.rs can run inference on any GPU with a Vulkan 1.2 driver (for example AMD
GPUs on Mesa RADV) through an experimental Vulkan backend in Candle. The backend is
feature-gated: builds without `--features vulkan` do not depend on Vulkan crates or
libraries.

> **Status: experimental bring-up.** GGUF text inference (validated: LFM3-1.7B
> Q4_K_M) runs fully on the GPU with zero CPU fallbacks. Throughput is still well
> below llama.cpp's Vulkan backend (see [Benchmarks](#benchmarks-amd-bc-250)).
> Multimodal, ASR and TTS paths have not been validated on Vulkan.

## Architecture

```
mistralrs-cli / mistralrs (SDK) / mistralrs-server-core
        │  `vulkan` feature, device selection (mistralrs_core::new_vulkan_device)
mistralrs-core   ── dtype policy (f32 on Vulkan), device maps, memory reporting, diagnostics
mistralrs-quant  ── GGUF QMatMul → candle QTensor (native), native RoPE, LFM3 causal conv,
        │           GGUF embedding gather, fused GLU composed from native ops
candle-core / candle-nn (fork, `vulkan` feature)
        └─ Device::Vulkan, VulkanStorage, QVulkanStorage, GLSL/WGSL kernels → SPIR-V (naga)
```

### Candle fork

The backend lives in a fork of Candle pinned in the workspace `Cargo.toml`
(`[patch.crates-io]`), so every crate resolves a single `candle-core`/`candle-nn`:

- Repository: <https://github.com/nglmercer/candle>, branch `vulkan-0.11.1`
- Base: the Candle 0.11.1 release commit (`59f8e903`), i.e. exactly what
  mistral.rs uses from crates.io, plus the Vulkan commits on top.
- Source of the backend: the Vulkan backend from
  [rexlunae/candle-core](https://github.com/rexlunae/candle-core/tree/joshua-vulkan),
  taken from its newer vendored copy in
  [rexlunae/joshua](https://github.com/rexlunae/joshua/tree/main/vendor/candle-core)
  (more kernels, including quantized GEMV/GEMM). Only the Vulkan parts were ported;
  that tree's OpenCL/SYCL backends, IQ2_XXS type, CPU grouped-Q8 paths and other
  unrelated changes were left out. See the commit messages on the branch for the
  full list of changes made during the port.

Key properties of the backend:

- **Bindings and shaders.** `ash` (with `loaded`, so `libvulkan` is loaded at
  runtime and not linked) and `naga` (GLSL/WGSL → SPIR-V at runtime). No `glslc`,
  `wgpu` or `vulkano`, and no GGML/llama.cpp execution dependency.
- **Memory.** Tensors are sub-allocated from 128 MiB blocks of host-visible,
  host-coherent memory, which is device-local when available (unified memory on
  APUs such as the BC-250). Uploads and downloads are `memcpy`s through the mapped
  pointer.
- **Execution.** Kernels are recorded into one command buffer per device and
  submitted lazily: at host reads, on `synchronize`, or when the batch is full.
  Buffers referenced by a pending batch are released after it completes.
- **Quantized weights stay quantized.** GGUF blocks are uploaded as-is
  (`QVulkanStorage`). Decode (`m ≤ 8` rows) uses a quantized GEMV. Prefill
  dequantizes one weight at a time into a single reused scratch buffer, then runs
  an f32 GEMM. Embedding lookups gather and dequantize only the selected rows.
- **Devices are shared per ordinal.** `Device::new_vulkan(n)` returns handles on
  one logical device while any handle is alive, so tensors created through
  separate calls can be combined.
- **f32 compute.** The native kernels are f32. mistral.rs therefore selects
  `F32` automatically on Vulkan, and warns if `bf16`/`f16` is forced, since those
  would run mostly through the CPU fallback.

### CPU fallback is always counted

Any op without a native kernel for its dtype, layout or shape takes a CPU
round-trip (download, CPU kernel, upload). This is never silent: every round-trip
is recorded per op, and the default `vulkan_fwd` of every custom op does the same.

```rust
let stats = candle_core::vulkan_backend::stats(); // or mistralrs::vulkan_execution_report()
println!("{stats}");
// native dispatches 41198, CPU fallbacks 0, H2D 0.1 MiB, D2H 32.2 MiB, syncs 836,
// GPU wait 5454.2 ms, dequantizations 0
```

`stats()` reports:

| Counter | Meaning |
|---|---|
| native dispatches | compute kernels recorded |
| CPU fallbacks (+ per-op breakdown) | CPU round-trips |
| H2D / D2H | bytes uploaded / read back |
| syncs | submissions the host waited on |
| GPU wait | host time blocked on the GPU |
| dequantizations | full-tensor dequantizations of quantized weights |

`mistralrs bench` resets these counters before the timed runs and prints them
afterwards.

## Building

Requirements:

- A Vulkan 1.2+ loader and driver at runtime (e.g. `vulkan-icd-loader` + Mesa RADV).
- No Vulkan SDK, `glslc` or headers are needed at build time.

```bash
cargo build --release -p mistralrs-cli --features vulkan
```

The `vulkan` feature exists on `mistralrs-quant`, `mistralrs-core`, `mistralrs`,
`mistralrs-server-core` and `mistralrs-cli`, following the `cuda`/`metal`
conventions. SDK users enable `mistralrs/vulkan`.

## Device selection

- **Automatic.** In a Vulkan build (without CUDA or Metal), the CLI, server and SDK
  `best_device` open adapter `MISTRALRS_VULKAN_DEVICE` (default `0`, the `GPUn`
  index from `vulkaninfo --summary`).
  - If it cannot be opened, startup fails with the list of available adapters.
  - `MISTRALRS_VULKAN_FALLBACK=cpu` allows the CPU instead, with a warning.
- **Explicit (SDK).** Pass `.with_device(mistralrs::new_vulkan_device(n)?)` to a
  model builder. This never falls back to the CPU.
- **Explicit (CLI).** `--cpu` forces the CPU. `-n ORD:LAYERS;...` maps layers to
  Vulkan ordinals.
- **Software adapters** (lavapipe/llvmpipe, `deviceType = CPU`) are refused unless
  `CANDLE_VULKAN_ALLOW_CPU=1`, so a missing GPU driver is never reported as GPU
  acceleration.
- **Startup log.** The adapter actually used is logged, e.g.
  `Using Vulkan device vulkan:0 AMD BC-250 (RADV GFX1013) (integrated, driver radv Mesa 26.2.5, Vulkan 1.4.354, 5.3 GiB device-local, unified memory)`.
- **Diagnostics.** `mistralrs doctor` lists Vulkan adapters, including software ones.

Memory reporting uses the size of the device-local heap minus this process's
allocator usage. On the BC-250 that heap is the 5.3 GiB carve-out, which bounds
the weights and KV cache that fit.

## Supported operations

Native f32 kernels from the Candle backend:

- Elementwise unary/binary/affine/powf/elu/cmp/where.
- Casts, with RNE f32→f16.
- Reductions and argmax/argmin.
- Strided copies and `copy2d`.
- `index_select`, `gather`, `scatter`, `index_add`.
- Batched matmul.
- `softmax_last_dim`, `rms_norm`, `sigmoid`, `rope`/`rope_i`.

mistral.rs additions:

- Quantized matmul (GGUF block formats, below).
- Quantized embedding gather.
- Positioned RoPE (neox and GPT-J pairing, partial rotary dims).
- LFM3 gated causal depthwise convolution.
- GLU composed as activation × gate.

### GGUF formats

Weights are uploaded in their block format. Quantized GEMV/GEMM and embedding
kernels exist for Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K,
Q8_K, F16, BF16 and F32. Validated end to end: Q4_K_M, which mixes Q4_K and Q6_K.
The other formats are covered by Candle's kernel-vs-CPU unit tests
(`vulkan_qgemv_kernels_match_cpu`, `vulkan_quantized_parity`), not by full models.

### Known unsupported or CPU-fallback operations

- Anything in bf16/f16 (use f32).
- `conv1d`/`conv2d`/`conv_transpose*`, pooling, upsampling.
- Custom ops without a `vulkan_fwd`; they take the counted CPU round-trip.
- PagedAttention, flash attention, and the CUDA/Metal-only fused kernels: Vulkan
  uses the eager attention path.
- Direct loading of sub-byte safetensors dtypes (F4/F6/F8E8M0) onto Vulkan
  (clear error).
- Multimodal encoders (LFM3-Omni vision/audio, mmproj) and speech generation are
  **untested** on Vulkan. Expect fallbacks or missing kernels.

## Debugging and profiling

| Variable | Effect |
|---|---|
| `CANDLE_VULKAN_TRACE=1` | log every CPU fallback (with the reason / kernel error) and every full dequantization |
| `CANDLE_VULKAN_NATIVE=0` | force every op through the CPU round-trip (A/B correctness checks) |
| `CANDLE_VULKAN_QGEMV=generic\|cubecl\|cubecl-packed\|cubecl-next\|mali\|mali-int8` | choose the quantized GEMV kernel |
| `CANDLE_VULKAN_ALLOW_CPU=1` | allow software (lavapipe) adapters |
| `MISTRALRS_VULKAN_DEVICE=n` | adapter for automatic selection |
| `MISTRALRS_VULKAN_FALLBACK=cpu` | let automatic selection fall back to the CPU (with a warning) |

Tests (they skip themselves when no hardware Vulkan device is available):

```bash
cd candle && cargo test -p candle-core --features vulkan --lib vulkan
cd candle && cargo test -p candle-nn --features vulkan --test vulkan_ops
cargo test -p mistralrs-quant --features vulkan --lib -- vulkan_rope lfm3_conv
```

The mistral.rs Vulkan tests also assert that their op recorded **no** CPU
fallback, so a kernel that fails to compile cannot pass by falling back to the
CPU path.

## Benchmarks (AMD BC-250)

Setup:

- Hardware: AMD BC-250 (RDNA2 `gfx1013` APU), RADV, Mesa 26.2.5, Vulkan
  1.4.354, unified memory (5.3 GiB device-local heap).
- Model: `LiquidAI/LFM3-1.7B-GGUF` `LFM3-1.7B-Q4_K_M.gguf`.
- mistral.rs: `mistralrs bench`, greedy, 1 warmup run, mean of 2–3 timed runs.
  Throughput is computed from TTFT and decode time.
- llama.cpp: `llama-bench -ngl 99 -r 3`, Vulkan backend, commit `6c01da6`.

| Workload | mistral.rs CPU | mistral.rs Vulkan | llama.cpp Vulkan | Vulkan vs llama.cpp |
|---|---:|---:|---:|---:|
| Prefill, 64 tokens | 105 tok/s | 177 tok/s | 1065 tok/s | 17 % |
| Prefill, 512 tokens | 118 tok/s | 459 tok/s | 1530 tok/s | 30 % |
| Decode, 32 tokens @ depth 64 | 22.6 tok/s | 13.6 tok/s | 157 tok/s | 9 % |
| Model load (warm cache) | — | ~5 s | — | |

Vulkan decode runs with 0 CPU fallbacks, ~0.5 MiB/token read back (the logits) and
~620 dispatches/token; GPU wait accounts for ~95 % of wall time.

Correctness: with greedy decoding, Vulkan and CPU produce the same first ~35
tokens. They then diverge at a near-tie: both outputs are coherent, and the
difference comes from the Q4_K accumulation order. 24 repeated requests on one
server showed stable process memory (no leak).

## Future work

Roughly in order of expected impact:

1. **Dispatch overhead.** Decode is bound by ~620 small dispatches/token, each
   followed by a full compute barrier.
   - Add GPU timestamp queries to profile them.
   - Fuse RMSNorm+matmul, SwiGLU, residual adds and the attention steps.
   - Use finer-grained barriers.
2. **RDNA-tuned quantized GEMV.** The current kernels were tuned on Intel/Arm, and
   all variants perform within ±15 % of each other here. Port the subgroup and
   wave64 Q4_K/Q6_K techniques from llama.cpp's `ggml-vulkan` (see also
   `llama.cpp-bc250`) once profiling shows where time goes.
3. **Quantized prefill GEMM** without dequantizing to an f32 scratch.
4. **Fewer host syncs per token.** Sample on the GPU, and avoid
   `synchronize()`/host reads in the step loop.
5. **f16 activations and KV cache** (`shaderFloat16`) to halve bandwidth.
6. **Coverage.**
   - Conv kernels for the LFM3-Omni audio encoder and vision tower.
   - Validate ASR/TTS.
   - Persistent/recurrent state paths for realtime sessions.
