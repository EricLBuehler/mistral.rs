# F4 (4-bit) KV cache storage assessment — Phase 0 evidence

Date: 2026-08-16. Unit 3 (llama.cpp-equivalent long context) Phase 0.
All claims verified against the pinned candle-core git dependency at rev
35d7ae7ca5c93e17c77359c3617376b8a72e96a4 (Cargo.lock) and the mistral.rs
working tree at HEAD 6f4b4cce4.

## Verdict

candle-core 0.11.0 (rev 35d7ae7) CANNOT store or cast DType::F4 on CUDA.
The 4-bit KV cache must use custom packed storage on DType::U8 tensors with
pack/unpack inside the mistralrs-paged-attn kernels (llama.cpp-style). This
matches the fallback the plan anticipated; no candle patch is needed.

## Evidence, from the pinned candle source

- DType::F4 exists (candle-core/src/dtype.rs:35) with size_in_bytes() == 0
  (dtype.rs:110) and a CUDA storage variant CudaStorageSlice::F4(CudaSlice<u8>)
  (cuda_backend/mod.rs:114).
- CUDA allocation is REJECTED: CudaDevice::alloc_uninit rejects
  DType::F6E2M3 | F6E3M2 | F4 | F8E8M0 with "Dummy types not supported in CUDA
  backend" (cuda_backend/device.rs:644). Tensor::empty(.., DType::F4, cuda)
  therefore fails. This is the alloc path CacheEngine::allocate_gpu_cache uses
  for the KV block tensors (cache_engine.rs:164, 209, 266, 325).
- CUDA casts are REJECTED both directions: to_dtype rejects F4 as a source
  (cuda_backend/mod.rs:1564, UnsupportedDtype op to_dtype) and as a target
  (mod.rs:1672, "Dummy types not supported in CUDA backend").
- No F4 cast kernels exist in candle-kernels (cast.cu defines casts only for
  f16/bf16/f32/f64/u8/u32/fp8).

## Design implication for Phase 2b

- PagedCacheType::F4 is added to the enum (cache_engine.rs:13-17) and its
  to_dtype returns DType::U8 for storage, not DType::F4.
- Block shape math (cache_engine.rs:379-390) needs a sub-byte branch: today
  x = 16 / element_size, which divides by zero for F4 (size_in_bytes() == 0).
  For F4 use x = 32 (16 bytes per x-row, 2 values per byte), keeping the same
  byte stride as the F8E4M3 layout so the kernel pointer math stays uniform.
- The kernels already take cache_dtype + k_scale/v_scale pointers (paged-attn
  ffi.rs: reshape_and_cache, paged_attention_v1/v2_*, gather_kv_cache), so an
  F4 branch means in-kernel pack (F32 -> 4-bit + block scale) on write and
  dequant on read, mirroring the existing F8E4M3 + kv_scale_update flow
  (paged_attention.rs:747-760, update_kvscales.cu).
- Reference format: llama.cpp q4_0 KV blocks (f16 scale per 32 values,
  4-bit symmetric values biased by 8), the format of the reference
  implementation this unit is matching.

## KV cache numbers (authoritative, source-derived)

Model: Qwen3.6-14B-A3B FableVibes Q4_K_M, architecture qwen35moe.
Full-attention geometry (GGUF tensor shapes + config.rs ModelConfigMetadata
at qwen3_next.rs:925-937): 16 Q heads x 256, 2 KV heads x 256 (K and V both
256), so 1024 KV elements per token per layer.

- Today (40 layers paged, BF16): 80 KB/token -> 640 MiB at 8192 (measured,
  matches the block math: 256 blocks x 40 layers x 64 KB/block).
- With Phase 2a (10 paged layers): BF16 20 KB/token, F8E4M3 10 KB/token,
  F4 5 KB/token (0.5 B/el) or ~5.6 KB/token (q4_0 with f16 scales).
- At 320000 tokens: F8E4M3 ~3.1 GiB, F4 ~1.56 GiB (0.5 B/el).
  The handoff's "4.88 KB/token" (F4) and "9.77 KB/token" (F8E4M3) figures
  were close but derived from a slightly different element count; the
  authoritative per-token figures are 5 KB/token (F4) and 10 KB/token
  (F8E4M3).

## GDN prefill accounting (Phase 3 ground truth)

The deployed CUDA build never runs the CPU f32 path the plan assumed.
apply_recurrence (gdn/backend.rs:487-500) dispatches CUDA devices to
recurrence_cuda; RECURRENCE_CHUNK_THRESHOLD = 64 (backend.rs:9), so any
prefill >= 64 tokens uses chunked_gated_delta_rule_recurrence_cuda or
warp_gated_delta_rule_recurrence_cuda (head_k_dim 64|128) — both already
memory-light. Per GDN layer, sequential peak:

- q/k/v/g/beta prep buffers (cuda/gdn.rs:553-558): 16 k-heads x 128 and 32
  v-heads x 128 in f32 -> ~40 KB/token.
- output_buf (cuda/gdn.rs:187): bh x v_dim f32 -> 8 KB/token.
- mixed_qkv input (conv_dim = 2 x 2048 + 4096 = 6144) -> ~24 KB/token.
- Total ~72 KB/token per layer, freed per layer (30 GDN layers run
  sequentially). This does NOT explain the measured "~0.48 MB/token"
  (+480 MB for a 1k prompt). Phase 3 must re-measure empirically before
  changing anything; the CUDA path is already chunked.

## Repro

mistralrs-core/tests/f4_cache_storage.rs in this unit:
- Asserts candle F4 size_in_bytes() == 0 (sub-byte).
- Asserts a CUDA F4 Tensor::empty fails with the expected "Dummy types"
  error (cuda feature gate).
- Asserts a DType::U8 CUDA tensor round-trips (the storage vehicle for the
  packed cache).
- Asserts the q4_0-style pack/unpack round-trip on CPU U8 storage closes to
  the original f32 values within the format's tolerance (2^-4 relative on
  the block scale).
