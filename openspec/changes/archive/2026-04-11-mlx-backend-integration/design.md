## Context

mistral.rs uses Candle as its tensor framework. On macOS, the `metal` feature enables Candle's Metal backend for GPU-accelerated forward passes. Separately, the `kvcache-compression` feature enables TurboQuant (WHT + PolarQuant/QJL) for compressing KV cache entries, currently running on CPU via SIMD (AVX-512/AVX2/NEON).

The KV cache compression hot path — `push_token()` and `decompress_to_tensors()` — accounts for measurable overhead during long-context inference. On Apple Silicon, MLX can fuse these operations into single Metal kernel dispatches via its lazy evaluation + compilation model, while Candle's Metal backend dispatches each op individually.

The existing `KvCache` enum in `mistralrs-core/src/kv_cache/mod.rs` already has a `Compressed` variant gated by `#[cfg(feature = "kvcache-compression")]`. The MLX integration adds a parallel path at the same abstraction level.

**Stakeholders**: macOS Apple Silicon users running long-context inference with TurboQuant enabled.

**Constraints**:
- Forward pass stays on Candle+Metal — we do not replace model execution
- `mlx-rs` v0.25.3 API is the boundary; some ops may be incomplete
- Bridge between Candle and MLX requires a copy (unified memory, no PCIe hop)

## Goals / Non-Goals

**Goals:**
- Route KV cache compress/decompress through MLX Metal kernels on macOS when `mlx` feature is active
- Provide a Candle↔MLX tensor bridge that handles dtype conversion (F32, F16, BF16)
- Maintain identical numerical behavior: MLX TurboQuant must produce outputs within cosine similarity >0.99 of the CPU TurboQuant path
- Zero impact when `mlx` feature is disabled — all code compiles out via `#[cfg]`
- Feature flag propagation across all workspace crates following the existing pattern

**Non-Goals:**
- Replacing Candle's forward pass with MLX (model execution stays on Candle)
- Zero-copy Metal buffer sharing (documented as future upgrade path, not v1)
- MLX-based PagedAttention (out of scope — PagedAttention stays on Candle Metal/CUDA)
- QJL "cold window" compression (v1 implements PolarQuant hot path only; QJL is a follow-up)
- Python SDK `mistralrs-mlx` package variant
- Supporting Intel Macs (MLX requires Apple Silicon)

## Decisions

### D1: MLX only for KV cache, not forward pass

**Decision**: The MLX backend handles only KV cache storage and TurboQuant compression/decompression. The forward pass remains on Candle+Metal.

**Rationale**: Replacing the forward pass would require reimplementing every model architecture in MLX. The KV cache lifecycle is a self-contained, high-value target — it's where TurboQuant operates, and MLX's kernel fusion directly benefits the compress/decompress pipeline.

**Alternative considered**: Full MLX backend (rejected — would require model reimplementation and duplicate the Candle model code).

### D2: New `KvCache::MlxCompressed` enum variant, not a separate backend abstraction

**Decision**: Add a new `MlxCompressed` variant to the existing `KvCache` enum in `kv_cache/mod.rs`, gated by `#[cfg(all(feature = "mlx", feature = "kvcache-compression"))]`. Do NOT introduce a `Backend` enum or `backend/` module tree.

**Rationale**: The existing codebase dispatches KV cache behavior through the `KvCache` enum with `match` arms. The `Compressed` variant already demonstrates the pattern. A separate `Backend` abstraction would add indirection that doesn't exist elsewhere and would complicate the `NormalCache` / `HybridCache` wrappers. Following the established pattern minimizes diff size and review burden.

**Alternative considered**: `KvCacheBackend` enum wrapping `Candle` vs `Mlx` (from the design notes) — rejected because it adds a layer of abstraction inconsistent with the codebase and forces changes to `NormalCache`, `HybridCache`, `EitherCache`, and all call sites.

### D3: MLX module lives in `mistralrs-core/src/kv_cache/mlx/`

**Decision**: Place the MLX bridge, cache, and TurboQuant code under `kv_cache/mlx/` rather than a top-level `backend/mlx/`.

**Rationale**: The MLX code is exclusively KV cache infrastructure. Placing it under `kv_cache/` reflects this scope and avoids creating a top-level `backend/` directory that implies broader framework integration.

**Structure**:
```
mistralrs-core/src/kv_cache/
├── mod.rs              # [mod] add MlxCompressed variant + mlx module
├── mlx/
│   ├── mod.rs          # feature gate + re-exports
│   ├── bridge.rs       # candle_to_mlx / mlx_to_candle
│   ├── cache.rs        # MlxCompressedCache type
│   └── turboquant.rs   # WHT + PolarQuant on MLX arrays
├── single_cache.rs     # unchanged
├── rotating_cache.rs   # unchanged
├── hybrid_cache.rs     # [mod] add mlx dispatch in apply_compression
└── full_cache.rs       # unchanged
```

### D4: `mlx` feature implies `kvcache-compression`

**Decision**: `mlx = ["dep:mlx-rs", "kvcache-compression"]` — enabling `mlx` automatically enables TurboQuant.

**Rationale**: The sole purpose of the MLX backend is accelerating TurboQuant operations. Without compression enabled, the MLX path has no function. Making the implication explicit prevents nonsensical feature combinations and simplifies cfg guards to `#[cfg(feature = "mlx")]` (no need for `all(feature = "mlx", feature = "kvcache-compression")`).

### D5: Copy-based bridge (v1) with documented zero-copy path

**Decision**: v1 bridge copies tensor data through CPU. On Apple Silicon unified memory, this is a memcpy within the same physical RAM — no PCIe transfer.

**Rationale**: Neither Candle nor mlx-rs expose stable raw Metal buffer handles. The copy cost is bounded by KV cache slice size (typically `[1, n_kv_heads, 1, head_dim]` per token — ~256 bytes at FP16 with 8 heads × 128 dim). Profiling may show this is negligible vs. the compression compute.

**Zero-copy upgrade**: When mlx-rs exposes `Array::from_metal_buffer`, replace the body of `candle_to_mlx` with a pointer share tied to the source tensor lifetime.

### D6: Pin mlx-rs to exact version `=0.25.3`

**Decision**: Use `mlx-rs = "=0.25.3"` (exact pin) in workspace dependencies.

**Rationale**: mlx-rs tracks MLX upstream minor versions and has broken API compatibility between minors. Exact pinning prevents surprise build breaks. Version bumps become explicit, reviewable PRs.

### D7: WHT and codebook tables precomputed at cache initialization

**Decision**: The Hadamard matrix and Lloyd-Max codebook are computed once when `MlxCompressedCache::new()` is called and stored as MLX arrays on-device.

**Rationale**: These are constants that depend only on `head_dim` and `bits`. Computing them per-token or per-call would waste Metal dispatch cycles. Storing as MLX arrays keeps them on the Metal device for zero-copy use in matmul/gather.

## Risks / Trade-offs

**[mlx-rs API instability]** → Pin exact version; wrap all mlx-rs calls behind `kv_cache/mlx/` module; bridge module provides a narrow surface area that's easy to update on version bumps.

**[Bridge copy overhead per token]** → Profile first. The copy is ~256 bytes per token per layer for typical models. If hot, pursue Metal buffer ptr sharing (D5 upgrade path). Alternatively, batch multiple tokens before bridging.

**[WHT/codebook divergence from turboquant-rs]** → Add round-trip fuzz test: compress with MLX path, decompress with CPU path (and vice versa), compare cosine similarity. The codebook tables are hardcoded constants — they either match or they don't.

**[Incomplete mlx-rs ops]** → `ops::bitwise_or`, `ops::left_shift`, `ops::right_shift` needed for 4-bit packing may not be in mlx-rs 0.25.3. Mitigation: fall back to `mlx-sys` FFI for missing ops, or implement packing on the CPU side (packing is not compute-heavy).

**[Lazy eval stalls]** → Call `array.eval()` at `append()` and `get()` boundaries to force computation and keep memory usage predictable. Document this requirement in the module.

**[First-build time]** → `mlx-sys` builds `mlx-c` from source via cmake (~5-15 min on first build). Document this in CARGO_FEATURES.md and CI caching strategy.

## Open Questions

1. **Batch vs. per-token bridging**: Should we accumulate N tokens on the Candle side before bridging to MLX for compression, or bridge each token individually? Batching amortizes bridge cost but adds latency for short sequences.

2. **BF16 support in mlx-rs**: Does mlx-rs 0.25.3 handle BF16 natively, or do we need to cast to F16 on the bridge? This affects models that use BF16 KV caches (e.g., Llama 3).

3. **HybridCache integration**: Should `MlxCompressed` support the `HybridCache` attention layers, or is it `NormalCache`-only for v1?

4. **Testing on CI**: macOS CI runners with Apple Silicon are limited. Should we use conditional compilation tests (`#[cfg(target_os = "macos")]`) or mock the MLX path for Linux CI?
