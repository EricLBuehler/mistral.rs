## 1. Feature Flags & Dependencies

- [x] 1.1 Add `mlx-rs = { version = "=0.25.3", optional = true }` to workspace root `Cargo.toml` under `[target.'cfg(target_os = "macos")'.dependencies]`
- [x] 1.2 Add `mlx` feature to `mistralrs-core/Cargo.toml`: `mlx = ["dep:mlx-rs", "kvcache-compression"]`
- [x] 1.3 Propagate `mlx` feature to `mistralrs-cli/Cargo.toml`: `mlx = ["mistralrs-core/mlx", "mistralrs-server-core/mlx"]`
- [x] 1.4 Propagate `mlx` feature to `mistralrs-server-core/Cargo.toml`: `mlx = ["mistralrs-core/mlx"]`
- [x] 1.5 Propagate `mlx` feature to `mistralrs/Cargo.toml` (SDK): `mlx = ["mistralrs-core/mlx"]`
- [x] 1.6 Propagate `mlx` feature to `mistralrs-pyo3/Cargo.toml`: `mlx = ["mistralrs-core/mlx"]`
- [x] 1.7 Propagate `mlx` feature to `mistralrs-server/Cargo.toml`: `mlx = ["mistralrs-core/mlx", "mistralrs-server-core/mlx"]`
- [x] 1.8 Verify `cargo check` passes without `mlx` feature on Linux (no regressions)
- [ ] 1.9 Verify `cargo check --features mlx` compiles (macOS or with appropriate cfg stubs)

## 2. Candle-MLX Tensor Bridge

- [x] 2.1 Create `mistralrs-core/src/kv_cache/mlx/mod.rs` with feature gate and module declarations
- [x] 2.2 Implement `candle_to_mlx()` in `bridge.rs` supporting F32, F16, BF16 dtypes with shape preservation
- [x] 2.3 Implement `mlx_to_candle()` in `bridge.rs` supporting F32, F16, BF16 dtypes with device placement
- [x] 2.4 Add error handling for unsupported dtypes (U8, U32, I64, etc.)
- [x] 2.5 Write unit tests: round-trip identity for each dtype, shape preservation, unsupported dtype error
- [x] 2.6 Document zero-copy upgrade path as comments in `bridge.rs`

## 3. MLX TurboQuant Compressor

- [x] 3.1 Implement `wht_matrix(dim)` in `turboquant.rs` — Sylvester construction, normalized, returns on-device MLX array
- [x] 3.2 Implement `lloyd_max_codebook(bits)` in `turboquant.rs` — 4-bit Gaussian-optimal centroids and boundaries as MLX arrays
- [x] 3.3 Implement `MlxTurboQuantCompressor::new()` with precomputed WHT matrix and codebook
- [x] 3.4 Implement `compress()` — WHT rotation via matmul, scalar quantization via boundary comparison, 4-bit packing into uint8
- [x] 3.5 Implement `decompress()` — uint8 unpacking, centroid gather, inverse WHT via matmul
- [x] 3.6 Add `eval()` calls at compress/decompress boundaries
- [x] 3.7 Handle missing mlx-rs ops: if `bitwise_or`/`left_shift`/`right_shift` unavailable, implement packing fallback via CPU or `mlx-sys` FFI
- [x] 3.8 Write unit tests: compress output shape, decompress restores shape, round-trip cosine similarity > 0.99, non-power-of-two rejection
- [ ] 3.9 Write cross-validation test: compress with MLX, decompress with CPU turboquant-rs (and vice versa), verify cosine similarity

## 4. MLX KV Cache Type

- [x] 4.1 Implement `MlxCompressedCache` struct in `cache.rs` with fields: `keys`, `values` (Vec of Option MLX arrays), `max_seq_len`, `compressor`, `total_seq_len`
- [x] 4.2 Implement `MlxCompressedCache::new(n_kv_heads, head_dim, max_seq_len, config)`
- [x] 4.3 Implement `append(&mut self, k: &Tensor, v: &Tensor)` — bridge to MLX, compress, concatenate, eval
- [x] 4.4 Implement `k()` / `v()` — decompress, bridge back to Candle
- [x] 4.5 Implement `current_seq_len()`, `reset()`, `set_len()`
- [x] 4.6 Write unit tests: append single token, append multiple tokens, reset, set_len truncation, shape correctness

## 5. KvCache Enum Integration

- [x] 5.1 Add `MlxCompressed` variant to `KvCache` enum in `kv_cache/mod.rs` gated by `#[cfg(feature = "mlx")]`
- [x] 5.2 Add `MlxCompressed` match arms to `KvCache::k()`, `KvCache::v()`, `KvCache::append()`, `KvCache::current_seq_len()`, `KvCache::reset()`, `KvCache::set_len()`, `KvCache::try_set_len()`
- [x] 5.3 Add `KvCache::new_mlx_compressed()` constructor
- [x] 5.4 Add `KvCache::is_mlx_compressed()` predicate
- [x] 5.5 Add `MlxCompressed` handling in `NormalCache::new_with_mlx_compression()`
- [x] 5.6 Add `NormalCache::apply_mlx_compression()` conversion method
- [x] 5.7 Add `MlxCompressed` handling in `clone_in_cache()` / `clone_out_cache()` for continuous batching
- [x] 5.8 Add `MlxCompressed` handling in `HybridCache::apply_mlx_compression()` for attention layers

## 6. Documentation & CI

- [x] 6.1 Update `docs/CARGO_FEATURES.md` — add `mlx` row to quick reference table, requirements section, recommended combinations
- [x] 6.2 Add `mlx` feature to docs/CARGO_FEATURES.md conflicting features section (note: no conflicts with metal/cuda)
- [x] 6.3 Create `docs/mlx-mistralrs-design.md` with the design rationale (based on the design artifact, for repository documentation)
- [x] 6.4 Run `cargo check` without features to verify no regressions
- [x] 6.5 Run `cargo check --features kvcache-compression` to verify no regressions to existing TurboQuant path
- [ ] 6.6 Run `cargo clippy --features mlx` and fix any warnings
- [ ] 6.7 Run `cargo test --features mlx -p mistralrs-core` — all MLX unit tests pass
