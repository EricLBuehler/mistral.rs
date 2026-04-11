## Why

On macOS with Apple Silicon, the TurboQuant KV cache compression pipeline (WHT + PolarQuant/QJL) runs on Candle's Metal backend, which lacks fused kernel dispatch — each operation is a separate Metal command buffer submission. Apple's MLX framework can fuse the entire compress/decompress pipeline into a single Metal kernel via `mlx_rs::compile!`, eliminating per-op dispatch overhead on the hottest path in long-context inference. By routing only the KV cache lifecycle (store → compress → retrieve → decompress) through MLX while keeping the forward pass on Candle+Metal, we get MLX's kernel fusion where it matters most without rewriting model implementations.

## What Changes

- **New `mlx` Cargo feature flag** across the workspace crates (`mistralrs-core`, `mistralrs-cli`, `mistralrs-server-core`, `mistralrs`, `mistralrs-pyo3`) — macOS-only, implies `kvcache-compression`
- **New `mistralrs-core/src/backend/mlx/` module** containing:
  - Candle-to-MLX tensor bridge (copy-based v1, zero-copy upgrade path documented)
  - `MlxKvCache` type storing K/V as MLX arrays with optional TurboQuant compression
  - `MlxTurboQuantCompressor` implementing WHT + 4-bit PolarQuant compress/decompress on MLX arrays
- **Modified KV cache dispatch** in `mistralrs-core/src/kv_cache/` to route through `MlxKvCache` when the `mlx` feature is active on macOS
- **New workspace dependency** on `mlx-rs = "=0.25.3"` (pinned, macOS-only target dependency)
- **No breaking changes** — the `mlx` feature is opt-in; all existing behavior is unchanged when disabled

## Capabilities

### New Capabilities
- `mlx-kv-backend`: MLX-backed KV cache storage and TurboQuant compression for macOS Apple Silicon, including Candle↔MLX tensor bridge, MLX array cache management, and MLX-native WHT/PolarQuant operations
- `mlx-feature-flags`: Workspace-wide `mlx` feature flag propagation with macOS-only conditional compilation, no conflicts with existing `metal`/`cuda`/`accelerate`/`kvcache-compression` features

### Modified Capabilities
<!-- No existing specs to modify — this is a new opt-in backend that leaves current behavior untouched -->

## Impact

- **Cargo.toml (workspace root)**: New `mlx-rs` workspace dependency under `[target.'cfg(target_os = "macos")'.dependencies]`
- **Cargo.toml (6 crates)**: New `mlx` feature flag in `mistralrs-core`, `mistralrs-cli`, `mistralrs-server-core`, `mistralrs`, `mistralrs-pyo3`, `mistralrs-kvcache-compression`
- **mistralrs-core/src/backend/**: New module tree (`mod.rs`, `mlx/mod.rs`, `mlx/bridge.rs`, `mlx/kv_cache.rs`, `mlx/turboquant.rs`)
- **mistralrs-core/src/kv_cache/**: Modified to add `KvCacheBackend` dispatch enum
- **Build system**: macOS CI needs `mlx` feature test job; Linux/Windows CI unaffected (feature compiles out)
- **Dependencies**: `mlx-rs =0.25.3` → `mlx-sys` → `mlx-c` (C library, built from source via cmake on first compile ~5-15 min)
- **Runtime**: No new runtime dependencies; MLX ships as a static library linked into the binary
- **Python SDK**: `mistralrs-mlx` pip package variant possible (future, not in scope)
