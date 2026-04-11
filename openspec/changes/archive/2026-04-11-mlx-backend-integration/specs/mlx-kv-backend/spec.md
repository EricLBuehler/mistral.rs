## ADDED Requirements

### Requirement: Candle-to-MLX tensor bridge
The system SHALL provide functions `candle_to_mlx` and `mlx_to_candle` that convert tensors between Candle and MLX array representations. The bridge MUST support F32, F16, and BF16 dtypes. The bridge MUST preserve tensor shape and numerical values (bitwise identical for lossless dtypes). All bridge functions MUST be gated by `#[cfg(feature = "mlx")]`.

#### Scenario: Convert F16 Candle tensor to MLX array
- **WHEN** `candle_to_mlx` is called with a Candle tensor of dtype F16 and shape `[1, 8, 1, 128]`
- **THEN** the returned MLX Array has the same shape and contains identical F16 values

#### Scenario: Round-trip preserves values
- **WHEN** a Candle tensor is converted to MLX via `candle_to_mlx` and back via `mlx_to_candle`
- **THEN** the resulting Candle tensor is bitwise identical to the original

#### Scenario: Unsupported dtype returns error
- **WHEN** `candle_to_mlx` is called with a Candle tensor of dtype U8
- **THEN** the function returns an error indicating the dtype is unsupported

### Requirement: MLX compressed KV cache storage
The system SHALL provide an `MlxCompressedCache` type that stores K/V tensors as MLX arrays with optional TurboQuant compression. The cache MUST accept Candle tensors via `append()` (bridging internally) and return Candle tensors via `get()` (bridging back). The cache MUST be integrated as a `KvCache::MlxCompressed` enum variant.

#### Scenario: Append single token to empty cache
- **WHEN** `append()` is called on an empty `MlxCompressed` cache with K and V tensors of shape `[1, n_kv_heads, 1, head_dim]`
- **THEN** the cache stores the token and `current_seq_len()` returns 1

#### Scenario: Append and retrieve preserves shape
- **WHEN** 10 tokens are appended sequentially and then `k()` and `v()` are called
- **THEN** the returned tensors have shape `[1, n_kv_heads, 10, head_dim]` and are on the same Candle device as the input tensors

#### Scenario: Reset clears all cached data
- **WHEN** `reset()` is called on a cache with stored tokens
- **THEN** `current_seq_len()` returns 0 and `k()` / `v()` return `None`

#### Scenario: set_len truncates cache
- **WHEN** `set_len(5)` is called on a cache with 10 tokens
- **THEN** `current_seq_len()` returns 5 and retrieved tensors have sequence length 5

### Requirement: MLX TurboQuant compress and decompress
The system SHALL provide an `MlxTurboQuantCompressor` that implements WHT rotation followed by 4-bit PolarQuant scalar quantization on MLX arrays. Compression MUST produce packed uint8 arrays (two 4-bit indices per byte). Decompression MUST reconstruct the original tensor shape. The compress→decompress round-trip MUST achieve cosine similarity > 0.99 with the uncompressed original for typical KV cache distributions.

#### Scenario: Compress reduces memory footprint
- **WHEN** `compress()` is called on an MLX array of shape `[seq, heads, head_dim]` with F32 dtype
- **THEN** the returned array has shape `[seq, heads, head_dim/2]` with uint8 dtype (4x smaller than F32, 2x smaller than F16)

#### Scenario: Decompress restores original shape
- **WHEN** `decompress()` is called on a compressed array of shape `[seq, heads, head_dim/2]`
- **THEN** the returned array has shape `[seq, heads, head_dim]`

#### Scenario: Round-trip quality meets threshold
- **WHEN** a random F32 tensor sampled from N(0,1) is compressed and then decompressed
- **THEN** the cosine similarity between original and reconstructed is > 0.99

#### Scenario: WHT matrix is precomputed
- **WHEN** `MlxTurboQuantCompressor::new()` is called with head_dim=128
- **THEN** the WHT matrix is computed once and stored as an on-device MLX array, not recomputed on each compress/decompress call

#### Scenario: Non-power-of-two head_dim rejected
- **WHEN** `MlxTurboQuantCompressor::new()` is called with head_dim=96
- **THEN** the function returns an error (WHT requires power-of-two dimensions)

### Requirement: MlxCompressed integrates with NormalCache
The `NormalCache` type MUST support creating and managing `MlxCompressed` cache layers. The system SHALL provide `NormalCache::new_with_mlx_compression()` and `NormalCache::apply_mlx_compression()` factory/conversion methods, following the same pattern as the existing `new_with_compression()` and `apply_compression()`.

#### Scenario: Create NormalCache with MLX compression
- **WHEN** `NormalCache::new_with_mlx_compression(n_layers, config)` is called
- **THEN** all layers use `KvCache::MlxCompressed` variants

#### Scenario: Convert existing NormalCache to MLX compression
- **WHEN** `apply_mlx_compression(config)` is called on a `NormalCache` with `Normal` caches
- **THEN** all `Normal` variants are replaced with `MlxCompressed` variants preserving existing cached data

#### Scenario: MlxCompressed participates in clone_in/clone_out
- **WHEN** `clone_in_cache()` and `clone_out_cache()` are called on sequences using `MlxCompressed` caches
- **THEN** the cache state is correctly saved and restored for continuous batching

### Requirement: MLX eval boundaries prevent lazy eval stalls
The system MUST call `array.eval()` at the end of `append()` and at the start of `get()` / `k()` / `v()` to force MLX lazy computation and prevent unbounded memory growth from deferred operations.

#### Scenario: Append forces evaluation
- **WHEN** `append()` stores a compressed MLX array
- **THEN** `eval()` is called on the stored array before `append()` returns

#### Scenario: Get forces evaluation before bridge
- **WHEN** `get()` decompresses an MLX array to return a Candle tensor
- **THEN** `eval()` is called on the decompressed array before `mlx_to_candle` conversion
