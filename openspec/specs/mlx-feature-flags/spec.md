## ADDED Requirements

### Requirement: Workspace mlx-rs dependency
The workspace root `Cargo.toml` SHALL declare `mlx-rs` as a workspace dependency under `[target.'cfg(target_os = "macos")'.dependencies]` with an exact version pin (`=0.25.3`). The dependency MUST be optional.

#### Scenario: Dependency declared for macOS only
- **WHEN** the workspace `Cargo.toml` is parsed on Linux
- **THEN** `mlx-rs` is not resolved or downloaded

#### Scenario: Exact version pin prevents unintended upgrades
- **WHEN** `cargo update` is run
- **THEN** `mlx-rs` stays at version 0.25.3 (not upgraded to 0.26+)

### Requirement: mlx feature flag in mistralrs-core
`mistralrs-core/Cargo.toml` SHALL define an `mlx` feature that enables `dep:mlx-rs` and implies the `kvcache-compression` feature. The feature MUST be gated to macOS via `cfg(target_os = "macos")` at the code level (not in `Cargo.toml` feature definitions, since Cargo features are platform-agnostic).

#### Scenario: Enabling mlx enables kvcache-compression
- **WHEN** `cargo build --features mlx` is run on macOS
- **THEN** the `kvcache-compression` feature is also active and `mistralrs-kvcache-compression` is compiled

#### Scenario: mlx code compiles out on Linux
- **WHEN** `cargo build --features mlx` is run on Linux
- **THEN** all `#[cfg(all(feature = "mlx", target_os = "macos"))]` code is excluded and the build succeeds (mlx-rs dependency is not pulled)

#### Scenario: Build without mlx feature succeeds unchanged
- **WHEN** `cargo build` is run without the `mlx` feature
- **THEN** no mlx-rs code is compiled and behavior is identical to the current codebase

### Requirement: mlx feature propagation across workspace crates
The `mlx` feature SHALL propagate through the workspace crate dependency chain following the established pattern used by `metal`, `cuda`, and `kvcache-compression` features. Specifically:
- `mistralrs-cli`: `mlx = ["mistralrs-core/mlx", "mistralrs-server-core/mlx"]`
- `mistralrs-server-core`: `mlx = ["mistralrs-core/mlx"]`
- `mistralrs` (SDK crate): `mlx = ["mistralrs-core/mlx"]`
- `mistralrs-pyo3`: `mlx = ["mistralrs-core/mlx"]`
- `mistralrs-server`: `mlx = ["mistralrs-core/mlx", "mistralrs-server-core/mlx"]`

#### Scenario: CLI enables mlx for core and server-core
- **WHEN** `cargo build -p mistralrs-cli --features mlx` is run
- **THEN** both `mistralrs-core` and `mistralrs-server-core` have the `mlx` feature active

#### Scenario: Python SDK enables mlx for core
- **WHEN** `maturin build --features mlx` is run for `mistralrs-pyo3`
- **THEN** `mistralrs-core` has the `mlx` feature active

### Requirement: mlx feature does not conflict with metal or cuda
The `mlx` feature MUST be independent of the `metal` and `cuda` features. Users MUST be able to enable `mlx` alongside `metal` (both use Metal but through different binding paths). Users MUST NOT be required to enable `metal` to use `mlx`. The `mlx` feature MUST NOT activate any CUDA dependencies.

#### Scenario: mlx and metal enabled together
- **WHEN** `cargo build --features "mlx metal"` is run on macOS
- **THEN** the build succeeds with both Candle Metal forward pass and MLX KV cache backend active

#### Scenario: mlx without metal
- **WHEN** `cargo build --features mlx` is run on macOS (without `metal`)
- **THEN** the build succeeds — Candle runs on CPU for the forward pass, MLX handles KV cache compression on Metal

#### Scenario: mlx does not activate cuda
- **WHEN** `cargo build --features mlx` is run
- **THEN** no CUDA-related dependencies are compiled

### Requirement: CARGO_FEATURES.md documentation updated
The `docs/CARGO_FEATURES.md` file SHALL be updated to document the `mlx` feature including: description, requirements (macOS, Apple Silicon, cmake), what it enables, relationship to `kvcache-compression`, and recommended combinations.

#### Scenario: Feature table includes mlx entry
- **WHEN** the CARGO_FEATURES.md quick reference table is read
- **THEN** there is an `mlx` row with description "MLX KV cache acceleration", platform "macOS", and requires "Apple Silicon, cmake"

#### Scenario: Recommended combinations include mlx
- **WHEN** the recommended combinations table is read
- **THEN** Apple Silicon row includes `metal accelerate mlx` as an option
