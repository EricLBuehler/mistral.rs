# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

mistral.rs is a blazing-fast LLM inference engine written in Rust. It supports text, vision, image generation, and speech models with multiple APIs (Rust, Python, OpenAI HTTP, MCP).

## Essential Commands

### Building
```bash
# Basic release build
cargo build --release

# With CUDA support (Linux)
cargo build --release --features "cuda flash-attn cudnn"

# With Metal support (macOS)
cargo build --release --features metal

# Install server binary
cargo install --path mistralrs-server --features <features>
```

### Testing & Quality
```bash
# Run core tests
cargo test -p mistralrs-core -p mistralrs-quant -p mistralrs-vision

# Format code (uses rustfmt, ruff, clang-format)
make fmt

# Check formatting
cargo fmt --all -- --check

# Run clippy
cargo clippy --workspace --tests --examples -- -D warnings
```

### Running Models
```bash
# Run interactive mode with plain model
cargo run --release --features <features> -- -i plain -m <model_id> -a <arch>

# Run with GGUF quantized model
cargo run --release --features <features> -- -i gguf -f <file> -t <tokenizer>

# Run server
cargo run --release --features <features> -- --port 1234 <model_args>
```

## Architecture Overview

### Workspace Structure
- `mistralrs-core/` - Core inference engine, model implementations, pipelines
- `mistralrs-server/` - CLI binary entry point
- `mistralrs-server-core/` - HTTP server routing, OpenAI API implementation
- `mistralrs-pyo3/` - Python bindings (PyO3)
- `mistralrs/` - High-level Rust API
- `mistralrs-vision/` - Vision model support
- `mistralrs-quant/` - Quantization implementations (ISQ, GGUF, GPTQ, etc.)
- `mistralrs-paged-attn/` - PagedAttention implementation
- `mistralrs-audio/` - Audio processing
- `mistralrs-mcp/` - Model Context Protocol client
- `mistralrs-bench/` - Benchmarking tools

### Key Design Patterns

1. **Pipeline Architecture**: All models implement the `Pipeline` trait in `mistralrs-core/src/pipeline/mod.rs`. Different model types (Plain, GGUF, GGML, Vision) have their own pipeline implementations.

2. **Model Loading**: Models are loaded through `Loader` traits that handle different formats and quantizations. See `mistralrs-core/src/loader.rs`.

3. **Request Handling**: The server uses message passing with `MistralRs` struct managing a background thread pool. Requests flow through `mistralrs-core/src/engine/mod.rs`.

4. **Device Management**: Automatic and manual device mapping for multi-GPU setups handled in `mistralrs-core/src/device_map.rs`.

### Adding New Features

When adding new model architectures:
1. Implement the model in `mistralrs-core/src/models/`
2. Add pipeline support in `mistralrs-core/src/pipeline/`
3. Update model detection in `mistralrs-core/src/pipeline/normal.rs`
4. Add architecture enum variant in `mistralrs-core/src/lib.rs`
5. Update CLI args in `mistralrs-server/src/main.rs`

When adding new quantization methods:
1. Implement in `mistralrs-quant/src/`
2. Add to quantization loading logic in pipelines
3. Update documentation in `docs/QUANTIZATION.md`

### Important Files to Know

- `mistralrs-core/src/engine/mod.rs` - Main engine orchestration
- `mistralrs-core/src/pipeline/mod.rs` - Pipeline trait and common logic
- `mistralrs-server-core/src/routes.rs` - HTTP API endpoints
- `mistralrs-pyo3/src/lib.rs` - Python API entry point
- `mistralrs/examples/` - Usage examples for Rust API

### Testing Approach

- Unit tests are colocated with source files
- Integration tests in `tests/` directories
- Use `cargo test -p <crate>` to test specific components
- Python tests require building and installing the package first

### Common Pitfalls

1. **Feature Flags**: Many features are gated behind Cargo features. Always check what features are needed for your use case.
2. **Device Indices**: CUDA device selection uses 0-based indexing
3. **Chat Templates**: Models may need specific chat templates - check `chat_templates/` directory
4. **Quantization**: Different quantization methods have different hardware requirements