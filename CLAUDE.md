# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

mistral.rs is a blazing-fast LLM inference engine written in Rust. It supports text, vision, image generation, and speech models with Rust and Python SDKs, plus OpenAI HTTP and MCP APIs.

## Essential Commands

### Building
```bash
# Basic release build
cargo build --release

# With CUDA support (Linux)
cargo build --release --features "cuda flash-attn cudnn"

# With Metal support (macOS)
cargo build --release --features metal

# Install CLI binary
cargo install --path mistralrs-cli --features <features>
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
# Run interactive mode (model type auto-detected)
mistralrs run -m <model_id>

# Run with GGUF quantized model
mistralrs run --format gguf -m <repo> -f <file>

# Run server
mistralrs serve -p 1234 -m <model_id>

# Run server with web UI
mistralrs serve --ui -m <model_id>

# Run benchmarks
mistralrs bench -m <model_id>
```

## Models

When integrating a new model, make sure it respects all of the varbuilder `.pp` calls. In Candle, a VarBuilder maintains an internal path vector that acts like a “current working directory” for model weights; every call to pp("sub") (alias for push_prefix) clones the builder and appends sub, so successive calls accumulate a dotted prefix such as transformer.h.0 while leaving the original builder untouched . When you eventually call get(...), Candle joins that prefix with the tensor name (prefix + "." + name) and looks it up in the checkpoint backend, producing keys that exactly match the dot-separated names emitted by PyTorch’s state_dict/named_parameters, which means PyTorch-trained weights can be loaded without any renaming  ￼. This lets you recreate the PyTorch module tree in Rust by “walking” it: e.g. vb.pp("word_embeddings") grabs word_embeddings.*, while a chain like vb.pp("encoder").pp("layers").pp(i.to_string()) targets keys such as encoder.layers.0.*, exactly as shown in community tutorials porting Transformers models to Candle  ￼. As one maintainer put it, the prefix system lets you “cd” around the parameter hierarchy, giving a lightweight namespace mechanism that keeps Candle fully compatible with PyTorch naming conventions while remaining ergonomic to use.

You should also look for a model.safetensors.index.json file for the model at hand to verify correct structure.

## Architecture Overview

### Workspace Structure
- `mistralrs-core/` - Core inference engine, model implementations, pipelines
- `mistralrs-cli/` - Unified CLI binary (commands: run, serve, bench, from-config)
- `mistralrs-server-core/` - HTTP server routing, OpenAI API implementation
- `mistralrs-pyo3/` - Python SDK (PyO3 bindings)
- `mistralrs/` - Rust SDK (high-level crate)
- `mistralrs-vision/` - Vision model support
- `mistralrs-quant/` - Quantization implementations (ISQ, GGUF, GPTQ, etc.)
- `mistralrs-paged-attn/` - PagedAttention implementation
- `mistralrs-audio/` - Audio processing
- `mistralrs-mcp/` - Model Context Protocol client
- `mistralrs-bench/` - (Deprecated) Use `mistralrs bench` instead

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
5. Update CLI args in `mistralrs-cli/src/main.rs`

When adding new quantization methods:
1. Implement in `mistralrs-quant/src/`
2. Add to quantization loading logic in pipelines
3. Update documentation in `docs/QUANTIZATION.md`

### Important Files to Know

- `mistralrs-core/src/engine/mod.rs` - Main engine orchestration
- `mistralrs-core/src/pipeline/mod.rs` - Pipeline trait and common logic
- `mistralrs-server-core/src/routes.rs` - HTTP API endpoints
- `mistralrs-pyo3/src/lib.rs` - Python SDK entry point
- `mistralrs/examples/` - Usage examples for Rust SDK

### Pull Requests

Never include a "Test plan" section in PR descriptions.

### Testing Approach

You should *always* run `cargo check`/`cargo c` before returning to make sure code compiles. If code does not compile, only make edits.

Avoid returning TODOs.

- Unit tests are colocated with source files
- Integration tests in `tests/` directories
- Use `cargo test -p <crate>` to test specific components
- Python tests require building and installing the package first

### Common Pitfalls

1. **Feature Flags**: Many features are gated behind Cargo features. Always check what features are needed for your use case.
2. **Device Indices**: CUDA device selection uses 0-based indexing
3. **Chat Templates**: Models may need specific chat templates - check `chat_templates/` directory
4. **Quantization**: Different quantization methods have different hardware requirements
5. **Never use `Tensor::{from_vec,arange}` in hot loops**: `Tensor::{from_vec,arange}` with a GPU device causes a CPU-to-GPU sync. If you need a small tensor on GPU during forward, either precompute it at model init or start of forward pass.

### Vision/Audio Model Pitfalls

6. **Vision encoder attention must be bidirectional (non-causal)**:  `Sdpa.run_attention` with `flash_params: None` defaults to `causal = seq_len > 1` on the CUDA flash-attn path, which silently breaks vision/audio encoders. Always pass `FlashParams { causal: false, cumulative_seqlens_q: HashMap::new(), cumulative_seqlens_k: HashMap::new(), max_q: 0, max_k: 0 }` with `Some(&flash_params)` for any encoder that needs bidirectional attention. The empty `cumulative_seqlens` cause the flash backend to use the non-varlen kernel path, avoiding any tensor allocation in the forward pass.

7. **`torch.bucketize(right=True)` requires `Ok(i) => i + 1`**: Rust's `binary_search_by` returns `Ok(i)` at the found position (bisect_left semantics). For `right=True` (bisect_right), you must use `Ok(i) => i + 1` to insert after equal elements. `Err(i) => i` is correct for both.

8. **Mistral `consolidated.safetensors` stores Q/K weights with interleaved head dimensions**: When loading from Mistral-native `consolidated.safetensors` (as opposed to HF-converted `model.safetensors`), the Q and K projection weights use an interleaved layout within each head: `[x0, x_{d/2}, x1, x_{d/2+1}, ...]` instead of the sequential HF layout `[x0, x1, ..., x_{d/2-1}, x_{d/2}, ...]`. This means you must use `is_gptx=false` (GPT-J/adjacent-pair style) for `RotaryEmbedding`, NOT `is_gptx=true` (GPT-NeoX/half-split style). Using the wrong RoPE style produces completely wrong attention outputs (cosine similarity ~0.02 with reference). To diagnose: compare a Q or K weight tensor between `consolidated.safetensors` and `model.safetensors` — if they differ (cosine ~0.02), apply the un-interleave: `reshape(n_heads, head_dim/2, 2, dim).permute(0,2,1,3)` and verify cosine ~1.0.

9. **Causal Conv1d padding formula**: For causal convolution (left-pad only, no right-pad), the correct left padding is `effective_kernel_size - stride`, NOT `(kernel_size - 1) * dilation` (which is the total padding for non-causal). For example, with kernel_size=3, stride=2, dilation=1: left_pad = 3 - 2 = 1, not 2. Verify against the HF model's `VoxtralRealtimeCausalConv1d` or equivalent source.
