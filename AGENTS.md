<!-- AGENTS.md: Guidance for AI agents to navigate, build, test, and contribute to this repository -->
# AGENTS

This file provides instructions for AI agents to understand the layout of the `mistral.rs` repository, run builds/tests, and follow project conventions.

## Repository Structure

- `/mistralrs/`           : Main Rust crate (text & multimodal inference API)
- `/mistralrs-core/`      : Core inference logic and tensor operations (text models)
- `/mistralrs-vision/`    : Vision inference support (image-based inputs & vision-enabled models)
- `/mistralrs-quant/`     : Quantization support (ISQ, GGUF, GPTQ, AWQ, FP8, HQQ, etc.)
- `/mistralrs-paged-attn/`: PagedAttention implementation
- `/mistralrs-pyo3/`      : Python bindings (PyO3)
- `/mistralrs-server/`    : CLI & OpenAI-compatible HTTP server (subcommands: run/vision-plain, diffusion, speech)
- `/mistralrs-server-core/`: Shared server core logic
- `/mistralrs-web-chat/`  : Web chat application (static assets & backend integration)
- `/mistralrs-bench/`     : Benchmarking tools
- `/docs/`                : Markdown documentation for models, features, and guides
- `/examples/`            : Usage examples (Rust, Python, server samples, notebooks)
- `/chat_templates/`      : Chat formatting templates (JSON/Jinja)
- `/scripts/`             : Utility scripts (e.g., AWQ conversion)
  
## Feature Organization

Mistral.rs supports multiple model types and advanced features via dedicated crates and CLI subcommands:

- **Text Inference**
  - Crate: `mistralrs-core` (low-level ops), `mistralrs` (API wrapper)
  - CLI: `run` / `plain` subcommand in `mistralrs-server`
  - Docs: `docs/SAMPLING.md`, `docs/TOOL_CALLING.md`
- **Vision Models**
  - Crate: `mistralrs-vision`
  - CLI: `vision-plain` subcommand
  - Docs: `docs/VISION_MODELS.md`, `docs/IMAGEGEN_MODELS.md`, `docs/IMATRIX.md`
- **Diffusion Models**
  - CLI: `diffusion` subcommand
  - Docs: `docs/FLUX.md`
- **Speech Models**
  - CLI: `speech` subcommand
  - Docs: `docs/DIA.md`
- **Quantization & ISQ**
  - Crate: `mistralrs-quant`
  - Docs: `docs/QUANTS.md`, `docs/ISQ.md`
  - Conversion Script: `scripts/convert_awq_marlin.py`
- **Paged Attention**
  - Crate: `mistralrs-paged-attn`
  - Docs: `docs/PAGED_ATTENTION.md`
- **Adapters & LoRA/X-LoRA**
  - Docs: `docs/ADAPTER_MODELS.md`, `docs/LORA_XLORA.md`
- **Mixture of Experts (AnyMoE)**
  - Docs: `docs/ANYMOE.md`

## Building

1. Install Rust via rustup (Rust 2021 edition).
2. Choose optional features (e.g., `cuda`, `flash-attn`, `cudnn`, `metal`, `mkl`, `accelerate`).
3. Build the entire workspace:
   ```bash
   cargo build --workspace --release --features "<features>"
   ```
4. Or build/install only the server binary:
   ```bash
   cargo build --release --package mistralrs-server --features "<features>"
   cargo install --path mistralrs-server --features "<features>"
   ```

## Testing

- Core test suite (requires HF token for some tests):
  ```bash
  export HF_TOKEN=<your_token>  # or TESTS_HF_TOKEN for CI parity
  cargo test -p mistralrs-core -p mistralrs-quant -p mistralrs-vision
  ```
- Run all tests across workspace (may skip some crates without tests):
  ```bash
  cargo test --workspace
  ```

## Formatting & Linting

- Format all Rust code:
  ```bash
  cargo fmt --all
  make fmt       # also formats Python/CUDA/C++ files via ruff, clang-format
  ```
- Lint with Clippy:
  ```bash
  cargo clippy --workspace --tests --examples -- -D warnings
  ```

## Documentation

- Generate Rust docs for all crates:
  ```bash
  cargo doc --workspace
  ```
- Preview at `target/doc/` or publish to GitHub Pages as configured.
- Refer to `/docs/` for in-depth markdown guides (e.g., DEVICE_MAPPING.md, TOOL_CALLING.md).

## Examples

- Rust examples: `mistralrs/examples/`
- Python examples: `examples/python/`
- Server samples: `examples/server/`
- Run Python scripts:
  ```bash
  python3 examples/python/<script>.py
  ```
- Run server/CLI:
  ```bash
  ./target/release/mistralrs-server -i <mode> -m <model> [options]
  ```

## CI Parity

The CI pipeline is defined in `.github/workflows/ci.yml` and includes:
  - `cargo check` for all targets
  - `cargo test` on core crates
  - `cargo fmt -- --check`
  - `cargo clippy -D warnings`
  - `cargo doc`
  - Typos check (`crate-ci/typos`)

## Contribution Conventions

- Follow Rust 2021 idioms, keep code minimal and focused.
- Update `/docs/` and examples when adding features or breaking changes.
- Add tests and examples for new functionality.
- Commit messages should be clear and follow conventional style where possible.
  ```
  feat(crate): describe new feature
  fix(crate): describe bug fix
  docs: update docs for ...
  ```
 
---
*This AGENTS.md file is intended solely to improve AI-driven assistance and does not affect runtime behavior.*