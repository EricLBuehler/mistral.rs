<!-- AGENTS.md: Guidance for AI agents to navigate, build, test, and contribute to this repository -->
# AGENTS

This file provides instructions for AI agents to understand the layout of the `mistral.rs` repository, run builds/tests, and follow project conventions.

## Repository Structure

- `/mistralrs/`           : Main Rust crate (text & multimodal inference API)
- `/mistralrs-core/`      : Core inference logic and tensor operations (text models)
- `/mistralrs-vision/`    : Image processing utilities (resizing, preprocessing for multimodal models)
- `/mistralrs-quant/`     : Quantization support (ISQ, GGUF, GPTQ, AWQ, FP8, HQQ, etc.)
- `/mistralrs-paged-attn/`: PagedAttention implementation
- `/mistralrs-pyo3/`      : Python bindings (PyO3)
- `/mistralrs-cli/`       : Unified CLI binary (commands: run, serve, bench, from-config)
- `/mistralrs-server-core/`: Shared server core logic
- `/mistralrs-bench/`     : (Deprecated) Use `mistralrs bench` instead
- `/docs/`                : Markdown documentation for models, features, and guides
- `/examples/`            : Usage examples (Rust, Python, server samples, notebooks)
- `/chat_templates/`      : Chat formatting templates (JSON/Jinja)
- `/scripts/`             : Utility scripts (e.g., AWQ conversion)
  
## Feature Organization

Mistral.rs supports multiple model types and advanced features via dedicated crates and CLI subcommands:

- **Text Inference**
  - Crate: `mistralrs-core` (low-level ops), `mistralrs` (API wrapper)
  - CLI: `mistralrs run -m <model>` or `mistralrs serve -m <model>` (auto-detects model type)
  - Docs: `docs/SAMPLING.md`, `docs/TOOL_CALLING.md`
- **Multimodal Models**
  - Crate: `mistralrs-vision`
  - CLI: `mistralrs run -m <model>` (auto-detects multimodal models)
  - Docs: `docs/MULTIMODAL_MODELS.md`, `docs/IMAGEGEN_MODELS.md`, `docs/IMATRIX.md`
- **Diffusion Models**
  - CLI: `mistralrs run -m <model>` (auto-detects diffusion models)
  - Docs: `docs/FLUX.md`
- **Speech Models**
  - CLI: `mistralrs run -m <model>` (auto-detects speech models)
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
4. Or build/install only the CLI binary:
   ```bash
   cargo build --release --package mistralrs-cli --features "<features>"
   cargo install --path mistralrs-cli --features "<features>"
   ```

## Models

When integrating a new model, make sure it respects all of the varbuilder `.pp` calls. In Candle, a VarBuilder maintains an internal path vector that acts like a “current working directory” for model weights; every call to pp("sub") (alias for push_prefix) clones the builder and appends sub, so successive calls accumulate a dotted prefix such as transformer.h.0 while leaving the original builder untouched . When you eventually call get(...), Candle joins that prefix with the tensor name (prefix + "." + name) and looks it up in the checkpoint backend, producing keys that exactly match the dot-separated names emitted by PyTorch’s state_dict/named_parameters, which means PyTorch-trained weights can be loaded without any renaming  ￼. This lets you recreate the PyTorch module tree in Rust by “walking” it: e.g. vb.pp("word_embeddings") grabs word_embeddings.*, while a chain like vb.pp("encoder").pp("layers").pp(i.to_string()) targets keys such as encoder.layers.0.*, exactly as shown in community tutorials porting Transformers models to Candle  ￼. As one maintainer put it, the prefix system lets you “cd” around the parameter hierarchy, giving a lightweight namespace mechanism that keeps Candle fully compatible with PyTorch naming conventions while remaining ergonomic to use.

You should also look for a model.safetensors.index.json file for the model at hand to verify correct structure.

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

You should *always* run `cargo check`/`cargo c` before returning to make sure code compiles. If code does not compile, only make edits.

Avoid returning TODOs.

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
- Run CLI:
  ```bash
  mistralrs run -m <model>        # Interactive mode
  mistralrs serve -p 1234 -m <model>  # Server mode
  mistralrs bench -m <model>      # Benchmarking
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

## Code Style

**Comments.** Default to none. Only add when the *why* isn't obvious from the code: hidden constraints, invariants, surprising edge cases, references to a spec/HF source. Never paraphrase what the next line does, never restate the function name, never narrate steps.

- One line each, up to ~120 cols. No multi-paragraph `///` blocks, no bulleted lists in doc comments, no `// === Section ===` or `// ── Section ──` banners.
- ASCII only. No em-dashes (`—`), en-dashes (`–`), ellipses (`…`), smart quotes, or box-drawing characters. Use `--`, `...`, `"`, `'`.
- Don't reference the current task / PR / fix / commit in comments — that belongs in the PR description and rots as the codebase evolves.
- Trailing inline annotations like `// already sent above` are fine when terse.
- Tone matches existing files: terse, casual, sometimes fragmented. Look at older code for the house style before writing new doc comments.

**Magic values.** Hoist durations, sizes, sentinels, and other constants to named `const`s at the top of the file. A sentinel value that crosses module boundaries (e.g. one place sets `Some(0)`, another checks for it) must be a `pub const`, not a literal both sides happen to share.

**Function shape.** When a function passes 6+ args, prefer wrapping the invariants in a small context struct (e.g. `DispatchCtx<'a>`). Don't add error handling, fallbacks, or validation for scenarios that can't actually occur — trust internal code and framework guarantees. Don't add backwards-compatibility shims unless explicitly asked.

**Build hygiene.** `cargo check --workspace` and `cargo clippy -p <touched-crate> -- -D warnings` must both be clean before reporting work as done. Fix the lint, don't suppress it.
 
---
*This AGENTS.md file is intended solely to improve AI-driven assistance and does not affect runtime behavior.*