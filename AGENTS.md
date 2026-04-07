<!-- AGENTS.md: Guidance for AI agents to navigate, build, test, and contribute to this repository -->
# AGENTS

This file provides instructions for AI agents to understand the layout of the `mistral.rs` repository, run builds/tests, and follow project conventions.

## Repository Structure

- `/mistralrs/`           : Main Rust crate (text & multimodal inference API)
- `/mistralrs-core/`      : Core inference logic and tensor operations (text models)
- `/mistralrs-vision/`    : Image processing utilities (resizing, preprocessing for multimodal models)
- `/mistralrs-audio/`     : Audio processing
- `/mistralrs-quant/`     : Quantization support (ISQ, GGUF, GPTQ, AWQ, FP8, HQQ, etc.)
- `/mistralrs-paged-attn/`: PagedAttention implementation
- `/mistralrs-pyo3/`      : Python bindings (PyO3)
- `/mistralrs-cli/`       : Unified CLI binary (commands: run, serve, bench, from-config)
- `/mistralrs-server-core/`: Shared server core logic
- `/mistralrs-server/`    : Server binary (standalone, separate from CLI)
- `/mistralrs-serve/`     : Web UI and HTTP API (served by mistralrs-cli/mistralrs-server)
- `/mistralrs-mcp/`       : Model Context Protocol client
- `/mistralrs-macros/`    : Procedural macros for derive helpers
- `/mistralrs-web-chat/`  : (Deprecated) Use `mistralrs serve --ui` instead
- `/mistralrs-bench/`     : (Deprecated) Use `mistralrs bench` instead
- `/docs/`                : Markdown documentation for models, features, and guides
- `/examples/`            : Usage examples (Rust, Python, server samples, notebooks)
- `/chat_templates/`      : Chat formatting templates (JSON/Jinja)
- `/scripts/`             : Utility scripts (e.g., AWZ conversion)
  
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

### Building with CUDA on this machine

This machine has CUDA 13.2 at `E:\Cuda` and Visual Studio 2022 (v18) at `E:\Program Files\Microsoft Visual Studio\18\Community`. The shell session does NOT have MSVC in PATH by default — you must source `vcvars64.bat` first.

**The CCCL + MSVC preprocessor problem:** CUDA 13.2's CCCL library requires MSVC's conformant preprocessor (`/Zc:preprocessor`). MSVC defaults to the "traditional" preprocessor, causing a fatal `#error` in `<cuda/std/__cccl/preprocessor.h>`. The fix is to inject `--compiler-options /Zc:preprocessor` into every nvcc invocation. This is done via an nvcc wrapper script — see `nvcc_wrapper.bat` below. The wrapper is transparent to all CUDA builds (mistralrs-quant, mistralrs-core, candle-kernels) because `cudaforge` respects the `NVCC` env var.

**Local build scripts (DO NOT commit):**

`nvcc_wrapper.bat` — wraps nvcc, injecting `/Zc:preprocessor`:
```bat
@echo off
E:\Cuda\bin\nvcc.exe --compiler-options /Zc:preprocessor %*
```

`build.bat` — sets up MSVC toolchain, points NVCC to wrapper, runs cargo:
```bat
@echo off
call "E:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" >NUL 2>&1
set CUDA_COMPUTE_CAP=86
set NVCC=%~dp0nvcc_wrapper.bat
cargo build --release -p mistralrs-cli --features cuda %*
```

**Build from PowerShell (one-liner, no local scripts needed):**
```powershell
pwsh -NoProfile -Command 'cmd /c "call ""E:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"" >NUL 2>&1 && set CUDA_COMPUTE_CAP=86 && set NVCC=E:\Development\Rust\mistral.rs\nvcc_wrapper.bat && cargo build --release -p mistralrs-cli --features cuda 2>&1"'
```

**Key env vars:**
- `NVCC=<path_to_wrapper>` — tells `cudaforge` to use the wrapper instead of raw nvcc (required on this machine with CUDA 13.2 + MSVC)
- `CUDA_COMPUTE_CAP=86` — target GPU architecture (RTX 3070)
- `CCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING=1` — does NOT work reliably; the wrapper approach above is required instead

**Build only the CLI (no CUDA kernels):**
```bash
cargo build --release -p mistralrs-cli
```

**Deploy to Windows service:**
- Service name: `MistralRs` (managed by Servy at `C:\ProgramData\Servy`)
- Deploy dir: `E:\MistralRs\`
- Binary: `target\release\mistralrs.exe` → copy to `E:\MistralRs\mistralrs.exe`
- Start/stop requires admin elevation via Servy CLI or `Start-Process -Verb RunAs`
- Config: `E:\MistralRs\run_server.bat` — `mistralrs.exe serve --ui --idle-timeout-secs 1800 --models-dir "E:\MistralRs\models" -p 1234`
- Logs: `E:\MistralRs\service.log` / `E:\MistralRs\service_err.log`
- Models dir: `E:\MistralRs\models\` (contains llama-3.1-8b, mistral-7b-v03, phi-3.5-mini, qwen3-4b, qwen3-8b)
- UI override dir: `E:\MistralRs\ui\` (optional disk-based UI files)

**DO NOT commit:** `build.bat`, `deploy.ps1`, `nvcc_wrapper.bat` (local tooling)
**DO NOT touch:** `E:\MistralRs\` deployment directory (permission-blocked from git)

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
 
---
*This AGENTS.md file is intended solely to improve AI-driven assistance and does not affect runtime behavior.*