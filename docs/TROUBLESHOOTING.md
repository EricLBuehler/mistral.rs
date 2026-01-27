# Troubleshooting

Common issues and solutions for mistral.rs.

## Debug Mode

Enable debug mode for more information:

```bash
MISTRALRS_DEBUG=1 mistralrs run -m <model>
```

Debug mode causes:
- If loading a GGUF or GGML model, outputs a file containing the names, shapes, and types of each tensor:
  - `mistralrs_gguf_tensors.txt` or `mistralrs_ggml_tensors.txt`
- Increased logging verbosity

## System Diagnostics

Run the built-in diagnostics tool:

```bash
mistralrs doctor
```

This checks your system configuration and reports any issues.

## Common Issues

### CUDA Issues

**Setting the CUDA compiler path:**
- Set the `NVCC_CCBIN` environment variable during build

**Error: `recompile with -fPIE`:**
- Some Linux distributions require compiling with `-fPIE`
- Set during build: `CUDA_NVCC_FLAGS=-fPIE cargo build --release --features cuda`

**Error: `CUDA_ERROR_NOT_FOUND` or symbol not found:**
- For non-quantized models, specify the data type to load and run in
- Use one of `f32`, `f16`, `bf16` or `auto` (auto chooses based on device)
- Example: `mistralrs run -m <model> -d auto`

**Minimum CUDA compute capability:**
- The minimum supported CUDA compute cap is **5.3**
- Set a specific compute cap with: `CUDA_COMPUTE_CAP=80 cargo build --release --features cuda`

### Metal Issues (macOS)

**Metal not found (error: unable to find utility "metal"):**

1. Install Xcode:
   ```bash
   xcode-select --install
   ```

2. Set the active developer directory:
   ```bash
   sudo xcode-select --switch /Applications/Xcode.app/Contents/Developer
   ```

**error: cannot execute tool 'metal' due to missing Metal toolchain**

1. Install Metal Toolchain:
   ```bash
   xcodebuild -downloadComponent MetalToolchain
   ```

**Disabling Metal kernel precompilation:**
- By default, Metal kernels are precompiled during build time for better performance
- To skip precompilation (useful for CI or when Metal is not needed):
  ```bash
  MISTRALRS_METAL_PRECOMPILE=0 cargo build --release --features metal
  ```

### Memory Issues

**Disabling mmap loading:**
- Set `MISTRALRS_NO_MMAP=1` to disable memory-mapped file loading
- Forces all tensor data into memory
- Useful if you're seeing mmap-related errors

**Out of memory errors:**
- Try using quantization: `--isq q4k` or `--isq q8_0`
- Use device mapping to offload layers: `-n 0:16;cpu:16`
- Reduce context length with PagedAttention: `--pa-context-len 4096`

### Model Loading Issues

**Model type not auto-detected:**
- If auto-detection fails, please [raise an issue](https://github.com/EricLBuehler/mistral.rs/issues)
- You can manually specify the architecture if needed

**Chat template issues:**
- Templates are usually auto-detected
- Override with: `-c /path/to/template.jinja`
- See [Chat Templates](CHAT_TOK.md) for details

## Getting Help

If you're still stuck:

- [Discord](https://discord.gg/SZrecqK8qw) - Community support
- [Matrix](https://matrix.to/#/#mistral.rs:matrix.org) - Alternative chat
- [GitHub Issues](https://github.com/EricLBuehler/mistral.rs/issues) - Bug reports and feature requests

When reporting issues, please include:
1. Output of `mistralrs doctor`
2. Full error message
3. Command you ran
4. Hardware (GPU model, OS)
