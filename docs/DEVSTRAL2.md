# DRAFT — Devstral Small 2 Quickstart (Mistral.rs)

This draft documents how to run the Devstral-Small-2-24B-Instruct-2512 model with the Mistral.rs server using the Devstral-specific chat template and recommended defaults. Adjust paths/flags to match your environment.

## Prerequisites
- CUDA-capable GPU; tested with CUDA 12.x.
- Rust toolchain installed (2021 edition), and `cargo` in PATH.
- Hugging Face access token in `HF_TOKEN` (export before running).
- Optional: `module` system for loading CUDA/GCC (the helper script does this automatically).

## Fast path: use the helper script
Run the bundled script which sets sane defaults and passes through extra server args:

```bash
HF_TOKEN=xxxxx ./devstral2small.sh --pa-cache-type f8e4m3
```

Key defaults inside the script:
- Model: `mistralai/Devstral-Small-2-24B-Instruct-2512`
- Features: `cuda flash-attn`
- Port: `1234`
- Context: `262144` tokens (`--pa-ctxt-len`), topology: `topologies/devstral_small2_24b_default.yml`
- Sampling fallbacks if the client omits them: `--temperature 0.15`, `--top-p 0.9`, `--min-p 0.01` (and `--top-k` if set)
- Chat template: `chat_templates/devstral_fixed.jinja`

Examples:
```bash
# Minimal (uses defaults)
./devstral2small.sh --pa-cache-type f8e4m3

# Override sampling defaults
./devstral2small.sh --pa-cache-type f8e4m3 --temperature 0.3 --top-p 0.9 --top-k 64 --min-p 0.05

# Disable topology (advanced / debugging)
./devstral2small.sh --no-topology --pa-cache-type f8e4m3
```

## Manual run (no script)
If you prefer to invoke the server directly:

```bash
export HF_TOKEN=xxxxx
cargo build --release --package mistralrs-server --features "cuda flash-attn"

./target/release/mistralrs-server \
  --port 1234 \
  --jinja-explicit chat_templates/devstral_fixed.jinja \
  --pa-ctxt-len 262144 \
  --pa-cache-type f8e4m3 \
  --max-seqs 1 \
  --temperature 0.15 \
  --top-p 0.9 \
  --min-p 0.01 \
  run -m mistralai/Devstral-Small-2-24B-Instruct-2512 \
      --topology topologies/devstral_small2_24b_default.yml
```

Notes:
- Change `--pa-cache-type` to match your KV cache format (e.g., `auto`, `f16`, `f8e4m3`).
- To limit VRAM during prefill, set `MISTRALRS_PREFILL_CHUNK_SIZE` (the script exposes `--prefill-chunk-size`).
- If you need explicit default instructions for Responses API clients, export `MISTRALRS_DEFAULT_INSTRUCTIONS` (the script can auto-fill a sensible default).

## Troubleshooting
- “unexpected argument --temperature”: ensure you’re on this branch; the server now accepts fallback sampling flags.
- Topology errors: verify the YAML path or use `--no-topology` to disable device mapping during debugging.
- OOM during prefill: lower `CONTEXT_TOKENS` or set `MISTRALRS_PREFILL_CHUNK_SIZE=2048`.

