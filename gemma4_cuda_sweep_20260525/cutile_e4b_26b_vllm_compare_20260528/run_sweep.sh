#!/usr/bin/env bash
# Reprofile Gemma 4 E4B + 26B-A4B BF16 (cutile MoE backend, default on bf16/cuda) vs vLLM 0.21.0.
set -euo pipefail

ROOT=/home/ericbuehler/mistral.rs
OUT="$ROOT/gemma4_cuda_sweep_20260525/cutile_e4b_26b_vllm_compare_20260528"
BIN="$ROOT/target/release/mistralrs"
VLLM_PY=/home/ericbuehler/vllm/.venv-bench/bin/python
DRIVER="$ROOT/gemma4_cuda_sweep_20260525/26b_bf16_vllm_resweep_20260526/run_vllm_latency_delta.py"

E4B=google/gemma-4-E4B-it
M26B="$ROOT/../hf_models/gemma4_26b_a4b"
PLENS=128,512,2048,4096,8192,16384
DEPTHS=128,512,2048,4096,8192,16384
# warmup 3 (not 1): cutile MoE kernels JIT per-shape; 1 warmup leaves the JIT spike in a
# measured iteration (seen as huge stddev). 3 warmups absorb it so means reflect steady state.
COMMON="--iterations 3 --warmup 3 --paged-attn on --pa-context-len 20000 --max-seq-len 20000"

cd "$ROOT"

echo "### mistral.rs E4B BF16 prompt"
RUST_LOG=info "$BIN" bench -m "$E4B" --prompt-len $PLENS --gen-len 0 --depth 1 $COMMON \
  > "$OUT/mistralrs_e4b_bf16_prompt.txt" 2>&1
echo "### mistral.rs E4B BF16 decode"
RUST_LOG=info "$BIN" bench -m "$E4B" --prompt-len 128 --gen-len 256 --depth $DEPTHS $COMMON \
  > "$OUT/mistralrs_e4b_bf16_decode.txt" 2>&1

echo "### mistral.rs 26B-A4B BF16 prompt (cutile path)"
RUST_LOG=info "$BIN" bench -m "$M26B" --prompt-len $PLENS --gen-len 0 --depth 1 $COMMON \
  > "$OUT/mistralrs_26b_a4b_bf16_prompt.txt" 2>&1
echo "### mistral.rs 26B-A4B BF16 decode (cutile path)"
RUST_LOG=info "$BIN" bench -m "$M26B" --prompt-len 128 --gen-len 256 --depth $DEPTHS $COMMON \
  > "$OUT/mistralrs_26b_a4b_bf16_decode.txt" 2>&1

echo "### vLLM E4B BF16 (gpu-mem-util 0.85)"
( cd /home/ericbuehler/vllm && "$VLLM_PY" "$DRIVER" \
  --model "$E4B" --out "$OUT/vllm_e4b_bf16_latency_delta.json" \
  --gpu-memory-utilization 0.85 --max-model-len 20000 --repeats 3 ) \
  > "$OUT/vllm_e4b_bf16.log" 2>&1

echo "### vLLM 26B-A4B BF16 (gpu-mem-util 0.60)"
( cd /home/ericbuehler/vllm && "$VLLM_PY" "$DRIVER" \
  --model "$M26B" --out "$OUT/vllm_26b_a4b_bf16_latency_delta.json" \
  --gpu-memory-utilization 0.60 --max-model-len 20000 --repeats 3 ) \
  > "$OUT/vllm_26b_a4b_bf16.log" 2>&1

echo "ALL SWEEPS DONE"
