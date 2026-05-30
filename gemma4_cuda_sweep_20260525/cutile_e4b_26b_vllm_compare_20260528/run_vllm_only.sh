#!/usr/bin/env bash
# vLLM-only rerun (mistral.rs numbers already captured). Clean-GPU standalone runs.
set -uo pipefail

ROOT=/home/ericbuehler/mistral.rs
OUT="$ROOT/gemma4_cuda_sweep_20260525/cutile_e4b_26b_vllm_compare_20260528"
VLLM_PY=/home/ericbuehler/vllm/.venv-bench/bin/python
DRIVER="$ROOT/gemma4_cuda_sweep_20260525/26b_bf16_vllm_resweep_20260526/run_vllm_latency_delta.py"
E4B=google/gemma-4-E4B-it
M26B="$ROOT/../hf_models/gemma4_26b_a4b"

echo "### vLLM E4B BF16 (gpu-mem-util 0.85)"
( cd /home/ericbuehler/vllm && "$VLLM_PY" "$DRIVER" \
  --model "$E4B" --out "$OUT/vllm_e4b_bf16_latency_delta.json" \
  --gpu-memory-utilization 0.85 --max-model-len 20000 --repeats 3 ) \
  > "$OUT/vllm_e4b_bf16.log" 2>&1
echo "E4B exit=$?"

echo "### vLLM 26B-A4B BF16 (gpu-mem-util 0.60)"
( cd /home/ericbuehler/vllm && "$VLLM_PY" "$DRIVER" \
  --model "$M26B" --out "$OUT/vllm_26b_a4b_bf16_latency_delta.json" \
  --gpu-memory-utilization 0.60 --max-model-len 20000 --repeats 3 ) \
  > "$OUT/vllm_26b_a4b_bf16.log" 2>&1
echo "26B exit=$?"
echo "VLLM RERUN DONE"
