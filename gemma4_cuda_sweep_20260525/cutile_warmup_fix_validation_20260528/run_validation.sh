#!/usr/bin/env bash
# Validate the cutile warmup fix: model-warmup (engine thread + real shapes) should compile all
# kernels at init, so the bench (even with --warmup 1) sees ZERO JIT in measured forwards and
# tight variance. CUTILE_JIT_TIMING=1 prints per-compile timing so we can place each on the timeline.
set -uo pipefail

ROOT=/home/ericbuehler/mistral.rs
OUT="$ROOT/gemma4_cuda_sweep_20260525/cutile_warmup_fix_validation_20260528"
BIN="$ROOT/target/release/mistralrs"
M26B="$ROOT/../hf_models/gemma4_26b_a4b"
PLENS=128,512,2048,4096,8192,16384
DEPTHS=128,512,2048,4096,8192,16384
COMMON="--iterations 3 --warmup 1 --paged-attn on --pa-context-len 20000 --max-seq-len 20000"

cd "$ROOT"

echo "### 26B-A4B BF16 prompt (cutile, warmup-fix, JIT timing on)"
CUTILE_JIT_TIMING=1 RUST_LOG=info "$BIN" bench -m "$M26B" \
  --prompt-len $PLENS --gen-len 0 --depth 1 $COMMON \
  > "$OUT/mistralrs_26b_a4b_bf16_prompt.txt" 2>&1
echo "prompt exit=$?"

echo "### 26B-A4B BF16 decode (cutile, warmup-fix, JIT timing on)"
CUTILE_JIT_TIMING=1 RUST_LOG=info "$BIN" bench -m "$M26B" \
  --prompt-len 128 --gen-len 256 --depth $DEPTHS $COMMON \
  > "$OUT/mistralrs_26b_a4b_bf16_decode.txt" 2>&1
echo "decode exit=$?"
echo "VALIDATION DONE"
