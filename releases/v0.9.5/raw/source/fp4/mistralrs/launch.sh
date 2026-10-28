#!/usr/bin/env bash
set -euo pipefail
export NVCC=/usr/local/cuda-13.3/bin/nvcc
export CUDA_ROOT=/usr/local/cuda-13.3
export CUDA_HOME=/usr/local/cuda-13.3
export CUDA_TOOLKIT_PATH=/usr/local/cuda-13.3
export CUTILE_TILEIRAS_PATH=/usr/local/cuda-13.3/bin/tileiras
export PATH=/usr/local/cuda-13.3/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.3/compat:/usr/local/cuda-13.3/lib64
export RUST_BACKTRACE=1
exec /home/ericbuehler/mistral.rs/target/release/mistralrs serve \
  -m /home/ericbuehler/.cache/huggingface/hub/models--nvidia--Qwen3.8-Flash-Next-FP4-Local/snapshots/b9ce18a6887e18c089fdea8509a33e63429a3300 \
  --host 127.0.0.1 --port 8239 --dtype bf16 \
  --max-model-len 16384 --max-seqs 1 --max-num-batched-tokens 4096 \
  --max-prefill-chunk-tokens 4096 --prefix-cache-n 0 --kv-cache 2GiB \
  --no-ui --disable-access-log "$@"
