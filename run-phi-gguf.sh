#!/bin/bash

# Llama 3.2 1B with GGUF quantization (4-bit, ~1GB)
# Very fast and efficient for M1 Mac
# Using parking-lot-scheduler for better performance
# Note: Phi-4 GGUF has tensor naming issues

cargo run --release --features metal,parking-lot-scheduler -p mistralrs-server -- \
  --port 1234 \
  gguf \
  -m bartowski/Llama-3.2-1B-Instruct-GGUF \
  -f Llama-3.2-1B-Instruct-Q4_K_M.gguf
