#!/bin/bash

# Phi-4 Full Precision (no quantization)
# Uses the already-downloaded model (~27GB)
# Best quality, your 64GB RAM can handle it
# Using parking-lot-scheduler for better performance

cargo run --release --features metal,parking-lot-scheduler -p mistralrs-server -- \
  --port 1234 \
  --paged-attn \
  plain \
  -m microsoft/Phi-4 \
  -a phi4 \
  --max-num-seqs 4
