#!/bin/bash

# FLUX.1 Dev - Best quality image generation
# Using parking-lot-scheduler for better performance

cargo run --release --features metal,parking-lot-scheduler -p mistralrs-server -- \
  --port 1234 \
  diffusion \
  -m black-forest-labs/FLUX.1-dev \
  -a flux