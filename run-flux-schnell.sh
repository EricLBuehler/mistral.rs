#!/bin/bash

# FLUX.1 Schnell - Fast image generation
# Using parking-lot-scheduler for better performance

cargo run --release --features metal,parking-lot-scheduler -p mistralrs-server -- \
  --port 1234 \
  --isq Q8_0 \
  diffusion \
  -m black-forest-labs/FLUX.1-schnell \
  -a flux
