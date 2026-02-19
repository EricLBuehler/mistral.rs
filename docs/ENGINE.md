# Engine Internals

This document describes internal engine behaviors in mistral.rs.

## Overview

The mistral.rs engine manages model inference through a background thread pool. Each loaded model runs in its own engine thread, which handles request queuing, batching, and execution.

## Warmup Run

When a text or vision model is loaded in a multi-threaded runtime, mistral.rs automatically performs a warmup ("dummy") run:

- Sends a short completion request ("hello" with max 1 token) to initialize CUDA kernels and caches
- Logs "Beginning dummy run." when starting and "Dummy run completed in Xs." when finished
- Helps ensure more consistent performance for the first real user request
- Only runs for text and vision models (not diffusion/speech)

This warmup ensures that CUDA kernel compilation and memory allocation happens during model loading rather than during the first user request.

## Automatic Engine Recovery

If the inference engine thread dies unexpectedly (e.g., due to a panic), mistral.rs can automatically recover:

- Detects dead engine threads when sending requests
- Automatically reboots the engine using saved configuration
- Logs "Engine {model_id} is dead, rebooting" followed by "Successfully rebooted engine {model_id}"
- Preserves all original configuration including KV cache settings, prefix cache, and tool callbacks

This ensures high availability without manual intervention.

## Thread Model

Each model loaded in mistral.rs runs in its own dedicated engine thread:

1. **Main Thread**: Handles HTTP requests, CLI interaction, and dispatches work to engine threads
2. **Engine Threads**: Each loaded model has a dedicated thread for inference
3. **Background Workers**: Tokenization and other preprocessing can run in parallel

For multi-model setups, each model gets its own engine thread, allowing true parallel inference across different models.

## See Also

- [Multi-Model Support](multi_model/overview.md) - Load and manage multiple models
- [Configuration](CONFIGURATION.md) - Environment variables affecting engine behavior
- [PagedAttention](PAGED_ATTENTION.md) - Memory management for high throughput
