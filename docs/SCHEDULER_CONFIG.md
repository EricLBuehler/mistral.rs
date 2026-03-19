# Parking Lot Scheduler Configuration

The parking-lot scheduler in mistral.rs provides configurable thread pool management and resource-aware request scheduling. This document explains how to configure and tune the scheduler for optimal performance.

## Table of Contents

- [Overview](#overview)
- [Configuration File](#configuration-file)
- [Configuration Priority](#configuration-priority)
- [Configuration Options](#configuration-options)
- [Hardware-Specific Tuning](#hardware-specific-tuning)
- [CLI Overrides](#cli-overrides)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)

## Overview

The parking-lot scheduler uses the `prometheus_parking_lot` crate to provide:

- **Dedicated worker threads** for CPU/GPU-bound inference work
- **Resource tracking** (GPU VRAM / KV cache blocks)
- **Priority-based queue management**
- **Graceful degradation** under load
- **Configurable limits** and timeouts

This scheduler is enabled with the `parking-lot-scheduler` feature flag.

## Configuration File

Configuration is specified in YAML format with two main sections:

### Pool Configuration

Controls the thread pool settings:

```yaml
pool:
  worker_threads: 4           # Number of worker threads
  thread_stack_size: 2097152  # Stack size in bytes (2MB)
```

### Limits Configuration

Controls resource allocation and queue management:

```yaml
limits:
  max_units: 4096        # Maximum KV cache blocks
  max_queue_depth: 100   # Maximum queued requests
  timeout_secs: 300      # Request timeout in seconds
```

## Configuration Priority

Configuration values are resolved in this priority order (highest to lowest):

1. **CLI flags** (e.g., `--worker-threads 8`)
2. **`--scheduler-config <path>`** flag
3. **`MISTRALRS_SCHEDULER_CONFIG`** environment variable
4. **`~/.mistralrs-server/scheduler.yaml`** (default location)
5. **Built-in defaults** (if no config found)

This allows you to:
- Set base configuration in a file
- Override specific values via CLI for testing
- Use different configs for different environments

## Configuration Options

### `pool.worker_threads`

**Type:** Integer  
**Default:** Number of CPU cores (`num_cpus::get()`)  
**Range:** 1-1024

Number of dedicated worker threads for processing inference requests.

**Recommendations:**
- **M1/M2 Macs:** 4-8 (matches performance cores)
- **Server (single GPU):** 4-16
- **Server (multi-GPU):** 16-32
- **High-throughput:** Match or exceed physical core count

**Notes:**
- More threads ≠ better GPU performance
- GPU inference is bottlenecked by VRAM and compute, not thread count
- Optimal value usually matches CPU cores for batch scheduling

### `pool.thread_stack_size`

**Type:** Integer (bytes)  
**Default:** 2097152 (2MB)  
**Minimum:** 65536 (64KB)  
**Platform:** Native only (ignored on WASM)

Stack size allocated per worker thread.

**Recommendations:**
- Use default (2MB) for most models
- Increase if you encounter stack overflow with deep architectures
- No need to change unless you have specific issues

### `limits.max_units`

**Type:** Integer  
**Default:** 16384  
**Range:** 1+

Maximum resource units representing KV cache blocks available for allocation.

**Calculation:**
```
max_units ≈ (GPU_VRAM_MB / block_size_mb)
```

For 16-token blocks:
```
max_units ≈ (GPU_VRAM_MB / 16)
```

**Examples:**
| GPU Memory | Recommended max_units | Use Case |
|------------|----------------------|----------|
| 8 GB | 1024-2048 | Small models (Phi-3, Qwen-0.5B) |
| 16 GB | 4096-8192 | Medium models (Mistral-7B, Llama-8B) |
| 24 GB | 8192-16384 | Large models (Llama-13B) |
| 40 GB | 16384-32768 | Very large models |
| 80 GB | 32768-65536 | Massive models or high concurrency |

**Notes:**
- Leave headroom for model weights and activations
- Directly impacts memory usage
- Lower value = fewer concurrent requests but more stable

### `limits.max_queue_depth`

**Type:** Integer  
**Default:** 1000  
**Range:** 1-100000

Maximum number of requests that can be queued before new requests are rejected.

**Recommendations:**
- **Memory-constrained (M1/M2 Mac):** 50-100
- **Typical server:** 500-1000
- **High-throughput production:** 2000-5000

**Trade-offs:**
- **Higher depth:** More buffering, slower rejection, more memory for queued requests
- **Lower depth:** Faster rejection, less memory, tighter control

### `limits.timeout_secs`

**Type:** Integer (seconds)  
**Default:** 120 (2 minutes)  
**Range:** 1+

Maximum time a request can take before being cancelled.

**Recommendations:**
- **Short responses:** 60-120 seconds
- **Long-context generation:** 300-600 seconds
- **Very long documents:** 900-1200 seconds
- **Slow models:** 600+ seconds

**Guidelines:**
- Set to 2-3x your expected max generation time
- Too short: legitimate requests get cancelled
- Too long: stuck requests block resources

## Hardware-Specific Tuning

### M1/M2 Mac (16GB-32GB Unified Memory)

```yaml
pool:
  worker_threads: 4
  thread_stack_size: 2097152
limits:
  max_units: 4096
  max_queue_depth: 100
  timeout_secs: 300
```

**Rationale:**
- 4 threads matches performance cores
- 4096 units ≈ 4GB KV cache for small models
- Lower queue depth for memory constraints

### Multi-GPU Server (4x A100 80GB)

```yaml
pool:
  worker_threads: 16
  thread_stack_size: 2097152
limits:
  max_units: 32768
  max_queue_depth: 2000
  timeout_secs: 600
```

**Rationale:**
- More workers for parallel inference across GPUs
- 32GB+ KV cache capacity
- Higher queue depth for throughput

### High-Throughput Production (8+ GPUs)

```yaml
pool:
  worker_threads: 32
  thread_stack_size: 2097152
limits:
  max_units: 65536
  max_queue_depth: 5000
  timeout_secs: 900
```

**Rationale:**
- Maximum parallelism
- Large KV cache pool
- Deep queue for burst handling

### Memory-Constrained (8GB GPU or CPU-only)

```yaml
pool:
  worker_threads: 2
  thread_stack_size: 2097152
limits:
  max_units: 1024
  max_queue_depth: 25
  timeout_secs: 180
```

**Rationale:**
- Minimal thread overhead
- Conservative memory allocation
- Small queue to avoid memory pressure

## CLI Overrides

All configuration values can be overridden via CLI flags:

```bash
# Override worker threads
--worker-threads 8

# Override thread stack size (in bytes)
--thread-stack-size 4194304  # 4MB

# Override max resource units
--scheduler-max-units 8192

# Override max queue depth
--scheduler-max-queue 500

# Override timeout (in seconds)
--scheduler-timeout 600
```

**Example:**
```bash
cargo run --release --features parking-lot-scheduler -- \
  --scheduler-config ~/.mistralrs-server/scheduler.yaml \
  --worker-threads 16 \
  --scheduler-max-units 16384 \
  plain -m microsoft/Phi-3-mini-4k-instruct
```

## Examples

### Basic Configuration

**~/.mistralrs-server/scheduler.yaml:**
```yaml
pool:
  worker_threads: 4
limits:
  max_units: 4096
  max_queue_depth: 100
  timeout_secs: 300
```

### Custom Location

```bash
# Via CLI flag
./mistralrs-server --scheduler-config /path/to/config.yaml plain ...

# Via environment variable
export MISTRALRS_SCHEDULER_CONFIG=/path/to/config.yaml
./mistralrs-server plain ...
```

### Development vs Production

**development.yaml:**
```yaml
pool:
  worker_threads: 2
limits:
  max_units: 1024
  max_queue_depth: 10
  timeout_secs: 60
```

**production.yaml:**
```yaml
pool:
  worker_threads: 16
limits:
  max_units: 32768
  max_queue_depth: 5000
  timeout_secs: 900
```

```bash
# Development
./mistralrs-server --scheduler-config development.yaml plain ...

# Production
./mistralrs-server --scheduler-config production.yaml plain ...
```

## Troubleshooting

### High Memory Usage

**Symptom:** Excessive memory consumption

**Solutions:**
- Reduce `max_units` (e.g., halve the value)
- Reduce `max_queue_depth`
- Ensure you're not overallocating for available VRAM

### Requests Timing Out

**Symptom:** 503 errors or timeout messages

**Solutions:**
- Increase `timeout_secs`
- Reduce `max_queue_depth` if queue is consistently full
- Add more worker threads if CPU-bound
- Increase `max_units` if requests are queuing for resources

### Low Throughput

**Symptom:** Slower than expected request processing

**Solutions:**
- Increase `worker_threads` (but not beyond physical cores)
- Increase `max_units` if resource-constrained
- Increase `max_queue_depth` if requests are being rejected
- Check GPU utilization (may be model/hardware bottleneck)

### Stack Overflow Errors

**Symptom:** Thread panics with stack overflow

**Solutions:**
- Increase `thread_stack_size` (e.g., double it to 4MB)
- Check for recursive operations in custom code

### Configuration Not Loading

**Symptom:** Default values used instead of config file

**Solutions:**
- Verify file path: `~/.mistralrs-server/scheduler.yaml` exists
- Check YAML syntax with a validator
- Set `MISTRALRS_DEBUG=1` for more logging
- Try explicit `--scheduler-config` flag
- Verify `parking-lot-scheduler` feature is enabled in build

## Monitoring

When the parking-lot scheduler is active, you'll see log messages like:

```
🔧 Scheduler Configuration:
   Worker threads: 8
   Thread stack size: 2097152 bytes
   Max units: 8192
   Max queue depth: 500
   Timeout: 300s
🚀 Initializing prometheus_parking_lot WorkerPool for inference
🔧 WorkerPool settings:
   Worker threads: 8
   Thread stack size: 2097152 bytes
   Max units: 8192
   Max queue depth: 500
   Timeout: 300s
✅ WorkerPool initialized successfully
```

This confirms your configuration is active.

## See Also

- [Example configuration file](../examples/scheduler-config.yaml) with detailed comments
- [Minimal configuration](../examples/scheduler-config-minimal.yaml) for quick start
- [PagedAttention documentation](PAGED_ATTENTION.md) for KV cache configuration
- [Performance tuning guide](../README.md#performance)
