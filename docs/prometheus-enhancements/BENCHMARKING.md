# Performance Benchmarking Guide

This document explains how to benchmark the parking_lot migration improvements in mistral.rs.

## Lock Performance Benchmarks

The lock benchmarks measure mutex and rwlock performance under various contention levels.

### Running Lock Benchmarks

```bash
# Run all lock benchmarks with output
cargo test -p mistralrs-bench lock_benchmarks::tests::bench_comparison -- --nocapture

# Run specific contention levels
cargo test -p mistralrs-bench lock_benchmarks::tests::test_mutex_low_contention -- --nocapture
cargo test -p mistralrs-bench lock_benchmarks::tests::test_mutex_medium_contention -- --nocapture
cargo test -p mistralrs-bench lock_benchmarks::tests::test_mutex_high_contention -- --nocapture
```

### Expected Results

Based on parking_lot benchmarks, you should see:

**Low Contention (2 threads)**:
- ~10% improvement over std::sync::Mutex

**Medium Contention (8 threads)**:
- ~32% improvement

**High Contention (32 threads)**:
- ~36% improvement

### Interpreting Results

The benchmarks report total time to complete N iterations across M threads.

**Lower time = Better performance**

Example output:
```
=== Lock Performance Benchmarks ===

Mutex Contention:
  2 threads: 1.2ms
  8 threads: 4.5ms
  16 threads: 8.3ms
  32 threads: 15.1ms
```

## Scheduler Performance Benchmarks

### Default Scheduler (std::sync)

Run without feature flag:
```bash
cargo run --release -p mistralrs-bench -- \
  --model-id "HuggingFaceH4/zephyr-7b-beta" \
  --prompt-batchsize 2,4,8 \
  --n-prompt 512 \
  --n-gen 128 \
  --repetitions 3
```

### Worker Pool Scheduler (parking_lot)

Run with feature flag:
```bash
cargo run --release -p mistralrs-bench --features parking-lot-scheduler -- \
  --model-id "HuggingFaceH4/zephyr-7b-beta" \
  --prompt-batchsize 2,4,8 \
  --n-prompt 512 \
  --n-gen 128 \
  --repetitions 3
```

### Metrics to Compare

- **Throughput**: tokens/sec at various concurrency levels
- **Latency**: time to first token (TTFT), time per output token (TPOT)
- **Queue time**: time requests spend waiting
- **Resource utilization**: GPU memory usage, cache hit rate

## Integration Tests

Run the parking_lot module tests:

```bash
# All parking_lot tests
cargo test -p mistralrs-core parking_lot::tests

# Specific test
cargo test -p mistralrs-core parking_lot::tests::test_worker_pool_config
```

All tests should pass:
```
test result: ok. 12 passed; 0 failed; 0 ignored
```

## Stress Testing

For production validation:

```bash
# High concurrency test (requires actual model)
cargo run --release -p mistralrs-bench -- \
  --model-id "microsoft/Phi-3-mini-4k-instruct" \
  --prompt-batchsize 32,64 \
  --n-prompt 256 \
  --n-gen 64 \
  --repetitions 10
```

## Continuous Benchmarking

Add to CI pipeline:

```yaml
- name: Run lock benchmarks
  run: |
    cargo test -p mistralrs-bench lock_benchmarks::tests::bench_comparison -- --nocapture
```

## Baseline Metrics

Record baseline before migration:
```bash
# Save baseline
cargo run --release -p mistralrs-bench -- \
  --model-id "MODEL" \
  --prompt-batchsize 4,8 \
  --n-prompt 512 \
  --n-gen 128 \
  --repetitions 5 > baseline_metrics.txt
```

Then compare after migration:
```bash
# Compare with parking_lot
cargo run --release -p mistralrs-bench --features parking-lot-scheduler -- \
  --model-id "MODEL" \
  --prompt-batchsize 4,8 \
  --n-prompt 512 \
  --n-gen 128 \
  --repetitions 5 > parking_lot_metrics.txt

diff baseline_metrics.txt parking_lot_metrics.txt
```

## Performance Goals

Target improvements from parking_lot migration:

- **Lock contention**: -40% (high concurrency scenarios)
- **Memory footprint**: -40% per lock (24 vs 40 bytes)
- **Throughput**: +10-15% (concurrent requests)
- **Latency p99**: -5-10% (reduced tail latencies)
- **No poisoning overhead**: Eliminated panic handling cost
