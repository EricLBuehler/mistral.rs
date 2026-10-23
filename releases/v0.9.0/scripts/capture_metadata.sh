#!/usr/bin/env bash
# Capture host + engine metadata alongside bench results.
set -u
OUT="${1:-releases/v0.9.0/raw/metadata.txt}"
{
  echo "date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "host: $(hostname)"
  echo "kernel: $(uname -r)"
  echo "mistralrs_commit: $(git -C ~/mistral.rs rev-parse HEAD)"
  echo "mistralrs_version: $(~/mistral.rs/target/release/mistralrs --version)"
  echo "llamacpp_commit: $(git -C ~/llama.cpp rev-parse HEAD)"
  echo "llamacpp_build: build-cpu (GGML_CUDA=OFF GGML_NATIVE=ON Release)"
  echo "rustc: $(rustc --version)"
  echo ""
  echo "== cpu topology =="
  lscpu | grep -E "^CPU\(s\)|Model name|Core|Socket|NUMA node0"
  for c in $(seq 0 19); do
    printf "cpu%d capacity=%s max_khz=%s\n" "$c" \
      "$(cat /sys/devices/system/cpu/cpu$c/cpu_capacity)" \
      "$(cat /sys/devices/system/cpu/cpu$c/cpufreq/cpuinfo_max_freq)"
  done
  echo ""
  echo "== memory =="
  free -h | head -2
  echo ""
  echo "== model files =="
  ls -la ~/hf_models/bench_v090/*/ 2>/dev/null
} > "$OUT"
echo "wrote $OUT"
