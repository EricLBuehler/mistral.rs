#!/usr/bin/env bash
set -euo pipefail
mode=${1:-exact}
ple_memory=${2:-disk}
case "$mode" in
  exact) head_quant=0 ;;
  recommended-head) head_quant=1 ;;
  *) echo 'Expected exact or recommended-head' >&2; exit 2 ;;
esac
case "$ple_memory" in
  disk) container_name="qwen38-fp4-vllm-$mode"; allocator_fraction=0.8; gpu_utilization=0.8 ;;
  device|ram) container_name="qwen38-fp4-vllm-$mode-$ple_memory"; allocator_fraction=0.9; gpu_utilization=0.85 ;;
  *) echo 'Expected disk, device, or ram PLE placement' >&2; exit 2 ;;
esac
ple_config="{\"ple_table_memory\":\"$ple_memory\"}"
exec docker run --rm --name qwen38-fp4-c1-c8 \
  --gpus all --ipc host --network host --shm-size=16g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --security-opt seccomp=unconfined --security-opt apparmor=unconfined \
  -v /home/ericbuehler/.cache/huggingface:/root/.cache/huggingface:ro \
  -v /tmp/qwen38-fp4-vllm/cache/vllm:/root/.cache/vllm \
  -v /tmp/qwen38-fp4-vllm/cache/flashinfer:/root/.cache/flashinfer \
  -v /tmp/qwen38-fp4-vllm/cache/triton:/root/.triton \
  -v /tmp/qwen38-fp4-vllm/cache/b12x:/root/.cache/b12x \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e "PYTORCH_ALLOC_CONF=per_process_memory_fraction:$allocator_fraction" \
  -e B12X_DISK_BACKEND=io_uring -e CUTE_DSL_ARCH=sm_121a -e SAFETENSORS_FAST_GPU=1 \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn -e VLLM_SSM_CONV_STATE_LAYOUT=DS \
  -e VLLM_USE_AOT_COMPILE=1 -e VLLM_USE_MEGA_AOT_ARTIFACT=1 \
  -e VLLM_USE_V2_MODEL_RUNNER=1 -e B12X_POLICY_MODE=auto \
  -e "VLLM_MXFP8_LM_HEAD=$head_quant" \
  -e VLLM_B12X_MOE_FP4_LAYER_MAX_INPUT_SCALE=w13 \
  -e VLLM_B12X_MOE_FP4_FORCE_A16=1 -e B12X_W4A16_A4_PREFILL=1 \
  --entrypoint vllm eugr/spark-vllm-b12x@sha256:57c24fc397c66d4d28a7d0d5f1d792934fd01a70f4574cff6315fb7bc329d5cb \
  serve /root/.cache/huggingface/hub/models--nvidia--Qwen3.8-Flash-Next-FP4-Local/snapshots/b9ce18a6887e18c089fdea8509a33e63429a3300 \
  --served-model-name nvidia/Qwen3.8-Flash-Next-FP4-Local \
  --host 127.0.0.1 --port 8238 \
  --tensor-parallel-size 1 --pipeline-parallel-size 1 \
  --mamba-cache-mode align --no-enable-prefix-caching --enable-chunked-prefill \
  --dtype bfloat16 --kv-cache-dtype bfloat16 --quantization modelopt_mixed \
  --block-size 16 --load-format b12x --max-model-len 16384 \
  --max-num-seqs 8 --max-num-batched-tokens 4096 \
  --max-cudagraph-capture-size 8 --kv-cache-memory-bytes 2147483648 \
  --gdn-decode-kernel b12x --linear-backend b12x --moe-backend b12x \
  --no-enable-flashinfer-autotune --mm-encoder-tp-mode data \
  --reasoning-parser qwen3 --tool-call-parser qwen3_xml --enable-auto-tool-choice \
  --compilation-config '{"pass_config":{"fuse_act_quant":true}}' \
  --additional-config "$ple_config" \
  --gpu-memory-utilization "$gpu_utilization"
