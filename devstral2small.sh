#!/usr/bin/env bash
set -euo pipefail

FEATURES="${FEATURES:-cuda flash-attn}"
MODEL="${MODEL:-mistralai/Devstral-Small-2-24B-Instruct-2512}"
PORT="${PORT:-1234}"
CONTEXT_TOKENS="${CONTEXT_TOKENS:-262144}"
# If TOPOLOGY is unset, default to a device-only topology. If TOPOLOGY is set to an empty string,
# disable topology entirely (useful for debugging).
TOPOLOGY="${TOPOLOGY-topologies/devstral_small2_24b_default.yml}"
PA_CTXT_LEN="${PA_CTXT_LEN:-${CONTEXT_TOKENS}}"
# Leave unset by default; setting both ctxt-len and gpu-mem-usage makes the server pick one,
# and gpu-mem-usage can easily reserve too much KV cache to fit weights fully on GPU.
PA_GPU_MEM_USAGE="${PA_GPU_MEM_USAGE-}"
PA_CACHE_TYPE="${PA_CACHE_TYPE:-auto}"
MAX_SEQS="${MAX_SEQS:-1}"
# Default sampling params (applied only when client omits them).
TEMPERATURE="${TEMPERATURE:-0.15}"
TOP_P="${TOP_P:-0.9}"
MIN_P="${MIN_P:-0.01}"
TOP_K="${TOP_K-}"
# Optional: chunk prompt prefill into smaller pieces to reduce peak VRAM usage.
# Set to e.g. 2048 when Codex sends ~10k token prompts and prefill OOMs.
PREFILL_CHUNK_SIZE="${PREFILL_CHUNK_SIZE-2048}"
# Optional: prepend default Responses API `instructions` if the client doesn't set them.
# Useful to enforce response language/format with Codex and other OpenAI-compatible clients.
DEFAULT_INSTRUCTIONS="${DEFAULT_INSTRUCTIONS-}"
CUDA_MODULE="${CUDA_MODULE:-cuda/12.5.1-b6iqzzi}"
GCC_MODULE="${GCC_MODULE:-gcc/12.2.0-w7lhsaj}"
CUDA_NVCC_FLAGS="${CUDA_NVCC_FLAGS:-}"
RUSTFLAGS="${RUSTFLAGS:--C opt-level=2}"
HF_HOME="${HF_HOME:-$PWD/.hf}"
HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
HF_TOKEN="${HF_TOKEN:-${HUGGINGFACE_TOKEN:-}}"

# Script-only flags (not passed to mistralrs-server).
# These exist to avoid huge `env ... bash devstral2small.sh ...` one-liners.
PASSTHROUGH_ARGS=()
SCRIPT_PREFILL_CHUNK_SIZE_SET=0
SCRIPT_PREFILL_CHUNK_SIZE=""
SCRIPT_TOPOLOGY_SET=0
SCRIPT_TOPOLOGY=""
NO_VISION=0

usage() {
  cat <<'EOF'
Usage: devstral2small.sh [script flags] [--] [mistralrs-server flags]

Script flags:
  --prefill-chunk-size N   Sets MISTRALRS_PREFILL_CHUNK_SIZE (default: 2048 if unset)
  --topology PATH          Uses PATH as the topology file (default: topologies/devstral_small2_24b_default.yml)
  --no-topology            Disables topology (equivalent to TOPOLOGY='')
  --no-vision              Disable loading the vision component (text-only)
  -h, --help               Show this help

Examples:
  ./devstral2small.sh --pa-cache-type f8e4m3 --temperature 0.6
  ./devstral2small.sh --prefill-chunk-size 4096 --pa-cache-type f8e4m3
  ./devstral2small.sh --no-topology --pa-cache-type f8e4m3
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --prefill-chunk-size)
      shift
      SCRIPT_PREFILL_CHUNK_SIZE_SET=1
      SCRIPT_PREFILL_CHUNK_SIZE="${1-}"
      shift || true
      ;;
    --prefill-chunk-size=*)
      SCRIPT_PREFILL_CHUNK_SIZE_SET=1
      SCRIPT_PREFILL_CHUNK_SIZE="${1#*=}"
      shift
      ;;
    --topology)
      shift
      SCRIPT_TOPOLOGY_SET=1
      SCRIPT_TOPOLOGY="${1-}"
      shift || true
      ;;
    --topology=*)
      SCRIPT_TOPOLOGY_SET=1
      SCRIPT_TOPOLOGY="${1#*=}"
      shift
      ;;
    --no-topology)
      SCRIPT_TOPOLOGY_SET=1
      SCRIPT_TOPOLOGY=""
      shift
      ;;
    --no-vision)
      NO_VISION=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      while [[ $# -gt 0 ]]; do
        PASSTHROUGH_ARGS+=("$1")
        shift
      done
      break
      ;;
    *)
      PASSTHROUGH_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ "${SCRIPT_PREFILL_CHUNK_SIZE_SET}" == "1" ]]; then
  PREFILL_CHUNK_SIZE="${SCRIPT_PREFILL_CHUNK_SIZE}"
fi
if [[ "${SCRIPT_TOPOLOGY_SET}" == "1" ]]; then
  TOPOLOGY="${SCRIPT_TOPOLOGY}"
fi

# Prefer the system/module toolchain for nvcc host compilation (avoid conda/pixi GCC 15+).
unset NVCC_PREPEND_FLAGS

if [[ "${SKIP_MODULES:-0}" != "1" ]]; then
  if type module >/dev/null 2>&1; then
    module load "${GCC_MODULE}"
    module load "${CUDA_MODULE}"
  fi
fi

export NVCC_CCBIN="${NVCC_CCBIN:-$(command -v gcc)}"
export HF_HOME HF_HUB_CACHE
if [[ -n "${HF_TOKEN}" ]]; then
  export HF_TOKEN
fi
if [[ -n "${PREFILL_CHUNK_SIZE}" ]]; then
  export MISTRALRS_PREFILL_CHUNK_SIZE="${PREFILL_CHUNK_SIZE}"
fi

# If the caller did not set DEFAULT_INSTRUCTIONS at all, provide a good "Codex server" default.
# To disable, set DEFAULT_INSTRUCTIONS='' explicitly.
if [[ -z "${DEFAULT_INSTRUCTIONS+x}" ]]; then
  TODAY="$(date -I 2>/dev/null || true)"
  if [[ -z "${TODAY}" ]]; then
    TODAY="$(date '+%Y-%m-%d')"
  fi

  DEFAULT_INSTRUCTIONS="$(cat <<'EOF'
You are Devstral-Small-2-24B-Instruct-2512, served behind an OpenAI-compatible Responses API (often used by Codex CLI).
Today is {today}.

You are a coding-focused assistant. Prioritize correctness, safety, and following the user's most recent request.

Tool use:
- If tools are available, use them when they materially improve correctness (e.g. reading files, running commands, checking outputs).
- Never invent tool results. If a tool call fails or is unavailable, say so briefly and continue with best-effort reasoning.
- Do not claim you can browse the web unless a dedicated web/search tool is explicitly available in the provided tool list.
- When you produce tool calls, follow the provided tool schema exactly and emit valid JSON arguments.

Behavior:
- Do not repeat or explain any “harness” / environment / tool-schema text that may appear in the conversation history.
- Ask a clarifying question if the request is ambiguous or missing required details.
- Use English by default unless the user requests another language.
EOF
)"
  DEFAULT_INSTRUCTIONS="${DEFAULT_INSTRUCTIONS//\{today\}/${TODAY}}"
fi

if [[ -n "${DEFAULT_INSTRUCTIONS}" ]]; then
  export MISTRALRS_DEFAULT_INSTRUCTIONS="${DEFAULT_INSTRUCTIONS}"
fi
mkdir -p "${HF_HUB_CACHE}"

TOKEN_SOURCE_ARGS=()
if [[ -n "${HF_TOKEN}" ]]; then
  TOKEN_SOURCE_ARGS=(--token-source "env:HF_TOKEN")
fi

has_arg() {
  local needle="$1"
  shift || true
  for arg in "$@"; do
    if [[ "${arg}" == "${needle}" || "${arg}" == "${needle}="* ]]; then
      return 0
    fi
  done
  return 1
}

echo "Building mistralrs-server with features: ${FEATURES}"
env NVCC_CCBIN="${NVCC_CCBIN}" CUDA_NVCC_FLAGS="${CUDA_NVCC_FLAGS}" RUSTFLAGS="${RUSTFLAGS}" cargo build --release --package mistralrs-server --features "${FEATURES}"

SERVER_ARGS=(--port "${PORT}")
SERVER_ARGS+=(--jinja-explicit "chat_templates/devstral_fixed.jinja")
SERVER_ARGS+=("${TOKEN_SOURCE_ARGS[@]}")
if [[ -n "${PA_CTXT_LEN}" ]] && ! has_arg "--pa-ctxt-len" "${PASSTHROUGH_ARGS[@]}"; then
  SERVER_ARGS+=(--pa-ctxt-len "${PA_CTXT_LEN}")
fi
if [[ -n "${PA_GPU_MEM_USAGE}" ]] && ! has_arg "--pa-gpu-mem-usage" "${PASSTHROUGH_ARGS[@]}"; then
  SERVER_ARGS+=(--pa-gpu-mem-usage "${PA_GPU_MEM_USAGE}")
fi
if [[ -n "${PA_CACHE_TYPE}" ]] && ! has_arg "--pa-cache-type" "${PASSTHROUGH_ARGS[@]}"; then
  SERVER_ARGS+=(--pa-cache-type "${PA_CACHE_TYPE}")
fi
if [[ -n "${MAX_SEQS}" ]] && ! has_arg "--max-seqs" "${PASSTHROUGH_ARGS[@]}"; then
  SERVER_ARGS+=(--max-seqs "${MAX_SEQS}")
fi
if [[ -n "${TEMPERATURE}" ]] && ! has_arg "--temperature" "${PASSTHROUGH_ARGS[@]}"; then
  SERVER_ARGS+=(--temperature "${TEMPERATURE}")
fi
if [[ -n "${TOP_P}" ]] && ! has_arg "--top-p" "${PASSTHROUGH_ARGS[@]}" && ! has_arg "--top_p" "${PASSTHROUGH_ARGS[@]}"; then
  SERVER_ARGS+=(--top-p "${TOP_P}")
fi
if [[ -n "${MIN_P}" ]] && ! has_arg "--min-p" "${PASSTHROUGH_ARGS[@]}" && ! has_arg "--min_p" "${PASSTHROUGH_ARGS[@]}"; then
  SERVER_ARGS+=(--min-p "${MIN_P}")
fi
if [[ -n "${TOP_K}" ]] && ! has_arg "--top-k" "${PASSTHROUGH_ARGS[@]}" && ! has_arg "--top_k" "${PASSTHROUGH_ARGS[@]}"; then
  SERVER_ARGS+=(--top-k "${TOP_K}")
fi


RUN_ARGS=(run -m "${MODEL}")
if [[ -n "${TOPOLOGY}" ]]; then
  echo "Using topology: ${TOPOLOGY}"
  RUN_ARGS+=(--topology "${TOPOLOGY}")
fi
if [[ "${NO_VISION}" == "1" ]]; then
  RUN_ARGS+=(--no-vision)
fi

echo "Starting Devstral Small 2 (context ${CONTEXT_TOKENS} tokens) on port ${PORT}"
exec ./target/release/mistralrs-server \
  "${SERVER_ARGS[@]}" \
  "${PASSTHROUGH_ARGS[@]}" \
  "${RUN_ARGS[@]}"
