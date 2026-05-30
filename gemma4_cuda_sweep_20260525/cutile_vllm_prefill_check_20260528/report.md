# Gemma 4 26B-A4B BF16 prefill check

Date: 2026-05-28

Model: `../hf_models/gemma4_26b_a4b`

Configuration:
- mistral.rs: current worktree, fresh `target/release/mistralrs`, `MISTRALRS_MOE_BACKEND=cutile`, BF16, PagedAttention on, context length 20000.
- vLLM: `vllm 0.21.0`, BF16, `language_model_only=True`, prefix caching disabled, max model length 20000.
- Both use sequential token IDs starting at 1000.
- Both use 3 measured repetitions. mistral.rs used same-shape prewarm via `MISTRALRS_NSYS_PREWARM=1`; vLLM warms each prompt/output shape in the script.

## Prefill T/s

| Prompt tokens | mistral.rs Cutile | vLLM |
| ---: | ---: | ---: |
| 128 | 1005.7 | 1014.4 |
| 512 | 2493.6 | 2566.4 |
| 2048 | 3903.5 | 4585.6 |
| 4096 | 4047.6 | 5132.7 |
| 8192 | 3873.9 | 5194.5 |
| 16384 | 3583.0 | 4379.2 |

## TTFT / output-1 latency

| Prompt tokens | mistral.rs TTFT ms | vLLM o1 mean ms |
| ---: | ---: | ---: |
| 128 | 127.33 | 126.18 |
| 512 | 205.33 | 199.50 |
| 2048 | 524.67 | 446.61 |
| 4096 | 1012.00 | 798.02 |
| 8192 | 2114.67 | 1577.05 |
| 16384 | 4572.67 | 3741.30 |

## Commands

```bash
CUDA_COMPUTE_CAP=121 cargo build --release --package mistralrs-cli --features cuda,flash-attn,cutile
```

```bash
MISTRALRS_MOE_BACKEND=cutile MISTRALRS_CUDA_GRAPHS=1 MISTRALRS_FLASHINFER_DECODE=1 MISTRALRS_NSYS_PREWARM=1 target/release/mistralrs bench -m ../hf_models/gemma4_26b_a4b --prompt-len 128,512,2048,4096,8192,16384 --gen-len 0 --depth 1 --iterations 3 --warmup 1 --paged-attn on --pa-context-len 20000 --max-seq-len 20000
```

```bash
cd ../vllm
.venv-bench/bin/python ../mistral.rs/gemma4_cuda_sweep_20260525/26b_bf16_vllm_resweep_20260526/run_vllm_latency_delta.py --model ../hf_models/gemma4_26b_a4b --out ../mistral.rs/gemma4_cuda_sweep_20260525/cutile_vllm_prefill_check_20260528/vllm_26b_a4b_bf16_latency_delta.json --gpu-memory-utilization 0.60 --max-model-len 20000 --repeats 3 --lengths 128,512,2048,4096,8192,16384 --prompt-pattern sequential
```

Notes:
- The first mistral.rs sweep without same-shape prewarm showed large variance at p128 and p2048, so it was discarded.
- The old vLLM p512 number around 6040 T/s was from the script default repeat-token prompt. The matched sequential-token vLLM p512 result here is 2566.4 T/s.
