# Gemma 4 26B-A4B BF16 mistral.rs vs vLLM

Models:
- mistral.rs: `../hf_models/gemma4_26b_a4b`
- vLLM: `../hf_models/gemma4_26b_a4b`

Configuration:
- mistral.rs: `target/release/mistralrs bench`, BF16/no quant, CUDA, FlashAttention, PagedAttention on, `--pa-context-len 20000 --max-seq-len 20000`, 3 iterations, 1 warmup, CUDA graphs and FlashInfer decode explicitly enabled.
- vLLM: `vllm 0.21.0`, BF16/no quant, `language_model_only=True`, prefix caching disabled, `max_model_len=20000`, `gpu_memory_utilization=0.60`, 3 measured repetitions.
- vLLM runs one warmup for each prompt/output shape before measuring, to avoid including first-use Triton JIT in the samples.
- vLLM forced `disable_chunked_mm_input` for multimodal-bidirectional attention and reported the default MoE config for `E=128,N=704` on NVIDIA GB10.
- vLLM prompt T/s is approximated from `input_len / latency(output_len=1)`, so it includes one generated token.
- vLLM decode T/s is computed as `256 / (latency(output_len=257) - latency(output_len=1))`.

## Prompt T/s

| Prompt tokens | mistral.rs BF16 | vLLM BF16 approx |
| ---: | ---: | ---: |
| 128 | 403.4 | 2652.9 |
| 512 | 700.2 | 6041.6 |
| 2048 | 674.0 | 6499.7 |
| 4096 | 605.0 | 6327.2 |
| 8192 | 586.7 | 5754.5 |
| 16384 | 568.8 | 4757.7 |

## Decode T/s

| Depth | mistral.rs BF16 | vLLM BF16 delta |
| ---: | ---: | ---: |
| 128 | 15.6 | 24.4 |
| 512 | 15.4 | 24.1 |
| 2048 | 15.3 | 23.8 |
| 4096 | 15.2 | 23.7 |
| 8192 | 15.1 | 23.4 |
| 16384 | 14.9 | 22.9 |

## Artifacts

- `26b_a4b_bf16_prompt_tps.png`
- `26b_a4b_bf16_decode_tps.png`
- `results.json`
- `vllm_26b_a4b_bf16_latency_delta.json`
- `run_vllm_latency_delta.py`

Raw outputs are in this directory as `mistralrs_*.txt` and `vllm_*.log`.

## Commands

```bash
MISTRALRS_CUDA_GRAPHS=1 MISTRALRS_FLASHINFER_DECODE=1 target/release/mistralrs bench -m ../hf_models/gemma4_26b_a4b --prompt-len 128,512,2048,4096,8192,16384 --gen-len 0 --depth 1 --iterations 3 --warmup 1 --paged-attn on --pa-context-len 20000 --max-seq-len 20000
MISTRALRS_CUDA_GRAPHS=1 MISTRALRS_FLASHINFER_DECODE=1 target/release/mistralrs bench -m ../hf_models/gemma4_26b_a4b --prompt-len 128 --gen-len 256 --depth 128,512,2048,4096,8192,16384 --iterations 3 --warmup 1 --paged-attn on --pa-context-len 20000 --max-seq-len 20000
cd ../vllm && .venv-bench/bin/python ../mistral.rs/gemma4_cuda_sweep_20260525/26b_bf16_vllm_resweep_20260526/run_vllm_latency_delta.py --model ../hf_models/gemma4_26b_a4b --out ../mistral.rs/gemma4_cuda_sweep_20260525/26b_bf16_vllm_resweep_20260526/vllm_26b_a4b_bf16_latency_delta.json --gpu-memory-utilization 0.60 --max-model-len 20000 --repeats 3
```
