# FlashInfer MoE Microbench

Model shape: Gemma4 26B-A4B BF16 MoE, `E=128`, `top_k=8`, `hidden=2816`, `intermediate=704`, activation `GeGLU`.

## TRT-LLM BF16 Routed MoE

This path is not directly viable for Gemma4:

- `trtllm_bf16_routed_moe` requires `intermediate_size % 128 == 0`.
- Gemma4 26B-A4B uses `intermediate_size=704`.
- Padding to 768 passes shape checks, but the installed FlashInfer BF16 runner does not provide a GeGLU tactic for this configuration.

## CUTLASS Fused MoE

This path supports Gemma4's exact `704` intermediate and GeGLU activation.

Command:

```bash
env PYTHONPYCACHEPREFIX=/tmp/mistralrs_pycache \
  FLASHINFER_WORKSPACE_BASE=/tmp/mistralrs_flashinfer \
  FLASHINFER_DISABLE_VERSION_CHECK=1 \
  ../vllm/.venv-bench/bin/python \
  gemma4_cuda_sweep_20260525/flashinfer_moe_microbench/bench_flashinfer_cutlass_moe.py \
  --tokens 1 16 128 512 \
  --warmup 2 \
  --iters 10 \
  --tune-max-num-tokens 512
```

| Tokens | Time / MoE Layer | Tokens/s |
|---:|---:|---:|
| 1 | 0.430 ms | 2,324 |
| 16 | 4.331 ms | 3,694 |
| 128 | 6.563 ms | 19,504 |
| 512 | 7.275 ms | 70,381 |

4096-token command:

```bash
env PYTHONPYCACHEPREFIX=/tmp/mistralrs_pycache \
  FLASHINFER_WORKSPACE_BASE=/tmp/mistralrs_flashinfer \
  FLASHINFER_DISABLE_VERSION_CHECK=1 \
  ../vllm/.venv-bench/bin/python \
  gemma4_cuda_sweep_20260525/flashinfer_moe_microbench/bench_flashinfer_cutlass_moe.py \
  --tokens 4096 \
  --warmup 2 \
  --iters 3 \
  --tune-max-num-tokens 4096
```

| Tokens | Time / MoE Layer | Tokens/s |
|---:|---:|---:|
| 4096 | 15.956 ms | 256,714 |

## Selected Kernels

Nsight Systems on the 512-token CUTLASS run showed the main kernels:

- `fused_moe::Fused_Moe_Kernel_sm80<..., Activation_Type=4>`: about 4.67 ms per call.
- `cutlass::gemm::kernel::MoeFCGemm<..., GemmShape<32,128,64>, ...>`: about 2.20 ms per call.

## Takeaway

FlashInfer CUTLASS is a compatible reference and likely faster than the current unquantized grouped WMMA MoE path, but it is not vLLM-level for prompt. Vendoring it cleanly would also pull in a large generated CUTLASS/TRT-LLM stack unless we extract a very narrow BF16 GeGLU subset.
