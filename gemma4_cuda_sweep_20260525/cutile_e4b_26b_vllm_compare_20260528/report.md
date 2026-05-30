# Gemma 4 BF16 mistral.rs (cutile MoE) vs vLLM

Branch `cuda_graphs_v1` @ `478e884e1` (cutile MoE backend, auto-default on bf16/CUDA). CUDA graphs + FlashInfer decode default-on. vLLM 0.21.0, same methodology as `26b_bf16_vllm_resweep_20260526` (language_model_only, prefix caching off, max_model_len 20000; E4B gpu-mem-util 0.85, 26B 0.60). vLLM prompt = input_len/latency(out=1); decode = 256/(lat(257)-lat(1)).

# E4B BF16 (dense - cutile does not apply; control)
## E4B Prompt T/s
| tokens | mistral.rs | vLLM | vs vLLM |
| ---: | ---: | ---: | ---: |
| 128 | 2419.0 | 2298.1 | +5% |
| 512 | 5665.5 | 6290.0 | -10% |
| 2048 | 7288.3 | 7199.0 | +1% |
| 4096 | 7207.1 | 7210.7 | -0% |
| 8192 | 7043.9 | 6627.2 | +6% |
| 16384 | 6605.6 | 5660.6 | +17% |

## E4B Decode T/s
| tokens | mistral.rs | vLLM | vs vLLM |
| ---: | ---: | ---: | ---: |
| 128 | 26.2 | 19.3 | +36% |
| 512 | 25.9 | 19.2 | +35% |
| 2048 | 25.6 | 19.1 | +34% |
| 4096 | 25.5 | 18.9 | +35% |
| 8192 | 24.9 | 18.5 | +35% |
| 16384 | 24.1 | 18.0 | +34% |

# 26B-A4B BF16 (cutile MoE path)
## 26B-A4B Prompt T/s
| tokens | mistral.rs (cutile) | before-cutile | vLLM | vs vLLM |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 946.9 | 403.4 | 2549.3 | -63% |
| 512 | 2370.6 | 700.2 | 6237.5 | -62% |
| 2048 | 3821.5 | 674.0 | 6373.8 | -40% |
| 4096 | 4034.2 | 605.0 | 6267.2 | -36% |
| 8192 | 3901.0 | 586.7 | 5786.8 | -33% |
| 16384 | 3582.0 | 568.8 | 4766.9 | -25% |

## 26B-A4B Decode T/s
| tokens | mistral.rs (cutile) | before-cutile | vLLM | vs vLLM |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 31.9 | 15.6 | 24.3 | +31% |
| 512 | 31.4 | 15.4 | 24.1 | +30% |
| 2048 | 30.7 | 15.3 | 23.7 | +29% |
| 4096 | 30.5 | 15.2 | 23.6 | +29% |
| 8192 | 30.1 | 15.1 | 23.3 | +29% |
| 16384 | 29.2 | 14.9 | 22.8 | +28% |

## Notes
- `before-cutile` column = `26b_bf16_vllm_resweep_20260526` (commit 1bd863a62, the pre-cutile commit, graphs+FlashInfer on). Same machine/methodology, so it isolates the cutile MoE delta.
- 26B-A4B was re-benched after the cuTile warmup fix (warmup runs on the engine thread + MoE shapes registered in the weight loaders), eliminating per-shape JIT during measured iterations. This corrected the previously JIT-depressed p128/p2048 means (659->947, 2811->3822) and collapsed their stddev (+/-419->+/-32, +/-1456->+/-50). Verified zero JIT via CUTILE_JIT_TIMING.
- cutile is bf16-only and auto-selected on the unquantized CUDA MoE path; E4B is dense so it is a control (warmup-3, unaffected by the MoE warmup fix; matches prior BF16 numbers).
- Remaining: fix C (pad M) to make prefill robust to arbitrary un-warmed lengths; warmup currently covers the bucket set {1..16384}.

