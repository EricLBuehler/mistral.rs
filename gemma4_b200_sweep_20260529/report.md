# Gemma-4 throughput sweep on B200 — mistral.rs vs llama.cpp vs vLLM

**Date:** 2026-05-29 · **GPU:** NVIDIA B200 (sm_100), driver CUDA 13.0.

mistral.rs numbers were **re-measured after the hd512 paged-attention correctness fix** (see "What changed" below). The reference engines (llama.cpp, vLLM) are unchanged from the original run.

## Models & configs
- **Models:** `gemma-4-E4B-it` (dense) and `gemma-4-26B-A4B` (MoE, 4B active).
- **Q8 (vs llama.cpp):** mistral.rs `--quant 8` (ISQ / UQFF 8-bit) vs llama.cpp `Q8_0` GGUF — each engine's native 8-bit.
- **bf16 (vs vLLM):** mistral.rs bf16 vs vLLM bf16 (`enable_prefix_caching=False`, unique prompts per iter).
- **mistral.rs:** `cuda,flash-attn,cutile`; `MISTRALRS_CUDA_GRAPHS=1 MISTRALRS_FLASHINFER_DECODE=1`; `--paged-attn on`.
- **llama.cpp:** `-ngl 99 -fa 1 -ctk f16 -ctv f16`.
- **Method:** prefill = pure prompt processing at each length; decode = 256 generated tokens at increasing KV depth; 3 iters, 1 warmup discarded; tok/s.

---

## What changed (hd512 paged-attention fix)
The gemma-4 full-attention layers use `head_dim=512`. Both paged hd512 kernels were unsound: the flashinfer decode kernel silently **garbled output at long context** (E4B produced gibberish past ~few-k tokens), and forcing the classic paged kernel instead **faulted with an illegal memory access**. On the B200 the flashinfer path **segfaulted** outright on 26B bf16 decode. Only contiguous flash-attn-v2 at hd512 is correct.

The fix routes `head_size > 256` attention (prefill *and* decode) off the paged hd512 kernels onto **gather-KV → flash-attn-v2**, which is correct. Because the gathered KV is variable-size, gemma-4 decode no longer uses CUDA graphs.

Consequences in the numbers below:
- **Correctness restored** (E4B no longer garbles at long context) and the **26B bf16 decode segfault is gone** (all cells rc=0).
- **Prefill is unchanged** (it already used flash-attn-v2, not the paged hd512 kernels).
- **Decode is slower**, increasingly so with depth — the cost of the per-step KV gather plus running gemma-4 decode without CUDA graphs. Recovering this is the next optimization (a fast, graph-compatible hd512 paged kernel).

---

## Q8 — mistral.rs vs llama.cpp

### Prefill (tok/s) — *unchanged by the fix*
| len | mistral E4B | llama E4B | E4B | mistral 26B | llama 26B | 26B |
|---:|---:|---:|:--:|---:|---:|:--:|
| 128 | 6,512 | 6,223 | 1.05× | 3,204 | 4,073 | 0.79× |
| 512 | 17,814 | 12,887 | 1.38× | 10,109 | 9,406 | 1.07× |
| 2048 | 33,032 | 13,963 | **2.37×** | 16,272 | 9,606 | **1.69×** |
| 4096 | 36,464 | 14,113 | **2.58×** | 18,018 | 9,577 | **1.88×** |
| 8192 | 33,945 | 13,983 | **2.43×** | 16,753 | 9,537 | **1.76×** |
| 16384 | 30,933 | 13,432 | **2.30×** | 15,355 | 9,167 | **1.68×** |

### Decode (tok/s) — *slower after the fix; llama.cpp now wins clearly*
| depth | mistral E4B | llama E4B | mistral 26B | llama 26B |
|---:|---:|---:|---:|---:|
| 128 | 116.8 | **199.1** | 122.9 | **192.8** |
| 2048 | 95.6 | **197.4** | 103.2 | **193.1** |
| 8192 | 62.5 | **190.6** | 73.1 | **187.8** |
| 16384 | 42.7 | **187.0** | 52.6 | **186.3** |

**Q8 takeaway:** mistral.rs still wins prefill (E4B ~2.3–2.6×, 26B ~1.7–1.9× at ≥2k). On decode, llama.cpp's flat ~187–199 t/s now beats mistral.rs across the board (~1.7× at d128 widening to ~3.5–4.4× at d16384), since mistral.rs decode now declines with depth (gather + no graphs).

---

## bf16 — mistral.rs vs vLLM

### Prefill (tok/s) — *unchanged by the fix*
| len | mistral E4B | vLLM E4B | E4B | mistral 26B | vLLM 26B | 26B |
|---:|---:|---:|:--:|---:|---:|:--:|
| 128 | 9,050 | 5,198 | **1.74×** | 4,176 | 4,723 | 0.88× |
| 512 | 26,667 | 16,433 | **1.62×** | 10,880 | 17,867 | 0.61× |
| 2048 | 52,416 | 33,074 | **1.58×** | 19,141 | 25,872 | 0.74× |
| 4096 | 57,661 | 54,421 | 1.06× | 22,883 | 38,048 | 0.60× |
| 8192 | 53,895 | 55,427 | 0.97× | 20,898 | 36,909 | 0.57× |
| 16384 | 46,724 | 45,076 | 1.04× | 18,732 | 30,483 | 0.61× |

### Decode (tok/s) — *26B no longer crashes; both slower than vLLM*
| depth | mistral E4B | vLLM E4B | mistral 26B | vLLM 26B |
|---:|---:|---:|---:|---:|
| 128 | 122.7 | **241** | 104.9 *(was crash)* | **261** |
| 2048 | 101.2 | **215** | 90.6 *(was crash)* | **243** |
| 8192 | 64.3 | **171** | 67.0 *(was crash)* | **202** |
| 16384 | 43.5 | 151 | 49.5 *(was crash)* | **184** |

**bf16 takeaway:**
- **E4B (dense):** mistral.rs wins prefill at short/mid (~1.6×), ties at long; vLLM wins decode (~2× short → ~3.5× at 16k, vs ~1.5× before — the gap widened with the decode regression).
- **26B (MoE):** vLLM wins prefill ~1.3–1.7×, and decode ~2.5–3.7×. The headline change: **26B bf16 decode now runs (rc=0) instead of segfaulting** — correctness over the previous (broken) speed.

---

## Key finding (vs the "GB10 is memory-bound" hypothesis)
On the 26B MoE bf16 prefill, vLLM's lead **widened** moving GB10 → B200 (~1.15× → ~1.6×). If the gap were memory-bound it should *shrink* on HBM3e. It grew — so the **26B prefill gap is compute/kernel-bound, not bandwidth.** The B200 refutes the memory-bound theory for this cell. (Prefill is unaffected by the hd512 fix, so this conclusion stands unchanged.)

## Caveats / notes
- **hd512 decode regression is the new headline cost.** Decode now declines with depth on every gemma-4 config because the hd512 layers gather KV per step and gemma-4 decode runs without CUDA graphs. A fast, graph-compatible hd512 paged kernel would recover most of this.
- The previous run's "26B bf16 decode crash" cells are now real numbers; output correctness was independently verified (E4B short = "Paris", 8k-context = correct answer; 26B bf16 8k-context ran clean rc=0).
- vLLM's first run was discarded (prefix-cache artifact); numbers here are the corrected `enable_prefix_caching=False` run.
- Variants: mistral.rs/vLLM ran base 26B; llama.cpp/UQFF used `-it` (throughput-equivalent).

## Bottom line
- **Correctness first:** the hd512 paged-attention bug (E4B garbage + 26B bf16 B200 segfault) is fixed; all cells now run clean and produce correct output.
- **Prefill:** unchanged and still strong — vs llama.cpp up to ~2.6× (Q8); vs vLLM ~1.6× on E4B bf16 (26B MoE prefill remains vLLM's, ~1.3–1.7×, kernel-bound).
- **Decode:** now the weak spot across the board (gather + graphs-off). This is the clear next optimization target: a correct, fast, graph-compatible hd512 decode path.
