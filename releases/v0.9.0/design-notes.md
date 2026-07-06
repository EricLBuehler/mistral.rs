# v0.9.0 CPU campaign - design notes

Implementation narrative removed from the public report; source material for the Floor
Attention writeup. Numbers here may be superseded by report.md.

## Correctness fix uncovered by this work

The optimization pass exposed a latent bug shipped with the aarch64 repacking kernels: the q4k/q5k tiled (m >= 4) matmul kernels read the interleaved q8k `bsums` with a row-major index while the quantizer stores them quarter-major, corrupting the `dmin` bias term. Any prefill whose token count is a multiple of 4 (i.e., every real chunked prefill) produced garbage activations; short chat prompts happened to take the non-multiple-of-4 generic path and benchmarks used random tokens, so nothing caught it.

Fixed in candle commit `25b498cd` with a repack-vs-reference regression test across m in {1, 4, 8, 23, 512} and all repacked quant types. Verified end to end with long-context recall prompts on qwen3-4b and gemma4-e4b.

## Optimization summary (initial sweep -> this report)

All measured on qwen3-4b q4k unless noted; each change validated by unit tests plus long-context generation checks.

1. Dynamic chunked dispatch in candle's barrier pool (`execute_chunked`, atomic cursor) replacing static per-thread slices in all repacked matmul kernels. Worker spin fell from 32% to 22% of cycles; LFM2.5 decode went from 159 to 456 t/s (2.9x) - tiny models were almost pure imbalance loss.
2. Decode attention (`single_q.rs`) rewritten: kv-axis splitting with online-softmax partial merge, GQA grouping (stream K/V once per kv head for the 4 to 8 q rows that share it), adaptive unit granularity. Deep decode flipped from far behind to ahead (d8192: 8.8 -> 16.7 t/s vs llama.cpp 15.6).
3. CPU fused qkv and gate/up projections sharing one barrier region and one lhs quantization (candle `QTensor::gemv_fused_shared_lhs` + mistralrs wiring), cutting ~72 barrier crossings per token.
4. Sliding-cursor rotating KV cache: the sliding-window cache now slides through slack capacity and relocates once per slack run instead of shifting the whole window every token.
5. `fused_glu` CPU moved from rayon to the barrier pool. This was the single largest win: rayon threads were fighting the barrier workers spinning between matmuls on the same pinned cores. Gemma decode +46%, qwen decode +14%.
6. Prefill attention (`full.rs`): q-blocked K/V streaming (8 query rows share each K/V pass), barrier pool instead of rayon, and binary-searched live kv ranges per row.
7. Blocked prefill attention kernel, the long-context unlock: scores a 128-position KV tile per pass with contiguous mask-row slices, applies the online-softmax correction once per tile instead of per position, and accumulates P*V with each V row shared across the q block. The tile restructure alone reached llama.cpp flash-kernel parity at 8192 tokens (47.8 -> 64.5 t/s on qwen3-4b Q8_0); NEON micro-kernels (4-wide dot with shared q registers, vectorized polynomial exp) pushed past it (67.0 t/s, 1.04x; Q4_K 70.7 t/s, 1.11x).
8. Direct CPU kernels for the GDN path used by qwen3-next-style models (fused causal conv1d + gated-delta-rule time scan), replacing per-timestep tensor-op chains; measured via unit tests, these do not affect the three benchmark models.

## Night 2: winning the curves, and a real CPU MoE path

The first pass won the measured grid; this pass targets the asymptotes (how throughput degrades
with context) and the MoE gap.

1. f16 KV cache on CPU (default; MISTRALRS_CPU_KV_F32=1 opts out). K/V convert once at cache
   append; attention kernels read them with fmlal/fmlal2 asm micro-kernels (f16 NEON intrinsics
   are unstable on stable rustc) and accumulate in f32. Q converts at the attention entry, output
   returns at the activation dtype. Halves attention memory traffic; recall verified on all models.
2. Tiled decode softmax: decode attention scores 128-position kv tiles with one vectorized
   max/exp correction per tile instead of a branchy scalar update per position. Combined with f16
   KV this cuts the decode depth slope from 5.1 to 2.36 ms/tok per 1k context vs llama.cpp's 4.35:
   the decode curve now wins asymptotically (~1.4x at 16384 depth, widening with depth).
3. Prefill slope: 0.89 -> 0.758 ms/tok per 1k (llama.cpp 0.596) via f16 KV plus skipping the
   O(len^2) mask-row reads for binary causal/window masks. Parity holds through 16k (0.97x at
   16384); beyond that llama.cpp's shallower slope would win - the remaining gap is a lane-packed
   fp16 score micro-kernel, a scoped follow-up.
4. Parallel elementwise in candle: contiguous unary/binary maps over 64k+ elements split across
   the barrier pool (with an in-pool reentrancy guard), and mimalloc as the CLI allocator to kill
   tensor-op allocation churn. Together these took lfm2.5-230m dense prefill from 0.69x at 8192 to
   0.92x, and its prefill means above parity (1.05-1.07x).
5. CPU MoE, rebuilt: gather_forward previously dequantized every expert on every call (the
   "dequantize-then-matmul" fallback), which made an 8B MoE effectively unusable on CPU (minutes
   per token). QTensor::indexed_gemv now runs each (token, expert) pair as a gemv against the
   expert's rows inside the repacked weights with one shared lhs quantization pass. lfm2.5-8B-A1B
   q4k: decode 81.2/79.7/75.9/70.3/68.4 t/s vs llama.cpp 79.7/77.1/75.0/69.9/61.8 across depths
   128-8192 (1.01-1.11x ahead, widening with depth), prefill 234.8/232.9/180.8 vs
   229.2/224.0/207.2 (1.02x/1.04x/0.87x after the expert-bucketed matmul; was 0.79-0.94x),
   recall verified. The dequantize path
   survives only as a fallback for unsupported layouts; expert-bucketed GEMM for prefill is the
   scoped follow-up.
6. bf16 activations on CPU: fully working, deliberately not the default. The first attempt looked like
   a lost cause (garbage output) and uncovered a latent candle bug: the bfdot-specialized
   CurrentCpuBF16 keeps raw bf16 bits in its vector unit, and the elementwise add kernels wrote
   against the widening variant's semantics, so every bf16 tensor add on FEAT_BF16 hardware
   returned f32-reinterpreted garbage. With that fixed (explicit widen/add/narrow kernels, vec_mul
   added, CPU matmul widening for unquantized bf16, bf16 lhs accepted by the packed quantized
   matmuls, bfdot attention kernels): qwen3-4b q4k decodes at 32.0 vs 28.0 t/s and prefills 4-10%
   faster than f32; gemma is at parity; the 8B MoE and all dense models pass recall in bf16.
   bf16 shares f32's exponent range, so the f16 residual-overflow hazard does not apply.
   Against the tuned f32 stack, however, bf16 loses 15-20% at decode: ISQ weights and the f16 KV
   cache already bank the memory wins, so bf16 only adds per-op conversion tax across the ~250
   ops per decode token. f32 activations remain the CPU default; --dtype bf16 is fully supported
   (and halves memory for unquantized models). bf16 becomes the default candidate again once a
   native bf16 elementwise pass and a bfmmla GEMM (2x f32 matmul for unquantized weights) land.

## x86: Sapphire Rapids (one day, one rented box)

Raw data: `raw/results_x86.jsonl` (normalized, includes both fa configs per point) and
`raw/x86_sweep.log` (bench stdout).

The aarch64 kernels above do not exist on x86, so a c7i.8xlarge (Xeon Platinum 8488C, 16 cores,
AVX512/VNNI/AMX) was rented to port them. llama.cpp at the same pinned commit (2d97363), native
build with its AMX path active. Baseline before the port: 0.38-0.79x across the board.

End-of-day board (qwen3-4b, mistral.rs / llama.cpp, ratio):

| q4k | 128 | 512 | 2048 | 8192 | 16384 |
|---|---|---|---|---|---|
| prefill | 0.42x | 0.69x | 0.79x | 0.86x | - |
| decode | 1.06x | 1.07x | 1.12x | 1.53x | **1.81x** |

| q8_0 | 128 | 512 | 2048 | 8192 |
|---|---|---|---|---|
| prefill | 0.79x | 0.66x | 0.72x | 0.81x |
| decode | 0.84x | 0.84x | 0.92x | **1.33x** |

llama.cpp's flash-attention CPU kernel inverts at depth on x86 (fa=1 beats fa=0 through ~2k,
then loses to it: 6.3 vs 8.8 t/s at 16384), so the decode ratios above take llama.cpp's best
configuration at every point, mixing fa=1 shallow and fa=0 deep.

Decode wins q4k at every depth and both quants at agent depths, with the same widening-with-depth
shape as the aarch64 curves (15.9 vs 8.8 t/s at 16384 depth); recall verified. Prefill trails llama.cpp's mature AMX path (a chip
feature most of the x86 fleet lacks; a non-AMX comparison point is a follow-up).

What was built (candle + mistralrs, all runtime-feature-detected):
1. AVX512-VNNI repacked kernels for q4k/q6k/q8_0: weights tiled [n/16][k][16 rows] so one
   vpdpbusd yields 16 output columns; gemv, m-tiled, and fused multi-projection paths share the
   layout; min/offset terms corrected through the q8k bsums.
2. An AMX-int8 macro-tile path (m>=32, q4k): 2x2 C tiles per A/B load pair via asm (the
   intrinsics are unstable), k=32 sub-tiles so per-subblock scales apply exactly.
3. AVX512 attention micro-kernels: dots, 16-lane Cephes exp, f16 K/V via vcvtph2ps, and the
   change that mattered most - register-tiled P.V accumulation holding 4 q-rows' accumulators in
   the full zmm register file across each kv tile. Decode attention went from 22 GB/s effective
   to 95 GB/s (the instance's bandwidth ceiling) with that one restructuring.
4. GEMM-structured prefill scoring: K tiles transposed once through a 16x16 shuffle network,
   16 scores per fma, no horizontal reductions.

The consumer tier landed: AVX-VNNI (vpdpbusd ymm) and pure AVX2 (maddubs) kernel variants plus
256-bit attention micro-ops and F16C f16 KV, so non-AVX512 x86 (Ryzen, pre-Ice-Lake Xeon,
laptops) runs real kernels instead of scalar fallbacks - forced-AVX2 on Sapphire Rapids reaches
~75% of the AVX512 path and passes recall. A direct aarch64 port of the register-tiled P.V was
measured and REJECTED (-9 to -13% at depth: NEON's register file forces per-row V re-streaming);
the ~2x aarch64 attention-bandwidth headroom (50 vs 130 GB/s) needs an ARM-native design.

Follow-ups: AMX epilogue amortization, a non-AMX comparison point, an ARM-native attention
bandwidth design, and the q8_0 shallow-decode cell (llama.cpp's q8_0 gemv sustains a few percent more effective
bandwidth per core; block-pairing landed, the rest needs deeper streaming work).

## Late addition: register-held P.V on aarch64

After the x86 register-tiled P.V trick landed, the aarch64 port took two attempts: a direct
copy regressed 9-13% at depth, and profiling showed the culprit was a per-position p == 0 skip
branch (never taken in decode, mispredicted every position), not the V re-streaming the smaller
NEON register file forces. The branch-free version (accumulator pinned in 64-f32 register
chunks, masked positions zeroed before the sweep) lands d8192 22.8 -> 23.3 and d16384
15.6 -> 16.6 t/s, bringing the ARM deep-decode ratio to 1.79x - within a hair of the x86 1.81x.
In-engine decode attention now runs at ~68 GB/s effective on GB10 against a ~130 GB/s machine
ceiling; the remaining gap (scoring-phase horizontal reductions, second V pass) is scoped for
the next release.
