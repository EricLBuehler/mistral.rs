# cuTile MoE prefill optimization — progress

Goal: match/beat vLLM on bf16 prefill (gemma4 26b-a4b) without regressing decode.

## Baseline & target (cutile_vllm_prefill_check_20260528, matched sequential tokens)

| Prompt | cuTile (us) | vLLM | gap |
| ---: | ---: | ---: | ---: |
| 2048 | 3903 | 4586 | +17% |
| 4096 | 4048 | 5133 | +27% |
| 8192 | 3874 | 5195 | +34% |
| 16384 | 3583 | 4379 | +22% |

My fresh baseline bench (prewarm, pa-ctx 10000): 2048=3955, 4096=4048, 8192=4064.

Key fact: that vLLM run used **Triton + the default E=128,N=704 config on GB10** — the SAME `get_default_config` we copied. So the gap is NOT config and NOT FlashInfer; it's **Triton's pipelined codegen (num_stages cp.async) vs our naive synchronous K-loop**. Our prefill T/s is FLAT across M (3955->4064) and ~10-20% of Blackwell tensor-core SOL => per-tile load latency not hidden (memory/latency-bound), lots of headroom.

Decode: we're already AHEAD of vLLM (~31 vs ~24 T/s). Must not regress.

## Diagnosis (static, no profiling — ncu crashes machine, nsys unreliable here)

- Our kernel uses `load_ptr_tko` (raw pointer tiles, register-direct, synchronous Latency::<0>) + manual per-iter masking + `select`.
- cuTile's idiomatic high-perf GEMM (its flash-attn test) uses `tensor.partition(tile).load([idx])` = `load_view_tko` with **TMA** (async bulk copy, compiler-pipelined) + `padding::Zero` auto-bounds.
- In-kernel view construction exists: `make_tensor_view(ptr, shape, strides, token)` + `.partition(tile)` -> TMA loads WITHOUT a host Tensor-param launch redesign.

## Experiments

- Exp1 — Latency::<0> -> Latency::<4> on A/B loads: NO help (2048 3811, 8192 3883, ~flat/slightly worse). cuTile does NOT software-pipeline the pointer-load loop via the latency hint. => pointer path can't pipeline; need TMA.
- Exp2 — B via TMA partition view (make_tensor_view + partition + load_view_tko, padding::Zero), A stays gather; dropped manual B masks + offs_bn%N; added e_size param. **JITs clean (0 warmup failures) + correct output (Paris).** In-kernel make_tensor_view + load_view_tko TMA works in our borrowed-context launch. Bench [running].
  - Gotchas: `Some(bf16::ZERO)` as load_ptr_tko padding fails JIT ("associated const in expression position"); use mask + `select(mask, load, constant(bf16::ZERO,..))`. Stale failing `mistralrs run` proc holds GPU mem (engine error-loops, never exits) -> kill by PID (NOT `pkill -f "mistralrs run"` — matches the wrapper bash). GB10 nvidia-smi shows memory "Not Supported" (unified mem).

## Plan / contingencies

- If Exp2 works + faster: escalate to pre-gather A into contiguous buffer -> both operands TMA -> fully pipelined (the standard high-perf MoE prefill design). Possibly split prefill vs decode kernels.
- If Exp2 correct but not faster: A's sync gather bottlenecks -> pre-gather A.
- If Exp2 JIT-fails: in-kernel view+TMA unsupported in our borrowed-context launch -> try host-side Tensor param, or high-level cutile gemm with pre-gathered contiguous A.

## RESOLUTION (2026-05-29): in-kernel TMA dead, config dead, host-Tensor-param TMA is the path

- In-kernel make_tensor_view + partition_permuted (Exp3): 2429/2682/2609 -- ~35% SLOWER than pointer (~4000). In-kernel views do NOT lower to real TMA (TMA descriptors are host-side). Reverted to pointer baseline (`git checkout HEAD`).
- MMA has NO transposed-rhs (NT) mode (mma/mmaf/mmai all lhs[M,K]xrhs[K,N]). With ENK [E,N,K], any view-based B load needs a register transpose -> slow.
- Config sweep (MISTRALRS_MOE_PREFILL_CFG hook in mod.rs, large-M regime only): baseline 128,128,64 (~4012/4213/4038) is BEST. 64,64,64=3376, 128,128,128=3239, 64,128,64~=baseline, 64,128,128~3880. Smaller tiles WORSE (occupancy hypothesis wrong; bigger acc/arith-intensity wins). Pointer ceiling firmly ~4000-4200.
- Idiomatic FAST cuTile GEMM (cutile-benchmarks/benches/gemm.rs, examples/batch_matmul.rs): host `&Tensor` params + `partition().load()` (real TMA) + **B stored [K,N]** (y=ones([k,n]), no transpose). batch_matmul: `b:&Tensor<{[-1,K,-1]}>` (=[batch,K,N]).
- KEY INSIGHT (transposed-output trick): compute acc[N,M] = W[N,K] @ A_gT[K,M] instead of acc[M,N]=A@W^T. Then lhs=W in HF-native [E,N,K] (partition [1,BN,BK] contiguous -> TMA, ZERO weight transpose, ZERO extra memory, decode weights untouched); rhs = A gathered as [K,EM] (gather writes this layout) -> TMA. Output transposed [N,EM] chains: transposed-gelu -> down GEMM -> transposed scatter-sum.
- Feasibility CONFIRMED: `cutile::tensor::Tensor::from_raw_parts(dptr, len_bytes, dev_id, shape, strides)` aliases candle buffers (no copy). DeviceBuffer is OWNING (Drop->free_async), so MUST `std::mem::forget` the cuTile wrappers after launch (candle keeps ownership). Launcher returns args back as tuple; forget the result. device_id = dev.cuda_stream().context().ordinal(). No cutile-rs edits.
- Design: NEW prefill path (fused_moe_prefill.rs), decode untouched. gather A->[K,EM] (a_gt), TMA GEMM moe_gemm_t (Tensor params, transposed out), transposed gelu, transposed moe_sum/scatter. BM=block_size=cfg.bm so each m-block=one expert (eids[m_idx]).
- gemma4 dims: gate_up N=1408 K=2816; down N=2816 K=704. cfg prefill bm=128 bn=128 bk=64. All divide exact.

## FINAL CONCLUSION (2026-05-29): the MoE is NOT the prefill bottleneck

Built + validated `cutile_grouped_gemm_t` (host-Tensor-param TMA GEMM, transposed-output trick, zero weight transpose) in fused_moe_prefill.rs. Correct (max_abs_err 0.12). Perf:
- TMA GEMM ~= pointer GEMM at all prefill shapes (0.85-1.09x). Both bandwidth/L2-bound (~58 TF). TMA does NOT help.
- Full cuTile MoE pipeline vs vLLM fused_experts (isolated, same shapes): ours 9.30/11.95/18.89ms vs vLLM 8.94/11.17/17.08ms at 2048/4096/8192. Only +4/+7/+11% slower. MoE is COMPETITIVE.

The prefill gap (4000 vs 5000 T/s, GROWS with seqlen) is dominated by **ucopy_bf16 copy churn**: nsys p8192 shows ucopy = 15.2% (was 9.7% at p2048 -> grows with n), ~20 strided transpose-copies/layer (~520us each). Root cause = attention BSHD<->BHSD transpose round-trips: gemma4 text.rs transposes q/k/v BSHD->BHSD (for rope + eager contract), attention/mod.rs run_attention re-transposes BHSD->BSHD for flash (flash_attn_v2 wants BSHD), flash returns BSHD, mod.rs:202 -> BHSD, text.rs:656 -> BSHD. Net: ~4 wasted transposes/layer for q + output, materialized as strided copies.

Prefill breakdown @ p8192: MoE ~40%, ucopy 15%, attention (flashinfer+flash, NOT eager) ~14%, dense nvjet GEMMs ~15%, norms ~7%, rest. Decode preserved (31.6 T/s).

**Recommended next step (NOT done -- cross-cutting + correctness-risky, deferred for review):** make the attention chain stay BSHD end-to-end (vLLM-style) so flash consumes/produces BSHD with no transposes. Requires a BSHD rope path + relaxing run_attention's BHSD return contract (touches flash/eager/paged/decode + all models) -> needs careful before/after logit validation. This is where the ~25% prefill gap actually lives. The cuTile MoE work (the original hypothesis) is already competitive and needs no change.
