YaRN source verification - Phase A evidence (2026-08-16)
==========================================================

Branch: feat/qwen35-yarn-long-context. Scope: pin the qwen35 YaRN partial
rotary to the reference implementations before any rebuild or serve.
Reference order (rule 15): llama.cpp source at /data/llama.cpp first, the
in-repo DeepSeek yarn as pattern, HF yarn formulas cited in llama.cpp
comments only. Nothing here was invented from memory; every number below
reproduces the llama.cpp code path.

Section 1 - What llama.cpp actually computes for Qwen3.5 YaRN

Command-level inputs (the planned serve equivalence):
  --rope-scaling yarn --rope-scale 1.220703125 --yarn-orig-ctx 262144
This sets rope_freq_scale = 1/1.220703125 = 0.8192 exactly
(arg.cpp:2333-2339). yarn_ext_factor defaults to 1.0 for YARN
(llama-context.cpp:172-174).

The GGUF carries NO rope.scaling metadata (only dimension_count 64,
dimension_sections [11,11,10,0], freq_base 1e7), so hparams.rope_yarn_log_mul
== 0 and the log-multiplier branch in llama-context.cpp is skipped:

  yarn_attn_factor = get_mscale(factor, 1.0)              (else branch, line 202)
  yarn_attn_factor *= 1 / (1 + 0.1*log(factor))           (cancellation, line 210)
  yarn_attn_factor *= rope_attn_factor (1.0)              (line 213)

  get_mscale(scale, m) = scale > 1.0 ? 0.1*m*log(scale) + 1.0 : 1.0  (lines 177-179)
  factor = 1/freq_scale = 1.220703125, log(factor) = 0.19942702

So yarn_attn_factor = (1.0199427025) * (1/1.0199427025) * 1.0 = 1.0.
This is the value that reaches ggml as the rope mscale/attn_factor op
param (llama-graph.cpp:1449).

The ggml rope kernel then REWRITES that mscale for ext_factor != 0
(ops.cpp rope_yarn, lines 5838-5845):

  mscale *= 1.0 + 0.1*logf(1.0/freq_scale) = 1.0 + 0.1*log(factor)
          = 1.0199427025

NET TABLE SCALE = yarn_attn_factor * (1 + 0.1*log(factor)) = 1.0199427025.
The cancellation in llama-context.cpp and the re-fold in the kernel cancel
against each OTHER; the leftover is get_mscale(factor, 1.0). A naive read
of only llama-context.cpp (yarn_attn_factor == 1.0) is therefore wrong.

Correctness trap (why the empty GGUF matters): the DeepSeek branch
(llama-context.cpp:184-200) divides get_mscale(factor, mscale) by
get_mscale(factor, mscale_all_dim) because DeepSeek's GGUF carries
rope.scaling.yarn_log_multiplier. DeepSeek models additionally pre-scale the
attention kq_scale in llama.cpp (deepseek2.cpp:438-446). Qwen3.5 has none
of that: qwen35moe.cpp:344 keeps kq_scale = 1/sqrt(head_dim) and the only
magnitude scaling is the 1.0199427025 folded into the rope tables. The
DeepSeek attention-scale pattern must NOT be copied into qwen3_next.

Section 2 - The rope cache geometry (mrope reduces to neox partial on text)

llama-model.cpp:2726-2732 pins QWEN35/QWEN35MOE to LLAMA_ROPE_TYPE_IMROPE.
For text batches llama-batch.cpp:781-788 broadcasts the one position to all
four sections (src_off = batch.token ? 0 : j*n_tokens). In
ggml_mrope_cache_init (ops.cpp:5866-5929) the four thetas all start at that
same position and each is advanced by theta_scale = base^(-2/n_dims) per
cache pair, so the sector selection (ops.cpp:5888-5919) always picks an
identical theta regardless of section. The claim IMROPE equals NEOX partial
on text holds for any rot dim because the section bases never differ for
text; for Qwen3.5 specifically n_rot = 64 = 2 * (11+11+10+0) so the dim
pairs tile the section cycle exactly. rotate_pairs (ops.cpp:5933-5943)
rotates dims (p, p + n_dims/2) with the cos/sin built for cache pair p, so
the value probe below compares one-to-one with new_partial_yarn's table.

corr_dims (ggml.c:4367-4381), with beta_fast = 32, beta_slow = 1:
  corr_dim(32) = 64*ln(262144/(32*2*pi))/(2*ln(1e7)) = 14.2411 -> low  = 14
  corr_dim(1)  = 64*ln(262144/(1*2*pi))/(2*ln(1e7))  = 21.1284 -> high = 22
  mistral.rs new_partial_yarn (layers.rs:2999-3004) matches (clamp to the
  pair count does not bite for these values).

Section 3 - Value evidence (exact llama.cpp pipeline vs new_partial_yarn)

Reference: /tmp/yarn_ref_llamacpp.py, which executes llama-context.cpp
get_mscale plus cancellation, ggml.c corr_dims, ops.cpp rope_yarn ramp and
mscale re-fold, and the shared theta_scale = 1e7^(-2/64). Results:

  corr_dims [14, 22]; theta_scale = 0.6042963902; net mscale = 1.0199427025.

Position 262145 (first beyond native), pair -> (sin, cos):
  pair  0  llama.cpp (-0.9015606498, -0.4769397353)  mistral (-0.9015606498, -0.4769397353)
  pair 15  llama.cpp ( 0.8629114077, -0.5437527184)  mistral ( 0.8629114077, -0.5437527184)
  pair 30  llama.cpp ( 0.0599455498,  1.0181795752)  mistral ( 0.0599455498,  1.0181795752)

The working-tree expectations in yarn_rotary.rs match to the last digit and
pass. The cos magnitude above 1.0 at pair 30 is the net mscale > 1 acting
on the table, exactly as in llama.cpp.

Boundary probes (added as yarn_matches_llama_cpp_reference_at_boundary_positions):
  pos 262144 pair 30: (0.0599453214, 1.0181795887)
  pos 262145 pair 30: (0.0599455498, 1.0181795752)
  pos 319999 pair 13: (-0.1119046827, 1.0137852131)
  pos 319999 pair 30: (0.0731545384, 1.0173158457)

High-frequency pairs (below corr low = 14) are pure extrapolation times the
net mscale: yarn/plain ratio measured 1.0199427025 for pairs 0, 5, 13 at
position 262145 - identical to get_mscale(1.220703125, 1.0).

Section 4 - The wrong fix that was reverted

Commit 641c8cd49 (qwen-model session) removed the '* yarn_mscale(factor,
1.0)' term, setting effective_mscale = 1.0 and correcting the golden
expectations to (-0.883932644, -0.467614244) and similar. It read
llama-context.cpp yarn_attn_factor == 1.0 as the final table scale and
missed the kernel re-fold of (1 + 0.1*log(factor)) in ops.cpp rope_yarn
(lines 5838-5845). Those expectations match no llama.cpp execution path.
The commit was dropped (the branch was unpushed) and the original
cherry-pick code confirmed against the reference.

Section 5 - Verification run (this session)

  cargo test -p mistralrs-core --test yarn_rotary
    yarn_matches_llama_cpp_reference_at_position_262145         ok
    yarn_high_frequency_pairs_are_pure_extrapolation_scaled_by_mscale ok
    yarn_table_extends_to_the_scaled_context                    ok
    yarn_matches_llama_cpp_reference_at_boundary_positions      ok (added)
  cargo test -p mistralrs-core --lib qwen35moe_yarn_override
    qwen35moe_yarn_override_extends_context_and_synthesizes_rope_scaling ok
    qwen35moe_yarn_override_wins_over_gguf_scaling_metadata             ok

  RopeOverride to apply_rope_override (normal_config.rs:195-237) produces
  rope_type yarn, factor 1.220703125, original ctx 262144, beta 32/1, and
  synthesized mscale = 1.0 / mscale_all_dim = 1.0 (no GGUF scaling metadata
  to read). supports_rope_override accepts Qwen3Next and Qwen3_5
  (normal_config.rs:167-172). The synthesized mscale pair is what collapses
  new_partial_yarn's ratio to the single llama.cpp net factor.

Section 6 - Conclusion

No divergence found between mistral.rs new_partial_yarn and the llama.cpp
YaRN pipeline for the Qwen3.5 synthesized config. Effective table mscale is
1.0199427025 in both engines; attention stays at 1/sqrt(head_dim) in both.
Phase A is complete; the next step is Phase B (CUDA rebuild of bin/mistralrs),
then Phase C (serve at 320000).

Section 7 - Rebuild and serve verification (Phases B-C, 2026-08-16)

Phase B: rebuilt bin/mistralrs with cuda + flash-attn
(PATH=/usr/local/cuda/bin cargo build --release --bin mistralrs --features
"cuda flash-attn"; 1m32s, all kernels cached). Copied to bin/ and verified
the serve subcommand exposes --rope-scaling, --rope-scale, --yarn-orig-ctx,
--override-ctx, --paged-attn, --pa-context-len, --pa-cache-type. Binary
reports git revision 5c594e8b2.

Phase C launch on the 12 GB RTX 4070 SUPER (port 8891, log
/tmp/mistralrs-qwen-yarn.log):

  bin/mistralrs serve -m /home/leandro/models/qwen36-a3b
    -f /home/leandro/models/qwen36-a3b/Qwen3.6-14B-A3B-FableVibes-Q4_K_M.gguf
    --port 8891 --host 0.0.0.0 --max-seqs 1
    --paged-attn on --pa-context-len 320000 --pa-cache-type f4
    --rope-scaling yarn --rope-scale 1.220703125 --yarn-orig-ctx 262144

Boot markers observed: PagedAttention KV cache type F4 (packed u8);
10000 GPU blocks, available context length 320000 tokens; 40 layers all on
cuda[0]; model loaded; /health OK; smoke completion OK.

Context proof (ctx_proof.py, target 320000, incremental turns, block-level
prefix caching):
  run 1: served prompts up to ~252k tokens with 99.83% prefix-cache hit
  rate and 267-272 T/s decode, then the step failed with
  CUDA_ERROR_OUT_OF_MEMORY (server survived and kept serving). GPU peak
  11807 MiB of 12282 MiB; cache cost measured 176 KB per 32-token block,
  i.e. 5.5 KB/token F4. The failing request was short the final ~370 MB
  of cache blocks. Cause: total VRAM pressure (weights ~9.1 GB + CUDA
  context + growing F4 cache), not a YaRN bug; positions 262145+ were not
  reached in run 1.
  run 2 (after closing Discord/Chrome GPU consumers, +~0.5 GB headroom):
  reached prompt 275837 / 280261 total on turn 1110 without OOM - past
  run 1's death point (252k) - then killed by the user before the 320000
  target and its recall check ran. Reclaiming the ~0.5 GB of non-server
  GPU memory was enough to cross the earlier ceiling; the F4 cache never
  overflowed on this run (GPU stayed under ~11.8 GB used).

VRAM arithmetic: the F4 per-token cost measured (5.5 KB/token) matches
packed U8 storage (16 bytes per 32 values); the 2x budget over-report in
the startup log is cosmetic (Phase E) - correcting it reallocates no
actual GPU memory, so it cannot unlock 320000 on this card.

RAM offload, for the record: CPU host layers are supported via an explicit
--device-layers split (the remainder of layers defaults to CPU,
device_map/mod.rs), but pipeline/normal.rs:850-856 disables PagedAttention
when the mapper uses any CPU device, so the F4 cache and CPU offload are
mutually exclusive by design. Offloaded layers run the standard bf16 KV
cache in system RAM; context then bounds by RAM instead of VRAM, at a
large decode-speed cost (MoE experts on CPU).