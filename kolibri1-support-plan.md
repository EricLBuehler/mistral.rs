# Kolibri-1 on CPU: support plan

Target: `Aleph-Alpha/Kolibri-1` (`model_type: kolibri1`, `Kolibri1ForCausalLM`) running
in mistral.rs on CPU, aarch64 NEON, 128 cores.

External references read while preparing this (they live outside this repo, so they are
named by path rather than linked, per the `doc-links` CI rule):

- `../vllm.cpp/.agents/specs/kolibri-1-cpu.md` (pinned architecture writeup, primary source)
- `../vllm.cpp/src/vllm/model_executor/models/kolibri1_forward.cpp` (working CPU forward)
- `../vllm.cpp/src/vllm/model_executor/models/kolibri1_registry.cpp` (config parse/validate)
- `../vllm.cpp/src/vllm/model_executor/models/kolibri1_weights.cpp` (real weight inventory)
- `../vllm.cpp/.agents/oracles/aleph-alpha-inference.md` (plugin pin `049a6a7bd240`)

## 1. Repo rules that shape this work

Read from `AGENTS.md` and `.github/workflows/ci.yml` before writing any code here.

### Hard CI gates every PR must pass

| Gate | Command | Notes for this work |
|---|---|---|
| check | `cargo check` on ubuntu, windows, macOS runners | no features, so CPU paths only. `#[cfg(target_arch = "aarch64")]` NEON code compiles on macOS-latest (arm64) but not on the linux/x86 runners |
| test | `cargo test -p mistralrs-core -p mistralrs-quant -p mistralrs-vision` | needs `TESTS_HF_TOKEN` for some tests |
| fmt | `cargo fmt --all -- --check` | |
| clippy | `cargo clippy --workspace --tests --examples -- -D warnings` | hand-written intrinsics trip lints, budget time for this |
| doc | `cargo doc --workspace` | public doc comments must compile |
| typos | `crate-ci/typos` with `.typos.toml` | |
| doc-links | greps every `*.md` outside `docs/` for relative links into `docs/` | this file therefore names site pages in prose and never links them relatively |

### Generated-doc gate specific to adding a model

`make docs-check` runs `cargo test -p mistralrs-core supported_models_matches_committed`.
Adding `NormalLoaderType::Kolibri1` changes the loader registry, so the committed
`docs/src/content/docs/reference/supported-models.md` goes stale and the gate fails until
you run:

```
cargo test -p mistralrs-core regenerate_supported_models -- --ignored
```

`model_metadata.rs` therefore needs three edits for the new arch, not one: `arch_metadata()`
(families / modalities / examples), `config_arch()`, and the loader enum itself.

### Code style constraints that apply to every line written here

- ASCII only. No em dashes, en dashes, ellipsis glyphs, smart quotes, or box drawing. No `--`.
- Comments are the exception, not the rule. One line, under ~120 cols, only when the *why* is
  not obvious from the code. No section banners, no narration of the next line, no restating
  the function name.
- Do not reference this plan, the task, or a PR number in comments. That rots.
- Magic values become named `const`s at the top of the file. A sentinel crossing a module
  boundary must be a `pub const` so both sides share it, not a literal both sides happen to
  spell the same way.
- Six or more arguments means wrap the invariants in a small context struct.
- No backwards compatibility shims unless explicitly asked for.
- Commit style: `feat(crate): ...`, `fix(crate): ...`, `docs: ...`.
- Run `cargo check` before returning anything. Do not hand back TODOs.

### Known gap worth flagging upstream

CI runs ubuntu/windows/macOS only (`ci.yml:24`). There is no aarch64 Linux runner, so the
target platform for this work (aarch64 Linux, 128 cores) gets compile coverage on macOS but
no linux coverage at all. The sibling `../vllm.cpp` repo has a `ci-aarch64-coverage.md`
spec and an explicit aarch64 lane; the equivalent here would be a cheap addition and is
worth proposing as its own PR, since every Track A change below is NEON code.

## 2. Architecture summary

Confirmed against the checkpoint index and the plugin, not the task brief.

| Field | Value |
|---|---|
| hidden_size | 2560 |
| num_hidden_layers | 50 |
| num_attention_heads / num_key_value_heads | 48 / 4 (GQA 12:1) |
| head_dim | 128 (explicit, full rotary, no partial) |
| layer_types | 4x `sliding_attention` then 1x `full_attention`, repeating |
| sliding_window | 513, `use_sliding_window: true` |
| rope_theta | 10000 |
| num_experts / num_experts_per_tok | 384 / 6 |
| moe_intermediate / shared_expert_intermediate | 512 / 512 |
| norm_topk_prob | false |
| rms_norm_eps | 1e-6, hidden_act silu |
| vocab_size | 128000 |
| max_position_embeddings | 262144 |
| quantization_config | `fp8`, dynamic activation, `weight_block_size: [128,128]` |
| dtype / head_dtype | bfloat16 / float32 |
| tie_word_embeddings | false |

78.1B total, 3.46B active, every layer is MoE.

### The four structural oddities

All four are now confirmed against the oracle source
(`aleph_alpha_inference/kolibri1.py` at pin `049a6a7bd240`) and the model card.

1. **RNoPE.** `Kolibri1Attention.__init__` sets `self.rotary_emb = None` and
   `sliding_window = None` when `is_full_attention`, and calls `get_rope(...)` otherwise. In
   `forward`, `if self.rotary_emb is not None: q, k = self.rotary_emb(positions, q, k)`.
   RoPE exists only on the 40 sliding layers. The model card states the same. No existing
   model in this repo skips rope on some layers.

2. **Sandwich post norms.** CONFIRMED, with exact placement, by `Kolibri1DecoderLayer.forward`:

   ```python
   if residual is None:
       residual = hidden_states
       hidden_states = self.input_layernorm(hidden_states)
   else:
       hidden_states, residual = self.input_layernorm(hidden_states, residual)
   hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states)
   hidden_states = self.post_attn_norm(hidden_states)
   hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
   hidden_states = self.mlp(hidden_states)
   hidden_states = self.post_ffn_norm(hidden_states)
   return hidden_states, residual
   ```

   This is the vLLM RMSNorm "residual out" contract: `input_layernorm`,
   `post_attention_layernorm` return `(normed, residual + normed)`, while `post_attn_norm`
   and `post_ffn_norm` are plain norms. The residual stream accumulates post normed tensors.
   On the first (only) call for a prefill chunk, `residual` is seeded from `hidden_states`
   and `input_layernorm` runs without one. In the steady state both fused norms run. mistral.rs
   `layers.rs:333` (`forward_residual`) is exactly the two-output form this needs.

3. **Sigmoid logit add router.** `sigmoid_logit_add_routing` in the oracle:

   ```python
   logits = gating_output.float()
   topk_ids = torch.topk(logits + e_score_correction_bias, k=topk, dim=-1)[1]
   topk_weights = torch.sigmoid(logits.gather(1, topk_ids))
   if renormalize: ...
   ```

   Selection on `logits + bias`, weight from unbiased `sigmoid(logits)`, fp32 throughout.
   `Kolibri1SparseMoeBlock` uses an ungated `shared_experts` Qwen3MoeMLP plus 384 routed
   experts at top-6. The oracle's own test (`test_routing_semantics`) pins the reference to
   torchtitan's `TokenChoiceTopKRouter` with `route_norm=False, route_scale=1.0`, and
   explicitly asserts selection DIFFERS from vLLM's built-in sigmoid scoring. In mistral.rs
   terms: `moe_router_topk` with `score_function: Raw`, `selected_weight: Sigmoid`,
   `renormalize: false`, and `selection_bias: Some(expert_bias)`.

4. **FP8 block checkpoint.** The oracle refuses DeepGEMM's E8M0 path with an explicit error:
   "Kolibri 1 FP8 weights carry fp32 block scales, which DeepGEMM's E8M0 path would round to
   powers of two." This independently confirms the f32 scales we measured on disk, and warns
   that any power-of-two scale optimization is wrong for this model.

### Window semantics: resolved

The card states the sliding window as **"512 preceding tokens plus the current token"**.
That is 513 total, i.e. left distance 512, exactly what `sliding_window: 513` encodes and
exactly what `attention/mod.rs:70-72` computes with `window - 1`. Risk R3 in section 7 is
closed by the primary source. The oracle passes `per_layer_sliding_window=sliding_window` to
vLLM's `Attention`, so the mask comes from vLLM's SWA layer; we reproduce it via the mask.

### Attention details worth copying exactly

- Separate q/k/v projections, no bias (`QKVParallelLinear` with `bias=False`), not a fused
  bias-carrying path. `o_proj` is a `RowParallelLinear`, also no bias.
- Per head qk norm BEFORE rope, in that order: `q_norm`/`k_norm` are
  `RMSNorm(self.head_dim, eps)`, applied to a view of shape `[..., num_heads, head_dim]`.
  The gamma is shared across heads (one vector per projection). mistral.rs
  `forward_qk_norm` in `qwen3_moe.rs:221-229` matches this shape.
- Scale is `head_dim ** -0.5` = `128 ** -0.5`.
- `config.rope_parameters` is read via `get_rope(...)`. The test asserts
  `config.rope_parameters["rope_theta"] == 10000.0`, so the plugin's config class normalizes
  flat `rope_theta` into `rope_parameters`. Accept both forms (risk R2).

### What else the source settles

- `head_dim` is read from config, not derived. Config says 32 in the tiny test and 128 in the
  real checkpoint.
- No attention bias, `attention_dropout: 0.0`.
- `lm_head` untied (`ParallelLMHead`), `logits_processor` standard.
- The plugin does nothing special for GGUF or CPU; it is a vLLM GPU plugin. Our CPU port is
  independent.

### Everything else the card confirms

Hidden 2560, 50 layers, 48 heads / 4 KV heads, head size 128, per head RMSNorm on q and k,
SwiGLU, vocab 128,000, 4:1 SWA to full GQA, `norm_topk_prob` false, untied lm head. The card
gives 78,103,074,560 total parameters and 3,457,573,120 active per token. Native context
262,144, with 1,048,576 validated by extrapolation via a `max_position_embeddings` override.

Two details for the serving work (B4):

- **Recommended sampling is `temperature=1.0, top_p=0.97, top_k=128`**, not greedy. Any
  token-exactness gate must set `temperature=0` explicitly rather than inheriting a default
  that is not the model's own.
- The reference server invocation uses `--kv-cache-dtype fp8`. mistral.rs has no CPU fp8 KV
  cache path (`F8E4M3` appears only as a weight/ISQ type and a CUDA paged cache type at
  `pipeline/loaders/auto_device_map.rs:593`). On CPU the KV cache will be bf16, costing
  100 KiB/token against roughly 50 for fp8. That is a real deviation from the reference
  serving config and should be documented rather than left implicit.

### Real weight names

Confirmed against the oracle's `WeightsMapper` and the plugin's synthetic checkpoint
generator (`tests/checkpoints.py`), which agrees with the index we measured on disk.

```
model.embed_tokens.weight
lm_head.weight                                  (untied)
model.norm.weight
model.layers.N.input_layernorm.weight
model.layers.N.post_attn_norm.weight
model.layers.N.post_attention_layernorm.weight
model.layers.N.post_ffn_norm.weight
model.layers.N.self_attn.{q,k,v,o}_proj.weight      + .weight_scale_inv
model.layers.N.self_attn.{q,k}_norm.weight
model.layers.N.mlp.gate.weight                 (router gate, bf16)
model.layers.N.moe.router.expert_bias          (outside mlp, bf16)
model.layers.N.mlp.experts.{i}.{gate,up,down}_proj.weight + .weight_scale_inv
model.layers.N.mlp.shared_experts.{gate,up,down}_proj.weight + .weight_scale_inv
```

The gate and the router bias do not share a prefix. The oracle maps
`.moe.router.expert_bias` onto the internal `.mlp.gate.e_score_correction_bias` by substring
rewrite (`orig_to_new_substr`), and fuses shared-expert gate/up into one `gate_up_proj`.
We load the on-disk names directly and do not need any of that remapping. 32 shards,
`metadata.total_size = 78,827,029,120`.

## 3. What mistral.rs already provides

Reusable, with locations:

- Per layer sliding/full cache split: `gpt_oss.rs:52-57` (`LayerType` enum) and
  `gpt_oss.rs:851-860` (`NormalCacheType::SlidingWindow` / `Normal` per layer).
- Per layer mask selection from two prebuilt masks: `gpt_oss.rs:887-938`.
- Sigmoid router with selection bias: `ops.rs:274-344`. `score_function: Raw` plus
  `selected_weight: Sigmoid` plus a `selection_bias` *is* sigmoid logit add.
- Per head qk norm before rope: `qwen3_moe.rs:221-229` via `forward_qk_norm`.
- Add rms norm and residual carrying norm: `layers.rs:329-360`, portable fallback at
  `layers.rs:454-460`.
- Ungated shared expert added to routed output: `glm4_moe.rs:504`, `:544-578`.
- Expert projection naming: `moe/experts/config.rs:62-68`, `DEFAULT` is already
  `gate_proj`/`up_proj`/`down_proj`.
- FP8 config parsing incl. the `weight_scale` / `weight_scale_inv` alias:
  `mistralrs-quant/src/lib.rs:527-564`, `fp8_config.rs:16`, `:30-73`.
- CPU attention kernels with hand written NEON: `attention/backends/cpu/neon.rs` (812
  lines), `avx.rs` (832 lines), dispatch at `attention/mod.rs:284-290` and `:296`.

## 4. Track A: generic CPU work

These are **not** Kolibri1 features. Each benefits every model on CPU and should land as its
own PR, benchmarked independently, with Kolibri1 as a downstream consumer rather than the
thing that discovers the problem. The advantage is that none of them is gated on downloading
a 78 GB checkpoint.

### A0. CPU thread-count policy

Folded into the GEMV item below as its step 2. The measurements that motivated it (decode
19.8 tok/s at 16 threads vs 7.5 at 32 on a dedicated node; 3.5 at 128 whole-host) are in
section 8, and the design constraints (scale with cores actually available, shared helper
across both pool construction sites, `MISTRALRS_CPU_THREADS` override following the existing
`MISTRALRS_*` convention) are carried into A0-gemv. Kept as a separate commit only if it
lands ahead of the GEMV work, since it is independently testable and independently useful.

### A1. Cache FP8 dequantization

`blockwise_fp8/mod.rs:447-454` is the fallback in `forward_raw`:

```rust
let weight = self.dequantize_w()?;
let unquant = UnquantLinear::new(...)?;
unquant.forward(x)
```

`dequantize_w` (`:287-294`) is uncalled-once, uncached, and runs *per forward*. On GPU the
CUDA early returns above it make this dead code; on CPU it means every decode step re-expands
the full weight matrix of every active expert.

Change: a byte budgeted cache of dequantized weight arenas inside `BlockwiseFP8Linear`.
Entries own their bytes, the budget is a `const`/config knob, and a lease is held across the
matmul so an eviction cannot free memory under a running GEMM.

Constraints: the change must sit strictly *after* every `cfg(feature = "cuda")` early
return in `forward_raw` so the CUDA fast paths are provably untouched. This is a
`QuantMethod` impl shared by all fp8 paths, so the blast radius is every fp8 model.

The sibling repo measured this lever at 2.52x on the same model, which is the number to
beat.

Gate: dequant output must be bit-identical to the uncached path (it is a pure cache, so any
difference is a bug). Unit test for eviction under concurrent-ish use, plus a bench at
several budget settings.

Commit: `feat(quant): budgeted dequant cache for blockwise fp8 on cpu`

### A2. Give the CPU attention kernels a sliding window

The CPU kernels only ever consume a materialized mask tensor. `attention/mod.rs:284-290`
routes `AttentionMask::Custom` into them, and `sdpa_params.sliding_window` is never read on
CPU (grepping `sliding` across `attention/backends/cpu/*.rs` finds only bench literals).
The mask is applied per element inside the inner loop at `single_q.rs:334-339` via the
`MaskInfo` indirection in `mask.rs`.

So on CPU every SWA model pays O(q_len x kv_len) mask memory and a per element mask lookup
in the hot loop, where CUDA's flash attn takes the window as a scalar. Affected today:
gemma2, gemma3, qwen3, qwen3_moe, glm4, gpt_oss, and Kolibri1.

Change: pass the window into `CpuAttnCtx` and bounds check arithmetically in place of the
per element lookup, keeping the mask path for genuinely custom masks.

Constraints: this is a shared kernel. It needs a golden output comparison per architecture
that uses sliding windows, not just a unit test, and the mask path must remain for
bidirectional and vision attention.

Gate: identical logits to the current mask path on SWA models, and the window 513 boundary
case specifically (a short prompt hides an off by one; see risk R3 below).

Commit: `feat(core): sliding window in cpu attention kernels`

### A0-gemv. CPU decode GEMV: reach the memory bandwidth (the real bottleneck) [DONE: 8.2x]

Section 8 measured the actual problem: decode tops out at 10.1 GB/s on a node that streams at
122.7 GB/s, and 10.1 GB/s is exactly what a single thread achieves. The CPU linear layer is
the bottleneck for every model, dense or MoE, and it is why nothing else moves.

The decode step is a batch-1 quantized or bf16 GEMV, `[1, K] x [K, N]`. It must stream the
entire weight tensor once per token, so it is purely bandwidth-bound and the ceiling is
`weight_bytes / bandwidth`. Before this work it was neither bandwidth-saturating nor scaling
with cores: it ran at one core's bandwidth whether given 1 thread or 32.

What this item delivered, in order:

1. **Parallelize the weight stream.** DONE. `mistralrs-quant/src/gemv_cpu.rs` partitions N
   across the rayon pool, one output row per thread.
2. **NEON inner loop.** DONE. Four-accumulator `vfmaq_f32` dot, matching the CPU attention
   backend.
3. **Pool sizing.** DONE. `cpu_threads::resolve()` sizes the decode GEMV pool and the
   flash-attention pool together from `MISTRALRS_CPU_THREADS`, then `RAYON_NUM_THREADS`,
   then candle's core count, so the pools mistral.rs owns cannot drift apart. No ratio is
   applied: a half-core default was tried and measured worse (see section 8). The tokio
   engine runtime (`mistralrs-core/src/lib.rs:36`) still uses candle's core count directly;
   it is sized for the engine rather than for the decode inner loop, so it was left alone.
4. **Fix the quantized dequant path.** NOT DONE. `mistralrs-quant/src/gguf/cpu.rs:74-75`
   still calls `qtensor.dequantize(device)` per call. This is why the 0.8B Q4_K_M number did
   not move. This is now the top remaining item inside A0-gemv and overlaps A1.
5. **Parallelize the aarch64 packed MoE GEMV.** NOT DONE. Candle's `quantized/repack.rs` has
   no parallel constructs, so `indexed_gemv` is serial (see A5).

Bit-exactness is the binding constraint, and it held: the parallel path is asserted
bit-identical to the serial loop (`parallel_matches_serial_bit_exactly`), the NEON dot keeps a
fixed per-row reduction order, and the only difference against candle's vectorized matmul is
FMA-vs-scalar rounding (`gemv_tracks_candle_matmul`, tolerance 1e-3 relative).

Result: Qwen3-4B bf16 decode 0.6 to 6.5 tok/s (10.8x) with `RAYON_NUM_THREADS=16`, or 2.6
tok/s (4.3x) at the default; Qwen3.5-9B bf16 decode 0.3 to 2.1 tok/s (7x). All on one node,
prefill unchanged.

Gate: MET for the bf16 dense path. Open for the GGUF quantized path and the packed MoE path.

Commits: `perf(quant): output-parallel CPU GEMV for decode`,
`perf(quant): NEON dot for the output-parallel CPU GEMV`,
`perf(core): size the CPU attention pool from the shared resolver`

### A5. Parallelize the CPU quantized MoE path (new)

Raised by the 35B diagnostic in section 8, which found the CPU MoE decode running on 1.9 of
128 cores with 1 running thread out of 178.

Two independent defects, both in the path a Kolibri-1 decode would take on every layer:

1. **The fallback dequantizes everything every call.**
   `mistralrs-quant/src/gguf/cpu.rs:74-75` calls `qtensor.dequantize(device)` to expand all
   expert weights to f32 on every forward, then routes through `UnquantLinear::gather_forward`.
   This is the same failure mode as A1, and for a 384-expert model it is catastrophic.
2. **The packed path that avoids it is serial.** `QTensor::indexed_gemv` has an aarch64 CPU
   implementation, but the underlying `candle-core/src/quantized/repack.rs` has no parallel
   constructs at all, so even the fast path uses one core.

Change: parallelize the aarch64 packed indexed GEMV over experts (and/or over output rows),
and make the dequantize fallback cached rather than per-call, as A1 describes. Both need to be
bit-reproducible under the new partitioning, since expert output is summed and accumulation
order must not drift.

Gate: decode on a MoE checkpoint must report more than one core busy, and must scale with
thread count up to a plateau. Bit-identical logits before and after.

Commit: `perf(quant): parallelize cpu quantized moe forward`

Note this is where candle is a dependency, not just mistral.rs: `repack.rs` is upstream. If
the packed path cannot be fixed in candle, the alternative is a mistral.rs-side packed
implementation for aarch64, which is more work but stays in this repo.

### A3. NUMA aware weight allocation

The only CPU topology code in the repo is `set_thread_affinity`
(`attention/backends/cpu/threading.rs:7`). There is no first touch policy and no
interleaving anywhere.

The sibling repo measured this precisely: default first touch leaves weight pages on the
loading node, and `numactl --interleave=all` with 32 threads took decode from 3.17 to 4.69
tok/s, a 47% gain. On a 128 core box this is plausibly the largest single lever available,
and it is a small, contained change to the weight arena allocation policy.

Gate: numerics must be bit-identical (it is purely an allocation policy), so the existing
greedy output tests are the gate. Measure with and without.

Commit: `perf(core): numa aware first touch for cpu weight arenas`

### A4. CPU paged attention (stretch)

`pipeline/normal.rs:961-967` disables paged attention whenever the device map touches CPU.
That costs continuous batching, prefix caching, and bounded KV memory, so it is the main
thing between "CPU works" and "CPU is a serving target". Much larger than A1 to A3 and worth
scheduling separately rather than bundling.

## 5. Track B: the Kolibri1 model

Genuinely Kolibri1 specific, but almost all of it is assembly from existing parts.

### B1. Registry (mechanical)

`mistralrs-core/src/models/mod.rs`, then in `pipeline/loaders/normal_loaders.rs`:
`NormalLoaderType::Kolibri1` plus the four match arms (`model_type_name`,
`from_causal_lm_name`, `FromStr`, `Display`), a `Kolibri1Loader`, and the
`AutoNormalLoader::get_loader` arm. Then `pipeline/normal.rs:503-540`, and
`model_metadata.rs` (`arch_metadata` *and* `config_arch`, per section 1). Then
`mistralrs-pyo3/src/which.rs`.

Do not add the GGUF registry entries yet (`gguf/normal_registry.rs`,
`gguf/normal_config.rs`). We are safetensors first, and a GGUF arch with no converter is
dead surface that the supported models generator would then advertise.

Required: `cargo test -p mistralrs-core regenerate_supported_models -- --ignored` and commit
the result, or `make docs-check` fails.

### B2. Config and weight loading

`Config` per section 2, including `head_dim`, `layer_types`, `head_dtype`, and both the flat
`rope_theta` and the `rope_parameters` object form (risk R2). Validation mirrors
`kolibri1_registry.cpp:140-150`: `layer_types.len() == num_hidden_layers`, entries are one of
the two kinds, `use_sliding_window` implies a positive window.

Weight load walks `.pp("model")` and handles the `mlp.gate.weight` /
`moe.router.expert_bias` split explicitly, since they are not siblings under one prefix.

Gate: a synthetic tiny checkpoint in the exact layout loads tensor complete, plus a
manifest test that enumerates names from the real `model.safetensors.index.json` without
reading tensor data. The `gpt_oss.rs:1046-1163` in-module test is the template for building
a tiny model in memory on `Device::Cpu`.

### B3. Forward

Cache types from `layer_types` per `gpt_oss.rs:851-860`; two masks and per layer selection
per `gpt_oss.rs:887-938`. The sandwich chain:

```
dhn    = input_layernorm(xs)                     plain rms norm
attn   = self_attn(dhn)                          rope iff sliding
attn_n = post_attn_norm(attn)                    plain rms norm
(dhn2, res) = post_attention_layernorm.forward_add_rms_norm(attn_n, res)
moe    = moe(dhn2)
moe_n  = post_ffn_norm(moe)                      plain rms norm
res    = res + moe_n
```

This threads an explicit residual through `DecoderLayer::forward`, which no existing model
does (they keep it implicitly in `xs`). That signature belongs to the new model only; the
final `model.norm.forward_add_rms_norm(&final, &res)` consumes the residual at the end.

Router:

```rust
let logits = self.gate.forward(&xs_flat)?.to_dtype(DType::F32)?;
let topk = ops::moe_router_topk(&logits, MoeRouterTopKConfig {
    top_k: cfg.num_experts_per_tok,
    score_function: Raw,
    selected_weight: Sigmoid,
    renormalize: false,
    output_scale: 1.0,
    norm_min: 1e-20,
    logit_clip: None,
}, Some(&self.expert_bias), None)?;
```

Getting `Raw` versus `Sigmoid` backwards here is silent, so unit test the router against
hand computed values *before* wiring it into the layer.

`supports_paged_attention` returns `false` for this loader, since A4 is not done.

### B4. Serving surface (optional, non blocking)

The tokenizer pre-tokenizer regex writes `\p{N}{1}` where this repo's recognized constant
writes `\p{N}`. `{1}` is the identity quantifier so the semantics are identical, but this
repo has its own pattern recognition and the sibling repo's fix does not transfer. Plus a
chat template and reasoning/tool parsers. None of this should block a working forward.

## 6. Sequencing

| Order | Item | Independent PR | Gate | State |
|---|---|---|---|---|
| 1 | **A0-gemv bandwidth-bound decode GEMV** | yes | decode exceeds single-thread bandwidth, scales to plateau, bit-identical | **DONE for bf16 dense** (8.2x at 4B, 7x at 9B); GGUF quantized and packed MoE paths still open |
| 3 | **A5 CPU stacked-MoE gather-GEMV parallel** | yes | MoE gather uses >1 core; bit-identical partitioning | **DONE** (bf16 42x, f32 6.9x on the isolated gather; `mistralrs-quant/src/moe_gather_cpu.rs`) |
| 2 | A0-gemv.4 GGUF quantized dequant + cache | yes | 0.8B Q4_K_M decode moves; bit identical to uncached | next |
| 4 | A2 sliding window in kernels | yes | golden output per SWA architecture | |
| 5 | A3 NUMA placement | yes | bit identical output, benched at 32/64/128 threads | |
| 6 | **B1 registry** | yes | `make docs-check` green | **DONE** |
| 7 | **B2 config and loader** | yes | synthetic + manifest tests, bf16 scale-ignoring test | **DONE** |
| 8 | **B3 forward** | yes | unit tests, then end to end greedy | **DONE**, unit tests green; golden run against the reference fingerprints in progress |
| 9 | A1 fp8 dequant cache | yes, follow-up | bit identical to uncached, eviction test, bench | |
| 10 | B4 tokenizer and template | yes, non blocking | encode matches HF ids | |
| - | aarch64 CI lane | propose separately | compiles on aarch64 linux | |
| - | A4 CPU paged attention | separate project | serving benchmarks |

### B-state detail

B1-B3 landed as three commits on top of `5ed83cf4e`:

- `feat(core): add Kolibri-1 (kolibri1) text model` - the model, the loader, the registry
  entries and the `kolibri1` GGUF architecture.
- `fix(core): Kolibri-1 RNoPE layers need a contiguous q/k before qk-norm` - the transposed
  q/k are non-contiguous and `qk_rms_norm_rope` copies them before the norm; the norm-only
  RNoPE branch did not, so every fifth layer failed. Includes a scale sweep so the dimensions
  only the real checkpoint exercises fail in seconds rather than after an eight-minute load.
- `perf(mistralrs-quant): parallel CPU gather-GEMV for stacked MoE experts` - A5.
- `fix(core): Kolibri-1 residual is the pre-norm stream, and CPU FP8 MoE now runs` - see below.

### The golden validation

`kolibri1_matches_reference_goldens` (ignored by default) teacher-forces each of the 8
reference prompts and compares the final-position top-8 logits against `vllm.cpp
tests/vllm/models/kolibri1_goldens.json`. That file is the only oracle for this model:
`transformers` has no `Kolibri1ForCausalLM`, and the plugin imports vLLM throughout so it
cannot run under a plain torch stack.

It earned its keep immediately. Two defects that unit tests and a synthetic 4 KB checkpoint
both passed straight through:

**The residual.** The reference's norm-with-residual is the pre-norm contract:
`residual += block_output`, and the norm reads that sum. An earlier revision had it as the
post-norm contract - the residual accumulating the *normalized* block output. That runs
without error, which is what makes it dangerous: against the goldens it gave argmax 0/8 and a
hidden-state RMS walking from 0.12 at layer 0 to 10.5 at layer 49 with outliers past 1296.
With the pre-norm stream, 6 of the 8 prompts reproduce the reference's top-1 token as the
global argmax, against a worst top-k logit gap of 6.8. The trace that
localized it was one `eprintln!` of the per-layer hidden RMS, which is worth keeping in
reserve for the next model.

**CPU FP8 MoE.** Three independent hard failures, none of them Kolibri-1 specific; see
`todo/mistralrs/06-cpu-fp8-stacked-moe-does-not-run.md`. The dequant kernel rejects the
rank-3 expert stack, the FP8 gather implements only the flattened CUDA layout, and the
non-LoRA stacked forward under-reshapes the down projection. A fourth issue is in the
checkpoint rather than the code: the shipped fp8 `config.json` lists only the per-layer
routers in `modules_to_not_convert`, though `lm_head`, `model.embed_tokens` and `model.norm`
also ship without a scale.

### What the goldens do not cover yet

The goldens were produced from the bf16 checkpoint, and **bf16 does not fit on this host.** The
checkpoint is 149 GB and `Tensor::stack` copies every expert slab, so the stacked weights add
~151 GB of resident copies on top; the bf16 run peaks past 255 GB and is OOM-killed. The fp8
checkpoint (74 GB) loads in 33 s and runs, so that is what the comparison above uses, at a
tolerance that reflects 3-mantissa-bit weights rather than bf16. A bf16 run needs either a
host with ~300 GB or an expert path that aliases the mmap instead of copying it, which is the
same work as `todo/mistralrs/02`. |

The ordering is forced by measurement. A0-gemv is first because section 8 showed decode reaches
only 10.1 GB/s on a node that streams at 122.7 GB/s, and 10.1 GB/s is what one thread gets on
its own. That single defect explains the thread cliff, the 9B being thread-insensitive, and the
MoE models being unusable, so it is the one thing that has to move first. A2 and A3 follow
because they are cheap and measurable on the same harness. A5 moves after A0-gemv rather than
before it: optimizing MoE dispatch on top of a GEMV that reaches 8% of bandwidth would still
leave the 35B and Kolibri-1 untestable. Nothing about Kolibri1 (rows 5 to 7) is worth
starting until the dense decode path is fast, because Kolibri-1 inherits every one of these
problems and its 78 GB weights make them 150x more expensive to hit.

## 7. Risks

- **R1 window semantics.** RESOLVED by the model card: "512 preceding tokens plus the current
  token" is 513 total, left distance 512, matching `sliding_window: 513` and the `window - 1`
  in `attention/mod.rs:70-72`. Keep the >513 token prompt test anyway, since agreeing on the
  number is not the same as agreeing on the mask.
- **R1b fp8 KV cache.** The reference serving config uses `--kv-cache-dtype fp8`; we will
  run bf16 KV on CPU. A documented deviation, not a bug.
- **R2 `rope_parameters`.** The plugin reads `config.rope_parameters["rope_theta"]` through
  the transformers base class; the checkpoint has a flat `rope_theta`. Accept both.
- **R3 no committed reference.** There is no golden token file anywhere. Transformers has no
  native Kolibri1 and the plugin imports vLLM, so the documented adaptation is a pure torch
  transcription over the dequantized bf16 weights. This is slow and it blocks the end to end
  gate, so start it first and in parallel with A3.
- **R4 `head_dtype: float32`.** The card does not list it and the plugin ignores it. Treat as
  a config annotation; resolve against the oracle at gate time, do not silently apply it.
- **R5 tokenizer.** The card says the tokenizer is **UniBPE**, trained with a Unigram
  objective but decoding as greedy bottom-up BPE, tuned for German compound morphology
  (4.7 bytes/token DE, 4.2 EN) over a 128,000 vocab with 20 added specials. So the *merge*
  order is custom. The vllm.cpp R7 finding is about the pre-tokenizer split regex only and
  says nothing about the merges, so do not assume it transfers.
- **R6 A1 blast radius.** `BlockwiseFP8Linear` serves every fp8 model. A cache bug is a
  correctness bug for all of them, not a perf regression.
- **R9 workspace clippy debt.** `cargo clippy --workspace --tests --examples -- -D warnings`
  reports 16 pre-existing `must_use` errors (`mistralrs-core/src/pipeline/mod.rs:1748`,
  `mistralrs-mcp/src/client.rs:17`, `mistralrs-mcp/src/transport.rs:12`, and others). These
  are on pristine `master`, not introduced here, but they mean the CI clippy gate cannot be
  expected to pass until they are fixed. `mistralrs-quant` on its own is clean.
- **R7 scales must stay f32.** The oracle explicitly refuses any power-of-two scale path for
  this model, because Kolibri-1's fp32 block scales are not representable as E8M0. Any
  fp8 KV or fp8 scale optimization for this model must not round scales to powers of two.
- **R8 tiny reference checkpoint.** The plugin ships a synthetic tiny checkpoint layout
  (`tests/checkpoints.py`: hidden 256, 6 layers, 8 experts, head_dim 32, sliding_window 65,
  4 sliding + 2 full layer_types). That is an ideal hermetic fixture for our B2 loader test
  and mirrors what vllm.cpp did. It does NOT include fp8 scales, so it only covers the
  unquantized bf16 loader path; the fp8 path needs our own scale tensors.

## 8. Checkpoints on this host (measured 2026-10-11)

Both formats are already downloaded, and this decides most of the plan.

| Path | Shards | Actual bytes | `metadata.total_size` | Weight dtype |
|---|---|---|---|---|
| `/mnt/models/Aleph-Alpha/Kolibri-1` | 32 | 78.8 GB | 78,827,029,120 | `F8_E4M3` + f32 scales |
| `/mnt/models/Aleph-Alpha/Kolibri-1-bf16` | 32 | 156.2 GB | 78,827,029,120 (stale) | `BF16` + f32 scales |

Both configs declare `Kolibri1ForCausalLM`, `model_type kolibri1`, 50 layers, 384 experts,
`moe_intermediate_size` 512, `shared_expert_intermediate_size` 512, `head_dtype float32`,
`tie_word_embeddings false`. The bf16 config has no `quantization_config`; the fp8 one has
`weight_block_size [128,128]`, dynamic activation, and `mlp.gate` in
`modules_to_not_convert`.

### The bf16 checkpoint is the exact dequant of the fp8 one

Spot checked `model.layers.0.mlp.shared_experts.gate_proj`, 10240 elements across 4 rows
spanning all four 128 row blocks: `dequant(fp8) * scale_inv` matches the bf16 weight to
within bf16 rounding on 100% of sampled elements, worst relative deviation 3.3e-3 (one bf16
ulp). So the bf16 tree is the same model, not a separately trained one.

This is worth stating plainly because it gives the port an oracle: the bf16 checkpoint is
what the fp8 path is *supposed* to produce, so a dequant bug shows up as a tensor diff
rather than as drifting logits.

### Two traps in the bf16 checkpoint

- **Stale index.** `metadata.total_size` says 78.8 GB but the files are 156.2 GB. Nothing in
  the loading path reads it (`total_size` appears only in a test fixture at
  `pipeline/paths.rs:785` and in an unrelated tuning context), so this is cosmetic, but do
  not trust it when sizing anything.
- **Vestigial scales.** Every quantized weight still ships its `weight_scale_inv` sibling
  (57950 of them), and `config.json` has no `quantization_config`. So the loader must ignore
  them. Worth an explicit test, because a loader that treats "weight plus scale sibling"
  as a signal to quantize would silently misread this tree.

### Memory, and the decision this forces

Host: aarch64, 128 cores, 255 GB RAM, ~205 GB available, no swap.

```
bf16 resident weights          156.2 GB
fp8 resident weights            78.8 GB
KV cache                        100 KiB/token  (3.4 GB at 32k, 26.8 GB at 262k)
per-token weight bytes touched   7.7 GB  (attention 4.98, routed experts 2.36, shared 0.39)
```

Reading the numbers:

1. **bf16 fits, but only just.** 156.2 GB resident leaves roughly 50 GB for KV and
   activations at 32k context. It works. It is not comfortable, and there is no swap, so an
   overshoot is an OOM rather than a slowdown.
2. **Decode is memory bound and attention dominates.** 4.98 GB of the 7.7 GB touched per
   token is attention projections, which are dense (no MoE sparsity to exploit), against
   2.36 GB of routed experts. So the single highest leverage optimization for this model is
   *not* MoE expert scheduling, it is the dense attention GEMMs and the weights behind them.
   This reorders A3 and A1 relative to what the vllm.cpp row measured on the same model,
   where MoE-shaped work led; on 128 cores with a dense-dominated decode, GEMM bandwidth and
   NUMA placement matter more.
3. **bf16 is the right first target.** It avoids the fp8 dequant path entirely, which means
   A1 is no longer on the critical path for getting Kolibri1 running, and the bf16 tree works
   as the correctness oracle for the forward itself.

Revised scope: **A1 (fp8 dequant cache) is demoted to follow-up, not a prerequisite.** It is
still the right long term work since the fp8 checkpoint halves resident memory and is what
you would actually deploy, but it is no longer what unblocks the model.

### Measured baseline (2026-10-11, this host)

Built `mistralrs-cli` at `3f2515e9b` and benchmarked CPU inference on
`/mnt/models/unsloth-qwen3.5-0.8B-gguf` (Qwen3.5-0.8B Q4_K_M, 508 MB, 24 layers, hybrid
GDN + full attention). Command shape:

```
mistralrs bench -m <dir> -f <file.gguf> --format gguf --cpu \
  --prompt-len 512 --depth 512 --gen-len 32 --iterations 3 --warmup 1
```

CPU inference works end to end on this box. The engine logs
`Using f16 KV cache on CPU` and places all 24 layers on `cpu`. Prefill is healthy. Decode is
not.

Host: Ampere-1a, 128 cores, 4 NUMA nodes x 32 cores. **The final numbers below are pinned to
NUMA node 3 alone** (`numactl --cpunodebind=3 --interleave=all`, cpus 96-127) because nodes
0-2 are shared with other users. Earlier whole-host sweeps are contaminated and are used only
for direction.

Decode thread scaling, node 3 only (32 cores available):

| threads | decode tok/s | TPOT |
|---|---|---|
| 4 | 13.5 +/- 0.2 | 74.0 ms |
| 8 | 17.7 +/- 3.3 | 58.5 ms |
| **16** | **19.8 +/- 4.1** | 52.7 ms |
| 24 | 9.8 +/- 1.3 | 103.4 ms |
| 32 | 7.5 +/- 0.1 | 133.5 ms |

Decode peaked at **16 threads on a 32-core node, exactly half the cores**, then fell off a
cliff. The earlier contaminated whole-host sweep showed the same 16-thread peak and the same
collapse past 32, so the shape is real and not contention noise.

This table is the **pre-fix** baseline, recorded before the output-parallel GEMV existed, and
it is what motivated the item. After the fix the same sweep is nearly flat (section 8), which
is why the half-core conclusion drawn from it did not survive.

Root cause is visible in the source: `candle_core::utils::get_num_threads()` returns
`rayon_num_threads()`, which defaults to the core count. That number is used twice
(`mistralrs-core/src/attention/backends/cpu/threading.rs:6` for the flash-attention rayon
pool, `mistralrs-core/src/lib.rs:36` for the tokio engine runtime). At core count the decode
spawns one thread per core for a batch-1 workload of ~24 sequential layers of small GEMVs,
and most threads are burning time at barriers.

Second effect, NUMA placement, is real but second-order and only visible once threads are not
oversubscribed. At 64 threads (contaminated whole-host run, for direction only):

| configuration | decode tok/s |
|---|---|
| default | 6.6 +/- 0.2 |
| `--interleave=all` alone | 6.2 +/- 0.5 |
| `--cpunodebind=0,1,2,3 --interleave=all` | 15.2 +/- 1.2 |

Interleaving alone does nothing; binding threads across nodes together with interleaved pages
is worth 2.3x. Pinning to a single node removes A3 as a concern for single-node serving, but
A3 still matters once more than one node is used.

Read together: the default CPU configuration is roughly 2 to 3x off the best measured
configuration on one node and worse across all four, and the dominant cause is thread count,
not memory placement.

Caveats: single model, GGUF Q4_K_M, batch 1, depth 512, small enough (508 MB) to fit in caches,
so the NUMA effect is a lower bound relative to a 78 GB model. `RAYON_NUM_THREADS` was used to
sweep because there is no mistral.rs flag for it.

#### The core CPU problem: decode uses ~8% of memory bandwidth

This is the finding that reframes everything, and it is what makes "speed up everything first"
the right call before touching the MoE path.

Measured on node 3 (32 dedicated cores), streaming read of an 8 GB working set:

| threads | aggregate read bandwidth |
|---|---|
| 1 | 10.1 GB/s |
| 4 | 38.5 GB/s |
| 8 | 37.6 GB/s |
| 16 | 76.5 GB/s |
| 32 | **122.7 GB/s** |

The node can stream at 122.7 GB/s. The 0.8B decode is memory-bound (it must read the whole
508 MB weight tensor every token), so its achievable throughput is just `bytes / bandwidth`.
What it actually achieves:

| threads | TPOT | achieved | per core |
|---|---|---|---|
| 4 | 74.0 ms | 7.2 GB/s | 1.80 GB/s |
| 8 | 58.5 ms | 9.1 GB/s | 1.14 GB/s |
| 16 | 52.7 ms | **10.1 GB/s** | 0.63 GB/s |
| 24 | 103.4 ms | 5.2 GB/s | 0.21 GB/s |
| 32 | 133.5 ms | 4.0 GB/s | 0.12 GB/s |

The best decode throughput on this box is **10.1 GB/s, which is exactly what one thread gets
on its own.** Parallelism is contributing nothing; past 16 threads it is actively harmful. The
model is leaving roughly 12x on the table, and no thread-count policy, NUMA tweak, or dequant
cache fixes that on its own.

So the actual bottleneck is the CPU linear layer itself: whatever runs the quantized
`[1, K] x [K, N]` GEMV per decode step is neither bandwidth-saturating nor scaling. That
single fact is why the 9B is thread-insensitive and why the MoE models are unusable, and it is
the same defect in every case. It also explains the earlier MoE finding: the aarch64 packed
GEMV in candle `repack.rs` has no parallel constructs, and the dense path appears to be in the
same family of serial, bandwidth-starved kernels.

Implication for sequencing: **fix the CPU GEMV before the MoE path.** A 12x dense win
compounds through the 384-expert path (and through the dense attention projections that
dominate Kolibri-1's decode). Conversely, optimising MoE dispatch on top of a GEMV that only
reaches 8% of bandwidth is optimizing the wrong thing, and would still leave the 35B and
Kolibri-1 untestable in reasonable time.

#### Isolated GEMV probe

`mistralrs-quant/examples/cpu_decode_gemv.rs` isolates the decode linear (`[1,K] x [K,N]`) from
the full model. On node 3:

| threads | f32 GEMV | bf16 GEMV | plain weight read |
|---|---|---|---|
| 1 | 16.1 GB/s | 2.8 GB/s | 13.0 GB/s |
| 2 | 13.3 | 2.7 | 13.5 |
| 4 | 12.7 | 2.8 | 12.8 |
| 8 | 13.1 | 2.8 | 13.4 |
| 16 | 12.6 | 2.8 | 13.3 |
| 32 | 13.3 | 2.8 | 13.2 |

The decode GEMV **does not scale with thread count at all**: it sits at ~13 GB/s from 1 thread
to 32, and a plain `sum` over the same weight tensor reads at the same ~13 GB/s. Both are far
below the 122.7 GB/s the node can stream. So candle's CPU path runs this batch-1 GEMV at
roughly one core's bandwidth and gets no benefit from the other 31, and the bf16 GEMV is
~5x worse still. This is the root cause, isolated from the model.

This is the same conclusion the vllm.cpp project reached from the other side (their
`gguf-cpu-threadpool.md`: "our CPU path is single-threaded scalar loops", closed by a chunked
output-parallel GEMM for ~8-13x, bit-identical by partitioning outputs, not reductions).

#### What landed: output-parallel CPU GEMV

`mistralrs-quant/src/gemv_cpu.rs` plus `cpu_threads.rs`. Decode is `[1,K] x [K,N]` and must
stream the whole weight once, so it is bandwidth-bound and the only lever is how many cores
stream at once. The new path partitions `N` across the rayon pool, one output row per thread,
each walking `k` in serial order, so no cross-thread reduction is introduced and results are
bit-identical to the serial loop (asserted exactly in
`gemv_cpu::tests::parallel_matches_serial_bit_exactly`). `MISTRALRS_CPU_THREADS` then
`RAYON_NUM_THREADS` then candle's core count sizes the pool (`cpu_threads::resolve`).

Isolated probe, f32 `[1,2048]x[2048,5632]` on node 3, GB/s:

| threads | candle dense | output-parallel GEMV |
|---|---|---|
| 1 | 16.9 | 2.7 |
| 2 | 17.4 | 5.2 |
| 4 | 18.3 | 10.1 |
| 8 | 18.4 | 18.0 |
| 16 | 15.1 | 30.2 |
| 32 | 16.6 | **50.6** |

The new path scales to 50.6 GB/s where candle stays flat at ~17 GB/s regardless of threads.
That confirms the diagnosis: candle's CPU GEMV was single-threaded in practice, and partitioning
the output is what reaches aggregate bandwidth. The isolated bf16 shape is slower still
(8.9 GB/s at 32 threads) because the scalar inner loop is the limit there, not bandwidth; that
is the obvious next step and is not yet done.

End-to-end on a bf16 safetensors model (Qwen3-4B, which is the `UnquantLinear` path Kolibri-1
will take), node 3, 32 threads:

| build | decode tok/s | TPOT | TTFT tok/s |
|---|---|---|---|
| parent commit | 0.6 +/- 0.0 | 1662 ms | 59.8 |
| with GEMV | 1.9 +/- 0.1 | 526 ms | 61.3 |

**Decode is 3.2x faster end to end** on the bf16 dense path, with prefill unchanged (+2.5%, at
the edge of noise). This is the path that matters for Kolibri-1 (safetensors, bf16 or fp8
dequantized to a dense linear), and it is the first real movement on the decode axis.

Two things this does NOT yet cover, both measured not assumed:

- The GGUF quantized path. Q4_K_M still routes through candle's `QMatMul` and does not hit
  this code, which is why the 0.8B Q4_K_M number did not move. Extending the win to GGUF means
  a cached dequant feeding this GEMV, which is A1.
- The scalar `dot` inner loop, which caps the bf16 shape at ~9 GB/s. Vectorizing it is the
  next lever inside this item.

#### Thread policy: what the measurements actually support

Qwen3-4B bf16, node 3 (`numactl --cpunodebind=3`), depth 512, decode tok/s:

| configuration | decode tok/s |
|---|---|
| master | 0.6 |
| this branch, default (no env) | 2.6 |
| this branch, `RAYON_NUM_THREADS=16` | **6.5** |
| this branch, `RAYON_NUM_THREADS=32` | 4.2 |
| this branch, `RAYON_NUM_THREADS=64` | 5.3 |

An earlier revision of this plan divided the core count by two, and that turned out to be
wrong twice over. It was measured *worse*: sizing mistral.rs's own pools to 16 while candle's
global Rayon pool stayed at 32 is an inconsistent configuration, and the mismatch cost more
than the ratio won (2.2 against 4.2 for a consistent 32). And "half the cores" is the wrong
shape on a large host regardless, because it keeps shrinking as cores grow, long past the
point where a batch-1 layer-serial decode can absorb more parallelism. The default now simply
delegates to candle's core count, and `MISTRALRS_CPU_THREADS` is a plain override.

Separately, leaving `RAYON_NUM_THREADS` **unset** under `numactl` is a distinct slow path
that predates this work: with every pool reporting 32, decode occupies 15.1 of 32 cores when
the variable is unset against 21.8 when it is set. The cause is in candle's pool sizing
(`default_num_threads` reads `/sys/.../cpu_capacity`, which ignores a `numactl` affinity
mask), not in this branch. It is left as a documented operational note rather than patched,
because fixing candle's global pool is outside this change; `RAYON_NUM_THREADS` is the lever
that resolves it in practice.

Worth noting that the end-to-end thread curve is now nearly flat across a wide range (6.5 at
16 threads, 5.3 at 64), whereas before the GEMV fix it fell off a cliff past half the cores.
That flatness is the actual deliverable; the specific default is a much smaller effect on top
of it.

#### Second lever inside A0-gemv: NEON dot

With the outputs partitioned, the scalar inner loop became the limit (bf16 shape capped near
9 GB/s). Adding a four-accumulator NEON `vfmaq_f32` dot, same construction as the CPU
attention backend already uses (`attention/backends/cpu/neon.rs`), keeps the reduction order
fixed per output element so the bit-exactness property is preserved (the serial-vs-parallel
test still passes).

Isolated probe on node 3, GB/s, output-parallel GEMV before and after the NEON dot:

| threads | f32 scalar | f32 NEON | bf16 scalar | bf16 NEON |
|---|---|---|---|---|
| 8 | 18.0 | 100.1 | 9.4 | 45.8 |
| 16 | 30.2 | 114.6 | 17.1 | 74.7 |
| 32 | 50.6 | **144.4** | 8.9 | **115.8** |

End to end on Qwen3-4B bf16 (node 3, depth 512):

| build | threads | decode tok/s | TPOT |
|---|---|---|---|
| parent commit | 32 | 0.6 | 1662 ms |
| partitioned GEMV, scalar dot | 32 | 1.9 | 526 ms |
| partitioned GEMV, NEON dot | 16 | **4.9** | 207 ms |
| partitioned GEMV, NEON dot | 32 | 4.1 | 246 ms |

**Decode is 8.2x faster than the baseline** (0.6 to 4.9 tok/s), and the optimum has moved to
16 threads now that the kernel is bandwidth-bound rather than barrier-bound. Prefill unchanged.

Generalizes across scale. Qwen3.5-9B bf16 (32 layers, hybrid GDN + full attention), node 3,
depth 512:

| build | threads | decode tok/s | TPOT | TTFT tok/s |
|---|---|---|---|---|
| before | 128 (default) | 0.3 +/- 0.0 | 2989 ms | 30.1 |
| after | 32 | 2.1 +/- 0.0 | 478 ms | 27.8 |
| after | 16 | 2.1 +/- 0.1 | 483 ms | 20.1 |

7x on decode at 9B, which is the shape Kolibri-1 has. Note both isolated GEMV shapes now
exceed 122.7 GB/s, the figure the earlier numpy probe reported as the node ceiling. That probe
was a conservative lower bound (python threads over 32 separate arrays), so the real ceiling is
higher and these numbers are not contradicting it.

### Second and third models: the picture gets much worse

Two more checkpoints, same harness.

**Qwen3.5-9B** (safetensors bf16, 32 text layers, hidden 4096, GDN + full attention interval
4, vision depth 27; `Qwen3_5ForConditionalGeneration`):

| threads | decode tok/s | TTFT tok/s |
|---|---|---|
| 16 | 0.3 +/- 0.0 | 22.7 +/- 2.7 |
| 128 | 0.3 +/- 0.0 | 30.1 +/- 6.6 |

Decode is completely **thread-insensitive** at 0.3 tok/s, about 100x slower than the 0.8B.
Prefill does scale (22.7 to 30.1). A 9B bf16 model reading ~18 GB of weights per token cannot
sustain better than ~0.3 tok/s here, which implies roughly 5 GB/s of effective read
bandwidth on a single-socket Ampere host that should do several hundred.

**Qwen3.6-35B-A3B** (GGUF Q4_K_M, MoE): did not complete. The first configuration ran over
30 minutes without finishing, so it is reported as a diagnostic, not a throughput number.
Sampling `/proc` during that run:

```
178 threads alive, 177 sleeping, 1 running
1.91 of 128 cores busy
```

Decode is **effectively single-threaded**.

### Root cause of the MoE result

Traced to two places.

1. `mistralrs-quant/src/gguf/cpu.rs:74-75` is the fallback when the packed path does not
   apply, and it is exactly the failure mode A1 predicts for fp8:

   ```rust
   // Dequantize all weights to f32
   let weights = qtensor.dequantize(device)?;
   ```

   That dequantizes **every expert** to f32 on **every forward**, then calls
   `UnquantLinear::gather_forward`.

2. The fast path that is supposed to avoid this, `QTensor::indexed_gemv`, exists for aarch64
   CPU, but its implementation in candle `quantized/repack.rs` contains **no parallel
   constructs at all**: no rayon, no `par_iter`, no scoped threads. Grepping that file for
   `par_iter|rayon|into_par|spawn` returns nothing. So even when the packed path is taken, it
   runs serially.

Together these explain a MoE decode that cannot use the machine regardless of thread count,
and they mean A1 as originally scoped (an fp8 dequant cache) is not the whole story: the GGUF
CPU MoE path needs the same treatment, and the aarch64 packed GEMV needs parallelizing.

This also means the A0 thread-count finding is necessary but not sufficient. On the 0.8B,
thread policy is worth 7x. On the 9B and 35B, no thread policy helps, because the work is
serial to begin with.

## 9. Open questions

1. ~~Does a bf16 checkpoint exist?~~ Resolved: yes, on disk, 156.2 GB, verified to be the
   exact dequant of the fp8 tree.
2. ~~Is there enough RAM to hold the dequantized model?~~ Resolved: yes for the bf16 tree as
   shipped (156.2 GB of 255 GB), tight at 32k context, no swap.
3. What does `moe/experts/backends.rs` fall back to on CPU today? The routed expert
   gather/scatter is 2.36 GB of the 7.7 GB per token and I have not measured it.
4. Target context length for the first run? 32k costs 3.4 GB of KV, 262k costs 26.8 GB,
   which on top of 156.2 GB of weights does not leave headroom.
5. Is the chat template and parser work (B4) in scope for this effort?
## 10. Kolibri-1 measured

Replaces the projection this plan used to carry. Two of the projection's assumptions turned
out to be wrong: the model could not be loaded until L1 landed, and the CPU decode GEMV could
not read a bf16 weight at all.

### What a decode step actually moves

The first version of this arithmetic omitted the attention projections and came out at
4.37 GB/token, which pointed at the routed-expert gather as the thing to fix. That was wrong:

| component | bytes/token | share |
| --- | --- | --- |
| attention (q, k, v, o) | 3.41 GB | 50.0% |
| routed experts (6 of 384, x 50 layers) | 2.36 GB | 34.6% |
| lm_head | 0.66 GB | 9.6% |
| shared expert | 0.39 GB | 5.8% |
| **total** | **6.82 GB** | |

Attention is half the traffic and the routed gather is a third. Optimising the gather alone
caps out at a third of the problem.

### Measured

`kolibri1_decode_benchmark`, bf16 (149 GB, 32 shards), node 3, batch 1, 40 decode tokens.
Every row below the first was taken with memory-local placement
(`--cpunodebind=3 --interleave=3`) at 32 threads, which the next subsection retracts:

| | tok/s | TPOT |
| --- | --- | --- |
| when this section was first written | 1.73 | 577 ms |
| after broadcasting activations by addressing | 1.92 | 519 ms |
| after sending 512-output projections to the CPU GEMV | 4.32 | 231 ms |
| after fusing q/k/v | 4.64 | 216 ms |
| after moving the decode kernels onto the barrier pool | **5.80-5.94** | **168-172 ms** |
| re-measured with headroom: `--interleave=all`, 16 threads | **7.04-7.18** | **137-141 ms** |

(at 32 threads; a 25-token run on a later build measured 5.01 tok/s, so these are +-10% apart
run to run - see the note at the end.)

**2.7x** in total, and the middle step is the large one: a GQA k or v projection is exactly
512 outputs, which sat just under `MIN_PARALLEL_OUTPUTS` and therefore fell through to candle's
generic GEMM. That path packs both operands into scratch and builds a `BarrierPool` per call,
and it was 8.6% of decode for 3.8% of the bytes.

At 4.64 tok/s a step moves 6.82 GB in 216 ms, which is **31.5 GB/s** against a measured ceiling
of 98 GB/s. Roughly a third of the way to the memory system's limit, up from an eighth.

Thread scaling at this baseline, 20 tokens:

| threads | 4 | 8 | 16 | 32 |
| --- | --- | --- | --- | --- |
| tok/s | 2.60 | 3.02 | 3.74 | **4.59** |

Still only 1.77x across 4 to 32 threads, and still short of the ceiling, so the scaling
question is open rather than answered.

### NUMA placement and thread count: both answers here were confounded

Every measurement above was taken while node 3 had **1.08 GB free of 67 GB**, with 58.7 GB of
that resident as file pages. This is a shared box. Re-measured with node 3 at 15.8 GB free and a
load average of 1.46, two answers reversed:

| configuration | tok/s |
| --- | --- |
| `--cpunodebind=3 --interleave=all`, 16 threads | **7.04 / 7.18** |
| `--cpunodebind=3 --interleave=all`, 32 threads | 6.33 |
| `--cpunodebind=3 --interleave=3`, 32 threads | 5.83 |

The plan had recorded that memory-local placement beat interleaving by 23%, and that 32 threads
beat 16 by 30%. **Both were wrong, and wrong the same way**: under page-cache thrashing the thing
being measured was reclaim, not bandwidth.

| threads | 8 | 12 | **16** | 24 | 32 |
| --- | --- | --- | --- | --- | --- |
| tok/s | 4.62 | 6.08 | **7.04 / 7.18** | 6.65 | 6.33 |

`--interleave=all` beats `--interleave=3` by 12%, and the isolated gather explains it: the same
benchmark reads **175.5 GB/s interleaved across all four nodes against 68.3 GB/s local**. The
threads are pinned to one node, so local placement confines them to that one node's DRAM while
interleaving lets all four serve them. The pure-streaming bench agrees that fewer threads are
enough once bandwidth is available - 71.6 GB/s at 16 against 64.8 at 32 - so 16 threads cover it
and the other 16 only add synchronisation.

**Recommendation: `--cpunodebind=3 --interleave=all` with `RAYON_NUM_THREADS=16`.** That is
+23% over the configuration this plan had been recommending, with no code change at all.

This cuts against the operational constraint the work was set under, which was to stay on node
3. It does not need more *cores*; it needs those cores allowed to draw memory from the other
three nodes, which is a placement decision rather than a capacity one.

The general lesson is the one already recorded about micro-benchmarks, and it is worth restating
because this time it was the headline configuration rather than a kernel: on a shared box,
measure with the memory state you intend to run in, and re-measure anything load-bearing when
that state changes.

### Profile at the corrected configuration

Decode-only, `perf record -F 997 -g --delay 140000`, `--interleave=all`, 16 threads, 250
tokens, at 7.0-7.2 tok/s:

| % | symbol |
| --- | --- |
| 53.42 | barrier-pool trampoline, attention path |
| 18.29 | `gemv_cpu::par_chunks::<moe_gather_cpu::par_fill>` |
| 15.27 | barrier-pool trampoline, decode GEMV path |
| 5.43 | `gemv_cpu::Weight::dot` |
| 1.05 | `gemm_f32::microkernel::neon::f32::x4x4` |
| 0.94 | `crossbeam_epoch` |
| 0.74 | `BarrierPool::execute_chunked` |
| 0.61 | `ReduceSum` |
| 0.61 | `single_q::compute_group_range` |
| 0.58 | `gemm_f32::gemm::gemm_basic` |
| 0.53 | `Stealer::steal` |

**Synchronisation is now ~1.5%**, down from 28% at the start of this line of work and 13% at
the previous profile. `crossbeam_epoch` fell from 9.98% to 0.94% and `Stealer::steal` from
3.39% to 0.53%, which is what halving the thread count against four nodes' worth of bandwidth
does to the coordination cost. About 95% of a decode step is now kernel work.

That closes the thread-synchronisation question and moves the remaining cost inside the
kernels. At 139 ms/token the step moves 6.82 GB at **49 GB/s aggregate**, against 71.6 GB/s
from the streaming bench at the same thread count and 175 GB/s from the isolated gather under
the same interleaving. So the kernels are not saturating the memory system, and the gap is
now per-thread streaming behaviour rather than anything structural - slab-to-slab transitions
in the gather, and the attention kernels, which are half the bytes and untouched.

### Per-phase decode timing, and the router

`perf` cannot read a CPU decode step on this host. The kernels run inside the barrier pool and
their frames inline into `call_trampoline`, so the interesting part of a step collapses into its
caller. `mistralrs-core/src/utils/decode_prof.rs` splits a step into nine named phases instead.
That is what makes an A/B readable - a change to one phase moves that phase's line rather than
being lost in a 15% band around the total. Instrumentation overhead is 0.7 ms on a 136 ms step
and is reported, not assumed.

Its first output was wrong in the direction that flatters you: 19 ms *more* attributed time than
wall clock, because two guards sat in one scope and the second shadowed the first without
dropping it, so o_proj was counted twice. Only the unattributed column gave it away.

At 7.34 tok/s, `--interleave=all`, 16 threads:

| phase | ms/step | share | GB/s |
| --- | --- | --- | --- |
| routed | 50.75 | 37.5% | 46 |
| attn_proj | 47.91 | 35.4% | 71 |
| router | 19.59 | 14.5% | **10** |
| shared | 6.25 | 4.6% | 62 |
| lm_head | 4.61 | 3.4% | 143 |
| attention | 3.21 | **2.4%** | 47 |
| norms | 3.04 | 2.2% | - |

**Two earlier conclusions died here.** The attention *op* is 2.4%, not the 50% of bytes that
pointed at it - that 50% was always the attention *projections*, which are dense GEMV this plan
had already optimised twice. Attacking the attention path would have been 2.4% of a step at
best, and it was the largest thing left to attack.

The router was the real find. 10 GB/s against 46-143 for everything else, 14.5% of a decode step
to choose 6 experts out of 384. It is a raw `Tensor::matmul`, so it bypassed every decode fix in
this plan: candle runs a batch-1 matmul as a tiled GEMM at m=1 on the global pool, whose workers
park between calls.

The cause was a threshold counting the wrong quantity. `MIN_PARALLEL_OUTPUTS = 512` counts
outputs, but what the kernel streams is bytes. The router is 384 outputs and so missed it, while
being 3.9 MB of f32 weight - more than the 2.6 MB bf16 k/v projection that set the 512 threshold
in the first place. Added a byte criterion as a second way to qualify, as a strict `||` so it
cannot take any shape away from the fast path. `router` 19.59 -> 3.52 ms, 5.6x.

### The pool is a resource, not a default

The router fix did not deliver its 15.8 ms. A same-session A/B, only the router changed:

| phase | before | after |
| --- | --- | --- |
| router | 19.29 | 3.52 |
| routed | 50.23 | **56.20** |
| attn_proj | 48.29 | 49.58 |
| total | 136.44 ms (7.33 tok/s) | 129.7 ms (7.71 tok/s) |

The router moved from Rayon's pool to the barrier pool and the gather got 6 ms slower for it.
Four combinations, all measured:

| router | gather | tok/s |
| --- | --- | --- |
| candle matmul / rayon | barrier | 7.33 |
| **gemv / barrier** | **barrier** | **7.71** |
| gemv / rayon | barrier | 5.80 |
| gemv / rayon | rayon | 4.71 |

Two pools are worse than either pool alone: with the router on Rayon, `attn_proj` also degrades
from 48.3 to 62.3 ms. One pool for all decode work wins, and the barrier pool is worth ~40% over
Rayon - the same parking result as before, now confirmed a second way.

So the honest total is **+4.5%** (7.33 -> 7.71), not the +13% the phase saving suggested, and the
gap is a real and reproducible cost rather than measurement noise.

### Where the time actually goes

Two corrections first, both from getting the measurement wrong before trusting it.

**The streaming ceiling was wrong.** The first version of the read benchmark accumulated into a
single `acc`, which is a serial dependency chain, so it measured add latency rather than memory:
2.7 GB/s at one thread. With four independent accumulators (`bw2.rs`, which is what makes it a
bandwidth test):

| threads | 1 | 4 | 8 | 16 | 32 |
| --- | --- | --- | --- | --- | --- |
| GB/s | 7.8 | 30.4 | 58.0 | **98.0** | 86.9 |

98 GB/s at 16 threads, and 32 is *worse* than 16. So the ceiling is ~98, not 72.8, and pure
streaming does not want 32 threads on this node.

**The first profile was 97% model load.** `perf record` over the whole benchmark captured a
180 s load and about 6 s of decode, so every percentage in it described loading - the 14% of
`memcpy` was tensor copies at load, and the single hottest symbol was hashbrown dropping a LoRA
registry. Re-profiled with `--delay` to skip the load (`perf3.data`, 40 decode tokens):

| % | symbol | |
| --- | --- | --- |
| 24.9 | `moe_gather_cpu::par_fill` | routed-expert gather |
| 18.0 | `gemv_cpu::Weight::dot` | dense GEMV |
| **23.7** | `crossbeam_epoch` + `Stealer::steal` | **rayon synchronisation** |
| 8.6 | candle gemm, `pack_operands`, `BarrierPool::new` | k/v |
| 5.5 | misc | |

`memcpy` does not appear. Rayon overhead at 23.7% is the largest addressable block.

### The gather kernel is not the problem

Worth recording because the first round of this plan set out to fix it. At Kolibri-1's shapes,
cold, rotating over a pool large enough to defeat the cache (`tests/routed_gather_bench.rs`):

| expert pool | resident | GB/s |
| --- | --- | --- |
| 6 | 16 MB | 129.3 (cache-resident, above DRAM) |
| 48 | 126 MB | 97.8 (at the ceiling) |
| 192 | 503 MB | 88.0 (90% of the 98 GB/s ceiling) |

So at the time, the gather looked like it ran at 90% of what the memory system can deliver, and
prefetching or re-tiling it - the original plan's proposal - would buy nothing.

**That conclusion is unsafe and should not be relied on.** The same benchmark now reads 177.4
GB/s under `--interleave=all` and 68.7 under `--interleave=3`, a 2.6x swing caused only by where
the benchmark's own 500 MB of anonymous memory lands. Node 3 has been between 1 and 60 GB free
across this session depending on what else is running. An isolated number from this box is a
measurement of the box's memory state as much as of the kernel, and only the end-to-end tok/s is
comparable between runs.

What survives is the weaker claim: the gather is not obviously broken, and no change to it has
been shown to help end to end. Whether prefetching it would help is still open.

### What was tried, and what is left

Tried, in the order the profile suggested:

1. **Broadcast activations by addressing.** Both gather layouts were copying every activation
   row `top_k` times. Worth +11%.
2. **The routed-expert gather kernel.** Measured cold at Kolibri-1's shapes over pools large
   enough to defeat the cache: 88-98 GB/s, i.e. 90% of the machine's streaming ceiling.
   **No work needed** - the original plan's premise, that this kernel needed prefetching or
   retiling, was wrong.
3. **`MIN_PARALLEL_OUTPUTS` 1024 to 512.** **2.35x.** The largest single win, from a constant.
4. **Fused q/k/v into one GEMV region.** 1.3x on the operation cold, ~7% end-to-end.

**Fusing gate and up: tried, measured, reverted.** The obvious next cut, since it is the last
big pair of regions the MoE opens. Implemented as one parallel region over both sets of slabs,
held bit-exact against two separate gathers. End-to-end it read **+5%**, and it was reverted,
because the mechanism does not support the number:

- a barrier costs ~10-20 us and a MoE gather covers ~178 us of work at 88 GB/s, so removing
  half the MoE's barriers should be worth about **0.3%**
- the fused form also saves one pass over a 10 KB activation row per projection, which is
  nothing

A 5% reading on a box with ~20% run-to-run variance is not evidence of a 5% effect. The
implementation also had to be put in the branch CPU actually takes - `uses_flattened_gather` is
CUDA/NVFP4 only, so the first attempt was measured having never fired at all. That is the
second time this session that an unexercised path produced a confident number.

What this says about the remaining 27% of rayon synchronisation: it is not concentrated in the
large gathers, because those barriers are cheap against the work they cover. It has to be in the
many small regions, which means reducing the *number* of calls rather than fusing the big ones -
and nothing obvious presents itself there.

**Thread count: 32, not 16.** Worth separating from the streaming benchmark, which says the
opposite:

| threads | 12 | 16 | 24 | 32 |
| --- | --- | --- | --- | --- |
| decode tok/s | 3.65 | 3.88 | 4.20 | **5.01** |

Pure streaming on this node peaks at 16 threads (98.0 GB/s against 86.9 at 32), but that is one
long region. Decode is a few hundred short ones, and there the extra threads hide latency rather
than contend for bandwidth. **Do not take the streaming optimum as the decode optimum** - they
are different regimes on the same machine, which is worth remembering given that this plan
previously concluded the opposite from a dense-GEMV measurement.

At 5.01 tok/s a step moves 6.82 GB in 200 ms, which is **34.1 GB/s** against the 86.9 GB/s this
node streams at 32 threads - 39% of it, up from an eighth when this section was written.

### What the 28% actually was: workers parking between regions

It was never barrier cost. rayon-core 1.13 gives an idle worker
`ROUNDS_UNTIL_SLEEPY = 32` `sched_yield` rounds before sleeping, and
`spin_threads_before_yielding` was removed in that version so the knob cannot be
raised. A batch-1 decode token opens around 400 parallel regions, so the workers
park and get woken ~400 times per token - and the profile shows it as
`crossbeam_epoch` and `cas4_acq` rather than under anything named "overhead".

candle's `barrier_pool` already solves this and was already in the tree: it spins
25,000 before sleeping, and the CPU attention runs on it every layer. Putting the
decode GEMVs and the routed gathers there too keeps the workers warm across the
attention/GEMV alternation:

| | tok/s |
| --- | --- |
| Rayon's global pool | 4.88 |
| barrier pool, GEMVs only | 5.05 |
| barrier pool, GEMVs and gathers | **5.80 / 5.94** |

**+20%.** This is what makes the four negatives below coherent: splitting at 512
outputs pays 2.8x, serializing a seventh of the regions costs 45%, and the
barrier pool's spin limit changes nothing - all true, and none of them were about
the cost of joining. The cost was arriving.

### Four hypotheses about the remaining 28%, none of them

A decode-only profile at the current baseline, 80 tokens, is 56% kernel and 28% rayon
synchronisation, with `crossbeam_epoch` (epoch guards, i.e. rayon's join machinery) and
`Stealer::steal` as the two symbols making up the latter. Four explanations were tested:

| hypothesis | test | result |
| --- | --- | --- |
| regions too small to be worth splitting | parallel vs serial time per output count | wrong: splitting at 512 outputs pays **2.8x**, at 4096 it pays 5.8x. The 512 threshold is well placed |
| barrier spin from three pools on 32 cores | `CANDLE_BARRIER_POOL_SPIN_LIMIT` = default / 0 / 200 | no effect: 4.76 / 4.64 / 4.78 tok/s |
| gate and up should be fused | implemented, bit-exact, measured | +5%, reverted - see above |
| the `Vec<f32>` -> `Tensor` -> `to_dtype` plumbing | full `gemv` vs the bare kernel | 3.4% at n=512, 21% at n=4096. Real, but too small to be 28% |

That is four failures in a row on the same number, which is worth recording as what it is: the
28% is not explained, and the obvious candidates are all wrong. The thread accounting is
genuinely odd - rayon 32, `FLASH_ATTN_POOL` 32, candle's `barrier_pool` `candle_num_threads()
- 1` = 31, so up to 95 threads exist for 32 cores - but shrinking the spin behaviour did not
change the result, so the pools are evidently not contending the way the thread count suggests.

What that leaves: fewer *calls*, not better-tuned ones. Every kernel here is already at 90% of
the streaming ceiling cold, so the remaining work is call count and the machinery around it, and
nothing in the decode path presents an obvious way to cut that without a structural change to
how a layer is executed.

A methodological note, because it cost real time more than that: every isolated benchmark
here initially reported a cache-resident number. The first read-bandwidth test measured add
latency rather than bandwidth (2.7 GB/s at one thread against a real 7.8). The first gather
benchmark reused one set of weights (15.7 MB, L3-resident) and reported 52 GB/s where cold it
is 88. The first fused-qkv benchmark did the same. Rotate enough independent weight sets that
each pass reads what the last did not, or the number is fiction. And match the benchmark's NUMA
placement to the model's, or the number is a measurement of page placement rather than of a
kernel - that one is worth 2.6x here.

Throughput measured on a model that is not yet bit-exact against the goldens
(`todo/mistralrs/07`) should be read with that in mind.


## An explicit weight LRU: measured, and what it is actually worth

Question: the model is 151 GB and the box has 255 GB, so a dedicated LRU of the hot weights was
proposed. Measured on the real routing rather than on the `6 of 384` independence argument,
which turns out to be wrong by a factor of 36.

Per decode step, 50 layers x 6 experts, recording which experts are actually selected:

| | degenerate golden chain | diverse 251-token prompt |
| --- | --- | --- |
| distinct experts touched per step | 212 / 384 (55%) | **211 / 384 (55%)** |
| step-to-step overlap | 65.4% | **57.6%** |
| chance, if draws were independent | 1.6% | 1.6% |
| reuse within 1 step | 94.1% | 83.0% |
| reuse within 5 steps | 98.9% | 96.8% |
| reuse within 10 steps | 99.7% | 98.6% |

**A step touches half the expert pool, not 6 experts.** The `K/E` estimate assumes independent
draws; consecutive hidden states are far more similar than that, so routing barely moves
between steps. The 55%-of-pool figure is the robust one - it is identical on a degenerate
repetition chain and on varied prose - while the overlap drops from 65% to 58% when the text
is real. Any number quoted from the golden chains alone is optimistic; the diverse column is
the one to design against.

An LRU therefore does not need to be large. At 2.36 GB of expert weight per step:

| cache | steps of history | re-selections caught (diverse) |
| --- | --- | --- |
| 4.7 GB | 2 | 83.0% |
| 11.8 GB | 5 | 96.8% |
| 23.6 GB | 10 | 98.6% |

The 4.46 GB/token of *fixed* weights - attention projections, the shared expert, `lm_head` - is
the same memory every step and is held by anything above ~5 GB forever. So **the knee is near
5-10 GB, and a 32 GB cache is roughly 3x over-provisioned**: it buys the last ~2% over 24 GB.

### What it would buy, which is not speed

The kernel's page cache is already an LRU, and on this host it is 67 GB - larger than anything
proposed here. Measured during decode: **61 major faults in 20 s**, i.e. no disk I/O at all. So
an explicit LRU would not make this build faster today; there is nothing un-cached to cache.

What it would buy is independence from the box. At 1.08 GB free on node 3 earlier in this
project - with 58.7 GB of that resident as file pages - the kernel's LRU was evicting our pages
under other tenants, and every number in this plan moved for that reason rather than for any
code change. A resident cache we control removes that variable. That is reproducibility, not
throughput.

There is also nothing to extend: `BoundedCache` is in `cutile/routed_lora.rs` and
`ExpertCudaCache` in `lora/dynamic/`, both CUDA. mistral.rs has no CPU weight cache at all, so
this is new machinery that copies the weights into anonymous memory - which is the L1 decision
this plan already backed away from at 149 GB, at a scale that is now affordable at 10 GB.


## The residency cache, and why 16 threads per node

An LRU of the hot expert weights was built (`mistralrs-quant/src/moe_cache.rs`, byte-budgeted,
default off, `MISTRALRS_MOE_CACHE_MB`). **Measured, it is a 13% loss and there is no disk I/O
for it to prevent.**

| | tok/s | `routed` |
| --- | --- | --- |
| cache off | 7.14 | 62.4 ms |
| cache on, 16 GB | 6.42 | 73.1 ms |

Steady state, not fill cost: 100 steps gives 6.26 and 400 steps gives 6.42, so it does not
amortise. Meanwhile **11 major faults in 25 s of decode**, even with the box busier and the
model's page cache evicted to 48 GB. The 6.82 GB working set is re-read every step and the
kernel keeps it resident whatever else is running, so there is nothing for a private cache to
save. Part of the penalty is placement: the model's file pages are ~97% node-3 local, while
anonymous copies get striped, and `--preferred=3` recovers about 6%.

Sizing was done on hit rate and should not have been presented as a performance result. The
reuse measurements stand - 212 of 384 experts touched per step, 57.6% step-to-step overlap,
99.1% within ten steps - but a high hit rate on a *slower* copy is not a win.

## Thread count: half a node per instance, and why

Every node has 32 cores, and the earlier sweeps used 16 per node throughout. Filling them:

| nodes | threads/node | tok/s |
| --- | --- | --- |
| 1 | **16** | **7.14** |
| 1 | 32 | 6.80 |
| 2 | 16 | 7.33 |
| 2 | 32 | 3.98 |
| 4 | 16 | 5.35 |
| 4 | 32 | 2.55 |

**Doubling threads per node roughly halves throughput.** Same mechanism as the parking fix, one
step further: the barrier pool spins 25,000 times before sleeping, so at 16 threads its 15
workers leave half a node free to absorb the spin, while at 128 threads there are 127 spinners
on 128 cores contending with the work they exist to accelerate. Half a node per instance is the
right setting for that pool, not a coincidence.

Spreading across nodes does not help this model: 2 nodes at 16 threads each is a wash against
one (7.33 vs 7.14, inside noise, with visibly wider variance), and 4 nodes is clearly worse at
either thread count. The earlier whole-host result in this plan that favoured all four nodes
was measured on a 508 MB model that fitted in cache; a 151 GB model moves its *page cache*
across nodes with it, and trades local DRAM for re-faulting.

Re-verified with the box unbusied - all four nodes at 0.1-0.7% busy, which retires the
constraint this work was scoped under (nodes 0-2 shared with other users). The
single-node, 16-thread recommendation survives it.


## The routed phase, and the one hypothesis still standing

`routed` is 44% of a step at ~39 GB/s against a 106 GB/s streaming ceiling. Everything below
is what has been eliminated, and it is most of the list.

- **Not disk.** 61 major faults in 20 s; 11 in 25 s with the box busier and the model's page
  cache evicted to 48 GB. The 6.82 GB working set is re-read every step and the kernel keeps it
  resident.
- **Not the kernel.** Isolated, cold, 503 MB pool rotating: 106.5 GB/s, which *is* the ceiling.
- **Not barrier-pool scheduling.** `execute_chunked` uses a shared atomic cursor with ~8 grabs
  per thread - dynamic, and the pool is 16-way (15 workers plus the caller), so no wave
  imbalance.
- **Not redundant traffic.** The per-expert path reads selected slabs in place; nothing copies
  the weight. It did take 57,600 `RwLock` reads per step to guard memory it never touched, now
  fixed, worth ~2 ms.
- **Not anonymous-versus-file-backed.** The residency cache answers this: it puts the routed
  reads on anonymous copies and `routed` gets *slower*, 62.4 -> 73.1 ms. Both backings are
  ~35-40 GB/s in-model; the isolated bench, which is anonymous, is 106.
- **Not the expert count per step.** ~212 of 384 are touched, but the isolated bench also
  rotates through scattered slabs and still reaches 106.

So the variable is **in-model versus isolated**, not any property of the access pattern. The
one hypothesis left that fits the numbers is page placement:

| | GB/s |
| --- | --- |
| isolated gather, anonymous, `--interleave=all` | 175.5 |
| isolated gather, anonymous, `--interleave=3` | 68.3 |
| in-model, file-backed, pages ~97% on node 3 | 39-42 |

The model's pages are effectively node-local, so the honest ceiling is the **one-node** number,
68 GB/s, not 106. At 39-42 the routed phase is at **57-62% of its real ceiling**, and the gap
is 1.6x rather than 2.7x. Threads are pinned to node 3, so a striped pool would draw from four
nodes' DRAM.

**This could not be tested.** Evicting the model's pages with `POSIX_FADV_DONTNEED` and letting
them re-fault under `--interleave=all` did not stripe them: 20.9 GB on node 0, 0.5 on node 1,
0.3 on node 2, 35.1 on node 3. btrfs file-cache faults do not honour the interleave policy the
way anonymous allocation does, and throughput fell to 6.32 tok/s because only 56 GB of 145 GB
was resident afterwards.

Testing it properly needs the weights in anonymous memory under the interleave policy - which is
the L1 stacking decision, at 151 GB, or a filesystem whose cache faults stripe.

Two measurement notes, because both were my error rather than the machine's:

- A first attempt at the pattern probe summed **one byte per slab** and reported 17,399 GB/s.
- The corrected probe summed every byte and reported 7.3 GB/s, which is numpy's uint8 reduction,
  not memory. Neither was a bandwidth measurement; the real kernel was needed.


## The governor: every number above this line was taken at 29% of peak

`scaling_governor` was `schedutil` (cppc_cpufreq), and it never ramps on this host:

```
scaling_min_freq  1,000,000 kHz
scaling_cur_freq  1,000,000 kHz   <- idle AND under load
scaling_max_freq  3,400,000 kHz
```

Under a 16-thread decode: 1.0 GHz. Under a 64-thread load across all four nodes: 1.0 GHz in 11
of 12 samples, 1.21 GHz once. So **every measurement earlier in this plan - every tok/s, every
GB/s, every "we are bandwidth-bound" conclusion - was taken at 1.0 GHz on a 3.4 GHz part.**

Switching the governor (needs sudo, which this account has passwordless):

| | tok/s | `attn_proj` | `routed` |
| --- | --- | --- | --- |
| 1 node, 16 threads, `schedutil` @ 1.0 GHz | 7.14 | 49.6 ms | 62.4 ms |
| 1 node, 16 threads, `performance` @ 3.4 GHz | **9.56** | **36.6 ms** | **46.8 ms** |

**+34%.** Not 3.4x, which is the interesting part: the kernels were partly memory-bound after
all, so the clock was not the whole story - but it was a third of it.

### Thread scaling got *worse*, which confirms the spin theory

| threads | 16 | 24 | 32 |
| --- | --- | --- | --- |
| tok/s @ 3.4 GHz | **9.56** | 7.29 | 7.16 |
| tok/s @ 1.0 GHz | 7.14 | 6.80 | 6.33 |

The optimum is still 16, but the penalty for oversubscribing is now sharper. That is exactly
what the barrier-pool spin account predicts: the pool spins 25,000 iterations before parking,
and at 3.4 GHz a spinner burns 3.4x more per iteration, so the cost of running 24 or 32
workers on a 16-thread's worth of useful work goes up rather than down.

### What this invalidates

- **"We are memory-bound, so compute optimisation cannot help."** That was inferred from the
  kernel sitting at 94% of single-thread streaming bandwidth - but both were measured at
  1 GHz, where compute is cheap relative to memory and the two are indistinguishable. The BFMMLA
  microbenchmark (1.167x) was also taken at 1 GHz; if anything a compute saving should have
  looked *worse* there, so its value at full clock is likely higher, not lower.
- **Every "measured" GB/s in this plan** should be read as "GB/s at 1.0 GHz".

### Policy for the governor

The 1 GHz cap is intentional and is the box's normal operating state - **`schedutil` stays on**.
`performance` is for top-speed measurement only:

```
sudo sh -c 'for c in /sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_governor; do echo performance > "$c"'
```

Runtime only, does not survive a reboot, and **put it back afterwards**. Record which
governor produced any published number.

The cap is invisible unless you go looking: a 128-core box that never leaves 29% of its clock
silently caps every benchmark taken on it by 3.4x. That is the reason this section exists, not
a request to change the policy.



## Head-to-head against llama.cpp on the same box

llama.cpp is the only other implementation of this model on CPU, and it is a fair yardstick:
stock upstream at `62318cd57` (the Kolibri-1 merge) plus 16 lines that only expose tensors
and change no arithmetic. Both sides run the same bf16 weights of the same checkpoint on node 3
at 3.4 GHz, on a quiet box (load 1.68).

**Decode** (token generation):

| threads | mistral.rs | llama.cpp | ratio |
|---|---|---|---|
| 8 | 6.97 tok/s | 1.54 tok/s | 4.53x |
| 16 | **8.73 tok/s** | 2.97 tok/s | 2.94x |
| 32 | 7.32 tok/s | **5.46 tok/s** | 1.34x |

**Prefill** (512 tokens, both sides):

| | mistral.rs | llama.cpp | ratio |
|---|---|---|---|
| p512 | **21.86 tok/s** | 7.04 tok/s | 3.10x |
| p64 | 12.43 tok/s | - | - |

**The shape of the curves is the interesting part.** llama.cpp scales almost linearly with
threads - 1.54, 2.97, 5.46, about 1.9x per doubling - while mistral.rs peaks at 16 threads and
*regresses* at 32. That regression is the barrier-pool spin problem, and it is now the whole of
our remaining gap: at 32 threads we are handicapped by a cost llama.cpp does not have. Fix the
spin and the ratio at 32 threads would be about 1.8x rather than 1.34x, with the same headroom
to spare.

That makes the spin the highest-value remaining perf work by a wide margin, ahead of the
`MIN_PARALLEL_OUTPUTS` re-sweep.

### Caveats

- Different weight containers: llama.cpp reads a 145.57 GiB bf16 GGUF, mistral.rs reads 149 GB
  of bf16 safetensors. Same model, same weights, not the same file.
- llama-bench's `tg32` starts from an empty prompt; the mistral.rs benchmark prefills 38 tokens
  and then times 120 decode steps. Both are decode-bound, but the KV contexts differ.
- llama.cpp's numbers are from its stock CPU path. The primary oracle this work is measured
  against is vLLM on GPU, so none of this is a statement about that.


## The thread-scaling regression is a spin limit, not a thread count

llama.cpp scales about 1.9x per doubling of threads; we peaked at 16 and regressed at 32. The
cause is candle's barrier pool: a worker spins `DEFAULT_BARRIER_POOL_SPIN_LIMIT = 25,000`
times before parking (`candle-core/src/utils.rs:10`, spin/park branch at `:163-169`), and the
pool is sized `candle_num_threads() - 1` workers. A decode token opens around 400 such regions,
so the spin is paid 400 times a token.

`CANDLE_BARRIER_POOL_SPIN_LIMIT` is an env var, so this is testable without a code change.
Decode mean tok/s, schedutil, node 3:

| spin limit | 16 threads | 32 threads |
|---|---|---|
| 0 | 6.28 | 7.33 |
| 256 | 6.33 | 7.79 |
| **1024** | 6.26 | **8.01** |
| 4096 | - | 6.87 |
| 25000 (default) | **7.66** | 6.88 |

Two findings.

**32 threads was the right thread count with the wrong spin limit.** The peak at 1024 is 8.01,
which beats the best 16-thread configuration (7.66 at the default limit) by 4.4%. The curve at
32 threads is sharp: 8.01 at 1024, 6.87 at 4096.

**The right limit depends on the thread count, and candle's single default is right for neither
of our operating points.** At 16 threads the default 25,000 is much better than 1024 (7.66
against 6.26): few enough workers that the burn is cheap and the fast wake-up dominates. At 32
threads the surplus workers' burn dominates and a short spin wins.

**These are 1.0 GHz numbers and must not be acted on yet.** The burn is clock-dependent: at
3.4 GHz the same spin costs 3.4x more, which moves the optimum down, while the wake-up latency
becomes 3.4x more expensive relative to the work, which moves it up. The 3.4 GHz sweep is
pending.

A correction to an earlier explanation: the surplus workers are not idle. The chunk count is
sized to the thread count (`chunk = n.div_ceil(threads)`), so every worker does work. The cost
is the spin *between* the roughly 400 parallel regions per token, paid by every worker in the
pool each time, and it grows with the number of workers.


## The spin limit at 3.4 GHz, and a three-way comparison

3.4 GHz, node 3, decode mean tok/s, `CANDLE_BARRIER_POOL_SPIN_LIMIT`:

| threads | 0 | 256 | 1024 | 4096 | 25000 (default) |
|---|---|---|---|---|---|
| 16 | 8.45 | - | 8.91 | - | **9.11** |
| 32 | **8.58** | 8.45 | 8.50 | 7.85 | 7.27 |

At 16 threads more spinning is better; at 32 threads less is better, by 18.0% against the
default. The shape is coherent: the surplus waiters cost more than the fast wake-up saves, but
only once there are more of them than there is work for.

**Confirmed in a second, independent codebase.** The vllm.cpp CPU tier, whose pool yields
after a bounded spin rather than parking, shows the same preference at 32 threads: never-yield
wins by 16.4%. Two implementations with different wait policies, same preference, same thread
count.

The 1.0 GHz result had the peak at 1024 for 32 threads; at 3.4 GHz the peak is at 0. The
optimum is clock-dependent, as expected, and the 1.0 GHz table must not be used to pick a value
for a 3.4 GHz deployment.

### Three-way comparison, and why the headline is not yet quotable

| threads | mistral.rs (bf16 safetensors) | llama.cpp (bf16 GGUF) | vllm.cpp (fp8, dequantised in process) |
|---|---|---|---|
| 8 | 6.97 | 1.54 | 1.36 |
| 16 | 8.73-9.11 | 2.97 | 2.56 |
| 32 | 7.27-8.58 | 5.46 | 3.49 (4.07 at spin 0, 4.97 with the dequant cache) |

**This table is not a comparison.** Three different containers - bf16 safetensors, bf16 GGUF,
fp8 dequantised in process - which is different arithmetic and different memory traffic, not just
a different runtime. The spin setting moves each row's optimum, and the vllm.cpp rows are
without its dequant cache, which their own shipping-equivalent number is 4.97 rather than 3.49.
The mistral.rs cells are single runs with no error bars, while the vllm.cpp cells are three runs
each.

A ratio is only meaningful with the same directory, the same spin setting, and repeats on both
sides. The agreed protocol is: both sides on the fp8 checkpoint, sequential rather than
concurrent, three repeats per cell, spin 0 and default at 8/16/32.

The prefill comparison is more surprising than the decode one: vllm.cpp's p512 is 43.6 tok/s at
32 threads, llama.cpp's pp512 is 7.04, and mistral.rs's p512 is 21.86. mistral.rs's decode is
the best of the three and its prefill is second, which is the opposite ordering and worth
looking at.

### Caveats on this section

- The last two 32-thread points (4096 and 25000) were taken while a 31-process test suite ran
  for 3.6 seconds on the box. They are flagged, not discarded, and the two cells are to be
  re-run.
- Every cell is a single run.


## The branch split, and what the entanglement taught us

71 commits on `cpu-decode-bandwidth` are being split into a stack of independently mergeable
branches. Four are built and verified so far:

| branch | commits | what it is |
|---|---|---|
| `pr/01-rmsnorm-scale-f32` | 1 | the RMSNorm scale fix, standalone, 1425 tests pass |
| `pr/02-flash-attention-causality` | 1 | the causality fix, standalone, 1427 tests pass |
| `pr/03-output-parallel-cpu-gemv` | 7 | the output-parallel GEMV, the NEON dot, the f32-weights fix, the pool sizing |
| `pr/04-parallel-expert-gather` | 8 | the per-expert gather, the broadcast-by-addressing, the barrier pool, the q/k/v fusion |

**The split is at the crate boundary, not the commit boundary, and it had to be.** The
Kolibri-1 model work and the CPU gather work are entangled in the original history:
`f31fafb3a` ("stop stacking MoE experts on CPU") adds the per-expert gather path that the
model's FP8 MoE fix needs, and it lands *after* the model commits. Cherry-picking the model
commits first fails on `moe_gather_cpu.rs` and `per_expert.rs` with modify/delete conflicts,
because the files do not exist yet.

So PR 4 takes the `mistralrs-quant` side of the gather work plus the one core commit that
builds without the model, and the core-side wiring lands with the model PR. The alternative -
one large PR - was rejected because a 20-commit PR with no reviewable increments is how this
branch became 71 commits in the first place.

**The plan doc is excluded from every PR branch.** 24 of the 71 commits touch
`kolibri1-support-plan.md`, and it is a local working log rather than upstream documentation,
so it lives on its own branch. Commits that modify it conflict as modify/delete and are
resolved as "deleted by us".

## The spin question is not resolvable, and the number that says so

Two single runs of one cell, same binary, same weights, same cell: **8.45** at 16 threads with
spin 0 from the 1.0 GHz sweep, and **8.11** from the 3.4 GHz grid. **4 percent apart.**

That is the noise floor of a single-run cell, measured directly, and it invalidates the shape
that was read out of the 32-thread row. The three cells that looked like a plateau - 8.58,
8.45, 8.50 - differ by 1.5 percent, and the two that looked like a step down - 7.85 and 7.27 -
are the two a 31-process test suite landed in. So:

- The 8.50 to 7.85 step is 8.5 percent, about twice the noise, in the two cells that were
  contaminated. The shape is not established.
- The 7 to 18 percent range reported earlier was the contended box, not the spin.
- The peer agent's clean three-run data puts their 32-thread spin=0 against default at 4.3
  percent, with spreads of 0.47 and 0.21 - the same order, and they said so themselves.

**So neither runtime shows a resolvable spin effect at three runs.** The claim that the spin
is the whole of the 32-thread regression was a story that fit the data I had, and the data was
noisy. It is withdrawn in the same way the one-position finding was.

What survives does not depend on the spin at all: our bf16 path is 1.67x llama.cpp at
bit-identical weights, and our fp8 path is 74 s per token against their 0.25 s.

### The retry path was self-amplifying, and box-wide

The grid's failure handling dropped the whole page cache, on the theory that a failed run was
the previous run's cache starving the load. That is backwards: dropping the cache evicts the
149 GB of weights, so the next run re-reads all of it from disk, which makes it I/O-bound and
slower and more likely to fail, which drops the cache again. Two of three repeats in the first
cell failed.

And a cache drop is a box-wide event. It would have evicted a concurrent measurement's model
pages and turned it into a disk benchmark that nobody chose. The grid was stopped rather than
patched for that reason, and the rewrite drops the cache from the retry path entirely: a cold
149 GB re-read is exactly what makes a run fail, so depending on it was the wrong shape for a
retry. Results are appended rather than truncated, the repeats are numbered, and whether a
value followed a cache drop is recorded in the row.

## A correction worth keeping in the writeup

The 32-thread spin row was described as a slope with a cliff at 4096. Reconstructing the clock
against the log showed the three clean cells were 8.58, 8.45 and 8.50 - a 1.5 percent spread,
inside the noise for one run per cell - so the clean rows are a **plateau**, and the only two
low points were the two cells a 31-process test suite happened to land in.

The shape may be a plateau then a step rather than a slope, and the edge is bracketed in
(1024, 4096]. One point at spin 2048 would locate it, which is worth more than another repeat
of a cell already measured, and the step is the part that transfers between runtimes.

This is the same failure mode as the feature that started the work: a bug that hid behind a
plausible story until somebody read the raw data end to end. A table described before the
clock was checked deserves a sentence about it.


## The bf16 GGUF and the bf16 safetensors are the same weights

Checked, because it decides whether the llama.cpp row can sit beside ours with a precise
label rather than a caveat. Two tensors in two different layers, compared as raw bf16 bits:

| tensor | identical bf16 bits |
|---|---|
| `model.layers.20.mlp.experts.100.gate_proj.weight` | 12800/12800 (100.00%) |
| `model.layers.7.mlp.experts.3.up_proj.weight` | 12800/12800 (100.00%) |

So the GGUF and the safetensors are the same fp8 dequantisation rounded to bf16, written
twice. **llama.cpp and mistral.rs are therefore already weight-matched**, and the difference
between them is the implementation rather than the weights. Only the vllm.cpp row is genuinely
different, because it dequantises fp8 in process instead of reading a bf16 container.

Two traps worth recording, because both produced a wrong answer first:

- The GGUF stores its tensors as `(expert, f, h_bytes)` with the row padded to 5120 bytes, not
  as ggml's `(n_embd, n_ff, n_expert)`. Indexing it as the latter compares the wrong elements.
- Comparing bf16 as int16 on one side and uint16 on the other makes every negative value look
  like a mismatch. The first comparison came out at 50 percent "identical" - which is the
  signature of the sign bit, not of a real difference.

## The matched comparison, and why the headline was not quotable

The three-way table was not a comparison: three containers, single runs on our side, and a
spin setting that moves each row's optimum. The agreed protocol, with peer-Ce:

- Both sides on the fp8 directory (74 GB) so it fits alongside, sequential, never concurrent.
- The same spin settings: 0, 1024, 2048 and the 25,000 default at 32 threads; 0 and the default
  at 8 and 16.
- Three repeats per cell on both sides, reported with the spread.
- A 2048 cell on both sides, so the edge is bracketed in both.

The point estimate is not quoted until that is done. The direction - that 32 threads wants less
waiting - already survives on the clean cells alone.


## The final PR stack

Seven branches, one commit each, in order. The code stack tip reproduces
`cpu-decode-bandwidth`'s source exactly - the `.rs` diff between them is empty - so the split
loses nothing.

| branch | what it is |
|---|---|
| `pr/01-rmsnorm-scale-f32` | the RMSNorm scale fix, standalone, and the candle issue it produced |
| `pr/02-cpu-decode-engine` | the output-parallel CPU GEMV, the NEON dot, the pool sizing, the per-expert gather, the broadcast by addressing, the barrier pool, the q/k/v fusion, the phase profiler |
| `pr/03-kolibri1-model` | the Kolibri-1 model and its three correctness fixes, and the "Canberra" assertion |
| `pr/04-bfmmla-reduction-order` | the BFMMLA reduction order, default off |
| `pr/05-moe-residency-cache` | the byte-budgeted MoE residency cache, default off |
| `pr/06-layer-taps-and-bench` | the per-layer taps and the configurable benchmark prompt |
| `docs/perf-record` | this working log and the todo/ issue reports |

**The split is a tree-based one, not a replay of the history.** Cherry-picking commit by
commit fought the interleaving: the model work and the gather work are entangled, because
`f31fafb3a` adds the per-expert gather path the model's FP8 MoE fix needs and lands after
the model commits. Each branch instead takes the main branch's final state for the paths it
owns, on top of the previous branch's tree. Two files are owned by two PRs each and are split
at the hunk level: `models/kolibri1.rs` (PR 3 and PR 6) and `dot_cpu.rs` (PR 2 and PR 4).

**PR 2 is one commit and it is large.** That is a deliberate trade against a 20-commit PR with
no reviewable increments, which is how this branch became 71 commits in the first place. It
would be better split, but only along lines that do not re-introduce the entanglement.

**The working log and the todo/ tree are excluded from every code PR.** 24 of the 71 commits
touch the plan doc, and it is a local record rather than upstream documentation.

### Not yet verified

Each branch needs to build and pass its own tests. That needs cargo, and cargo breaks the
quiet-box precondition the peer agent is waiting on, so verification is deferred until that
window closes. The tree-level equivalence above is a structural check, not a build.
