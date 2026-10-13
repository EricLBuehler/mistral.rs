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

`kolibri1_decode_benchmark`, bf16 (149 GB, 32 shards), node 3 with memory-local placement
(`--cpunodebind=3 --interleave=3`), batch 1, 40 decode tokens:

| | tok/s | TPOT |
| --- | --- | --- |
| when this section was first written | 1.73 | 577 ms |
| after broadcasting activations by addressing | 1.92 | 519 ms |
| after sending 512-output projections to the CPU GEMV | 4.32 | 231 ms |
| after fusing q/k/v | **4.64** | **216 ms** |

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

So the gather runs at 90% of what the memory system can deliver. Prefetching it and tuning its
tiling, which was the original plan, would buy almost nothing.

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

A methodological note, because it cost real time three times over: every isolated benchmark
here initially reported a cache-resident number. The first read-bandwidth test measured add
latency rather than bandwidth (2.7 GB/s at one thread against a real 7.8). The first gather
benchmark reused one set of weights (15.7 MB, L3-resident) and reported 52 GB/s where cold it
is 88. The first fused-qkv benchmark did the same. Rotate enough independent weight sets that
each pass reads what the last did not, or the number is fiction.

Throughput measured on a model that is not yet bit-exact against the goldens
(`todo/mistralrs/07`) should be read with that in mind.
