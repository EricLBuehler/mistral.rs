# Flint is supported (RecursiveSI), on the `flint_by_flint` branch

- **Repo:** mistral.rs
- **Components:** `mistralrs-core/src/models/flint.rs`, `mistralrs-core/src/gguf/normal_registry.rs`,
  `mistralrs-core/src/gguf/normal_bindings.rs`, `mistralrs-core/src/gguf/normal_config.rs`,
  `mistralrs-core/src/gguf/gguf_tokenizer.rs`, `mistralrs-core/src/pipeline/{normal,gguf,chat_template}.rs`
- **Status:** implemented on the `flint_by_flint` worktree branch (`/home/lu_zero/Sources/flint`,
  4 commits on `36db2a0ea`). The GGUF route runs end to end. Nothing pushed, nothing PRed.
- **Impact:** `flint` was rejected with `Unknown normal-model GGUF architecture`; it now loads,
  reasons and answers.

## What it is

Flint is a 40B-parameter MoE with 2.5B active, from RecursiveSI (not Mistral; the
`unsloth/flint-GGUF` and `RecursiveSI/flint` repos are the same model). Released 2026-10-21.

It is **the same MoE family as Kolibri-1**, with four changes: attention sinks, a per-layer
no-RoPE pattern, a two-layer dense lead, and a renormalised sigmoid router without the expert
bias.

| | Kolibri-1 | Flint |
|---|---|---|
| layers | 50 | 49 |
| hidden | 2560 | 2048 |
| heads / kv heads | 48 / 4 | 32 / 4 |
| head dim | 128 | 128 |
| experts | 384 | 256 |
| experts per token | 6 | 6 |
| shared experts | 1 (512) | 1 (1024), GGUF says `expert_shared_count = 2` |
| moe intermediate | 512 | 512 |
| dense ff (lead layers) | 512 | 6144, lead is `mlp_only_layers = [0, 1]` |
| sliding window | 513 | 513 |
| vocab | 128,000 | 150,144 |
| router | sigmoid, no renorm, with `exp_probs_b.bias` | sigmoid, **renormalised**, no bias |
| attention sinks | no | **yes** (`attn_sinks.weight`, 32 per layer) |
| rope | every layer | **14 layers have no rope** (`no_rope_layer_indices`) |
| norms | four per layer (sandwich) | two per layer (standard pre-norm) |

The last row is the one the survey missed: Kolibri-1's sandwich residual is *not* shared. Flint
is a standard pre-norm layer (`h = attn(ln1(x)) + x; out = mlp(ln2(h)) + h`), so the model is
closer to Qwen3-MoE with sinks than to Kolibri-1.

## What was implemented

1. **The model**, `mistralrs-core/src/models/flint.rs`. Reuses `MoEExperts` with
   `ExpertProjNames::DEFAULT` (the checkpoint's per-expert layout
   `mlp.experts.<i>.<gate|up,down>_proj.weight` is what `ExpertSourceLayout::PerExpert`
   detects), the sinks backend (`SdpaParams::sinks`, already wired for gpt-oss), and the f32
   router. The router selects on `sigmoid(logits)` and weights with the same value, then
   renormalises - `MoeRouterScoreFunction::Sigmoid` +
   `MoeRouterSelectedWeight::Sigmoid` + `renormalize`, no selection bias.
2. **The registry.** `CanonicalGgufArchitecture::Flint`, schema (requires `expert_count`,
   `expert_used_count`, `leading_dense_block_count`, and the `.attn_sinks.` tensor),
   `RopePairing::HalfSplit`, loader `NormalLoaderType::Flint`, config synthesis
   (`build_flint`), and the `flint` BPE pre-tokenizer profile (three regexes, as upstream
   llama.cpp splits them).
3. **The bindings.** Flint keeps the checkpoint's own names: the sink is
   `self_attn.softmax_offset` (not `sinks`), and the shared expert is `mlp.shared_expert`
   (singular). Everything else takes the default targets (`mlp.gate`, `mlp.experts.*_proj`,
   `mlp.{gate,up,down}_proj` for the dense lead).
4. **RoPE.** The no-rope list wins when present; without it the full-attention layers are the
   ones without positional encoding. Upstream llama.cpp puts `LLM_ARCH_FLINT` in the NeoX
   group, which matches HF's `rotate_half`, which matches `HalfSplit`.
5. **Stop tokens.** `tokenizer.ggml.eog_token_ids` is read and joined into the stop set. A
   GGUF can name several turn-ending tokens and the single `eos_token_id` does not cover them.
   `calculate_eos_tokens` grew the extra argument.

## Evidence

The run, on the K-quant build (`lmstudio-community/flint-GGUF/flint-Q4_K_M.gguf`, 24.17 GB,
sha-verified by size against the HF `x-linked-size`), node 2 of the shared box under load
(schedutil, 16 threads, `RAYON_NUM_THREADS=16`, load average ~150 from other agents' tests):

```
finish: stop
reasoning: 1.  The user asks for the capital of France in one word.
           2.  The capital of France is Paris.
           3.  The answer should be just one word.
content: Paris
tok/s: 10.08
```

Arithmetic is right and formatted: "45 minutes = 0.75 hours, 60 km / 0.75 h = **80 km/h**".
Decode throughput 4-11 tok/s depending on how loaded the box was, which brackets Kolibri-1's
bf16 8.65 tok/s and beats the Tenstorrent P150's 1.82 tok/s. Not a clean cell: the box was
busy throughout.

The architecture rejection is gone:

```
$ mistralrs run -f flint-preview-UD-Q4_K_XL.gguf -i "hello"
Error: ... uses dtype IQ2_S (22) for native binding `model.layers.10.mlp.experts.down_proj.weight`
```

which is the *quantisation* failing, not the architecture: config synthesis, the tokenizer and
every tensor binding resolve first.

## What is still open

1. **The published IQ GGUFs are still unreadable.** `flint-preview-UD-IQ1_M.gguf` and
   `flint-preview-UD-Q4_K_XL.gguf` need an IQ dequantiser (`todo/gaps/04`). The K-quant
   `flint-Q4_K_M.gguf` is the only runnable one today.
2. **The fp8 route.** `HawkBearPig/flint-FP8-BF16` uses the same blockwise fp8 recipe and the
   same per-expert tensor names as Kolibri-1, but its `config.json` declares
   `quant_method: "dgpp_mixed"`, which `QuantizedConfig` does not parse. That is the next
   increment, and it is the first test of the fp8 fast path on a checkpoint we did not author.
3. **The legacy lmstudio GGUF carries no `eog_token_ids`**, so it needs the `stop` parameter
   (or a caller-supplied stop list) to end a turn cleanly. Newer converters emit the key; the
   shim for files without it is not worth carrying.
4. **The upstream PR.** Four commits, ready to stack on `origin/master`. Nothing leaves the
   machine without an explicit go-ahead.
