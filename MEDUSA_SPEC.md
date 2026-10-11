# Medusa-Style Multi-Head Speculative Decoding for mistral.rs

## Status: Proposal / Architecture Design (not started)
**Target Milestone:** Raise single-stream decode throughput (~25 T/s on Blackwell/PCIe-bound setups) by drafting several tokens per base-model weight load  
**Related Documents:**
* [`MULTIMODAL_CUDA_GRAPH_SPEC.md`](MULTIMODAL_CUDA_GRAPH_SPEC.md) - Shares the CUDA-graph decode gate in `forward_step`
* `mistralrs-core/src/speculative/` - The existing speculative framework this spec plugs into (read first)

> [!IMPORTANT]
> **Instructions for a local LLM picking this up.** The first draft of this spec assumed code that does not exist and
> duplicated code that does. Section 2 is the verified truth (2026-10-11). Section 2.2 lists symbols from the old
> draft that are **fake**: never reference them. Build Medusa as a new `SpeculativeProposer` inside the existing
> framework; do not add a second engine-loop interception. Do one Work Package at a time and run its acceptance
> commands before continuing.

---

## 1. Executive Summary & Problem Context

Autoregressive decode is memory-bandwidth bound: each token reloads all weights. Medusa adds small extra heads on the
final hidden state that each guess a future token; the base model then verifies all guesses in one forward pass and
keeps the longest accepted prefix.

### 1.1 What the repo already gives us
mistral.rs already has speculative decoding (MTP for Gemma4/Qwen3.5, DFlash). The pieces Medusa needs are present:

```mermaid
flowchart LR
    CFG[SpeculativeConfig] --> PIPE[Pipeline: SpeculativePipelineExt]
    PIPE -->|speculative_propose| PROP[SpeculativeProposer]
    PROP -->|SpeculativeProposalBatch| DRV[driver.rs]
    DRV -->|build_speculative_verify_inputs| FWD[target forward]
    FWD --> VER[verifier.rs accept/reject]
    VER -->|rollback_to| KV[trim_request_to_num_tokens]
```

### 1.2 The gap
Existing proposals are **linear** (`SpeculativeTokens::{Host,Device}`). Medusa's strength is a **tree** of candidates
(top-k per head). Tree verification needs a non-causal attention mask in the verify pass, which does not exist.
Therefore the plan is staged: linear Medusa first (cheap, reuses everything), tree second.

---

## 2. Ground Truth (verified in tree)

### 2.1 Exists
| Symbol | Where |
|---|---|
| `SpeculativeProposer { proposal_len, propose(ctx, target_embedder) -> SpeculativeProposalBatch }` | `speculative/proposer.rs:444` |
| `SpeculativeProposal { tokens: SpeculativeTokens, distribution: Option<SpeculativeProposalDistribution> }` | `speculative/proposer.rs:316` |
| `enum SpeculativeTokens { Host(Vec<u32>), Device(Tensor) }` | `speculative/proposer.rs:240` |
| `trait SpeculativePipelineExt` (`speculative_propose`, `speculative_commit`, `build_speculative_verify_inputs`, `has_speculative_proposer`, ...) | `speculative/driver.rs:~97` |
| `enum SpeculativeConfig { Off, Mtp(MtpConfig) }` | `speculative/config.rs:~17` |
| KV rollback: `PagedSpeculativeCacheGuard::rollback_to` calls `KVCacheManager::trim_request_to_num_tokens(request_id, num_tokens)` | `speculative/cache.rs:~120`, `paged_attention/kv_cache_manager.rs:382` |
| Gemma4 MTP proposer (closest template to copy) | `vision_models/gemma4/mtp.rs`, `has_speculative_proposer` at `vision_models/gemma4/mod.rs:1167` |
| Pipeline step entry (`submit_step`, takes `is_prompt`, `StepLookahead`) | `pipeline/mod.rs:2002` |
| Paged attention decode kernels `paged_attention_v1_kernel` / `v2_kernel` (single query token per sequence) | `mistralrs-paged-attn/src/cuda/pagedattention.cuh:490/518` |
| Block copy kernel | `mistralrs-paged-attn/src/cuda/copy_blocks_kernel.cu` |

### 2.2 Does NOT exist (old-draft fiction, do not use)
`SpeculativeTree`, `SpeculativeTreeNode`, `SpeculativeTokens::Tree`, `MedusaScheduler`, `pipeline.has_medusa()`,
`pipeline.submit_medusa_draft`, `build_tree_mask_tensor`, `evaluate_tree`, `compact_accepted_kv_blocks`,
`PagedAttentionMeta.tree_mask`, `ctx.set_medusa_logits`, `medusa_head` field on `qwen2.rs`, `paged_tree_attention_v1_kernel`,
and an interception "around line 1923 of `engine/mod.rs`" (the engine does not host speculative logic; `driver.rs` does).

### 2.3 Must confirm before coding (use `grep`, do not guess)
* Where `ModelForwardContext` is defined and whether it already carries hidden-state taps: `grep -rn "struct ModelForwardContext" mistralrs-core/src`.
* How the verify pass attends over multiple query tokens per sequence (it is not the q_len=1 paged v1/v2 kernel): read `build_speculative_verify_inputs` in `pipeline/normal.rs` and `pipeline/multimodal.rs` and follow to the attention call.
* How `speculative_target_hiddens(rows)` exposes the final hidden state to a proposer (Medusa heads consume it).

---

## 3. Design

### 3.1 Medusa heads (new file `mistralrs-core/src/speculative/medusa.rs`)
Each head k (k = 1..K) is a residual SiLU block followed by an LM-head projection, applied to the **post-norm final
hidden state** (`[rows, hidden]`), predicting token t+k+1:
```
logits_k = lm_head_k( h + silu(W_k h) )
```
Heads load from a separate safetensors checkpoint (new `MedusaConfig { model: String, num_heads: Option<usize> }`),
with `VarBuilder.pp(...)` prefixes verified against that checkpoint's `model.safetensors.index.json` (AGENTS.md "Models").
Heads may reuse the base `lm_head` weights (`medusa_use_base_lm_head`) to save VRAM.

### 3.2 Linear Medusa as a `SpeculativeProposer` (WP2)
`MedusaProposer::propose` takes the target hidden states, runs K heads, takes argmax (or samples via `sample_draft_rows`)
per head, and returns `SpeculativeProposal { tokens: Host(vec_of_K_tokens), distribution }`. The existing driver and
verifier then work unchanged, including rollback.

### 3.3 Config wiring (WP3)
Add `SpeculativeConfig::Medusa(MedusaConfig)`; handle it everywhere `SpeculativeConfig::Mtp` is matched
(`grep -rn "SpeculativeConfig::" mistralrs-core/src mistralrs/src mistralrs-cli/src mistralrs-pyo3/src`). Model support
is opt-in per model through `has_speculative_proposer`; start with Gemma4 text (`vision_models/gemma4/text.rs`) and
the plain Qwen2 text model.

### 3.4 Tree verification (WP4, only after WP2-3 are measured)
Prerequisite findings from Section 2.3. Candidate flattening (parents array, <= 64 nodes), a `[num_seqs, T, T]` boolean
ancestor mask, and accepted-path KV compaction (copy accepted slots into contiguous positions with
`copy_blocks_kernel.cu`, then `trim_request_to_num_tokens`). The attention change belongs to whichever kernel the verify
pass actually uses (Section 2.3), not to `paged_attention_v1_kernel`, which only serves q_len = 1.

```rust
pub struct SpeculativeTree {
    pub tokens: Vec<u32>,
    pub parents: Vec<Option<usize>>, // index into `tokens`
    pub leaf_indices: Vec<usize>,
}
```
Do not add a `Tree` variant to `SpeculativeTokens` until verifier/driver call sites are listed
(`grep -rn "SpeculativeTokens::" mistralrs-core/src`); each `match` must be updated.

### 3.5 CUDA-graph interaction
The verify pass has its own graph handling (`SpeculativeGraphState`, `SpeculativeGraphPlan` in `speculative/policy.rs`
and `target.rs`). Medusa must not capture graphs in WP2; add them only by following the MTP path.

---

## 4. Work Packages

| WP | Scope | Files | Acceptance |
|---|---|---|---|
| WP1 | Read-only: answer the three items in Section 2.3 and write the answers into this doc | none | Section 2.3 replaced with concrete file:line facts |
| WP2 | `medusa.rs` heads + `MedusaProposer` (linear), unit tests with random weights | `speculative/medusa.rs`, `speculative/mod.rs` | `cargo test -p mistralrs-core medusa` |
| WP3 | `SpeculativeConfig::Medusa`, loader + CLI/API plumbing, Gemma4 and Qwen2 `has_speculative_proposer` | `speculative/config.rs`, model files, `mistralrs-cli`, `mistralrs` | `cargo check --workspace`; `cargo test -p mistralrs-core speculative` |
| WP4 | Tree candidates, mask, KV compaction | decided by WP1 | acceptance rate >= linear on a fixed prompt set |
| WP5 | Benchmark | `mistralrs bench` | table in Section 6 filled in |

Rules for every WP: `cargo fmt --all`; `cargo check` (with and without `--features cuda`); `cargo clippy -p mistralrs-core -- -D warnings`;
no new dependencies; follow the Code Style section of `AGENTS.md` (ASCII only, no multi-line comments, hoist constants).

---

## 5. Verification Plan

### 5.1 Correctness
Greedy decoding with Medusa on must equal greedy decoding with speculation off, token for token, over a fixed set of
>= 20 prompts. Any divergence is a verifier/rollback bug, not a quality tradeoff.

### 5.2 Cache integrity
After each request, the KV block count for the sequence must equal `prompt_len + generated_len` (no leaked draft blocks).
Unit-test `rollback_to` with partial accepts.

### 5.3 Performance table (fill in at WP5)
| Setup | Tokens/s | Mean accepted tokens per step |
|---|---|---|
| Baseline (speculation off) | | 1.0 |
| Linear Medusa K=3 | | |
| Tree Medusa | | |

---

## 6. Open Questions (ask the user, do not guess)
1. Which Medusa checkpoint is the target? Heads are model-specific and must be trained or downloaded; none exist in the repo.
2. Is Medusa preferred over the existing Gemma4 MTP path? If a Gemma4 MTP assistant checkpoint is available, Medusa may add little.
