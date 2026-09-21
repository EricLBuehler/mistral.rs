# Blueprint: Multi-Head Speculative Decoding (Medusa/EAGLE) for mistral.rs

## Objective
Break the PCIe memory bandwidth bottleneck during the autoregressive decode phase by trading surplus FLOPs (available on Blackwell architecture) for token throughput. 
Implement Medusa/EAGLE speculative decoding to generate and verify multiple draft tokens per single base-model weight load, accelerating decoding from ~25 T/s to 100+ T/s.

---

## 1. The Sequence Tree State (`mistralrs-core/src/speculative/proposer.rs` & `kv_cache_manager.rs`)

### 1a. Struct Modification
Currently, `staged_speculative_tokens` is a linear enum:
```rust
pub enum SpeculativeTokens {
    Host(Vec<u32>),
    Device(Tensor),
}
```
We will introduce `SpeculativeTree` to support Medusa's branching graph, flattening the tree into arrays for GPU cache locality.

```rust
#[derive(Clone, Debug)]
pub struct SpeculativeTreeNode {
    pub token_id: u32,
    pub parent_idx: Option<usize>, // Index into `nodes`
}

#[derive(Clone, Debug)]
pub struct SpeculativeTree {
    pub nodes: Vec<SpeculativeTreeNode>, // Max length typically 64
    pub leaf_indices: Vec<usize>,
}

pub enum SpeculativeTokens {
    Host(Vec<u32>),
    Device(Tensor),
    Tree(SpeculativeTree), // NEW
}
```

### 1b. KV Cache Truncation & Rollback
In `mistralrs-core/src/paged_attention/kv_cache_manager.rs`, blocks are allocated sequentially in `req.block_ids`. If a tree evaluates 64 tokens, we allocate blocks for all 64. If only a 3-token branch is accepted, we must slice off the rejected blocks.

**Exact interception:**
```rust
// In `MedusaScheduler::step` after evaluating tree acceptance:
let accepted_len = evaluate_tree(base_logits, &mut seq);

// 1. Identify how many tokens to keep (base context + accepted drafts)
let new_total_len = seq.prompt_tokens() + seq.recognized_len() + accepted_len;

// 2. Call existing KVCacheManager function
let mut kv_mgr = get_mut_arcmutex!(kv_cache_manager);
kv_mgr.trim_request_to_num_tokens(seq.id(), new_total_len);

// 3. Compact accepted tree branch into the contiguous slots via CUDA copy_blocks kernel
// (If the accepted branch was evaluating at arbitrary offsets in the block table)
compact_accepted_kv_blocks(seq.id(), &accepted_branch_offsets);
```

---

## 2. The CUDA Tree Kernel (`mistralrs-paged-attn/src/cuda/pagedattention.cuh`)

### 2a. C++ Header / Signature
The current causal kernel `paged_attention_v1_kernel` hardcodes causal logic (`mask = token_idx >= context_len`). We must inject a `tree_mask`.

**Draft Signature:**
```cpp
__global__ void paged_tree_attention_v1_kernel(
    scalar_t *__restrict__ out,
    const scalar_t *__restrict__ q,
    const cache_t *__restrict__ k_cache,
    const cache_t *__restrict__ v_cache,
    const int num_kv_heads,
    const float scale, const float softcapping,
    const uint32_t *__restrict__ block_tables,
    const uint32_t *__restrict__ context_lens,
    const int max_num_blocks_per_seq,
    const float *__restrict__ alibi_slopes,
    const int q_stride, const int kv_block_stride, const int kv_head_stride,
    const float *k_scale, const float *v_scale,
    const float *__restrict__ sinks,
    // --- NEW EXACT FFI ARGS ---
    const int32_t *__restrict__ tree_mask, // Flat 1D array representing 2D boolean mask [num_seqs, max_tree_len, max_tree_len]
    const int max_tree_len
)
```

### 2b. Passing the Mask across FFI
In `mistralrs-paged-attn/src/cuda/ffi.rs`, the mask will be allocated as a `candle_core::Tensor` of `DType::I32` (acting as boolean 1/0) on the device.
It will be passed across the C-interface as an opaque `*const c_void`:
```rust
extern "C" {
    pub fn paged_tree_attention_v1(
        // ... existing args ...
        tree_mask: *const std::ffi::c_void,
        max_tree_len: i32,
    );
}

// Passed from Rust:
let tree_mask_ptr = tree_mask.as_ptr() as *const std::ffi::c_void;
```

---

## 3. The Engine Loop Interception (`mistralrs-core/src/engine/mod.rs`)

### Exact Execution Flow Hijack
Around line 1923 in `engine/mod.rs`, the engine submits a step. We will intercept the decode step (when `!is_prompt`).

```rust
// Current code around L1923:
// pipeline.submit_step( &mut guards_mut, is_prompt, ... ).await

// NEW MEDUSA INTERCEPTION:
let submission = if pipeline.has_medusa() && !is_prompt {
    // Phase 1: Medusa Draft (Drafting using Medusa MLPs on the last hidden state)
    let medusa_drafts = pipeline.submit_medusa_draft(&mut guards_mut).await?;
    
    // Update sequence state with tree
    for (seq, draft) in guards_mut.iter_mut().zip(medusa_drafts) {
        seq.staged_speculative_tokens = SpeculativeTokens::Tree(draft.tree);
    }

    // Build the 2D tree mask and attach to metadata
    let tree_mask = build_tree_mask_tensor(&guards_mut, device)?;
    metadata.tree_mask = Some(tree_mask); // Requires adding `tree_mask` to `PagedAttentionMeta`

    // Phase 2: Base Model Verify (Executes paged_tree_attention_v1_kernel)
    let base_logits = pipeline.submit_step(
        &mut guards_mut, 
        is_prompt, 
        /* ... */ 
        CacheBackendMetadata::PagedAttention { metadata }
    ).await?;

    // Phase 3: Acceptance & KV Rollback
    for (seq, logits) in guards_mut.iter_mut().zip(base_logits) {
        let accepted_len = evaluate_tree(logits, seq);
        let new_len = seq.prompt_tokens() + seq.recognized_len() + accepted_len;
        get_mut_arcmutex!(kv_cache_manager).trim_request_to_num_tokens(seq.id(), new_len);
    }
    
    base_logits // return the accepted outputs
} else {
    // Fallback to standard execution
    pipeline.submit_step(&mut guards_mut, is_prompt, /* ... */).await
};
```

---

## 4. Model Deserialization (`qwen2.rs` & `gemma4/text.rs`)

### 4a. Layer Splice Index
The Medusa heads operate on the final normalized hidden states *before* the Language Model Head (`lm_head`).

In `mistralrs-core/src/models/qwen2.rs` inside `Model::forward_embed`:
```rust
for (i, layer) in self.layers.iter().enumerate() {
    xs = layer.forward(&xs, ...)?;
}
let xs = xs.to_device(&self.device)?;
let xs = xs.apply(&self.norm)?;

// === MEDUSA SPLICE POINT ===
// Here `xs` has shape [batch, seq_len, hidden_size]
if let Some(medusa_head) = &self.medusa_head {
    // We stash the Medusa logits directly into the forward context
    let medusa_logits = medusa_head.forward(&xs)?;
    ctx.set_medusa_logits(medusa_logits);
}
// ===========================

let xs = ctx.logits(&xs)?;
ctx.lm_head(&*self.lm_head, &xs)
```

### 4b. Synchronization Across Thread Pool
Because `Model::forward` executes asynchronously via the `pipeline.submit_step` Rayon threadpool (`mistralrs-core/src/pipeline/mod.rs`), the Medusa logits must be synchronized. 

**Solution:**
We will add `medusa_logits: Option<Tensor>` to the `ModelForwardContext<'a>` struct in `mistralrs-core/src/pipeline/mod.rs`. The model populates this during the forward pass. Once `submit_step` returns the `StepSubmissionKind::Ready`, the engine extracts both `lm_head` logits and `ctx.medusa_logits()`, syncing them natively back into the main `Scheduler` thread without blocking the GPU streams.
