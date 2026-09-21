# Blueprint: Multi-Modal CUDA Graph Compilation Fix (Issue #2438)

## Objective
Prevent CUDA panic and memory pool pollution caused by dynamic vision embeddings injecting into statically captured CUDA graphs during the decode phase.

---

## 1. Graph Capture Interception (`mistralrs-core/src/vision_models/mod.rs` & `inputs_processor`)

### The Flaw
Currently, the pipeline checks if `pixel_values.is_none()` to determine if a batch is eligible for CUDA graph decode (since `pixel_values` are only sent during prefill). However, for multi-modal requests in the decode phase, `pixel_values` is `None` but the sequence still processes vision embeddings, altering tensor shapes and breaking the static graph.

### The Fix
We will introduce a boolean flag in `ModelInputs` to definitively track multimodal sequences.

**Struct Modification (`mistralrs-core/src/vision_models/mod.rs`):**
```rust
pub struct ModelInputs {
    pub input_ids: Tensor,
    pub seqlen_offsets: Vec<usize>,
    pub context_lens: Vec<(usize, usize)>,
    pub position_ids: Vec<usize>,
    pub pixel_values: Option<Tensor>,
    pub model_specific_args: Box<dyn Any>,
    pub paged_attn_meta: Option<PagedAttentionInputMetadata>,
    pub flash_meta: FlashParams,
    pub recurrent_batch_kind: RecurrentBatchKind,
    pub adapter_leases: Arc<[Option<crate::AdapterLease>]>,
    pub has_multimodal_data: bool, // NEW FIELD
}
```

**Initialization logic (in vision model `InputsProcessor` implementations, e.g., `llava_inputs_processor.rs`):**
```rust
let has_multimodal_data = input_seqs.iter().any(|seq| {
    seq.has_images() || seq.has_audios() || seq.has_videos()
});

let inputs: Box<dyn Any> = Box::new(ModelInputs {
    // ... existing fields
    has_multimodal_data,
});
```

---

## 2. The Engine Fallback Mechanism (`mistralrs-core/src/pipeline/multimodal.rs`)

### Interception Logic
We intercept the continuous batcher's forward pass before it attempts to replay the CUDA graph. In `MultimodalPipeline::forward_step`, we unpack the new flag and prevent execution.

**Target File:** `mistralrs-core/src/pipeline/multimodal.rs`
**Target Function:** `forward_step` (around line 2678)

```rust
let ModelInputs {
    input_ids,
    // ...
    pixel_values,
    model_specific_args,
    has_multimodal_data, // Extract the new flag
} = *inputs.downcast::<ModelInputs>().expect("Downcast failed.");

// Bypasses the CUDA graph runner (CudaGraphRunner::replay) if the batch contains vision tokens
if lora_execution.is_none() && !return_raw_logits && pixel_values.is_none() && !has_multimodal_data {
    match self.try_cuda_decode_graph_forward(CudaDecodeGraphForwardInput {
        // ...
    }) {
        Ok(Some(replay)) => {
            return Ok(ForwardStepResult::cuda_decode(
                ForwardInputsResult::CausalGeneration { logits: replay.logits },
                replay.launch,
            ))
        }
        Ok(None) => {}
        Err(err) => {
            if !self.disable_cuda_decode_graph(&err) {
                return Err(err);
            }
        }
    }
}
```
By failing the conditional, the pipeline skips `try_cuda_decode_graph_forward` and elegantly falls through to standard eager execution (`self.forward_inputs()`) on the main thread for these batches.

---

## 3. Memory Pool Isolation (`mistralrs-core/src/vision_models/`)

### The Memory Leak Origin
CUDA Graph capture (`prepare_cuda_graph_memory_pool` in `cuda_graph.rs`) locks the CUDA stream's memory pool and sets the release threshold to `u64::MAX`, retaining allocations. Because the vision encoder (e.g., CLIP) runs on the **same CUDA stream** as the text generation by default, its massive ViT allocations become trapped in the graph's memory pool and are never returned to the OS.

### Isolation Strategy
To isolate vision embeddings, we must decouple the Vision Tower's execution from the Text Decoder's CUDA stream. 

**Implementation (e.g., `llava15.rs` / `cached_encode_images`):**
1. **Dedicated Vision Stream:** Instantiate a separate CUDA stream strictly for vision/audio encoders.
   ```rust
   let vision_device = Device::new_cuda_with_stream(device_id, dedicated_vision_stream_id);
   ```
2. **Device Swap:** Pass this `vision_device` to the `ClipVisionTower` during `Model::new`. 
3. **Synchronization:** Execute `clip_vision_tower.forward(x)` on the isolated stream. After processing, transfer the resulting `image_features` tensor back to the primary stream used by the `LLaVALLM` via `.to_device(&primary_device)?`.

This completely prevents vision embedding buffers from polluting the stream-ordered allocator targeted by `sys::CUgraphMem_attribute::CU_GRAPH_MEM_ATTR_RESERVED_MEM_CURRENT`, fixing the memory exhaustion.

## Verification Protocol
To guarantee the swarm implemented the graph fallback and memory isolation correctly, the following verifications must be met:

### 1. Swarm-Generated Unit Tests
- Add a `#[test]` in `mistralrs-core/src/pipeline/multimodal.rs`.
- The test must mock a `ModelInputs` struct with `has_multimodal_data = true`.
- Assert that the execution path explicitly skips `try_cuda_decode_graph_forward` and cleanly falls through to eager execution.

### 2. Tracing Verification
- Inject a trace log immediately at the interception point in `forward_step`:
  `tracing::info!("Bypassing CUDA graph replay: Multimodal sequence detected");`
- This ensures silent failures do not happen; the fallback mechanism must be visibly verifiable in the terminal output when an image is processed.

### 3. The "Smoke Test"
- Serve a small vision model natively (e.g., `microsoft/Phi-3-vision-128k-instruct`) with CUDA graphs explicitly enabled:
  `cargo run --features cuda -- serve --cuda-graph -m microsoft/Phi-3-vision-128k-instruct`
- Pass an image to the prompt.
- **Success Criteria:** 
  - If it panics instantly, the CUDA graph fallback failed.
  - If it hallucinates garbage, the memory pool isolation failed (vision embeddings were overwritten).
  - If it correctly describes the image and prints the tracing log, the implementation is flawless.
