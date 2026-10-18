---
title: Watermark GPU integration boundaries
description: Compact candidate integration and remaining host sampling boundaries.
---

The watermark adapter uses the same Candle package as mistral. GPU algorithms
remain in `llm-watermarking`; no watermark kernel or GPU runtime is implemented
in mistral. The library's CUDA feature uses NVRTC for its own metadata kernels.

## Connected path

The single-sequence CUDA/Metal top-k path calls the existing
`cuda_topk_logits_f32_packed` or `metal_topk_logits_packed` operation. It retains
actual vocabulary token IDs, original logits, and the existing softmax normalizer.
Candle prepares the same top-p/min-p masks as the host sampler. The adapter binds
the U32 candidate IDs and original vocabulary size with
`IndexedCandidates::new_trusted`, then calls the scheme's `prepare_indexed` and
`apply_trusted` on the K filtered weights. The host selects a token after its
existing compact readback. Keyed sampler outputs use argmax; probability-transform
outputs use the existing weighted categorical draw.

Metal tensor views with a nonzero storage offset are copied on the GPU because
the existing top-k helper accepts a base buffer without an offset. This performs
no host transfer and preserves the view's contents.

The readback contains three arrays of K values: watermark weights/scores, token
IDs, and original reporting probabilities. Carrying original probabilities
preserves logprob semantics. There is one readback, not a CPU round trip for the
watermark operation, and no vocabulary-sized probability download. The adapter
uses no library strict-validation scalar readback.

## Compact probabilities and vocabulary metadata

All seven token schemes consume compact `[K]` weights and return values in the same
candidate order. The adapter no longer scatters weights into a zero-filled
vocabulary row or gathers dense results. Top-k supplies unique, in-range token
IDs; filtering supplies finite nonnegative weights with positive mass. These
invariants permit the trusted constructors and application without a validation
readback. Zero weights keep excluded candidates ineligible.

Hashes and partitions still use actual vocabulary IDs and the full vocabulary
definition. SynthID, exponential race, and textGrain hash only K candidates. textGrain computes its block transport on the input device. KGW and MPAC
still reconstruct the full-vocabulary partition for each changed context;
Unigram and inverse transform cache full-vocabulary metadata per device. The
library owns these algorithms. Removing dense probability work does not remove
all O(V) metadata work or establish a throughput improvement.

The adapter uses the sequence's existing host history and prompt length.
Position-keyed schemes use the configured start position plus the generated-token
count, modulo their period. Replaying or discarding a branch does not advance
shared watermark state.

## Sampling traces

The HTTP `sampling_trace` extension requires the existing logprob path, which
performs sampling on the host even when model inference uses Metal or CUDA.
It calls the library's scalar traced methods and adds no device readback beyond
that path's existing logits transfer. Requests without traces keep the compact
GPU path and its single readback unchanged. See the
[sampling trace contract](/guides/customize/watermarking/#inspect-sampling-with-and-without-watermarking).

The library also provides device-resident dense/indexed trace tensors, but this
HTTP integration does not export them. The current compact sampler chooses on
the host after its readback, so gathering trace rows on the GPU after selection
would require another transfer. A future GPU trace export must include bounded
metadata in the existing packed readback. Device-history traces cannot identify
whether an inactive step was warmup or a repeated context; that reason must stay
unspecified rather than be inferred.

## Native textGrain sampling

textGrain's default probability-update policy uses the same compact device
insertion point. Its separate `block_then_token` policy calls the library's
scalar native sampler with the live sequence RNG and bypasses compact selection.
This policy requires ordinary token decoding; speculative and block decoding are
rejected. Native trace capture uses the same host path, including costs, coupling,
entropy diagnostics, and the actual block/token draw intervals. Capture does not
change the solver or RNG consumption. See the
[native trace contract](/guides/customize/watermarking/#native-textgrain-transport-traces).

## Host boundary: fused CUDA samplers

The current CUDA batch/resident and sparse speculative paths combine filtering,
selection, or acceptance inside existing kernels. They do not expose the filtered
probability tensor at the needed point. Removing their watermark exclusion would
silently skip watermarking or use the wrong acceptance distribution.

They remain excluded. Supporting them requires an explicit host insertion point
or composable sampling stages that apply watermarks before selection and use the
correct watermarked distributions and branch positions for acceptance.
Implementing a second GPU sampler merely to bypass this boundary would duplicate
mistral's inference stack and is outside this integration.

The library now provides `DeviceHistory`, device position preparation, and
`PreparedIndexedBatch` for independent rows. These APIs are available for future
resident integration, but the current top-k adapter does not use them. A resident
host must preserve independent per-sequence keys, histories, prompt lengths,
positions, payloads, and retry semantics. The library batch helper submits row
operations; it does not fuse mistral's filtering, selection, or acceptance stages.

## SemStamp boundary

SemStamp operates on sentence embeddings and returns signatures/acceptance masks.
Mistral's token sampler has no sentence-encoder and candidate-sentence retry hook.
This is a workflow mismatch, not a missing CUDA or Metal implementation in the
library. Its embedding operations are exposed through Rust and Python helpers;
ordinary text-generation requests reject the scheme explicitly.
`POST /v1/watermark/detect` accepts supplied SemStamp sentence embeddings for CPU
detection; it does not add the missing generation workflow.

## Verification

Tests cover every token scheme against the CPU reference, actual token IDs in
compact rows with K=1, K=4, and K=128, filtering order, exclusion support, original
reporting probabilities, warmup, repeated contexts, replay, wrapped key-stream
positions, and SemStamp embedding evidence. Feature-gated CUDA
and Metal tests execute both the device tensor operations and the existing GPU
top-k sampling entry point. Run them on the corresponding hardware:

```bash
cargo test -p mistralrs-core --features metal watermark -- --include-ignored
cargo test -p mistralrs-core --features cuda watermark -- --include-ignored
```

GPU floating-point reductions can differ from the scalar f64 reference. Hashes
and partitions must agree, while probability comparisons use tolerances. Synthetic
integration tests do not establish model quality or performance; benchmark the
intended model, vocabulary, top-k, batch sizes, and scheme.
