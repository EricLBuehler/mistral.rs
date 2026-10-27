# Attention and shared decode optimization evidence

These artifacts describe the integrated source of the seventh canonical benchmark cell.
The [same-GGUF result](mrs-attention-8192-serving.json) and full prewarm completed 24requests
without errors. Decode leads the best measured llama.cpp configuration by 9.37%, 7.76% and
13.42%; the uniform 10% target remains unmet. All 74 distinct repository tests passed
(52 targeted plus 22 additional CPU), as did the release build and CUDA/CPU checks.
The optional dense-source packing patch remains outside the repository.

| Change | Standalone evidence | Production scope |
|---|---|---|
| cuTile split-K attention | [Evidence](cutile-qsa-decode-evidence.json): corrected 128MiB visited-KV rotation, BF16/F16, scalar CUDA control including reduction | Shared decode helper; head dimensions 128/256, grouped heads 1..16, runtime context/selection metadata; three probability components preserve precision. Unsupported layouts, dtypes and indexing/alignment bounds use existing CUDA attention. |
| Vectorized HC combine and normalization | [Evidence](hyper-vector-combine-evidence.json): 576 shape/dtype/stride combinations, 2808 normalized variant comparisons, hot and 128MiB rotating controls | Shared architecture kernel; BF16/F16/F32, four-element vectors, rounded residuals retained in registers, runtime dimensions. Odd, unaligned, small or over-capacity rows retain the original kernel. |
| Shared expert gate and residual addition | [Evidence](shared-gate-evidence.json): 3.96M exact element comparisons plus 74.84M prefill comparisons per tested variant; C1 and 17/512/8192-token timings | Existing generic scalar GEMV plus dtype-exact sigmoid/multiply/add, shared by Qwen3Next, Qwen3.5 MoE and Qwen4Exp. Existing LoRA ordering and general broadcast fallback remain. |
| Shared GGUF activation quantization | [Evidence](shared-mmvq-evidence.json), [repeat controls](gguf-shared-mmvq-repeat-summary.json) | Ten supported dense GGUF formats, including mixed-format groups; BF16/F16/F32 and 1..8 input rows. Each projection retains its own established CUDA geometry and explicit fused GLU/QKV/pair routes keep priority. |
| Sparse selection | [Evidence](qsa-selection-summary.json) | Shared-score staging and parallel integer histogram prefix; source-specific correctness and timing records accompany the summary. |

The cuTile prototype used explicit split variants. Production derives the split geometry
from SM count and tensor capacity and uses exact-f32 runtime scalars to avoid graph-time
JIT specialization changes. Its production tests passed, but the prototype timings alone
do not verify its complete-model speedup.

HC vectorization changes the FP32 sum association. The production graph test originally
used a CPU FP32 normalization reference that accumulated 2.44e-6 error on a constant
16,384-wide row. The CUDA result was 1.19e-7 from an independent FP64 oracle. The test
reference was changed to FP64 with the original tolerances and production kernel unchanged;
the evidence retains this diagnosis. Exact combined residual values and the separately
rounded input to normalization were checked independently.

[Validation snapshot](attention-optimization-evidence-index.json) separates repository
tests from standalone prototypes. The [release validation](attention-build-validation.json) and
[source manifest](attention-source-manifest.json) identify source `4ac5bbc7...` and
binary `2f66d0a6...`. Per-operation gains cannot be added and presented as a measured model gain.
