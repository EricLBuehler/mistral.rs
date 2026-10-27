# mistral.rs v0.9.5 Qwen3.8-Flash-Next CUDA C1 Benchmark

This report retains canonical Qwen/Qwen3.8-Flash-Next measurements on one NVIDIA GB10, using the same
UD-Q4_K_XL GGUF checkpoint in mistral.rs and freshly rebuilt llama.cpp. It also records
the changes that replace narrow model-specific kernels with shared architecture and
quantization implementations.

The latest measured source `24b72dc9...` and release binary `6681778b...` achieve the requested
10% decode lead in all three measured C1 prompt-length cases using the same GGUF
checkpoint. Against the best historical or fresh llama.cpp median selected separately
for each metric and length, prompt throughput leads by 60.5% to 69.1% and decode by
13.81%, 11.45% and 17.15%. No vLLM performance comparison is claimed.

Both fresh llama.cpp controls, at 512- and 8,192-token batch limits, completed full
prewarm and measured sweeps after the final mistral.rs run. Their
[512 metadata](raw/llama-verify-paired512-metadata.json) and
[8,192 metadata](raw/llama-verify-paired8192-metadata.json) confirm unchanged binaries
and loaded libraries, with no compiler or profiler active during measurement.

The three historical mistral.rs runs use source `b55a21db...` and binary `eca4ba52...`.
The next mistral.rs run uses `464e5a6f...` / `81d1e23b...`, followed by the
attention-optimization run `4ac5bbc7...` / `2f66d0a6...` and the latest measured paired-projection
run `24b72dc9...` / `6681778b...`. The llama.cpp controls use commit `7aca0d2e...` and
binary `b6704403...`. Every retained result identifies its own engine and provenance.

A subsequent cleanup moved tests into sibling modules, removed unused declarations,
unreachable kernel instantiations and stale comments, and clarified shared API documentation.
Kernel arithmetic and dispatch policies are unchanged. Its
[validation record](raw/cleanup-validation.json) and
[kernel reachability audit](raw/kernel-reachability.json) are separate from the benchmark
provenance; no serving sweep was rerun after cleanup.

## Latest measured same GGUF, best recorded llama.cpp controls

Each baseline cell independently selects the highest median across historical and fresh
llama.cpp controls at both batch limits. This is a conservative per-metric comparison;
it does not imply one llama configuration simultaneously produced every selected value.
The latest measured mistral.rs run uses 8,192-token prompt batches in every row.

| Prompt tokens | mistral.rs prompt tok/s | Best llama.cpp prompt tok/s | Difference | mistral.rs decode tok/s | Best llama.cpp decode tok/s | Difference |
|---:|---:|---:|---:|---:|---:|---:|
| 512 | 1,239.52 | 772.23 | +60.5% | 31.48 | 27.66 | +13.8% |
| 2,048 | 1,849.43 | 1,093.76 | +69.1% | 30.58 | 27.44 | +11.4% |
| 8,192 | 1,949.88 | 1,211.21 | +61.0% | 30.44 | 25.98 | +17.2% |

The [final comparison](raw/final-comparison.json) records the winning control file and
batch limit for every cell. All ten canonical runs remain in the
[manifest](raw/run-manifest.json), and the [summary](raw/summary.json) recomputes their
medians and comparisons from the retained request records.

## Latest measured run validation

The checkpoint, fixture, precision settings, 16,384-token context limit and concurrency
are unchanged. All 48 mistral.rs decoder layers remain on CUDA. Values are medians of
three measured requests after a full prewarm sweep and one per-length warmup.
Both engines use BF16 KV caches. Existing runtime arithmetic differs: mistral.rs uses
BF16 activations and casts F32 dense weights to BF16, while llama.cpp retains F32 dense
storage. This behavior predates these optimizations. Greedy outputs and expert routing
differ, so this is fixed-prompt serving rather than teacher-forced identical-token decode.
The [independent methodology audit](raw/paired-independent-methodology-audit.json)
records these limits and checks the fixture, counts, source and build provenance.

The [canonical result](raw/mrs-paired-8192-serving.json) and
[prewarm sweep](raw/mrs-paired-8192-prewarm.json) completed all 24 requests without
NaNs or request errors. Startup took 155.4 seconds and is excluded from throughput.
The maximum measured inter-token interval was 45.70 ms, with 20 intervals above 40 ms.
The [post-sweep scan](raw/mrs-paired-8192-health.json) found 26.813 GiB of PLE pages
resident and 0.031 GiB resident in eligible copied-source ranges. These observations
do not establish the cause of latency variation.

Compared with the preceding source, decode improves by 4.1%, 3.4% and 3.3%.
The short, medium and long prompt rates change by +0.7%, -0.5% and -0.1%; the strong
prefill lead is preserved. This result measures mixed GGUF/dense projection fusion,
compatible dense-source packing and accompanying GEMV correctness fixes together.
The [historical matched-batch comparison](raw/paired-comparison.json),
[command](raw/mrs-paired-8192-command.json),
[source manifest](raw/paired-source-manifest.json) and
[build/test record](raw/paired-build-validation.json) identify the measured release.

## Previous same GGUF, attention-optimization source

This section records source `4ac5bbc7...` and binary `2f66d0a6...`.

The checkpoint, fixture, precision settings, context limit and concurrency are unchanged.
Both engines use 8,192-token prompt batch limits. All 48 mistral.rs decoder layers remain
on CUDA. Values are medians of three measured requests after the full prewarm sweep and
per-length warmup.

| Prompt tokens | mistral.rs prompt tok/s | llama.cpp prompt tok/s | Difference | mistral.rs decode tok/s | llama.cpp decode tok/s | Difference |
|---:|---:|---:|---:|---:|---:|---:|
| 512 | 1,231.24 | 762.02 | +61.6% | 30.25 | 27.38 | +10.5% |
| 2,048 | 1,858.96 | 1,093.76 | +70.0% | 29.57 | 26.98 | +9.6% |
| 8,192 | 1,952.44 | 1,210.51 | +61.3% | 29.47 | 25.31 | +16.4% |

The [canonical result](raw/mrs-attention-8192-serving.json) and
[prewarm sweep](raw/mrs-attention-8192-prewarm.json) completed all 24 requests without
NaNs or request errors. No measured inter-token interval exceeded 40 ms. The
[post-sweep scan](raw/mrs-attention-8192-health.json) found the complete 26.822 GiB PLE
table resident and 0.016 GiB resident in eligible copied-source ranges. Startup took
158.1 seconds and is excluded from throughput.

Compared with the intermediate source below, decode improves by 0.8%, 1.3% and 1.1%.
These combined serving gains are smaller than the sum of isolated kernel improvements;
a fresh profile confirms the new cuTile attention, HC and scalar-gate kernels execute.
Profiled kernel savings exceed serving savings, so isolated timings are not used to
predict additive model gains.
Against the best measured llama.cpp decode configuration, the short and medium cells
needed about 0.190 and 0.690 ms/token of improvement to reach a 10% lead. The
[comparison record](raw/attention-comparison.json), [command](raw/mrs-attention-8192-command.json),
[source manifest](raw/attention-source-manifest.json) and
[build/test record](raw/attention-build-validation.json) identify this result.

## Intermediate same GGUF, matched 8,192-token batches

This section records source `464e5a6f...` and binary `81d1e23b...`.

The checkpoint, fixture, precision settings, context limit and concurrency are unchanged
from the historical same-GGUF runs. Both engines use 8,192-token prompt batch limits.
All 48 mistral.rs decoder layers remain on CUDA. Values are medians of three measured
requests after the full prewarm sweep and per-length warmup.

| Prompt tokens | mistral.rs prompt tok/s | llama.cpp prompt tok/s | Difference | mistral.rs decode tok/s | llama.cpp decode tok/s | Difference |
|---:|---:|---:|---:|---:|---:|---:|
| 512 | 1,230.75 | 762.02 | +61.5% | 30.00 | 27.38 | +9.6% |
| 2,048 | 1,836.31 | 1,093.76 | +67.9% | 29.19 | 26.98 | +8.2% |
| 8,192 | 1,928.97 | 1,210.51 | +59.4% | 29.15 | 25.31 | +15.2% |

The intermediate [canonical result](raw/mrs-resumed-8192-serving.json) and
[prewarm sweep](raw/mrs-resumed-8192-prewarm.json) completed all 24 requests without
NaNs or request errors. No measured client inter-token interval exceeded 40 ms.
The [post-sweep scan](raw/mrs-resumed-8192-health.json) found the complete 26.822 GiB
PLE table resident and only 0.032 GiB resident in the eligible copied-source ranges.
These observations are consistent with reduced memory pressure, but the combined run
does not isolate the contributions of cache reclamation, router dispatch and expert staging.

The process took 141.3 seconds to become ready; startup remains excluded from throughput.
The [command](raw/mrs-resumed-8192-command.json),
[source manifest](raw/resumed-source-manifest.json) and
[build/test record](raw/resumed-build-validation.json) identify this result. Greedy
outputs still differ across engines, so decode compares fixed-prompt serving rather
than teacher-forced identical routing.

## Historical same GGUF, matched 512-token chunks

Each value is the median of three measured requests. Both engines use BF16 KV caches,
one active sequence, a 16,384-token context limit, and 128 generated tokens. mistral.rs
selects BF16 activations; llama.cpp retains its internal activation implementation.
Prompt throughput includes all work through the first streamed output token.

| Prompt tokens | mistral.rs prompt tok/s | llama.cpp prompt tok/s | Difference | mistral.rs decode tok/s | llama.cpp decode tok/s | Difference |
|---:|---:|---:|---:|---:|---:|---:|
| 512 | 1,218.29 | 769.22 | +58.4% | 29.17 | 27.66 | +5.5% |
| 2,048 | 1,291.51 | 809.55 | +59.5% | 28.61 | 27.44 | +4.3% |
| 8,192 | 1,122.91 | 799.47 | +40.5% | 26.90 | 25.98 | +3.6% |

The canonical files are [mistral.rs](raw/mrs-production-512-serving.json) and
[llama.cpp](raw/llama-clean-serving.json). The historical mistral.rs release binary completed
a fresh-server 12-request prewarm sweep followed by the complete 12-request measurement
sweep without NaNs or request errors. Three per-length warmup requests are excluded
from each table's medians.

## Historical same GGUF, matched 8,192-token batches

Both engines were also measured with 8,192-token prompt limits. llama.cpp uses
`-b 8192 -ub 8192`; mistral.rs sets both batched-token and prefill-chunk limits to 8,192.
The binaries, checkpoint, KV precision, context limit, fixture and sampling settings
are otherwise unchanged. All 48 mistral.rs decoder layers remain on CUDA.

| Prompt tokens | mistral.rs prompt tok/s | llama.cpp prompt tok/s | Difference | mistral.rs decode tok/s | llama.cpp decode tok/s | Difference |
|---:|---:|---:|---:|---:|---:|---:|
| 512 | 1,224.88 | 762.02 | +60.7% | 28.67 | 27.38 | +4.7% |
| 2,048 | 1,811.11 | 1,093.76 | +65.6% | 25.23 | 26.98 | -6.5% |
| 8,192 | 1,889.75 | 1,210.51 | +56.1% | 25.67 | 25.31 | +1.4% |

The canonical files are [mistral.rs](raw/mrs-production-8192-serving.json) and
[llama.cpp](raw/llama-chunk8192-serving.json). Both completed a full prewarm sweep
before the measured sweep. The fresh mistral.rs process completed all 24 requests
without NaNs or request errors. The [llama.cpp metadata](raw/llama-chunk8192-metadata.json)
also confirms that its loaded artifacts did not change during measurement.

Larger batches improve both engines' long-prompt throughput. mistral.rs decode is
slower in the 2,048-token cell; these measurements alone do not identify its cause.
The tested limits are two configurations, not an exhaustive search for either engine's
best configuration.

## Historical native quant-4, separately quantized weights

The same mistral.rs release binary was also measured with its existing `--quant 4`
preset, loading the native checkpoint with an 8,192-token prompt limit. This preset
was not changed by this work. It uses a different mixed-precision weight distribution
from the UD-Q4_K_XL GGUF; the following comparison therefore combines engine and
quantization differences.

| Prompt tokens | Native quant-4 prompt tok/s | llama.cpp GGUF prompt tok/s | Difference | Native quant-4 decode tok/s | llama.cpp GGUF decode tok/s | Difference |
|---:|---:|---:|---:|---:|---:|---:|
| 512 | 1,210.65 | 762.02 | +58.9% | 38.63 | 27.38 | +41.1% |
| 2,048 | 1,808.77 | 1,093.76 | +65.4% | 37.40 | 26.98 | +38.7% |
| 8,192 | 1,907.41 | 1,210.51 | +57.6% | 37.27 | 25.31 | +47.3% |

The [native results](raw/mrs-production-native4-8192-serving.json) use the same fixture,
BF16 KV cache, context limit, sampling and warmup policy as the GGUF runs. All 48
decoder layers remained on CUDA. A fresh 12-request prewarm sweep and the complete
12-request measured sweep passed without NaNs or request errors.

| Component | Native quant-4 | UD-Q4_K_XL GGUF |
|---|---|---|
| Token embedding and output head | Q6_K | Q8_0 |
| Attention/GDN projections | Q4_K | Mostly Q8_0; GDN a/b stored as F32 |
| Shared experts | Q4_K gate/up; Q4_1 down | Q8_0 |
| Hyper-connection down/up/injection | Q6_K / Q5_1 / Q6_K | Q8_0 / Q8_0 / F32 |
| Indexer and PLE projections | Q6_K | F32 indexer; Q8_0 PLE |
| Routed expert gate/up | Q4_K in all 48 layers | Q4_K in 47; Q5_K in one |
| Routed expert down | Q4_1 in all 48 layers | Q5_1 in 43; Q8_0 in five |
| PLE table | Symmetric Q4, groups of 32 with F16 scales | IQ4_NL |

The native mapped Q4 table was confirmed at runtime. Its packed size is the same as
the GGUF IQ4_NL table, but its values differ. The routers and shared-expert gates remain
dense. For the GGUF route, mistral.rs loads the listed F32 projection tensors as BF16,
while llama.cpp retains F32; identical source files do not imply identical runtime
arithmetic even in the same-GGUF tables.

The cataloged native text weights excluding PLE occupy 73.825 GB versus 82.362 GB
for mistral.rs's GGUF representation. An idealized single-pass active-weight payload
estimate is 4.216 GB versus 6.173 GB per decoded token, about 31.7% less. This estimate
is not measured memory traffic or allocated GPU memory; its assumptions and exclusions
are recorded in the [quantization disclosure](raw/native-quant4-disclosure.md) and
[inventory summary](raw/native-quant4-disclosure.json).

The smaller payload can materially improve decode throughput. These results do not
establish equal model quality, isolate kernel improvements, or demonstrate a new decode
gain over the earlier branch's native preset. The same-GGUF comparisons remain the
appropriate results for separating this precision change from the engine comparison.

## Shared implementation in the measured source

Hyper-connections now live in `mistralrs-core/src/layers/hyper_connection.rs` and the
shared CUDA module. Their configuration supplies stream count, hidden width, projection
rank and normalization epsilon. Qwen4Exp text and MTP use this layer; the implementation
does not depend on a model configuration or fixed 4-by-2,560-by-320 dimensions.

The gated up projection can quantize scaled-SiLU activations and reduce sigmoid-weighted
residual streams without materializing the expanded gate. The fused input preparation
preserves the input-dtype rounding after both division and SiLU. Dense packed down and
injection projections feed row-strided consumers directly, avoiding copies of the split
views. CPU execution and unsupported fusion cases retain ordinary operations.

GGUF vector-dot math is shared across dense projections, indexed experts and gated
mixing. The new dispatch policies use weight storage, reduction width, routing and
hardware capabilities rather than an exact model shape. Compact expert prefill uses
adaptive 16/32/64/128-column tiles and handles partial rows. Indexed-expert prefetch
supports one or two tokens. An earlier exact-shape exception for 24 to 31 tokens was
removed after routing-dependent comparisons failed to justify a general replacement.

| Operation | Accelerated coverage | Fallback |
|---|---|---|
| Dense and fused QKV prefetch | Ten established dense GGUF formats; BF16/F16/F32; eligible SM121 single-token shapes | Existing multi-row MMVQ |
| Indexed expert prefetch | Eleven GGUF vector formats; BF16/F16/F32; SM121; one or two tokens; runtime expert and top-k dimensions | Existing indexed expert dispatch |
| Compact expert prefill | Ten MMQ formats; SM121; runtime routing and partial rows | Regular MMQ when worklist limits are exceeded |
| Quantized gated projection and mixing | Eleven vector formats; BF16/F16/F32; one token; runtime streams, width and rank | Projection followed by shared mixing |
| Paired projections | Existing FP8/NVFP4 and GGUF shared-input preparation; eligible single-token mixed GGUF/dense pairs in ten dense formats and BF16/F16/F32 | Independent projections, retaining affine/Marlin precedence |
| GGUF embedding gathers | Sixteen storage formats; CPU, resident CUDA and supported mapped CUDA memory | CPU gather when mapped GPU access is unavailable |

The ten dense/MMQ formats are Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K, Q3_K, Q4_K,
Q5_K and Q6_K. Indexed experts and gated mixing additionally support Q8_1. Embedding
gathers support all eleven plus F32, F16, BF16, Q8_K and IQ4_NL. The prefetch policy
preserves existing kernels for small matrices and unpacking-bound low-bit reductions.
Other quantization backends keep their established kernels. Bias, active dynamic LoRA,
statistics capture and incompatible layouts retain the relevant unfused paths.

Independent tests exposed an inherited Q8_1 dot-product error: the stored block sum is
metadata, not an affine offset. The shared implementation removes the erroneous product
of block sums and is checked against a dequantized reference.

## Embedding storage and startup

The large per-layer embedding table retains its source quantization. Generic GGUF row
gathers handle its narrow IQ4_NL rows, including widths that the previous CUDA route
could not accept. The mapped-memory path retains reclaimable file-backed storage.

On devices that support pageable access through host page tables, a shared raw-byte
preparation operation touches host pages and performs a synchronized GPU page-table
sweep before serving. In the standalone actual-table workload, startup preparation took
40.7 seconds and subsequent mapped gathers averaged 7.08 microseconds. The unprepared
mapped path's median per-round mean was 352.76 microseconds. These are isolated gather
measurements, not a claim of a corresponding model-level speedup.

Preparation moves a real cost into startup. Development runs observed roughly 29 to
40 seconds; the historical GGUF512 server logged 56 seconds for table preparation and took
198.5 seconds to become ready overall. GGUF8192 logged 103 seconds for preparation
and 236.3 seconds to become ready. Loading times depend on filesystem cache and
memory pressure, so they are not presented as a cold-start comparison. Prefaulting does
not prevent later eviction. Callback-based gathers and a GPU row cache were evaluated
but were not selected for production.

The native quant-4 process took 540.4 seconds to become ready, including native weight
quantization. Preparing its 26 GiB mapped Q4 PLE table, including quantization and
prefaulting, took 171 seconds. This preparation time is not directly comparable to the
prefault-only GGUF table timing.

## Kernel measurements

These development microbenchmarks explain implementation choices. They are separate
from the full-model serving results above.

| Operation | Previous | Retained implementation | Unit |
|---|---:|---:|---|
| BF16 scaled-SiLU plus Q8_1 input preparation | 1.584 | 0.907 | us |
| BF16 preparation, gated up projection and mix, rotating weights | 20.839 | 20.105 | us |
| F16 preparation, gated up projection and mix, rotating weights | 21.638 | 20.519 | us |
| Fused QKV projection, rotating weights | 155.635 | 148.258 | us |

The complete dense consumer benchmark includes both projections, scaled SiLU, residual
combine and the next grouped normalization:

| Dtype | Tokens | Separate projections, us | Packed projection with strided consumers, us |
|---|---:|---:|---:|
| BF16 | 1 | 12.821 | 11.028 |
| BF16 | 17 | 23.460 | 20.598 |
| BF16 | 512 | 250.942 | 242.383 |
| BF16 | 2,048 | 1,069.949 | 886.584 |
| F16 | 1 | 12.292 | 10.247 |
| F16 | 17 | 22.604 | 17.543 |
| F16 | 512 | 248.548 | 240.589 |
| F16 | 2,048 | 1,068.928 | 889.314 |

Packing is a loader capability. In the earlier measured source, GGUF and immediate-ISQ
sources declined this packing path; the GGUF checkpoint's mixed quantized-down and
dense-injection pair therefore remained separate. The latest measured source adds compatible
dense GGUF binding packing and a distinct mixed-projection kernel. The dense table above
does not measure that mixed GGUF pair. Padding to 336 rows helped some larger cases but lost to 324 rows for one token,
so no model-specific padding policy was added.

The generic up-and-mix kernel retained the previous rotating-weight timing, approximately
19.5 microseconds. Smaller warp subgroups and register-retained residual-combine
prototypes did not consistently improve the relevant workloads. Adaptive compact
prefill was neutral for many 128/512-token cases and 1.31 to 1.41 times faster for the
tested balanced and single-hot 2,048-token routes. Aggregate samples and rejected
experiments are recorded in [kernel-summary.json](raw/kernel-summary.json).

## Additional shared changes in the latest measured source

The new `layers/router.rs` helper sends eligible CUDA BF16/F16 single-token dense
routers through the existing GEMV kernel. Nine Linear-based router implementations
share it: Qwen3MoE, Qwen3Next, Qwen3VL-MoE, Qwen3.5-MoE, Qwen4Exp, Flint, LFM2,
GPT-OSS and Gemma4. It retains the original F32 and multi-token paths, applies bias
after output rounding, and preserves each caller's LoRA delta. Existing QuantMethod
routers keep their established dispatch. GEMV read guards now track the actual launch
stream for inputs, weights and bias.

At the tested 512-by-2,560 BF16 router geometry, rotating-weight latency changed from
12.570 to 11.459 us; cached latency was 3.603 versus 3.585 us. That isolated difference
amounts to about 0.053 ms across 48 routers, rather than the full router cost seen in
a profiler. Independent graph and close-tie checks passed; new repository tests cover
bias rounding, views, cross-stream weights and replay. Their coordinated build/run is
recorded separately from the historical 35-test suite.

GGUF sources now retain their open file handles and track which source tensors were
successfully copied to CUDA, including transformed and sharded bindings. A shared
weight-source hook runs after loading and context synchronization. On Linux it drops
mapped page-table entries and requests eviction of complete interior file-cache pages
for those uploaded sources. Untouched raw mapped tables remain available; optional
embedding prefault follows this reclamation. This changes memory residency, not weight
values or precision. Reload-after-unlink and shared-binding/untouched-neighbor tests
cover retained file ownership and source availability.

Indexed GGUF experts also gain cooperative asynchronous copies into shared memory.
This path uses the existing eleven-format vector-dot helpers and preserves activation
quantization and aggregation. It applies to SM121 single-token requests when the
relevant expert bank exceeds the device's queried L2 size, the copy is at least
four-byte aligned, and each staged operand fits the 24 KiB limit. Copy alignment
selects eight- or sixteen-row tiles. Cache-fitting banks, two-token requests and
unsupported geometry retain the previous kernels.

| Isolated expert operation, rotating weights | Previous, us | Staged, us |
|---|---:|---:|
| Q4_K gate, 640 x 2,560, top-10 | 88.265 | 81.501 |
| Q5_1 down, 2,560 x 640, top-10 | 56.094 | 52.495 |
| Q4_1 down, 2,560 x 640, top-10 | 47.367 | 42.704 |
| Q8_0 down, 2,560 x 640, top-10 | 78.112 | 73.577 |

A bank larger than L2 predicts locality imperfectly: repeatedly resident selected
experts showed 20% to 27% down-projection regressions. The standalone gain estimate of
roughly 0.5 ms/token is not an isolated full-model gain. The 558 staging cases
included eleven formats, partial rows, changed inputs/routing and three replays per
case. Gate outputs were bitwise equal; maximum down relative RMS error was
5.74e-8. All three added production staging tests subsequently passed in the corresponding
validation batch; the intermediate serving run measures these three shared changes together.

Implementation scope, dispatch limits and status are in
[resumed-development.json](raw/resumed-development.json),
[router-final-summary.json](raw/router-final-summary.json) and
[quant-resume-staging-summary.json](raw/quant-resume-staging-summary.json).

## Shared attention and residual optimizations

Source `4ac5bbc7...` and release binary `2f66d0a6...` contain the additional shared decode
optimizations below. Their completed same-GGUF result is retained in the preceding
source table. These changes remain in the latest measured paired-projection source.

A shared cuTile split-K attention kernel now supports rectangular decode and verification
with BF16/F16 KV caches, head dimensions 128/256 and grouped query heads up to 16.
It reads live device context lengths and selection metadata, handles partial tails and
uses an SM-count-derived split policy. Q/K/V retain their original dtype; softmax and
partial reduction use F32, with three dtype components representing probabilities before
the value MMA. Unsupported layouts, dtypes, pointer alignment and indexing ranges retain
the existing CUDA path. Independent higher-precision references, poisoned unused pages,
changed-metadata graph replay and cross-stream tests passed.

The shared HC combine kernel uses aligned four-element accesses and retains the rounded
combined residual in registers for the next normalization. Runtime dimensions and
capacity/alignment checks select it; the previous kernel handles the remaining cases.
The shared expert gate uses the existing scalar GEMV and a dtype-exact sigmoid, multiply
and residual-add operation across Qwen3Next, Qwen3.5 MoE and Qwen4Exp. Its separate
rounding points, broadcast fallbacks and LoRA ordering remain intact. F32 row tiles reuse
the sigmoid when beneficial; BF16/F16 retain the measured faster elementwise variant.

A generic GGUF helper also quantizes a common activation once for multiple compatible
projections while retaining each weight's established CUDA geometry. It supports all ten
existing dense formats, mixed-format groups, BF16/F16/F32 activations and one through
eight input rows. Explicit fused GLU/QKV/pair paths retain priority. GDN QKV/Z uses this
shared capability; unsupported methods and larger input batches keep their existing paths.
Sparse selection separately gains staged score loads and a parallel integer histogram
prefix, preserving its score threshold and eligible boundary-tie pool.

| Standalone operation | Previous, us | Candidate, us |
|---|---:|---:|
| BF16 dense QSA attention and reduction, rotating KV, context 640 | 33.078 | 15.658 |
| BF16 sparse QSA attention and reduction, rotating KV, context 2,176 | 49.998 | 29.840 |
| BF16 HC combine and normalization, rotating inputs, width 2,560 | 5.895 | 3.265 |
| BF16 scalar shared gate and residual addition, width 2,560 | 8.901 | 2.305 |

These are isolated measurements, not additive model gains. The QSA figures come from
selected prototype split variants; the integrated runtime split policy is independently
tested; the complete serving result measures all new changes together. Corrected QSA and HC cold controls rotate at least
128 MiB of actually visited inputs. The shared-gate benchmark includes the projection and
addition expression; its prefill controls also improved at 17, 512 and 8,192 rows.
Detailed scope and references are in the [evidence bundle](raw/attention-optimization-evidence.md).

That release CLI build, CUDA/CPU `cargo check`, formatting and whitespace checks
passed. Its suite has 74 distinct passing tests: 52 targeted CUDA/host tests and 22 additional
CPU archive/binding tests. Earlier test batches overlap and are not added. One new HC
test initially exposed CPU F32 reference accumulation error on a 16,384-wide constant
row; an independent FP64 oracle confirmed the CUDA result, and the reference was corrected
without changing the kernel or tolerance. The seven-test HC group then passed.

| Previous attention-optimization provenance | SHA256 |
|---|---|
| Source | `4ac5bbc7122f69a3391861b0208879e7aa45125ffe94db4b77703cac77c4040c` |
| Release binary | `2f66d0a61e4070e86a761fd01b59c98668837f868305e6b838fe1ff2035db23e` |

The [build and exact test record](raw/attention-build-validation.json),
[source manifest](raw/attention-source-manifest.json) and
[serving command](raw/mrs-attention-8192-command.json) identify this measured release.
The paired-projection development batch described below is not part of this measured build.

## Paired projections in the latest measured source

The latest measured source combines eligible GGUF and dense projections of one input in
one CUDA grid. It preserves the existing vector-dot arithmetic, dense FMA/reduction,
activation quantization and output dtype. All ten dense GGUF formats and BF16/F16/F32
are supported, with runtime dimensions and either operand order. Unsupported layouts,
odd quantized row counts, bias, statistics, active LoRA and nontrivial collectives retain
the established paths. A separate forward-weight capability preserves runtime output
order when canonical export order differs.

The ordinary mixed kernel beat separate projections in all 134 cached/rotating
comparisons, and 1,620 standalone projection comparisons were bitwise equal. At the
representative Q8_0 320-plus-4 by 10,240 shape, BF16 rotating-weight latency changed
from 21.420 to 18.476 us. A further mixed-only prefetch policy uses row payload and
queried device occupancy on SM121; it retains measured cached-versus-rotating tradeoffs
and does not change standalone GGUF dispatch.

The batch also enables common dense-source packing for compatible GGUF bindings and
fixes existing GEMV vector-alignment eligibility and statistics collection before its
early return. The release CLI build, CUDA/CPU checks and 95 distinct tests passed:
72 targeted CUDA/host tests and 23 additional CPU archive/binding tests. The additional
CUDA check with optional cuTile disabled also passed, along with final formatting and
whitespace checks. The [final snapshot](raw/final-source-verification.json) confirms that the
source and release binary matched the measured artifacts at benchmark completion. The fresh canonical sweep
measures this batch together; isolated kernel gains are not added to predict throughput.
All earlier canonical tables retain their original provenance.
The [paired evidence bundle](raw/paired-optimization-evidence.md) records the exact
scope, timing controls, completed tests and raw aggregate results. Source `24b72dc9...`
and binary `6681778b...` are identified by the [manifest](raw/paired-source-manifest.json)
and [build/test record](raw/paired-build-validation.json).

| Latest measured release | SHA256 |
|---|---|
| Source | `24b72dc9f7a4bab0a215fbe87543ad0a7c23ad20b3bd856590d313d950c1ac77` |
| Release binary | `6681778b528263db9c729a69c98f74c0f1a8327f4d337d04cd3c79a7485b7ec5` |

## Further decode experiments

Two independent cuTile alternatives were compiled and measured with Rust cuTile 0.3.1
and CUDA 13.2 on SM121. Both retained original Q8_0/Q8_1 weight bytes and Q8_1
activation quantization; neither introduced BF16 weight dequantization or new
quantization. The first compared SIMT integer block reductions with batched integer
tensor-core MMA and split-K. The second fused a short-rank projection, projection-output
rounding, sigmoid and residual-stream reduction in parallel stream tiles.

| cuTile experiment | Existing hot, us | Best cuTile hot, us | Existing rotating, us | Best cuTile rotating, us |
|---|---:|---:|---:|---:|
| Projection, 320 x 10,240 | 3.926 | 9.471 | 17.656 | 30.707 |
| Gated mix, 4 streams x 2,560 hidden x 320 rank | 6.085 | 12.248 | 19.858 | 24.599 |

Each experiment passed 48 independent correctness cases, including Q8_1, odd widths
and multiple runtime geometries. All measured cuTile schedules were slower. These
prototypes were rejected before expanding format decoding or adding production paths;
there is no Q8-only cuTile production policy. The best hot and rotating entries can
come from different schedules. Full aggregate results are in
[cutile-gguf-summary.json](raw/cutile-gguf-summary.json) and
[cutile-gatedmix-summary.json](raw/cutile-gatedmix-summary.json).

Programmatic dependent launch experiments checked all eleven formats, changed-input
and changed-weight graph replay, delayed producers, poisoned scratch, fallback launches
and generated instruction ordering. They produced some cached wins but inconsistent
cold-weight results; the final ordering strategy regressed cold Q4_K by 7.58%.
No PDL variant was integrated. Dense multirow panels likewise produced small or
inconsistent gains across Q8_0 and Q4_K neighbors, including only about 1.1% on the
Q8_0 output head, so existing dense dispatch remains.

CUDA parallel-stream gated mixing was slower at the representative shape. Raw-panel
staging reduced rotating-weight latency from 19.814 to 18.081 us, but the same variant
increased cached latency from 6.570 to 8.030 us and regressed other geometries. It was
also rejected. These decisions and controls are retained in
[pdl-summary.json](raw/pdl-summary.json), [dense-panel-summary.json](raw/dense-panel-summary.json),
[gated-mix-parallel-summary.json](raw/gated-mix-parallel-summary.json) and
[gated-mix-staged-summary.json](raw/gated-mix-staged-summary.json).

## Intermediate source validation

This section records the completed `464e5a6f...` benchmark source. The latest measured source
and its 95-test validation are identified separately above.

The additional cache, router and expert-staging changes passed the CUDA release CLI
build, CUDA `cargo check`, CPU `cargo check`, formatting and whitespace checks. The
refreshed suite passed 41 targeted CUDA/host tests plus 22 additional CPU archive and
binding tests, for 63 distinct tests. This includes the three new router tests, three
staging tests and two CUDA cache-reclamation tests. The earlier 35-test batch overlaps
this coverage and is not added to that count. The CUDA 13.3 NVFP4 provider test remains
excluded from this CUDA 13.2 build.

| Intermediate build provenance | SHA256 |
|---|---|
| Source | `464e5a6f7001b3d52d15548fca3a91d8e7f59a8d20c79fb70bbdbe1c0b81150a` |
| Release binary | `81d1e23b618ef09bdd8b183c420b0b4a3999c164b1385832c288cbcfce36df65` |

The intermediate build's commands, exact test names and outcomes are in
[resumed-build-validation.json](raw/resumed-build-validation.json). The fresh same-GGUF 8,192-token
prewarm and measured sweeps also completed 24 requests without errors. Historical
full-model results remain separately identified.

## Historical validation

The historical measured source passed the CUDA CLI build, CUDA `cargo check`, CPU `cargo check`,
and the refreshed core/quant test build. The targeted suite passed 33 tests, including
CUDA execution, plus two CPU shared-activation tests, for 35 total.

Coverage includes all eligible quantization formats; independent Q8_1 arithmetic;
changed-input and changed-routing CUDA graph replay; odd QKV rows; compact expert
partial tiles; resident and mapped embeddings; unaligned storage; offset and strided
packed projections; cross-stream ownership; GDN prefill; and scored-topk attention
with poisoned unused cache pages. The NVFP4 provider-specific test was not compiled:
its existing CUDA 13.3 minimum excludes this CUDA 13.2 build. It is not counted as a pass.

Earlier development binaries encountered a first-use NaN during the 2,048-token sweep.
Four GDN `ldmatrix` instructions now explicitly specify the `.shared` address space,
correcting their PTX contract. Both original and corrected standalone poisoned-memory
cases passed, so the change's causal relationship to the earlier NaN is unproven.
The temporary finite-stage diagnostic was removed before the historical release build.
That binary passed fresh-server full prewarm and measured sweeps at both GGUF chunk
sizes and with native quant-4: 72 completed requests across three mistral.rs processes.

Exact test names, counts and commands are in [validation.json](raw/validation.json).
These checks establish the tested arithmetic and execution behavior; they are not a
perplexity or model-quality evaluation, nor proof of identical generated text between engines.
Greedy outputs differ between engines, which also changes expert routing during decode.
The results compare fixed-prompt, fixed-length serving rather than identical-token,
teacher-forced decode. The [fairness audit](raw/benchmark-fairness-audit.json) records
these limitations. mistral.rs also loads the vision projector, while llama.cpp serves
text only; no vision inference is included, but the projector consumes extra memory.

## Historical method and provenance

- Hardware: NVIDIA GB10, compute capability 12.1, aarch64 host, 128 GB unified memory,
  driver 580.126.09 and CUDA 13.2. One serving process used the GPU at a time. Profilers
  and compilers were absent during canonical measurements.
- Fixture: four deterministic code-review prompts at each of 512, 2,048 and 8,192
  tokens. The first request at each length is a warmup; the remaining three supply the
  median. A full fixture sweep precedes measurement to warm loading and graph paths.
- Sampling: greedy temperature zero, EOS ignored, 128 requested output tokens, no
  prefix/prompt caching, and exact concurrency one.
- Transport: llama.cpp receives token IDs; mistral.rs receives text after an exact
  tokenizer round-trip assertion. Server-reported counts must match the fixture and
  requested completion length. Every canonical request produced 128 nonempty chunks.
- Metrics: prompt tokens divided by TTFT; decode throughput is 127 divided by the
  interval from first to last output chunk. These include serving overhead. Startup is
  excluded. Only three measured requests per length are used, with no confidence interval.
- Checkpoint: `unsloth/Qwen3.8-Flash-Next-GGUF`, revision
  `38bb39ee97821de2c9009abb7e93950eec396e66`, four UD-Q4_K_XL shards. The source
  tokenizer is from `Qwen/Qwen3.8-Flash-Next` revision
  `de4b8e4d43b917e7706784d8bb445c9af86a3540`.

| Component | Revision or digest |
|---|---|
| mistral.rs base commit | `142bc95a506512eb21d9458bce73400e6ec9f335` |
| Historical measured source SHA256 | `b55a21dbf070239b30336520810c85906eb2ab69a9e31c51742afd68d005a965` |
| Historical release binary SHA256 | `eca4ba522dc65558ab91800341ae9459d53f0dc36115c4b685444f2a09898b93` |
| Candle | `66a8cf184a5a519671454066b1b9efd446ec9f5c` |
| llama.cpp | `7aca0d2ea8d92b56911233b822db01f9fbb1ae50` |
| vLLM, source review only | `3d81a510f9ebc802a7864ced38832182295f4cad` |
| Rust / LLVM | `1.99.1` / `23.1.1` |

The base commit alone does not identify the measured worktree. The
[source manifest](raw/production-source-manifest.json) hashes every tracked or unignored
new file outside `releases/`, omitting deleted files, and records the aggregate source
digest and exact release binary. This report does not use the earlier development binary's
profiled or incomplete serving results as a performance baseline.

## Reproduction

Build the source matching the manifest with the command used for the measured binary:

```bash
cargo --config profile.release.package.candle-core.codegen-units=1 \
  build --release -p mistralrs-cli --features cuda,flash-attn,cutile -j10
```

The initial release build hit a reproducible LLVM 23.1.1 `SIGTRAP` in
`VPDerivedIVRecipe::execute` while compiling Candle. Setting only Candle's codegen units
to one avoided the compiler failure. Optimization level 3, native CPU targeting and
vectorization remained enabled; no dependency source patch or workspace profile change
was required. The isolated Candle reproduction passed, followed by the complete release
build. Commands and outcomes are retained in [build-workaround.json](raw/build-workaround.json).

Start the mistral.rs server using the exact argument array in
[mrs-paired-8192-command.json](raw/mrs-paired-8192-command.json). Historical engine
commands remain in [commands.json](raw/commands.json); adjust model and binary paths
for the local machine. For the 512-token cells,
llama.cpp uses `-b 512 -ub 512`; mistral.rs uses
`--max-num-batched-tokens 512 --max-prefill-chunk-tokens 512`. The larger cells change
both engines' limits to 8,192. The shared benchmark runner is
[`scripts/bench_c1_serving.py`](../../scripts/bench_c1_serving.py).

For llama.cpp, run the command below once to warm the fixture and again to collect a
separate canonical result:

```bash
python3 scripts/bench_c1_serving.py \
  --url http://127.0.0.1:1235 --model qwen38 \
  --prompts releases/v0.9.5/raw/prompts.json \
  --lengths 512 2048 8192 --warmup 1 --repetitions 3 --output-tokens 128 \
  --output /tmp/llama-c1-serving.json
```

For mistral.rs, use port 1234, model name `Qwen/Qwen3.8-Flash-Next`,
`--mistralrs-metrics`, and `--text-tokenizer` pointing to the pinned tokenizer JSON.
The native cell uses the separately recorded command with `--quant 4`, without the
GGUF shard or projector arguments; retain its quantization label when comparing results.
Use the committed prompt fixture rather than regenerating it from changed source files.
Regenerate and validate the stored medians with
`python3 releases/v0.9.5/scripts/summarize.py`.

## Artifacts

- [run-manifest.json](raw/run-manifest.json): hardware, workload, checkpoint, metric
  definitions, canonical cells and software identifiers.
- [summary.json](raw/summary.json): validated medians, matched-chunk ratios and independently
  selected best recorded llama controls for each prompt length and metric.
- [prompts.json](raw/prompts.json): exact token-ID fixture.
- Canonical `*-serving.json` files: individual request timings, output text, token
  timestamps, counts and warmup flags; prewarm files are retained separately.
- [commands.json](raw/commands.json), [build-workaround.json](raw/build-workaround.json)
  and [validation.json](raw/validation.json): reproduction and validation evidence.
- [kernel-summary.json](raw/kernel-summary.json): isolated kernel results supporting the historical measured source.
- [resumed-development.json](raw/resumed-development.json): later implementation scope,
  isolated evidence for the intermediate measured source, kept separate from canonical timing samples.
