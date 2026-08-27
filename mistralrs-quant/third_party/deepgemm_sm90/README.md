# DeepGEMM SM90 provider

The official kernel family under `include/official/deep_gemm` is from
DeepGEMM 2.6.1 commit `559d79fb6994a58b8a15b4b93bf13ccc16edf247`.
Its pinned CUTLASS commit `f3fde58372d33e9a5650ba7b80fc48b3b49d40c8`
is fetched by cudaforge during the build. The Rust build generates an embedded
runtime header bundle, so serving does not depend on Python, vLLM, or a
machine-local CUTLASS checkout.

The headers directly under `include/deep_gemm` are the previous TensorRT-LLM
DeepGEMM subset distributed by FlashInfer at commit
`4927c0e15cb63a2abb6df09019c39a172222f0eb`. Production selects its swap-AB
kernel with BN=8 or BN=16 WGMMA tiles for measured small-M projection shapes.
The official SM90 1D2D family handles every other shape and uses measured
small-M tiles where they beat its cost heuristic. The remaining TensorRT-LLM
kernel paths stay behind `MISTRALRS_DEEPGEMM_ENABLE_LEGACY_DIAGNOSTICS` for
internal family A/B benchmarks and regression diagnostics.

The imported files retain their upstream SPDX and copyright headers. See
`NOTICE`, `LICENSE-APACHE`, `LICENSE-MIT`, and `LICENSE-BSD-3-CLAUSE` for
provenance and license terms. Files carrying a mistral.rs modification notice
differ from their pinned source.

The local integration replaces upstream process exits with status-based error
propagation, uses an owner-private XDG cache, materializes its JIT header bundle
from the binary, accepts caller-owned workspace, and prepares immutable kernel
handles before CUDA graph capture. The provider accepts BF16 activations or
prequantized E4M3 activations with native padded column-major scales, E4M3
weights, FP32 128x128 weight scales, and BF16 output on SM90. The official SM90
1D2D kernel consumes arbitrary FP32 scales directly; packed E8M0 scales are not
substituted for that interface. Official launches enable PDL by default and
`MISTRALRS_DEEPGEMM_PDL=0` is available for diagnostic comparison.

To update either snapshot, copy the corresponding `deep_gemm/include/deep_gemm`
tree, update the pinned commits and licenses, then run the provider selection
and smoke tests, dense serving-shape parity and graph tests, and the matched
family benchmarks.
