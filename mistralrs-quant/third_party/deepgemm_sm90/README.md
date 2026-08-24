# DeepGEMM SM90 provider

The headers under `include/deep_gemm` are the minimal TensorRT-LLM DeepGEMM
subset distributed by FlashInfer at commit
`4927c0e15cb63a2abb6df09019c39a172222f0eb`. The provider ABI, tests, and
BF16-to-FP8 activation quantizer live in this directory so the integration is
independent of a system FlashInfer installation.

The imported files retain their upstream SPDX and copyright headers. See
`NOTICE`, `LICENSE-APACHE`, and `LICENSE-MIT` for provenance and license terms.
Files carrying a mistral.rs modification notice differ from that pinned source.

The local integration replaces upstream process exits with status-based error
propagation, uses an owner-private XDG cache, materializes its JIT header bundle
from the binary, accepts caller-owned workspace, and prepares immutable kernel
handles before CUDA graph capture. The provider contract is BF16 activations,
E4M3 weights, FP32 128x128 weight scales, and BF16 output on SM90.

To update the vendor snapshot, copy only the headers reachable from
`fp8_gemm.cuh` at the new pinned FlashInfer revision, preserve upstream headers,
reapply and mark integration changes, update `NOTICE` and this revision, then
run the provider smoke test, dense serving-shape parity and graph test, and the
matched CUTLASS/DeepGEMM production-shape benchmark.
