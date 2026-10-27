# Paired projection development evidence

This bundle records measured source `24b72dc9...`, following source `4ac5bbc7...`.
Release build, final CUDA/CPU checks and 95 distinct tests passed: 72 targeted
CUDA/host tests and 23 additional CPU archive/binding tests. The fresh 24-request
canonical sweep is complete. Its measured same-GGUF decode rates exceed the best historical or fresh
llama controls by 13.81%, 11.45% and 17.15%. Both fresh controls completed successfully.
The additional CUDA build check without optional cuTile also passed. Final source and
release binary hashes exactly match the measured artifacts.

The mixed GGUF/dense helper uses one CUDA grid with dense rows first, preserving the
existing dense vector2 FMA/reduction and quantized Q8_1 activation/vector-dot arithmetic.
It supports the ten established dense GGUF formats, BF16/F16/F32, both operand orders,
and runtime dimensions. It neither requantizes weights nor retains duplicate weights.

The ordinary helper passed 1,620 bitwise projection comparisons across formats, dtypes,
changed graph inputs and layouts. Its 134 cached/rotating comparisons all beat separate
production projections, with latency reductions from 4.74% to 35.15%. Odd-row standalone
fixtures used explicit padding; production declines odd quantized rows instead.

On SM121, the mixed prefetch policy additionally requires at least 8 KiB per quantized
row and enough row CTAs to fill the queried device thread capacity. This is 288 rows on
the tested GB10. Each supported format has eligible rotating-weight measurements.
Some cached cases are slower than ordinary mixed fusion; selected cases still beat the
original separate projections. Standalone dense dispatch is unchanged.

The same source batch enables compatible dense GGUF bindings in common packed-projection
factories and fixes existing GEMV alignment and statistics early-return behavior.
An explicit forward-order dense-weight capability keeps runtime row remapping separate
from canonical export order. The production test batch includes nonidentity row layouts,
all-format selected prefetch paths, changed graph replay, cross-stream ownership,
fallback guards, source transforms and exactly-once statistics.

Evidence:

- [Implementation and production test scope](paired-development.json)
- [Completed checks and exact runtime test results](paired-build-validation.json)
- [Source and release binary provenance](paired-source-manifest.json)
- [Final compatibility checks](paired-final-compatibility.json)
- [Unchanged final source and binary verification](final-source-verification.json)
- [Standalone validation summary](mixed-pair-standalone-summary.json)
- [Ordinary mixed-pair timings](mixed-pair-plain-timings.json)
- [Prefetch policy timings](mixed-pair-prefetch-timings.json)

All timing controls include input quantization. Rotating runs visit at least 128 MiB of
weight storage and alternate variants inside repeated CUDA graph event measurements.
These are isolated kernel measurements, not additive predictions of serving throughput.
