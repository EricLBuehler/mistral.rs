# FlashAttention SM90 forward subset

This directory contains the transitive source closure needed by the FP8 paged decode provider.
It is derived from vLLM's FlashAttention fork at commit
`f3e1a4f74c99145c0717709860bf765de1703779` and is licensed under BSD-3-Clause.

The build pins NVIDIA CUTLASS commit `62750a2b75c802660e4894434dc55e839f322277`.
Local changes remove the Torch header dependency, return CUDA failures to the C ABI instead of
terminating the process, pass the CUDA device ordinal from the persistent decode plan, and retain
only the noncausal forward and BF16 hdim-256 combine instantiations used by this provider.
Trailing whitespace in the vendored source subset is normalized for repository checks.
