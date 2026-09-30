# Captured routing: interpretation notes

This review covers 26 native target-layer-8 captures and 24 completed requests.
All requested outputs contain 128 tokens; prompt counts agree across C1/C6/C8.
Independent checks reproduced each expert occupancy vector from saved U32 IDs,
verified distinct in-range top-10 selections, finite nonnegative routing weights,
and small-file hashes. The largest selected-weight sum error was 1.53e-7.
The pinned config hash matches the architecture comparison. Full input/output
finiteness is separately established by the original capture validator.

| Captured call | Samples | Assignments | Active experts | Mean rows per active expert | Largest expert group |
| --- | ---: | ---: | ---: | ---: | ---: |
| C1 decode, 1 row | 8 | 10 | 10 | 1.00 | 1 |
| C1 verification, 7 rows | 8 | 70 | 25-38 | 1.84-2.80 | 6-7 |
| C6 verification, 42 rows | 5 | 420 | 107-195 | 2.15-3.93 | 11-30 |
| C8 verification, 56 rows | 5 | 560 | 132-219 | 2.56-4.24 | 16-45 |

Ranges describe these captured calls, not a population estimate. C1 takes the
first matching shape for each of eight prompts. C6/C8 capture shape visits
1, 5, 9, 13 and 17. The diagnostic disables CUDA graphs, fixes six MTP proposals,
and synchronizes capture writes. Its requests are not throughput benchmarks,
and its rows include draft verification work rather than accepted output count.

The C8 calls spread 560 assignments over 132-219 experts. Most active groups are
small, but some experts are hot: 5-17 experts per C8 sample receive more than
eight rows, with maxima of 16-45. This records uneven matrix reuse within the
routed path. It does not establish bandwidth saturation, GPU occupancy, or the
cause of end-to-end scaling. Every native C6/C8 sample exceeds an eight-row
expert bound, so the earlier synthetic bound-of-eight control cannot be applied
to these routes without excluding real work.

The hook captures only the routed expert contribution before the shared expert
is added. It excludes attention/GDN, PLE, hyper-connections, shared-expert work,
scheduling, graph coverage and speculative acceptance. Replay reuses exact
post-ISQ Q4K gate/up and Q4_1 down weights, BF16 inputs, IDs and routing weights.

Numerical comparison between alternate kernels is separate from reproducing the
captured native result. The first replay stopped at a 3.3298% relative-RMS
cross-kernel difference under an old 3% guard; that alone identifies neither
kernel as incorrect. No timing or numerical-quality conclusion is included
here. The follow-up records cross-kernel differences, retains exact native
reproduction, and uses an independent dequantized-weight FP32 reference.
