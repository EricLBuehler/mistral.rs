# Dense FFN backend probe

Actual layer-0 Qwen3.8-27B-FP8 weights; hidden 5120, intermediate 17408. Each timing covers gate/up, SiLU product, and down projection through the production dispatch and fusion hooks. FP8 uses packed gate/up; immediate Q4K uses separate weights with fused computation.

Q4K is derived from the same FP8 checkpoint after dequantization. This double-quantized experiment tests backend timing, not model quality. Both paths passed checks against their own dequantized-weight FP32 references.

| Rows | Default FP8 ms | Default Q4K ms | Affine-on Q4K ms | Default / affine Q4K |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 1.2610 | 0.8074 | 0.7799 | 1.035x |
| 6 | 1.3997 | 0.9290 | 0.8779 | 1.058x |
| 8 | 1.2737 | 1.0728 | 0.7386 | 1.453x |
| 24 | 1.3505 | 0.8994 | 0.7361 | 1.222x |
| 32 | 1.2185 | 0.9379 | 0.7341 | 1.278x |
| 40 | 1.3440 | 0.9323 | 0.7504 | 1.242x |
| 56 | 1.3303 | 0.9249 | 0.7991 | 1.157x |

Default M1-to-M8 aggregate row-throughput scaling is 7.92x for FP8 and 6.02x for Q4K. The optional affine run prepares all three Q4K projections from M8 onward and reaches 8.45x scaling; M1 and M6 still use MMVQ. The default FFN probe is consistent with a partial backend contribution to the full-model scaling difference.

Source-dispatch evidence: FP8 uses its eight-column MMA path through M32, then cuTile above M32. Default Q4K uses MMVQ through M8 and MMQ above M8. The optional `MISTRALRS_GGUF_AFFINE_BACKEND=on` run records successful affine preparation at M8 and above. These are checked dispatch branches, not measured kernel traces.

Seven alternating rounds, ten iterations per round, three warmups; all raw stream/host intervals, errors and backend preparation flags are retained. Default and affine-on run in separate fresh processes. Compilation and model serving were absent during both probes; editor processes remained stopped. Release build, relevant release `cargo check`, and rustfmt passed.

The inputs are synthetic BF16. Upstream fused activation quantization, graph replay, model-wide cache behavior, speculative acceptance, and serving overhead are outside this test. CUDA-event intervals include CPU launch gaps. Repacking setup costs are excluded. These layer timings do not predict a full-model affine speedup.

Reproduction: build with the command in `comparison.json`, then run the archived controller with `--binary <test-executable> --output-dir <new-directory>`. Repeat in a fresh process with the explicit affine environment override. The controller uses the pinned local snapshot recorded in provenance; update its paths on another machine. `build/` contains the successful build/check logs; each run preserves source, runner, results, provenance and an inner manifest. External originals remain under `/home/ericbuehler/qwen4exp_work/kernel_comparison_20260930/`.
