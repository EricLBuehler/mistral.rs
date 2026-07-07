# Reddit post draft (r/LocalLLaMA)

**Title (locked):** mistral.rs v0.9.0: up to 1.8x faster CPU decode than llama.cpp on x86 and ARM

**Image:** figures/pro_decode_vs_depth.png

---

Hey all! After the v0.8.2 CUDA release, several of you asked the right question: "cool, but
what about hardware I actually own?" So this release is entirely about CPU.

The result: mistral.rs now decodes faster than llama.cpp on CPU - on ARM and on x86 - and the
lead grows with context depth. At 16K context: 1.81x on a Xeon, 1.79x on ARM. Q4_K decode is at
or ahead of llama.cpp at every context length we measured, on both architectures.

The interesting part is the shape of the curve, not one number: our decode attention runs at
memory bandwidth (accumulators live in registers across each KV tile, V converts once per
position), so throughput degrades with context at the hardware's floor. That is exactly where
long chats and agent workloads live.

**Try it in 60 seconds** (no model files to hunt down - it quantizes on the fly from
safetensors, so any HF model id works directly):

```
curl --proto '=https' --tlsv1.2 -sSf https://raw.githubusercontent.com/EricLBuehler/mistral.rs/master/install.sh | sh
mistralrs run --cpu -m Qwen/Qwen3-4B --quant 4
```

Or an OpenAI-compatible server with a web UI: `mistralrs serve -m Qwen/Qwen3-4B --quant 4 --cpu`

**Numbers** (both engines built from source with native flags, llama.cpp at its best config
per point - including where fa=0 beat its own fa=1):

- gemma-4-E4B q4k prefill: 2.4-3.2x, decode 1.3x (ARM)
- qwen3-4B q4k decode: 1.06x -> 1.81x as depth grows 128 -> 16K (x86, Sapphire Rapids w/ AMX)
- qwen3-4B q4k decode: 1.09x -> 1.79x (ARM, GB10)
- LFM2.5-8B MoE: decode ahead at every depth (the CPU MoE path was rebuilt from scratch)
- Q8_0 (scheme-identical on both engines, for the quant-fairness folks): decode 1.02-1.33x ARM,
  ahead at depth on x86

**Honest caveats, before you find them:** prefill on AMX-equipped Xeons trails llama.cpp
(0.7-0.9x - their AMX path is mature, ours is a day old; most x86 lacks AMX). Two more amber
cells (MoE 8K prefill, q8_0 shallow x86 decode) are in the report with causes and fixes named.
These are single-request numbers; batched/serving benchmarks are next. Prebuilt binaries land
within ~8% of source builds.

Full report with every number, raw JSONL data, and reproduction scripts:
https://github.com/EricLBuehler/mistral.rs/blob/master/releases/v0.9.0/report.md

Reproductions, criticism, and benchmark suggestions are very welcome - especially on hardware
I have not tested (Ryzen, M-series, Graviton).

---

# Prepared replies (do not post preemptively)

**"This is one exotic ARM box"** -> Second architecture is a stock Xeon (c7i.8xlarge anyone can
rent for $1.40/hr); exact instance type, commands, and raw logs in the report. Ryzen/M-series
reproductions welcome - the kernels runtime-dispatch (AVX512-VNNI / AVX-VNNI / AVX2 tiers).

**"ISQ q4k vs Q4_K_M is not the same quant"** -> Correct, and disclosed in the report. That is
why the Q8_0 rows exist - identical scheme both engines, still ahead. Perplexity table is a
fair ask; on the list.

**"Did you tune llama.cpp properly?"** -> We ran an affinity/threading matrix for it, verified
fa=auto against fa=0 at every depth, and report whichever won per point (their flash-attn CPU
kernel inverts at depth on x86 - fa=0 is faster there; we report that number, not the weaker
default). Build: GGML_NATIVE=ON, pinned commit in the report.

**"What about batched serving / n=8?"** -> Single-request release; continuous batching exists,
serving benchmarks are the next report. This one is about the curves.

**"Numbers don't reproduce on my machine"** -> Which binary? Prebuilt installers are within ~8%
of the table (portable builds); the repro scripts build from source and match the table. Post
your `mistralrs bench` + `llama-bench` lines and I will look.

**"Mainline llama.cpp CPU is known-slow, compare against ik_llama.cpp"** -> We did (bbc7de4,
same box, same GGUF, its best fa config per point). x86 qwen3-4b Q4_K decode, ik vs us:
d512 32.9 vs 32.4 (tie), d2048 28.2 vs 29.2, d8192 18.0 vs 21.5 (+19% us), d16384 12.2 vs 15.9
(+30% us). ik's prefill is excellent (391 t/s pp512, beats us - its GEMM kernels are the best
in the family and it nearly matches mainline's AMX path without AMX). Its flash attention is
far better than mainline's (12.2 vs 8.8 at depth) but the depth curve still diverges in our
favor. Could not build ik on aarch64 (HEAD does not compile on gcc 13 / Ubuntu 24.04; three
separate errors). The comparison in the post targets mainline because that is what people run.

**gusbags-class perf bug reports** -> collect exact model/quant/flags + the log header
(DType/paged attn lines) and file an issue; the Spark NCCL 8 t/s report from the v0.8.2 thread
is still on the follow-up list.
