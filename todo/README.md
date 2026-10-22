# todo

Working notes for issues found while adding Kolibri-1 support to the CPU path. Each file is
self-contained enough to be filed upstream on its own, with the evidence that produced it.

Nothing here is a proposal yet. Each entry has a status and needs a decision before any of it
is sent anywhere.

## Artifacts

`artifacts/` holds data and scripts, not issue files. `artifacts/llamacpp-oracle/` is the
llama.cpp third oracle and the scripts that compare it against the other two implementations.

## Layout

| Directory | Upstream | Meaning |
|---|---|---|
| `mistralrs/` | this repo | defects and limitations in mistral.rs itself |
| `candle/` | huggingface/candle | defects in the pinned candle revision, reached through mistral.rs |
| `gaps/` | this repo | missing features, not defects |

One entry (`gaps/04`) is unrequested rather than found-in-the-wild; it is marked as such in
the index so it is not mistaken for a bug report.

## Open measurement tasks

Things decided but not yet done. Kept here rather than in a plan file so they do not get lost.

| task | why it is still open |
|---|---|
| Measure this build's chains against the llama.cpp goldens | The transcription and llama.cpp agree on 7 of 8 full sequences, so the transcription number should carry over - but that is an inference. The model needs ~150 GB resident and the box was occupied when it came up. |
| A CPU-to-CPU per-layer dump, from llama.cpp | `mistralrs/09`. Ours against the peer agent's tt-metal build shows no structural divergence (layer 0 cosine 0.99999), but that separates devices as much as implementations. llama.cpp on the same CPU is the isolating measurement. |
| Re-sweep `MIN_PARALLEL_OUTPUTS` at 3.4 GHz | The 1024 -> 512 change was the largest single win of the session (2.35x) and is a work-partitioning threshold. Its optimum was found at 1.0 GHz and has almost certainly moved. Self-contained. Set the governor to `performance` for the run and put it back afterwards - `schedutil` is the normal state. |
| Re-test the MoE residency cache at 3.4 GHz | Measured -13% at 1.0 GHz. A memory-traffic optimisation is the kind of thing that can flip at full clock. One env var. Same governor discipline. |
| Compare generated chains against the primary oracle | `07` measures against the transcription. The primary comparison is the one that would settle whether BFMMLA converges. Blocked on the gate-schema question (logprobs vs logits). |
| Re-verify `mistralrs/06` end to end | The fix is reasoned from shapes, not run. The only FP8 checkpoint available is 74 GB on a shared host. |

## Index

### mistral.rs

| # | Issue | Status |
|---|---|---|
| 01 | [CPU FP8 dequantizes the whole weight on every forward](mistralrs/01-cpu-fp8-dequant-every-forward.md) | open, not started |
| 02 | [CPU GGUF MoE dequantizes every expert on every forward](mistralrs/02-cpu-gguf-moe-dequant-every-call.md) | open, not started |
| 03 | [CPU attention kernels ignore `sliding_window`](mistralrs/03-cpu-attention-ignores-sliding-window.md) | open, not started |
| 04 | [PagedAttention is disabled whenever CPU is in the device map](mistralrs/04-cpu-disables-paged-attention.md) | open, design decision |
| 05 | [17 clippy warnings block the CI clippy gate](mistralrs/05-clippy-double-must-use-blocks-ci.md) | open, trivial. Pre-existing on `master`; our own one (`kolibri1.rs` `type_complexity`) is fixed on this branch. |
| 06 | [CPU blockwise-FP8 stacked MoE does not run](mistralrs/06-cpu-fp8-stacked-moe-does-not-run.md) | fixed on the `cpu-decode-bandwidth` branch by `56044c0e9`, all three |
| 07 | [Kolibri-1 CPU forward does not reproduce the reference logits](mistralrs/07-kolibri1-logit-gap.md) | **cause found and fixed** (`d9198c578`): candle's fused RMSNorm rounds the scale to bf16 before dividing, which moved the router's selection. "The capital of Australia is" now answers "Canberra", matching every reference. The transcription is not a valid oracle - it computes the router in bf16. |
| 08 | [A default CPU build is not pure Rust](mistralrs/08-cpu-build-pulls-native-tls.md) | open. `aws-lc-sys` + `ring` + `cc` + `cmake` via `hf-hub`'s TLS, not the compute. |
| 09 | [The forward disagrees with the reference semantically](mistralrs/09-forward-disagrees-with-the-reference-semantically.md) | open. Not a position shift - the self-consistency test is 23/23 at offset 0. Two of six prompts have the reference's top-1 ahead by more than 5 nats, so this is semantic. The previous revision of this entry claimed a one-position shift; that was wrong. |
| 10 | [Flint (RecursiveSI) is unsupported](mistralrs/10-flint-is-unsupported.md) | open, not started. Not a new model family: Flint is the same MoE family as Kolibri-1 (sigmoid router, shared experts, sliding window 513, per-layer pattern) with attention sinks, a per-layer no-rope pattern, two shared experts, and a renormalised router. The GGUF route needs only the architecture, the bindings and the tokenizer variant - `lmstudio-community/flint-GGUF` publishes a K-quant-only build that candle can read. The FP8 checkpoint uses the same blockwise recipe and tensor names as Kolibri-1's, so the scoped gather applies unchanged. |

### candle

| # | Issue | Status |
|---|---|---|
| 01 | [CPU decode GEMV does not scale with thread count](candle/01-cpu-decode-gemv-single-threaded.md) | already reported as [candle#1103](https://github.com/huggingface/candle/issues/1103) and [gemm#20](https://github.com/sarah-ek/gemm/issues/20), both open since 2023. Our measurement is the addition to comment on them with, not a new issue. |
| 02 | ~~[`RAYON_NUM_THREADS` unset under `numactl` under-uses the pinned cores](candle/02-numa-affinity-ignored.md)~~ | **withdrawn.** The count is correct under pinning (32, matching `available_parallelism`); `linux_detect_preferred_cpus` returns `None` on a homogeneous host and the `num_cpus` fallback respects affinity. The observed symptom was a misattribution of 01. |
| 03 | [aarch64 packed MoE GEMV is serial](candle/03-aarch64-packed-moe-gemv-serial.md) | open, not yet reported. Verified still true at candle `12bb5d1`. |

### Gaps

| # | Gap | Status |
|---|---|---|
| 01 | [No EXL3 quantization support](gaps/01-no-exl3-quantization.md) | open, unrequested |
| 02 | [No OPT architecture](gaps/02-no-opt-architecture.md) | open, unrequested |
| 03 | [`qwen3_5moe` has no text-only loader](gaps/03-qwen3-5moe-has-no-text-loader.md) | open, unrequested |
| 04 | [No expert streaming, and the expert stack is materialized on CPU](gaps/04-no-expert-streaming.md) | open, unrequested. Now costed: the non-expert set is 4.85 GB, the routed experts 140.6 GB bf16, and 2.56 GB per token of traffic. A 32 GB box is I/O bound by two orders of magnitude, and the hit rate - not the cache size - is the unknown. The fix is `todo/02`. |

## Status vocabulary

- `open, not started`: understood and evidenced, no code written.
- `open, fixed on the `cpu-decode-bandwidth` branch`: the branch has a fix that has not been
  upstreamed, so the report still stands and the diff is available if it is wanted.
- `open, worked around`: understood, bypassed locally, upstream still unfixed.
- `open, trivial`: understood, small fix.
- `open, design decision`: needs a maintainer view before anyone works on it.

## Conventions

Every file carries the same shape: what, where, evidence, how it was found, impact, and a
proposed direction that is explicitly labelled as a suggestion rather than a prescription.
Measurements are pinned to the host described in `evidence-host.md`; numbers from other hosts
are marked as such.

Spellings follow the repository (American). No em dashes or other non-ASCII.

## Filing upstream

None of these are filed yet. Before sending anything:

- The candle entries go to `huggingface/candle`, and should say they were found from
  mistral.rs and are pinned to the revision mistral.rs uses. A maintainer reproducing them
  needs that rev, since two of the three are inside code paths that change between releases.
- The mistral.rs entries are written as reports, not patches. Attaching a proposed diff
  changes the conversation from "is this real" to "review this", which is a bigger ask to
  make first, especially for 02 and 04 where the right direction is not settled.
- Every performance claim should carry the host. On a different machine the absolute numbers
  will not reproduce, and a report whose numbers do not reproduce reads as unreliable even when
  the ratio is right.
- The two entries with the strongest evidence are 02 (178 threads, 1 running, 1.9 cores busy)
  and candle/03 (1062 lines, zero parallel constructs). Those are checkable by reading the
  code, which is worth leading with.
- `gaps/04` is unrequested and overlaps an open PR (#2380, sparse per-expert MoE on CPU).
  Check that PR before filing, and lead with "the stack is copied into anonymous memory"
  rather than "please add streaming", since the first is a statement about the code and the
  second asks for work nobody has scoped.

## Not here

The Kolibri-1 port plan itself lives in `kolibri1-support-plan.md` at the repository root. This
directory is only for things that are mistral.rs's or candle's problem, not ours.