# todo

Working notes for issues found while adding Kolibri-1 support to the CPU path. Each file is
self-contained enough to be filed upstream on its own, with the evidence that produced it.

Nothing here is a proposal yet. Each entry has a status and needs a decision before any of it
is sent anywhere.

## Layout

| Directory | Upstream | Meaning |
|---|---|---|
| `mistralrs/` | this repo | defects and limitations in mistral.rs itself |
| `candle/` | huggingface/candle | defects in the pinned candle revision, reached through mistral.rs |
| `gaps/` | this repo | missing features, not defects |

## Index

### mistral.rs

| # | Issue | Status |
|---|---|---|
| 01 | [CPU FP8 dequantizes the whole weight on every forward](mistralrs/01-cpu-fp8-dequant-every-forward.md) | open, not started |
| 02 | [CPU GGUF MoE dequantizes every expert on every forward](mistralrs/02-cpu-gguf-moe-dequant-every-call.md) | open, not started |
| 03 | [CPU attention kernels ignore `sliding_window`](mistralrs/03-cpu-attention-ignores-sliding-window.md) | open, not started |
| 04 | [PagedAttention is disabled whenever CPU is in the device map](mistralrs/04-cpu-disables-paged-attention.md) | open, design decision |
| 05 | [17 clippy warnings block the CI clippy gate](mistralrs/05-clippy-double-must-use-blocks-ci.md) | open, trivial |
| 06 | [CPU blockwise-FP8 stacked MoE does not run](mistralrs/06-cpu-fp8-stacked-moe-does-not-run.md) | open, fixed on the `cpu-decode-bandwidth` branch |

### candle

| # | Issue | Status |
|---|---|---|
| 01 | [CPU decode GEMV runs at one core's bandwidth](candle/01-cpu-decode-gemv-single-threaded.md) | open, worked around locally |
| 02 | [`RAYON_NUM_THREADS` unset under `numactl` under-uses the pinned cores](candle/02-numa-affinity-ignored.md) | open, workaround known |
| 03 | [aarch64 packed MoE GEMV is serial](candle/03-aarch64-packed-moe-gemv-serial.md) | open, blocks quantized MoE decode |

### Gaps

| # | Gap | Status |
|---|---|---|
| 01 | [No EXL3 quantization support](gaps/01-no-exl3-quantization.md) | open, unrequested |
| 02 | [No OPT architecture](gaps/02-no-opt-architecture.md) | open, unrequested |
| 03 | [`qwen3_5moe` has no text-only loader](gaps/03-qwen3-5moe-has-no-text-loader.md) | open, unrequested |

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

## Not here

The Kolibri-1 port plan itself lives in `kolibri1-support-plan.md` at the repository root. This
directory is only for things that are mistral.rs's or candle's problem, not ours.