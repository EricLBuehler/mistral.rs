# mmap-backed GGUF loading for quantised tensors

- **Repo:** `huggingface/candle`
- **Pinned rev:** `66a8cf184a5a519671454066b1b9efd446ec9f5c` (v0.11.0)
- **Status:** PR open. https://github.com/huggingface/candle/pull/4061
  (`lu-zero:mmap-quantized-gguf` onto `main`, commit `ff6c630`, base `12bb5d1`).
  No issue comment. mistral.rs pin not bumped.
- **Origin:** positive feedback on candle issue #3708 ("No lazy/mmap loading for GGUF
  weights"); the maintainer called the `QStorage` `Cow` route the smallest first step.
- **Impact:** every candle GGUF load copies the whole checkpoint into anonymous memory.

## Why it matters here

mistralrs's K-quant GGUF route maps the shard and hands `QStorage::from_data` a
`Cow::Borrowed` window (`mistralrs-quant/src/gguf/archive.rs`), and candle copies it into
a `Vec` anyway, because `QStorage::Cpu` is a `Box<dyn QuantizedType>` and `'static` kills
any borrow. For the 25 GiB Flint Q4_K_M checkpoint that is 25 GiB of anonymous memory on
top of the page cache. The IQ route already dodges this by owning its own `Arc<Mmap>`
(`gguf/iq.rs`); a candle-side fix removes the copy for every K-quant/dtype too.

## The patch

One mechanism, three layers, 5 files, ~370 insertions:

1. `quantized/mod.rs`: private `enum Blocks<T>` (`Owned(Vec<T>)` / `Mapped { Arc<Mmap>,
   start, len }`) replaces `impl QuantizedType for Vec<T>` (one impl, no forwarding
   duplication). `GgmlDType::from_mapped` + `QStorage::from_mapped` validate whole blocks,
   alignment, and range, and box the mapped arm. `cpu_zeros`/`from_data` wrap in
   `Blocks::Owned`. `QuantizedType::from_float{,_imatrix}` now return `Result<()>` since
   mapped storage is read-only; the four `QStorage::quantize*` call sites propagate. This
   signature change is the only breaking bit (no other impl exists in-repo).
2. `quantized/gguf_file.rs`: `TensorInfo::read_mapped`, `Content::tensor_mapped`,
   `Content::from_mmap` (`unsafe`, mirroring `MmapedSafetensors::new`). Non-CPU devices
   still copy via `qtensor_from_ggml` (the upload needs the bytes anyway).
3. `candle-transformers/src/quantized_var_builder.rs`: `VarBuilder::from_gguf` maps the
   file once and hands out mapped tensors; each `QTensor` holds an `Arc` clone of the map.
   `from_gguf_buffer` untouched.

Tests: three new in `candle-core/tests/gguf_tests.rs` - mapped equals copied (bytes and
dequantised values), window validation errors, and `QMatMul::forward` bit-identical
through the repack path. Suite green (209+13), clippy and fmt clean.

## Measured, 2026-10-23

Throwaway `/tmp/gguf-mmap-measure`, candle tree `/tmp/candle-for-grok`, file
`/mnt/models/flint-GGUF/flint-Q4_K_M.gguf` (24166841184 bytes, the checkpoint
mistral.rs loads through `qtensor_from_gguf_data`). aarch64, `schedutil`, loadavg
3.68, caches not dropped, no cgroup. `/proc/self/smaps_rollup`:

| snapshot | Anonymous | what changed |
|---|---|---|
| header mapped | 24640 kB | parse only |
| all 779 tensors via `from_mapped`, unread | 24960 kB | 24161371264 weight bytes borrowed, 0 failures |
| read `output.weight` (252241920 bytes, Q6K) | 24960 kB | `Private_Clean` 6080 -> 252544 kB, file pages |
| one-row `QMatMul` on that tensor | 41920 kB | the output, 149 ms; bytes match the `from_data` copy |
| `from_data(Cow::Borrowed)` of 2.07 GiB (13 tensors, 357 ms) | 2215104 kB | delta 2.07 GiB anonymous, the call at `archive.rs` |

Every tensor in the file accepted `from_mapped`, so the align fallback in
`qtensor_from_gguf_data` does not fire on this checkpoint. Adoption in mistral.rs
is still a separate change and does not bump the `66a8cf1` pin.

## Before it can be filed

- Filed as #4061. `origin/main` had not moved past `12bb5d1`, so the rebase was empty.
- An issue comment on #3708 still needs its own approval. The PR body already says
  "Toward #3708".

## Build note for anyone touching this tree

On this box, `~/.cargo/config.toml`'s `[target.aarch64-unknown-linux-gnu]` rustflags
override candle's `[build] rustflags`, so `target-cpu=native` never reaches rustc and
`gemm-common`'s f16 asm fails with "instruction requires: fullfp16" during test builds.
Prefix cargo invocations with `RUSTFLAGS="-C target-cpu=native"`. A bare `cargo check`
passes regardless (no codegen), so green check proves nothing there.

## Related

- `../mistralrs/02-cpu-gguf-moe-dequant-every-call.md` is the dequant-side of GGUF MoE;
  this file is the residency-side.
- The wider upstream plan lives in `../mistralrs/09-upstream-pr-plan.md`.
