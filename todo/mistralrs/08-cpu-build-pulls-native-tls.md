# A default CPU build is not pure Rust, and not because of the matmul

- **Repo:** mistral.rs
- **Components:** `mistralrs-core/Cargo.toml` (via `hf-hub` -> `reqwest` -> `hyper-rustls` -> `rustls`)
- **Status:** open, unrequested. Recorded because it caps a claim we were about to make.
- **Impact:** anyone who wants a CPU inference build with no C toolchain, or a static binary

## What

"The model math is pure Rust" is now true on this branch: `gemv_cpu.rs`, `dot_cpu.rs` and
`attention/backends/cpu/neon.rs` are intrinsics and inline asm with no native dependency. But
the binary is not, and the reason is not the compute:

```
$ cargo tree -p mistralrs-core -i aws-lc-sys
aws-lc-sys v0.37.0
└── aws-lc-rs v1.15.4
    └── rustls v0.23.36
        └── hyper-rustls v0.27.7
            └── reqwest v0.12.28
                └── hf-hub v0.4.3
                    └── mistralrs-core
```

`aws-lc-sys`, `ring`, `cc` and `cmake` are all on the inference path. Every one of them arrives
through the TLS stack used to download models from the Hub. `libmimalloc-sys` is *not* on this
path - it is a workspace-level dependency only.

So a CPU build that never touches the network still compiles a TLS implementation in C, and
requires a C toolchain to build.

## Why it is worth recording

It inverts the usual assumption. The thing that looked like the blocker - a pure-Rust matmul -
is the part that was solved. The remaining native code is incidental, arrived with model
loading, and is not load-bearing for inference.

## Fix

Feature-gate `hf-hub` so a model supplied as a local path does not pull it. That is the
straightforward version and it covers the CPU use case, where the checkpoint is on disk anyway:

```toml
[features]
default = ["hf-hub"]
hf-hub = ["dep:hf-hub"]
```

with the loader split so the path-only constructor does not reference it. `cc` and `cmake` then
leave the graph with the TLS crates.

Not attempted here. It touches the loader plumbing that the Kolibri-1 PR already modifies, so
it should not ride along with that PR.

## Related

- `../candle/01-cpu-decode-gemv-single-threaded.md` is the compute-side half: even with the C
  gone, candle has no alternative matmul on aarch64, so the pure-Rust path would still be
  carrying a bespoke decode GEMV until gemm PR #41 lands.
