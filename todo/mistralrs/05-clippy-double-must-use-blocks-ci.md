# 16 `double_must_use` clippy errors block the CI clippy gate

- **Repo:** mistral.rs
- **Component:** lint debt on `master`
- **Status:** open, trivial
- **Impact:** CI, and anyone running the documented lint command locally

## What

`AGENTS.md` and CI both prescribe:

```
cargo clippy --workspace --tests --examples -- -D warnings
```

On pristine `master` at `3f2515e9b` this fails with 16 instances of
`clippy::double_must_use`: a `#[must_use]` attribute with no message on a function that
already returns a `#[must_use]` type.

Four files, 16 sites:

| File | Lines |
|---|---|
| `mistralrs-core/src/pipeline/mod.rs` | 1748 (and neighbours) |
| `mistralrs-core/src/utils/varbuilder_utils.rs` | 299 |
| `mistralrs-mcp/src/client.rs` | 17 |
| `mistralrs-mcp/src/transport.rs` | 12 |

## Why it matters

The clippy job in `.github/workflows/ci.yml` runs exactly this command, so a PR cannot go green
on lint without either fixing these or carrying an `allow`. Since they predate any current
work, they are the kind of thing that gets rediscovered by whoever runs the gate next, which
is why it is written down here rather than left to be rediscovered again.

## Fix

Either drop the redundant attribute or give it a reason:

```rust
#[must_use]                       // currently, rejected
#[must_use = "returns the built model"]  // accepted
```

`cargo clippy --fix` handles the attribute removal mechanically, but the reason strings are
worth writing by hand, since a `#[must_use]` on a builder is usually load-bearing and the
message is what tells a caller why.

## Verification

```
cargo clippy --workspace --tests --examples -- -D warnings
```

should be silent. Note that this command needs the same features as CI does not: CI runs it
with no `--features`, so run it that way too.

## Note on scope

This was found while verifying a CPU performance branch and is unrelated to it. It is recorded
here because it blocks the documented local check and would otherwise be attributed to
whatever change happened to be in flight.