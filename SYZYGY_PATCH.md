# Explicit engine shutdown for Syzygy

Upstream baseline: `9397956728e6508d1c7a288043161ad1d70b658f`, matching the
`mistralrs-core 0.8.1` registry source. The SDK and its sibling crates remain at
0.8.1. No model algorithm, scheduler, or tensor implementation is changed.

## Problem and boundary

`MistralRs` privately owns each native `std::thread::JoinHandle`. Upstream Drop
only tries to send `Request::Terminate`; remove/unload/reboot can discard a
handle without joining it. Dropping the public SDK model therefore cannot prove
that its native runtime and resources have stopped. The SDK exposes neither the
handle nor a native completion barrier, so the consumer cannot repair this at
its own Arc boundary.

## Patch

- `MistralRs::shutdown()` closes admission, waits for already admitted model
  management, signals the engines, joins their actual threads, then releases the
  corresponding model resources outside the registry lock.
- Concurrent and repeated shutdown callers share a retained result, including
  native-thread and resource-destructor failures. A model retirement failure
  fences later model management instead of silently enabling a replacement.
- Remove, unload, and reboot use the same join path. The existing reload set
  reserves the model while its worker is retiring; a scoped guard also clears
  this reservation on early returns. Unload checks loader configuration before
  removing the engine. A new worker that cannot be registered is joined before
  that error returns.
- The original engine thread and its Tokio worker/blocking threads carry only
  thread-local execution identity, so a callback cannot synchronously join the
  runtime executing it. This marker owns no model and is not a model registry.
- Constructor metadata is read before the new worker can lock the pipeline;
  a failed warmup send no longer panics and discards the newly created handle.

Explicit shutdown is blocking and must run on a blocking worker. The upstream
best-effort Drop behavior remains an emergency signal, not a join guarantee.
Syzygy owns the public model inside its existing shared `ModelRuntime`, stops
new generation, drains admitted native responses, calls this shutdown, and only
then releases the counted model lease and awaits retirement.

## Verification

The offline tests in `mistralrs-core/src/tests/shutdown.rs` exercise real standard
threads and Tokio workers, retained close results, admission fencing, all-engine
join, resource release ordering, and self-join rejection. Consumer tests in
`syzygy-local-llm` additionally cover cancelled waiters, disconnected HTTP
clients, native close failure, counted exclusive leases, and streaming errors.

Run from the Syzygy workspace using its pinned patch graph:

```sh
cargo test -p mistralrs-core --lib tests::shutdown::
cargo check -p syzygy-local-llm -p syzygy-backend-ai
cargo test -p syzygy-local-llm --lib
cargo test -p syzygy-backend-ai --lib tests::gateway::
cargo clippy -p syzygy-local-llm -p syzygy-backend-ai --all-targets -- -D warnings
```

The patch is tracked under `20260915-LOCAL-LLM-RUNTIME-OWNER`. No upstream PR has
been submitted. The parent integration task runs Cargo and records the actual
consumer results; source formatting alone is not a completed verification.
