Codebase Alignment Check

Existing pattern(s) found: (paths + symbols)
- `/Users/gabe/Desktop/mistral.rs/mistralrs/src/messages.rs` `RequestBuilder::set_sampler_max_len` and `set_deterministic_sampler`
- `/Users/gabe/Desktop/mistral.rs/mistralrs-core/src/sampler.rs` `SamplingParams::deterministic` (`max_len: None`)
- `/Users/gabe/Desktop/mistral.rs/mistralrs-core/src/sequence.rs` `Sequence::is_done` length stop logic (`StopReason::Length`)
- `/Users/gabe/Desktop/mistral.rs/mistralrs/examples/llguidance/main.rs` and `/Users/gabe/Desktop/mistral.rs/mistralrs/examples/json_schema/main.rs` existing bounded example style (`.set_sampler_max_len(...)`)

Call-chain trace (2+ hops): (caller → callee → callee)
- `examples/*/main.rs` request construction → `Model::send_chat_request` (`/Users/gabe/Desktop/mistral.rs/mistralrs/src/model.rs`) → `Engine::add_request` sequence creation with `request.sampling_params.max_len` (`/Users/gabe/Desktop/mistral.rs/mistralrs-core/src/engine/add_request.rs`) → `Sequence::is_done` length termination (`/Users/gabe/Desktop/mistral.rs/mistralrs-core/src/sequence.rs`)

Reuse plan (or divergence justification):
- Reuse existing bounded sampling mechanism (`RequestBuilder::set_sampler_max_len`) in the two affected examples.
- No new abstractions, dependencies, or runtime mechanisms added.

Why this wasn’t already done this way

Constraints or design reasons (with evidence):
- Legacy example code used `TextMessages` convenience type (`/Users/gabe/Desktop/mistral.rs/mistralrs/examples/isq/main.rs` and `/Users/gabe/Desktop/mistral.rs/mistralrs/examples/uqff/main.rs`), which always takes deterministic defaults via `RequestLike::take_sampling_params` and did not expose max length override.
- Deterministic defaults intentionally leave max length unset (`/Users/gabe/Desktop/mistral.rs/mistralrs-core/src/sampler.rs`).

Why now: (what requirement or bug forces this)
- GitHub issue #1845 reports apparent hangs in ISQ/UQFF examples after dummy run when EOS is not sampled quickly.

Tradeoffs

Pros:
- Guarantees bounded completion for the affected examples.
- Aligns with existing bounded examples in this repo.
- Small localized diff in leaf examples + one regression test.

Cons:
- Caps outputs at 256 tokens in those examples.
- Example behavior differs from prior unbounded defaults.

Mitigations:
- Cap selected is high enough for demo responses and can be adjusted by users.
- Change is example-scoped; core library defaults/API behavior unchanged.

Pareto check: Pareto-improving? If not, list regressing dimensions and why unavoidable.
- Pareto-improving: yes (no known regressions for runtime/library behavior; only example output length is intentionally bounded).

Antagonistic Case Review

List 3–6 relevant worst cases
- EOS never sampled under quantized run.
- Long repetitive output that would continue indefinitely.
- Prompt near model max context, with completion still requested.

For each: before/after behavior, guardrails, tests
- EOS never sampled: before unbounded apparent hang; after capped at 256 tokens via `.set_sampler_max_len(256)`; regression test checks both example call sites are bounded.
- Repetitive output: before could run until model limit/stop conditions far later; after deterministic length stop reached.
- Near-context prompt: existing engine logic already clamps keep length and applies sampling max (`/Users/gabe/Desktop/mistral.rs/mistralrs-core/src/engine/add_request.rs`); this change preserves that path while ensuring explicit completion cap.

Before/After Evidence

Before: (failing test / repro command output / error)
- Deterministic repro commands on baseline examples:
  - `rg -o "\\.set_sampler_max_len\\(" mistralrs/examples/isq/main.rs | wc -l` => `0`
  - `rg -o "\\.set_sampler_max_len\\(" mistralrs/examples/uqff/main.rs | wc -l` => `0`
  - `rg -n "TextMessages::new\\("` showed two call sites in isq and one in uqff.

After: (passing test / corrected output)
- Deterministic repro commands after patch:
  - `...isq...` count => `2`
  - `...uqff...` count => `2`
  - `TextMessages::new(` no longer present in either example.
- Compile validation:
  - `CARGO_TARGET_DIR=/tmp/mistralrs_issue1845_final2 cargo check -p mistralrs`
  - Result: success (`Finished 'dev' profile [optimized + debuginfo] target(s) in 4m 21s`).

Artifacts: (test names, logs, snapshots)
- Regression test: `/Users/gabe/Desktop/mistral.rs/mistralrs/tests/issue_1845_examples_are_bounded.rs`
- Logs: `/tmp/issue1845_before.log`, `/tmp/issue1845_after_test.log`, `/tmp/issue1845_after_check.log` (where available)

Merge Conflict Readiness

Repo mode: online/offline
- Offline mode (no upstream sync performed in this run)

Base reference: origin/main@<date> or <commit sha>
- Local `HEAD` at time of patch (no remote fetch/rebase in this run)

Upstream sync performed: yes/no
- No

Conflict risk level: low/medium/high + why
- Low: localized edits in two example files + one new test file.

Likely conflict hotspots: file paths + why
- `/Users/gabe/Desktop/mistral.rs/mistralrs/examples/isq/main.rs` (example churn)
- `/Users/gabe/Desktop/mistral.rs/mistralrs/examples/uqff/main.rs` (example churn)

If conflicts occur, resolution guidance: (2–6 bullets)
- Keep `RequestBuilder` usage in both files.
- Preserve `.set_sampler_max_len(256)` on each request call site.
- Ensure `TextMessages` import remains removed where not used.
- Re-run grep evidence commands for counts and `cargo check -p mistralrs`.

Verification after conflict resolution: tests/repro
- Run deterministic grep counts (before/after criteria listed above).
- Run `cargo check -p mistralrs`.

Repro/Test Plan

Repro steps:
- Run grep count commands for `set_sampler_max_len` and `TextMessages::new` in both examples.

Tests to run:
- `cargo test -p mistralrs issue_1845_examples_are_bounded -- --nocapture`
- `cargo check -p mistralrs`

Expected results:
- Regression test passes with exactly two capped call sites in each example and no `TextMessages::new` usage.
- Cargo check passes.
