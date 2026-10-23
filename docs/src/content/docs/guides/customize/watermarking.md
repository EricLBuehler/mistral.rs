---
title: Text watermarking
description: Select library watermark schemes through Rust, Python, and HTTP, with CPU and GPU sampling integration.
---

mistral.rs delegates watermark algorithms to the standalone Rust `llm-watermarking`
library. Six token schemes are available through Rust, Python, and HTTP generation:
SynthID-Text, KGW, Unigram, exponential-race, inverse-transform, and MPAC. SemStamp
is available through embedding helpers; it cannot be attached to token generation.

Watermarking is opt-in. Omit `watermark` or set it to `null` for ordinary sampling.
All four generation endpoints accept the same configuration: `/v1/chat/completions`,
`/v1/completions`, `/v1/responses`, and `/v1/messages`. Interactive slash commands
do not configure watermarks. Image and audio generation are outside this feature.

For local development, place `llm_watermarking` beside this repository. The workspace
uses `llm-watermarking = { path = "../llm_watermarking", version = "0.1.0" }`.
Both projects use the same pinned Candle Git revision. Mistral enables the library's
`candle` feature and forwards its `cuda` and `metal` build features.

## Configuration

Every scheme requires a secret 32-byte key encoded as 64 hexadecimal characters.
Generate it once and retain it with the tokenizer revision, scheme parameters,
vocabulary size, and implementation version:

```bash
export MISTRALRS_WATERMARK_KEY="$(python3 -c 'import secrets; print(secrets.token_hex(32))')"
```

The examples read this variable; the server does not automatically read or enforce
it. Rust debug formatting redacts keys. Serialized configurations and request-body
logs contain the key, so store and transmit them accordingly.

| `scheme` | Parameters beyond `key` | Detection evidence |
| --- | --- | --- |
| `synthid` | `ngram_len=5`, `depth=30`, `generation_policy="probability_updates"` | `tokens_scored`, `mean_g_value` |
| `kgw` | Required `vocab_size`; `context_width=1`, `green_fraction=0.5`, `delta=2`, `ignore_repeated_ngrams=true` | Green counts, expected/observed rate, nominal z-score |
| `unigram` | Required `vocab_size`; `green_fraction=0.5`, `delta=2`, `ignore_repeated_tokens=true` | Green counts, expected/observed rate, nominal z-score |
| `exponential` | Required `vocab_size`; `sequence_len=1024`, `start_position=0` | `tokens_scored`, `mean_cost` (lower is stronger) |
| `inverse_transform` | Required `vocab_size`; `sequence_len=1024`, `start_position=0` | `tokens_scored`, `mean_cost` (lower is stronger) |
| `mpac` | Required `vocab_size`, nonempty `payload`; `radix=2`, `context_width=1`, `delta=2`, `ignore_repeated_ngrams=true` | Decoded payload, per-symbol votes, winning fraction |
| `semstamp` | Required `embedding_dim`; `num_hyperplanes=8`, `green_fraction=0.25`, `margin=0.02`, `max_attempts=100`, `ignore_repeated_transitions=true` | Valid sentence transitions and nominal z-score |

`vocab_size` is the model's logits width, including padded output slots. It need
not equal the tokenizer's reported vocabulary size. A mismatch is rejected before
building vocabulary-sized watermark tables. MPAC payload entries are radix-r
symbols; they are bits only when `radix=2`. Positions for the two keyed samplers
count generated tokens, excluding the prompt, and wrap at `sequence_len`.

Native library validation checks parameter ranges. Unknown schemes and unknown
fields are rejected. Existing HTTP configurations without `scheme` still select
SynthID. The existing Rust/Python `SynthIdTextWatermarkConfig` remains supported.

## Examples for every scheme

`examples/watermarking/` contains a JSON file for each of the seven schemes. The
six generation configurations use the [Qwen3-4B output vocabulary of 151936](https://huggingface.co/Qwen/Qwen3-4B/raw/main/config.json).
The SemStamp file uses three-dimensional synthetic embeddings for a small demo.
All runners replace the fixture key with `MISTRALRS_WATERMARK_KEY`.

```bash
cargo run --release -p mistralrs --features metal --example watermarking -- examples/watermarking/kgw.json
python examples/python/watermarking.py kgw
python examples/server/watermarking.py kgw --endpoint chat/completions
```

Replace `kgw` with `synthid`, `unigram`, `exponential`, `inverse_transform`, or
`mpac` to run each token scheme. For NVIDIA builds use `--features cuda`; omit the
feature for CPU builds. Python uses the backend compiled into its extension.
The examples set `top_k=40`, enabling the supported GPU top-k insertion point.
The HTTP runner also supports `completions`, `responses`, and `messages`.

## Rust

```rust
use mistralrs::{RequestBuilder, TextMessageRole, Watermark, WatermarkConfig};

let config: WatermarkConfig = serde_json::from_value(serde_json::json!({
    "scheme": "kgw",
    "key": std::env::var("MISTRALRS_WATERMARK_KEY")?,
    "vocab_size": 151936,
    "delta": 2.0
}))?;
let detector = Watermark::new(&config)?;
let request = RequestBuilder::new()
    .add_message(TextMessageRole::User, "Write a long story about a lunar garden.")
    .set_sampler_temperature(0.8)
    .set_sampler_topk(40)
    .set_sampler_watermark(config);
// Send the request with the model, then score its original token IDs.
let evidence = detector.detect(&tokens, prompt_len, &eos_token_ids)?;
```

`SamplingParams.watermark` accepts `WatermarkConfig` directly. Existing SynthID
configurations convert with `.into()`; `set_sampler_watermark` accepts either type.
`Watermark::apply_tensor` delegates dense probability rows to the library without
reading their values back. `WatermarkTensor::Probabilities` retains categorical
selection; `WatermarkTensor::SelectionScores` requires argmax. Its input contract
is finite, nonnegative weights with positive mass. The output remains on the input
device. The host supplies already-filtered weights and retains sampling ownership.

## Python

```python
import os
from mistralrs import ChatCompletionRequest, WatermarkConfig

watermark = WatermarkConfig(
    os.environ["MISTRALRS_WATERMARK_KEY"],
    scheme="mpac",
    vocab_size=151936,
    payload=[1, 0, 1, 1],
)
request = ChatCompletionRequest(
    model="Qwen/Qwen3-4B",
    messages="Write a long story about a lunar garden.",
    temperature=0.8,
    top_k=40,
    max_tokens=512,
    watermark=watermark,
)
evidence = watermark.detect(tokens, prompt_len=prompt_len, eos_token_ids=eos_ids)
```

The generic detector returns a dictionary with scheme-specific fields and a
`kind` discriminator. The older `SynthIdTextWatermarkConfig.detect` continues
returning `(mean_g_value, tokens_scored)`.

## HTTP

For an OpenAI client, put the configuration in `extra_body`:

```python
response = client.chat.completions.create(
    model="Qwen/Qwen3-4B",
    messages=[{"role": "user", "content": "Write a long story about a lunar garden."}],
    temperature=0.8,
    max_tokens=512,
    extra_body={
        "top_k": 40,
        "watermark": {
            "scheme": "unigram",
            "key": os.environ["MISTRALRS_WATERMARK_KEY"],
            "vocab_size": 151936,
        },
    },
)
```

SemStamp configurations deserialize but generation rejects them with an explicit
sentence-embedding requirement. HTTP sentence-rejection generation is not supported.

### HTTP detection

`POST /v1/watermark/detect` accepts the same `watermark` configuration used for
generation and returns the library's scheme-specific evidence as JSON. No model
inference is needed for supplied token IDs or sentence embeddings. Text input
uses the selected serving model's tokenizer.

Prefer original prompt-plus-generation token IDs for exact detection:

```json
{
  "watermark": {
    "scheme": "kgw",
    "key": "0000000000000000000000000000000000000000000000000000000000000000",
    "vocab_size": 151936
  },
  "input": {"type": "tokens", "tokens": [1, 2, 3, 4, 5, 6]},
  "prompt_len": 4,
  "eos_token_ids": [151645]
}
```

Replace the example key, token IDs, vocabulary size, and EOS IDs with the values
from generation. `prompt_len` defaults to zero and counts prefix tokens excluded
from evidence. Scoring stops at the first generated token in `eos_token_ids`,
which defaults to an empty list. Prompt EOS tokens do not stop detection. Use
the same key, scheme parameters, MPAC payload configuration, and key-stream
`start_position` as generation. The HTTP detector accepts vocabulary sizes up to
1,048,576 to bound vocabulary-table allocations.

To score an HTTP generation response directly, send its text:

```python
import requests

evidence = requests.post(
    "http://localhost:1234/v1/watermark/detect",
    json={
        "watermark": watermark_config,
        "input": {
            "type": "text",
            "model": "Qwen/Qwen3-4B",
            "text": response.choices[0].message.content,
        },
    },
)
evidence.raise_for_status()
print(evidence.json())
```

Text input requires `model`, which may be `"default"` or a serving model alias.
It is tokenized as one raw string without a chat template. `add_special_tokens`
defaults to false. If the string includes the prompt, provide its length in the
resulting token sequence as `prompt_len`. Generated text alone can be scored
with zero prompt length, but missing prompt context leaves initial contextual
positions unscored. Retokenization, removed special tokens, or extracted reasoning
can change token IDs or positions; original IDs preserve the exact generation
sequence. For position-keyed schemes, adjust `start_position` if the text omits a
known generated prefix.

For SemStamp, use `input: {"type": "embeddings", "embeddings": [[...], ...]}`.
Here `prompt_len` counts prompt sentences, and `eos_token_ids` must be empty.
Use the same sentence encoder and segmentation as generation. The endpoint does
not generate sentence embeddings or run sentence retries.

Successful responses contain the same `kind` and evidence fields as the Rust
and Python detectors: mean g-values, count statistics, keyed sampling costs,
MPAC payload votes, or SemStamp sentence statistics. They contain no calibrated
watermark verdict or probability of authorship. Empty or insufficient evidence
has zero scored items and null statistics where appropriate. Invalid keys,
parameters, prompt lengths, token IDs, embedding shapes, or scheme/input
combinations return HTTP 400 in the standard `error` envelope. Missing text
models return 404; malformed bodies, content types, and body limits follow the
other server endpoints.

The runnable example can generate and then detect returned text:

```bash
python examples/server/watermarking.py kgw --endpoint completions --detect
```

Set `MISTRALRS_WATERMARK_KEY` to the generation key. Use `--model` and
`--vocab-size` when serving a model other than the example's Qwen configuration.

## Inspect sampling with and without watermarking

Chat completions and completions accept an opt-in `sampling_trace` extension:

```json
{
  "model": "default",
  "messages": [{"role": "user", "content": "Write a story about a lunar garden."}],
  "max_tokens": 64,
  "seed": 42,
  "logprobs": true,
  "top_logprobs": 10,
  "sampling_trace": {"max_steps": 32, "max_candidates": 32, "max_layers": 8}
}
```

Omit `watermark` for a baseline, or add the same watermark configuration used for
generation. `/v1/completions` uses its numeric `logprobs` option instead of the
chat boolean. Tracing requires logprobs and `n=1`; completions also require
`best_of` absent or 1. Speculative and block diffusion decoding are rejected.
Chat tracing requires a model without registered server tool callbacks and no
automatic tool execution, tool rounds, tool dispatch, or input files. Ordinary
client-handled function calls and their grammar retries are supported.

Read `choices[i].sampling_trace.steps`. Each step includes the actual selected
token ID, generated-token index, context length, and accepted attempt number
(0 normally, 1 after a grammar retry). The first candidate is always the selected
token. Remaining rows alternate leading pre/post candidates, deduplicated with
ties ordered by vocabulary ID. Keyed schemes use leading input candidates;
greedy input ranking uses reporting probabilities. `candidate_count` is the
complete vocabulary row length; `candidates_truncated` means some rows were
omitted, not that their probabilities are zero. Token text is a decoded piece,
which need not be a whole word.

| Candidate field | Meaning |
| --- | --- |
| `input_logit` | Actual sampler input before penalties; a grammar retry can already have masked it. |
| `processed_logit` | After penalties and custom processors, before temperature. |
| `reporting_probability` | After temperature and softmax, before filtering and watermarking; existing logprobs retain this meaning. |
| `pre_watermark_probability` | Filtered input normalized over the full vocabulary row. |
| `post_watermark_probability` | Actual categorical sampling weights normalized over the full row. |
| `post_watermark_log_probability` | Natural log of that probability, not a model logit. |

Displayed rows are never renormalized as a smaller distribution. Baseline
pre/post probabilities match. Null logits represent nonfinite masked values;
zero post probability has a null log probability. Position-keyed samplers omit
post probabilities, expose typed `selection_score` values, and report
`selection_rule: "keyed_argmax"`. A null score with
`selection_score_status: "negative_infinity"` is excluded from selection.
Greedy reports `selection_rule: "greedy"` and skips watermarking.

KGW and Unigram expose `membership.kind: "green"` with `favored: true` for green
and false for red. MPAC uses `kind: "favored"` plus payload position and symbol;
unfavored includes other colors and the unassigned vocabulary remainder. Full
MPAC colors are not exported. SynthID `layers` contain candidate-aligned g-values,
input/output probabilities, and full-support green mass and normalization.
They show the production probability updates. A separate teaching bracket can
be requested as described below.
Warmup and repeated-context skips are explicit and have no fabricated membership
or layers. Inverse transform also exposes threshold, rank, and CDF intervals in
the original input weight units. Keys and prompt text are absent from traces.

Defaults are 32 steps, 32 candidates, and 8 captured layers. Limits are 256 steps,
128 candidates, and 32 layers; zero layers is allowed. The product
`max_steps * max_candidates * max(1, max_layers)` must not exceed 65,536.
Invalid limits return HTTP 400. Capture also has an 8 MiB serialized trace budget,
including allowance for trace framing and summaries. Exceeding a step/byte budget
stops observation while generation continues; `truncated` and
`truncation_reason` explain why. `layers_truncated` means only a prefix of layers
was captured; all configured watermark layers still execute.

With `stream: true`, ordinary SSE choice chunks include `sampling_trace` with
newly committed steps. Trace chunks can have empty text deltas: use
`generated_index`, since UTF-8, stop strings, tool calls, and reasoning parsing can
buffer text. Each step is delivered once, followed by a final trace summary
before the terminal choice and `[DONE]`. Stop/EOS selections can appear in traces
even when their text is hidden. Streaming trace history is drained as it is sent.

This uses the existing host logprob sampling path, including when the model runs
on a GPU. There is no additional device readback for diagnostics. Requests without
`sampling_trace` keep their existing sampler path and omit trace response fields.
Full compact GPU trace export is not part of this HTTP path.

```bash
python examples/server/watermarking.py synthid --compare --max-tokens 64
python examples/server/watermarking.py kgw --endpoint completions --trace --detect
```

Both comparison runs use the same seed and settings. Once their generated
prefixes differ, subsequent distributions have different contexts. The pre/post
values within a marked step compare the watermark effect on the same prefix.

### Actual generation tournaments

`POST /v1/completions` accepts `sampling_trace.generation_tournament`. It captures
the bracket that actually selected the emitted token. Select the generation
policy independently inside `watermark`:

```json
{
  "model": "default",
  "prompt": "Write a story about a lunar garden.",
  "max_tokens": 64,
  "seed": 42,
  "temperature": 0.8,
  "top_k": 40,
  "logprobs": 5,
  "watermark": {
    "scheme": "synthid",
    "key": "<your 64 hexadecimal characters>",
    "ngram_len": 5,
    "depth": 4,
    "generation_policy": "tournament"
  },
  "sampling_trace": {
    "max_steps": 1,
    "max_candidates": 128,
    "max_layers": 32,
    "generation_tournament": {"max_matches": 4095}
  }
}
```

Read `choices[i].sampling_trace.steps[j].generation_tournament`. Chat completions
support the same nested settings with their usual boolean `logprobs` option.
The existing tracing restrictions (`n=1`, logprobs, ordinary token decoding)
still apply. Unsupported configurations are rejected before streaming starts.

`generation_policy` defaults to `probability_updates`. That policy executes
probability updates followed by a categorical draw; its generation report has
`status: "no_production_bracket"`, `used_for_generation: false`, and `winner: null`.
Teaching tournaments and marginal probability layers are never relabelled as
actual matches. Requesting capture alone does not select tournament sampling.

The `tournament` policy supports depth 1 through 20; depth 30 is rejected rather
than reduced to a capture limit. It executes the library's scalar sampler on
complete filtered weights and emits its returned token directly. Its CPU
sampling choice applies with tracing enabled or disabled, including when the
model itself runs on CUDA or Metal. The library currently has no device explicit
bracket implementation. The nested capture flag adds no GPU fallback or readback.
Speculative and block decoding are unsupported for this policy, even without
tracing. At depth D, generation performs up to `2^D` draws; limiting capture does
not reduce that work.

Applied reports have `origin: "production"`, `used_for_generation: true`, and
`winner.token_id == step.selected_token_id`, matching the committed token. They
include actual configured depth, executed rounds, full draw/match counts, retained
draws and matches, and collapsed subtrees. Draw IDs distinguish repeated draws
of the same token. Match IDs and draw IDs are separate, potentially sparse
namespaces; do not index arrays by IDs. Entrant `source.kind` is `draw`, `match`,
or `collapsed_subtree`; source IDs refer to that namespace, with collapsed sources
using the omitted root match ID. Each entrant includes its token, draw ID, and
g-value. Match `winner` is `left` or `right`; reasons are `higher_g` or `random_tie`.

Draw `probability` is the full-input probability before watermarking. Explicit
sampling does not compute a marginal post-watermark vector, so candidate post
probabilities are omitted and `selection_rule` is `synthid_tournament`. Reporting
logprobs retain their existing unwatermarked meaning. Warmup and repeated-context
steps have explicit statuses and no fabricated bracket. Disabled watermarking,
greedy sampling, and other algorithms report `watermark_disabled`, `greedy`, or
`unsupported_sampling` respectively.

`max_matches` defaults to 4095 and is bounded by 65535 plus the existing aggregate
observation and byte budgets. `max_layers` limits captured rounds nearest the
winner. Either may be zero to retain only the collapsed root and advancing draw.
`max_candidates` limits the ordinary candidate table, never bracket identities.
Observation accounting reserves up to four records per retained match plus three
root records per step (two when `max_layers=0`). Large combinations are rejected
before generation; reduce `max_steps` or `max_matches` if needed.

Capture reserves space for a minimal current-step report within the existing
8 MiB request budget. If export exceeds that budget, already captured records
collapse to the real root and winner, with `max_bytes` recorded; subsequent steps
continue with the untraced tournament sampler and unchanged RNG. Captured draw
text is capped at 64 KiB per bracket; exceeding it collapses the bracket with
`max_text_bytes`. `text_status` distinguishes decoded text, an unavailable
tokenizer, decode errors, and budget omission. No replacement token text is used.

The report includes `rng_version`, decimal-string `effective_seed`, and the
library's `sampling_version`. Explicit sampling owns a per-sequence ISAAC64
stream; an omitted request seed is resolved once regardless of capture. Existing
shared-RNG sampling is unchanged: if its request seed is unavailable, no-bracket
reports use `effective_seed: null` and
`rng_provenance: "unavailable_shared_stream"`. A seed identifies an initial
stream, not a snapshot; retain the step index, accepted attempt, and generation
settings for replay. Rejected grammar attempts are discarded from the emitted
trace. Streaming delivers each committed trace once, before the terminal event.

```bash
python examples/server/watermarking.py synthid --endpoint completions --model default \
  --generation-policy tournament --depth 4 --generation-tournament --trace-steps 1
```

Remove `--generation-tournament` to run the same policy without capture. Omit
`--generation-policy tournament` to inspect the default policy's honest
no-bracket report. Neither trace form exposes the watermark key.

### SynthID teaching tournaments

To see a small sampled tournament alongside the production trace, add:

```json
{
  "logprobs": true,
  "watermark": {"scheme": "synthid", "key": "<your 64 hex characters>", "depth": 16},
  "sampling_trace": {
    "max_steps": 32,
    "max_candidates": 32,
    "max_layers": 8,
    "teaching_tournament": {"rounds": 3, "seed": 42}
  }
}
```

This is a **teaching simulation**, labeled `origin: "teaching_simulation"` and
`used_for_generation: false`. Production SynthID updates probabilities directly;
it does not select the generated token by running this bracket. The simulation
uses the first requested layers with the same key, hash domain, and context.
Its winner can differ from the actual `step.selected_token_id`, particularly when
the configured depth is greater than the requested rounds. A single bracket is
not an estimate of the final token probabilities.

Each step's optional `teaching_tournament` includes:

- `status`, `requested_rounds`, executed `rounds`, `configured_depth`, and
  zero-based `layer_indices`.
- `draws`, with distinct `draw_id`, vocabulary `token_id`, decoded `text`, and
  `probability` normalized over the complete filtered pre-watermark distribution.
  Draws are with replacement; repeated tokens keep separate IDs. Contestants can
  fall outside the ordinary displayed `candidates`.
- `matches`, in round order, with `match_id`, zero-based `round`, `left` and
  `right` entrants. Each entrant retains its original `draw_id`, binary `g_value`,
  and a `source` such as `{"type":"draw","id":0}` or `{"type":"match","id":0}`.
  A match source refers to that earlier match's winning draw.
- `winner` side (`left` or `right`), `winner_draw_id`, and `reason` (`higher_g`
  or `random_tie`) on each match. The top-level `winner` identifies the final
  match, draw, and token separately from the generated token.
- `rng_version: "sha256_tournament_demo_v1"` and `effective_seed` as a decimal
  string, preserving the full 64-bit value in JavaScript clients.

The diagnostic seed is independent of generation's `seed`. For generated index
`i`, the effective seed is the first eight SHA-256 bytes interpreted as a
little-endian u64, hashing `b"mistralrs-synthid-teaching-seed-v1\0"`, then the
little-endian u64 request diagnostic seed, then little-endian u64 `i`.
The library receives this effective seed. Replaying the same inputs, configuration,
and diagnostic seed reproduces the bracket without consuming generation RNG.
Grammar retries reuse the same position seed with the retry's masked input;
only the accepted attempt is returned.

Applied simulations report `status: "demonstrated"`. Warmup, repeated-context,
and host greedy skips report `warmup`, `repeated_context`, or `greedy`, with zero
executed rounds, empty draws/matches, and a null winner. No bracket is fabricated
for skipped steps.

An empty request object defaults to two rounds and diagnostic seed zero. Rounds
must be 1 to 4 and cannot exceed configured SynthID depth. Missing or non-SynthID
watermarks return HTTP 400. Existing trace restrictions still apply. `max_layers`
independently bounds production layer capture and may be zero. A four-round
bracket has 16 draws and 15 matches; `max_candidates` does not truncate it.
Admission counts all bracket records:

`max_steps * (max_candidates * max(1,max_layers) + (2^(rounds+1)-1)) <= 65536`

The additional term is zero when no teaching tournament is requested. The whole
bracket counts toward the existing 8 MiB byte budget. A step that would exceed
the budget is dropped whole, preserving valid match references. SSE delivery and
truncation summaries follow the existing trace behavior. Omitting the option
performs no simulation and omits its response field. No extra GPU readback occurs.

```bash
python examples/server/watermarking.py synthid --compare --teaching-tournament 3 --teaching-seed 42 --max-tokens 64
```

The example includes the tournament only in the marked comparison run.

## SemStamp

SemStamp accepts sentence embeddings from a fixed encoder. It needs complete
candidate sentences, embedding, acceptance testing, and retry/commit handling.
Those operations are not part of mistral's token sampler. The adapter exposes the
library's semantic watermark without constructing another generation pipeline.

Rust callers use `Watermark::semstamp()` for acceptance, signatures, and the
library's `prepare_tensor` operation. `detect_embeddings` scores host embeddings;
`detect_embeddings_tensor` scores Candle embeddings on their existing device.
Python exposes `accepts_embedding(previous, candidate)` and
`detect_embeddings(embeddings, prompt_len=0, device_name="cpu")`; the latter also
accepts `metal` and `cuda` when compiled in. Python lists are uploaded when a GPU
is requested; detection reads only the library's validation status and signatures.

```bash
MISTRALRS_WATERMARK_DEVICE=metal cargo run --release -p mistralrs --features metal --example watermarking -- examples/watermarking/semstamp.json
python examples/python/watermarking.py semstamp --device metal
```

These examples use synthetic vectors, not sentence generation or a robustness
benchmark. Real generation and detection must use the same embedding model.

## Sampling and device behavior

The order remains penalties/processors, temperature and softmax, top-k/top-p/min-p
filters, watermarking, and token selection. Excluded tokens remain excluded.
Reported logprobs refer to the unwatermarked, pre-filter model probabilities.
Greedy sampling has no watermark choice.

CPU generation supports all six token schemes, including speculative probabilities.
The position-keyed samplers produce a point mass for a fixed key, position, and
model distribution; speculative acceptance uses that distribution rather than
treating selection scores as probabilities. Context and position are derived from
sequence history, so clones and discarded branches do not advance shared state.

CUDA and Metal single-sequence top-k sampling run the watermark transformation
on the GPU. Existing top-k kernels identify candidates; Candle prepares filtered
weights; the library's indexed API applies its algorithm directly to K candidates
using their actual vocabulary IDs. No vocabulary-sized probability row is
allocated for watermarking. One compact readback supplies the
existing host draw and original reporting probabilities. No full probability
vector is downloaded, and the library's strict scalar validation readback is
avoided by its trusted tensor API.

This path currently requires active temperature, `top_k` from 1 to 128, no requested
top logprobs, no speculative sampling, and no multiple-sequence mode. Existing
GPU penalty support is reused; custom logits processors and active DRY use the
CPU path. Metal logits bias also uses the CPU path. Other request shapes retain
existing host sampling and CPU watermarking. CUDA fused batch/resident and sparse
speculative verification remain ineligible for watermarked requests.

See [GPU integration boundaries](/guides/customize/watermarking-gpu/) for exact
metadata costs and remaining host boundaries. This feature does not replace
mistral's GPU backend.

## Detection and format

Prefer original token IDs. Detokenizing and re-encoding can change IDs, and
completion-only detection loses initial context for context-dependent schemes.
Provide the prompt length and EOS IDs when available. For position-keyed sampling,
use the generation's starting position or adjust it for a known cropped prefix.

Scores are uncalibrated evidence, not probabilities of AI authorship. Counts,
mean g-values, alignment costs, and decoded payload votes are different statistics.
Repetitions, short samples, low-entropy output, editing, and retokenization affect
detection. Calibrate each scheme with representative marked and unmarked data.
Possession of a key permits generating its signal; a watermark is not a signature.

SynthID retains the previous mistral domain `mistralrs-synthid-text-v1` followed
by a zero byte. Its digest is SHA256(domain || decoded key || LE32(ngram_len) ||
LE32(context tokens) || LE32(candidate)); layer bits are read least significant bit
first within each byte. Other schemes use the library's versioned default domains.
See `llm_watermarking/docs/formats.md` for their precise formats. Matching only the
algorithm name does not establish byte compatibility with another implementation.
None of these independent keys detect private Claude or Gemini watermarks.

## Prior implementation discussions

Before implementation, on 2026-09-25, the upstream
`EricLBuehler/mistral.rs` and fork `iamorlando/mistral.rs` GitHub issue/PR searches
were checked for `watermark`, `watermarking`, and `SynthID`, including closed items
and comment searches. Local tracked code and available Git history were also
checked, starting at commit `2370966bb`.

No relevant text-watermarking discussion, abandoned implementation, or recorded
maintainer rationale was found. Upstream [PR #2030](https://github.com/EricLBuehler/mistral.rs/pull/2030) mentions a CUDA memory-pool
watermark, which is unrelated. There is consequently no evidenced reason to
attribute the previous absence to a maintainer decision.

The implementation concerns identified in the code were sampler ordering,
speculative verification, GPU paths that bypass ordinary sampling, repeated
contexts, and a matching detector. These are engineering findings from this
change, not historical reasons given by maintainers. Anthropic's undisclosed
production parameters also prevent an exact replica of its deployed watermark.
