---
title: Use decision models (System One)
description: Run CLM locally through the Jev-compatible System One API and Pydantic AI.
---

mistral.rs serves [CLM-v0.1-8B](https://huggingface.co/Contrastive-LM/CLM-v0.1-8B)
through `POST /v1/systemone`, the typed decision API used by TypeSafe's Jev.
The model answers yes/no questions, selects labels, and scores ordered rubrics.
It returns distributions rather than generating text.

## Start the server

```bash
mistralrs serve -m Contrastive-LM/CLM-v0.1-8B -p 1234
```

The auto loader reads the CLM repository's `config.json`, downloads its
`CLM_v0.1-8B.pt` projection checkpoint and the frozen `Qwen/Qwen3-8B` encoder,
and runs both natively in Rust. No Python, PyTorch, or embedding server is needed
at runtime. Use the normal `metal` or `cuda` Cargo feature for GPU execution;
CPU execution is also supported. The full encoder needs the usual 8B-model
memory. In-situ quantization can reduce encoder memory, but changes its
embeddings and can affect decision quality.

Automatic device mapping reserves encoder workspace and uses CPU layers when
the weights and workspace exceed the GPU budget. On Apple Silicon, GPU and CPU
allocations are counted against the same physical RAM. Cold requests reduce
their encoder batch size when GPU memory is tight, and Metal reclaims unused
scratch buffers between layers under memory pressure. Larger GPUs keep the
normal batches and asynchronous execution. These adjustments preserve model
weights and precision; CPU offloading can increase latency.

The trained heads require **Qwen3-8B**, not Qwen3-Embedding-8B or another encoder.
Model loading verifies the encoder identity in the head checkpoint and its
hidden dimension. Loading a different model under the same local directory
name cannot be detected from that metadata alone.

## Request and response

```bash
curl http://localhost:1234/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Contrastive-LM/CLM-v0.1-8B",
    "state": "My invoice was charged twice and nobody answers the phone!",
    "questions": {
      "urgent": {"type": "noul", "instructions": "Is this urgent?"},
      "team": {
        "type": "choice",
        "instructions": "Which team should handle this?",
        "criteria": {"billing": "Charges, invoices, refunds", "technical": "Bugs and outages"}
      },
      "anger": {
        "type": "score",
        "instructions": "How frustrated is the customer?",
        "criteria": ["Calm", "Frustrated", "Very angry"]
      }
    }
  }'
```

`model` selects a loaded model or registered alias. `"default"` selects mistral's
default model. The response identifies the actual loaded CLM repository/path;
it never identifies CLM as Jev. Use the server's normal Bearer authentication
when enabled.

`state` and each question's `instructions` accept a string, object, or array.
CLM also accepts omitted or null instructions, as used by Pydantic's contract.
Question names are identifiers returned unchanged; they do not affect inference.
Structured values are rendered using the upstream CLM prose layout with JSON
key order preserved. Each question embeds its state followed by its instructions.

| Question | Criteria | Answer |
| --- | --- | --- |
| `noul` | Optional `{"true": description, "false": description}` | `{"type":"noul", "noul": probability_of_true}` |
| `choice` | 1 to 255 named options; descriptions may be text, structured values, or null | `type`, `choice`, `confidence`, and `probabilities` keyed by option |
| `score` | 2 to 10 ordered text or structured levels | `type`, `score`, `confidence`, `probabilities`, and `legend` keyed by zero-based level |

Every response contains `model`, `answers`, and `usage`. `usage.input_tokens`
counts encoder tokens spent on cache misses, with duplicate texts encoded once
within a request. An entirely cached request uses zero encoder tokens.
`usage.output_tokens` is zero. `billing_units` counts questions for wire
compatibility and does not represent a local charge.

The optional `temperature` defaults to `1.0` and must be in `(0, 100]`. It divides
CLM's candidate logits before softmax. It is not a generation temperature.
Each state combined with its instructions, and each candidate, must fit in
**2048 tokens**. Oversized or empty tokenized inputs return an error instead of
silently truncating. These are CLM-serving limits, not Jev's context limits.

`GET /v1/models` retains mistral's OpenAI `object` and `data` fields and adds a
TypeSafe-compatible `models` array when decision models are loaded. Existing
chat, embeddings, Responses, and Anthropic routes keep their contracts.
Decision models reject generation and embedding requests. Invalid inputs return
HTTP 400, unknown models return 404, and inference failures return 500 using
the server's existing JSON error format. Decisions do not stream.

## Pydantic AI

[Pydantic's `SystemOneModel`](https://pydantic.dev/docs/ai/models/system-one/)
already implements the `DecisionModel` contract for this HTTP API. No custom
Pydantic provider or model adapter is needed.

```bash
pip install pydantic-ai-slim
export SYSTEM_ONE_BASE_URL=http://localhost:1234
```

```python
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.system_one import SystemOneModel
from pydantic_ai.profiles.decision import DecisionModelProfile
from pydantic_ai.providers.system_one import SystemOneProvider

class Ticket(BaseModel):
    urgent: bool = Field(description="Does this need an urgent response?")
    team: Literal["billing", "technical", "account"] = Field(
        description="Which team should handle this request?"
    )

model = SystemOneModel(
    "Contrastive-LM/CLM-v0.1-8B",
    provider=SystemOneProvider(base_url="http://localhost:1234"),
    profile=DecisionModelProfile(
        context_window=2048,
        decision_max_choice_options=255,
        decision_max_score_levels=10,
    ),
)
result = Agent(model, output_type=Ticket).run_sync("My invoice was charged twice.")
print(result.output)
print(result.response.provider_details)
```

Use a Pydantic AI release that includes `SystemOneModel`; the contract is tested
with `pydantic-ai-slim==2.54.0`. Boolean thresholds, tool routing, structured
outputs, and fallback decisions belong to Pydantic's `DecisionModel`. Free-form
text fields are not supported by this decision model.

The TypeSafe SDK can also target this endpoint by setting `base_url` to
`http://localhost:1234` and selecting the CLM model. It requires an API key
argument even when local authentication is disabled; a placeholder value works
in that case.

Complete examples:

- [HTTP request with all three primitives](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/server/system_one.py)
- [Pydantic structured output](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/server/system_one_pydantic.py)
- [Native Rust `Model::decide`](https://github.com/EricLBuehler/mistral.rs/blob/master/mistralrs/examples/advanced/system_one/main.rs)

## Scoring and execution

The implementation follows the [CLM reference implementation](https://github.com/Contrastive-LM/CLM/tree/bb42c6c5bf914fd449bed2f6ca65be80602cb1f7):

1. Encode plain text with causal Qwen3 and pool the last non-padding token.
2. Normalize the encoder vector and apply the appropriate trained state/action
   MLP, including its activation, layer normalization, and residual settings.
3. Normalize the projected vectors and compute scaled cosine similarity.
   The learned scale is `min(exp(logit_scale), 100)`.
4. Softmax candidate logits independently for each question. `score` is the
   expected zero-based level. Confidence is the top probability minus the mean
   probability of the other options. Ties select the first option.

Encoder inference reuses mistral's Qwen3 embedding implementation, device
mapping, and quantization. Cache misses are sorted into batches of up to 32
texts, bounded by 2048 padded token positions and a maximum 4:1 token-length
ratio. The padded-token budget decreases when available GPU memory cannot hold
the normal encoder workspace. Each text is pooled at its true last token. Heads run in float32 on the
encoder's output device and project cache misses in batches. Encoder embeddings
and projections stay on device; final candidate logits are copied to the host
once per request for answer construction. Requests use
mistral's bounded engine queue, with one decision request evaluated at a time
per model. This encoder path does not perform autoregressive sampling or use
the text-generation pipeline's paged KV cache.

### Shared state computation

On CPU and Metal, Qwen3 batches with at least 32 common starting tokens compute
that prefix once within the request. Each question's remaining tokens retain
their original rotary positions, and a branching attention mask lets them see
only the shared prefix and their own preceding tokens. Last-token pooling and
CLM projection heads are unchanged. This reduces repeated encoder work for
multiple questions about a new state, even with all caches disabled; it does
not reuse answers from similar text.

This uses the existing Candle attention and tensor operations, with no extra
embedding readback. Short or unrelated inputs use ordinary batching. CUDA
retains its existing batched path because its varlen FlashAttention interface
cannot represent the branching mask. Sliding-window models also retain the
ordinary path. Token usage counts logical input tokens before prefix sharing;
debug logs report both logical and computed tokens. Reduced precision can
produce small rounding differences when the matrix shapes change.

### State and action caching

Like the [CLM reference vector cache](https://github.com/Contrastive-LM/CLM/blob/bb42c6c5bf914fd449bed2f6ca65be80602cb1f7/src/clm/cache.py),
the native server reserves a fixed device-memory arena for normalized state and
action projections. A hit skips tokenization, the encoder, and the projection
head. Repeated schemas reuse candidate projections when the state changes;
repeated complete states and instructions also reuse their state projections.
With vector caching alone, an exact repeat needs only vector gathering, dot
products, and answer assembly.

The default is **64 MiB per loaded CLM model**. This is a fixed local-serving
default; upstream defaults to a 2% device-memory budget. Set `CLM_ACTION_CACHE`
before starting the server to choose either convention:

```bash
CLM_ACTION_CACHE=128MiB mistralrs serve -m Contrastive-LM/CLM-v0.1-8B -p 1234
CLM_ACTION_CACHE=0.02 mistralrs serve -m Contrastive-LM/CLM-v0.1-8B -p 1234
CLM_ACTION_CACHE=0 mistralrs serve -m Contrastive-LM/CLM-v0.1-8B -p 1234
```

Sizes accept B, KB, MB, GB, KiB, MiB, and GiB. A bare fraction uses the device's
reported memory budget; `0` disables caching. GPU reservation is capped at 90%
of memory available after loading, with an additional reserve for encoder
workspace; CPU uses the configured size. The arena
never grows. Least-recently-used
entries are evicted; requests larger than the cache still complete using their
freshly computed vectors. Cache keys use SHA-256 of the rendered text and
separate state and action namespaces, with bounded host metadata rather than
retaining the original text. For the released 512-dimensional heads, 64 MiB
holds 32,768 projections.

Caches belong to the loaded model instance. Reloading creates a fresh cache;
runtime requantization and calibration invalidate it. Caching is bypassed while
collecting calibration activations. Cached values are float32 on CPU, Metal, or
CUDA, with no embedding readback or GPU backend reimplementation. The native
cache stores final projections; it does not retain a second host embedding
cache or reuse KV prefixes between different texts.

Repeated comparisons also reuse a bounded CPU cache of the final float32
scores already read back for the response. A score-cache hit performs no GPU
operations or additional readbacks. Keys include the rendered states, ordered
candidates, and temperature; answers are rebuilt for the current question IDs.
This cache uses hashed keys rather than retaining input text, and reserves at
most 1 MiB of score entries, limited to 1/64 of the vector-cache byte budget,
plus bounded map metadata. It is disabled with `CLM_ACTION_CACHE=0`, bypassed
during calibration, and cleared when vectors are replaced, evicted, or
invalidated. New inputs still require encoder inference.

With the server's metrics endpoint enabled, the following counters carry a
`model` label: `mistralrs_decision_cache_hits_total`,
`mistralrs_decision_cache_misses_total`, `mistralrs_decision_encoder_batches_total`,
and `mistralrs_decision_encoder_tokens_total`. Hits and misses count unique
state/action projection keys per request. Startup logs report reserved bytes
and capacity; debug logs report vector-cache lookups and encoder work.
Projection hits include vectors covered by a final-score cache hit.
`mistralrs_decision_score_cache_hits_total` counts requests served from final
scores. `mistralrs_decision_inference_duration_seconds` measures successful
inference including cache lookup and response construction, with a `cache`
label of `hit` or `miss`. `mistralrs_decision_dispatch_duration_seconds` measures
dispatch through engine response, including queue wait; neither duration
includes model loading or the application's HTTP proxy.

CLM probabilities are relative to the supplied candidates. Matching Jev's
[API shape](https://docs.typesafe.ai/api) does not reproduce Jev's model,
accuracy, calibration, pricing, or rate limits. Evaluate thresholds on your
own data, particularly after quantization or temperature changes.

CLM UQFF import/export is currently rejected because encoder-only artifacts
would omit the projection checkpoint. Original Hugging Face or local CLM
directories are supported, including the existing cache and offline mode. A
local CLM `config.json` may point `base_model` to a relative encoder directory;
the head's recorded encoder identity must match that value.

## Validation

`cargo test -p mistralrs-core --test decision` checks the native encoder and
heads against a deterministic PyTorch/Transformers fixture. The same suite
has Metal and CUDA tests under their respective features. It includes mixed
input lengths and compares batched answers with individual questions.

The published heads can be checked separately:

```bash
CLM_CHECKPOINT=/path/to/CLM_v0.1-8B.pt \
  cargo test -p mistralrs-core --lib published_clm_heads_match_pytorch -- --ignored
```

`cargo test -p mistralrs-server-core --test system_one` exercises the HTTP
endpoint, model aliases, discovery, and errors using the tiny model. Set
`MISTRALRS_TEST_SYSTEM_ONE_PYTHON` to a Python interpreter containing
`pydantic-ai-slim==2.54.0` and `typesafe-sdk==0.7.2` to also run both real clients
against that test's temporary server. No model download is needed for this test.

For full-model performance comparisons, save a baseline, restart with the new
build, and replay the same requests:

```bash
python3 scripts/benchmark_decisions.py --base-url http://127.0.0.1:1234 --output /tmp/clm-before.json
python3 scripts/benchmark_decisions.py --base-url http://127.0.0.1:1234 --compare /tmp/clm-before.json --output /tmp/clm-after.json
```

Add `--idle-seconds 5` to measure requests spaced five seconds apart. The
default sends requests back to back, which can substantially understate
interactive latency after idle periods. Compare runs using the same delay.

The benchmark excludes each scenario's warmup and reports median latency,
encoder tokens, and numerical answer differences. It covers repeated requests,
new states with fixed actions, and new states with 32 fixed candidates. Running
one server at a time avoids memory pressure from loading two encoders. The
benchmark rejects a new-state sample that uses zero encoder tokens, preventing
an already-cached replay from being reported as fresh-input performance.

Measured on 2026-10-04 with the full Qwen3-8B encoder in BF16 on an M2 Max
(30 GPU cores, 64 GiB RAM), using a 16 MiB cache. These are five-call medians
after each scenario's warmup, with the two builds run sequentially:

| Scenario | Initial integration (`5d4a6b6c9`) | Vector cache (`1919767c6`) |
| --- | --- | --- |
| Repeated three-question request | 627 ms | 1.3 ms |
| New state, same three-question schema | 929 ms | 437 ms |
| New state, 32 fixed candidates | 1414 ms | 171 ms |

The latter two scenarios append a unique ticket reference to the state on every
call. The maximum absolute numerical answer difference was below `0.000001`,
and selected choices were unchanged. Entirely cached requests used zero encoder
tokens. These figures exclude model loading and first-use kernel warmup; new
texts still require encoder inference.

Interactive checks on the same machine, through the app's HTTP proxy, exposed
225-490 ms delays for vector-cache hits after 2-15 seconds idle. Final-score
caching reduced repeated pull-request-review comparisons to 11-13 ms after
those idle intervals, with roughly 0.03-0.06 ms inside inference. A separate
30-second idle check took 34 ms through the proxy and returned identical
answers. These checks used the default 64 MiB vector cache. New-state variants with the same criteria
still took about 1.1 seconds for 255 encoder tokens. The millisecond result
applies to previously computed comparisons, not arbitrary new inputs; model
loading and first-use warmup are excluded.


A subsequent shared-prefix benchmark used new versions of the three-question
code-review request, with the same BF16 encoder and default 64 MiB cache. Both
builds were run sequentially on the same M2 Max. Candidate projections were
warm, but every measured request had new state vectors and positive encoder
token usage. Direct HTTP medians exclude the first sample in each group:

| Fresh-input workload | Before prefix sharing | With prefix sharing | Measured calls |
| --- | --- | --- | --- |
| Three questions about a short code review | 626 ms | 313 ms | 6 |
| Three questions about a longer state | 2080 ms | 754 ms | 3 |
| One question (control) | 315 ms | 310 ms | 3 |
| Different code changes | 457 ms | 302 ms | 5 |

Selected choices were unchanged. Across all 21 comparisons, the largest
absolute answer difference was 0.01954 (a Noul probability); for the short
review group it was 0.00636. Single-question answers were identical. These
are BF16 rounding differences from the changed computation layout, not
approximate retrieval or reuse of another input's answer. The deterministic
float32 CPU and Metal fixture also compares shared computation against
independent question evaluations.

Fresh short-review requests through the app after 5 and 15 seconds idle took
740 and 722 ms respectively, each processing 246 logical encoder tokens.
Those interactive timings remain substantially higher than the 313 ms warm
backend median. Prefix sharing reduces encoder work; it does not make new
8B-model decisions run in 12 ms on this machine.
