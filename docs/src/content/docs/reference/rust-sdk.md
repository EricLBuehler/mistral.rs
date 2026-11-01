---
title: Rust SDK reference
description: The Model API surface of the mistralrs crate, with signatures and links to runnable examples.
---

Model builders return a `Model`. Reuse it by reference or in an `Arc`; request methods take `&self`. See [getting started](/guides/rust/getting-started/) for setup and [rustdoc](https://docs.rs/mistralrs) for complete signatures.

Most request methods have a `*_with_model(..., model_id: Option<&str>)` twin for multi-model setups; `None` targets the default model. The twins are omitted below.

## Chat

| Method | Result |
| --- | --- |
| `chat(message: impl ToString)` | Assistant text for a single user message. |
| `send_chat_request(request: impl RequestLike)` | `ChatCompletionResponse`. Accepts `TextMessages`, `MultimodalMessages`, or `RequestBuilder`. |
| `stream_chat_request(request: impl RequestLike)` | `Stream<'_>` yielding `Response` values and borrowing the model. See [streaming](/guides/rust/streaming/). |
| `send_raw_chat_request(request: impl RequestLike)` | `(Vec<Tensor>, Vec<u32>)`: first generated-token logits and prompt tokens. See [perplexity](/examples/rust/advanced/perplexity/). |

All four are async and return `Result`.

### Chat speech output

For LFM3-Audio or LFM3-Omni, convert text or multimodal messages into a `RequestBuilder` and select the output mode:

```rust
use mistralrs::{ChatAudioFormat, ChatAudioMode, RequestBuilder};

let request = RequestBuilder::from(messages)
    .with_audio_output(ChatAudioMode::TextAudio, ChatAudioFormat::Wav);
let response = model.send_chat_request(request).await?;
let audio = response.choices[0].message.audio.as_ref();
```

`ChatAudioMode::Audio` requests speech alone; `TextAudio` requests text and speech. `message.audio.data` contains base64 file bytes. Formats are `Wav`, `Pcm16`, `Mp3`, `Flac`, `Opus`, and `Aac`; compressed formats require FFmpeg. For `stream_chat_request`, choose `Pcm16` and concatenate decoded `choices[].delta.audio.data` bytes, including the final tail. LFM3 PCM is mono, signed 16-bit little-endian at 24 kHz.

Keep assistant text for later turns; native SDKs have no HTTP `audio.id` history cache. See [speech sampling and limits](/guides/models/model-family-notes/#lfm3-speech-input-and-output) for defaults and incompatible options.

## Reasoning

`ReasoningEffort::{Off, Low, Medium, High, XHigh}` is accepted by `TextMessages::with_reasoning_effort`, `MultimodalMessages::with_reasoning_effort`, `RequestBuilder::with_reasoning_effort`, and `AgentBuilder::with_reasoning_effort`. The existing `enable_thinking(bool)` or `with_enable_thinking(bool)` methods remain available. Omission leaves effort unspecified with thinking enabled for text output. Chat requests with speech output default to thinking disabled. Explicit controls take precedence; contradictory explicit controls return a request-validation error.

```rust
let messages = TextMessages::new()
    .add_message(TextMessageRole::User, "Solve this carefully.")
    .with_reasoning_effort(ReasoningEffort::High);
```

The effort is passed to the model's chat template; it does not change sampling parameters directly.

## Structured output

```rust
async fn generate_structured<T>(&self, messages: impl Into<RequestBuilder>) -> Result<T>
where T: DeserializeOwned + JsonSchema
```
Constrains generation to the JSON schema derived from `T` (via `schemars`), then deserializes the reply into `T`. Example: [structured](/examples/rust/cookbook/structured/).

## Agentic tools

Configure executors on the model builder, then enable them on a `RequestBuilder`:

| Request method | Purpose |
| --- | --- |
| `with_code_execution()` | Python execution. |
| `with_shell_execution()` | Shell execution. |
| `with_shell_skill(name, description, path)` | Mount a local Skill. |
| `with_input_file(InputFile::from_text(...))` | Attach a file. |
| `with_max_tool_rounds(6)` | Limit tool-loop rounds. |

See [code execution](/guides/agents/enable-code-execution/), [shell](/guides/agents/enable-shell/), [Skills](/guides/agents/skills/), and [file inputs](/guides/agents/file-inputs/) for builder configuration and examples.

## Embeddings

```rust
async fn generate_embeddings(&self, request: EmbeddingRequestBuilder) -> Result<Vec<Vec<f32>>>
```
One embedding vector per input, in insertion order. Example: [embeddings](/examples/rust/advanced/embeddings/).

```rust
async fn generate_embedding(&self, prompt: impl ToString) -> Result<Vec<f32>>
```
Single-input convenience wrapper.

## Image and speech generation

```rust
async fn generate_image(&self, prompt: impl ToString, response_format: ImageGenerationResponseFormat,
    generation_params: DiffusionGenerationParams, save_file: Option<PathBuf>) -> Result<ImageGenerationResponse>
```
Diffusion image generation. Example: [diffusion](/examples/rust/models/diffusion/).

```rust
async fn generate_speech(&self, prompt: impl ToString) -> Result<(Arc<Vec<f32>>, usize, usize)>
```
Text to speech; returns `(pcm, sample_rate, channels)`. This works with dedicated speech models such as Dia and chat speech models such as LFM3-Audio and LFM3-Omni. For LFM3 it requests audio-only synthesis with the checkpoint's TTS system prompt, thinking disabled, and its default voice. Examples: [LFM3 speech](/examples/rust/models/lfm3-audio/) and [Dia speech](/examples/rust/models/speech/).

Use [chat speech output](#chat-speech-output) for custom messages, multimodal input, encoded formats, generation controls, or streaming. Native helpers have no HTTP `instructions`/`speed` fields; put synthesis instructions in chat messages.

## Quantization

```rust
async fn re_isq_model(&self, isq_type: IsqType) -> Result<()>
```
Reapply [ISQ (in-situ quantization)](/reference/quantization-types/) to the loaded model in place, on whatever device it is already on.

```rust
async fn begin_calibration(&self) -> Result<CalibrationStatus>
async fn calibration_status(&self) -> Result<CalibrationStatus>
async fn apply_calibration(&self, save_cimatrix: Option<PathBuf>) -> Result<CalibrationStatus>
```
Online calibration trio (model must be loaded with ISQ):

- `begin_calibration` - start collecting activation statistics from live traffic.
- `calibration_status` - report per-layer progress.
- `apply_calibration` - requantize from source weights and hot-swap the layers; `save_cimatrix` optionally writes the importance matrix to a `.cimatrix` file for reuse.

Guide: [online calibration](/guides/quantization/online-calibration/); example: [online-calibration](/examples/rust/quantization/online-calibration/).

## Tokenization

```rust
async fn tokenize(&self, text: Either<TextMessages, String>, tools: Option<Vec<Tool>>,
    add_special_tokens: bool, add_generation_prompt: bool, enable_thinking: Option<bool>) -> Result<Vec<u32>>
```
Tokenize raw text or chat messages (messages go through the chat template; `tools` only applies to messages).

```rust
async fn tokenize_with_reasoning_effort(&self, text: Either<TextMessages, String>,
    tools: Option<Vec<Tool>>, add_special_tokens: bool, add_generation_prompt: bool,
    enable_thinking: Option<bool>, reasoning_effort: Option<ReasoningEffort>) -> Result<Vec<u32>>
```

Use this variant for chat messages when tokenization must reflect an explicit effort. Raw strings are tokenized directly and do not render the chat template.

```rust
async fn detokenize(&self, tokens: Vec<u32>, skip_special_tokens: bool) -> Result<String>
```

## Introspection and management

- `config() -> Result<MistralRsConfig>`: modalities and device info for the loaded model.
- `max_sequence_length() -> Result<Option<usize>>`.
- Multi-model: `list_models`, `add_model`, `remove_model`, `unload_model`, `reload_model`, `get_default_model_id`, `set_default_model_id`, `list_models_with_status`. Example: [multi-model](/examples/rust/advanced/multi-model/).
- Sessions: `export_session`, `import_session`, `delete_session`, `fork_session`, `list_session_ids`. Guide: [sessions](/guides/agents/persist-sessions/).
- `list_mcp_tools(model_id)`: [MCP (Model Context Protocol)](/guides/agents/connect-mcp-server/)-provided tools registered for a model, as `(name, description)` pairs.
- `find_file(id)`: fetch the full body of a file emitted by the agentic runtime.
- `inner() -> &MistralRs`: escape hatch to the underlying engine; `Model::new(Arc<MistralRs>)` wraps one back up.

## Decision requests

Use `Model::send_decision_request(DecisionRequest)` with d1 models. `DecisionQuestion` and `DecisionAnswer` are typed enums; requests carry arbitrary JSON state, named questions, and optional `DecisionMedia` values. `send_decision_request_with_model(request, Some(model_id))` selects a loaded model. See [decision request semantics](/reference/decisions/) and the [Rust example](/examples/rust/models/lfm3-d1/).
