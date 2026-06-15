---
title: Rust SDK reference
description: The Model API surface of the mistralrs crate, with signatures and links to runnable examples.
---

`Model` is the object every builder (`ModelBuilder`, `GgufModelBuilder`, `EmbeddingModelBuilder`, ...) returns. All methods take `&self`; share one instance by reference or in an `Arc`. This page lists the surface; [docs.rs/mistralrs](https://docs.rs/mistralrs) has full rustdoc, and the [Rust examples](/mistral.rs/examples/) are runnable.

Most request methods have a `*_with_model(..., model_id: Option<&str>)` twin for multi-model setups; `None` targets the default model. The twins are omitted below.

## Chat

```rust
async fn chat(&self, message: impl ToString) -> Result<String>
```
Quick one-shot: send a single user message, get the assistant's text reply.

```rust
async fn send_chat_request<R: RequestLike>(&self, request: R) -> Result<ChatCompletionResponse>
```
Generate non-streaming. Accepts `TextMessages`, `MultimodalMessages`, or `RequestBuilder`. Example: [text-generation](/mistral.rs/examples/rust/getting-started/text-generation/).

```rust
async fn stream_chat_request<R: RequestLike>(&self, request: R) -> Result<Stream<'_>>
```
Generate streaming. The returned `Stream` implements `futures::Stream<Item = Response>` and borrows the model. Guide: [streaming](/mistral.rs/guides/rust/streaming/).

```rust
async fn send_raw_chat_request<R: RequestLike>(&self, request: R) -> Result<(Vec<Tensor>, Vec<u32>)>
```
Returns raw logits of the first generated token plus the prompt tokens. Example: [perplexity](/mistral.rs/examples/rust/advanced/perplexity/).

## Structured output

```rust
async fn generate_structured<T>(&self, messages: impl Into<RequestBuilder>) -> Result<T>
where T: DeserializeOwned + JsonSchema
```
Constrains generation to the JSON schema derived from `T` (via `schemars`), then deserializes the reply into `T`. Example: [structured](/mistral.rs/examples/rust/cookbook/structured/).

## Agentic tools

Enable built-in executors on the model builder, then opt in per request:

```rust
let model = ModelBuilder::new("Qwen/Qwen3-4B")
    .with_code_execution(CodeExecutionConfig::default())
    .with_shell_execution(ShellConfig::default())
    .build()
    .await?;

let req = RequestBuilder::from(messages)
    .with_code_execution()
    .with_shell_skill("my-skill", "Local task-specific skill.", "skills/my-skill")
    .with_max_tool_rounds(6);
```

`with_input_file(InputFile::from_text(...))` attaches user-provided request files. `with_shell_execution()` enables plain shell for a request. `with_shell_skill(...)` mounts a local skill directory using the same directory shape as OpenAI-compatible Skills. Guides: [file inputs](/mistral.rs/guides/agents/file-inputs/), [code execution](/mistral.rs/guides/agents/enable-code-execution/), [shell execution](/mistral.rs/guides/agents/enable-shell/), [OpenAI-compatible Skills](/mistral.rs/guides/agents/skills/). Examples: [file inputs](/mistral.rs/examples/rust/advanced/file-inputs/), [code execution](/mistral.rs/examples/rust/advanced/code-execution/), [shell](/mistral.rs/examples/rust/advanced/shell/), [shell skills](/mistral.rs/examples/rust/advanced/shell-skills/).

## Embeddings

```rust
async fn generate_embeddings(&self, request: EmbeddingRequestBuilder) -> Result<Vec<Vec<f32>>>
```
One embedding vector per input, in insertion order. Example: [embeddings](/mistral.rs/examples/rust/advanced/embeddings/).

```rust
async fn generate_embedding(&self, prompt: impl ToString) -> Result<Vec<f32>>
```
Single-input convenience wrapper.

## Image and speech generation

```rust
async fn generate_image(&self, prompt: impl ToString, response_format: ImageGenerationResponseFormat,
    generation_params: DiffusionGenerationParams, save_file: Option<PathBuf>) -> Result<ImageGenerationResponse>
```
Diffusion image generation. Example: [diffusion](/mistral.rs/examples/rust/models/diffusion/).

```rust
async fn generate_speech(&self, prompt: impl ToString) -> Result<(Arc<Vec<f32>>, usize, usize)>
```
Text to speech; returns `(pcm, sample_rate, channels)`. Example: [speech](/mistral.rs/examples/rust/models/speech/).

## Quantization

```rust
async fn re_isq_model(&self, isq_type: IsqType) -> Result<()>
```
Reapply [ISQ (in-situ quantization)](/mistral.rs/reference/quantization-types/) to the loaded model in place, on whatever device it is already on.

```rust
async fn begin_calibration(&self) -> Result<CalibrationStatus>
async fn calibration_status(&self) -> Result<CalibrationStatus>
async fn apply_calibration(&self, save_cimatrix: Option<PathBuf>) -> Result<CalibrationStatus>
```
Online calibration trio (model must be loaded with ISQ):

- `begin_calibration` - start collecting activation statistics from live traffic.
- `calibration_status` - report per-layer progress.
- `apply_calibration` - requantize from source weights and hot-swap the layers; `save_cimatrix` optionally writes the importance matrix to a `.cimatrix` file for reuse.

Guide: [online calibration](/mistral.rs/guides/quantization/online-calibration/); example: [online-calibration](/mistral.rs/examples/rust/quantization/online-calibration/).

## Tokenization

```rust
async fn tokenize(&self, text: Either<TextMessages, String>, tools: Option<Vec<Tool>>,
    add_special_tokens: bool, add_generation_prompt: bool, enable_thinking: Option<bool>) -> Result<Vec<u32>>
```
Tokenize raw text or chat messages (messages go through the chat template; `tools` only applies to messages).

```rust
async fn detokenize(&self, tokens: Vec<u32>, skip_special_tokens: bool) -> Result<String>
```

## Introspection and management

- `config() -> Result<MistralRsConfig>`: modalities and device info for the loaded model.
- `max_sequence_length() -> Result<Option<usize>>`.
- Multi-model: `list_models`, `add_model`, `remove_model`, `unload_model`, `reload_model`, `get_default_model_id`, `set_default_model_id`, `list_models_with_status`. Example: [multi-model](/mistral.rs/examples/rust/advanced/multi-model/).
- Sessions: `export_session`, `import_session`, `delete_session`, `fork_session`, `list_session_ids`. Guide: [sessions](/mistral.rs/guides/agents/persist-sessions/).
- `list_mcp_tools(model_id)`: [MCP (Model Context Protocol)](/mistral.rs/guides/agents/connect-mcp-server/)-provided tools registered for a model, as `(name, description)` pairs.
- `find_file(id)`: fetch the full body of a file emitted by the agentic runtime.
- `inner() -> &MistralRs`: escape hatch to the underlying engine; `Model::new(Arc<MistralRs>)` wraps one back up.
