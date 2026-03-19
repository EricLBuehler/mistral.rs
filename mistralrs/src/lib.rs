//! # mistralrs — Blazing-Fast LLM Inference in Rust
//!
//! The Rust SDK for [mistral.rs](https://github.com/EricLBuehler/mistral.rs), a high-performance
//! LLM inference engine supporting text, vision, speech, image generation, and embedding models.
//!
//! ## Quick Start
//!
//! ```no_run
//! use mistralrs::{IsqBits, ModelBuilder, TextMessages, TextMessageRole};
//!
//! #[tokio::main]
//! async fn main() -> mistralrs::error::Result<()> {
//!     let model = ModelBuilder::new("Qwen/Qwen3-4B")
//!         .with_auto_isq(IsqBits::Four)
//!         .build()
//!         .await?;
//!
//!     let response = model.chat("What is Rust's ownership model?").await?;
//!     println!("{response}");
//!     Ok(())
//! }
//! ```
//!
//! ## Capabilities
//!
//! | Capability | Builder | Example |
//! |---|---|---|
//! | Any model (auto-detect) | [`ModelBuilder`] | `examples/getting_started/text_generation/` |
//! | Text generation | [`TextModelBuilder`] | `examples/getting_started/text_generation/` |
//! | Vision (image+text) | [`VisionModelBuilder`] | `examples/getting_started/vision/` |
//! | GGUF quantized models | [`GgufModelBuilder`] | `examples/getting_started/gguf/` |
//! | Image generation | [`DiffusionModelBuilder`] | `examples/models/diffusion/` |
//! | Speech synthesis | [`SpeechModelBuilder`] | `examples/models/speech/` |
//! | Embeddings | [`EmbeddingModelBuilder`] | `examples/getting_started/embedding/` |
//! | Structured output | [`Model::generate_structured`] | `examples/advanced/json_schema/` |
//! | Tool calling | [`Tool`], [`ToolChoice`] | `examples/advanced/tools/` |
//! | Agents | [`AgentBuilder`] | `examples/advanced/agent/` |
//! | Multi-model | [`MultiModelBuilder`] | `examples/advanced/multi_model/` |
//! | LoRA / X-LoRA | [`LoraModelBuilder`], [`XLoraModelBuilder`] | `examples/advanced/lora/` |
//! | AnyMoE | [`AnyMoeModelBuilder`] | `examples/advanced/anymoe/` |
//! | MCP client | [`McpClientConfig`] | `examples/advanced/mcp_client/` |
//!
//! ## Model Loading
//!
//! All models are created through builder structs that follow a consistent pattern:
//!
//! ```no_run
//! # use mistralrs::*;
//! # async fn example() -> error::Result<()> {
//! let model = ModelBuilder::new("Qwen/Qwen3-4B")
//!     .with_auto_isq(IsqBits::Four)            // In-situ quantization (auto-selects best type)
//!     .with_logging()                        // Enable logging
//!     .with_paged_attn(PagedAttentionMetaBuilder::default().build()?)
//!     .build()
//!     .await?;
//! # Ok(())
//! # }
//! ```
//!
//! Use [`ModelBuilder::with_auto_isq`] for automatic platform-optimal quantization (e.g., `with_auto_isq(IsqBits::Four)`),
//! or [`ModelBuilder::with_isq`] with a specific [`IsqType`]: `Q4_0`, `Q4_1`, `Q4K`, `Q5_0`, `Q5_1`, `Q5K`,
//! `Q6K`, `Q8_0`, `Q8_1`, `HQQ4`, `HQQ8`, and more.
//!
//! ## Choosing a Request Type
//!
//! | Type | Use When | Sampling |
//! |---|---|---|
//! | [`TextMessages`] | Simple text-only chat, no special settings needed | Deterministic |
//! | [`VisionMessages`] | Your prompt includes images or audio | Deterministic |
//! | [`RequestBuilder`] | You need tools, logprobs, custom sampling, constraints, adapters, or web search | Configurable |
//!
//! `TextMessages` and `VisionMessages` can be converted into a [`RequestBuilder`] via
//! `Into<RequestBuilder>` if you start simple and later need more control.
//!
//! ## Streaming
//!
//! The stream returned by [`Model::stream_chat_request`] implements
//! [`futures::Stream`], so you can use `StreamExt` combinators:
//!
//! ```no_run
//! use futures::StreamExt;
//! use mistralrs::*;
//!
//! # async fn example(model: Model) -> error::Result<()> {
//! let messages = TextMessages::new()
//!     .add_message(TextMessageRole::User, "Tell me a joke.");
//!
//! let mut stream = model.stream_chat_request(messages).await?;
//! while let Some(chunk) = stream.next().await {
//!     if let Response::Chunk(c) = chunk {
//!         if let Some(text) = c.choices.first().and_then(|ch| ch.delta.content.as_ref()) {
//!             print!("{text}");
//!         }
//!     }
//! }
//! # Ok(())
//! # }
//! ```
//!
//! ## Structured Output
//!
//! Derive [`schemars::JsonSchema`] on your type and the model will be constrained to
//! produce valid JSON matching the schema:
//!
//! ```no_run
//! use mistralrs::*;
//! use schemars::JsonSchema;
//! use serde::Deserialize;
//!
//! #[derive(Deserialize, JsonSchema)]
//! struct City {
//!     name: String,
//!     country: String,
//!     population: u64,
//! }
//!
//! # async fn example(model: Model) -> error::Result<()> {
//! let messages = TextMessages::new()
//!     .add_message(TextMessageRole::User, "Give me info about Paris.");
//!
//! let city: City = model.generate_structured::<City>(messages).await?;
//! println!("{}: pop. {}", city.name, city.population);
//! # Ok(())
//! # }
//! ```
//!
//! ## Blocking API
//!
//! For non-async applications, use [`blocking::BlockingModel`]:
//!
//! ```no_run
//! use mistralrs::blocking::BlockingModel;
//! use mistralrs::{IsqBits, ModelBuilder};
//!
//! fn main() -> mistralrs::error::Result<()> {
//!     let model = BlockingModel::from_auto_builder(
//!         ModelBuilder::new("Qwen/Qwen3-4B")
//!             .with_auto_isq(IsqBits::Four),
//!     )?;
//!     let answer = model.chat("What is 2+2?")?;
//!     println!("{answer}");
//!     Ok(())
//! }
//! ```
//!
//! ## Error Handling
//!
//! All public methods return [`error::Result<T>`](error::Result) with a structured
//! [`error::Error`] enum. Variants include [`ModelLoad`](error::Error::ModelLoad),
//! [`Inference`](error::Error::Inference), [`RequestValidation`](error::Error::RequestValidation),
//! and more. The error type implements `std::error::Error`, so it works seamlessly with
//! `anyhow` and `eyre`.
//!
//! ## MCP (Model Context Protocol)
//!
//! ```no_run
//! # use mistralrs::*;
//! # async fn example() -> error::Result<()> {
//! let mcp_config = McpClientConfig {
//!     servers: vec![/* your server configs */],
//!     auto_register_tools: true,
//!     tool_timeout_secs: Some(30),
//!     max_concurrent_calls: Some(5),
//! };
//!
//! let model = ModelBuilder::new("path/to/model")
//!     .with_auto_isq(IsqBits::Eight)
//!     .with_mcp_client(mcp_config)
//!     .build()
//!     .await?;
//! # Ok(())
//! # }
//! ```
//!
//! ## Feature Flags
//!
//! | Flag | Effect |
//! |---|---|
//! | `cuda` | CUDA GPU support |
//! | `flash-attn` | Flash Attention 2 kernels (requires `cuda`) |
//! | `cudnn` | cuDNN acceleration (requires `cuda`) |
//! | `nccl` | Multi-GPU via NCCL (requires `cuda`) |
//! | `metal` | Apple Metal GPU support |
//! | `accelerate` | Apple Accelerate framework |
//! | `mkl` | Intel MKL acceleration |
//!
//! The default feature set (no flags) builds with pure Rust — no C compiler or system
//! libraries required.
//!
//! ## Architecture
//!
//! ```text
//! ModelBuilder / TextModelBuilder / VisionModelBuilder / GgufModelBuilder / ...
//!     │
//!     ▼
//!   Model ──── send_chat_request() ──► Engine ──► Pipeline ──► Output
//!     │                                  │
//!     ├── chat()                    Scheduler + PagedAttention
//!     ├── stream_chat_request()
//!     ├── generate_structured()
//!     └── send_*_with_model()       (multi-model dispatch)
//! ```

#[macro_use]
mod builder_macros;
mod agent;
mod anymoe;
mod auto_model;
pub mod blocking;
mod diffusion_model;
mod embedding_model;
pub mod error;
mod gguf;
mod gguf_lora_model;
mod gguf_xlora_model;
mod isq_setting;
mod lora_model;
mod messages;
mod model;
pub mod model_builder_trait;
mod speculative;
mod speech_model;
mod text_model;
mod vision_model;
mod xlora_model;

pub(crate) use isq_setting::resolve_isq;
pub use isq_setting::{IsqBits, IsqSetting};

pub use agent::{
    Agent, AgentBuilder, AgentConfig, AgentEvent, AgentResponse, AgentStep, AgentStopReason,
    AgentStream, AsyncToolCallback, ToolCallbackType, ToolResult,
};
pub use anymoe::AnyMoeModelBuilder;
pub use auto_model::ModelBuilder;
pub use diffusion_model::DiffusionModelBuilder;
pub use embedding_model::{EmbeddingModelBuilder, UqffEmbeddingModelBuilder};
pub use gguf::GgufModelBuilder;
pub use gguf_lora_model::GgufLoraModelBuilder;
pub use gguf_xlora_model::GgufXLoraModelBuilder;
pub use lora_model::LoraModelBuilder;
pub use messages::{
    EmbeddingRequest, EmbeddingRequestBuilder, EmbeddingRequestInput, RequestBuilder, RequestLike,
    TextMessageRole, TextMessages, VisionMessages,
};
pub use mistralrs_core::{
    McpClient, McpClientConfig, McpServerConfig, McpServerSource, McpToolInfo,
};
pub use mistralrs_core::{SearchCallback, SearchResult, ToolCallback};
pub use model::{best_device, Model};
pub use model_builder_trait::{AnyModelBuilder, MultiModelBuilder};
pub use speculative::TextSpeculativeBuilder;
pub use speech_model::SpeechModelBuilder;
pub use text_model::{PagedAttentionMetaBuilder, TextModelBuilder, UqffTextModelBuilder};
pub use vision_model::{UqffVisionModelBuilder, VisionModelBuilder};
pub use xlora_model::XLoraModelBuilder;

pub use candle_core::{DType, Device, Result, Tensor};
pub use candle_nn::loss::cross_entropy as cross_entropy_loss;

/// Low-level types and internals re-exported from `mistralrs_core`.
///
/// Most users don't need these types directly. They're available for advanced
/// use cases like custom pipelines, device mapping, or direct engine access.
pub mod core;

// ========== Response Types ==========
pub use mistralrs_core::{
    ChatCompletionChunkResponse, ChatCompletionResponse, Choice, ChunkChoice, CompletionResponse,
    Delta, Logprobs, Response, ResponseMessage, TopLogprob, Usage,
};

// ========== Request Types ==========
pub use mistralrs_core::{Constraint, LlguidanceGrammar, MessageContent, NormalRequest, Request};

// ========== Sampling ==========
pub use mistralrs_core::{DrySamplingParams, SamplingParams, StopTokens};

// ========== Tool Types ==========
pub use mistralrs_core::{
    CalledFunction, Function, Tool, ToolCallResponse, ToolCallType, ToolChoice, ToolType,
};

// ========== Config Types ==========
pub use mistralrs_core::{
    DefaultSchedulerMethod, IsqType, MemoryGpuConfig, ModelDType, PagedAttentionConfig,
    PagedCacheType, SchedulerConfig, WebSearchOptions,
};

// ========== Audio Types ==========
pub use mistralrs_core::AudioInput;

// ========== Custom Logits ==========
pub use mistralrs_core::CustomLogitsProcessor;

// ========== Model Category ==========
pub use mistralrs_core::ModelCategory;

// ========== Search Types ==========
pub use mistralrs_core::{SearchEmbeddingModel, SearchFunctionParameters};

// ========== Speech Types ==========
pub use mistralrs_core::{speech_utils, SpeechLoaderType};

// ========== AnyMoe Types ==========
pub use mistralrs_core::{AnyMoeConfig, AnyMoeExpertType};

// ========== Diffusion Types ==========
pub use mistralrs_core::{
    DiffusionGenerationParams, DiffusionLoaderType, ImageGenerationResponseFormat,
};

// ========== Speculative Types ==========
pub use mistralrs_core::SpeculativeConfig;

// ========== Device Mapping ==========
pub use mistralrs_core::{AutoDeviceMapParams, DeviceMapSetting};

// ========== Topology ==========
pub use mistralrs_core::{LayerTopology, Topology};

// ========== Loader Types ==========
pub use mistralrs_core::{NormalLoaderType, VisionLoaderType};

// ========== Token Source ==========
pub use mistralrs_core::TokenSource;

// ========== Engine (Advanced) ==========
pub use mistralrs_core::{IntervalLogger, MistralRs, RequestMessage, ResponseOk};

// ========== Utilities ==========
pub use mistralrs_core::{initialize_logging, paged_attn_supported, parse_isq_value};

// ========== llguidance ==========
pub use mistralrs_core::llguidance;

// Re-export the tool proc macro for ergonomic tool definition
pub use mistralrs_macros::tool;

// Re-export schemars for use in tool definitions
pub use schemars;
