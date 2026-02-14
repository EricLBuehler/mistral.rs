//! This crate is the Rust SDK for `mistral.rs`, providing an asynchronous interface for LLM inference.
//!
//! To get started loading a model, check out the following builders:
//! - [`TextModelBuilder`]
//! - [`LoraModelBuilder`]
//! - [`XLoraModelBuilder`]
//! - [`GgufModelBuilder`]
//! - [`GgufLoraModelBuilder`]
//! - [`GgufXLoraModelBuilder`]
//! - [`VisionModelBuilder`]
//! - [`AnyMoeModelBuilder`]
//!
//! For loading multiple models simultaneously, use [`MultiModelBuilder`].
//! The returned [`Model`] supports `_with_model` method variants and runtime
//! model management (unload/reload).
//!
//! ## Example
//! ```no_run
//! use anyhow::Result;
//! use mistralrs::{
//!     IsqType, PagedAttentionMetaBuilder, TextMessageRole, TextMessages, TextModelBuilder,
//! };
//!
//! #[tokio::main]
//! async fn main() -> Result<()> {
//!     let model = TextModelBuilder::new("microsoft/Phi-3.5-mini-instruct".to_string())
//!         .with_isq(IsqType::Q8_0)
//!         .with_logging()
//!         .with_paged_attn(|| PagedAttentionMetaBuilder::default().build())?
//!         .build()
//!         .await?;
//!
//!     let messages = TextMessages::new()
//!         .add_message(
//!             TextMessageRole::System,
//!             "You are an AI agent with a specialty in programming.",
//!         )
//!         .add_message(
//!             TextMessageRole::User,
//!             "Hello! How are you? Please write generic binary search function in Rust.",
//!         );
//!
//!     let response = model.send_chat_request(messages).await?;
//!
//!     println!("{}", response.choices[0].message.content.as_ref().unwrap());
//!     dbg!(
//!         response.usage.avg_prompt_tok_per_sec,
//!         response.usage.avg_compl_tok_per_sec
//!     );
//!
//!     Ok(())
//! }
//! ```
//!
//! ## Streaming example
//! ```no_run
//!    use anyhow::Result;
//!    use mistralrs::{
//!        ChatCompletionChunkResponse, ChunkChoice, Delta, IsqType, PagedAttentionMetaBuilder,
//!        Response, TextMessageRole, TextMessages, TextModelBuilder,
//!    };
//!
//!    #[tokio::main]
//!    async fn main() -> Result<()> {
//!        let model = TextModelBuilder::new("microsoft/Phi-3.5-mini-instruct".to_string())
//!            .with_isq(IsqType::Q8_0)
//!            .with_logging()
//!            .with_paged_attn(|| PagedAttentionMetaBuilder::default().build())?
//!            .build()
//!            .await?;
//!
//!        let messages = TextMessages::new()
//!            .add_message(
//!                TextMessageRole::System,
//!                "You are an AI agent with a specialty in programming.",
//!            )
//!            .add_message(
//!                TextMessageRole::User,
//!                "Hello! How are you? Please write generic binary search function in Rust.",
//!            );
//!
//!        let mut stream = model.stream_chat_request(messages).await?;

//!        while let Some(chunk) = stream.next().await {
//!            if let Response::Chunk(ChatCompletionChunkResponse { choices, .. }) = chunk {
//!                if let Some(ChunkChoice {
//!                    delta:
//!                        Delta {
//!                            content: Some(content),
//!                            ..
//!                        },
//!                    ..
//!                }) = choices.first()
//!                {
//!                    print!("{}", content);
//!                };
//!            }
//!        }
//!        Ok(())
//!    }
//! ```
//!
//! ## MCP example
//!
//! The MCP client integrates seamlessly with mistral.rs model builders:
//!
//! ```rust,no_run
//! use mistralrs::{TextModelBuilder, IsqType, McpClientConfig, McpServerConfig, McpServerSource};
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     let mcp_config = McpClientConfig {
//!         servers: vec![/* your server configs */],
//!         auto_register_tools: true,
//!         tool_timeout_secs: Some(30),
//!         max_concurrent_calls: Some(5),
//!     };
//!     
//!     let model = TextModelBuilder::new("path/to/model".to_string())
//!         .with_isq(IsqType::Q8_0)
//!         .with_mcp_client(mcp_config)  // MCP tools automatically registered
//!         .build()
//!         .await?;
//!     
//!     // MCP tools are now available for automatic tool calling
//!     Ok(())
//! }
//! ```

mod agent;
mod anymoe;
mod diffusion_model;
mod embedding_model;
mod gguf;
mod gguf_lora_model;
mod gguf_xlora_model;
mod lora_model;
mod messages;
mod model;
pub mod model_builder_trait;
mod speculative;
mod speech_model;
mod text_model;
mod vision_model;
mod xlora_model;

pub use agent::{
    Agent, AgentBuilder, AgentConfig, AgentEvent, AgentResponse, AgentStep, AgentStopReason,
    AgentStream, AsyncToolCallback, ToolCallbackType, ToolResult,
};
pub use anymoe::AnyMoeModelBuilder;
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
    SchedulerConfig, WebSearchOptions,
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
