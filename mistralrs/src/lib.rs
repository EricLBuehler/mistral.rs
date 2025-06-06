//! This crate provides an asynchronous API to `mistral.rs`.
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
//!        IsqType, PagedAttentionMetaBuilder, Response, TextMessageRole, TextMessages,
//!        TextModelBuilder,
//!    };
//!    use mistralrs_core::{ChatCompletionChunkResponse, ChunkChoice, Delta};
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

mod anymoe;
mod diffusion_model;
mod gguf;
mod gguf_lora_model;
mod gguf_xlora_model;
mod lora_model;
mod messages;
mod model;
mod speculative;
mod speech_model;
mod text_model;
mod vision_model;
mod xlora_model;

pub use anymoe::AnyMoeModelBuilder;
pub use diffusion_model::DiffusionModelBuilder;
pub use gguf::GgufModelBuilder;
pub use gguf_lora_model::GgufLoraModelBuilder;
pub use gguf_xlora_model::GgufXLoraModelBuilder;
pub use lora_model::LoraModelBuilder;
pub use messages::{RequestBuilder, RequestLike, TextMessageRole, TextMessages, VisionMessages};
pub use mistralrs_core::{SearchCallback, SearchResult, ToolCallback};
pub use model::{best_device, Model};
pub use speculative::TextSpeculativeBuilder;
pub use speech_model::SpeechModelBuilder;
pub use text_model::{PagedAttentionMetaBuilder, TextModelBuilder, UqffTextModelBuilder};
pub use vision_model::{UqffVisionModelBuilder, VisionModelBuilder};
pub use xlora_model::XLoraModelBuilder;

pub use candle_core::{DType, Device, Result, Tensor};
pub use candle_nn::loss::cross_entropy as cross_entropy_loss;
pub use mistralrs_core::*;
