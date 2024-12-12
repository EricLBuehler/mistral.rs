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
//! Check out the [`v0_4_api`] module for concise documentation of this, newer API.
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
//! use anyhow::Result;
//! use mistralrs::{
//!     IsqType, PagedAttentionMetaBuilder, TextMessageRole, TextMessages, TextModelBuilder, Response
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
//!     let mut stream = model.stream_chat_request(messages).await?;
//!
//!     while let Some(chunk) = stream.next().await {
//!         if let Response::Chunk(chunk) = chunk{
//!             print!("{}", chunk.choices[0].delta.content);
//!         }
//!         // Handle the error cases.
//!
//!     }
//!     Ok(())
//! }
//! ```

mod anymoe;
mod diffusion_model;
mod gguf;
mod gguf_lora_model;
mod gguf_xlora_model;
mod lora_model;
mod messages;
mod model;
mod text_model;
mod vision_model;
mod xlora_model;

/// This will be the API as of v0.4.0. Other APIs will *not* be deprecated, but moved into a module such as this one.
pub mod v0_4_api {
    pub use super::anymoe::AnyMoeModelBuilder;
    pub use super::diffusion_model::DiffusionModelBuilder;
    pub use super::gguf::GgufModelBuilder;
    pub use super::gguf_lora_model::GgufLoraModelBuilder;
    pub use super::gguf_xlora_model::GgufXLoraModelBuilder;
    pub use super::lora_model::LoraModelBuilder;
    pub use super::messages::{
        RequestBuilder, RequestLike, TextMessageRole, TextMessages, VisionMessages,
    };
    pub use super::model::{best_device, Model};
    pub use super::text_model::{PagedAttentionMetaBuilder, TextModelBuilder};
    pub use super::vision_model::VisionModelBuilder;
    pub use super::xlora_model::XLoraModelBuilder;
}

pub use v0_4_api::*;

pub use candle_core::{DType, Device, Result, Tensor};
pub use candle_nn::loss::cross_entropy as cross_entropy_loss;
pub use mistralrs_core::*;
