//! This crate provides an asynchronous, multithreaded API to `mistral.rs`.
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

mod gguf;
mod gguf_lora_model;
mod gguf_xlora_model;
mod lora_model;
mod messages;
mod model;
mod text_model;
mod vision_model;
mod xlora_model;

/// Upcoming API for v0.4.0. Other APIs will be deprecated.
pub mod v0_4_api {
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
pub use mistralrs_core::*;
