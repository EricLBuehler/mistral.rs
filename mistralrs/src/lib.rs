//! This crate provides an asynchronous, multithreaded API to `mistral.rs`.
//!
//! ## Example
//! ```no_run
//! use anyhow::Result;
//! use mistralrs::{
//!     IsqType, PagedAttentionMetaBuilder, TextMessageRole, TextMessages, TextModel,
//! };
//!
//! #[tokio::main]
//! async fn main() -> Result<()> {
//!     let model = TextModel::builder("microsoft/Phi-3.5-mini-instruct".to_string())
//!         .with_isq(IsqType::Q8_0)
//!         .with_logging()
//!         .with_paged_attn(PagedAttentionMetaBuilder::default().build()?)
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

pub mod messages;
pub use messages::{RequestBuilder, RequestLike, TextMessageRole, TextMessages};

pub mod text_model;
pub use text_model::{PagedAttentionMetaBuilder, TextModel, TextModelBuilder};

pub use candle_core::{DType, Device, Result, Tensor};
pub use mistralrs_core::*;
