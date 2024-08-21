//! This crate provides an asynchronous, multithreaded API to `mistral.rs`.
//!
//! ## Example
//! ```no_run
//! use std::sync::Arc;
//! use tokio::sync::mpsc::channel;
//!
//! use mistralrs::{
//!     Constraint, DeviceMapMetadata, MistralRs, MistralRsBuilder,
//!     NormalLoaderType, NormalRequest, Request, RequestMessage, Response,
//!     SamplingParams, SchedulerConfig, DefaultSchedulerMethod, TokenSource,
//! };
//!
//! fn setup() -> anyhow::Result<Arc<MistralRs>> {
//!     // See the examples for how to load your model.
//!     todo!()
//! }
//!
//! fn main() -> anyhow::Result<()> {
//!     let mistralrs = setup()?;
//!
//!     let (tx, mut rx) = channel(10_000);
//!     let request = Request::Normal(NormalRequest {
//!         messages: RequestMessage::Completion {
//!             text: "Hello! My name is ".to_string(),
//!             echo_prompt: false,
//!             best_of: 1,
//!         },
//!         sampling_params: SamplingParams::default(),
//!         response: tx,
//!         return_logprobs: false,
//!         is_streaming: false,
//!         id: 0,
//!         constraint: Constraint::None,
//!         suffix: None,
//!         adapters: None,
//!         tool_choice: None,
//!         tools: None,
//!     });
//!     mistralrs.get_sender()?.blocking_send(request)?;
//!
//!     let response = rx.blocking_recv().unwrap();
//!     match response {
//!         Response::CompletionDone(c) => println!("Text: {}", c.choices[0].text),
//!         _ => unreachable!(),
//!     }
//!     Ok(())
//! }
//! ```

pub use candle_core::{DType, Device, Result, Tensor};
pub use mistralrs_core::*;
