//! Synchronous (blocking) wrappers for the mistralrs SDK.
//!
//! This module mirrors the async API but blocks the current thread.
//! It is intended for scripts, CLI tools, and non-async applications.
//!
//! **Important:** `BlockingModel` must NOT be used from within an existing
//! tokio runtime — calling `block_on` inside an async context will panic.
//!
//! # Example
//! ```no_run
//! use mistralrs::blocking::BlockingModel;
//! use mistralrs::{IsqBits, TextModelBuilder, TextMessages, TextMessageRole};
//!
//! fn main() -> mistralrs::error::Result<()> {
//!     let model = BlockingModel::from_builder(
//!         TextModelBuilder::new("microsoft/Phi-3.5-mini-instruct")
//!             .with_auto_isq(IsqBits::Four),
//!     )?;
//!
//!     let messages = TextMessages::new()
//!         .add_message(TextMessageRole::User, "Hello!");
//!
//!     let response = model.send_chat_request(messages)?;
//!     println!("{}", response.choices[0].message.content.as_ref().unwrap());
//!     Ok(())
//! }
//! ```

use std::sync::Arc;

use crate::error::Error as SdkError;
use crate::model::Stream;
use crate::{ChatCompletionResponse, Model, ModelBuilder, RequestLike, Response, TextModelBuilder};

/// A synchronous wrapper around [`Model`].
///
/// Owns a tokio runtime internally and blocks the calling thread
/// for all async operations.
pub struct BlockingModel {
    inner: Model,
    rt: Arc<tokio::runtime::Runtime>,
}

impl BlockingModel {
    /// Build a model from a [`TextModelBuilder`] synchronously.
    ///
    /// This creates a dedicated tokio runtime for the model. Panics if called
    /// from within an existing tokio runtime.
    pub fn from_builder(builder: TextModelBuilder) -> crate::error::Result<Self> {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .map_err(|e| SdkError::Inference(e.into()))?;
        let inner = rt
            .block_on(builder.build())
            .map_err(|e| SdkError::ModelLoad(e.into()))?;
        Ok(Self {
            inner,
            rt: Arc::new(rt),
        })
    }

    /// Build a model from a [`ModelBuilder`] (auto-detecting) synchronously.
    ///
    /// This creates a dedicated tokio runtime for the model. Panics if called
    /// from within an existing tokio runtime.
    pub fn from_auto_builder(builder: ModelBuilder) -> crate::error::Result<Self> {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .map_err(|e| SdkError::Inference(e.into()))?;
        let inner = rt
            .block_on(builder.build())
            .map_err(|e| SdkError::ModelLoad(e.into()))?;
        Ok(Self {
            inner,
            rt: Arc::new(rt),
        })
    }

    /// Wrap an existing async [`Model`] for blocking use.
    ///
    /// The caller provides a tokio runtime to use for blocking operations.
    pub fn new(model: Model, rt: Arc<tokio::runtime::Runtime>) -> Self {
        Self { inner: model, rt }
    }

    /// Send a blocking chat request.
    pub fn send_chat_request<R: RequestLike>(
        &self,
        request: R,
    ) -> crate::error::Result<ChatCompletionResponse> {
        self.rt.block_on(self.inner.send_chat_request(request))
    }

    /// Quick chat: send a single user message and get the assistant's text reply.
    pub fn chat(&self, message: impl ToString) -> crate::error::Result<String> {
        self.rt.block_on(self.inner.chat(message))
    }

    /// Send a chat request and get a blocking stream of response chunks.
    pub fn stream_chat_request<R: RequestLike>(
        &self,
        request: R,
    ) -> crate::error::Result<BlockingStream> {
        let stream: Stream<'_> = self.rt.block_on(self.inner.stream_chat_request(request))?;
        Ok(BlockingStream {
            rx: stream.into_receiver(),
            rt: self.rt.clone(),
        })
    }

    /// Send a chat request constrained to a JSON schema derived from `T`, then
    /// deserialize the response into the target type.
    pub fn generate_structured<T>(
        &self,
        messages: impl Into<crate::RequestBuilder>,
    ) -> crate::error::Result<T>
    where
        T: serde::de::DeserializeOwned + schemars::JsonSchema,
    {
        self.rt
            .block_on(self.inner.generate_structured::<T>(messages))
    }

    /// Get a reference to the underlying async [`Model`].
    pub fn inner(&self) -> &Model {
        &self.inner
    }
}

/// A blocking iterator over streaming response chunks.
///
/// Implements [`Iterator`] so it can be used in `for` loops:
/// ```no_run
/// # use mistralrs::blocking::{BlockingModel, BlockingStream};
/// # use mistralrs::*;
/// # fn example(model: &BlockingModel, messages: TextMessages) -> error::Result<()> {
/// for chunk in model.stream_chat_request(messages)? {
///     if let Response::Chunk(c) = chunk {
///         if let Some(text) = c.choices.first().and_then(|ch| ch.delta.content.as_ref()) {
///             print!("{text}");
///         }
///     }
/// }
/// # Ok(())
/// # }
/// ```
pub struct BlockingStream {
    rx: tokio::sync::mpsc::Receiver<Response>,
    rt: Arc<tokio::runtime::Runtime>,
}

impl Iterator for BlockingStream {
    type Item = Response;

    fn next(&mut self) -> Option<Self::Item> {
        self.rt.block_on(self.rx.recv())
    }
}
