//! Error types and conversions for the mistralrs SDK.

use thiserror::Error;

/// Error type for the mistralrs SDK.
///
/// All public methods in this crate return [`Result<T>`](type@Result).
/// This enum has a `#[non_exhaustive]` attribute so new variants can be
/// added in minor releases without breaking downstream code.
#[derive(Error, Debug)]
#[non_exhaustive]
pub enum Error {
    /// Model loading failed (download, parsing, device mapping, etc.).
    #[error("model loading failed: {0}")]
    ModelLoad(Box<dyn std::error::Error + Send + Sync>),

    /// The inference engine returned an error.
    #[error("inference error: {0}")]
    Inference(Box<dyn std::error::Error + Send + Sync>),

    /// A validation error on the request (e.g., empty messages, bad constraint).
    #[error("request validation error: {0}")]
    RequestValidation(String),

    /// An error from the model itself during generation.
    #[error("model error: {message}")]
    ModelError {
        /// Human-readable description of the model error.
        message: String,
        /// The partial / incomplete response that was produced before the error.
        partial_response: Option<Box<mistralrs_core::ChatCompletionResponse>>,
    },

    /// Channel communication error (sender/receiver dropped unexpectedly).
    #[error("channel error: {0}")]
    Channel(String),

    /// Model management error (not found, already loaded, etc.).
    /// Wraps the existing [`mistralrs_core::MistralRsError`].
    #[error(transparent)]
    Management(#[from] mistralrs_core::MistralRsError),

    /// JSON serialization/deserialization error.
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),

    /// An unexpected response type was received.
    #[error("unexpected response type: expected {expected}")]
    UnexpectedResponse {
        /// Description of the response type that was expected.
        expected: &'static str,
    },
}

/// Convenience type alias for `std::result::Result<T, Error>`.
pub type Result<T> = std::result::Result<T, Error>;

impl Error {
    /// Returns a reference to the underlying source error for the
    /// [`ModelLoad`](Self::ModelLoad) and [`Inference`](Self::Inference) variants.
    ///
    /// Returns `None` for all other variants.
    pub fn source_inner(&self) -> Option<&(dyn std::error::Error + Send + Sync)> {
        match self {
            Self::ModelLoad(e) | Self::Inference(e) => Some(e.as_ref()),
            _ => None,
        }
    }

    /// Attempt to downcast the boxed inner error of [`ModelLoad`](Self::ModelLoad) or
    /// [`Inference`](Self::Inference) to a concrete type `T`.
    ///
    /// Returns `None` for other variants or if `T` does not match the inner error type.
    ///
    /// # Example
    /// ```no_run
    /// # use mistralrs::error::Error;
    /// # fn example(err: Error) {
    /// if let Some(io_err) = err.downcast_ref::<std::io::Error>() {
    ///     eprintln!("I/O error: {io_err}");
    /// }
    /// # }
    /// ```
    pub fn downcast_ref<T: std::error::Error + 'static>(&self) -> Option<&T> {
        match self {
            Self::ModelLoad(e) | Self::Inference(e) => {
                let err_ref: &(dyn std::error::Error + 'static) = e.as_ref();
                err_ref.downcast_ref::<T>()
            }
            _ => None,
        }
    }
}

impl From<anyhow::Error> for Error {
    /// Convert from `anyhow::Error` (mapped to [`Error::Inference`]).
    fn from(e: anyhow::Error) -> Self {
        Error::Inference(e.into())
    }
}

impl From<tokio::sync::mpsc::error::SendError<mistralrs_core::Request>> for Error {
    /// Convert from a channel send error (mapped to [`Error::Channel`]).
    fn from(e: tokio::sync::mpsc::error::SendError<mistralrs_core::Request>) -> Self {
        Error::Channel(e.to_string())
    }
}

impl From<Box<mistralrs_core::ResponseErr>> for Error {
    /// Convert from a boxed engine response error.
    fn from(e: Box<mistralrs_core::ResponseErr>) -> Self {
        match *e {
            mistralrs_core::ResponseErr::InternalError(inner) => Error::Inference(inner),
            mistralrs_core::ResponseErr::ValidationError(inner) => {
                Error::RequestValidation(inner.to_string())
            }
            mistralrs_core::ResponseErr::ModelError(message, response) => Error::ModelError {
                message,
                partial_response: Some(Box::new(response)),
            },
            mistralrs_core::ResponseErr::CompletionModelError(message, _) => Error::ModelError {
                message,
                partial_response: None,
            },
        }
    }
}
