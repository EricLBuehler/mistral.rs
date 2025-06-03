//! ## Custom types used in mistral.rs server core.

use std::sync::Arc;

use axum::extract::State;
use mistralrs_core::{MistralRs, Pipeline};

/// This is the underlying instance of mistral.rs.
pub type SharedMistralRsState = Arc<MistralRs>;

/// This is the `SharedMistralRsState` that has been extracted for an axum handler.
pub type ExtractedMistralRsState = State<SharedMistralRsState>;

pub(crate) type LoadedPipeline = Arc<tokio::sync::Mutex<dyn Pipeline + Send + Sync>>;
