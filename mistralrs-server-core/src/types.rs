/// Custom types
use std::sync::Arc;

use axum::extract::State;
use mistralrs_core::{MistralRs, Pipeline};

pub type SharedMistralState = Arc<MistralRs>;
pub type ExtractedMistralState = State<SharedMistralState>;
pub(crate) type LoadedPipeline = Arc<tokio::sync::Mutex<dyn Pipeline + Send + Sync>>;
