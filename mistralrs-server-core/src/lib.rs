use std::sync::Arc;

use axum::extract::State;
use mistralrs_core::{MistralRs, ModelSelected, Pipeline};

pub mod chat_completion;
mod completions;
pub mod defaults;
mod handlers;
mod image_generation;
pub mod mistralrs_for_server_builder;
pub mod mistralrs_server_router_builder;
pub mod openai;
pub mod openapi_doc;
mod speech_generation;
mod util;

pub type SharedMistralState = Arc<MistralRs>;
pub type ExtractedMistralState = State<SharedMistralState>;
type LoadedPipeline = Arc<tokio::sync::Mutex<dyn Pipeline + Send + Sync>>;
