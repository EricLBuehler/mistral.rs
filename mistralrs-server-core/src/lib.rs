use std::sync::Arc;

use anyhow::Result;
use axum::{extract::State, Router};
use mistralrs_core::{initialize_logging, MistralRs, ModelSelected, Pipeline};

pub mod chat_completion;
mod completions;
pub mod defaults;
mod handlers;
mod image_generation;
pub mod openai;
pub mod openapi_doc;
pub mod router;
pub mod server_builder;
mod speech_generation;
mod util;

use crate::router::get_router;

pub type SharedMistralState = Arc<MistralRs>;
pub type ExtractedMistralState = State<SharedMistralState>;
type LoadedPipeline = Arc<tokio::sync::Mutex<dyn Pipeline + Send + Sync>>;

pub async fn bootstrap_mistralrs_router_from_state(
    mistralrs: SharedMistralState,
    include_swagger_routes: bool,
    base_path: Option<&str>,
) -> Result<Router> {
    initialize_logging();

    let app = get_router(mistralrs, include_swagger_routes, base_path);

    Ok(app)
}
