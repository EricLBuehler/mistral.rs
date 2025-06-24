//! ## General mistral.rs server route handlers.

use anyhow::Result;
use axum::extract::{Json, State};
use mistralrs_core::{parse_isq_value, MistralRs, Request};
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use crate::{
    openai::{ModelObject, ModelObjects},
    types::ExtractedMistralRsState,
};

#[utoipa::path(
  get,
  tag = "Mistral.rs",
  path = "/v1/models",
  responses((status = 200, description = "Served model info", body = ModelObjects))
)]
pub async fn models(State(state): ExtractedMistralRsState) -> Json<ModelObjects> {
    // Get all available models
    let available_models = state.list_models().unwrap_or_default();
    let mut model_objects = Vec::new();

    // Add "default" as a special model option
    model_objects.push(ModelObject {
        id: "default".to_string(),
        object: "model",
        created: state.get_creation_time(),
        owned_by: "local",
        tools_available: None,
        mcp_tools_count: None,
        mcp_servers_connected: None,
    });

    for model_id in available_models {
        // Get model-specific information
        let tools_count = state.get_tools_count(Some(&model_id)).unwrap_or(0);
        let has_mcp = state.has_mcp_client(Some(&model_id)).unwrap_or(false);

        let (tools_available, mcp_tools_count, mcp_servers_connected) =
            if has_mcp || tools_count > 0 {
                (Some(tools_count > 0), Some(tools_count), Some(1)) // Simplified MCP info
            } else {
                (None, None, None)
            };

        model_objects.push(ModelObject {
            id: model_id,
            object: "model",
            created: state.get_creation_time(),
            owned_by: "local",
            tools_available,
            mcp_tools_count,
            mcp_servers_connected,
        });
    }

    Json(ModelObjects {
        object: "list",
        data: model_objects,
    })
}

#[utoipa::path(
  get,
  tag = "Mistral.rs",
  path = "/health",
  responses((status = 200, description = "Server is healthy"))
)]
pub async fn health() -> &'static str {
    "OK"
}

#[derive(Debug, Clone, Deserialize, Serialize, ToSchema)]
pub struct ReIsqRequest {
    #[schema(example = "Q4K")]
    ggml_type: String,
}

#[utoipa::path(
  post,
  tag = "Mistral.rs",
  path = "/re_isq",
  request_body = ReIsqRequest,
  responses((status = 200, description = "Reapply ISQ to a non GGUF or GGML model."))
)]
pub async fn re_isq(
    State(state): ExtractedMistralRsState,
    Json(request): Json<ReIsqRequest>,
) -> Result<String, String> {
    let repr = format!("Re ISQ: {:?}", request.ggml_type);
    MistralRs::maybe_log_request(state.clone(), repr.clone());
    let request = Request::ReIsq(parse_isq_value(&request.ggml_type, None)?);
    state.get_sender(None).unwrap().send(request).await.unwrap();
    Ok(repr)
}
