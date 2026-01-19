//! ## General mistral.rs server route handlers.

use anyhow::Result;
use axum::extract::{Json, Path, State};
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
    let mut model_objects = Vec::new();

    // Add "default" as a special model option
    model_objects.push(ModelObject {
        id: "default".to_string(),
        object: "model",
        created: state.get_creation_time(),
        owned_by: "local",
        status: None,
        tools_available: None,
        mcp_tools_count: None,
        mcp_servers_connected: None,
    });

    // Get all models with their status (loaded, unloaded, reloading)
    let models_with_status = state.list_models_with_status().unwrap_or_default();

    for (model_id, status) in models_with_status {
        // Get model-specific information (only available for loaded models)
        let (tools_available, mcp_tools_count, mcp_servers_connected) = if status == "loaded" {
            let tools_count = state.get_tools_count(Some(&model_id)).unwrap_or(0);
            let has_mcp = state.has_mcp_client(Some(&model_id)).unwrap_or(false);

            if has_mcp || tools_count > 0 {
                (Some(tools_count > 0), Some(tools_count), Some(1)) // Simplified MCP info
            } else {
                (None, None, None)
            }
        } else {
            (None, None, None)
        };

        model_objects.push(ModelObject {
            id: model_id,
            object: "model",
            created: state.get_creation_time(),
            owned_by: "local",
            status: Some(status.to_string()),
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

/// Response for model status operations
#[derive(Debug, Clone, Deserialize, Serialize, ToSchema)]
pub struct ModelStatusResponse {
    #[schema(example = "my-model")]
    pub model_id: String,
    #[schema(example = "loaded")]
    pub status: String,
    #[schema(example = "Model unloaded successfully")]
    pub message: String,
}

#[utoipa::path(
  post,
  tag = "Mistral.rs",
  path = "/v1/models/{model_id}/unload",
  params(
    ("model_id" = String, Path, description = "The model ID to unload")
  ),
  responses(
    (status = 200, description = "Model unloaded successfully", body = ModelStatusResponse),
    (status = 400, description = "Failed to unload model", body = ModelStatusResponse)
  )
)]
pub async fn unload_model(
    State(state): ExtractedMistralRsState,
    Path(model_id): Path<String>,
) -> Json<ModelStatusResponse> {
    match state.unload_model(&model_id) {
        Ok(()) => Json(ModelStatusResponse {
            model_id: model_id.clone(),
            status: "unloaded".to_string(),
            message: format!("Model {} unloaded successfully", model_id),
        }),
        Err(e) => Json(ModelStatusResponse {
            model_id: model_id.clone(),
            status: "error".to_string(),
            message: e,
        }),
    }
}

#[utoipa::path(
  post,
  tag = "Mistral.rs",
  path = "/v1/models/{model_id}/reload",
  params(
    ("model_id" = String, Path, description = "The model ID to reload")
  ),
  responses(
    (status = 200, description = "Model reloaded successfully", body = ModelStatusResponse),
    (status = 400, description = "Failed to reload model", body = ModelStatusResponse)
  )
)]
pub async fn reload_model(
    State(state): ExtractedMistralRsState,
    Path(model_id): Path<String>,
) -> Json<ModelStatusResponse> {
    match state.reload_model(&model_id).await {
        Ok(()) => Json(ModelStatusResponse {
            model_id: model_id.clone(),
            status: "loaded".to_string(),
            message: format!("Model {} reloaded successfully", model_id),
        }),
        Err(e) => Json(ModelStatusResponse {
            model_id: model_id.clone(),
            status: "error".to_string(),
            message: format!("{:?}", e),
        }),
    }
}

#[utoipa::path(
  get,
  tag = "Mistral.rs",
  path = "/v1/models/{model_id}/status",
  params(
    ("model_id" = String, Path, description = "The model ID to check status")
  ),
  responses(
    (status = 200, description = "Model status", body = ModelStatusResponse),
    (status = 404, description = "Model not found", body = ModelStatusResponse)
  )
)]
pub async fn get_model_status(
    State(state): ExtractedMistralRsState,
    Path(model_id): Path<String>,
) -> Json<ModelStatusResponse> {
    match state.get_model_status(&model_id) {
        Ok(Some(status)) => Json(ModelStatusResponse {
            model_id: model_id.clone(),
            status: status.to_string(),
            message: format!("Model {} is {}", model_id, status),
        }),
        Ok(None) => Json(ModelStatusResponse {
            model_id: model_id.clone(),
            status: "not_found".to_string(),
            message: format!("Model {} not found", model_id),
        }),
        Err(e) => Json(ModelStatusResponse {
            model_id: model_id.clone(),
            status: "error".to_string(),
            message: e,
        }),
    }
}
