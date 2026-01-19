//! ## General mistral.rs server route handlers.

use anyhow::Result;
use axum::extract::{Json, State};
use mistralrs_core::{
    parse_isq_value, MistralRs, MistralRsError, ModelStatus as CoreModelStatus, Request,
};
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
        let (tools_available, mcp_tools_count, mcp_servers_connected) =
            if status == CoreModelStatus::Loaded {
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

/// Request for model operations (unload, reload, status)
#[derive(Debug, Clone, Deserialize, Serialize, ToSchema)]
pub struct ModelOperationRequest {
    #[schema(example = "my-model")]
    pub model_id: String,
}

/// Model status enum
#[derive(Debug, Clone, Copy, Deserialize, Serialize, ToSchema, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ModelStatus {
    Loaded,
    Unloaded,
    Reloading,
    NotFound,
    /// Cannot unload the last remaining model
    CannotUnloadLast,
    /// Model doesn't have loader config for reload
    NoLoaderConfig,
    /// Internal error (e.g., lock poisoned)
    InternalError,
}

/// Response for model status operations
#[derive(Debug, Clone, Deserialize, Serialize, ToSchema)]
pub struct ModelStatusResponse {
    #[schema(example = "my-model")]
    pub model_id: String,
    pub status: ModelStatus,
    /// Error message when status indicates an error condition
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

#[utoipa::path(
  post,
  tag = "Mistral.rs",
  path = "/v1/models/unload",
  request_body = ModelOperationRequest,
  responses(
    (status = 200, description = "Model unloaded successfully", body = ModelStatusResponse),
    (status = 400, description = "Failed to unload model", body = ModelStatusResponse)
  )
)]
pub async fn unload_model(
    State(state): ExtractedMistralRsState,
    Json(request): Json<ModelOperationRequest>,
) -> Json<ModelStatusResponse> {
    let model_id = request.model_id;
    match state.unload_model(&model_id) {
        Ok(()) => Json(ModelStatusResponse {
            model_id,
            status: ModelStatus::Unloaded,
            error: None,
        }),
        Err(e) => {
            let (status, error) = match &e {
                MistralRsError::ModelNotFound(_) => (ModelStatus::NotFound, None),
                MistralRsError::ModelAlreadyUnloaded(_) => (ModelStatus::Unloaded, None),
                MistralRsError::CannotUnloadLastModel => (ModelStatus::CannotUnloadLast, None),
                MistralRsError::NoLoaderConfig(_) => (ModelStatus::NoLoaderConfig, None),
                _ => (ModelStatus::InternalError, Some(e.to_string())),
            };
            Json(ModelStatusResponse {
                model_id,
                status,
                error,
            })
        }
    }
}

#[utoipa::path(
  post,
  tag = "Mistral.rs",
  path = "/v1/models/reload",
  request_body = ModelOperationRequest,
  responses(
    (status = 200, description = "Model reloaded successfully", body = ModelStatusResponse),
    (status = 400, description = "Failed to reload model", body = ModelStatusResponse)
  )
)]
pub async fn reload_model(
    State(state): ExtractedMistralRsState,
    Json(request): Json<ModelOperationRequest>,
) -> Json<ModelStatusResponse> {
    let model_id = request.model_id;
    match state.reload_model(&model_id).await {
        Ok(()) => Json(ModelStatusResponse {
            model_id,
            status: ModelStatus::Loaded,
            error: None,
        }),
        Err(e) => {
            let (status, error) = match &e {
                MistralRsError::ModelNotFound(_) => (ModelStatus::NotFound, None),
                MistralRsError::ModelReloading(_) => (ModelStatus::Reloading, None),
                MistralRsError::ModelAlreadyLoaded(_) => (ModelStatus::Loaded, None),
                MistralRsError::ReloadFailed(msg) => {
                    (ModelStatus::InternalError, Some(msg.clone()))
                }
                _ => (ModelStatus::InternalError, Some(e.to_string())),
            };
            Json(ModelStatusResponse {
                model_id,
                status,
                error,
            })
        }
    }
}

#[utoipa::path(
  post,
  tag = "Mistral.rs",
  path = "/v1/models/status",
  request_body = ModelOperationRequest,
  responses(
    (status = 200, description = "Model status", body = ModelStatusResponse),
    (status = 404, description = "Model not found", body = ModelStatusResponse)
  )
)]
pub async fn get_model_status(
    State(state): ExtractedMistralRsState,
    Json(request): Json<ModelOperationRequest>,
) -> Json<ModelStatusResponse> {
    let model_id = request.model_id;
    match state.get_model_status(&model_id) {
        Ok(Some(core_status)) => {
            let status = match core_status {
                CoreModelStatus::Loaded => ModelStatus::Loaded,
                CoreModelStatus::Unloaded => ModelStatus::Unloaded,
                CoreModelStatus::Reloading => ModelStatus::Reloading,
            };
            Json(ModelStatusResponse {
                model_id,
                status,
                error: None,
            })
        }
        Ok(None) => Json(ModelStatusResponse {
            model_id,
            status: ModelStatus::NotFound,
            error: None,
        }),
        Err(e) => Json(ModelStatusResponse {
            model_id,
            status: ModelStatus::InternalError,
            error: Some(e.to_string()),
        }),
    }
}
