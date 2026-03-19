//! ## General mistral.rs server route handlers.

use axum::extract::{Json, State};
use mistralrs_core::{
    auto_tune, collect_system_info, parse_isq_value, run_doctor, AutoDeviceMapParams,
    AutoTuneRequest, AutoTuneResult, MistralRs, MistralRsError, ModelDType, ModelSelected,
    ModelStatus as CoreModelStatus, Request, TokenSource, TuneProfile,
};
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use crate::{
    openai::{ModelObject, ModelObjects},
    types::ExtractedMistralRsState,
};

#[derive(Debug, Clone, Copy, Deserialize, Serialize, ToSchema)]
#[serde(rename_all = "kebab-case")]
pub enum TuneProfileRequest {
    Quality,
    Balanced,
    Fast,
}

#[derive(Debug, Clone, Deserialize, Serialize, ToSchema)]
pub struct UnloadModelRequest {
    #[schema(example = "default")]
    model_id: String,
}

#[derive(Debug, Clone, Deserialize, Serialize, ToSchema)]
pub struct ReloadModelRequest {
    #[schema(example = "default")]
    model_id: String,
}

#[derive(Debug, Clone, Deserialize, Serialize, ToSchema)]
pub struct ModelStatusRequest {
    #[schema(example = "default")]
    model_id: String,
}

#[derive(Debug, Clone, Deserialize, Serialize, ToSchema)]
pub struct TuneModelRequest {
    model: ModelSelected,
    token_source: TokenSource,
    hf_revision: Option<String>,
    #[serde(default)]
    force_cpu: bool,
    profile: TuneProfileRequest,
    requested_isq: Option<String>,
}

#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct ModelStatusResponse {
    model_id: String,
    status: Option<String>,
}

impl From<TuneProfileRequest> for TuneProfile {
    fn from(value: TuneProfileRequest) -> Self {
        match value {
            TuneProfileRequest::Quality => TuneProfile::Quality,
            TuneProfileRequest::Balanced => TuneProfile::Balanced,
            TuneProfileRequest::Fast => TuneProfile::Fast,
        }
    }
}

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

pub async fn system_info() -> Json<mistralrs_core::SystemInfo> {
    Json(collect_system_info())
}

pub async fn system_doctor() -> Json<mistralrs_core::DoctorReport> {
    Json(run_doctor())
}

#[utoipa::path(
    post,
    tag = "Mistral.rs",
    path = "/v1/models/unload",
    request_body = UnloadModelRequest,
    responses((status = 200, description = "Unload a loaded model."))
)]
pub async fn unload_model(
    State(state): ExtractedMistralRsState,
    Json(request): Json<UnloadModelRequest>,
) -> std::result::Result<String, String> {
    state
        .unload_model(&request.model_id)
        .map_err(|e| e.to_string())?;
    Ok(format!("Unloaded model {}", request.model_id))
}

#[utoipa::path(
    post,
    tag = "Mistral.rs",
    path = "/v1/models/reload",
    request_body = ReloadModelRequest,
    responses((status = 200, description = "Reload an unloaded model."))
)]
pub async fn reload_model(
    State(state): ExtractedMistralRsState,
    Json(request): Json<ReloadModelRequest>,
) -> std::result::Result<String, String> {
    state
        .reload_model(&request.model_id)
        .await
        .map_err(|e| e.to_string())?;
    Ok(format!("Reloaded model {}", request.model_id))
}

#[utoipa::path(
    post,
    tag = "Mistral.rs",
    path = "/v1/models/status",
    request_body = ModelStatusRequest,
    responses((status = 200, description = "Get model status.", body = ModelStatusResponse))
)]
pub async fn get_model_status(
    State(state): ExtractedMistralRsState,
    Json(request): Json<ModelStatusRequest>,
) -> std::result::Result<Json<ModelStatusResponse>, String> {
    let status = state
        .get_model_status(&request.model_id)
        .map_err(|e| e.to_string())?
        .map(|s| s.to_string());
    Ok(Json(ModelStatusResponse {
        model_id: request.model_id,
        status,
    }))
}

#[utoipa::path(
    post,
    tag = "Mistral.rs",
    path = "/v1/models/tune",
    request_body = TuneModelRequest,
    responses((status = 200, description = "Auto-tune a model configuration.", body = AutoTuneResult))
)]
pub async fn tune_model(
    Json(request): Json<TuneModelRequest>,
) -> std::result::Result<Json<AutoTuneResult>, String> {
    let requested_isq = request
        .requested_isq
        .as_deref()
        .map(|value| parse_isq_value(value, None))
        .transpose()?;
    let result = auto_tune(AutoTuneRequest {
        model: request.model,
        token_source: request.token_source,
        hf_revision: request.hf_revision,
        force_cpu: request.force_cpu,
        profile: request.profile.into(),
        requested_isq,
    })
    .map_err(|e| e.to_string())?;
    Ok(Json(result))
}

#[cfg(feature = "parking-lot-scheduler")]
pub async fn metrics() -> Json<serde_json::Value> {
    Json(serde_json::json!({"status": "parking-lot-scheduler-enabled"}))
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
) -> std::result::Result<String, String> {
    let repr = format!("Re ISQ: {:?}", request.ggml_type);
    MistralRs::maybe_log_request(state.clone(), repr.clone());
    let request = Request::ReIsq(parse_isq_value(&request.ggml_type, None)?);
    state.get_sender(None).unwrap().send(request).await.unwrap();
    Ok(repr)
}
