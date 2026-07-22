//! ## General mistral.rs server route handlers.

use anyhow::Result;
use axum::extract::Path;
use axum::extract::{Json, State};
use axum::http::StatusCode;
use mistralrs_core::{
    auto_tune, collect_system_info, parse_isq_value, run_doctor, AutoDeviceMapParams,
    AutoTuneRequest, AutoTuneResult, MistralRs, MistralRsError, ModelDType, ModelSelected,
    ModelStatus as CoreModelStatus, Request, SerializedSession, TokenSource, TuneProfile,
};
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use crate::{
    lora_adapters::list_lora_adapter_models,
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
        root: Some("default".to_string()),
        parent: None,
        adapter_generation: None,
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
            root: Some(model_id.clone()),
            id: model_id,
            object: "model",
            created: state.get_creation_time(),
            owned_by: "local",
            parent: None,
            adapter_generation: None,
            status: Some(status.to_string()),
            tools_available,
            mcp_tools_count,
            mcp_servers_connected,
        });
    }

    for adapter_model in list_lora_adapter_models(&state).unwrap_or_default() {
        model_objects.push(ModelObject {
            root: Some(adapter_model.adapter.alias.clone()),
            id: adapter_model.id,
            object: "model",
            created: state.get_creation_time(),
            owned_by: "local",
            parent: Some(adapter_model.parent),
            adapter_generation: Some(adapter_model.adapter.generation.to_string()),
            status: Some(CoreModelStatus::Loaded.to_string()),
            tools_available: None,
            mcp_tools_count: None,
            mcp_servers_connected: None,
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

#[utoipa::path(
  get,
  tag = "Mistral.rs",
  path = "/v1/system/info",
  responses((status = 200, description = "Host, device, and build information"))
)]
pub async fn system_info() -> Json<mistralrs_core::SystemInfo> {
    Json(collect_system_info())
}

#[utoipa::path(
  post,
  tag = "Mistral.rs",
  path = "/v1/system/doctor",
  responses((status = 200, description = "Environment diagnostics report"))
)]
pub async fn system_doctor() -> Json<mistralrs_core::DoctorReport> {
    Json(run_doctor())
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

/// Request body for applying online calibration.
#[derive(Debug, Clone, Deserialize, Serialize, ToSchema)]
pub struct CalibrationApplyRequest {
    /// Optionally save the collected imatrix to this `.cimatrix` path before requantizing.
    #[serde(default)]
    pub save_cimatrix: Option<String>,
}

async fn send_calibration(
    state: &crate::types::SharedMistralRsState,
    action: mistralrs_core::CalibrationAction,
) -> Result<axum::Json<mistralrs_core::CalibrationStatus>, String> {
    let (tx, mut rx) = tokio::sync::mpsc::channel(1);
    let request = Request::Calibration(mistralrs_core::CalibrationRequest {
        action,
        response: tx,
    });
    state.get_sender(None).unwrap().send(request).await.unwrap();
    match rx.recv().await {
        Some(Ok(status)) => Ok(axum::Json(status)),
        Some(Err(e)) => Err(e.to_string()),
        None => Err("Engine closed the calibration channel.".to_string()),
    }
}

#[utoipa::path(
  post,
  tag = "Mistral.rs",
  path = "/calibration/start",
  responses((status = 200, description = "Begin collecting activation statistics from live traffic.", body = mistralrs_core::CalibrationStatus))
)]
pub async fn calibration_start(
    State(state): ExtractedMistralRsState,
) -> Result<axum::Json<mistralrs_core::CalibrationStatus>, String> {
    MistralRs::maybe_log_request(state.clone(), "Calibration start".to_string());
    send_calibration(&state, mistralrs_core::CalibrationAction::Start).await
}

#[utoipa::path(
  get,
  tag = "Mistral.rs",
  path = "/calibration/status",
  responses((status = 200, description = "Per-layer calibration collection progress.", body = mistralrs_core::CalibrationStatus))
)]
pub async fn calibration_status(
    State(state): ExtractedMistralRsState,
) -> Result<axum::Json<mistralrs_core::CalibrationStatus>, String> {
    send_calibration(&state, mistralrs_core::CalibrationAction::Status).await
}

#[utoipa::path(
  post,
  tag = "Mistral.rs",
  path = "/calibration/apply",
  request_body = CalibrationApplyRequest,
  responses((status = 200, description = "Requantize with collected statistics and hot-swap the layers.", body = mistralrs_core::CalibrationStatus))
)]
pub async fn calibration_apply(
    State(state): ExtractedMistralRsState,
    Json(request): Json<CalibrationApplyRequest>,
) -> Result<axum::Json<mistralrs_core::CalibrationStatus>, String> {
    MistralRs::maybe_log_request(state.clone(), "Calibration apply".to_string());
    send_calibration(
        &state,
        mistralrs_core::CalibrationAction::Apply {
            save_cimatrix: request.save_cimatrix.map(std::path::PathBuf::from),
        },
    )
    .await
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

#[derive(Debug, Clone, Deserialize, Serialize, ToSchema)]
pub struct TuneModelRequest {
    #[schema(example = "meta-llama/Llama-3.2-3B-Instruct")]
    pub model_id: String,
    /// Optional model dtype (auto, f16, bf16, etc)
    #[serde(default)]
    pub dtype: Option<String>,
    /// Optional max sequence length for tuning
    #[serde(default)]
    pub max_seq_len: Option<usize>,
    /// Optional max batch size for tuning
    #[serde(default)]
    pub max_batch_size: Option<usize>,
    /// Optional max num images (multimodal)
    #[serde(default)]
    pub max_num_images: Option<usize>,
    /// Optional max image length (multimodal)
    #[serde(default)]
    pub max_image_length: Option<usize>,
    /// Optional tuning profile
    #[serde(default)]
    pub profile: Option<TuneProfileRequest>,
    /// Optional fixed ISQ level to test (e.g., Q4K)
    #[serde(default)]
    pub requested_isq: Option<String>,
    /// Optional HF token source
    #[serde(default)]
    pub token_source: Option<String>,
    /// Optional HF revision
    #[serde(default)]
    pub hf_revision: Option<String>,
    /// Force CPU-only tuning
    #[serde(default)]
    pub cpu: Option<bool>,
}

#[utoipa::path(
  post,
  tag = "Mistral.rs",
  path = "/v1/models/tune",
  request_body = TuneModelRequest,
  responses(
    (status = 200, description = "Auto-tune result with recommended settings"),
    (status = 500, description = "Tuning failed")
  )
)]
pub async fn tune_model(
    Json(request): Json<TuneModelRequest>,
) -> Result<Json<AutoTuneResult>, String> {
    let token_source = match request.token_source {
        Some(value) => value
            .parse()
            .map_err(|err| format!("Invalid token_source: {err}"))?,
        None => TokenSource::CacheToken,
    };

    let dtype = request
        .dtype
        .as_deref()
        .unwrap_or("auto")
        .parse::<ModelDType>()
        .map_err(|err| format!("Invalid dtype: {err}"))?;

    let max_seq_len = request
        .max_seq_len
        .unwrap_or(AutoDeviceMapParams::DEFAULT_MAX_SEQ_LEN);
    let max_batch_size = request
        .max_batch_size
        .unwrap_or(AutoDeviceMapParams::DEFAULT_MAX_BATCH_SIZE);

    let model_selected = ModelSelected::Run {
        model_id: request.model_id.clone(),
        tokenizer_json: None,
        dtype,
        topology: None,
        organization: None,
        write_uqff: None,
        from_uqff: None,
        imatrix: None,
        calibration_file: None,
        max_edge: None,
        max_seq_len,
        max_batch_size,
        max_num_images: request.max_num_images,
        max_image_length: request.max_image_length,
        hf_cache_path: None,
        matformer_config_path: None,
        matformer_slice_name: None,
    };

    let requested_isq = match request.requested_isq {
        Some(value) => {
            Some(parse_isq_value(&value, None).map_err(|err| format!("Invalid isq value: {err}"))?)
        }
        None => None,
    };

    let tune_request = AutoTuneRequest {
        model: model_selected,
        token_source,
        hf_revision: request.hf_revision,
        force_cpu: request.cpu.unwrap_or(false),
        profile: request
            .profile
            .map(Into::into)
            .unwrap_or(TuneProfile::Balanced),
        requested_isq,
    };

    auto_tune(tune_request)
        .map(Json)
        .map_err(|err| err.to_string())
}

/// GET `/v1/sessions/{session_id}`. 404 if the session doesn't exist.
#[utoipa::path(
    get,
    tag = "Mistral.rs",
    path = "/v1/sessions/{session_id}",
    params(("session_id" = String, Path, description = "Session ID to export")),
    responses(
        (status = 200, description = "Serialized agentic session", body = SerializedSession),
        (status = 404, description = "Session not found"),
    )
)]
pub async fn get_session(
    State(state): ExtractedMistralRsState,
    Path(session_id): Path<String>,
) -> Result<Json<SerializedSession>, (StatusCode, String)> {
    match state.export_session(None, &session_id) {
        Ok(Some(session)) => Ok(Json(session)),
        Ok(None) => Err((
            StatusCode::NOT_FOUND,
            format!("Session {session_id} not found"),
        )),
        Err(e) => Err((StatusCode::INTERNAL_SERVER_ERROR, e.to_string())),
    }
}

/// PUT `/v1/sessions/{session_id}`. Replaces any existing session.
#[utoipa::path(
    put,
    tag = "Mistral.rs",
    path = "/v1/sessions/{session_id}",
    params(("session_id" = String, Path, description = "Session ID to import as")),
    request_body = SerializedSession,
    responses(
        (status = 200, description = "Session imported"),
        (status = 400, description = "Invalid session payload"),
    )
)]
pub async fn put_session(
    State(state): ExtractedMistralRsState,
    Path(session_id): Path<String>,
    Json(session): Json<SerializedSession>,
) -> Result<StatusCode, (StatusCode, String)> {
    state
        .import_session(None, session_id, session)
        .map(|()| StatusCode::OK)
        .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))
}

/// DELETE `/v1/sessions/{session_id}`. Idempotent: returns 200 either way.
#[utoipa::path(
    delete,
    tag = "Mistral.rs",
    path = "/v1/sessions/{session_id}",
    params(("session_id" = String, Path, description = "Session ID to delete")),
    responses((status = 200, description = "Session deleted (or did not exist)"))
)]
pub async fn delete_session(
    State(state): ExtractedMistralRsState,
    Path(session_id): Path<String>,
) -> Result<StatusCode, (StatusCode, String)> {
    state
        .delete_session(None, &session_id)
        .map(|_| StatusCode::OK)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))
}
