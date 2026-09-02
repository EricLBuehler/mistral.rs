//! ## General mistral.rs server route handlers.

use axum::extract::{rejection::JsonRejection, Json, Path, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use mistralrs_core::{
    auto_tune, collect_system_info, parse_isq_value, run_doctor, AutoDeviceMapParams,
    AutoTuneRequest, MistralRs, MistralRsError, ModelDType, ModelSelected,
    ModelStatus as CoreModelStatus, Request, SerializedSession, TokenSource, TuneProfile,
};
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use crate::{
    handler_core::{openai_error_from_error, openai_error_response, ApiError, ApiErrorKind},
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
  responses(
    (status = 200, description = "Served model info", body = ModelObjects),
    (status = 500, description = "Failed to inspect the model registry")
  )
)]
pub async fn models(State(state): ExtractedMistralRsState) -> Response {
    let mut model_objects = Vec::new();

    let models_with_status = match state.list_models_with_status() {
        Ok(models) => models,
        Err(error) => return openai_error_from_error(&error, ApiErrorKind::Internal),
    };

    if !models_with_status.is_empty() {
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
    }

    for (model_id, status) in models_with_status {
        let (tools_available, mcp_tools_count, mcp_servers_connected) =
            if status == CoreModelStatus::Loaded {
                let tools_count = match state.get_tools_count(Some(&model_id)) {
                    Ok(count) => count,
                    Err(_) => return openai_error_response(ApiError::internal()),
                };
                let has_mcp = match state.has_mcp_client(Some(&model_id)) {
                    Ok(has_mcp) => has_mcp,
                    Err(_) => return openai_error_response(ApiError::internal()),
                };

                if has_mcp || tools_count > 0 {
                    (Some(tools_count > 0), Some(tools_count), Some(1))
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

    let adapter_models = match list_lora_adapter_models(&state) {
        Ok(models) => models,
        Err(error) => return openai_error_from_error(&error, ApiErrorKind::Internal),
    };
    for adapter_model in adapter_models {
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
    .into_response()
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
  responses(
    (status = 200, description = "Reapply ISQ to a model that was loaded with ISQ."),
    (status = 400, description = "Invalid ISQ type"),
    (status = 500, description = "Failed to dispatch the ISQ request")
  )
)]
pub async fn re_isq(
    State(state): ExtractedMistralRsState,
    payload: Result<Json<ReIsqRequest>, JsonRejection>,
) -> Response {
    let request = match payload {
        Ok(Json(request)) => request,
        Err(error) => return openai_error_response(ApiError::from_json_rejection(error)),
    };
    let repr = format!("Re ISQ: {:?}", request.ggml_type);
    MistralRs::maybe_log_request(state.clone(), repr.clone());
    let level = match parse_isq_value(&request.ggml_type, None) {
        Ok(level) => level,
        Err(error) => {
            return openai_error_response(ApiError::new(
                ApiErrorKind::InvalidRequest,
                error,
                Some("invalid_isq"),
                Some("ggml_type"),
            ));
        }
    };
    let sender = match state.get_sender(None) {
        Ok(sender) => sender,
        Err(error) => return openai_error_from_error(&error, ApiErrorKind::Internal),
    };
    if let Err(error) = sender.send(Request::ReIsq(level)).await {
        tracing::error!(%error, "failed to dispatch ISQ request");
        return openai_error_response(ApiError::internal());
    }
    (StatusCode::OK, repr).into_response()
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
) -> Response {
    let (tx, mut rx) = tokio::sync::mpsc::channel(1);
    let request = Request::Calibration(mistralrs_core::CalibrationRequest {
        action,
        response: tx,
    });
    let sender = match state.get_sender(None) {
        Ok(sender) => sender,
        Err(error) => return openai_error_from_error(&error, ApiErrorKind::Internal),
    };
    if let Err(error) = sender.send(request).await {
        tracing::error!(%error, "failed to dispatch calibration request");
        return openai_error_response(ApiError::internal());
    }
    match rx.recv().await {
        Some(Ok(status)) => Json(status).into_response(),
        Some(Err(error)) => {
            MistralRs::maybe_log_error(state.clone(), error.as_ref());
            openai_error_response(ApiError::internal())
        }
        None => openai_error_response(ApiError::internal()),
    }
}

#[utoipa::path(
  post,
  tag = "Mistral.rs",
  path = "/calibration/start",
  responses((status = 200, description = "Begin collecting activation statistics from live traffic.", body = mistralrs_core::CalibrationStatus))
)]
pub async fn calibration_start(State(state): ExtractedMistralRsState) -> Response {
    MistralRs::maybe_log_request(state.clone(), "Calibration start".to_string());
    send_calibration(&state, mistralrs_core::CalibrationAction::Start).await
}

#[utoipa::path(
  get,
  tag = "Mistral.rs",
  path = "/calibration/status",
  responses((status = 200, description = "Per-layer calibration collection progress.", body = mistralrs_core::CalibrationStatus))
)]
pub async fn calibration_status(State(state): ExtractedMistralRsState) -> Response {
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
    payload: Result<Json<CalibrationApplyRequest>, JsonRejection>,
) -> Response {
    let request = match payload {
        Ok(Json(request)) => request,
        Err(error) => return openai_error_response(ApiError::from_json_rejection(error)),
    };
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
}

/// Response for model status operations
#[derive(Debug, Clone, Deserialize, Serialize, ToSchema)]
pub struct ModelStatusResponse {
    #[schema(example = "my-model")]
    pub model_id: String,
    pub status: ModelStatus,
}

fn model_operation_request(
    payload: Result<Json<ModelOperationRequest>, JsonRejection>,
) -> Result<ModelOperationRequest, ApiError> {
    payload
        .map(|Json(request)| request)
        .map_err(ApiError::from_json_rejection)
}

fn model_status_response(model_id: String, status: ModelStatus) -> Response {
    Json(ModelStatusResponse { model_id, status }).into_response()
}

fn unload_model_result(model_id: String, result: Result<(), MistralRsError>) -> Response {
    match result {
        Ok(()) | Err(MistralRsError::ModelAlreadyUnloaded(_)) => {
            model_status_response(model_id, ModelStatus::Unloaded)
        }
        Err(error) => openai_error_from_error(&error, ApiErrorKind::Internal),
    }
}

fn reload_model_result(model_id: String, result: Result<(), MistralRsError>) -> Response {
    match result {
        Ok(()) | Err(MistralRsError::ModelAlreadyLoaded(_)) => {
            model_status_response(model_id, ModelStatus::Loaded)
        }
        Err(error) => openai_error_from_error(&error, ApiErrorKind::Internal),
    }
}

fn get_model_status_result(
    model_id: String,
    result: Result<Option<CoreModelStatus>, MistralRsError>,
) -> Response {
    match result {
        Ok(Some(CoreModelStatus::Loaded)) => model_status_response(model_id, ModelStatus::Loaded),
        Ok(Some(CoreModelStatus::Unloaded)) => {
            model_status_response(model_id, ModelStatus::Unloaded)
        }
        Ok(Some(CoreModelStatus::Reloading)) => {
            model_status_response(model_id, ModelStatus::Reloading)
        }
        Ok(None) => openai_error_from_error(
            &MistralRsError::ModelNotFound(model_id),
            ApiErrorKind::Internal,
        ),
        Err(error) => openai_error_from_error(&error, ApiErrorKind::Internal),
    }
}

#[utoipa::path(
  post,
  tag = "Mistral.rs",
  path = "/v1/models/unload",
  request_body = ModelOperationRequest,
  responses(
    (status = 200, description = "Model unloaded or already unloaded", body = ModelStatusResponse),
    (status = 400, description = "Invalid request or model cannot be unloaded"),
    (status = 404, description = "Model not found"),
    (status = 409, description = "Model state conflicts with the operation"),
    (status = 413, description = "Request body is too large"),
    (status = 415, description = "Request content type is not JSON"),
    (status = 500, description = "Model registry failure")
  )
)]
pub async fn unload_model(
    State(state): ExtractedMistralRsState,
    payload: Result<Json<ModelOperationRequest>, JsonRejection>,
) -> Response {
    let request = match model_operation_request(payload) {
        Ok(request) => request,
        Err(error) => return openai_error_response(error),
    };
    let model_id = request.model_id;
    let result = state.unload_model(&model_id);
    unload_model_result(model_id, result)
}

#[utoipa::path(
  post,
  tag = "Mistral.rs",
  path = "/v1/models/reload",
  request_body = ModelOperationRequest,
  responses(
    (status = 200, description = "Model reloaded or already loaded", body = ModelStatusResponse),
    (status = 400, description = "Invalid request or model cannot be reloaded"),
    (status = 404, description = "Model not found"),
    (status = 409, description = "Model state conflicts with the operation"),
    (status = 413, description = "Request body is too large"),
    (status = 415, description = "Request content type is not JSON"),
    (status = 500, description = "Model reload failure")
  )
)]
pub async fn reload_model(
    State(state): ExtractedMistralRsState,
    payload: Result<Json<ModelOperationRequest>, JsonRejection>,
) -> Response {
    let request = match model_operation_request(payload) {
        Ok(request) => request,
        Err(error) => return openai_error_response(error),
    };
    let model_id = request.model_id;
    let result = state.reload_model(&model_id).await;
    reload_model_result(model_id, result)
}

#[utoipa::path(
  post,
  tag = "Mistral.rs",
  path = "/v1/models/status",
  request_body = ModelOperationRequest,
  responses(
    (status = 200, description = "Model status", body = ModelStatusResponse),
    (status = 400, description = "Invalid request"),
    (status = 404, description = "Model not found"),
    (status = 413, description = "Request body is too large"),
    (status = 415, description = "Request content type is not JSON"),
    (status = 500, description = "Model registry failure")
  )
)]
pub async fn get_model_status(
    State(state): ExtractedMistralRsState,
    payload: Result<Json<ModelOperationRequest>, JsonRejection>,
) -> Response {
    let request = match model_operation_request(payload) {
        Ok(request) => request,
        Err(error) => return openai_error_response(error),
    };
    let model_id = request.model_id;
    let result = state.get_model_status(&model_id);
    get_model_status_result(model_id, result)
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
    (status = 400, description = "Invalid tuning request"),
    (status = 413, description = "Request body is too large"),
    (status = 415, description = "Request content type is not JSON"),
    (status = 500, description = "Tuning failed")
  )
)]
pub async fn tune_model(payload: Result<Json<TuneModelRequest>, JsonRejection>) -> Response {
    let request = match payload {
        Ok(Json(request)) => request,
        Err(error) => return openai_error_response(ApiError::from_json_rejection(error)),
    };
    let token_source = match request.token_source {
        Some(value) => match value.parse() {
            Ok(token_source) => token_source,
            Err(error) => {
                return openai_error_response(ApiError::new(
                    ApiErrorKind::InvalidRequest,
                    format!("Invalid token_source: {error}"),
                    Some("invalid_token_source"),
                    Some("token_source"),
                ));
            }
        },
        None => TokenSource::CacheToken,
    };

    let dtype = match request
        .dtype
        .as_deref()
        .unwrap_or("auto")
        .parse::<ModelDType>()
    {
        Ok(dtype) => dtype,
        Err(error) => {
            return openai_error_response(ApiError::new(
                ApiErrorKind::InvalidRequest,
                format!("Invalid dtype: {error}"),
                Some("invalid_dtype"),
                Some("dtype"),
            ));
        }
    };

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
        Some(value) => match parse_isq_value(&value, None) {
            Ok(value) => Some(value),
            Err(error) => {
                return openai_error_response(ApiError::new(
                    ApiErrorKind::InvalidRequest,
                    format!("Invalid isq value: {error}"),
                    Some("invalid_isq"),
                    Some("requested_isq"),
                ));
            }
        },
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

    match auto_tune(tune_request) {
        Ok(result) => Json(result).into_response(),
        Err(error) => {
            tracing::error!(%error, "model auto-tuning failed");
            openai_error_response(ApiError::internal())
        }
    }
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
) -> Response {
    match state.export_session(None, &session_id) {
        Ok(Some(session)) => Json(session).into_response(),
        Ok(None) => openai_error_response(ApiError::new(
            ApiErrorKind::NotFound,
            format!("Session '{session_id}' was not found."),
            Some("session_not_found"),
            Some("session_id"),
        )),
        Err(error) => openai_error_from_error(&error, ApiErrorKind::Internal),
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
    payload: Result<Json<SerializedSession>, JsonRejection>,
) -> Response {
    let session = match payload {
        Ok(Json(session)) => session,
        Err(error) => return openai_error_response(ApiError::from_json_rejection(error)),
    };
    match state.import_session(None, session_id, session) {
        Ok(()) => StatusCode::OK.into_response(),
        Err(MistralRsError::Other(message)) => openai_error_response(ApiError::new(
            ApiErrorKind::InvalidRequest,
            message,
            Some("invalid_session"),
            None,
        )),
        Err(error) => openai_error_from_error(&error, ApiErrorKind::Internal),
    }
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
) -> Response {
    match state.delete_session(None, &session_id) {
        Ok(_) => StatusCode::OK.into_response(),
        Err(error) => openai_error_from_error(&error, ApiErrorKind::Internal),
    }
}

#[cfg(test)]
mod tests {
    use axum::{
        body::{to_bytes, Body},
        extract::FromRequest,
        http::Request as HttpRequest,
    };

    use super::*;

    #[test]
    fn unload_model_results_use_operation_statuses() {
        assert_eq!(
            unload_model_result("model".to_string(), Ok(())).status(),
            StatusCode::OK
        );
        assert_eq!(
            unload_model_result(
                "model".to_string(),
                Err(MistralRsError::ModelAlreadyUnloaded("model".to_string())),
            )
            .status(),
            StatusCode::OK
        );
        assert_eq!(
            unload_model_result(
                "missing".to_string(),
                Err(MistralRsError::ModelNotFound("missing".to_string())),
            )
            .status(),
            StatusCode::NOT_FOUND
        );
        assert_eq!(
            unload_model_result(
                "model".to_string(),
                Err(MistralRsError::NoLoaderConfig("model".to_string())),
            )
            .status(),
            StatusCode::BAD_REQUEST
        );
        assert_eq!(
            unload_model_result(
                "model".to_string(),
                Err(MistralRsError::ModelReloading("model".to_string())),
            )
            .status(),
            StatusCode::CONFLICT
        );
        assert_eq!(
            unload_model_result("model".to_string(), Err(MistralRsError::EnginePoisoned),).status(),
            StatusCode::INTERNAL_SERVER_ERROR
        );
    }

    #[test]
    fn reload_and_status_results_preserve_idempotency() {
        assert_eq!(
            reload_model_result(
                "model".to_string(),
                Err(MistralRsError::ModelAlreadyLoaded("model".to_string())),
            )
            .status(),
            StatusCode::OK
        );
        assert_eq!(
            reload_model_result(
                "model".to_string(),
                Err(MistralRsError::ModelReloading("model".to_string())),
            )
            .status(),
            StatusCode::CONFLICT
        );
        assert_eq!(
            get_model_status_result("missing".to_string(), Ok(None)).status(),
            StatusCode::NOT_FOUND
        );
        assert_eq!(
            get_model_status_result("model".to_string(), Ok(Some(CoreModelStatus::Reloading)),)
                .status(),
            StatusCode::OK
        );
    }

    #[tokio::test]
    async fn lifecycle_json_rejections_use_openai_statuses() {
        let request = HttpRequest::builder()
            .header(axum::http::header::CONTENT_TYPE, "application/json")
            .body(Body::from("{}"))
            .unwrap();
        let rejection = Json::<ModelOperationRequest>::from_request(request, &())
            .await
            .unwrap_err();
        let error = model_operation_request(Err(rejection)).unwrap_err();

        assert_eq!(error.status(), StatusCode::BAD_REQUEST);
        assert_eq!(error.code.as_deref(), Some("invalid_request_body"));
    }

    #[tokio::test]
    async fn internal_model_errors_do_not_expose_details() {
        let response = reload_model_result(
            "model".to_string(),
            Err(MistralRsError::ReloadFailed("private failure".to_string())),
        );
        assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);

        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let body = String::from_utf8(body.to_vec()).unwrap();
        assert!(!body.contains("private failure"));
        assert!(body.contains("Internal server error"));
    }
}
