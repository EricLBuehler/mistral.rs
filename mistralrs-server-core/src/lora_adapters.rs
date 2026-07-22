use std::{
    collections::HashSet,
    fs::File,
    path::{Path, PathBuf},
};

use anyhow::{bail, Context};
use axum::{
    extract::{
        rejection::{JsonRejection, QueryRejection},
        Query, State,
    },
    http::StatusCode,
    response::{IntoResponse, Response},
    Extension, Json,
};
use mistralrs_core::{
    LoraAdapterError, LoraAdapterFiles, LoraAdapterInfo, LoraAdapterLoadPolicy, LoraAdapterRoute,
    MistralRs, MistralRsError, MAX_LORA_ALIAS_BYTES,
};
use serde::{Deserialize, Serialize};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use utoipa::{IntoParams, ToSchema};

use crate::types::ExtractedMistralRsState;

pub const ALLOW_RUNTIME_LORA_UPDATING_ENV: &str = "MISTRALRS_ALLOW_RUNTIME_LORA_UPDATING";
pub const LORA_ADAPTER_ROOT_ENV: &str = "MISTRALRS_LORA_ADAPTER_ROOT";

const LORA_ADAPTER_OBJECT: &str = "lora_adapter";
const LORA_ADAPTER_LIST_OBJECT: &str = "list";
const LORA_CONFIG_FILE: &str = "adapter_config.json";
const LORA_WEIGHTS_FILE: &str = "adapter_model.safetensors";
const DEFAULT_MODEL_ID: &str = "default";
const MAX_CONCURRENT_HTTP_LORA_LOADS: usize = 1;

#[derive(Clone, Debug)]
pub(crate) struct LoraAdapterModel {
    pub id: String,
    pub parent: String,
    pub adapter: LoraAdapterInfo,
}

pub(crate) fn list_lora_adapter_models(
    state: &MistralRs,
) -> Result<Vec<LoraAdapterModel>, MistralRsError> {
    let routes = state.list_lora_adapter_routes()?;
    adapter_models_from_routes(routes, |model_id| state.model_exists(model_id))
}

fn adapter_model_id_component(value: &str) -> String {
    let mut encoded = String::with_capacity(value.len());
    for character in value.chars() {
        match character {
            '%' => encoded.push_str("%25"),
            ':' => encoded.push_str("%3A"),
            '#' => encoded.push_str("%23"),
            character => encoded.push(character),
        }
    }
    encoded
}

fn adapter_models_from_routes(
    routes: Vec<LoraAdapterRoute>,
    mut model_exists: impl FnMut(&str) -> Result<bool, MistralRsError>,
) -> Result<Vec<LoraAdapterModel>, MistralRsError> {
    let mut used = HashSet::new();
    let mut models = Vec::with_capacity(routes.len());
    for route in routes {
        let alias = &route.adapter.alias;
        let mut id = format!(
            "{}::{}",
            adapter_model_id_component(&route.model_id),
            adapter_model_id_component(alias)
        );
        let base_id = id.clone();
        let mut suffix = 2usize;
        while used.contains(&id) || model_exists(&id)? || id == DEFAULT_MODEL_ID {
            id = format!("{base_id}#{suffix}");
            suffix += 1;
        }
        used.insert(id.clone());
        models.push(LoraAdapterModel {
            id,
            parent: route.model_id,
            adapter: route.adapter,
        });
    }
    Ok(models)
}

fn select_lora_adapter_model(
    selected: &LoraAdapterModel,
    model: &mut String,
    adapter: &mut Option<crate::openai::AdapterSelection>,
) -> Result<(), String> {
    if adapter.is_some() {
        return Err(format!(
            "model `{}` already selects LoRA adapter `{}`; omit `adapter` or use base model `{}`",
            selected.id, selected.adapter.alias, selected.parent
        ));
    }
    *model = selected.parent.clone();
    *adapter = Some(crate::openai::AdapterSelection::Alias(
        selected.adapter.alias.clone(),
    ));
    Ok(())
}

fn resolve_lora_adapter_model_from_models(
    models: &[LoraAdapterModel],
    model: &mut String,
    adapter: &mut Option<crate::openai::AdapterSelection>,
) -> Result<(), String> {
    if let Some(selected) = models.iter().find(|candidate| candidate.id == *model) {
        return select_lora_adapter_model(selected, model, adapter);
    }

    let alias_matches = models
        .iter()
        .filter(|candidate| candidate.adapter.alias == *model)
        .collect::<Vec<_>>();
    match alias_matches.as_slice() {
        [] => Ok(()),
        [selected] => select_lora_adapter_model(selected, model, adapter),
        _ => Err(format!(
            "LoRA adapter model `{model}` is ambiguous; use one of: {}",
            alias_matches
                .iter()
                .map(|candidate| candidate.id.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        )),
    }
}

pub(crate) fn is_resolvable_lora_adapter_model(models: &[LoraAdapterModel], model: &str) -> bool {
    models.iter().any(|candidate| candidate.id == model)
        || models
            .iter()
            .filter(|candidate| candidate.adapter.alias == model)
            .take(2)
            .count()
            == 1
}

pub(crate) fn resolve_lora_adapter_model(
    state: &MistralRs,
    model: &mut String,
    adapter: &mut Option<crate::openai::AdapterSelection>,
) -> Result<(), String> {
    if model == DEFAULT_MODEL_ID
        || state
            .model_exists(model)
            .map_err(|error| error.to_string())?
    {
        return Ok(());
    }

    let models = list_lora_adapter_models(state).map_err(|error| error.to_string())?;
    resolve_lora_adapter_model_from_models(&models, model, adapter)
}

/// Controls exposure and filesystem access for runtime LoRA management routes.
#[derive(Clone, Debug)]
pub struct LoraAdapterApiConfig {
    enabled: bool,
    allowed_root: Option<PathBuf>,
    load_gate: std::sync::Arc<Semaphore>,
}

impl Default for LoraAdapterApiConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            allowed_root: None,
            load_gate: std::sync::Arc::new(Semaphore::new(MAX_CONCURRENT_HTTP_LORA_LOADS)),
        }
    }
}

impl LoraAdapterApiConfig {
    pub fn from_env() -> Self {
        Self {
            enabled: runtime_lora_updates_enabled(),
            allowed_root: std::env::var_os(LORA_ADAPTER_ROOT_ENV)
                .filter(|value| !value.is_empty())
                .map(PathBuf::from),
            ..Self::default()
        }
    }

    pub fn with_enabled(mut self, enabled: bool) -> Self {
        self.enabled = enabled;
        self
    }

    pub fn with_allowed_root(mut self, root: impl Into<PathBuf>) -> Self {
        self.allowed_root = Some(root.into());
        self
    }

    pub fn enabled(&self) -> bool {
        self.enabled
    }

    pub fn allowed_root(&self) -> Option<&Path> {
        self.allowed_root.as_deref()
    }

    pub(crate) fn prepare(mut self) -> anyhow::Result<Self> {
        if !self.enabled {
            return Ok(self);
        }
        if let Some(root) = &self.allowed_root {
            let root = root.canonicalize().with_context(|| {
                format!("failed to resolve LoRA adapter root `{}`", root.display())
            })?;
            let metadata = root.metadata().with_context(|| {
                format!("failed to inspect LoRA adapter root `{}`", root.display())
            })?;
            if !metadata.is_dir() {
                bail!("LoRA adapter root `{}` is not a directory", root.display());
            }
            self.allowed_root = Some(root);
        }
        Ok(self)
    }

    fn open_adapter_files(&self, path: &Path) -> Result<OpenedAdapterFiles, LoraAdapterApiError> {
        let path = if path.is_relative() {
            self.allowed_root()
                .map(|root| root.join(path))
                .unwrap_or_else(|| path.to_path_buf())
        } else {
            path.to_path_buf()
        };
        let description = format!("adapter directory `{}`", path.display());
        let adapter_dir = path
            .canonicalize()
            .map_err(|error| adapter_filesystem_error(&description, error))?;
        let metadata = adapter_dir
            .metadata()
            .map_err(|error| adapter_filesystem_error(&description, error))?;
        if !metadata.is_dir() {
            return Err(LoraAdapterApiError::new(
                StatusCode::BAD_REQUEST,
                "invalid_adapter_path",
                format!(
                    "adapter path `{}` is not a directory",
                    adapter_dir.display()
                ),
            ));
        }
        self.ensure_allowed(&adapter_dir, "adapter directory")?;
        let (config_path, config) = self.open_adapter_file(&adapter_dir, LORA_CONFIG_FILE)?;
        let (weights_path, weights) = self.open_adapter_file(&adapter_dir, LORA_WEIGHTS_FILE)?;
        Ok(OpenedAdapterFiles {
            source: adapter_dir.display().to_string(),
            config_path,
            weights_path,
            config,
            weights,
        })
    }

    fn open_adapter_file(
        &self,
        adapter_dir: &Path,
        filename: &str,
    ) -> Result<(PathBuf, File), LoraAdapterApiError> {
        let path = adapter_dir.join(filename);
        let description = format!("adapter file `{filename}`");
        let path = path
            .canonicalize()
            .map_err(|error| adapter_filesystem_error(&description, error))?;
        self.ensure_allowed(&path, filename)?;
        let metadata = path
            .metadata()
            .map_err(|error| adapter_filesystem_error(&description, error))?;
        if !metadata.is_file() {
            return Err(LoraAdapterApiError::new(
                StatusCode::BAD_REQUEST,
                "invalid_adapter_file",
                format!("adapter file `{filename}` is not a regular file"),
            ));
        }
        let file =
            File::open(&path).map_err(|error| adapter_filesystem_error(&description, error))?;
        let verified_path = path
            .canonicalize()
            .map_err(|error| adapter_filesystem_error(&description, error))?;
        self.ensure_allowed(&verified_path, filename)?;
        verify_open_file(&file, &verified_path, filename)?;
        Ok((verified_path, file))
    }

    fn ensure_allowed(&self, path: &Path, description: &str) -> Result<(), LoraAdapterApiError> {
        if self
            .allowed_root()
            .is_some_and(|root| !path.starts_with(root))
        {
            return Err(LoraAdapterApiError::new(
                StatusCode::FORBIDDEN,
                "adapter_path_forbidden",
                format!("{description} resolves outside the configured LoRA adapter root"),
            ));
        }
        Ok(())
    }

    fn try_begin_load(&self) -> Result<OwnedSemaphorePermit, LoraAdapterApiError> {
        self.load_gate.clone().try_acquire_owned().map_err(|_| {
            LoraAdapterApiError::new(
                StatusCode::TOO_MANY_REQUESTS,
                "lora_load_busy",
                "another LoRA adapter load is already in progress",
            )
        })
    }
}

#[derive(Debug)]
struct OpenedAdapterFiles {
    source: String,
    config_path: PathBuf,
    weights_path: PathBuf,
    config: File,
    weights: File,
}

impl OpenedAdapterFiles {
    fn into_runtime_files(self) -> LoraAdapterFiles {
        LoraAdapterFiles::new(
            self.source,
            self.config_path,
            self.config,
            self.weights_path,
            self.weights,
        )
    }
}

fn adapter_filesystem_error(description: &str, error: std::io::Error) -> LoraAdapterApiError {
    let (status, code) = match error.kind() {
        std::io::ErrorKind::NotFound => (StatusCode::NOT_FOUND, "adapter_path_not_found"),
        std::io::ErrorKind::PermissionDenied => (StatusCode::FORBIDDEN, "adapter_path_forbidden"),
        std::io::ErrorKind::NotADirectory | std::io::ErrorKind::IsADirectory => {
            (StatusCode::BAD_REQUEST, "invalid_adapter_path")
        }
        std::io::ErrorKind::InvalidData | std::io::ErrorKind::InvalidInput => {
            (StatusCode::UNPROCESSABLE_ENTITY, "invalid_adapter_path")
        }
        _ => (
            StatusCode::SERVICE_UNAVAILABLE,
            "adapter_storage_unavailable",
        ),
    };
    LoraAdapterApiError::new(
        status,
        code,
        format!("failed to access {description}: {error}"),
    )
}

fn adapter_task_error(action: &str, error: tokio::task::JoinError) -> LoraAdapterApiError {
    LoraAdapterApiError::new(
        StatusCode::INTERNAL_SERVER_ERROR,
        "lora_load_task_failed",
        format!("{action} task failed: {error}"),
    )
}

fn verify_open_file(file: &File, path: &Path, filename: &str) -> Result<(), LoraAdapterApiError> {
    let opened = file
        .metadata()
        .map_err(|error| adapter_filesystem_error(&format!("adapter file `{filename}`"), error))?;
    if !opened.is_file() {
        return Err(LoraAdapterApiError::new(
            StatusCode::BAD_REQUEST,
            "invalid_adapter_file",
            format!("adapter file `{filename}` is not a regular file"),
        ));
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;

        let current = path.metadata().map_err(|error| {
            adapter_filesystem_error(&format!("adapter file `{filename}`"), error)
        })?;
        if opened.dev() != current.dev() || opened.ino() != current.ino() {
            return Err(LoraAdapterApiError::new(
                StatusCode::CONFLICT,
                "adapter_file_changed",
                format!("adapter file `{filename}` changed while it was being opened"),
            ));
        }
    }
    Ok(())
}

fn normalize_model_id(model: Option<String>) -> Option<String> {
    model.and_then(|model| {
        let model = model.trim();
        (!model.is_empty() && model != DEFAULT_MODEL_ID).then(|| model.to_string())
    })
}

pub fn runtime_lora_updates_enabled() -> bool {
    std::env::var(ALLOW_RUNTIME_LORA_UPDATING_ENV)
        .ok()
        .is_some_and(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
}

#[derive(Debug, Clone, Deserialize, Serialize, ToSchema)]
#[serde(deny_unknown_fields)]
pub struct LoadLoraAdapterRequest {
    /// Request-facing adapter alias.
    #[schema(example = "production")]
    pub lora_name: String,
    /// Local server filesystem directory containing PEFT safetensors files.
    #[schema(example = "/srv/adapters/production-v2")]
    pub lora_path: String,
    /// Atomically replace an existing alias. Defaults to false, matching vLLM.
    #[serde(default, alias = "replace")]
    #[schema(default = false, example = false)]
    pub load_inplace: bool,
    /// Replace only if the alias still points at this generation.
    #[serde(default)]
    #[schema(value_type = Option<String>, example = "5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a")]
    pub expected_generation: Option<mistralrs_core::AdapterGenerationId>,
    #[serde(default)]
    #[schema(ignore)]
    is_3d_lora_weight: bool,
    #[serde(default)]
    pub model: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize, ToSchema)]
#[serde(deny_unknown_fields)]
pub struct UnloadLoraAdapterRequest {
    pub lora_name: String,
    /// Remove only if the alias still points at this generation.
    #[serde(default)]
    #[schema(value_type = Option<String>, example = "5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a")]
    pub expected_generation: Option<mistralrs_core::AdapterGenerationId>,
    /// Accepted for vLLM request compatibility; aliases are authoritative in mistral.rs.
    #[serde(default)]
    pub lora_int_id: Option<u64>,
    #[serde(default)]
    pub model: Option<String>,
}

#[derive(Debug, Clone, Deserialize, IntoParams)]
#[into_params(parameter_in = Query)]
pub struct ListLoraAdaptersQuery {
    #[serde(default)]
    pub model: Option<String>,
}

#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct LoraAdapterObject {
    pub id: String,
    pub object: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub revision: Option<String>,
    pub generation: String,
    pub rank: usize,
    pub bytes: u64,
}

impl LoraAdapterObject {
    fn from_info(info: LoraAdapterInfo, expose_source: bool) -> Self {
        Self {
            id: info.alias,
            object: LORA_ADAPTER_OBJECT.to_string(),
            source: expose_source.then_some(info.source),
            revision: info.revision,
            generation: info.generation.to_string(),
            rank: info.rank,
            bytes: info.bytes,
        }
    }
}

#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct LoraAdapterListResponse {
    pub object: String,
    pub data: Vec<LoraAdapterObject>,
    pub generations: Vec<LoraResidentGenerationObject>,
    pub resident_generations: usize,
    pub retired_generations: usize,
    pub resident_bytes: u64,
    pub max_adapters: usize,
    pub max_rank: usize,
    pub max_bytes: u64,
}

#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct LoraResidentGenerationObject {
    pub generation: String,
    pub aliases: Vec<String>,
    pub rank: usize,
    pub bytes: u64,
    pub retired: bool,
    pub active_leases: usize,
}

impl From<mistralrs_core::LoraResidentGenerationInfo> for LoraResidentGenerationObject {
    fn from(info: mistralrs_core::LoraResidentGenerationInfo) -> Self {
        Self {
            generation: info.generation.to_string(),
            aliases: info.aliases,
            rank: info.rank,
            bytes: info.bytes,
            retired: info.retired,
            active_leases: info.active_leases,
        }
    }
}

#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct LoraAdapterErrorResponse {
    pub error: LoraAdapterErrorBody,
}

#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct LoraAdapterErrorBody {
    pub message: String,
    #[serde(rename = "type")]
    pub error_type: String,
    pub code: String,
}

#[derive(Debug)]
pub(crate) struct LoraAdapterApiError {
    status: StatusCode,
    body: LoraAdapterErrorResponse,
}

impl LoraAdapterApiError {
    fn new(status: StatusCode, code: &str, message: impl Into<String>) -> Self {
        let error_type = if status.is_server_error() {
            "server_error"
        } else {
            "invalid_request_error"
        };
        Self {
            status,
            body: LoraAdapterErrorResponse {
                error: LoraAdapterErrorBody {
                    message: message.into(),
                    error_type: error_type.to_string(),
                    code: code.to_string(),
                },
            },
        }
    }

    fn core(error: MistralRsError) -> Self {
        match error {
            MistralRsError::LoraAdapter(error) => Self::lora(error),
            error @ MistralRsError::ModelNotFound(_) => {
                Self::new(StatusCode::NOT_FOUND, "model_not_found", error.to_string())
            }
            error @ MistralRsError::ModelReloading(_)
            | error @ MistralRsError::ModelAlreadyLoaded(_)
            | error @ MistralRsError::ModelAlreadyUnloaded(_) => Self::new(
                StatusCode::CONFLICT,
                "model_state_conflict",
                error.to_string(),
            ),
            error @ MistralRsError::NoLoaderConfig(_) => Self::new(
                StatusCode::BAD_REQUEST,
                "invalid_model_operation",
                error.to_string(),
            ),
            error @ MistralRsError::EnginePoisoned
            | error @ MistralRsError::SenderPoisoned
            | error @ MistralRsError::ReloadFailed(_)
            | error @ MistralRsError::Other(_) => Self::new(
                StatusCode::INTERNAL_SERVER_ERROR,
                "internal_error",
                error.to_string(),
            ),
        }
    }

    fn json_rejection(error: JsonRejection) -> Self {
        let status = error.status();
        let code = match status {
            StatusCode::PAYLOAD_TOO_LARGE => "request_body_too_large",
            StatusCode::UNSUPPORTED_MEDIA_TYPE => "invalid_content_type",
            StatusCode::UNPROCESSABLE_ENTITY => "invalid_request_body",
            _ => "malformed_json",
        };
        Self::new(status, code, error.body_text())
    }

    fn query_rejection(error: QueryRejection) -> Self {
        Self::new(error.status(), "invalid_query", error.body_text())
    }

    fn lora(error: LoraAdapterError) -> Self {
        let (status, code) = match &error {
            LoraAdapterError::RuntimeUnavailable { .. }
            | LoraAdapterError::TensorParallelUnsupported { .. }
            | LoraAdapterError::RuntimeChanged { .. } => {
                (StatusCode::CONFLICT, "lora_runtime_unavailable")
            }
            LoraAdapterError::InvalidAlias => (StatusCode::BAD_REQUEST, "invalid_lora_name"),
            LoraAdapterError::AliasTooLong { .. } => (StatusCode::BAD_REQUEST, "invalid_lora_name"),
            LoraAdapterError::AliasLimit { .. } => {
                (StatusCode::CONFLICT, "lora_alias_limit_exceeded")
            }
            LoraAdapterError::LoadBusy => (StatusCode::TOO_MANY_REQUESTS, "lora_load_busy"),
            LoraAdapterError::AlreadyLoaded { .. } => {
                (StatusCode::CONFLICT, "lora_adapter_already_loaded")
            }
            LoraAdapterError::GenerationMismatch { .. } => {
                (StatusCode::CONFLICT, "lora_generation_mismatch")
            }
            LoraAdapterError::NotFound { .. } | LoraAdapterError::GenerationNotFound { .. } => {
                (StatusCode::NOT_FOUND, "lora_adapter_not_found")
            }
            LoraAdapterError::GenerationConflict { .. } => {
                (StatusCode::CONFLICT, "lora_generation_conflict")
            }
            LoraAdapterError::RankLimit { .. } => {
                (StatusCode::CONFLICT, "lora_rank_limit_exceeded")
            }
            LoraAdapterError::AdapterLimit { .. } => {
                (StatusCode::CONFLICT, "lora_adapter_limit_exceeded")
            }
            LoraAdapterError::ByteLimit { .. } => {
                (StatusCode::CONFLICT, "lora_byte_limit_exceeded")
            }
            LoraAdapterError::SlotExhausted => (StatusCode::CONFLICT, "lora_slot_space_exhausted"),
            LoraAdapterError::InvalidRuntimeConfig(_) => {
                (StatusCode::INTERNAL_SERVER_ERROR, "invalid_lora_runtime")
            }
            LoraAdapterError::SizeOverflow => {
                (StatusCode::UNPROCESSABLE_ENTITY, "invalid_lora_adapter")
            }
            LoraAdapterError::FileTooLarge { .. } => {
                (StatusCode::PAYLOAD_TOO_LARGE, "lora_adapter_file_too_large")
            }
            LoraAdapterError::Io { source, .. }
                if source.kind() == std::io::ErrorKind::NotFound =>
            {
                (StatusCode::NOT_FOUND, "adapter_file_not_found")
            }
            LoraAdapterError::Io { source, .. }
                if source.kind() == std::io::ErrorKind::PermissionDenied =>
            {
                (StatusCode::FORBIDDEN, "adapter_file_forbidden")
            }
            LoraAdapterError::Io { source, .. }
                if matches!(
                    source.kind(),
                    std::io::ErrorKind::InvalidData
                        | std::io::ErrorKind::InvalidInput
                        | std::io::ErrorKind::UnexpectedEof
                ) =>
            {
                (StatusCode::UNPROCESSABLE_ENTITY, "invalid_lora_adapter")
            }
            LoraAdapterError::Io { .. } => {
                (StatusCode::SERVICE_UNAVAILABLE, "lora_storage_unavailable")
            }
            LoraAdapterError::Config { .. } | LoraAdapterError::Format(_) => {
                (StatusCode::UNPROCESSABLE_ENTITY, "invalid_lora_adapter")
            }
            LoraAdapterError::Load(_) => {
                (StatusCode::SERVICE_UNAVAILABLE, "lora_device_load_failed")
            }
            LoraAdapterError::Task(_) => {
                (StatusCode::INTERNAL_SERVER_ERROR, "lora_load_task_failed")
            }
            _ => (StatusCode::INTERNAL_SERVER_ERROR, "internal_error"),
        };
        Self::new(status, code, error.to_string())
    }
}

fn lifecycle_json<T>(payload: Result<Json<T>, JsonRejection>) -> Result<T, LoraAdapterApiError> {
    payload
        .map(|Json(request)| request)
        .map_err(LoraAdapterApiError::json_rejection)
}

fn lifecycle_query<T>(payload: Result<Query<T>, QueryRejection>) -> Result<T, LoraAdapterApiError> {
    payload
        .map(|Query(query)| query)
        .map_err(LoraAdapterApiError::query_rejection)
}

impl IntoResponse for LoraAdapterApiError {
    fn into_response(self) -> Response {
        let retry = self.status == StatusCode::TOO_MANY_REQUESTS;
        let mut response = (self.status, Json(self.body)).into_response();
        if retry {
            response.headers_mut().insert(
                axum::http::header::RETRY_AFTER,
                axum::http::HeaderValue::from_static("1"),
            );
        }
        response
    }
}

pub(crate) fn lifecycle_body_too_large_response() -> Response {
    LoraAdapterApiError::new(
        StatusCode::PAYLOAD_TOO_LARGE,
        "request_body_too_large",
        "request body exceeds the configured limit",
    )
    .into_response()
}

fn validate_alias(alias: &str) -> Result<(), LoraAdapterApiError> {
    if alias.trim().is_empty() {
        return Err(LoraAdapterApiError::new(
            StatusCode::BAD_REQUEST,
            "invalid_lora_name",
            "lora_name must not be empty",
        ));
    }
    if alias.trim().len() > MAX_LORA_ALIAS_BYTES {
        return Err(LoraAdapterApiError::new(
            StatusCode::BAD_REQUEST,
            "invalid_lora_name",
            format!("lora_name must not exceed {MAX_LORA_ALIAS_BYTES} bytes"),
        ));
    }
    Ok(())
}

#[utoipa::path(
    post,
    tag = "LoRA adapters",
    path = "/v1/load_lora_adapter",
    description = "Registered only when runtime LoRA mutation is enabled. CLI servers enable it with MISTRALRS_ALLOW_RUNTIME_LORA_UPDATING.",
    request_body(
        content = LoadLoraAdapterRequest,
        example = json!({"lora_name": "production", "lora_path": "/srv/adapters/production"})
    ),
    responses(
        (status = 200, description = "LoRA adapter loaded", body = LoraAdapterObject),
        (status = 400, description = "Invalid request", body = LoraAdapterErrorResponse),
        (status = 403, description = "Adapter path is not allowed", body = LoraAdapterErrorResponse),
        (status = 404, description = "Model or adapter path was not found", body = LoraAdapterErrorResponse),
        (status = 409, description = "LoRA runtime is unavailable or at capacity", body = LoraAdapterErrorResponse),
        (status = 413, description = "Adapter input files exceed the configured safety limit", body = LoraAdapterErrorResponse),
        (status = 415, description = "Request content type is not JSON", body = LoraAdapterErrorResponse),
        (status = 422, description = "Adapter files are invalid or unsupported", body = LoraAdapterErrorResponse),
        (status = 429, description = "Another adapter load is already in progress", body = LoraAdapterErrorResponse),
        (status = 503, description = "Adapter storage or model device is unavailable", body = LoraAdapterErrorResponse),
        (status = 500, description = "Adapter loading task failed", body = LoraAdapterErrorResponse)
    )
)]
pub(crate) async fn load_lora_adapter(
    State(state): ExtractedMistralRsState,
    Extension(config): Extension<LoraAdapterApiConfig>,
    payload: Result<Json<LoadLoraAdapterRequest>, JsonRejection>,
) -> Result<Json<LoraAdapterObject>, LoraAdapterApiError> {
    let request = lifecycle_json(payload)?;
    validate_alias(&request.lora_name)?;
    let policy = lora_load_policy(&request)?;
    let LoadLoraAdapterRequest {
        lora_name,
        lora_path,
        model,
        ..
    } = request;
    let load_permit = config.try_begin_load()?;
    let adapter_path = PathBuf::from(lora_path);
    let model = normalize_model_id(model);
    let operation = tokio::spawn(async move {
        let _load_permit = load_permit;
        let files = tokio::task::spawn_blocking(move || config.open_adapter_files(&adapter_path))
            .await
            .map_err(|error| adapter_task_error("adapter file validation", error))??;
        state
            .load_lora_adapter_files_with_policy(
                model.as_deref(),
                lora_name,
                files.into_runtime_files(),
                policy,
            )
            .await
            .map_err(LoraAdapterApiError::core)
    });
    let info = operation
        .await
        .map_err(|error| adapter_task_error("adapter load", error))??;
    Ok(Json(LoraAdapterObject::from_info(info, true)))
}

fn lora_load_policy(
    request: &LoadLoraAdapterRequest,
) -> Result<LoraAdapterLoadPolicy, LoraAdapterApiError> {
    if request.expected_generation.is_some() && !request.load_inplace {
        return Err(LoraAdapterApiError::new(
            StatusCode::BAD_REQUEST,
            "invalid_lora_load_policy",
            "expected_generation requires load_inplace=true",
        ));
    }
    Ok(match (request.load_inplace, request.expected_generation) {
        (false, None) => LoraAdapterLoadPolicy::Create,
        (true, None) => LoraAdapterLoadPolicy::Upsert,
        (true, Some(generation)) => LoraAdapterLoadPolicy::CompareAndSwap(generation),
        (false, Some(_)) => unreachable!(),
    })
}

#[utoipa::path(
    post,
    tag = "LoRA adapters",
    path = "/v1/unload_lora_adapter",
    description = "Registered only when runtime LoRA mutation is enabled. CLI servers enable it with MISTRALRS_ALLOW_RUNTIME_LORA_UPDATING.",
    request_body(
        content = UnloadLoraAdapterRequest,
        example = json!({"lora_name": "production"})
    ),
    responses(
        (status = 200, description = "LoRA adapter unloaded", body = LoraAdapterObject),
        (status = 400, description = "Invalid request", body = LoraAdapterErrorResponse),
        (status = 404, description = "Model or adapter was not found", body = LoraAdapterErrorResponse),
        (status = 409, description = "LoRA runtime is unavailable", body = LoraAdapterErrorResponse),
        (status = 413, description = "Request body exceeds the configured limit", body = LoraAdapterErrorResponse),
        (status = 415, description = "Request content type is not JSON", body = LoraAdapterErrorResponse),
        (status = 422, description = "Request body is invalid", body = LoraAdapterErrorResponse),
        (status = 500, description = "Internal server error", body = LoraAdapterErrorResponse)
    )
)]
pub(crate) async fn unload_lora_adapter(
    State(state): ExtractedMistralRsState,
    payload: Result<Json<UnloadLoraAdapterRequest>, JsonRejection>,
) -> Result<Json<LoraAdapterObject>, LoraAdapterApiError> {
    let request = lifecycle_json(payload)?;
    let UnloadLoraAdapterRequest {
        lora_name,
        expected_generation,
        lora_int_id: _,
        model,
    } = request;
    validate_alias(&lora_name)?;
    let model = normalize_model_id(model);
    let info = state
        .unload_lora_adapter_if_generation(model.as_deref(), &lora_name, expected_generation)
        .await
        .map_err(LoraAdapterApiError::core)?;
    Ok(Json(LoraAdapterObject::from_info(info, true)))
}

#[utoipa::path(
    get,
    tag = "LoRA adapters",
    path = "/v1/lora_adapters",
    description = "Always registered for adapter discovery and status, even when runtime LoRA mutation is disabled.",
    params(ListLoraAdaptersQuery),
    responses(
        (status = 200, description = "Loaded LoRA adapters", body = LoraAdapterListResponse),
        (status = 400, description = "Invalid query", body = LoraAdapterErrorResponse),
        (status = 404, description = "Model was not found", body = LoraAdapterErrorResponse),
        (status = 409, description = "LoRA runtime is unavailable", body = LoraAdapterErrorResponse),
        (status = 500, description = "Internal server error", body = LoraAdapterErrorResponse)
    )
)]
pub(crate) async fn list_lora_adapters(
    State(state): ExtractedMistralRsState,
    Extension(config): Extension<LoraAdapterApiConfig>,
    payload: Result<Query<ListLoraAdaptersQuery>, QueryRejection>,
) -> Result<Json<LoraAdapterListResponse>, LoraAdapterApiError> {
    let query = lifecycle_query(payload)?;
    let model = normalize_model_id(query.model);
    let status = state
        .lora_adapter_status(model.as_deref())
        .await
        .map_err(LoraAdapterApiError::core)?;
    let data = status
        .adapters
        .into_iter()
        .map(|info| LoraAdapterObject::from_info(info, config.enabled()))
        .collect();
    let generations = status.generations.into_iter().map(Into::into).collect();
    Ok(Json(LoraAdapterListResponse {
        object: LORA_ADAPTER_LIST_OBJECT.to_string(),
        data,
        generations,
        resident_generations: status.resident_generations,
        retired_generations: status.retired_generations,
        resident_bytes: status.resident_bytes,
        max_adapters: status.limits.max_adapters,
        max_rank: status.limits.max_rank,
        max_bytes: status.limits.max_bytes,
    }))
}

#[cfg(test)]
mod tests {
    use std::io::Read;

    use axum::{body::Body, extract::FromRequest, http::Request};

    use super::*;

    fn route(model_id: &str, alias: &str, generation: u8) -> LoraAdapterRoute {
        LoraAdapterRoute {
            model_id: model_id.to_string(),
            adapter: LoraAdapterInfo {
                alias: alias.to_string(),
                source: "source".to_string(),
                revision: None,
                generation: mistralrs_core::AdapterGenerationId::from_bytes([generation; 32]),
                rank: 8,
                bytes: 16,
            },
        }
    }

    #[test]
    fn adapter_model_ids_are_stably_qualified() {
        let models = adapter_models_from_routes(
            vec![
                route("base-a", "code", 1),
                route("base-b", "code", 2),
                route("base-a", "math", 3),
            ],
            |_| Ok(false),
        )
        .unwrap();
        assert_eq!(
            models
                .iter()
                .map(|model| model.id.as_str())
                .collect::<Vec<_>>(),
            vec!["base-a::code", "base-b::code", "base-a::math"]
        );

        let models = adapter_models_from_routes(vec![route("base-a", "math", 3)], |model_id| {
            Ok(model_id == "math")
        })
        .unwrap();
        assert_eq!(models[0].id, "base-a::math");

        let before =
            adapter_models_from_routes(vec![route("base-a", "code", 1)], |_| Ok(false)).unwrap();
        let after = adapter_models_from_routes(
            vec![route("base-a", "code", 1), route("base-b", "code", 2)],
            |_| Ok(false),
        )
        .unwrap();
        assert_eq!(before[0].id, after[0].id);

        let models =
            adapter_models_from_routes(vec![route("a::b", "c", 1), route("a", "b::c", 2)], |_| {
                Ok(false)
            })
            .unwrap();
        assert_eq!(models[0].id, "a%3A%3Ab::c");
        assert_eq!(models[1].id, "a::b%3A%3Ac");

        let info = route("base-a", "math", 3).adapter;
        assert!(LoraAdapterObject::from_info(info.clone(), false)
            .source
            .is_none());
        assert_eq!(
            LoraAdapterObject::from_info(info, true).source.as_deref(),
            Some("source")
        );
    }

    #[test]
    fn adapter_model_resolution_accepts_stable_ids_and_unique_aliases() {
        let models = adapter_models_from_routes(
            vec![
                route("base-a", "code", 1),
                route("base-b", "code", 2),
                route("base-a", "math", 3),
            ],
            |_| Ok(false),
        )
        .unwrap();

        let mut model = "base-a::code".to_string();
        let mut adapter = None;
        resolve_lora_adapter_model_from_models(&models, &mut model, &mut adapter).unwrap();
        assert_eq!(model, "base-a");
        assert!(matches!(
            adapter,
            Some(crate::openai::AdapterSelection::Alias(alias)) if alias == "code"
        ));

        let mut model = "math".to_string();
        let mut adapter = None;
        resolve_lora_adapter_model_from_models(&models, &mut model, &mut adapter).unwrap();
        assert_eq!(model, "base-a");
        assert!(is_resolvable_lora_adapter_model(&models, "math"));
        assert!(!is_resolvable_lora_adapter_model(&models, "code"));

        let mut model = "code".to_string();
        let mut adapter = None;
        let error =
            resolve_lora_adapter_model_from_models(&models, &mut model, &mut adapter).unwrap_err();
        assert!(error.contains("base-a::code"));
        assert!(error.contains("base-b::code"));
    }

    #[test]
    fn adapter_root_blocks_sibling_paths() {
        let temp = tempfile::tempdir().unwrap();
        let allowed = temp.path().join("allowed");
        let sibling = temp.path().join("sibling");
        std::fs::create_dir(&allowed).unwrap();
        std::fs::create_dir(&sibling).unwrap();
        std::fs::File::create(allowed.join(LORA_CONFIG_FILE)).unwrap();
        std::fs::File::create(allowed.join(LORA_WEIGHTS_FILE)).unwrap();
        let config = LoraAdapterApiConfig::default()
            .with_enabled(true)
            .with_allowed_root(&allowed)
            .prepare()
            .unwrap();

        assert!(config.open_adapter_files(&allowed).is_ok());
        assert_eq!(
            config.open_adapter_files(&sibling).unwrap_err().status,
            StatusCode::FORBIDDEN
        );
    }

    #[cfg(unix)]
    #[test]
    fn adapter_root_blocks_files_resolving_outside_root() {
        use std::os::unix::fs::symlink;

        let temp = tempfile::tempdir().unwrap();
        let allowed = temp.path().join("allowed");
        let adapter = allowed.join("adapter");
        let outside_config = temp.path().join("adapter_config.json");
        std::fs::create_dir(&allowed).unwrap();
        std::fs::create_dir(&adapter).unwrap();
        std::fs::File::create(&outside_config).unwrap();
        std::fs::File::create(adapter.join(LORA_WEIGHTS_FILE)).unwrap();
        symlink(outside_config, adapter.join(LORA_CONFIG_FILE)).unwrap();
        let config = LoraAdapterApiConfig::default()
            .with_enabled(true)
            .with_allowed_root(allowed)
            .prepare()
            .unwrap();

        assert_eq!(
            config.open_adapter_files(&adapter).unwrap_err().status,
            StatusCode::FORBIDDEN
        );
    }

    #[test]
    fn adapter_loading_keeps_the_validated_file_handles() {
        let temp = tempfile::tempdir().unwrap();
        let adapter = temp.path().join("adapter");
        std::fs::create_dir(&adapter).unwrap();
        std::fs::write(adapter.join(LORA_CONFIG_FILE), b"original-config").unwrap();
        std::fs::write(adapter.join(LORA_WEIGHTS_FILE), b"original-weights").unwrap();
        let config = LoraAdapterApiConfig::default()
            .with_enabled(true)
            .with_allowed_root(temp.path())
            .prepare()
            .unwrap();
        let mut opened = config.open_adapter_files(&adapter).unwrap();

        std::fs::rename(
            adapter.join(LORA_CONFIG_FILE),
            adapter.join("old-adapter-config.json"),
        )
        .unwrap();
        std::fs::write(adapter.join(LORA_CONFIG_FILE), b"replacement-config").unwrap();
        let mut contents = String::new();
        opened.config.read_to_string(&mut contents).unwrap();
        assert_eq!(contents, "original-config");
    }

    #[test]
    fn relative_adapter_paths_resolve_under_the_allowed_root() {
        let temp = tempfile::tempdir().unwrap();
        let adapter = temp.path().join("production");
        std::fs::create_dir(&adapter).unwrap();
        std::fs::File::create(adapter.join(LORA_CONFIG_FILE)).unwrap();
        std::fs::File::create(adapter.join(LORA_WEIGHTS_FILE)).unwrap();
        let config = LoraAdapterApiConfig::default()
            .with_enabled(true)
            .with_allowed_root(temp.path())
            .prepare()
            .unwrap();

        assert!(config.open_adapter_files(Path::new("production")).is_ok());
    }

    #[test]
    fn lifecycle_requests_accept_vllm_fields_and_reject_typos() {
        let request: LoadLoraAdapterRequest = serde_json::from_value(serde_json::json!({
            "lora_name": "production",
            "lora_path": "production-v2",
            "load_inplace": true,
            "expected_generation": "0707070707070707070707070707070707070707070707070707070707070707",
            "is_3d_lora_weight": true
        }))
        .unwrap();
        assert!(request.load_inplace);
        assert!(request.is_3d_lora_weight);
        assert_eq!(
            lora_load_policy(&request).unwrap(),
            LoraAdapterLoadPolicy::CompareAndSwap(mistralrs_core::AdapterGenerationId::from_bytes(
                [7; 32]
            ))
        );
        assert_eq!(
            request.expected_generation,
            Some(mistralrs_core::AdapterGenerationId::from_bytes([7; 32]))
        );
        assert!(
            serde_json::from_value::<LoadLoraAdapterRequest>(serde_json::json!({
                "lora_name": "production",
                "lora_path": "production-v2",
                "load_inpalce": true
            }))
            .is_err()
        );
    }

    #[tokio::test]
    async fn lifecycle_json_rejections_use_the_stable_error_envelope() {
        let request = Request::builder()
            .header(axum::http::header::CONTENT_TYPE, "application/json")
            .body(Body::from("{"))
            .unwrap();
        let rejection = Json::<LoadLoraAdapterRequest>::from_request(request, &())
            .await
            .unwrap_err();
        let error = LoraAdapterApiError::json_rejection(rejection);
        assert_eq!(error.status, StatusCode::BAD_REQUEST);
        assert_eq!(error.body.error.code, "malformed_json");
        assert_eq!(error.body.error.error_type, "invalid_request_error");

        let request = Request::builder()
            .header(axum::http::header::CONTENT_TYPE, "application/json")
            .body(Body::from(
                r#"{"lora_name":"production","lora_path":"adapter","load_inpalce":true}"#,
            ))
            .unwrap();
        let rejection = Json::<LoadLoraAdapterRequest>::from_request(request, &())
            .await
            .unwrap_err();
        let error = LoraAdapterApiError::json_rejection(rejection);
        assert_eq!(error.status, StatusCode::UNPROCESSABLE_ENTITY);
        assert_eq!(error.body.error.code, "invalid_request_body");

        let request = Request::builder()
            .body(Body::from(
                r#"{"lora_name":"production","lora_path":"adapter"}"#,
            ))
            .unwrap();
        let rejection = Json::<LoadLoraAdapterRequest>::from_request(request, &())
            .await
            .unwrap_err();
        let error = LoraAdapterApiError::json_rejection(rejection);
        assert_eq!(error.status, StatusCode::UNSUPPORTED_MEDIA_TYPE);
        assert_eq!(error.body.error.code, "invalid_content_type");
    }

    #[test]
    fn http_load_admission_is_shared_and_non_queueing() {
        let config = LoraAdapterApiConfig::default();
        let permit = config.try_begin_load().unwrap();
        assert_eq!(
            config.clone().try_begin_load().unwrap_err().status,
            StatusCode::TOO_MANY_REQUESTS
        );
        drop(permit);
        assert!(config.try_begin_load().is_ok());

        let response = LoraAdapterApiError::lora(LoraAdapterError::LoadBusy).into_response();
        assert_eq!(
            response.headers().get(axum::http::header::RETRY_AFTER),
            Some(&axum::http::HeaderValue::from_static("1"))
        );
    }

    #[test]
    fn default_and_empty_model_ids_select_the_default_model() {
        assert_eq!(normalize_model_id(None), None);
        assert_eq!(normalize_model_id(Some(String::new())), None);
        assert_eq!(normalize_model_id(Some(" default ".to_string())), None);
        assert_eq!(
            normalize_model_id(Some(" model ".to_string())),
            Some("model".to_string())
        );
    }

    #[test]
    fn core_errors_map_to_stable_http_statuses() {
        assert_eq!(
            LoraAdapterApiError::core(MistralRsError::ModelNotFound("model".to_string())).status,
            StatusCode::NOT_FOUND
        );
        assert_eq!(
            LoraAdapterApiError::lora(LoraAdapterError::InvalidAlias).status,
            StatusCode::BAD_REQUEST
        );
        assert_eq!(
            LoraAdapterApiError::lora(LoraAdapterError::AdapterLimit { max: 1 }).status,
            StatusCode::CONFLICT
        );
        assert_eq!(
            LoraAdapterApiError::lora(LoraAdapterError::AliasLimit { max: 1 }).status,
            StatusCode::CONFLICT
        );
        assert_eq!(
            LoraAdapterApiError::lora(LoraAdapterError::AliasTooLong {
                bytes: MAX_LORA_ALIAS_BYTES + 1,
                max: MAX_LORA_ALIAS_BYTES,
            })
            .status,
            StatusCode::BAD_REQUEST
        );
        assert_eq!(
            LoraAdapterApiError::lora(LoraAdapterError::FileTooLarge {
                path: PathBuf::from("adapter_model.safetensors"),
                bytes: 2,
                max: 1,
            })
            .status,
            StatusCode::PAYLOAD_TOO_LARGE
        );
        assert_eq!(
            LoraAdapterApiError::lora(LoraAdapterError::LoadBusy).status,
            StatusCode::TOO_MANY_REQUESTS
        );
        assert_eq!(
            LoraAdapterApiError::lora(LoraAdapterError::AlreadyLoaded {
                alias: "production".to_string(),
                generation: mistralrs_core::AdapterGenerationId::from_bytes([1; 32]),
            })
            .status,
            StatusCode::CONFLICT
        );
        assert_eq!(
            adapter_filesystem_error(
                "adapter file `adapter_model.safetensors`",
                std::io::Error::other("disk failure"),
            )
            .status,
            StatusCode::SERVICE_UNAVAILABLE
        );
        assert_eq!(
            adapter_filesystem_error(
                "adapter directory `/missing`",
                std::io::Error::from(std::io::ErrorKind::NotFound),
            )
            .status,
            StatusCode::NOT_FOUND
        );
        assert_eq!(
            adapter_filesystem_error(
                "adapter directory `/forbidden`",
                std::io::Error::from(std::io::ErrorKind::PermissionDenied),
            )
            .status,
            StatusCode::FORBIDDEN
        );
        assert_eq!(
            adapter_filesystem_error(
                "adapter file `adapter_model.safetensors`",
                std::io::Error::from(std::io::ErrorKind::NotADirectory),
            )
            .status,
            StatusCode::BAD_REQUEST
        );
    }
}
