//! ## General mistral.rs server route handlers.

use anyhow::Result;
use axum::extract::{Json, Query, State};
use axum::response::IntoResponse;
use mistralrs_core::{parse_isq_value, MistralRs, Request};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use utoipa::ToSchema;

use crate::{
    openai::{ModelObject, ModelObjects},
    types::ExtractedMistralRsState,
};

#[derive(Debug, Clone, Deserialize)]
pub struct ModelsQuery {
    /// When present, Codex expects a Codex-specific `/models` response schema and uses it to
    /// determine context window and model capabilities. We serve that schema from `/v1/models`
    /// when `client_version` is set, while keeping the OpenAI-compatible response otherwise.
    client_version: Option<String>,
}

#[utoipa::path(
  get,
  tag = "Mistral.rs",
  path = "/v1/models",
  responses((status = 200, description = "Served model info", body = ModelObjects))
)]
pub async fn models(
    State(state): ExtractedMistralRsState,
    Query(query): Query<ModelsQuery>,
) -> impl IntoResponse {
    if query.client_version.is_some() {
        // Codex `/models` schema (codex_protocol::openai_models::ModelsResponse).
        // This enables Codex to compute "% context left" instead of falling back to "X used".
        let available_models = state.list_models().unwrap_or_default();
        let default_ctx = state
            .max_sequence_length(None)
            .ok()
            .flatten()
            .unwrap_or_default() as i64;
        let mut models: Vec<Value> = Vec::with_capacity(available_models.len().saturating_add(1));

        // Include a "default" alias so Codex can use it as a stable selection.
        models.push(json!({
            "slug": "default",
            "display_name": "default",
            "description": "Default model".to_string(),
            "default_reasoning_level": "medium",
            "supported_reasoning_levels": [
                {"effort": "low", "description": "low"},
                {"effort": "medium", "description": "medium"}
            ],
            "shell_type": "shell_command",
            "visibility": "list",
            "minimal_client_version": [0, 0, 0],
            "supported_in_api": true,
            "priority": 0,
            "upgrade": null,
            "base_instructions": null,
            "supports_reasoning_summaries": false,
            "support_verbosity": false,
            "default_verbosity": null,
            "apply_patch_tool_type": null,
            "truncation_policy": {"mode": "bytes", "limit": 10000},
            "supports_parallel_tool_calls": true,
            "context_window": if default_ctx > 0 { Some(default_ctx) } else { None },
            "reasoning_summary_format": "none",
            "experimental_supported_tools": [],
        }));

        for (idx, model_id) in available_models.into_iter().enumerate() {
            let ctx = state
                .max_sequence_length(Some(&model_id))
                .ok()
                .flatten()
                .unwrap_or_default() as i64;
            models.push(json!({
                "slug": model_id,
                "display_name": model_id,
                "description": null,
                "default_reasoning_level": "medium",
                "supported_reasoning_levels": [
                    {"effort": "low", "description": "low"},
                    {"effort": "medium", "description": "medium"}
                ],
                "shell_type": "shell_command",
                "visibility": "list",
                "minimal_client_version": [0, 0, 0],
                "supported_in_api": true,
                "priority": (idx as i32) + 1,
                "upgrade": null,
                "base_instructions": null,
                "supports_reasoning_summaries": false,
                "support_verbosity": false,
                "default_verbosity": null,
                "apply_patch_tool_type": null,
                "truncation_policy": {"mode": "bytes", "limit": 10000},
                "supports_parallel_tool_calls": true,
                "context_window": if ctx > 0 { Some(ctx) } else { None },
                "reasoning_summary_format": "none",
                "experimental_supported_tools": [],
            }));
        }

        return Json(json!({
            "models": models,
            "etag": "",
        }))
        .into_response();
    }

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
