//! ## OpenAPI doc functionality.

use utoipa::OpenApi;

use crate::{
    anthropic::{
        __path_anthropic_count_tokens, __path_anthropic_messages, AnthropicContainer,
        AnthropicContentBlock, AnthropicCountTokensResponse, AnthropicError, AnthropicErrorBody,
        AnthropicImageSource, AnthropicMessage, AnthropicMessageContent, AnthropicMessageResponse,
        AnthropicMessagesRequest, AnthropicResponseContentBlock, AnthropicSkillReference,
        AnthropicSystem, AnthropicThinking, AnthropicTool, AnthropicToolChoice, AnthropicUsage,
        AnthropicWebSearchUserLocation,
    },
    approvals::{
        __path_resolve_agent_approval, ApprovalDecision, ApprovalDecisionRequest,
        ApprovalDecisionResponse,
    },
    chat_completion::__path_chatcompletions,
    completions::__path_completions,
    embeddings::__path_embeddings,
    files::{
        __path_delete_file, __path_get_container_file, __path_get_container_file_content,
        __path_get_file, __path_get_file_content, __path_list_container_files, __path_list_files,
        __path_upload_file, ContainerFileMetadata, FileMetadata, SourceMeta,
    },
    handlers::{
        __path_calibration_apply, __path_calibration_start, __path_calibration_status,
        __path_delete_session, __path_get_model_status, __path_get_session, __path_health,
        __path_models, __path_put_session, __path_re_isq, __path_reload_model,
        __path_system_doctor, __path_system_info, __path_tune_model, __path_unload_model,
        CalibrationApplyRequest, ModelOperationRequest, ModelStatus, ModelStatusResponse,
        ReIsqRequest, TuneModelRequest, TuneProfileRequest,
    },
    image_generation::__path_image_generation,
    lora_adapters::{
        __path_list_lora_adapters, __path_load_lora_adapter, __path_unload_lora_adapter,
        LoadLoraAdapterRequest, LoraAdapterErrorBody, LoraAdapterErrorResponse,
        LoraAdapterListResponse, LoraAdapterObject, UnloadLoraAdapterRequest,
    },
    metrics::__path_metrics,
    openai::{
        AdapterGenerationSelection, AdapterSelection, AudioResponseFormat, ChatCompletionRequest,
        CompletionRequest, EmbeddingData, EmbeddingEncodingFormat, EmbeddingInput,
        EmbeddingRequest, EmbeddingResponse, EmbeddingUsage, EmbeddingVector, FunctionCalled,
        Grammar, ImageGenerationRequest, JsonSchemaResponseFormat, Message, MessageContent,
        MessageInnerContent, ModelObject, ModelObjects, OpenAiCodeInterpreterAutoContainer,
        OpenAiCodeInterpreterContainer, OpenAiCodeInterpreterContainerType,
        OpenAiCodeInterpreterTool, OpenAiCodeInterpreterToolType, OpenAiFunctionToolType,
        OpenAiResponsesFunctionTool, OpenAiShellEnvironment, OpenAiShellSkill, OpenAiShellTool,
        OpenAiShellToolType, OpenAiTool, OpenAiWebSearchTool, OpenAiWebSearchToolType,
        OpenAiWebSearchUserLocation, ResponseFormat, ResponsesAnnotation, ResponsesChunk,
        ResponsesContent, ResponsesCreateRequest, ResponsesDelta, ResponsesDeltaContent,
        ResponsesDeltaOutput, ResponsesError, ResponsesIncompleteDetails,
        ResponsesInputTokensDetails, ResponsesMessages, ResponsesObject, ResponsesOutput,
        ResponsesOutputTokensDetails, ResponsesUsage, SpeechGenerationRequest, StopTokens,
        ToolCall,
    },
    responses::{
        __path_cancel_response, __path_create_response, __path_delete_response, __path_get_response,
    },
    responses_types::content::{FileCitation, FilePathInfo, UrlCitation},
    skills::{
        __path_list_skill_versions, __path_list_skills, __path_upload_skill,
        __path_upload_skill_version, AnthropicSkillListObject, AnthropicSkillObject,
        AnthropicSkillVersionListObject, AnthropicSkillVersionObject, SkillListObject,
        SkillListQuery, SkillObject, SkillVersionObject,
    },
    speech_generation::__path_speech_generation,
};
use mistralrs_core::{
    ApproximateUserLocation, CalibrationStatus, Function, ImageGenerationResponseFormat,
    NamedFunctionToolChoice, SearchContextSize, SerializedSession, Tool, ToolChoice, ToolType,
    WebSearchContentType, WebSearchFilters, WebSearchImageSettings, WebSearchOptions,
    WebSearchReturnTokenBudget, WebSearchUserLocation,
};

/// This is used to generate the OpenAPI docs.
/// The mistral.rs server router will include these by default, but if you're
/// including the mistral.rs server core into another project, you can generate the
/// OpenAPI docs separately to merge with the other project OpenAPI docs.
///
/// ### Arguments
/// * `base_path` - the base path of the mistral.rs server instance (in case the mistral.rs server is being included in another axum project)
///
/// ### Example
/// ```ignore
/// // MyApp
/// use axum::{Router, routing::{get, post}};
/// use utoipa::OpenApi;
/// use utoipa_swagger_ui::SwaggerUi;
/// use mistralrs_server_core::openapi_doc::get_openapi_doc;
///
/// #[derive(OpenApi)]
/// #[openapi(
///     paths(root, controllers::custom_chat),
///     tags(
///         (name = "hello", description = "Hello world endpoints")
///     ),
///     info(
///         title = "Hello World API",
///         version = "1.0.0",
///         description = "A simple API that responds with a greeting"
///     )
/// )]
/// struct ApiDoc;
///
/// let mistral_base_path = "/api/mistral";
/// let mistral_doc = get_openapi_doc(Some(mistral_base_path));
/// let mut api_docs = ApiDoc::openapi();
/// api_docs.merge(mistral_doc);
///
/// let app = Router::new()
///   .route("/", get(root))
///   .merge(SwaggerUi::new("/api-docs").url("/api-docs/openapi.json", api_docs));
/// ```
pub fn get_openapi_doc(base_path: Option<&str>) -> utoipa::openapi::OpenApi {
    #[derive(OpenApi)]
    #[openapi(
        paths(models, health, chatcompletions, anthropic_messages, anthropic_count_tokens, completions, embeddings, re_isq, calibration_start, calibration_status, calibration_apply, image_generation, speech_generation, create_response, get_response, delete_response, cancel_response, upload_skill, list_skills, upload_skill_version, list_skill_versions, load_lora_adapter, unload_lora_adapter, list_lora_adapters, unload_model, reload_model, get_model_status, tune_model, system_info, system_doctor, get_session, put_session, delete_session, list_files, upload_file, get_file, get_file_content, delete_file, list_container_files, get_container_file, get_container_file_content, resolve_agent_approval, metrics),
        components(schemas(
            ApprovalDecision,
            ApprovalDecisionRequest,
            ApprovalDecisionResponse,
            AdapterGenerationSelection,
            AdapterSelection,
            ApproximateUserLocation,
            AnthropicContainer,
            AnthropicContentBlock,
            AnthropicCountTokensResponse,
            AnthropicError,
            AnthropicErrorBody,
            AnthropicImageSource,
            AnthropicMessage,
            AnthropicMessageContent,
            AnthropicMessageResponse,
            AnthropicMessagesRequest,
            AnthropicResponseContentBlock,
            AnthropicSkillReference,
            AnthropicSkillListObject,
            AnthropicSkillObject,
            AnthropicSkillVersionListObject,
            AnthropicSkillVersionObject,
            AnthropicSystem,
            AnthropicThinking,
            AnthropicTool,
            AnthropicToolChoice,
            AnthropicUsage,
            AnthropicWebSearchUserLocation,
            AudioResponseFormat,
            CalibrationStatus,
            ChatCompletionRequest,
            CompletionRequest,
            EmbeddingData,
            EmbeddingEncodingFormat,
            EmbeddingInput,
            EmbeddingRequest,
            EmbeddingResponse,
            EmbeddingUsage,
            EmbeddingVector,
            ContainerFileMetadata,
            FileMetadata,
            FileCitation,
            FilePathInfo,
            Function,
            FunctionCalled,
            Grammar,
            ImageGenerationRequest,
            LoadLoraAdapterRequest,
            LoraAdapterErrorBody,
            LoraAdapterErrorResponse,
            LoraAdapterListResponse,
            LoraAdapterObject,
            ImageGenerationResponseFormat,
            JsonSchemaResponseFormat,
            Message,
            MessageContent,
            MessageInnerContent,
            ModelObject,
            ModelObjects,
            NamedFunctionToolChoice,
            OpenAiCodeInterpreterAutoContainer,
            OpenAiCodeInterpreterContainer,
            OpenAiCodeInterpreterContainerType,
            OpenAiCodeInterpreterTool,
            OpenAiCodeInterpreterToolType,
            OpenAiFunctionToolType,
            OpenAiResponsesFunctionTool,
            OpenAiShellEnvironment,
            OpenAiShellSkill,
            OpenAiShellTool,
            OpenAiShellToolType,
            OpenAiTool,
            OpenAiWebSearchTool,
            OpenAiWebSearchToolType,
            OpenAiWebSearchUserLocation,
            ModelOperationRequest,
            ModelStatus,
            ModelStatusResponse,
            ReIsqRequest, CalibrationApplyRequest,
            ResponseFormat,
            ResponsesAnnotation,
            ResponsesChunk,
            ResponsesContent,
            ResponsesCreateRequest,
            ResponsesDelta,
            ResponsesDeltaContent,
            ResponsesDeltaOutput,
            ResponsesError,
            ResponsesIncompleteDetails,
            ResponsesInputTokensDetails,
            ResponsesMessages,
            ResponsesObject,
            ResponsesOutput,
            ResponsesOutputTokensDetails,
            ResponsesUsage,
            SearchContextSize,
            SerializedSession,
            SkillListObject,
            SkillListQuery,
            SkillObject,
            SkillVersionObject,
            SourceMeta,
            SpeechGenerationRequest,
            StopTokens,
            Tool,
            ToolCall,
            ToolChoice,
            ToolType,
            TuneModelRequest,
            TuneProfileRequest,
            UnloadLoraAdapterRequest,
            UrlCitation,
            WebSearchContentType,
            WebSearchFilters,
            WebSearchImageSettings,
            WebSearchOptions,
            WebSearchReturnTokenBudget,
            WebSearchUserLocation
        )),
        tags(
            (name = "Mistral.rs", description = "Mistral.rs API"),
            (name = "LoRA adapters", description = "Dynamic LoRA discovery and lifecycle operations")
        ),
        info(
            title = "Mistral.rs",
            license(
            name = "MIT",
        )
        )
    )]
    struct ApiDoc;

    let mut doc = ApiDoc::openapi();

    if let Some(prefix) = base_path {
        if !prefix.is_empty() {
            let mut prefixed_paths = utoipa::openapi::Paths::default();

            let original_paths = std::mem::take(&mut doc.paths.paths);

            for (path, item) in original_paths {
                let prefixed_path = format!("{prefix}{path}");
                prefixed_paths.paths.insert(prefixed_path, item);
            }

            prefixed_paths.extensions = doc.paths.extensions.clone();

            doc.paths = prefixed_paths;
        }
    }

    doc
}

#[cfg(test)]
mod tests {
    use super::*;

    const COMMITTED: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../docs/openapi.json");

    fn render() -> String {
        let doc = get_openapi_doc(None);
        serde_json::to_string_pretty(&doc).expect("openapi doc serializes") + "\n"
    }

    #[test]
    fn inference_responses_have_schemas_with_adapter_generation() {
        let value = serde_json::to_value(get_openapi_doc(None)).unwrap();
        for schema in [
            "ChatCompletionResponse",
            "ChatCompletionChunkResponse",
            "CompletionResponse",
            "CompletionChunkResponse",
            "ResponseResource",
        ] {
            assert!(
                value["components"]["schemas"][schema]["properties"]
                    .get("adapter_generation")
                    .is_some(),
                "missing adapter_generation from {schema}"
            );
        }

        for path in ["/v1/chat/completions", "/v1/completions", "/v1/responses"] {
            assert!(
                value["paths"][path]["post"]["responses"]["200"]["content"]
                    .as_object()
                    .is_some_and(|content| !content.is_empty()),
                "missing response content schema for {path}"
            );
        }

        for path in ["/v1/load_lora_adapter", "/v1/unload_lora_adapter"] {
            assert!(
                value["paths"][path]["post"]["description"]
                    .as_str()
                    .is_some_and(
                        |description| description.contains("only when runtime LoRA mutation")
                    )
            );
        }

        let list_model = &value["paths"]["/v1/lora_adapters"]["get"]["parameters"][0];
        assert_eq!(list_model["name"], "model");
        assert_eq!(list_model["in"], "query");
        assert_eq!(list_model["required"], false);

        let load_example = &value["paths"]["/v1/load_lora_adapter"]["post"]["requestBody"]
            ["content"]["application/json"]["example"];
        assert_eq!(
            load_example,
            &serde_json::json!({
                "lora_name": "production",
                "lora_path": "/srv/adapters/production"
            })
        );

        let unload_example = &value["paths"]["/v1/unload_lora_adapter"]["post"]["requestBody"]
            ["content"]["application/json"]["example"];
        assert_eq!(
            unload_example,
            &serde_json::json!({"lora_name": "production"})
        );

        assert!(value["paths"]["/v1/lora_adapters"]["get"]["description"]
            .as_str()
            .is_some_and(|description| description.contains("Always registered")));
    }

    // docs/openapi.json is a committed artifact consumed by the docs site.
    #[test]
    fn openapi_matches_committed() {
        let committed = std::fs::read_to_string(COMMITTED).unwrap_or_default();
        assert_eq!(
            render(),
            committed,
            "docs/openapi.json is stale; regenerate with: cargo test -p mistralrs-server-core regenerate_openapi -- --ignored"
        );
    }

    #[test]
    #[ignore]
    fn regenerate_openapi() {
        std::fs::write(COMMITTED, render()).expect("write openapi dump");
    }
}
