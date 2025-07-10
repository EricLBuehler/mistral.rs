//! ## OpenAPI doc functionality.

use utoipa::OpenApi;

use crate::{
    chat_completion::__path_chatcompletions,
    completions::__path_completions,
    handlers::{ReIsqRequest, __path_health, __path_models, __path_re_isq},
    image_generation::__path_image_generation,
    openai::{
        AudioResponseFormat, ChatCompletionRequest, CompletionRequest, FunctionCalled, Grammar,
        ImageGenerationRequest, JsonSchemaResponseFormat, Message, MessageContent,
        MessageInnerContent, ModelObject, ModelObjects, ResponseFormat, SpeechGenerationRequest,
        StopTokens, ToolCall,
    },
    speech_generation::__path_speech_generation,
};
use mistralrs_core::{
    ApproximateUserLocation, Function, ImageGenerationResponseFormat, SearchContextSize, Tool,
    ToolChoice, ToolType, WebSearchOptions, WebSearchUserLocation,
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
        paths(models, health, chatcompletions, completions, re_isq, image_generation, speech_generation),
        components(schemas(
            ApproximateUserLocation,
            AudioResponseFormat,
            ChatCompletionRequest,
            CompletionRequest,
            Function,
            FunctionCalled,
            Grammar,
            ImageGenerationRequest,
            ImageGenerationResponseFormat,
            JsonSchemaResponseFormat,
            Message,
            MessageContent,
            MessageInnerContent,
            ModelObject,
            ModelObjects,
            ReIsqRequest,
            ResponseFormat,
            SearchContextSize,
            SpeechGenerationRequest,
            StopTokens,
            Tool,
            ToolCall,
            ToolChoice,
            ToolType,
            WebSearchOptions,
            WebSearchUserLocation
        )),
        tags(
            (name = "Mistral.rs", description = "Mistral.rs API")
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
