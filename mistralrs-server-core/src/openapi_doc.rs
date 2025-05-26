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
                let prefixed_path = format!("{}{}", prefix, path);
                prefixed_paths.paths.insert(prefixed_path, item);
            }

            prefixed_paths.extensions = doc.paths.extensions.clone();

            doc.paths = prefixed_paths;
        }
    }

    doc
}
