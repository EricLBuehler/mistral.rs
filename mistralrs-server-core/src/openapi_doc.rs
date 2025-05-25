use utoipa::OpenApi;

use crate::{
    chat_completion::__path_chatcompletions,
    handlers::{__path_health, __path_models},
    openai::{
        ChatCompletionRequest, CompletionRequest, ImageGenerationRequest, Message, ModelObject,
        ModelObjects, StopTokens,
    },
};

pub fn get_openapi_doc(base_path: Option<&str>) -> utoipa::openapi::OpenApi {
    #[derive(OpenApi)]
    #[openapi(
      paths(models, health, chatcompletions),
      components(
          schemas(ModelObjects, ModelObject, ChatCompletionRequest, CompletionRequest, ImageGenerationRequest, StopTokens, Message)),
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
