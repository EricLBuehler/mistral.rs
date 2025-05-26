use axum::{
    extract::DefaultBodyLimit,
    http::{self, Method},
    routing::{get, post},
    Router,
};
use tower_http::cors::{AllowOrigin, CorsLayer};
use utoipa_swagger_ui::SwaggerUi;

use crate::{
    chat_completion::chatcompletions,
    completions::completions,
    defaults::MAX_BODY_LIMIT,
    handlers::{health, models, re_isq},
    image_generation::image_generation,
    openapi_doc::get_openapi_doc,
    speech_generation::speech_generation,
    SharedMistralState,
};

pub fn get_router(
    state: SharedMistralState,
    include_swagger_routes: bool,
    base_path: Option<&str>,
) -> Router {
    let allow_origin = AllowOrigin::any();
    let cors_layer = CorsLayer::new()
        .allow_methods([Method::GET, Method::POST])
        .allow_headers([http::header::CONTENT_TYPE, http::header::AUTHORIZATION])
        .allow_origin(allow_origin);

    // Use the provided base path or default to ""
    let prefix = base_path.unwrap_or("");

    let mut router = Router::new()
        .route("/v1/chat/completions", post(chatcompletions))
        .route("/v1/completions", post(completions))
        .route("/v1/models", get(models))
        .route("/health", get(health))
        .route("/", get(health))
        .route("/re_isq", post(re_isq))
        .route("/v1/images/generations", post(image_generation))
        .route("/v1/audio/speech", post(speech_generation))
        .layer(cors_layer)
        .layer(DefaultBodyLimit::max(MAX_BODY_LIMIT))
        .with_state(state);

    if include_swagger_routes {
        let doc = get_openapi_doc(None);

        router = router.merge(
            SwaggerUi::new(format!("{prefix}/docs"))
                .url(format!("{prefix}/api-doc/openapi.json"), doc),
        );
    }

    router
}
