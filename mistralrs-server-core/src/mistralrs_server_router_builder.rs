/// Builder for mistral.rs for server
use anyhow::Result;
use axum::{
    extract::DefaultBodyLimit,
    http::{self, Method},
    routing::{get, post},
    Router,
};
use mistralrs_core::initialize_logging;
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

pub struct MistralRsServerRouterBuilder {
    mistralrs: Option<SharedMistralState>,
    include_swagger_routes: bool,
    base_path: Option<String>,
}

impl Default for MistralRsServerRouterBuilder {
    fn default() -> Self {
        Self {
            mistralrs: None,
            include_swagger_routes: true,
            base_path: None,
        }
    }
}

impl MistralRsServerRouterBuilder {
    pub fn new() -> Self {
        Default::default()
    }

    pub fn with_mistralrs(mut self, mistralrs: SharedMistralState) -> Self {
        self.mistralrs = Some(mistralrs);
        self
    }

    pub fn with_include_swagger_routes(mut self, include_swagger_routes: bool) -> Self {
        self.include_swagger_routes = include_swagger_routes;
        self
    }

    pub fn with_base_path(mut self, base_path: &str) -> Self {
        self.base_path = Some(base_path.to_owned());
        self
    }

    pub async fn build(mut self) -> Result<Router> {
        initialize_logging();

        let mistralrs = self.mistralrs.ok_or_else(|| {
            anyhow::anyhow!("`mistralrs` instance must be set. Use `with_mistralrs`.")
        })?;

        let mistralrs_server_router = init_get_router(
            mistralrs,
            self.include_swagger_routes,
            self.base_path.as_deref(),
        );

        Ok(mistralrs_server_router)
    }
}

fn init_get_router(
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
