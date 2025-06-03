//! ## mistral.rs server router builder.

use anyhow::Result;
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
    handlers::{health, models, re_isq},
    image_generation::image_generation,
    openapi_doc::get_openapi_doc,
    speech_generation::speech_generation,
    types::SharedMistralRsState,
};

// NOTE(EricLBuehler): Accept up to 50mb input
const N_INPUT_SIZE: usize = 50;
const MB_TO_B: usize = 1024 * 1024; // 1024 kb in a mb

/// This is the axum default request body limit for the router. Accept up to 50mb input.
pub const DEFAULT_MAX_BODY_LIMIT: usize = N_INPUT_SIZE * MB_TO_B;

/// A builder for creating a mistral.rs server router with configurable options.
///
/// ### Examples
///
/// Basic usage:
/// ```ignore
/// use mistralrs_server_core::mistralrs_server_router_builder::MistralRsServerRouterBuilder;
///
/// let router = MistralRsServerRouterBuilder::new()
///     .with_mistralrs(mistralrs_instance)
///     .build()
///     .await?;
/// ```
///
/// With custom configuration:
/// ```ignore
/// use mistralrs_server_core::mistralrs_server_router_builder::MistralRsServerRouterBuilder;
///
/// let router = MistralRsServerRouterBuilder::new()
///     .with_mistralrs(mistralrs_instance)
///     .with_include_swagger_routes(false)
///     .with_base_path("/api/mistral")
///     .build()
///     .await?;
/// ```
pub struct MistralRsServerRouterBuilder {
    /// The shared mistral.rs instance
    mistralrs: Option<SharedMistralRsState>,
    /// Whether to include Swagger/OpenAPI documentation routes
    include_swagger_routes: bool,
    /// Optional base path prefix for all routes
    base_path: Option<String>,
    /// Optional CORS allowed origins
    allowed_origins: Option<Vec<String>>,
    /// Optional axum default request body limit
    max_body_limit: Option<usize>,
}

impl Default for MistralRsServerRouterBuilder {
    /// Creates a new builder with default configuration.
    fn default() -> Self {
        Self {
            mistralrs: None,
            include_swagger_routes: true,
            base_path: None,
            allowed_origins: None,
            max_body_limit: None,
        }
    }
}

impl MistralRsServerRouterBuilder {
    /// Creates a new `MistralRsServerRouterBuilder` with default settings.
    ///
    /// This is equivalent to calling `Default::default()`.
    ///
    /// ### Examples
    ///
    /// ```ignore
    /// use mistralrs_server_core::mistralrs_server_router_builder::MistralRsServerRouterBuilder;
    ///
    /// let builder = MistralRsServerRouterBuilder::new();
    /// ```
    pub fn new() -> Self {
        Default::default()
    }

    /// Sets the shared mistral.rs instance
    pub fn with_mistralrs(mut self, mistralrs: SharedMistralRsState) -> Self {
        self.mistralrs = Some(mistralrs);
        self
    }

    /// Configures whether to include OpenAPI doc routes.
    ///
    /// When enabled (default), the router will include routes for Swagger UI
    /// at `/docs` and the OpenAPI specification at `/api-doc/openapi.json`.
    /// These routes respect the configured base path if one is set.
    pub fn with_include_swagger_routes(mut self, include_swagger_routes: bool) -> Self {
        self.include_swagger_routes = include_swagger_routes;
        self
    }

    /// Sets a base path prefix for all routes.
    ///
    /// When set, all routes will be prefixed with the given path. This is
    /// useful when including the mistral.rs server instance in another axum project.
    pub fn with_base_path(mut self, base_path: &str) -> Self {
        self.base_path = Some(base_path.to_owned());
        self
    }

    /// Sets the CORS allowed origins.
    pub fn with_allowed_origins(mut self, origins: Vec<String>) -> Self {
        self.allowed_origins = Some(origins);
        self
    }

    /// Sets the axum default request body limit.
    pub fn with_max_body_limit(mut self, max_body_limit: usize) -> Self {
        self.max_body_limit = Some(max_body_limit);
        self
    }

    /// Builds the configured axum router.
    ///
    /// ### Examples
    ///
    /// ```ignore
    /// use mistralrs_server_core::mistralrs_server_router_builder::MistralRsServerRouterBuilder;
    ///
    /// let router = MistralRsServerRouterBuilder::new()
    ///     .with_mistralrs(mistralrs_instance)
    ///     .build()
    ///     .await?;
    /// ```
    pub async fn build(self) -> Result<Router> {
        let mistralrs = self.mistralrs.ok_or_else(|| {
            anyhow::anyhow!("`mistralrs` instance must be set. Use `with_mistralrs`.")
        })?;

        let mistralrs_server_router = init_router(
            mistralrs,
            self.include_swagger_routes,
            self.base_path.as_deref(),
            self.allowed_origins,
            self.max_body_limit,
        );

        mistralrs_server_router
    }
}

/// Initializes and configures the underlying axum router with MistralRs API endpoints.
///
/// This function creates a router with all the necessary API endpoints,
/// CORS configuration, body size limits, and optional Swagger documentation.
fn init_router(
    state: SharedMistralRsState,
    include_swagger_routes: bool,
    base_path: Option<&str>,
    allowed_origins: Option<Vec<String>>,
    max_body_limit: Option<usize>,
) -> Result<Router> {
    let allow_origin = if let Some(origins) = allowed_origins {
        let parsed_origins: Result<Vec<_>, _> = origins.into_iter().map(|o| o.parse()).collect();

        match parsed_origins {
            Ok(origins) => AllowOrigin::list(origins),
            Err(_) => anyhow::bail!("Invalid origin format"),
        }
    } else {
        AllowOrigin::any()
    };

    let router_max_body_limit = max_body_limit.unwrap_or(DEFAULT_MAX_BODY_LIMIT);

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
        .layer(DefaultBodyLimit::max(router_max_body_limit))
        .with_state(state);

    if include_swagger_routes {
        let doc = get_openapi_doc(None);

        router = router.merge(
            SwaggerUi::new(format!("{prefix}/docs"))
                .url(format!("{prefix}/api-doc/openapi.json"), doc),
        );
    }

    Ok(router)
}
