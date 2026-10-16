//! ## mistral.rs server router builder.

use anyhow::Result;
use axum::{
    extract::DefaultBodyLimit,
    http::{self, header::HeaderName, HeaderMap, Method, StatusCode, Uri},
    middleware,
    response::IntoResponse,
    routing::{get, post},
    Extension, Router,
};
use tower_http::cors::{AllowOrigin, CorsLayer};
#[cfg(feature = "swagger-ui")]
use utoipa_swagger_ui::SwaggerUi;

#[cfg(feature = "swagger-ui")]
use crate::openapi_doc::get_openapi_doc;
#[cfg(feature = "security")]
use crate::security::SecurityGuard;
use crate::{
    anthropic::{anthropic_count_tokens, anthropic_error_response, anthropic_messages},
    approvals::{resolve_agent_approval, ApprovalBroker},
    chat_completion::chatcompletions,
    completions::completions,
    embeddings::embeddings,
    files::{
        delete_file, get_container_file, get_container_file_content, get_file, get_file_content,
        list_container_files, list_files, upload_file,
    },
    handler_core::{openai_error_response, ApiError, ApiErrorKind},
    handlers::{
        calibration_apply, calibration_start, calibration_status, delete_session, get_model_status,
        get_session, health, models, put_session, re_isq, reload_model, system_doctor, system_info,
        tune_model, unload_model,
    },
    image_generation::image_generation,
    lora_adapters::{
        list_lora_adapters, load_lora_adapter, unload_lora_adapter, LoraAdapterApiConfig,
    },
    metrics::{metrics, metrics_disabled, observe_http, ObservabilityConfig, ObservabilityState},
    responses::{cancel_response, create_response, delete_response, get_response},
    route_registry::{
        AGENT_APPROVAL_ROUTE, ANTHROPIC_COUNT_TOKENS_ROUTE, ANTHROPIC_MESSAGES_ROUTE,
        CALIBRATION_APPLY_ROUTE, CALIBRATION_START_ROUTE, CALIBRATION_STATUS_ROUTE,
        CANCEL_RESPONSE_ROUTE, CHAT_COMPLETIONS_ROUTE, COMPLETIONS_ROUTE, CONTAINER_FILES_ROUTE,
        CONTAINER_FILE_CONTENT_ROUTE, CONTAINER_FILE_ROUTE, EMBEDDINGS_ROUTE, FILES_ROUTE,
        FILE_CONTENT_ROUTE, FILE_ROUTE, HEALTH_ROUTE, IMAGE_GENERATION_ROUTE,
        LIST_LORA_ADAPTERS_ROUTE, LOAD_LORA_ADAPTER_ROUTE, MODELS_ROUTE, MODEL_STATUS_ROUTE,
        RELOAD_MODEL_ROUTE, RESPONSES_ROUTE, RESPONSE_ROUTE, RE_ISQ_ROUTE, ROOT_ROUTE,
        SESSION_ROUTE, SKILLS_ROUTE, SKILL_VERSIONS_ROUTE, SPEECH_GENERATION_ROUTE,
        SYSTEM_DOCTOR_ROUTE, SYSTEM_INFO_ROUTE, TUNE_MODEL_ROUTE, UNLOAD_LORA_ADAPTER_ROUTE,
        UNLOAD_MODEL_ROUTE,
    },
    skills::{list_skill_versions, list_skills, upload_skill, upload_skill_version, SkillStore},
    speech_generation::speech_generation,
    types::SharedMistralRsState,
};

/// Server-level defaults for agentic features.
/// Injected as an axum Extension so handlers can apply them to incoming requests.
#[derive(Clone, Default)]
pub struct AgenticDefaults {
    pub max_tool_rounds: Option<usize>,
    pub tool_dispatch_url: Option<String>,
    pub agent_permission: Option<mistralrs_core::AgentPermission>,
    pub approval_broker: ApprovalBroker,
}

// NOTE(EricLBuehler): Accept up to 50mb input
const N_INPUT_SIZE: usize = 50;
const MB_TO_B: usize = 1024 * 1024; // 1024 kb in a mb
const ROUTE_NOT_FOUND_MESSAGE: &str = "The requested API route was not found.";
const METHOD_NOT_ALLOWED_MESSAGE: &str = "The requested HTTP method is not allowed for this route.";

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
    /// Whether to include Swagger/OpenAPI documentation routes.
    /// Only available when the `swagger-ui` feature is enabled.
    #[cfg(feature = "swagger-ui")]
    include_swagger_routes: bool,
    /// Optional base path prefix for Swagger UI routes.
    /// Only available when the `swagger-ui` feature is enabled.
    #[cfg(feature = "swagger-ui")]
    base_path: Option<String>,
    /// Optional CORS allowed origins
    allowed_origins: Option<Vec<String>>,
    /// Optional axum default request body limit
    max_body_limit: Option<usize>,
    /// Server-level agentic defaults
    agentic_defaults: AgenticDefaults,
    skills_dir: Option<std::path::PathBuf>,
    observability: ObservabilityConfig,
    lora_adapter_api: LoraAdapterApiConfig,
    #[cfg(feature = "security")]
    security: Option<SecurityGuard>,
}

impl Default for MistralRsServerRouterBuilder {
    /// Creates a new builder with default configuration.
    fn default() -> Self {
        Self {
            mistralrs: None,
            #[cfg(feature = "swagger-ui")]
            include_swagger_routes: true,
            #[cfg(feature = "swagger-ui")]
            base_path: None,
            allowed_origins: None,
            max_body_limit: None,
            agentic_defaults: AgenticDefaults::default(),
            skills_dir: None,
            observability: ObservabilityConfig::default(),
            lora_adapter_api: LoraAdapterApiConfig::from_env(),
            #[cfg(feature = "security")]
            security: None,
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
    ///
    /// Only available when the `swagger-ui` feature is enabled.
    #[cfg(feature = "swagger-ui")]
    pub fn with_include_swagger_routes(mut self, include_swagger_routes: bool) -> Self {
        self.include_swagger_routes = include_swagger_routes;
        self
    }

    /// Sets a base path prefix for Swagger UI routes.
    ///
    /// When set, Swagger UI routes will be prefixed with the given path. This is
    /// useful when including the mistral.rs server instance in another axum project.
    ///
    /// Only available when the `swagger-ui` feature is enabled.
    #[cfg(feature = "swagger-ui")]
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

    /// Sets the default maximum tool-call rounds for the agentic loop.
    pub fn with_max_tool_rounds(mut self, rounds: usize) -> Self {
        self.agentic_defaults.max_tool_rounds = Some(rounds);
        self
    }

    /// Sets the default maximum tool-call rounds if provided.
    pub fn with_max_tool_rounds_optional(mut self, rounds: Option<usize>) -> Self {
        if let Some(rounds) = rounds {
            self = self.with_max_tool_rounds(rounds);
        }
        self
    }

    /// Sets the URL to POST tool calls to for server-side execution.
    pub fn with_tool_dispatch_url(mut self, url: String) -> Self {
        self.agentic_defaults.tool_dispatch_url = Some(url);
        self
    }

    /// Sets the tool dispatch URL if provided.
    pub fn with_tool_dispatch_url_optional(mut self, url: Option<String>) -> Self {
        if let Some(url) = url {
            self = self.with_tool_dispatch_url(url);
        }
        self
    }

    pub fn with_agent_permission(mut self, permission: mistralrs_core::AgentPermission) -> Self {
        self.agentic_defaults.agent_permission = Some(permission);
        self
    }

    pub fn with_code_execution_permission(
        self,
        permission: mistralrs_core::CodeExecutionPermission,
    ) -> Self {
        self.with_agent_permission(permission.into())
    }

    pub fn with_approval_broker(mut self, broker: ApprovalBroker) -> Self {
        self.agentic_defaults.approval_broker = broker;
        self
    }

    pub fn with_skills_dir(mut self, skills_dir: impl Into<std::path::PathBuf>) -> Self {
        self.skills_dir = Some(skills_dir.into());
        self
    }

    pub fn with_skills_dir_optional(mut self, skills_dir: Option<std::path::PathBuf>) -> Self {
        if let Some(skills_dir) = skills_dir {
            self = self.with_skills_dir(skills_dir);
        }
        self
    }

    /// Sets server observability options.
    pub fn with_observability_config(mut self, observability: ObservabilityConfig) -> Self {
        self.observability = observability;
        self
    }

    /// Configures runtime LoRA management routes and their allowed adapter root.
    pub fn with_lora_adapter_api_config(mut self, config: LoraAdapterApiConfig) -> Self {
        self.lora_adapter_api = config;
        self
    }

    /// Adds the opt-in security layer. Serve the router with connect info so the guard sees the peer.
    #[cfg(feature = "security")]
    pub fn with_security(mut self, security: SecurityGuard) -> Self {
        self.security = Some(security);
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
        if self.observability.metrics {
            crate::metrics::install_prometheus_recorder();
        }
        let lora_adapter_api = self.lora_adapter_api.prepare()?;
        let mistralrs = self.mistralrs.ok_or_else(|| {
            anyhow::anyhow!("`mistralrs` instance must be set. Use `with_mistralrs`.")
        })?;
        let router_max_body_limit = self.max_body_limit.unwrap_or(DEFAULT_MAX_BODY_LIMIT);
        let observability = ObservabilityState::with_max_body_bytes(
            self.observability.clone(),
            mistralrs.clone(),
            router_max_body_limit,
        );

        #[allow(unused_mut)]
        let mut router = init_router(
            mistralrs,
            RouterEdgeConfig {
                allowed_origins: self.allowed_origins,
                max_body_limit: router_max_body_limit,
                #[cfg(feature = "security")]
                security: self.security,
            },
            self.agentic_defaults,
            self.skills_dir,
            self.observability,
            lora_adapter_api,
        )?;

        #[cfg(feature = "swagger-ui")]
        if self.include_swagger_routes {
            let prefix = self.base_path.as_deref().unwrap_or("");
            let doc = get_openapi_doc(None);
            router = router.merge(
                SwaggerUi::new(format!("{prefix}/docs"))
                    .url(format!("{prefix}/api-doc/openapi.json"), doc),
            );
        }

        router = router.layer(middleware::from_fn_with_state(observability, observe_http));

        Ok(router)
    }
}

struct RouterEdgeConfig {
    allowed_origins: Option<Vec<String>>,
    max_body_limit: usize,
    #[cfg(feature = "security")]
    security: Option<SecurityGuard>,
}

/// Initializes and configures the underlying axum router with MistralRs API endpoints.
///
/// This function creates a router with all the necessary API endpoints,
/// CORS configuration, and body size limits.
fn init_router(
    state: SharedMistralRsState,
    edge: RouterEdgeConfig,
    agentic_defaults: AgenticDefaults,
    skills_dir: Option<std::path::PathBuf>,
    observability: ObservabilityConfig,
    lora_adapter_api: LoraAdapterApiConfig,
) -> Result<Router> {
    let router_max_body_limit = edge.max_body_limit;
    let allow_origin = if let Some(origins) = edge.allowed_origins {
        let parsed_origins: Result<Vec<_>, _> = origins.into_iter().map(|o| o.parse()).collect();

        match parsed_origins {
            Ok(origins) => AllowOrigin::list(origins),
            Err(_) => anyhow::bail!("Invalid origin format"),
        }
    } else {
        AllowOrigin::any()
    };

    let skill_store = std::sync::Arc::new(SkillStore::new(
        skills_dir.unwrap_or_else(SkillStore::default_root),
    )?);
    let metrics_route = if observability.metrics {
        get(metrics)
    } else {
        get(metrics_disabled)
    };

    let cors_layer = CorsLayer::new()
        .allow_methods([Method::GET, Method::POST, Method::PUT, Method::DELETE])
        .allow_headers([
            http::header::CONTENT_TYPE,
            http::header::AUTHORIZATION,
            HeaderName::from_static("x-api-key"),
            HeaderName::from_static("anthropic-version"),
            HeaderName::from_static("anthropic-beta"),
            HeaderName::from_static("x-request-id"),
            HeaderName::from_static("request-id"),
        ])
        .expose_headers([
            HeaderName::from_static("x-request-id"),
            HeaderName::from_static("request-id"),
        ])
        .allow_origin(allow_origin);

    let mut router = Router::new()
        .route(CHAT_COMPLETIONS_ROUTE.path, post(chatcompletions))
        .route(ANTHROPIC_MESSAGES_ROUTE.path, post(anthropic_messages))
        .route(
            ANTHROPIC_COUNT_TOKENS_ROUTE.path,
            post(anthropic_count_tokens),
        )
        .route(COMPLETIONS_ROUTE.path, post(completions))
        .route(EMBEDDINGS_ROUTE.path, post(embeddings))
        .route(MODELS_ROUTE.path, get(models))
        .route(LIST_LORA_ADAPTERS_ROUTE.path, get(list_lora_adapters))
        .route(UNLOAD_MODEL_ROUTE.path, post(unload_model))
        .route(RELOAD_MODEL_ROUTE.path, post(reload_model))
        .route(MODEL_STATUS_ROUTE.path, post(get_model_status))
        .route(TUNE_MODEL_ROUTE.path, post(tune_model))
        .route(SYSTEM_INFO_ROUTE.path, get(system_info))
        .route(SYSTEM_DOCTOR_ROUTE.path, post(system_doctor))
        .route(HEALTH_ROUTE.path, get(health))
        .route("/metrics", metrics_route)
        .route(ROOT_ROUTE.path, get(health))
        .route(RE_ISQ_ROUTE.path, post(re_isq))
        .route(CALIBRATION_START_ROUTE.path, post(calibration_start))
        .route(
            CALIBRATION_STATUS_ROUTE.path,
            axum::routing::get(calibration_status),
        )
        .route(CALIBRATION_APPLY_ROUTE.path, post(calibration_apply))
        .route(IMAGE_GENERATION_ROUTE.path, post(image_generation))
        .route(FILES_ROUTE.path, get(list_files).post(upload_file))
        .route(FILE_ROUTE.path, get(get_file).delete(delete_file))
        .route(FILE_CONTENT_ROUTE.path, get(get_file_content))
        .route(CONTAINER_FILES_ROUTE.path, get(list_container_files))
        .route(CONTAINER_FILE_ROUTE.path, get(get_container_file))
        .route(
            CONTAINER_FILE_CONTENT_ROUTE.path,
            get(get_container_file_content),
        )
        .route(SPEECH_GENERATION_ROUTE.path, post(speech_generation))
        .route(AGENT_APPROVAL_ROUTE.path, post(resolve_agent_approval))
        .route(RESPONSES_ROUTE.path, post(create_response))
        .route(SKILLS_ROUTE.path, get(list_skills).post(upload_skill))
        .route(
            SKILL_VERSIONS_ROUTE.path,
            get(list_skill_versions).post(upload_skill_version),
        )
        .route(
            RESPONSE_ROUTE.path,
            get(get_response).delete(delete_response),
        )
        .route(CANCEL_RESPONSE_ROUTE.path, post(cancel_response))
        .route(
            SESSION_ROUTE.path,
            get(get_session).put(put_session).delete(delete_session),
        );

    if lora_adapter_api.enabled() {
        if let Some(root) = lora_adapter_api.allowed_root() {
            tracing::warn!(
                adapter_root = %root.display(),
                "runtime LoRA adapter management is enabled; authorized clients can load adapter files from this root"
            );
        } else {
            tracing::warn!(
                "runtime LoRA adapter management is enabled without an adapter root; authorized clients can load any adapter path readable by the server"
            );
        }
        router = router
            .route(LOAD_LORA_ADAPTER_ROUTE.path, post(load_lora_adapter))
            .route(UNLOAD_LORA_ADAPTER_ROUTE.path, post(unload_lora_adapter));
    }

    let router = router
        .fallback(api_route_not_found)
        .method_not_allowed_fallback(api_method_not_allowed);

    let router = apply_edge_layers(
        router,
        cors_layer,
        router_max_body_limit,
        #[cfg(feature = "security")]
        edge.security,
    )
    .layer(Extension(agentic_defaults.approval_broker.clone()))
    .layer(Extension(skill_store))
    .layer(Extension(agentic_defaults))
    .layer(Extension(lora_adapter_api))
    .with_state(state);

    Ok(router)
}

fn apply_edge_layers<S>(
    router: Router<S>,
    cors_layer: CorsLayer,
    max_body_limit: usize,
    #[cfg(feature = "security")] security: Option<SecurityGuard>,
) -> Router<S>
where
    S: Clone + Send + Sync + 'static,
{
    #[cfg(feature = "security")]
    let router = match security {
        Some(security) => security.apply(router, max_body_limit),
        None => router,
    };
    router
        .layer(cors_layer)
        .layer(DefaultBodyLimit::max(max_body_limit))
}

fn path_is_in_namespace(path: &str, namespace: &str) -> bool {
    path == namespace
        || path
            .strip_prefix(namespace)
            .is_some_and(|suffix| suffix.starts_with('/'))
}

fn is_anthropic_api_request(uri: &Uri, headers: &HeaderMap) -> bool {
    if path_is_in_namespace(uri.path(), ANTHROPIC_MESSAGES_ROUTE.path) {
        return true;
    }
    if !path_is_in_namespace(uri.path(), SKILLS_ROUTE.path) {
        return false;
    }
    headers.contains_key("anthropic-version")
        || headers.contains_key("anthropic-beta")
        || uri.query().is_some_and(|query| {
            url::form_urlencoded::parse(query.as_bytes()).any(|(key, _)| key == "source")
        })
}

fn is_versioned_api_path(path: &str) -> bool {
    path == "/v1" || path.starts_with("/v1/")
}

fn api_protocol_error(
    uri: &Uri,
    headers: &HeaderMap,
    status: StatusCode,
    error: ApiError,
) -> axum::response::Response {
    let mut response = if is_anthropic_api_request(uri, headers) {
        anthropic_error_response(error)
    } else {
        openai_error_response(error)
    };
    *response.status_mut() = status;
    response
}

async fn api_route_not_found(uri: Uri, headers: HeaderMap) -> axum::response::Response {
    if !is_versioned_api_path(uri.path()) {
        return StatusCode::NOT_FOUND.into_response();
    }

    api_protocol_error(
        &uri,
        &headers,
        StatusCode::NOT_FOUND,
        ApiError::new(
            ApiErrorKind::NotFound,
            ROUTE_NOT_FOUND_MESSAGE,
            Some("not_found"),
            None,
        ),
    )
}

async fn api_method_not_allowed(uri: Uri, headers: HeaderMap) -> axum::response::Response {
    if !is_versioned_api_path(uri.path()) {
        return StatusCode::METHOD_NOT_ALLOWED.into_response();
    }

    api_protocol_error(
        &uri,
        &headers,
        StatusCode::METHOD_NOT_ALLOWED,
        ApiError::new(
            ApiErrorKind::InvalidRequest,
            METHOD_NOT_ALLOWED_MESSAGE,
            Some("method_not_allowed"),
            None,
        ),
    )
}

#[cfg(test)]
mod tests {
    use axum::body::to_bytes;

    use super::*;

    async fn response_body(response: axum::response::Response) -> serde_json::Value {
        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        serde_json::from_slice(&body).unwrap()
    }

    #[tokio::test]
    async fn unknown_openai_routes_use_openai_errors() {
        let response = api_route_not_found(Uri::from_static("/v1/unknown"), HeaderMap::new()).await;
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
        let body = response_body(response).await;
        assert_eq!(body["error"]["type"], "invalid_request_error");
        assert_eq!(body["error"]["code"], "not_found");
    }

    #[tokio::test]
    async fn unknown_anthropic_message_routes_use_anthropic_errors() {
        let response =
            api_route_not_found(Uri::from_static("/v1/messages/unknown"), HeaderMap::new()).await;
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
        let body = response_body(response).await;
        assert_eq!(body["type"], "error");
        assert_eq!(body["error"]["type"], "not_found_error");
    }

    #[tokio::test]
    async fn wrong_openai_methods_use_openai_errors() {
        let response =
            api_method_not_allowed(Uri::from_static("/v1/chat/completions"), HeaderMap::new())
                .await;
        assert_eq!(response.status(), StatusCode::METHOD_NOT_ALLOWED);
        let body = response_body(response).await;
        assert_eq!(body["error"]["type"], "invalid_request_error");
        assert_eq!(body["error"]["code"], "method_not_allowed");
    }

    #[tokio::test]
    async fn wrong_anthropic_message_methods_use_anthropic_errors() {
        let response =
            api_method_not_allowed(Uri::from_static("/v1/messages"), HeaderMap::new()).await;
        assert_eq!(response.status(), StatusCode::METHOD_NOT_ALLOWED);
        let body = response_body(response).await;
        assert_eq!(body["type"], "error");
        assert_eq!(body["error"]["type"], "invalid_request_error");
    }

    #[tokio::test]
    async fn anthropic_skill_fallbacks_use_anthropic_errors() {
        let mut headers = HeaderMap::new();
        headers.insert(
            "anthropic-version",
            "2023-06-01".parse().expect("valid header value"),
        );
        let response = api_method_not_allowed(Uri::from_static("/v1/skills"), headers).await;
        assert_eq!(response.status(), StatusCode::METHOD_NOT_ALLOWED);
        let body = response_body(response).await;
        assert_eq!(body["type"], "error");
        assert_eq!(body["error"]["type"], "invalid_request_error");

        let response = api_route_not_found(
            Uri::from_static("/v1/skills/unknown?source=custom"),
            HeaderMap::new(),
        )
        .await;
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
        let body = response_body(response).await;
        assert_eq!(body["type"], "error");
        assert_eq!(body["error"]["type"], "not_found_error");
    }
}

#[cfg(all(test, feature = "security"))]
mod security_tests {
    use std::net::{IpAddr, SocketAddr};

    use axum::{
        body::Body,
        extract::ConnectInfo,
        http::{header, HeaderMap, Request},
        routing::any,
    };
    use http_body_util::BodyExt;
    use tower::ServiceExt;

    use super::*;
    use crate::{
        route_registry::{RouteInfo, MISTRALRS_API_ROUTES, RUNTIME_LORA_API_ROUTES},
        security::SecurityOptions,
    };

    const CONTAINER_ROUTES: [RouteInfo; 3] = [
        CONTAINER_FILES_ROUTE,
        CONTAINER_FILE_ROUTE,
        CONTAINER_FILE_CONTENT_ROUTE,
    ];
    const KEY: &str = "sk-test-key";
    const ALLOWED_PEER: &str = "10.0.0.5";
    const OUTSIDE_PEER: &str = "192.0.2.5";
    const PROXY: &str = "10.1.1.1";
    const TEST_BODY_LIMIT: usize = 4 * 1024 * 1024;
    const CONTROL_PATHS: &[(&str, &str)] = &[
        ("POST", "/v1/models/unload"),
        ("POST", "/v1/models/reload"),
        ("POST", "/v1/models/tune"),
        ("POST", "/re_isq"),
        ("POST", "/calibration/start"),
        ("GET", "/calibration/status"),
        ("POST", "/calibration/apply"),
        ("POST", "/v1/system/doctor"),
        ("GET", "/v1/system/info"),
        ("GET", "/v1/files"),
        ("GET", "/v1/files/file-1"),
        ("DELETE", "/v1/files/file-1"),
        ("GET", "/v1/files/file-1/content"),
        ("GET", "/v1/skills"),
        ("GET", "/v1/skills/skill-1/versions"),
        ("POST", "/v1/load_lora_adapter"),
        ("POST", "/v1/unload_lora_adapter"),
        ("POST", "/v1/agent/approvals/approval-1"),
        ("GET", "/v1/containers/ctr-1/files"),
        ("GET", "/v1/containers/ctr-1/files/file-1"),
        ("GET", "/v1/containers/ctr-1/files/file-1/content"),
    ];
    const GPU_PATHS: &[&str] = &[
        "/v1/chat/completions",
        "/v1/responses",
        "/v1/images/generations",
        "/v1/audio/speech",
        "/v1/embeddings",
        "/v1/completions",
        "/v1/messages",
    ];

    fn options() -> SecurityOptions {
        SecurityOptions {
            control_allow_ips: vec!["10.0.0.0/8".to_string()],
            api_keys: vec![KEY.to_string()],
            trusted_proxies: vec![PROXY.to_string()],
            ..SecurityOptions::default()
        }
    }

    fn cors() -> CorsLayer {
        CorsLayer::new().allow_origin(AllowOrigin::any())
    }

    fn registered_route_paths() -> Vec<&'static str> {
        MISTRALRS_API_ROUTES
            .iter()
            .chain(RUNTIME_LORA_API_ROUTES)
            .chain(&CONTAINER_ROUTES)
            .map(|route| route.path)
            .collect()
    }

    fn router_with(security: Option<SecurityGuard>) -> Router {
        let mut router = Router::new();
        for path in registered_route_paths() {
            router = router.route(path, any(|| async { "ok" }));
        }
        let router = router
            .fallback(api_route_not_found)
            .method_not_allowed_fallback(api_method_not_allowed);
        apply_edge_layers(router, cors(), TEST_BODY_LIMIT, security)
    }

    fn guarded(options: &SecurityOptions) -> Router {
        router_with(Some(SecurityGuard::new(options).unwrap()))
    }

    struct Call<'a> {
        method: &'a str,
        uri: &'a str,
        peer: &'a str,
        headers: Vec<(&'a str, String)>,
        body: Vec<u8>,
    }

    fn call<'a>(method: &'a str, uri: &'a str, peer: &'a str) -> Call<'a> {
        Call {
            method,
            uri,
            peer,
            headers: Vec::new(),
            body: Vec::new(),
        }
    }

    impl<'a> Call<'a> {
        fn header(mut self, name: &'a str, value: &str) -> Self {
            self.headers.push((name, value.to_string()));
            self
        }

        fn key(self) -> Self {
            self.header("x-api-key", KEY)
        }

        fn body(mut self, body: Vec<u8>) -> Self {
            self.body = body;
            self
        }

        async fn send(self, router: &Router) -> (StatusCode, HeaderMap, String) {
            let peer: IpAddr = self.peer.parse().unwrap();
            let mut request = Request::builder()
                .method(self.method)
                .uri(self.uri)
                .extension(ConnectInfo(SocketAddr::new(peer, 40000)));
            for (name, value) in self.headers {
                request = request.header(name, value);
            }
            let response = router
                .clone()
                .oneshot(request.body(Body::from(self.body)).unwrap())
                .await
                .unwrap();
            let status = response.status();
            let headers = response.headers().clone();
            let body = response.into_body().collect().await.unwrap().to_bytes();
            (status, headers, String::from_utf8_lossy(&body).into_owned())
        }
    }

    async fn status(router: &Router, call: Call<'_>) -> StatusCode {
        call.send(router).await.0
    }

    #[tokio::test]
    async fn security_off_leaves_the_router_unchanged() {
        let router = router_with(None);
        let attack = call(
            "GET",
            "/v1/models/unload?q=%3Cscript%3Ealert(1)%3C/script%3E",
            "192.0.2.10",
        );
        assert_eq!(status(&router, attack).await, StatusCode::OK);
        for _ in 0..200 {
            let request = call("POST", "/v1/chat/completions", "192.0.2.10");
            assert_eq!(status(&router, request).await, StatusCode::OK);
        }
        let (code, _, body) = call("GET", "/v1/unknown", "192.0.2.10").send(&router).await;
        assert_eq!(code, StatusCode::NOT_FOUND);
        assert!(body.contains("not_found"));
    }

    #[tokio::test]
    async fn control_routes_refuse_clients_outside_the_allowlist() {
        let router = guarded(&options());
        for (method, path) in CONTROL_PATHS {
            let outside = call(method, path, "192.0.2.11").key();
            assert_eq!(
                status(&router, outside).await,
                StatusCode::FORBIDDEN,
                "{path}"
            );
            let allowed = call(method, path, "10.0.1.11").key();
            assert_eq!(status(&router, allowed).await, StatusCode::OK, "{path}");
        }
    }

    #[tokio::test]
    async fn control_routes_need_the_api_key() {
        let router = guarded(&options());
        for (method, path) in CONTROL_PATHS {
            let missing = call(method, path, "10.0.2.12");
            assert_eq!(
                status(&router, missing).await,
                StatusCode::UNAUTHORIZED,
                "{path}"
            );
        }
        let wrong =
            call("POST", "/v1/models/unload", "10.0.2.13").header("x-api-key", "sk-test-kez");
        assert_eq!(status(&router, wrong).await, StatusCode::UNAUTHORIZED);
        let prefix = call("POST", "/v1/models/unload", "10.0.2.13").header("x-api-key", "sk-test");
        assert_eq!(status(&router, prefix).await, StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn api_key_header_is_configurable_and_accepts_bearer() {
        let router = guarded(&SecurityOptions {
            api_key_header: "Authorization".to_string(),
            ..options()
        });
        let bearer = call("POST", "/v1/models/unload", "10.0.3.14")
            .header("authorization", &format!("Bearer {KEY}"));
        assert_eq!(status(&router, bearer).await, StatusCode::OK);
        let bare = call("POST", "/v1/models/unload", "10.0.3.14").header("authorization", KEY);
        assert_eq!(status(&router, bare).await, StatusCode::OK);
        let wrong_header = call("POST", "/v1/models/unload", "10.0.3.14").key();
        assert_eq!(
            status(&router, wrong_header).await,
            StatusCode::UNAUTHORIZED
        );
    }

    #[tokio::test]
    async fn hex_and_base64_api_keys_are_not_flagged() {
        let keys = ["3f9a1c7e5b2d4a60b8c1e9f07d6a2b45", "c2stdGVzdC1rZXk_LS1h"];
        let router = guarded(&SecurityOptions {
            api_keys: keys.iter().map(|key| key.to_string()).collect(),
            ..options()
        });
        for key in keys {
            let request = call("POST", "/v1/models/unload", "10.0.4.15").header("x-api-key", key);
            assert_eq!(status(&router, request).await, StatusCode::OK, "{key}");
        }
    }

    #[tokio::test]
    async fn inference_routes_stay_reachable_without_credentials() {
        let router = guarded(&options());
        for path in GPU_PATHS {
            let request = call("POST", path, "192.0.2.16");
            assert_eq!(status(&router, request).await, StatusCode::OK, "{path}");
        }
        for path in [
            "/v1/models",
            "/health",
            "/",
            "/v1/models/status",
            "/v1/responses/r1",
        ] {
            let request = call("GET", path, "192.0.2.16");
            assert_eq!(status(&router, request).await, StatusCode::OK, "{path}");
        }
    }

    #[tokio::test]
    async fn protect_all_routes_extends_access_control_to_inference() {
        let router = guarded(&SecurityOptions {
            protect_all_routes: true,
            ..options()
        });
        let outside = call("POST", "/v1/chat/completions", "192.0.2.17").key();
        assert_eq!(status(&router, outside).await, StatusCode::FORBIDDEN);
        let no_key = call("POST", "/v1/chat/completions", "10.0.5.17");
        assert_eq!(status(&router, no_key).await, StatusCode::UNAUTHORIZED);
        let allowed = call("POST", "/v1/chat/completions", "10.0.5.17").key();
        assert_eq!(status(&router, allowed).await, StatusCode::OK);
    }

    #[tokio::test]
    async fn gpu_routes_are_budgeted_per_client_and_route() {
        let router = guarded(&SecurityOptions {
            gpu_rate_limit_per_minute: 3,
            ..options()
        });
        for _ in 0..3 {
            let request = call("POST", "/v1/chat/completions", "192.0.2.18");
            assert_eq!(status(&router, request).await, StatusCode::OK);
        }
        let (code, headers, _) = call("POST", "/v1/chat/completions", "192.0.2.18")
            .send(&router)
            .await;
        assert_eq!(code, StatusCode::TOO_MANY_REQUESTS);
        assert!(headers.contains_key(header::RETRY_AFTER));
        let other_route = call("POST", "/v1/embeddings", "192.0.2.18");
        assert_eq!(status(&router, other_route).await, StatusCode::OK);
        let other_client = call("POST", "/v1/chat/completions", "192.0.2.19");
        assert_eq!(status(&router, other_client).await, StatusCode::OK);
        for _ in 0..10 {
            let request = call("GET", "/v1/models", "192.0.2.18");
            assert_eq!(status(&router, request).await, StatusCode::OK);
        }
    }

    #[tokio::test]
    async fn global_budget_applies_to_every_route() {
        let router = guarded(&SecurityOptions {
            rate_limit_per_minute: 5,
            ..options()
        });
        for _ in 0..5 {
            assert_eq!(
                status(&router, call("GET", "/health", "192.0.2.20")).await,
                StatusCode::OK
            );
        }
        let over = call("GET", "/health", "192.0.2.20");
        assert_eq!(status(&router, over).await, StatusCode::TOO_MANY_REQUESTS);
    }

    #[tokio::test]
    async fn attack_payloads_in_path_query_and_headers_are_refused() {
        let router = guarded(&options());
        let attacks = [
            call("GET", "/v1/../../etc/hosts", "192.0.2.21"),
            call(
                "GET",
                "/v1/models?name=1%27%20OR%20%271%27=%271",
                "192.0.2.22",
            ),
            call(
                "GET",
                "/v1/models?q=%3Cscript%3Ealert(1)%3C/script%3E",
                "192.0.2.23",
            ),
            call("GET", "/v1/models", "192.0.2.24").header("x-custom", "<script>alert(1)</script>"),
            call("GET", "/v1/models", "192.0.2.25").header("x-custom", "../../../../etc/hosts"),
        ];
        for attack in attacks {
            let label = format!("{} {:?}", attack.uri, attack.headers);
            assert_eq!(
                status(&router, attack).await,
                StatusCode::BAD_REQUEST,
                "{label}"
            );
        }
        let clean = call("GET", "/v1/models?limit=10&name=llama-3.1-8b", "192.0.2.26");
        assert_eq!(status(&router, clean).await, StatusCode::OK);
    }

    #[tokio::test]
    async fn bodies_are_not_screened() {
        let router = guarded(&options());
        let prompt = br#"{"messages":[{"role":"user","content":"<script>alert(1)</script> ' OR 1=1 -- ../../etc/hosts"}]}"#;
        let request = call("POST", "/v1/chat/completions", "192.0.2.27")
            .header("content-type", "application/json")
            .body(prompt.to_vec());
        assert_eq!(status(&router, request).await, StatusCode::OK);
    }

    #[tokio::test]
    async fn large_inference_bodies_pass_up_to_the_router_body_limit() {
        let router = guarded(&options());
        let image_prompt =
            call("POST", "/v1/chat/completions", "192.0.2.28").body(vec![b'a'; 2 * 1024 * 1024]);
        assert_eq!(status(&router, image_prompt).await, StatusCode::OK);
        let too_large =
            call("POST", "/v1/chat/completions", "192.0.2.29")
                .body(vec![b'a'; TEST_BODY_LIMIT + 1]);
        assert_eq!(
            status(&router, too_large).await,
            StatusCode::PAYLOAD_TOO_LARGE
        );
    }

    #[tokio::test]
    async fn uploads_need_multipart_and_respect_the_size_limit() {
        let router = guarded(&SecurityOptions {
            max_upload_mb: Some(1),
            ..options()
        });
        for path in ["/v1/files", "/v1/skills", "/v1/skills/skill-1/versions"] {
            let json = call("POST", path, "10.0.6.30")
                .key()
                .header("content-type", "application/json");
            assert_eq!(
                status(&router, json).await,
                StatusCode::UNSUPPORTED_MEDIA_TYPE,
                "{path}"
            );
            let none = call("POST", path, "10.0.6.30").key();
            assert_eq!(
                status(&router, none).await,
                StatusCode::UNSUPPORTED_MEDIA_TYPE,
                "{path}"
            );
            let multipart = call("POST", path, "10.0.6.30")
                .key()
                .header("content-type", "multipart/form-data; boundary=x")
                .body(vec![b'a'; 1024]);
            assert_eq!(status(&router, multipart).await, StatusCode::OK, "{path}");
            let big = call("POST", path, "10.0.6.30")
                .key()
                .header("content-type", "multipart/form-data; boundary=x")
                .body(vec![b'a'; 1024 * 1024 + 1]);
            assert_eq!(
                status(&router, big).await,
                StatusCode::PAYLOAD_TOO_LARGE,
                "{path}"
            );
            let unauthenticated = call("POST", path, "10.0.6.31")
                .header("content-type", "multipart/form-data; boundary=x");
            assert_eq!(
                status(&router, unauthenticated).await,
                StatusCode::UNAUTHORIZED,
                "{path}"
            );
        }
        let get = call("GET", "/v1/files", "10.0.6.30").key();
        assert_eq!(status(&router, get).await, StatusCode::OK);
    }

    #[tokio::test]
    async fn repeated_failed_api_keys_ban_the_client() {
        let router = guarded(&options());
        let attempt =
            || call("POST", "/v1/models/unload", "10.0.7.40").header("x-api-key", "guess");
        for _ in 0..5 {
            assert_eq!(status(&router, attempt()).await, StatusCode::UNAUTHORIZED);
        }
        let tripping = attempt().send(&router).await;
        assert_eq!(tripping.0, StatusCode::UNAUTHORIZED);
        let correct_key = call("POST", "/v1/models/unload", "10.0.7.40").key();
        assert_eq!(status(&router, correct_key).await, StatusCode::FORBIDDEN);
        let inference = call("POST", "/v1/chat/completions", "10.0.7.40");
        assert_eq!(status(&router, inference).await, StatusCode::FORBIDDEN);
        let neighbour = call("POST", "/v1/models/unload", "10.0.7.41").key();
        assert_eq!(status(&router, neighbour).await, StatusCode::OK);
    }

    #[tokio::test]
    async fn repeated_not_founds_ban_the_client() {
        let router = guarded(&options());
        let probe = || call("GET", "/v1/nothing-here", "192.0.2.50");
        for _ in 0..30 {
            assert_eq!(status(&router, probe()).await, StatusCode::NOT_FOUND);
        }
        assert_eq!(status(&router, probe()).await, StatusCode::NOT_FOUND);
        let after = call("GET", "/health", "192.0.2.50");
        assert_eq!(status(&router, after).await, StatusCode::FORBIDDEN);
    }

    #[tokio::test]
    async fn spoofed_forwarded_for_from_an_untrusted_peer_is_ignored() {
        let router = guarded(&options());
        let impersonate = call("POST", "/v1/models/unload", OUTSIDE_PEER)
            .key()
            .header("x-forwarded-for", "10.0.0.7");
        assert_eq!(status(&router, impersonate).await, StatusCode::FORBIDDEN);
        let hide = call("POST", "/v1/models/unload", ALLOWED_PEER)
            .key()
            .header("x-forwarded-for", "192.0.2.99");
        assert_eq!(status(&router, hide).await, StatusCode::OK);
    }

    #[tokio::test]
    async fn spoofed_forwarded_for_does_not_reset_a_budget() {
        let router = guarded(&SecurityOptions {
            gpu_rate_limit_per_minute: 2,
            ..options()
        });
        for index in 0..4u8 {
            let request = call("POST", "/v1/chat/completions", "192.0.2.51")
                .header("x-forwarded-for", &format!("203.0.113.{index}"));
            let expected = if index < 2 {
                StatusCode::OK
            } else {
                StatusCode::TOO_MANY_REQUESTS
            };
            assert_eq!(status(&router, request).await, expected);
        }
    }

    #[tokio::test]
    async fn trusted_proxy_forwarded_for_is_honored() {
        let router = guarded(&options());
        let allowed = call("POST", "/v1/models/unload", PROXY)
            .key()
            .header("x-forwarded-for", "10.0.8.60");
        assert_eq!(status(&router, allowed).await, StatusCode::OK);
        let refused = call("POST", "/v1/models/unload", PROXY)
            .key()
            .header("x-forwarded-for", "192.0.2.60");
        assert_eq!(status(&router, refused).await, StatusCode::FORBIDDEN);
    }

    #[tokio::test]
    async fn dual_stack_peers_match_ipv4_rules() {
        let router = guarded(&options());
        let proxy = format!("::ffff:{PROXY}");
        let forwarded = call("POST", "/v1/models/unload", &proxy)
            .key()
            .header("x-forwarded-for", "192.0.2.61");
        assert_eq!(status(&router, forwarded).await, StatusCode::FORBIDDEN);
        let forwarded_in = call("POST", "/v1/models/unload", &proxy)
            .key()
            .header("x-forwarded-for", "10.0.9.61");
        assert_eq!(status(&router, forwarded_in).await, StatusCode::OK);
        let direct = call("POST", "/v1/models/unload", "::ffff:10.0.9.62").key();
        assert_eq!(status(&router, direct).await, StatusCode::OK);
        let outside = call("POST", "/v1/models/unload", "::ffff:192.0.2.62").key();
        assert_eq!(status(&router, outside).await, StatusCode::FORBIDDEN);
    }

    #[tokio::test]
    async fn guard_rejections_carry_cors_headers() {
        let router = guarded(&options());
        let blocked = call("POST", "/v1/models/unload", "192.0.2.70")
            .key()
            .header("origin", "http://localhost:3000");
        let (code, headers, _) = blocked.send(&router).await;
        assert_eq!(code, StatusCode::FORBIDDEN);
        assert_eq!(
            headers.get(header::ACCESS_CONTROL_ALLOW_ORIGIN).unwrap(),
            "*"
        );
        let attack = call(
            "GET",
            "/v1/models?q=%3Cscript%3Ealert(1)%3C/script%3E",
            "192.0.2.71",
        )
        .header("origin", "http://localhost:3000");
        let (code, headers, _) = attack.send(&router).await;
        assert_eq!(code, StatusCode::BAD_REQUEST);
        assert!(headers.contains_key(header::ACCESS_CONTROL_ALLOW_ORIGIN));
        let preflight = call("OPTIONS", "/v1/models/unload", "192.0.2.72")
            .header("origin", "http://localhost:3000")
            .header("access-control-request-method", "POST");
        let (code, headers, _) = preflight.send(&router).await;
        assert_eq!(code, StatusCode::OK);
        assert!(headers.contains_key(header::ACCESS_CONTROL_ALLOW_ORIGIN));
    }

    #[tokio::test]
    async fn local_and_lan_referrers_are_not_flagged() {
        let router = guarded(&options());
        for (index, referer) in [
            "http://localhost:1234/",
            "http://192.168.1.20:1234/",
            "http://127.0.0.1:1234/ui",
        ]
        .into_iter()
        .enumerate()
        {
            let peer = format!("192.0.2.{}", 80 + index);
            let request = call("GET", "/v1/models", &peer).header("referer", referer);
            assert_eq!(status(&router, request).await, StatusCode::OK, "{referer}");
        }
    }

    #[tokio::test]
    async fn adapter_excluded_headers_are_not_screened() {
        let router = guarded(&options());
        let payload = "http://x/<script>alert(1)</script>";
        for (index, name) in ["referer", "origin", "user-agent"].into_iter().enumerate() {
            let peer = format!("192.0.2.{}", 85 + index);
            let request = call("GET", "/v1/models", &peer).header(name, payload);
            assert_eq!(status(&router, request).await, StatusCode::OK, "{name}");
        }
        let request = call("GET", "/v1/models", "192.0.2.89").header("x-custom", payload);
        assert_eq!(status(&router, request).await, StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn emergency_mode_serves_only_the_whitelist() {
        let router = guarded(&SecurityOptions {
            emergency_mode: true,
            emergency_whitelist: vec!["198.51.100.7".to_string()],
            ..options()
        });
        let outside = call("GET", "/v1/models", "192.0.2.91");
        assert_eq!(
            status(&router, outside).await,
            StatusCode::SERVICE_UNAVAILABLE
        );
        let whitelisted = call("GET", "/v1/models", "198.51.100.7");
        assert_eq!(status(&router, whitelisted).await, StatusCode::OK);
    }

    #[tokio::test]
    async fn passive_mode_observes_attacks_but_keeps_the_control_allowlist() {
        let router = guarded(&SecurityOptions {
            passive_mode: true,
            ..options()
        });
        let attack = call(
            "GET",
            "/v1/models?q=%3Cscript%3Ealert(1)%3C/script%3E",
            "192.0.2.92",
        );
        assert_eq!(status(&router, attack).await, StatusCode::OK);
        let outside = call("POST", "/v1/models/unload", "192.0.2.92").key();
        assert_eq!(status(&router, outside).await, StatusCode::FORBIDDEN);
    }

    #[tokio::test]
    async fn one_guard_shares_bans_across_routers() {
        let guard = SecurityGuard::new(&options()).unwrap();
        let api = router_with(Some(guard.clone()));
        let ui = guard.apply(
            Router::new().route("/chat", any(|| async { "ui" })),
            TEST_BODY_LIMIT,
        );
        let attack = call(
            "GET",
            "/chat?q=%3Cscript%3Ealert(1)%3C/script%3E",
            "192.0.2.93",
        );
        assert_eq!(status(&ui, attack).await, StatusCode::BAD_REQUEST);
        for _ in 0..12 {
            let _ = status(
                &ui,
                call(
                    "GET",
                    "/chat?q=%3Cscript%3Ealert(1)%3C/script%3E",
                    "192.0.2.93",
                ),
            )
            .await;
        }
        assert_eq!(
            status(&api, call("GET", "/health", "192.0.2.93")).await,
            StatusCode::FORBIDDEN
        );
    }

    #[tokio::test]
    async fn requests_without_a_peer_address_fail_closed() {
        let router = guarded(&options());
        let request = Request::builder()
            .uri("/v1/models")
            .body(Body::empty())
            .unwrap();
        let response = router.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::FORBIDDEN);
    }

    #[test]
    fn control_and_gpu_routes_exist_in_the_registry() {
        let registered = registered_route_paths();
        for (_, path) in CONTROL_PATHS.iter() {
            assert!(
                registered
                    .iter()
                    .any(|template| crate::security::matches_template(template, path)),
                "{path}"
            );
        }
        for path in GPU_PATHS {
            assert!(registered.contains(path), "{path}");
        }
    }
}
