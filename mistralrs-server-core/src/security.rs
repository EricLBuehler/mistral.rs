//! Optional application-layer security for the HTTP server, backed by axum-guard-rs.

use std::{collections::BTreeSet, net::SocketAddr, sync::Arc};

use anyhow::Result;
use axum::{
    extract::{connect_info::ConnectInfo, Request},
    http::{header, HeaderName, Method, StatusCode},
    middleware::{self, Next},
    response::{IntoResponse, Response},
    Router,
};
use axum_guard_rs::{
    BehaviorRule, GuardClientIp, GuardLayer, RouteConfig, SecurityConfig, SecurityHeadersConfig,
};
use subtle::ConstantTimeEq;

use crate::route_registry::{
    AGENT_APPROVAL_ROUTE, ANTHROPIC_MESSAGES_ROUTE, CALIBRATION_APPLY_ROUTE,
    CALIBRATION_START_ROUTE, CALIBRATION_STATUS_ROUTE, CHAT_COMPLETIONS_ROUTE, COMPLETIONS_ROUTE,
    CONTAINER_FILES_ROUTE, CONTAINER_FILE_CONTENT_ROUTE, CONTAINER_FILE_ROUTE, EMBEDDINGS_ROUTE,
    FILES_ROUTE, FILE_CONTENT_ROUTE, FILE_ROUTE, IMAGE_GENERATION_ROUTE, LOAD_LORA_ADAPTER_ROUTE,
    RELOAD_MODEL_ROUTE, RESPONSES_ROUTE, RE_ISQ_ROUTE, SKILLS_ROUTE, SKILL_VERSIONS_ROUTE,
    SPEECH_GENERATION_ROUTE, SYSTEM_DOCTOR_ROUTE, SYSTEM_INFO_ROUTE, TUNE_MODEL_ROUTE,
    UNLOAD_LORA_ADAPTER_ROUTE, UNLOAD_MODEL_ROUTE,
};

pub const DEFAULT_API_KEY_HEADER: &str = "x-api-key";
pub const DEFAULT_RATE_LIMIT_PER_MINUTE: u32 = 1200;
pub const DEFAULT_GPU_RATE_LIMIT_PER_MINUTE: u32 = 120;

const RATE_LIMIT_WINDOW_SECS: u64 = 60;
const BODY_READ_TIMEOUT_SECS: f64 = 30.0;
const BEARER_PREFIX: &str = "Bearer ";
const UPLOAD_CONTENT_TYPE: &str = "multipart/form-data";
const MB_TO_B: u64 = 1024 * 1024;
const UNAUTHORIZED_BAN_THRESHOLD: u32 = 5;
const UNAUTHORIZED_BAN_WINDOW_SECS: u64 = 300;
const NOT_FOUND_BAN_THRESHOLD: u32 = 30;
const NOT_FOUND_BAN_WINDOW_SECS: u64 = 60;
const BEHAVIOR_BAN_SECS: u64 = 3600;
const UNSUPPORTED_MEDIA_TYPE_MESSAGE: &str = "Unsupported content type";

const CONTROL_ROUTES: &[&str] = &[
    UNLOAD_MODEL_ROUTE.path,
    RELOAD_MODEL_ROUTE.path,
    TUNE_MODEL_ROUTE.path,
    RE_ISQ_ROUTE.path,
    CALIBRATION_START_ROUTE.path,
    CALIBRATION_STATUS_ROUTE.path,
    CALIBRATION_APPLY_ROUTE.path,
    SYSTEM_DOCTOR_ROUTE.path,
    SYSTEM_INFO_ROUTE.path,
    FILES_ROUTE.path,
    FILE_ROUTE.path,
    FILE_CONTENT_ROUTE.path,
    SKILLS_ROUTE.path,
    SKILL_VERSIONS_ROUTE.path,
    LOAD_LORA_ADAPTER_ROUTE.path,
    UNLOAD_LORA_ADAPTER_ROUTE.path,
    AGENT_APPROVAL_ROUTE.path,
    CONTAINER_FILES_ROUTE.path,
    CONTAINER_FILE_ROUTE.path,
    CONTAINER_FILE_CONTENT_ROUTE.path,
];

const UPLOAD_ROUTES: &[&str] = &[
    FILES_ROUTE.path,
    SKILLS_ROUTE.path,
    SKILL_VERSIONS_ROUTE.path,
];

const GPU_ROUTES: &[&str] = &[
    CHAT_COMPLETIONS_ROUTE.path,
    RESPONSES_ROUTE.path,
    IMAGE_GENERATION_ROUTE.path,
    SPEECH_GENERATION_ROUTE.path,
    EMBEDDINGS_ROUTE.path,
    COMPLETIONS_ROUTE.path,
    ANTHROPIC_MESSAGES_ROUTE.path,
];

/// Access control covers the control routes only, unless `protect_all_routes` is set.
#[derive(Clone)]
pub struct SecurityOptions {
    pub control_allow_ips: Vec<String>,
    pub api_keys: Vec<String>,
    pub api_key_header: String,
    pub protect_all_routes: bool,
    pub trusted_proxies: Vec<String>,
    pub rate_limit_per_minute: u32,
    pub gpu_rate_limit_per_minute: u32,
    pub max_upload_mb: Option<u64>,
    pub blocked_countries: Vec<String>,
    pub allowed_countries: Vec<String>,
    pub ipinfo_token: Option<String>,
    pub block_cloud_providers: Vec<String>,
    pub emergency_mode: bool,
    pub emergency_whitelist: Vec<String>,
    pub passive_mode: bool,
}

impl Default for SecurityOptions {
    fn default() -> Self {
        Self {
            control_allow_ips: Vec::new(),
            api_keys: Vec::new(),
            api_key_header: DEFAULT_API_KEY_HEADER.to_string(),
            protect_all_routes: false,
            trusted_proxies: Vec::new(),
            rate_limit_per_minute: DEFAULT_RATE_LIMIT_PER_MINUTE,
            gpu_rate_limit_per_minute: DEFAULT_GPU_RATE_LIMIT_PER_MINUTE,
            max_upload_mb: None,
            blocked_countries: Vec::new(),
            allowed_countries: Vec::new(),
            ipinfo_token: None,
            block_cloud_providers: Vec::new(),
            emergency_mode: false,
            emergency_whitelist: Vec::new(),
            passive_mode: false,
        }
    }
}

struct RoutePolicies {
    control: Arc<RouteConfig>,
    upload: Arc<RouteConfig>,
    gpu: Arc<RouteConfig>,
    protect_all_routes: bool,
}

impl RoutePolicies {
    fn resolve(&self, method: &str, path: &str) -> Option<Arc<RouteConfig>> {
        let is_post = method == Method::POST.as_str();
        if is_post && matches_any(UPLOAD_ROUTES, path) {
            return Some(self.upload.clone());
        }
        if self.protect_all_routes || matches_any(CONTROL_ROUTES, path) {
            return Some(self.control.clone());
        }
        (is_post && matches_any(GPU_ROUTES, path)).then(|| self.gpu.clone())
    }
}

fn matches_any(templates: &[&str], path: &str) -> bool {
    templates
        .iter()
        .any(|template| matches_template(template, path))
}

pub(crate) fn matches_template(template: &str, path: &str) -> bool {
    let mut template_segments = template.split('/');
    let mut path_segments = path.split('/');
    loop {
        match (template_segments.next(), path_segments.next()) {
            (None, None) => return true,
            (Some(expected), Some(actual)) => {
                let is_parameter = expected.starts_with('{') && !actual.is_empty();
                if !is_parameter && expected != actual {
                    return false;
                }
            }
            _ => return false,
        }
    }
}

fn api_key_verifier(api_keys: Vec<String>, bearer_header: bool) -> axum_guard_rs::AuthVerifier {
    Arc::new(move |credential: &str| {
        let credential = if bearer_header {
            credential.strip_prefix(BEARER_PREFIX).unwrap_or(credential)
        } else {
            credential
        };
        api_keys.iter().fold(false, |matched, key| {
            matched | bool::from(key.as_bytes().ct_eq(credential.as_bytes()))
        })
    })
}

fn behavior_ban_rule(status: u16, threshold: u32, window: u64) -> BehaviorRule {
    BehaviorRule {
        rule_type: "return_pattern".to_string(),
        threshold,
        window,
        pattern: format!("status:{status}"),
        action: "ban".to_string(),
        ban_duration: Some(BEHAVIOR_BAN_SECS),
        correlate_with_detection: false,
    }
}

fn to_set(values: &[String]) -> BTreeSet<String> {
    values.iter().cloned().collect()
}

fn route_policies(options: &SecurityOptions) -> RoutePolicies {
    let has_api_keys = !options.api_keys.is_empty();
    let bearer_header = options
        .api_key_header
        .eq_ignore_ascii_case(header::AUTHORIZATION.as_str());
    let control = RouteConfig {
        ip_whitelist: (!options.control_allow_ips.is_empty())
            .then(|| options.control_allow_ips.clone()),
        api_key_required: has_api_keys,
        api_key_header: has_api_keys.then(|| options.api_key_header.clone()),
        api_key_verifier: has_api_keys
            .then(|| api_key_verifier(options.api_keys.clone(), bearer_header)),
        ..RouteConfig::default()
    };
    let upload = RouteConfig {
        max_request_size: options.max_upload_mb.map(|mb| mb * MB_TO_B),
        ..control.clone()
    };
    let gpu = RouteConfig {
        rate_limit: Some(options.gpu_rate_limit_per_minute),
        rate_limit_window: Some(RATE_LIMIT_WINDOW_SECS),
        ..RouteConfig::default()
    };
    RoutePolicies {
        control: Arc::new(control),
        upload: Arc::new(upload),
        gpu: Arc::new(gpu),
        protect_all_routes: options.protect_all_routes,
    }
}

fn guard_config(options: &SecurityOptions) -> SecurityConfig {
    let mut security_headers = SecurityHeadersConfig::reference_default();
    security_headers.enabled = false;
    SecurityConfig {
        enable_redis: false,
        redis_url: None,
        security_headers,
        enable_cors: false,
        detection_scan_body: false,
        body_read_timeout: BODY_READ_TIMEOUT_SECS,
        enable_agent: false,
        enable_rate_limiting: true,
        rate_limit: options.rate_limit_per_minute,
        rate_limit_window: RATE_LIMIT_WINDOW_SECS,
        enable_ip_banning: true,
        passive_mode: options.passive_mode,
        trusted_proxies: options.trusted_proxies.clone(),
        blocked_countries: to_set(&options.blocked_countries),
        whitelist_countries: to_set(&options.allowed_countries),
        ipinfo_token: options.ipinfo_token.clone(),
        block_cloud_providers: (!options.block_cloud_providers.is_empty())
            .then(|| to_set(&options.block_cloud_providers)),
        emergency_mode: options.emergency_mode,
        emergency_whitelist: options.emergency_whitelist.clone(),
        global_behavior_rules: vec![
            behavior_ban_rule(
                StatusCode::UNAUTHORIZED.as_u16(),
                UNAUTHORIZED_BAN_THRESHOLD,
                UNAUTHORIZED_BAN_WINDOW_SECS,
            ),
            behavior_ban_rule(
                StatusCode::NOT_FOUND.as_u16(),
                NOT_FOUND_BAN_THRESHOLD,
                NOT_FOUND_BAN_WINDOW_SECS,
            ),
        ],
        ..SecurityConfig::default()
    }
}

async fn resolve_peer_ip(mut request: Request, next: Next) -> Response {
    let peer = request
        .extensions()
        .get::<ConnectInfo<SocketAddr>>()
        .map(|ConnectInfo(peer)| GuardClientIp(peer.ip().to_canonical()));
    if let Some(peer) = peer {
        request.extensions_mut().insert(peer);
    }
    next.run(request).await
}

async fn require_multipart_upload(request: Request, next: Next) -> Response {
    let is_upload =
        request.method() == Method::POST && matches_any(UPLOAD_ROUTES, request.uri().path());
    let is_multipart = request
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .is_some_and(|value| {
            value
                .split(';')
                .next()
                .is_some_and(|media| media.trim().eq_ignore_ascii_case(UPLOAD_CONTENT_TYPE))
        });
    if is_upload && !is_multipart {
        return (
            StatusCode::UNSUPPORTED_MEDIA_TYPE,
            UNSUPPORTED_MEDIA_TYPE_MESSAGE,
        )
            .into_response();
    }
    next.run(request).await
}

/// Clones share rate-limit windows and bans, so one guard can protect several routers.
#[derive(Clone)]
pub struct SecurityGuard {
    layer: GuardLayer,
}

impl SecurityGuard {
    pub fn new(options: &SecurityOptions) -> Result<Self> {
        HeaderName::from_bytes(options.api_key_header.as_bytes())?;
        if options.control_allow_ips.is_empty() && options.api_keys.is_empty() {
            tracing::warn!(
                "security is enabled without a control IP allowlist or API key; control routes stay open"
            );
        }
        let policies = Arc::new(route_policies(options));
        let layer = axum_guard_rs::with_security_config(&guard_config(options))?
            .with_route_configs(Arc::new(move |method, path| policies.resolve(method, path)));
        Ok(Self { layer })
    }

    /// Add the CORS layer after this call so blocked responses keep their CORS headers.
    pub fn apply<S>(&self, router: Router<S>, body_cap: usize) -> Router<S>
    where
        S: Clone + Send + Sync + 'static,
    {
        router
            .layer(middleware::from_fn(require_multipart_upload))
            .layer(self.layer.clone().with_body_cap(body_cap))
            .layer(middleware::from_fn(resolve_peer_ip))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn templates_match_whole_segments_only() {
        assert!(matches_template("/v1/files/{id}", "/v1/files/abc"));
        assert!(matches_template("/v1/files", "/v1/files"));
        assert!(!matches_template("/v1/files/{id}", "/v1/files/"));
        assert!(!matches_template("/v1/files/{id}", "/v1/files/a/b"));
        assert!(!matches_template("/v1/files", "/v1/files/"));
        assert!(!matches_template("/v1/files", "//v1/files"));
        assert!(!matches_template("/v1/files", "/v1/filesx"));
        assert!(matches_template(
            "/v1/skills/{skill_id}/versions",
            "/v1/skills/s1/versions"
        ));
    }

    #[test]
    fn api_key_verifier_compares_whole_keys() {
        let verify = api_key_verifier(vec!["alpha".to_string(), "bravo".to_string()], false);
        assert!(verify("alpha") && verify("bravo"));
        assert!(!verify("alp") && !verify("alphaa") && !verify("Alpha") && !verify(""));
        assert!(!verify("Bearer alpha"));
        let bearer = api_key_verifier(vec!["alpha".to_string()], true);
        assert!(bearer("Bearer alpha") && bearer("alpha"));
        assert!(!bearer("Bearer beta") && !bearer("bearer alpha"));
    }

    #[test]
    fn route_policies_follow_the_route_kind() {
        let policies = route_policies(&SecurityOptions {
            api_keys: vec!["k".to_string()],
            max_upload_mb: Some(2),
            ..SecurityOptions::default()
        });
        assert!(
            policies
                .resolve("GET", "/v1/models/unload")
                .unwrap()
                .api_key_required
        );
        assert!(
            policies
                .resolve("POST", "/v1/files")
                .unwrap()
                .max_request_size
                == Some(2 * MB_TO_B)
        );
        assert!(policies
            .resolve("GET", "/v1/files")
            .unwrap()
            .max_request_size
            .is_none());
        assert!(policies
            .resolve("POST", "/v1/chat/completions")
            .unwrap()
            .rate_limit
            .is_some());
        assert!(policies.resolve("GET", "/v1/chat/completions").is_none());
        assert!(policies.resolve("GET", "/v1/models").is_none());
    }

    #[test]
    fn an_invalid_api_key_header_is_rejected() {
        let options = SecurityOptions {
            api_key_header: "bad header".to_string(),
            ..SecurityOptions::default()
        };
        assert!(SecurityGuard::new(&options).is_err());
    }
}
