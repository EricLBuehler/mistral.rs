#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RouteInfo {
    pub path: &'static str,
    pub methods: &'static str,
    pub kind: RouteKind,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RouteKind {
    OpenAi,
    Anthropic,
    MistralRs,
    Docs,
    Ui,
}

impl RouteInfo {
    pub const fn new(path: &'static str, methods: &'static str, kind: RouteKind) -> Self {
        Self {
            path,
            methods,
            kind,
        }
    }
}

pub const CHAT_COMPLETIONS_ROUTE: RouteInfo =
    RouteInfo::new("/v1/chat/completions", "POST", RouteKind::OpenAi);
pub const ANTHROPIC_MESSAGES_ROUTE: RouteInfo =
    RouteInfo::new("/v1/messages", "POST", RouteKind::Anthropic);
pub const ANTHROPIC_COUNT_TOKENS_ROUTE: RouteInfo =
    RouteInfo::new("/v1/messages/count_tokens", "POST", RouteKind::Anthropic);
pub const COMPLETIONS_ROUTE: RouteInfo =
    RouteInfo::new("/v1/completions", "POST", RouteKind::OpenAi);
pub const EMBEDDINGS_ROUTE: RouteInfo = RouteInfo::new("/v1/embeddings", "POST", RouteKind::OpenAi);
pub const MODELS_ROUTE: RouteInfo = RouteInfo::new("/v1/models", "GET", RouteKind::OpenAi);
pub const LOAD_LORA_ADAPTER_ROUTE: RouteInfo =
    RouteInfo::new("/v1/load_lora_adapter", "POST", RouteKind::MistralRs);
pub const UNLOAD_LORA_ADAPTER_ROUTE: RouteInfo =
    RouteInfo::new("/v1/unload_lora_adapter", "POST", RouteKind::MistralRs);
pub const LIST_LORA_ADAPTERS_ROUTE: RouteInfo =
    RouteInfo::new("/v1/lora_adapters", "GET", RouteKind::MistralRs);
pub const UNLOAD_MODEL_ROUTE: RouteInfo =
    RouteInfo::new("/v1/models/unload", "POST", RouteKind::MistralRs);
pub const RELOAD_MODEL_ROUTE: RouteInfo =
    RouteInfo::new("/v1/models/reload", "POST", RouteKind::MistralRs);
pub const MODEL_STATUS_ROUTE: RouteInfo =
    RouteInfo::new("/v1/models/status", "POST", RouteKind::MistralRs);
pub const TUNE_MODEL_ROUTE: RouteInfo =
    RouteInfo::new("/v1/models/tune", "POST", RouteKind::MistralRs);
pub const SYSTEM_INFO_ROUTE: RouteInfo =
    RouteInfo::new("/v1/system/info", "GET", RouteKind::MistralRs);
pub const SYSTEM_DOCTOR_ROUTE: RouteInfo =
    RouteInfo::new("/v1/system/doctor", "POST", RouteKind::MistralRs);
pub const HEALTH_ROUTE: RouteInfo = RouteInfo::new("/health", "GET", RouteKind::MistralRs);
pub const ROOT_ROUTE: RouteInfo = RouteInfo::new("/", "GET", RouteKind::MistralRs);
pub const RE_ISQ_ROUTE: RouteInfo = RouteInfo::new("/re_isq", "POST", RouteKind::MistralRs);
pub const CALIBRATION_START_ROUTE: RouteInfo =
    RouteInfo::new("/calibration/start", "POST", RouteKind::MistralRs);
pub const CALIBRATION_STATUS_ROUTE: RouteInfo =
    RouteInfo::new("/calibration/status", "GET", RouteKind::MistralRs);
pub const CALIBRATION_APPLY_ROUTE: RouteInfo =
    RouteInfo::new("/calibration/apply", "POST", RouteKind::MistralRs);
pub const IMAGE_GENERATION_ROUTE: RouteInfo =
    RouteInfo::new("/v1/images/generations", "POST", RouteKind::OpenAi);
pub const FILES_ROUTE: RouteInfo = RouteInfo::new("/v1/files", "GET, POST", RouteKind::OpenAi);
pub const FILE_ROUTE: RouteInfo =
    RouteInfo::new("/v1/files/{id}", "GET, DELETE", RouteKind::OpenAi);
pub const FILE_CONTENT_ROUTE: RouteInfo =
    RouteInfo::new("/v1/files/{id}/content", "GET", RouteKind::OpenAi);
pub const CONTAINER_FILES_ROUTE: RouteInfo = RouteInfo::new(
    "/v1/containers/{container_id}/files",
    "GET",
    RouteKind::OpenAi,
);
pub const CONTAINER_FILE_ROUTE: RouteInfo = RouteInfo::new(
    "/v1/containers/{container_id}/files/{file_id}",
    "GET",
    RouteKind::OpenAi,
);
pub const CONTAINER_FILE_CONTENT_ROUTE: RouteInfo = RouteInfo::new(
    "/v1/containers/{container_id}/files/{file_id}/content",
    "GET",
    RouteKind::OpenAi,
);
pub const SPEECH_GENERATION_ROUTE: RouteInfo =
    RouteInfo::new("/v1/audio/speech", "POST", RouteKind::OpenAi);
pub const AGENT_APPROVAL_ROUTE: RouteInfo = RouteInfo::new(
    "/v1/agent/approvals/{approval_id}",
    "POST",
    RouteKind::MistralRs,
);
pub const RESPONSES_ROUTE: RouteInfo = RouteInfo::new("/v1/responses", "POST", RouteKind::OpenAi);
pub const SKILLS_ROUTE: RouteInfo = RouteInfo::new("/v1/skills", "GET, POST", RouteKind::OpenAi);
pub const SKILL_VERSIONS_ROUTE: RouteInfo = RouteInfo::new(
    "/v1/skills/{skill_id}/versions",
    "GET, POST",
    RouteKind::OpenAi,
);
pub const RESPONSE_ROUTE: RouteInfo = RouteInfo::new(
    "/v1/responses/{response_id}",
    "GET, DELETE",
    RouteKind::OpenAi,
);
pub const CANCEL_RESPONSE_ROUTE: RouteInfo = RouteInfo::new(
    "/v1/responses/{response_id}/cancel",
    "POST",
    RouteKind::OpenAi,
);
pub const SESSION_ROUTE: RouteInfo = RouteInfo::new(
    "/v1/sessions/{session_id}",
    "GET, PUT, DELETE",
    RouteKind::MistralRs,
);

pub const MISTRALRS_API_ROUTES: &[RouteInfo] = &[
    ROOT_ROUTE,
    HEALTH_ROUTE,
    MODELS_ROUTE,
    LIST_LORA_ADAPTERS_ROUTE,
    UNLOAD_MODEL_ROUTE,
    RELOAD_MODEL_ROUTE,
    MODEL_STATUS_ROUTE,
    TUNE_MODEL_ROUTE,
    SYSTEM_INFO_ROUTE,
    SYSTEM_DOCTOR_ROUTE,
    CHAT_COMPLETIONS_ROUTE,
    RESPONSES_ROUTE,
    SKILLS_ROUTE,
    SKILL_VERSIONS_ROUTE,
    RESPONSE_ROUTE,
    CANCEL_RESPONSE_ROUTE,
    ANTHROPIC_MESSAGES_ROUTE,
    ANTHROPIC_COUNT_TOKENS_ROUTE,
    COMPLETIONS_ROUTE,
    EMBEDDINGS_ROUTE,
    IMAGE_GENERATION_ROUTE,
    SPEECH_GENERATION_ROUTE,
    FILES_ROUTE,
    FILE_ROUTE,
    FILE_CONTENT_ROUTE,
    SESSION_ROUTE,
    AGENT_APPROVAL_ROUTE,
    RE_ISQ_ROUTE,
    CALIBRATION_START_ROUTE,
    CALIBRATION_STATUS_ROUTE,
    CALIBRATION_APPLY_ROUTE,
];

pub const RUNTIME_LORA_API_ROUTES: &[RouteInfo] =
    &[LOAD_LORA_ADAPTER_ROUTE, UNLOAD_LORA_ADAPTER_ROUTE];

#[cfg(feature = "swagger-ui")]
pub const MISTRALRS_SWAGGER_ROUTES: &[RouteInfo] = &[
    RouteInfo::new("/api-doc/openapi.json", "GET", RouteKind::Docs),
    RouteInfo::new("/docs", "GET", RouteKind::Docs),
    RouteInfo::new("/docs/", "GET", RouteKind::Docs),
    RouteInfo::new("/docs/{*rest}", "GET", RouteKind::Docs),
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn route_registry_matches_lora_route_visibility() {
        assert!(MISTRALRS_API_ROUTES.contains(&LIST_LORA_ADAPTERS_ROUTE));
        assert!(!MISTRALRS_API_ROUTES.contains(&LOAD_LORA_ADAPTER_ROUTE));
        assert!(!MISTRALRS_API_ROUTES.contains(&UNLOAD_LORA_ADAPTER_ROUTE));
        assert_eq!(
            RUNTIME_LORA_API_ROUTES,
            &[LOAD_LORA_ADAPTER_ROUTE, UNLOAD_LORA_ADAPTER_ROUTE]
        );
    }
}
