use either::Either;
use indexmap::IndexMap;
use mistralrs_audio::AudioInput;
use mistralrs_quant::IsqType;
#[cfg(feature = "pyo3_macros")]
use pyo3::{pyclass, pymethods};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::VideoInput;

use crate::{
    response::Response, sampler::SamplingParams, tools::ToolChoice, AdapterSelection,
    AgentPermission, AgentToolApprovalHandler, CodeExecutionPermission, CustomLogitsProcessor,
    DiffusionGenerationParams, Tool,
};
use std::{fmt::Debug, path::PathBuf, sync::Arc};
use tokio::sync::mpsc::Sender;

pub type LlguidanceGrammar = llguidance::api::TopLevelGrammar;

#[derive(Clone, Serialize, Deserialize)]
/// Control the constraint with llguidance.
pub enum Constraint {
    Regex(String),
    Lark(String),
    JsonSchema(serde_json::Value),
    Llguidance(LlguidanceGrammar),
    None,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
#[cfg_attr(feature = "pyo3_macros", pyo3::pyclass(eq, eq_int))]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
/// Image generation response format
pub enum ImageGenerationResponseFormat {
    Url,
    B64Json,
}

pub type MessageContent = Either<String, Vec<IndexMap<String, Value>>>;

/// Reasoning effort passed to chat templates. `none` is a parse-only alias for `off`.
#[derive(Clone, Copy, Debug, Serialize, PartialEq, Eq)]
#[cfg_attr(feature = "pyo3_macros", pyo3::pyclass(eq, eq_int))]
#[serde(rename_all = "lowercase")]
pub enum ReasoningEffort {
    /// Disable reasoning.
    Off,
    /// Low reasoning effort.
    Low,
    /// Medium reasoning effort.
    Medium,
    /// High reasoning effort.
    High,
    /// Maximum reasoning effort.
    XHigh,
}

impl ReasoningEffort {
    /// Return the canonical wire value.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
            Self::XHigh => "xhigh",
        }
    }

    /// Return whether this effort disables reasoning.
    pub const fn is_off(self) -> bool {
        matches!(self, Self::Off)
    }
}

#[cfg(feature = "utoipa")]
impl utoipa::PartialSchema for ReasoningEffort {
    fn schema() -> utoipa::openapi::RefOr<utoipa::openapi::schema::Schema> {
        use utoipa::openapi::{schema::SchemaType, ObjectBuilder, RefOr, Schema, Type};

        RefOr::T(Schema::Object(
            ObjectBuilder::new()
                .schema_type(SchemaType::Type(Type::String))
                .description(Some(
                    "Reasoning effort. `none` aliases `off`; `max` aliases `xhigh`.",
                ))
                .enum_values(Some(
                    ["off", "none", "low", "medium", "high", "xhigh", "max"]
                        .into_iter()
                        .map(|value| Value::String(value.to_string()))
                        .collect::<Vec<_>>(),
                ))
                .build(),
        ))
    }
}

#[cfg(feature = "utoipa")]
impl utoipa::ToSchema for ReasoningEffort {}

impl std::fmt::Display for ReasoningEffort {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[error(
    "invalid reasoning effort `{value}`; expected one of: off, none, low, medium, high, xhigh, max"
)]
/// Error returned when a reasoning effort string is invalid.
pub struct ReasoningEffortParseError {
    value: String,
}

impl std::str::FromStr for ReasoningEffort {
    type Err = ReasoningEffortParseError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "off" | "none" => Ok(Self::Off),
            "low" => Ok(Self::Low),
            "medium" => Ok(Self::Medium),
            "high" => Ok(Self::High),
            "xhigh" | "max" => Ok(Self::XHigh),
            _ => Err(ReasoningEffortParseError {
                value: value.to_string(),
            }),
        }
    }
}

impl<'de> Deserialize<'de> for ReasoningEffort {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        value.parse().map_err(serde::de::Error::custom)
    }
}

/// Default thinking toggle when neither reasoning control is provided.
pub const DEFAULT_ENABLE_THINKING: bool = true;

/// Effective reasoning controls after validating their relationship.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResolvedReasoningControls {
    /// Effective thinking toggle.
    pub enable_thinking: bool,
    /// Explicit effort, if the caller selected one.
    pub reasoning_effort: Option<ReasoningEffort>,
}

/// Contradictory reasoning controls supplied by a caller.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum ReasoningControlError {
    #[error("reasoning effort `off` conflicts with enable_thinking=true")]
    OffWithThinkingEnabled,
    #[error("reasoning effort `{0}` conflicts with enable_thinking=false")]
    EffortWithThinkingDisabled(ReasoningEffort),
}

/// Validate reasoning controls and derive the effective thinking toggle.
pub fn resolve_reasoning_controls(
    enable_thinking: Option<bool>,
    reasoning_effort: Option<ReasoningEffort>,
) -> Result<ResolvedReasoningControls, ReasoningControlError> {
    let enable_thinking = match (enable_thinking, reasoning_effort) {
        (Some(true), Some(ReasoningEffort::Off)) => {
            return Err(ReasoningControlError::OffWithThinkingEnabled)
        }
        (Some(false), Some(effort)) if !effort.is_off() => {
            return Err(ReasoningControlError::EffortWithThinkingDisabled(effort))
        }
        (_, Some(effort)) => !effort.is_off(),
        (enable_thinking, None) => enable_thinking.unwrap_or(DEFAULT_ENABLE_THINKING),
    };

    Ok(ResolvedReasoningControls {
        enable_thinking,
        reasoning_effort,
    })
}

#[derive(Clone, Debug, Serialize, Deserialize)]
/// Message or messages for a [`Request`].
pub enum RequestMessage {
    Chat {
        messages: Vec<IndexMap<String, MessageContent>>,
        enable_thinking: Option<bool>,
        /// Reasoning effort level for templates with configurable reasoning
        reasoning_effort: Option<ReasoningEffort>,
    },
    Completion {
        text: String,
        echo_prompt: bool,
        best_of: Option<usize>,
    },
    CompletionTokens(Vec<u32>),
    MultimodalChat {
        #[serde(skip)] // TODO
        images: Vec<image::DynamicImage>,
        #[serde(skip)] // TODO
        audios: Vec<AudioInput>,
        #[serde(skip)]
        videos: Vec<VideoInput>,
        messages: Vec<IndexMap<String, MessageContent>>,
        enable_thinking: Option<bool>,
        /// Reasoning effort level for templates with configurable reasoning
        reasoning_effort: Option<ReasoningEffort>,
    },
    ImageGeneration {
        prompt: String,
        format: ImageGenerationResponseFormat,
        generation_params: DiffusionGenerationParams,
        save_file: Option<PathBuf>,
    },
    SpeechGeneration {
        prompt: String,
    },
    Embedding {
        prompt: String,
    },
    EmbeddingTokens {
        prompt: Vec<u32>,
    },
}

fn default_responder<T>() -> Sender<T> {
    let (sender, _) = tokio::sync::mpsc::channel(1);
    sender
}

#[cfg_attr(feature = "pyo3_macros", pyo3::pyclass(eq, eq_int))]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Default)]
pub enum SearchContextSize {
    #[serde(rename = "low")]
    Low,
    #[default]
    #[serde(rename = "medium")]
    Medium,
    #[serde(rename = "high")]
    High,
}

#[cfg_attr(feature = "pyo3_macros", pyclass(eq))]
#[cfg_attr(feature = "pyo3_macros", pyo3(get_all))]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ApproximateUserLocation {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub city: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub country: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub region: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timezone: Option<String>,
}

#[cfg(feature = "pyo3_macros")]
#[pymethods]
impl ApproximateUserLocation {
    #[new]
    fn py_new(city: String, country: String, region: String, timezone: String) -> Self {
        Self {
            city: Some(city),
            country: Some(country),
            region: Some(region),
            timezone: Some(timezone),
        }
    }
}

#[cfg_attr(feature = "pyo3_macros", pyclass(eq))]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type")]
pub enum WebSearchUserLocation {
    #[serde(rename = "approximate")]
    Approximate {
        approximate: ApproximateUserLocation,
    },
}

#[cfg(feature = "pyo3_macros")]
#[pymethods]
impl WebSearchUserLocation {
    #[staticmethod]
    fn approximate(approximate: ApproximateUserLocation) -> Self {
        Self::Approximate { approximate }
    }
}

#[cfg_attr(feature = "pyo3_macros", pyclass(eq))]
#[cfg_attr(feature = "pyo3_macros", pyo3(get_all))]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Default)]
pub struct WebSearchOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub search_context_size: Option<SearchContextSize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user_location: Option<WebSearchUserLocation>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub filters: Option<WebSearchFilters>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub external_web_access: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub return_token_budget: Option<WebSearchReturnTokenBudget>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub search_content_types: Option<Vec<WebSearchContentType>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image_settings: Option<WebSearchImageSettings>,
    /// Override the description for the search tool.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub search_description: Option<String>,
    /// Override the description for the extraction tool.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extract_description: Option<String>,
}

#[cfg_attr(feature = "pyo3_macros", pyclass(eq))]
#[cfg_attr(feature = "pyo3_macros", pyo3(get_all))]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Default)]
pub struct WebSearchFilters {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub allowed_domains: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub blocked_domains: Option<Vec<String>>,
}

#[cfg_attr(feature = "pyo3_macros", pyclass(eq, eq_int))]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum WebSearchContentType {
    Text,
    Image,
}

#[cfg_attr(feature = "pyo3_macros", pyclass(eq, eq_int))]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum WebSearchReturnTokenBudget {
    Default,
    Unlimited,
}

#[cfg_attr(feature = "pyo3_macros", pyclass(eq))]
#[cfg_attr(feature = "pyo3_macros", pyo3(get_all))]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Default)]
pub struct WebSearchImageSettings {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_results: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub caption: Option<bool>,
}

#[cfg(feature = "pyo3_macros")]
#[pymethods]
impl WebSearchOptions {
    #[new]
    #[pyo3(signature = (
        search_context_size = None,
        user_location = None,
        search_description = None,
        extract_description = None,
    ))]
    fn py_new(
        search_context_size: Option<SearchContextSize>,
        user_location: Option<WebSearchUserLocation>,
        search_description: Option<String>,
        extract_description: Option<String>,
    ) -> Self {
        Self {
            search_context_size,
            user_location,
            filters: None,
            external_web_access: None,
            return_token_budget: None,
            search_content_types: None,
            image_settings: None,
            search_description,
            extract_description,
        }
    }
}

#[derive(Clone, Serialize, Deserialize)]
/// A normal request request to the `MistralRs`.
/// - `messages`: Messages for the request
/// - `sampling_params`: Sampling parameters for generation
/// - `response`: Object to send the result through
/// - `return_logprobs`: Whether to return logprobs
/// - `is_streaming`: Control whether the request is streaming, if so chunk responses will be sent
/// - `id`: Request ID
/// - `constraint`: Constraint to use during generation
/// - `suffix`: Suffix to add
/// - `tools`: Tools available in this request
/// - `tool_choice`: Choice of tools
/// - `logits_processors`: Custom logits processors. Order of application:
///     1) Apply penalties from `sampling_params`
///     2) Apply these custom logits processors sequentially
///     3) Apply temperature and softmax
///     4) Sample the next token (topk, topp, minp, etc)
/// - `return_raw_logits`: Return raw logits.
/// - `truncate_sequence`: Whether to truncate the prompt if it exceeds the model's maximum context length.
pub struct NormalRequest {
    pub messages: RequestMessage,
    pub sampling_params: SamplingParams,
    #[serde(default = "default_responder")]
    #[serde(skip)]
    pub response: Sender<Response>,
    pub return_logprobs: bool,
    pub is_streaming: bool,
    pub id: usize,
    pub constraint: Constraint,
    pub suffix: Option<String>,
    pub tools: Option<Vec<Tool>>,
    pub tool_choice: Option<ToolChoice>,
    #[serde(skip)]
    pub logits_processors: Option<Vec<Arc<dyn CustomLogitsProcessor>>>,
    pub return_raw_logits: bool,
    pub web_search_options: Option<WebSearchOptions>,
    /// When true, registered code-execution tools are injected and the agentic loop runs.
    #[serde(default)]
    pub enable_code_execution: bool,
    /// When true, registered shell tools are injected and the agentic loop runs.
    #[serde(default)]
    pub enable_shell: bool,
    #[serde(default)]
    pub shell_options: Option<mistralrs_mcp::ShellOptions>,
    #[serde(default)]
    pub code_execution_permission: Option<CodeExecutionPermission>,
    #[serde(skip)]
    pub code_execution_approval_notifier: Option<Arc<mistralrs_mcp::CodeExecutionApprovalNotifier>>,
    #[serde(default)]
    pub agent_permission: Option<AgentPermission>,
    #[serde(skip)]
    pub agent_approval_handler: Option<AgentToolApprovalHandler>,
    #[serde(skip)]
    pub agent_approval_notifier: Option<Arc<mistralrs_mcp::AgentToolApprovalNotifier>>,
    pub max_tool_rounds: Option<usize>,
    /// URL to POST `{"name": ..., "arguments": ...}` to when no server-side callback is registered. Expects `{"content": "..."}` back.
    pub tool_dispatch_url: Option<String>,
    pub model_id: Option<String>,
    #[serde(default)]
    pub adapter: Option<AdapterSelection>,
    #[serde(default)]
    pub truncate_sequence: bool,
    /// Persistent agentic state. If `None`, a new session is created and the ID is returned in the response.
    #[serde(default)]
    pub session_id: Option<String>,
    /// Required output files. The runtime asks the model to produce them and surfaces a `File` (or error placeholder) for each.
    #[serde(default)]
    pub files: Option<Vec<crate::files::RequestedFile>>,
    /// User-provided input files attached to this request.
    #[serde(default)]
    pub input_files: Vec<crate::files::File>,
}

impl NormalRequest {
    pub fn new_simple(
        messages: RequestMessage,
        sampling_params: SamplingParams,
        response: Sender<Response>,
        id: usize,
        tools: Option<Vec<Tool>>,
        tool_choice: Option<ToolChoice>,
    ) -> Self {
        Self {
            messages,
            sampling_params,
            response,
            id,
            tools,
            tool_choice,
            return_logprobs: false,
            is_streaming: false,
            constraint: Constraint::None,
            suffix: None,
            logits_processors: None,
            return_raw_logits: false,
            web_search_options: None,
            enable_code_execution: false,
            enable_shell: false,
            shell_options: None,
            code_execution_permission: None,
            code_execution_approval_notifier: None,
            agent_permission: None,
            agent_approval_handler: None,
            agent_approval_notifier: None,
            max_tool_rounds: None,
            tool_dispatch_url: None,
            model_id: None,
            adapter: None,
            truncate_sequence: false,
            session_id: None,
            files: None,
            input_files: Vec::new(),
        }
    }
}

#[derive(Clone, Serialize, Deserialize)]
/// Request to tokenize some messages or some text.
/// - `add_generation_prompt` is only applicable if chat messages are provided and not a raw string.
pub struct TokenizationRequest {
    pub text: Either<Vec<IndexMap<String, MessageContent>>, String>,
    pub tools: Option<Vec<Tool>>,
    pub add_generation_prompt: bool,
    pub add_special_tokens: bool,
    pub enable_thinking: Option<bool>,
    pub reasoning_effort: Option<ReasoningEffort>,
    #[serde(default = "default_responder")]
    #[serde(skip)]
    pub response: Sender<anyhow::Result<Vec<u32>>>,
}

#[derive(Clone, Serialize, Deserialize)]
/// Request to detokenize some text.
pub struct DetokenizationRequest {
    pub tokens: Vec<u32>,
    pub skip_special_tokens: bool,
    #[serde(default = "default_responder")]
    #[serde(skip)]
    pub response: Sender<anyhow::Result<String>>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
/// Online calibration lifecycle action.
pub enum CalibrationAction {
    /// Begin collecting activation statistics from live traffic.
    Start,
    /// Report per-layer collection progress.
    Status,
    /// Requantize with the collected statistics and hot-swap the layers.
    Apply {
        save_cimatrix: Option<std::path::PathBuf>,
    },
}

#[derive(Clone, Serialize, Deserialize)]
pub struct CalibrationRequest {
    pub action: CalibrationAction,
    #[serde(default = "default_responder")]
    #[serde(skip)]
    pub response: Sender<anyhow::Result<crate::CalibrationStatus>>,
}

#[derive(Clone, Serialize, Deserialize)]
/// A request to the Engine, encapsulating the various parameters as well as
/// the `mpsc` response `Sender` used to return the [`Response`].
pub enum Request {
    Normal(Box<NormalRequest>),
    ReIsq(IsqType),
    Calibration(CalibrationRequest),
    Tokenize(TokenizationRequest),
    Detokenize(DetokenizationRequest),
    // Sending a terminate request causes the `run` function to return to the thread created in `MistralRs::new`,
    // and then Engine will be dropped.
    Terminate,
    TerminateAllSeqsNextStep,
}

impl Debug for Request {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Request::Normal(boxed_req) => {
                let NormalRequest {
                    messages,
                    sampling_params,
                    is_streaming,
                    id,
                    ..
                } = &**boxed_req;
                write!(
                    f,
                    "Request {id} {{ messages: `{messages:?}`, sampling_params: {sampling_params:?}, is_streaming: {is_streaming}}}",
                )
            }
            Request::ReIsq(tp) => {
                write!(f, "Re ISQ Request {tp:?}",)
            }
            Request::Calibration(req) => {
                write!(f, "Calibration Request {:?}", req.action)
            }
            Request::Tokenize(req) => {
                write!(f, "Tokenization Request {:?}", req.text)
            }
            Request::Detokenize(req) => {
                write!(f, "Tokenization Request {:?}", req.tokens)
            }
            Request::Terminate => write!(f, "Termination Request"),
            Request::TerminateAllSeqsNextStep => write!(f, "Terminate All Seqs Next Step"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reasoning_effort_parsing_is_canonical() {
        let cases = [
            ("off", ReasoningEffort::Off),
            (" none ", ReasoningEffort::Off),
            ("LOW", ReasoningEffort::Low),
            ("Medium", ReasoningEffort::Medium),
            ("high", ReasoningEffort::High),
            (" XHIGH\n", ReasoningEffort::XHigh),
            ("max", ReasoningEffort::XHigh),
        ];

        for (input, expected) in cases {
            assert_eq!(input.parse::<ReasoningEffort>().unwrap(), expected);
        }
        assert_eq!(
            "extreme"
                .parse::<ReasoningEffort>()
                .unwrap_err()
                .to_string(),
            "invalid reasoning effort `extreme`; expected one of: off, none, low, medium, high, xhigh, max"
        );
        assert_eq!(
            serde_json::from_str::<ReasoningEffort>(r#"" NoNe ""#).unwrap(),
            ReasoningEffort::Off
        );
        assert_eq!(
            serde_json::to_string(&ReasoningEffort::XHigh).unwrap(),
            r#""xhigh""#
        );
    }

    #[test]
    fn reasoning_controls_resolve_consistently() {
        let cases = [
            (None, None, true, None),
            (Some(true), None, true, None),
            (Some(false), None, false, None),
            (
                None,
                Some(ReasoningEffort::Off),
                false,
                Some(ReasoningEffort::Off),
            ),
            (
                Some(false),
                Some(ReasoningEffort::Off),
                false,
                Some(ReasoningEffort::Off),
            ),
            (
                None,
                Some(ReasoningEffort::Low),
                true,
                Some(ReasoningEffort::Low),
            ),
            (
                Some(true),
                Some(ReasoningEffort::XHigh),
                true,
                Some(ReasoningEffort::XHigh),
            ),
        ];

        for (enable_thinking, reasoning_effort, expected_enabled, expected_effort) in cases {
            let resolved = resolve_reasoning_controls(enable_thinking, reasoning_effort).unwrap();
            assert_eq!(resolved.enable_thinking, expected_enabled);
            assert_eq!(resolved.reasoning_effort, expected_effort);
        }

        assert_eq!(
            resolve_reasoning_controls(Some(true), Some(ReasoningEffort::Off)),
            Err(ReasoningControlError::OffWithThinkingEnabled)
        );
        assert_eq!(
            resolve_reasoning_controls(Some(false), Some(ReasoningEffort::High)),
            Err(ReasoningControlError::EffortWithThinkingDisabled(
                ReasoningEffort::High
            ))
        );
    }

    #[test]
    fn request_replication_keeps_an_exact_generation() {
        let (response, _) = tokio::sync::mpsc::channel(1);
        let mut request = NormalRequest::new_simple(
            RequestMessage::Completion {
                text: "hello".to_string(),
                echo_prompt: false,
                best_of: None,
            },
            SamplingParams::neutral(),
            response,
            0,
            None,
            None,
        );
        let generation = crate::AdapterGenerationId::from_bytes([7; 32]);
        request.adapter = Some(AdapterSelection::generation(generation));

        let serialized = serde_json::to_string(&Request::Normal(Box::new(request))).unwrap();
        let Request::Normal(request) = serde_json::from_str::<Request>(&serialized).unwrap() else {
            panic!("expected a normal request");
        };
        assert_eq!(
            request
                .adapter
                .as_ref()
                .and_then(AdapterSelection::resolved_generation),
            Some(generation)
        );
    }
}
