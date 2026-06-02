use image::DynamicImage;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::fmt;
use std::path::PathBuf;
use std::sync::Arc;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AgentToolSource {
    BuiltIn,
    User,
    Mcp,
    External,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AgentToolKind {
    CodeExecution,
    WebSearch,
    File,
    Custom,
    External,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AgentToolMetadata {
    pub source: AgentToolSource,
    pub kind: AgentToolKind,
    pub label: String,
}

#[derive(Clone, Debug)]
pub struct AgentToolApprovalRequest {
    pub approval_id: String,
    pub session_id: String,
    pub round: usize,
    pub tool: AgentToolMetadata,
    pub arguments: Value,
}

pub type AgentToolApprovalNotifier = dyn Fn(AgentToolApprovalRequest) + Send + Sync + 'static;

#[derive(Clone, Debug)]
pub struct CodeExecutionApprovalRequest {
    pub approval_id: String,
    pub session_id: String,
    pub round: usize,
    pub tool_name: String,
    pub code: String,
    pub outputs: Vec<String>,
    pub working_directory: Option<PathBuf>,
}

pub type CodeExecutionApprovalNotifier =
    dyn Fn(CodeExecutionApprovalRequest) + Send + Sync + 'static;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum AgentPermission {
    #[default]
    Auto,
    Ask,
    Deny,
}

impl AgentPermission {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Ask => "ask",
            Self::Deny => "deny",
        }
    }

    pub fn strictest(self, other: Self) -> Self {
        match (self, other) {
            (Self::Deny, _) | (_, Self::Deny) => Self::Deny,
            (Self::Ask, _) | (_, Self::Ask) => Self::Ask,
            (Self::Auto, Self::Auto) => Self::Auto,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum CodeExecutionPermission {
    #[default]
    Auto,
    Ask,
    Deny,
}

impl CodeExecutionPermission {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Ask => "ask",
            Self::Deny => "deny",
        }
    }

    pub fn strictest(self, other: Self) -> Self {
        match (self, other) {
            (Self::Deny, _) | (_, Self::Deny) => Self::Deny,
            (Self::Ask, _) | (_, Self::Ask) => Self::Ask,
            (Self::Auto, Self::Auto) => Self::Auto,
        }
    }
}

impl From<CodeExecutionPermission> for AgentPermission {
    fn from(value: CodeExecutionPermission) -> Self {
        match value {
            CodeExecutionPermission::Auto => Self::Auto,
            CodeExecutionPermission::Ask => Self::Ask,
            CodeExecutionPermission::Deny => Self::Deny,
        }
    }
}

impl From<AgentPermission> for CodeExecutionPermission {
    fn from(value: AgentPermission) -> Self {
        match value {
            AgentPermission::Auto => Self::Auto,
            AgentPermission::Ask => Self::Ask,
            AgentPermission::Deny => Self::Deny,
        }
    }
}

/// Context provided to tool callbacks by the agentic loop.
#[derive(Clone, Default)]
pub struct ToolCallContext {
    /// Use to key per-session state across invocations.
    pub session_id: Option<String>,
    pub round: Option<usize>,
    pub tool_name: Option<String>,
    pub agent_permission: Option<AgentPermission>,
    pub agent_approval_notifier: Option<Arc<AgentToolApprovalNotifier>>,
    pub code_execution_permission: Option<CodeExecutionPermission>,
    pub code_execution_approval_notifier: Option<Arc<CodeExecutionApprovalNotifier>>,
}

impl fmt::Debug for ToolCallContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ToolCallContext")
            .field("session_id", &self.session_id)
            .field("round", &self.round)
            .field("tool_name", &self.tool_name)
            .field("agent_permission", &self.agent_permission)
            .field(
                "agent_approval_notifier",
                &self.agent_approval_notifier.is_some(),
            )
            .field("code_execution_permission", &self.code_execution_permission)
            .field(
                "code_execution_approval_notifier",
                &self.code_execution_approval_notifier.is_some(),
            )
            .finish()
    }
}

/// Custom tool callback. Receives the called function and returns the tool output as a string.
pub type ToolCallback =
    dyn Fn(&CalledFunction, &ToolCallContext) -> anyhow::Result<String> + Send + Sync;

/// Callback that can return multimodal output (text + images).
pub type MultimodalToolCallback =
    dyn Fn(&CalledFunction, &ToolCallContext) -> anyhow::Result<ToolOutput> + Send + Sync;

/// A file produced by a tool, carried out of band from the text response. The engine converts it to a typed `File`.
#[derive(Debug, Clone)]
pub struct ToolFile {
    pub name: String,
    pub format: String,
    pub mime_type: Option<String>,
    /// Set for utf-8 readable files.
    pub text: Option<String>,
    /// Set for binary files.
    pub data_base64: Option<String>,
    pub size_bytes: u64,
    /// Set when the file was requested but not produced or failed to read.
    pub error: Option<String>,
}

impl ToolFile {
    pub fn is_text(&self) -> bool {
        self.text.is_some()
    }
    pub fn is_error(&self) -> bool {
        self.error.is_some()
    }
}

/// Tool output: text-only or multimodal.
pub enum ToolOutput {
    Text(String),
    Multimodal {
        text: String,
        images: Vec<DynamicImage>,
        /// Ordered. The caller assembles these (e.g. into a `VideoInput`).
        video_frames: Vec<DynamicImage>,
        /// Surfaced as typed `File`s in the chat response.
        files: Vec<ToolFile>,
    },
}

impl From<String> for ToolOutput {
    fn from(s: String) -> Self {
        ToolOutput::Text(s)
    }
}

impl ToolOutput {
    pub fn text(&self) -> &str {
        match self {
            ToolOutput::Text(s) => s,
            ToolOutput::Multimodal { text, .. } => text,
        }
    }

    pub fn images(&self) -> &[DynamicImage] {
        match self {
            ToolOutput::Text(_) => &[],
            ToolOutput::Multimodal { images, .. } => images,
        }
    }

    pub fn video_frames(&self) -> &[DynamicImage] {
        match self {
            ToolOutput::Text(_) => &[],
            ToolOutput::Multimodal { video_frames, .. } => video_frames,
        }
    }

    pub fn files(&self) -> &[ToolFile] {
        match self {
            ToolOutput::Text(_) => &[],
            ToolOutput::Multimodal { files, .. } => files,
        }
    }

    pub fn has_multimodal(&self) -> bool {
        match self {
            ToolOutput::Text(_) => false,
            ToolOutput::Multimodal {
                images,
                video_frames,
                ..
            } => !images.is_empty() || !video_frames.is_empty(),
        }
    }
}

/// Wraps either a text-only or multimodal tool callback.
pub enum ToolCallbackKind {
    /// Legacy text-only callback returning a String.
    Text(Arc<ToolCallback>),
    /// Multimodal callback that may return images alongside text.
    Multimodal(Arc<MultimodalToolCallback>),
}

impl Clone for ToolCallbackKind {
    fn clone(&self) -> Self {
        match self {
            ToolCallbackKind::Text(cb) => ToolCallbackKind::Text(Arc::clone(cb)),
            ToolCallbackKind::Multimodal(cb) => ToolCallbackKind::Multimodal(Arc::clone(cb)),
        }
    }
}

/// A tool callback with its associated Tool definition.
#[derive(Clone)]
pub struct ToolCallbackWithTool {
    pub callback: ToolCallbackKind,
    pub tool: Tool,
}

/// Collection of callbacks keyed by tool name.
pub type ToolCallbacks = HashMap<String, Arc<ToolCallback>>;

/// Collection of callbacks with their tool definitions keyed by tool name.
pub type ToolCallbacksWithTools = HashMap<String, ToolCallbackWithTool>;

/// Type of tool
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
#[derive(Clone, Debug, Deserialize, Serialize)]
pub enum ToolType {
    #[serde(rename = "function")]
    Function,
}

/// Function definition for a tool
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Function {
    pub description: Option<String>,
    pub name: String,
    #[serde(alias = "arguments")]
    pub parameters: Option<HashMap<String, Value>>,
    /// When `true`, the tool's `parameters` JSON schema is enforced on the
    /// generated arguments via constrained decoding (llguidance).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>,
}

impl Function {
    /// Returns the parameters as a JSON Schema [`Value`] when strict mode is
    /// enabled.  Returns `None` when strict is absent or `false`.
    pub fn strict_parameters_schema(&self) -> Option<Value> {
        if self.strict != Some(true) {
            return None;
        }
        match &self.parameters {
            Some(p) => match serde_json::to_value(p) {
                Ok(v) => Some(v),
                Err(e) => {
                    tracing::warn!(
                        "Failed to serialize parameters for strict tool `{}`: {e}. \
                         Falling back to generic object schema.",
                        self.name,
                    );
                    Some(json!({"type": "object"}))
                }
            },
            None => {
                tracing::warn!(
                    "Tool `{}` has strict: true but no parameters schema defined. \
                     Cannot enforce strict mode; falling back to generic object schema.",
                    self.name,
                );
                Some(json!({"type": "object"}))
            }
        }
    }
}

/// Tool definition
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Tool {
    #[serde(rename = "type")]
    pub tp: ToolType,
    pub function: Function,
}

/// Called function with name and arguments
#[cfg_attr(feature = "pyo3_macros", pyo3::pyclass)]
#[cfg_attr(feature = "pyo3_macros", pyo3(get_all))]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CalledFunction {
    pub name: String,
    pub arguments: String,
}
