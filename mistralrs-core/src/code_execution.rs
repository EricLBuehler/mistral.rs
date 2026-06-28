use std::{future::Future, path::PathBuf, pin::Pin, sync::Arc};

use mistralrs_mcp::{AgentPermission, AgentToolMetadata, CodeExecutionPermission};
use mistralrs_sandbox::SandboxPolicy;

pub const DEFAULT_CODE_EXEC_TIMEOUT_SECS: u64 = 60;
pub const DEFAULT_SHELL_TIMEOUT_SECS: u64 = 600;

#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct CodeExecutionConfig {
    #[serde(default = "default_python_path")]
    pub python_path: PathBuf,
    #[serde(default = "default_code_exec_timeout_secs")]
    pub timeout_secs: u64,
    #[serde(default)]
    pub working_directory: Option<PathBuf>,
    #[serde(default)]
    pub sandbox_policy: Option<SandboxPolicy>,
    #[serde(default)]
    pub permission: CodeExecutionPermission,
    #[serde(skip)]
    pub approval_callback: Option<CodeExecutionApprovalCallback>,
}

impl std::fmt::Debug for CodeExecutionConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CodeExecutionConfig")
            .field("python_path", &self.python_path)
            .field("timeout_secs", &self.timeout_secs)
            .field("working_directory", &self.working_directory)
            .field("sandbox_policy", &self.sandbox_policy)
            .field("permission", &self.permission)
            .field("approval_callback", &self.approval_callback.is_some())
            .finish()
    }
}

impl Default for CodeExecutionConfig {
    fn default() -> Self {
        Self {
            python_path: default_python_path(),
            timeout_secs: default_code_exec_timeout_secs(),
            working_directory: None,
            sandbox_policy: None,
            permission: CodeExecutionPermission::Auto,
            approval_callback: None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct CodeExecutionApproval {
    pub approval_id: String,
    pub session_id: String,
    pub code: String,
    pub outputs: Vec<String>,
    pub working_directory: Option<PathBuf>,
}

pub type CodeExecutionApprovalCallback =
    Arc<dyn Fn(&CodeExecutionApproval) -> bool + Send + Sync + 'static>;

#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct ShellConfig {
    #[serde(default = "default_shell_path")]
    pub shell_path: PathBuf,
    #[serde(default = "default_shell_timeout_secs")]
    pub timeout_secs: u64,
    #[serde(default)]
    pub working_directory: Option<PathBuf>,
    #[serde(default)]
    pub sandbox_policy: Option<SandboxPolicy>,
    #[serde(default)]
    pub permission: AgentPermission,
}

impl std::fmt::Debug for ShellConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ShellConfig")
            .field("shell_path", &self.shell_path)
            .field("timeout_secs", &self.timeout_secs)
            .field("working_directory", &self.working_directory)
            .field("sandbox_policy", &self.sandbox_policy)
            .field("permission", &self.permission)
            .finish()
    }
}

impl Default for ShellConfig {
    fn default() -> Self {
        Self {
            shell_path: default_shell_path(),
            timeout_secs: default_shell_timeout_secs(),
            working_directory: None,
            sandbox_policy: None,
            permission: AgentPermission::Auto,
        }
    }
}

#[derive(Clone, Debug)]
pub struct AgentToolApproval {
    pub approval_id: String,
    pub session_id: String,
    pub round: usize,
    pub tool: AgentToolMetadata,
    pub arguments: serde_json::Value,
}

#[derive(Clone, Debug)]
pub struct AgentToolApprovalDecision {
    pub approve: bool,
    pub remember_for_session: bool,
    pub message: Option<String>,
}

impl AgentToolApprovalDecision {
    pub fn approve() -> Self {
        Self {
            approve: true,
            remember_for_session: false,
            message: None,
        }
    }

    pub fn approve_for_session() -> Self {
        Self {
            approve: true,
            remember_for_session: true,
            message: None,
        }
    }

    pub fn deny(message: Option<String>) -> Self {
        Self {
            approve: false,
            remember_for_session: false,
            message,
        }
    }

    pub fn deny_with_message(message: impl Into<String>) -> Self {
        Self {
            approve: false,
            remember_for_session: false,
            message: Some(message.into()),
        }
    }

    pub fn with_remember_for_session(mut self, remember_for_session: bool) -> Self {
        self.remember_for_session = remember_for_session;
        self
    }
}

pub type AgentToolApprovalCallback =
    Arc<dyn Fn(&AgentToolApproval) -> AgentToolApprovalDecision + Send + Sync + 'static>;

pub type AgentToolApprovalFuture =
    Pin<Box<dyn Future<Output = AgentToolApprovalDecision> + Send + 'static>>;

pub type AgentToolApprovalAsyncCallback =
    Arc<dyn Fn(AgentToolApproval) -> AgentToolApprovalFuture + Send + Sync + 'static>;

#[derive(Clone)]
pub enum AgentToolApprovalHandler {
    Sync(AgentToolApprovalCallback),
    Async(AgentToolApprovalAsyncCallback),
}

impl AgentToolApprovalHandler {
    pub fn from_sync(callback: AgentToolApprovalCallback) -> Self {
        Self::Sync(callback)
    }

    pub fn from_async(callback: AgentToolApprovalAsyncCallback) -> Self {
        Self::Async(callback)
    }
}

fn default_python_path() -> PathBuf {
    if cfg!(windows) {
        PathBuf::from("python")
    } else {
        PathBuf::from("python3")
    }
}

fn default_code_exec_timeout_secs() -> u64 {
    DEFAULT_CODE_EXEC_TIMEOUT_SECS
}

fn default_shell_path() -> PathBuf {
    if cfg!(windows) {
        PathBuf::from("cmd")
    } else {
        PathBuf::from("/bin/sh")
    }
}

fn default_shell_timeout_secs() -> u64 {
    DEFAULT_SHELL_TIMEOUT_SECS
}
