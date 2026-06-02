mod output;
mod protocol;
mod session;
pub mod tools;

use std::collections::HashMap;
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use mistralrs_mcp::{
    CalledFunction, ToolCallbackKind, ToolCallbackWithTool, ToolCallbacksWithTools, ToolFile,
    ToolOutput,
};
use mistralrs_sandbox::{NetworkMode, Sandbox, SandboxPolicy};
use protocol::{ExecuteFile, ExecuteOutputSpec};
use serde::{Deserialize, Serialize};
use session::PythonSession;
use tokio::sync::Mutex;

pub use protocol::{ExecuteFile as CodeExecFile, ExecuteOutputSpec as CodeExecOutputSpec};
pub use tools::{
    build_list_files_tool, build_read_file_tool, code_exec_tool_called, EXECUTE_PYTHON_TOOL_NAME,
    LIST_FILES_TOOL_NAME, READ_FILE_TOOL_NAME, RESET_SESSION_TOOL_NAME,
};

/// Tailors the tool description to what the model can take as input.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InputModality {
    Text,
    Vision,
    Audio,
    Video,
}

const EXECUTOR_PY: &str = include_str!("../python/executor.py");

const REAP_INTERVAL: Duration = Duration::from_secs(300);
const SESSION_TTL: Duration = Duration::from_secs(3600);
const PYTHON_PREFIX_PROBE: &str = concat!(
    "import sys, site; ",
    "print(sys.prefix); ",
    "print(sys.base_prefix); ",
    "print(sys.executable); ",
    "print(site.getusersitepackages())",
);

/// Python code execution config.
#[derive(Clone, Serialize, Deserialize)]
pub struct CodeExecutionConfig {
    /// Defaults to `python3` (`python` on Windows).
    #[serde(default = "default_python_path")]
    pub python_path: PathBuf,
    /// Per-execution timeout. Defaults to 30s.
    #[serde(default = "default_timeout_secs")]
    pub timeout_secs: u64,
    /// If `None`, a temp dir is created. Otherwise this is the cwd for the model's code.
    #[serde(default)]
    pub working_directory: Option<PathBuf>,
    /// OS-level sandbox policy. `Some(policy)` enables the platform sandbox
    /// (Linux/macOS) with the given limits; `None` disables it entirely
    /// (NullSandbox). The CLI/server layer is responsible for choosing.
    #[serde(default)]
    pub sandbox_policy: Option<mistralrs_sandbox::SandboxPolicy>,
    #[serde(default)]
    pub permission: CodeExecutionPermission,
    #[serde(skip)]
    pub approval_callback: Option<Arc<CodeExecutionApprovalCallback>>,
}

impl fmt::Debug for CodeExecutionConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
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

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum CodeExecutionPermission {
    #[default]
    Auto,
    Ask,
    Deny,
}

impl CodeExecutionPermission {
    fn strictest(self, other: Self) -> Self {
        match (self, other) {
            (Self::Deny, _) | (_, Self::Deny) => Self::Deny,
            (Self::Ask, _) | (_, Self::Ask) => Self::Ask,
            (Self::Auto, Self::Auto) => Self::Auto,
        }
    }
}

impl From<mistralrs_mcp::CodeExecutionPermission> for CodeExecutionPermission {
    fn from(value: mistralrs_mcp::CodeExecutionPermission) -> Self {
        match value {
            mistralrs_mcp::CodeExecutionPermission::Auto => Self::Auto,
            mistralrs_mcp::CodeExecutionPermission::Ask => Self::Ask,
            mistralrs_mcp::CodeExecutionPermission::Deny => Self::Deny,
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
    dyn Fn(&CodeExecutionApproval) -> bool + Send + Sync + 'static;

fn default_python_path() -> PathBuf {
    if cfg!(windows) {
        PathBuf::from("python")
    } else {
        PathBuf::from("python3")
    }
}

fn default_timeout_secs() -> u64 {
    30
}

async fn resolve_python_prefixes(python_path: &Path) -> Vec<PathBuf> {
    let out = tokio::process::Command::new(python_path)
        .args(["-c", PYTHON_PREFIX_PROBE])
        .output()
        .await;
    let Ok(out) = out else { return Vec::new() };
    if !out.status.success() {
        return Vec::new();
    }
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .filter_map(|l| {
            let s = l.trim();
            if s.is_empty() {
                return None;
            }
            let p = PathBuf::from(s);
            let p = if p.is_file() {
                p.parent()?.parent()?.to_path_buf()
            } else {
                p
            };
            if p.exists() {
                Some(p)
            } else {
                None
            }
        })
        .collect()
}

impl Default for CodeExecutionConfig {
    fn default() -> Self {
        Self {
            python_path: default_python_path(),
            timeout_secs: default_timeout_secs(),
            working_directory: None,
            sandbox_policy: None,
            permission: CodeExecutionPermission::Auto,
            approval_callback: None,
        }
    }
}

pub struct CodeExecutionManager {
    config: CodeExecutionConfig,
    sessions: Arc<Mutex<HashMap<String, Arc<Mutex<PythonSession>>>>>,
    #[allow(dead_code)]
    executor_dir: Arc<tempfile::TempDir>,
    executor_script: PathBuf,
    installed_packages: String,
    sandbox: Arc<dyn Sandbox>,
    sandbox_policy: SandboxPolicy,
}

#[derive(Clone)]
struct SpawnCtx {
    #[allow(dead_code)]
    executor_dir: Arc<tempfile::TempDir>,
    python_path: PathBuf,
    executor_script: PathBuf,
    timeout: Duration,
    working_directory: Option<PathBuf>,
    sandbox: Arc<dyn Sandbox>,
    sandbox_policy: SandboxPolicy,
    permission: CodeExecutionPermission,
    approval_callback: Option<Arc<CodeExecutionApprovalCallback>>,
}

impl SpawnCtx {
    async fn session_handle(
        &self,
        sessions: &Mutex<HashMap<String, Arc<Mutex<PythonSession>>>>,
        session_id: &str,
    ) -> anyhow::Result<Arc<Mutex<PythonSession>>> {
        let mut map = sessions.lock().await;
        if let Some(existing) = map.get(session_id) {
            return Ok(Arc::clone(existing));
        }
        let session = PythonSession::new(
            &self.python_path,
            &self.executor_script,
            self.timeout,
            self.working_directory.as_deref(),
            Arc::clone(&self.sandbox),
            self.sandbox_policy.clone(),
        )
        .await?;
        let arc = Arc::new(Mutex::new(session));
        map.insert(session_id.to_string(), Arc::clone(&arc));
        Ok(arc)
    }
}

fn write_executor_script() -> anyhow::Result<(Arc<tempfile::TempDir>, PathBuf)> {
    let executor_dir = Arc::new(
        tempfile::Builder::new()
            .prefix("mistralrs-executor-")
            .tempdir()?,
    );
    let executor_script = executor_dir.path().join("executor.py");
    std::fs::write(&executor_script, EXECUTOR_PY)?;
    Ok((executor_dir, executor_script))
}

async fn validate_python(python_path: &Path) -> anyhow::Result<()> {
    let output = tokio::process::Command::new(python_path)
        .arg("--version")
        .output()
        .await
        .map_err(|e| {
            anyhow::anyhow!(
                "Python interpreter not found at '{}': {e}",
                python_path.display()
            )
        })?;
    if !output.status.success() {
        anyhow::bail!(
            "Python interpreter at '{}' returned non-zero status",
            python_path.display()
        );
    }
    Ok(())
}

async fn installed_packages(python_path: &Path) -> String {
    let output = tokio::process::Command::new(python_path)
        .args(["-m", "pip", "list", "--format=freeze"])
        .output()
        .await;
    match output {
        Ok(o) if o.status.success() => String::from_utf8_lossy(&o.stdout).to_string(),
        _ => "(pip list unavailable)".to_string(),
    }
}

fn spawn_reaper(sessions: Arc<Mutex<HashMap<String, Arc<Mutex<PythonSession>>>>>) {
    tokio::spawn(async move {
        loop {
            tokio::time::sleep(REAP_INTERVAL).await;
            let mut map = sessions.lock().await;
            let before = map.len();
            let mut to_remove = Vec::new();
            for (id, session_arc) in map.iter() {
                if let Ok(session) = session_arc.try_lock() {
                    if session.seconds_since_last_active() >= SESSION_TTL.as_secs() {
                        to_remove.push(id.clone());
                    }
                }
            }
            for id in &to_remove {
                map.remove(id);
            }
            let reaped = before - map.len();
            if reaped > 0 {
                tracing::debug!(
                    "Reaped {reaped} idle code execution session(s) ({} remaining)",
                    map.len()
                );
            }
        }
    });
}

async fn sandbox_for_config(
    config: &CodeExecutionConfig,
    executor_dir: &Path,
) -> anyhow::Result<(Arc<dyn Sandbox>, SandboxPolicy)> {
    let (sandbox, mut policy): (Arc<dyn Sandbox>, SandboxPolicy) =
        match config.sandbox_policy.clone() {
            Some(policy) => (Arc::from(mistralrs_sandbox::detect()), policy),
            None => (
                Arc::from(mistralrs_sandbox::null()),
                SandboxPolicy::default(),
            ),
        };

    policy.extra_fs_read.push(executor_dir.to_path_buf());
    for prefix in resolve_python_prefixes(&config.python_path).await {
        if !policy.extra_fs_read.contains(&prefix) {
            policy.extra_fs_read.push(prefix);
        }
    }

    validate_strict_policy(sandbox.as_ref(), &policy)?;
    Ok((sandbox, policy))
}

fn validate_strict_policy(sandbox: &dyn Sandbox, policy: &SandboxPolicy) -> anyhow::Result<()> {
    if !policy.strict {
        return Ok(());
    }

    let effective = sandbox.effective(policy);
    let mut missing = Vec::new();
    if !effective.rlimits_applied && sandbox.name() != "macos" {
        missing.push("rlimits");
    }
    if !effective.fs_isolated {
        missing.push("filesystem isolation");
    }
    if policy.network != NetworkMode::Full && !effective.network_isolated {
        missing.push("network isolation");
    }
    if missing.is_empty() {
        Ok(())
    } else {
        anyhow::bail!(
            "sandbox strict mode requested but {} unavailable for {} sandbox",
            missing.join(", "),
            sandbox.name()
        );
    }
}

impl CodeExecutionManager {
    pub async fn new(config: CodeExecutionConfig) -> anyhow::Result<Self> {
        let (executor_dir, executor_script) = write_executor_script()?;
        validate_python(&config.python_path).await?;
        let installed_packages = installed_packages(&config.python_path).await;

        let sessions: Arc<Mutex<HashMap<String, Arc<Mutex<PythonSession>>>>> =
            Arc::new(Mutex::new(HashMap::new()));
        spawn_reaper(Arc::clone(&sessions));

        let (sandbox, sandbox_policy) = sandbox_for_config(&config, executor_dir.path()).await?;
        let workdir = config
            .working_directory
            .as_ref()
            .map(|path| path.display().to_string())
            .unwrap_or_else(|| "per-session temp dir".to_string());

        if sandbox.name() == "macos" {
            tracing::info!(
                "code execution sandbox: {} (workdir={}, rlimits=not enforced, network={:?}, strict={})",
                sandbox.name(),
                workdir,
                sandbox_policy.network,
                sandbox_policy.strict,
            );
        } else {
            tracing::info!(
                "code execution sandbox: {} (workdir={}, memory={}MB, cpu={}s, procs={}, network={:?}, strict={})",
                sandbox.name(),
                workdir,
                sandbox_policy.max_memory_mb,
                sandbox_policy.max_cpu_secs,
                sandbox_policy.max_procs,
                sandbox_policy.network,
                sandbox_policy.strict,
            );
        }

        Ok(Self {
            config,
            sessions,
            executor_dir,
            executor_script,
            installed_packages,
            sandbox,
            sandbox_policy,
        })
    }

    pub fn is_sandboxed(&self) -> bool {
        self.effective_protection().any()
    }

    pub fn network_mode(&self) -> Option<mistralrs_sandbox::NetworkMode> {
        let policy = self.config.sandbox_policy.as_ref()?;
        if self.effective_protection().network_isolated {
            Some(policy.network)
        } else {
            None
        }
    }

    pub fn effective_protection(&self) -> mistralrs_sandbox::EffectiveProtection {
        self.sandbox.effective(&self.sandbox_policy)
    }

    fn spawn_ctx(&self) -> SpawnCtx {
        SpawnCtx {
            executor_dir: Arc::clone(&self.executor_dir),
            python_path: self.config.python_path.clone(),
            executor_script: self.executor_script.clone(),
            timeout: Duration::from_secs(self.config.timeout_secs),
            working_directory: self.config.working_directory.clone(),
            sandbox: Arc::clone(&self.sandbox),
            sandbox_policy: self.sandbox_policy.clone(),
            permission: self.config.permission,
            approval_callback: self.config.approval_callback.clone(),
        }
    }

    /// Tool callbacks to register with the engine. `input_modalities` tunes which capabilities the tool description advertises.
    pub fn get_tool_callbacks(&self, input_modalities: &[InputModality]) -> ToolCallbacksWithTools {
        let mut callbacks = ToolCallbacksWithTools::new();

        let execute_tool = tools::build_execute_python_tool(
            self.config.timeout_secs,
            &self.installed_packages,
            input_modalities,
            self.effective_protection(),
            self.network_mode(),
        );

        let reset_tool = tools::build_reset_session_tool();

        let sessions = Arc::clone(&self.sessions);
        let ctx = self.spawn_ctx();

        let execute_callback: Arc<mistralrs_mcp::MultimodalToolCallback> = Arc::new(
            move |func: &CalledFunction, tc: &mistralrs_mcp::ToolCallContext| {
                let sessions = Arc::clone(&sessions);
                let ctx = ctx.clone();

                let session_id = tc
                    .session_id
                    .clone()
                    .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());

                let args: serde_json::Value = serde_json::from_str(&func.arguments)?;
                let code = args
                    .get("code")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| anyhow::anyhow!("Missing 'code' argument"))?
                    .to_string();
                let outputs = args
                    .get("outputs")
                    .and_then(|v| v.as_array())
                    .map(|arr| parse_output_specs(arr))
                    .unwrap_or_default();

                let output_names: Vec<String> = outputs.iter().map(|o| o.name.clone()).collect();
                if let Some(text) =
                    denied_by_permission(&ctx, tc, &session_id, &code, &output_names)
                {
                    return Ok(ToolOutput::Multimodal {
                        text,
                        images: vec![],
                        video_frames: vec![],
                        files: vec![],
                    });
                }

                let handle = tokio::runtime::Handle::current();
                tokio::task::block_in_place(|| {
                    handle.block_on(async {
                        let session_arc = ctx.session_handle(&sessions, &session_id).await?;

                        let mut session = session_arc.lock().await;
                        let result = session.execute_with_outputs(&code, &outputs).await;
                        let files: Vec<ToolFile> =
                            result.files.iter().map(execute_file_to_tool_file).collect();

                        Ok(ToolOutput::Multimodal {
                            text: result.text,
                            images: result.images,
                            video_frames: result.video_frames,
                            files,
                        })
                    })
                })
            },
        );

        callbacks.insert(
            EXECUTE_PYTHON_TOOL_NAME.to_string(),
            ToolCallbackWithTool {
                callback: ToolCallbackKind::Multimodal(execute_callback),
                tool: execute_tool,
            },
        );

        let sessions = Arc::clone(&self.sessions);
        let ctx = self.spawn_ctx();

        let reset_callback: Arc<mistralrs_mcp::ToolCallback> = Arc::new(
            move |_func: &CalledFunction, tc: &mistralrs_mcp::ToolCallContext| {
                let sessions = Arc::clone(&sessions);
                let ctx = ctx.clone();

                let session_id = tc
                    .session_id
                    .clone()
                    .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());

                if let Some(text) =
                    denied_by_permission(&ctx, tc, &session_id, "reset Python session", &[])
                {
                    return Ok(text);
                }

                let handle = tokio::runtime::Handle::current();
                tokio::task::block_in_place(|| {
                    handle.block_on(async {
                        let session_arc = ctx.session_handle(&sessions, &session_id).await?;

                        let mut session = session_arc.lock().await;
                        session.reset().await?;

                        Ok(serde_json::json!({"status": "success", "message": "Session reset. All variables and imports have been cleared."}).to_string())
                    })
                })
            },
        );

        callbacks.insert(
            RESET_SESSION_TOOL_NAME.to_string(),
            ToolCallbackWithTool {
                callback: ToolCallbackKind::Text(reset_callback),
                tool: reset_tool,
            },
        );

        // Schemas only. The engine owns the FileStore and dispatches these directly.
        let read_file_stub: Arc<mistralrs_mcp::ToolCallback> = Arc::new(|_, _| {
            Err(anyhow::anyhow!(
                "read_file must be dispatched by the engine"
            ))
        });
        callbacks.insert(
            READ_FILE_TOOL_NAME.to_string(),
            ToolCallbackWithTool {
                callback: ToolCallbackKind::Text(read_file_stub),
                tool: tools::build_read_file_tool(),
            },
        );

        let list_files_stub: Arc<mistralrs_mcp::ToolCallback> = Arc::new(|_, _| {
            Err(anyhow::anyhow!(
                "list_files must be dispatched by the engine"
            ))
        });
        callbacks.insert(
            LIST_FILES_TOOL_NAME.to_string(),
            ToolCallbackWithTool {
                callback: ToolCallbackKind::Text(list_files_stub),
                tool: tools::build_list_files_tool(),
            },
        );

        callbacks
    }
}

fn denied_by_permission(
    ctx: &SpawnCtx,
    tool_ctx: &mistralrs_mcp::ToolCallContext,
    session_id: &str,
    code: &str,
    outputs: &[String],
) -> Option<String> {
    let permission = tool_ctx
        .code_execution_permission
        .map(Into::into)
        .map(|request_permission| ctx.permission.strictest(request_permission))
        .unwrap_or(ctx.permission);

    let reason = match permission {
        CodeExecutionPermission::Auto => return None,
        CodeExecutionPermission::Deny => "Code execution was denied by policy.",
        CodeExecutionPermission::Ask => {
            let Some(callback) = &ctx.approval_callback else {
                return Some(code_execution_denied_text(
                    "Code execution requires approval, but no approval handler is configured.",
                ));
            };
            let approval_id = format!("appr_{}", uuid::Uuid::new_v4().simple());
            let approval = CodeExecutionApproval {
                approval_id: approval_id.clone(),
                session_id: session_id.to_string(),
                code: code.to_string(),
                outputs: outputs.to_vec(),
                working_directory: ctx.working_directory.clone(),
            };
            if let Some(notifier) = &tool_ctx.code_execution_approval_notifier {
                notifier(mistralrs_mcp::CodeExecutionApprovalRequest {
                    approval_id,
                    session_id: approval.session_id.clone(),
                    round: tool_ctx.round.unwrap_or_default(),
                    tool_name: tool_ctx
                        .tool_name
                        .clone()
                        .unwrap_or_else(|| EXECUTE_PYTHON_TOOL_NAME.to_string()),
                    code: approval.code.clone(),
                    outputs: approval.outputs.clone(),
                    working_directory: approval.working_directory.clone(),
                });
            }
            if callback(&approval) {
                return None;
            }
            "Code execution was denied by the user."
        }
    };

    Some(code_execution_denied_text(reason))
}

fn code_execution_denied_text(reason: &str) -> String {
    serde_json::json!({
        "status": "denied",
        "exception": reason,
    })
    .to_string()
}

fn parse_output_specs(arr: &[serde_json::Value]) -> Vec<ExecuteOutputSpec> {
    arr.iter()
        .filter_map(|v| {
            let name = v.get("name")?.as_str()?.to_string();
            let format = v
                .get("format")
                .and_then(|f| f.as_str())
                .map(|s| s.to_string());
            Some(ExecuteOutputSpec { name, format })
        })
        .collect()
}

fn execute_file_to_tool_file(f: &ExecuteFile) -> ToolFile {
    ToolFile {
        name: f.name.clone(),
        format: f.format.clone(),
        mime_type: f.mime_type.clone(),
        text: f.text.clone(),
        data_base64: f.data_base64.clone(),
        size_bytes: f.size_bytes,
        error: f.error.clone(),
    }
}
