mod output;
mod protocol;
mod session;
pub mod tools;

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use mistralrs_mcp::{
    CalledFunction, ToolCallbackKind, ToolCallbackWithTool, ToolCallbacksWithTools, ToolFile,
    ToolOutput,
};
use mistralrs_sandbox::{Sandbox, SandboxPolicy};
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

/// Interval between idle-session reaper sweeps.
const REAP_INTERVAL: Duration = Duration::from_secs(300);
/// A code-exec session is reaped after this much inactivity.
const SESSION_TTL: Duration = Duration::from_secs(3600);

/// Python code execution config.
#[derive(Clone, Debug, Serialize, Deserialize)]
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
}

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

/// Ask the configured interpreter for `sys.prefix` and `sys.base_prefix` so
/// the sandbox can grant read access to the actual install path. Without
/// this, virtualenvs (`.venv/bin/python3`) fail at startup because the
/// interpreter cannot read its own stdlib. Best-effort: returns empty if
/// the probe fails or python prints nothing.
async fn resolve_python_prefixes(python_path: &Path) -> Vec<PathBuf> {
    // Ask Python where its install + user site live. Cover three cases:
    //   - sys.prefix:        venv root (`.venv/`) or system root (`/usr`).
    //   - sys.base_prefix:   underlying python install for venvs.
    //   - sys.executable:    needed when prefix doesn't already include the
    //                        interpreter binary (some custom installs).
    //   - site.getusersitepackages(): `~/.local/lib/python3.X/site-packages`
    //                        for `pip install --user` workflows.
    let out = tokio::process::Command::new(python_path)
        .args([
            "-c",
            "import sys, site; \
             print(sys.prefix); \
             print(sys.base_prefix); \
             print(sys.executable); \
             print(site.getusersitepackages())",
        ])
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
                // sys.executable is a file - take its install root.
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
        }
    }
}

/// Python code execution sessions keyed by client-provided ID. Different sessions run in parallel; same-session calls queue FIFO.
pub struct CodeExecutionManager {
    config: CodeExecutionConfig,
    sessions: Arc<Mutex<HashMap<String, Arc<Mutex<PythonSession>>>>>,
    /// Dedicated temp dir holding `executor.py`. Held to keep the file alive
    /// (the `TempDir` deletes its contents on drop); its path is in
    /// `sandbox_policy.extra_fs_read` so the sandboxed interpreter can read it.
    #[allow(dead_code)]
    executor_dir: Arc<tempfile::TempDir>,
    executor_script: PathBuf,
    installed_packages: String,
    sandbox: Arc<dyn Sandbox>,
    sandbox_policy: SandboxPolicy,
}

/// Invariant per-session state bundled together so it can be cloned cheaply
/// into the tool callback closures without holding `&self`. Clone is shallow
/// thanks to the `Arc` fields.
#[derive(Clone)]
struct SpawnCtx {
    python_path: PathBuf,
    executor_script: PathBuf,
    timeout: Duration,
    working_directory: Option<PathBuf>,
    sandbox: Arc<dyn Sandbox>,
    sandbox_policy: SandboxPolicy,
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

impl CodeExecutionManager {
    pub async fn new(config: CodeExecutionConfig) -> anyhow::Result<Self> {
        // Put executor.py in a dedicated tempdir so we can grant the
        // sandboxed interpreter read access to just this directory. A bare
        // tempfile lives in /tmp; allowing all of /tmp would be too broad.
        let executor_dir = Arc::new(
            tempfile::Builder::new()
                .prefix("mistralrs-executor-")
                .tempdir()?,
        );
        let executor_script = executor_dir.path().join("executor.py");
        std::fs::write(&executor_script, EXECUTOR_PY)?;

        // Validate python path.
        let output = tokio::process::Command::new(&config.python_path)
            .arg("--version")
            .output()
            .await
            .map_err(|e| {
                anyhow::anyhow!(
                    "Python interpreter not found at '{}': {e}",
                    config.python_path.display()
                )
            })?;
        if !output.status.success() {
            anyhow::bail!(
                "Python interpreter at '{}' returned non-zero status",
                config.python_path.display()
            );
        }

        // Capture installed packages.
        let installed_packages = {
            let output = tokio::process::Command::new(&config.python_path)
                .args(["-m", "pip", "list", "--format=freeze"])
                .output()
                .await;
            match output {
                Ok(o) if o.status.success() => String::from_utf8_lossy(&o.stdout).to_string(),
                _ => "(pip list unavailable)".to_string(),
            }
        };

        let sessions: Arc<Mutex<HashMap<String, Arc<Mutex<PythonSession>>>>> =
            Arc::new(Mutex::new(HashMap::new()));

        // Reap idle sessions. Sessions whose lock can't be acquired with `try_lock` are busy and skipped this round.
        let sessions_for_reaper = Arc::clone(&sessions);
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(REAP_INTERVAL).await;
                let mut map = sessions_for_reaper.lock().await;
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
                    tracing::info!(
                        "Reaped {reaped} idle code execution session(s) ({} remaining)",
                        map.len()
                    );
                }
            }
        });

        let (sandbox, mut sandbox_policy): (Arc<dyn Sandbox>, SandboxPolicy) =
            match config.sandbox_policy.clone() {
                Some(policy) => (Arc::from(mistralrs_sandbox::detect()), policy),
                None => (
                    Arc::from(mistralrs_sandbox::null()),
                    SandboxPolicy::default(),
                ),
            };

        // The interpreter has to read `executor.py` and its own stdlib /
        // site-packages. Without these, Landlock returns EACCES at startup
        // and Python dies with "Permission denied". Both paths are constant
        // for the lifetime of the manager.
        sandbox_policy
            .extra_fs_read
            .push(executor_dir.path().to_path_buf());
        for prefix in resolve_python_prefixes(&config.python_path).await {
            if !sandbox_policy.extra_fs_read.contains(&prefix) {
                sandbox_policy.extra_fs_read.push(prefix);
            }
        }

        tracing::info!(
            "code execution sandbox: {} (memory={}MB, cpu={}s, procs={}, network={:?}, strict={})",
            sandbox.name(),
            sandbox_policy.max_memory_mb,
            sandbox_policy.max_cpu_secs,
            sandbox_policy.max_procs,
            sandbox_policy.network,
            sandbox_policy.strict,
        );

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

    /// True if a real sandbox policy is in effect (`Some(policy)` was passed
    /// in `CodeExecutionConfig`). Used by the engine for accurate logging
    /// and tool-prompt wording.
    pub fn is_sandboxed(&self) -> bool {
        self.config.sandbox_policy.is_some()
    }

    /// Network mode the model is running under. Returns `None` when no
    /// sandbox policy was set (NullSandbox; no enforcement).
    pub fn network_mode(&self) -> Option<mistralrs_sandbox::NetworkMode> {
        self.config.sandbox_policy.as_ref().map(|p| p.network)
    }

    /// Build a [`SpawnCtx`] capturing all invariant per-session state. Cloned
    /// once per tool callback so closures don't borrow `&self`.
    fn spawn_ctx(&self) -> SpawnCtx {
        SpawnCtx {
            python_path: self.config.python_path.clone(),
            executor_script: self.executor_script.clone(),
            timeout: Duration::from_secs(self.config.timeout_secs),
            working_directory: self.config.working_directory.clone(),
            sandbox: Arc::clone(&self.sandbox),
            sandbox_policy: self.sandbox_policy.clone(),
        }
    }

    /// Tool callbacks to register with the engine. `input_modalities` tunes which capabilities the tool description advertises.
    pub fn get_tool_callbacks(&self, input_modalities: &[InputModality]) -> ToolCallbacksWithTools {
        let mut callbacks = ToolCallbacksWithTools::new();

        let execute_tool = tools::build_execute_python_tool(
            self.config.timeout_secs,
            &self.installed_packages,
            input_modalities,
            self.is_sandboxed(),
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
