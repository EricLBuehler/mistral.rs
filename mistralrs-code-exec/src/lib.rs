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

impl Default for CodeExecutionConfig {
    fn default() -> Self {
        Self {
            python_path: default_python_path(),
            timeout_secs: default_timeout_secs(),
            working_directory: None,
        }
    }
}

/// Python code execution sessions keyed by client-provided ID. Different sessions run in parallel; same-session calls queue FIFO.
pub struct CodeExecutionManager {
    config: CodeExecutionConfig,
    sessions: Arc<Mutex<HashMap<String, Arc<Mutex<PythonSession>>>>>,
    executor_script: Arc<tempfile::NamedTempFile>,
    installed_packages: String,
}

impl CodeExecutionManager {
    pub async fn new(config: CodeExecutionConfig) -> anyhow::Result<Self> {
        // Write executor.py to a temp file held by the manager. Cleaned when the last `Arc` drops.
        let executor_script = {
            use std::io::Write;
            let mut f = tempfile::Builder::new().suffix(".py").tempfile()?;
            f.write_all(EXECUTOR_PY.as_bytes())?;
            f.flush()?;
            Arc::new(f)
        };

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

        Ok(Self {
            config,
            sessions,
            executor_script,
            installed_packages,
        })
    }

    /// Get or create a session. Outer map lock is released immediately so cross-session calls run in parallel.
    async fn session_handle(
        sessions: &Mutex<HashMap<String, Arc<Mutex<PythonSession>>>>,
        session_id: &str,
        python_path: &Path,
        executor_script: &Path,
        timeout: Duration,
        working_directory: Option<&Path>,
    ) -> anyhow::Result<Arc<Mutex<PythonSession>>> {
        let mut map = sessions.lock().await;
        if let Some(existing) = map.get(session_id) {
            return Ok(Arc::clone(existing));
        }
        let session =
            PythonSession::new(python_path, executor_script, timeout, working_directory).await?;
        let arc = Arc::new(Mutex::new(session));
        map.insert(session_id.to_string(), Arc::clone(&arc));
        Ok(arc)
    }

    /// Tool callbacks to register with the engine. `input_modalities` tunes which capabilities the tool description advertises.
    pub fn get_tool_callbacks(&self, input_modalities: &[InputModality]) -> ToolCallbacksWithTools {
        let mut callbacks = ToolCallbacksWithTools::new();

        let execute_tool = tools::build_execute_python_tool(
            self.config.timeout_secs,
            &self.installed_packages,
            input_modalities,
        );

        let reset_tool = tools::build_reset_session_tool();

        let sessions = Arc::clone(&self.sessions);
        let python_path = self.config.python_path.clone();
        let executor_script = Arc::clone(&self.executor_script);
        let timeout = Duration::from_secs(self.config.timeout_secs);
        let working_directory = self.config.working_directory.clone();

        let execute_callback: Arc<mistralrs_mcp::MultimodalToolCallback> = Arc::new(
            move |func: &CalledFunction, ctx: &mistralrs_mcp::ToolCallContext| {
                let sessions = Arc::clone(&sessions);
                let python_path = python_path.clone();
                let executor_script = Arc::clone(&executor_script);
                let working_directory = working_directory.clone();

                let session_id = ctx
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

                // Models sometimes emit LaTeX operators instead of Python ones.
                let code = sanitize_latex_operators(&code);

                let handle = tokio::runtime::Handle::current();
                tokio::task::block_in_place(|| {
                    handle.block_on(async {
                        let session_arc = Self::session_handle(
                            &sessions,
                            &session_id,
                            &python_path,
                            executor_script.path(),
                            timeout,
                            working_directory.as_deref(),
                        )
                        .await?;

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
        let python_path_reset = self.config.python_path.clone();
        let executor_script_reset = Arc::clone(&self.executor_script);
        let working_directory_reset = self.config.working_directory.clone();

        let reset_callback: Arc<mistralrs_mcp::ToolCallback> = Arc::new(
            move |_func: &CalledFunction, ctx: &mistralrs_mcp::ToolCallContext| {
                let sessions = Arc::clone(&sessions);
                let python_path = python_path_reset.clone();
                let executor_script = Arc::clone(&executor_script_reset);
                let working_directory = working_directory_reset.clone();

                let session_id = ctx
                    .session_id
                    .clone()
                    .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());

                let handle = tokio::runtime::Handle::current();
                tokio::task::block_in_place(|| {
                    handle.block_on(async {
                        let session_arc = Self::session_handle(
                            &sessions,
                            &session_id,
                            &python_path,
                            executor_script.path(),
                            timeout,
                            working_directory.as_deref(),
                        )
                        .await?;

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

/// Replace bare LaTeX math operators with Python equivalents. Skips matches followed by an ASCII letter (`\left`, `\length`).
fn sanitize_latex_operators(code: &str) -> String {
    let mut result = code.to_string();
    result = replace_latex_op(&result, "\\le", "<=");
    result = replace_latex_op(&result, "\\ge", ">=");
    result = replace_latex_op(&result, "\\ne", "!=");
    result = replace_latex_op(&result, "\\times", "*");
    result = replace_latex_op(&result, "\\cdot", "*");
    result
}

/// Replace `pattern` with `replacement` only when not followed by an ASCII letter.
fn replace_latex_op(haystack: &str, pattern: &str, replacement: &str) -> String {
    let mut result = String::with_capacity(haystack.len());
    let mut remaining = haystack;
    while let Some(pos) = remaining.find(pattern) {
        result.push_str(&remaining[..pos]);
        let after = pos + pattern.len();
        let next_char = remaining[after..].chars().next();
        if next_char.is_some_and(|c| c.is_ascii_alphabetic()) {
            // Part of a longer word like \left, \length. Leave it.
            result.push_str(pattern);
        } else {
            result.push_str(replacement);
        }
        remaining = &remaining[after..];
    }
    result.push_str(remaining);
    result
}
