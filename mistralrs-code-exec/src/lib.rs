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

/// Input modalities the model supports, used to tailor the tool description.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InputModality {
    Text,
    Vision,
    Audio,
    Video,
}

const EXECUTOR_PY: &str = include_str!("../python/executor.py");

/// Configuration for Python code execution.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CodeExecutionConfig {
    /// Path to the Python interpreter. Defaults to `"python3"`.
    #[serde(default = "default_python_path")]
    pub python_path: PathBuf,
    /// Execution timeout in seconds. Default: 30.
    #[serde(default = "default_timeout_secs")]
    pub timeout_secs: u64,
    /// Working directory for code execution. If `None`, a temporary directory
    /// is created (with prefix `mistralrs-code-`). If set, the model's code
    /// runs in this directory and files are saved there.
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

/// Manages Python code execution sessions.
///
/// Sessions are keyed by a client-provided session ID (set via the
/// `CURRENT_SESSION_ID` thread-local). When the same session ID is reused
/// across requests, the Python subprocess and its state persist.
/// Sessions idle for >10 minutes are automatically reaped.
pub struct CodeExecutionManager {
    config: CodeExecutionConfig,
    sessions: Arc<Mutex<HashMap<String, PythonSession>>>,
    /// Path to the persisted executor script. Kept as PathBuf after
    /// `NamedTempFile::keep()` so the file survives the manager's lifetime.
    executor_script_path: std::path::PathBuf,
    installed_packages: String,
}

impl CodeExecutionManager {
    pub async fn new(config: CodeExecutionConfig) -> anyhow::Result<Self> {
        // Write executor.py to a temp file and persist it (so it survives
        // after the manager is dropped — callbacks hold the path).
        let executor_script_path = {
            use std::io::Write;
            let mut f = tempfile::Builder::new().suffix(".py").tempfile()?;
            f.write_all(EXECUTOR_PY.as_bytes())?;
            f.flush()?;
            let (_, path) = f
                .keep()
                .map_err(|e| anyhow::anyhow!("Failed to persist executor script: {e}"))?;
            path
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

        let sessions: Arc<Mutex<HashMap<String, PythonSession>>> =
            Arc::new(Mutex::new(HashMap::new()));

        // Reap sessions idle for >1 hour. Runs every 5 minutes.
        let sessions_for_reaper = Arc::clone(&sessions);
        tokio::spawn(async move {
            const REAP_INTERVAL_SECS: u64 = 300;
            const SESSION_TTL_SECS: u64 = 3600;
            loop {
                tokio::time::sleep(Duration::from_secs(REAP_INTERVAL_SECS)).await;
                let mut map = sessions_for_reaper.lock().await;
                let before = map.len();
                map.retain(|_id, session| session.seconds_since_last_active() < SESSION_TTL_SECS);
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
            executor_script_path,
            installed_packages,
        })
    }

    /// Get or create a session for the given session ID.
    async fn get_or_create_session(
        sessions: &Mutex<HashMap<String, PythonSession>>,
        session_id: &str,
        python_path: &Path,
        executor_script: &Path,
        timeout: Duration,
        working_directory: Option<&Path>,
    ) -> anyhow::Result<()> {
        let mut map = sessions.lock().await;
        if !map.contains_key(session_id) {
            let session =
                PythonSession::new(python_path, executor_script, timeout, working_directory)
                    .await?;
            map.insert(session_id.to_string(), session);
        }
        Ok(())
    }

    /// Build tool callbacks for registration with the engine.
    ///
    /// `input_modalities` controls which capabilities are advertised in the
    /// tool description (e.g. image/video feedback is only mentioned when the
    /// model supports the corresponding input modality).
    pub fn get_tool_callbacks(&self, input_modalities: &[InputModality]) -> ToolCallbacksWithTools {
        let mut callbacks = ToolCallbacksWithTools::new();

        let execute_tool = tools::build_execute_python_tool(
            self.config.timeout_secs,
            &self.installed_packages,
            input_modalities,
        );

        let reset_tool = tools::build_reset_session_tool();

        // execute_python callback
        let sessions = Arc::clone(&self.sessions);
        let python_path = self.config.python_path.clone();
        let executor_script = self.executor_script_path.clone();
        let timeout = Duration::from_secs(self.config.timeout_secs);
        let working_directory = self.config.working_directory.clone();

        let execute_callback: Arc<mistralrs_mcp::MultimodalToolCallback> = Arc::new(
            move |func: &CalledFunction, ctx: &mistralrs_mcp::ToolCallContext| {
                let sessions = Arc::clone(&sessions);
                let python_path = python_path.clone();
                let executor_script = executor_script.clone();
                let working_directory = working_directory.clone();

                // Get session ID from context.
                let session_id = ctx
                    .session_id
                    .clone()
                    .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());

                // Parse the code + outputs args.
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

                // Run async code execution in the current runtime.
                let handle = tokio::runtime::Handle::current();
                tokio::task::block_in_place(|| {
                    handle.block_on(async {
                        Self::get_or_create_session(
                            &sessions,
                            &session_id,
                            &python_path,
                            &executor_script,
                            timeout,
                            working_directory.as_deref(),
                        )
                        .await?;

                        let mut map = sessions.lock().await;
                        let session = map.get_mut(&session_id).unwrap();
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

        // reset_python_session callback
        let sessions = Arc::clone(&self.sessions);
        let python_path_reset = self.config.python_path.clone();
        let executor_script_reset = self.executor_script_path.clone();
        let working_directory_reset = self.config.working_directory.clone();

        let reset_callback: Arc<mistralrs_mcp::ToolCallback> = Arc::new(
            move |_func: &CalledFunction, ctx: &mistralrs_mcp::ToolCallContext| {
                let sessions = Arc::clone(&sessions);
                let python_path = python_path_reset.clone();
                let executor_script = executor_script_reset.clone();
                let working_directory = working_directory_reset.clone();

                let session_id = ctx
                    .session_id
                    .clone()
                    .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());

                let handle = tokio::runtime::Handle::current();
                tokio::task::block_in_place(|| {
                    handle.block_on(async {
                        Self::get_or_create_session(
                            &sessions,
                            &session_id,
                            &python_path,
                            &executor_script,
                            timeout,
                            working_directory.as_deref(),
                        )
                        .await?;

                        let mut map = sessions.lock().await;
                        let session = map.get_mut(&session_id).unwrap();
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

        // read_file and list_files: schemas registered here so the model
        // sees them. Dispatch happens in the engine (it owns the
        // FileStore).
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

/// Replace common LaTeX math operators that models sometimes emit with
/// their Python equivalents so the code actually runs.
///
/// Only replaces when the LaTeX command is NOT followed by an ASCII letter,
/// so `\left` / `\length` / `\next` etc. are left intact.
fn sanitize_latex_operators(code: &str) -> String {
    let mut result = code.to_string();
    result = replace_latex_op(&result, "\\le", "<=");
    result = replace_latex_op(&result, "\\ge", ">=");
    result = replace_latex_op(&result, "\\ne", "!=");
    result = replace_latex_op(&result, "\\times", "*");
    result = replace_latex_op(&result, "\\cdot", "*");
    result
}

/// Replace `pattern` with `replacement` only when the character immediately
/// after `pattern` is not an ASCII letter (i.e. the command is complete).
fn replace_latex_op(haystack: &str, pattern: &str, replacement: &str) -> String {
    let mut result = String::with_capacity(haystack.len());
    let mut remaining = haystack;
    while let Some(pos) = remaining.find(pattern) {
        result.push_str(&remaining[..pos]);
        let after = pos + pattern.len();
        let next_char = remaining[after..].chars().next();
        if next_char.is_some_and(|c| c.is_ascii_alphabetic()) {
            // Part of a longer word like \left, \length — keep as-is.
            result.push_str(pattern);
        } else {
            result.push_str(replacement);
        }
        remaining = &remaining[after..];
    }
    result.push_str(remaining);
    result
}
