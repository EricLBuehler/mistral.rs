mod output;
mod protocol;
mod session;
pub mod tools;

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::thread::ThreadId;
use std::time::Duration;

use mistralrs_mcp::{
    CalledFunction, ToolCallbackKind, ToolCallbackWithTool, ToolCallbacksWithTools, ToolOutput,
};
use serde::{Deserialize, Serialize};
use session::PythonSession;
use tokio::sync::Mutex;

pub use tools::{code_exec_tool_called, EXECUTE_PYTHON_TOOL_NAME, RESET_SESSION_TOOL_NAME};

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
}

fn default_python_path() -> PathBuf {
    PathBuf::from("python3")
}

fn default_timeout_secs() -> u64 {
    30
}

impl Default for CodeExecutionConfig {
    fn default() -> Self {
        Self {
            python_path: default_python_path(),
            timeout_secs: default_timeout_secs(),
        }
    }
}

/// Manages Python code execution sessions.
///
/// Sessions are created per-request (keyed by thread ID) and persist across
/// multiple tool calls within the same agentic loop iteration.
pub struct CodeExecutionManager {
    config: CodeExecutionConfig,
    sessions: Arc<Mutex<HashMap<ThreadId, PythonSession>>>,
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

        Ok(Self {
            config,
            sessions: Arc::new(Mutex::new(HashMap::new())),
            executor_script_path,
            installed_packages,
        })
    }

    /// Get or create a session for the current thread.
    async fn get_or_create_session(
        sessions: &Mutex<HashMap<ThreadId, PythonSession>>,
        python_path: &Path,
        executor_script: &Path,
        timeout: Duration,
    ) -> anyhow::Result<()> {
        let tid = std::thread::current().id();
        let mut map = sessions.lock().await;
        if !map.contains_key(&tid) {
            let session = PythonSession::new(python_path, executor_script, timeout).await?;
            map.insert(tid, session);
        }
        Ok(())
    }

    /// Build tool callbacks for registration with the engine.
    pub fn get_tool_callbacks(&self) -> ToolCallbacksWithTools {
        let mut callbacks = ToolCallbacksWithTools::new();

        let execute_tool =
            tools::build_execute_python_tool(self.config.timeout_secs, &self.installed_packages);

        let reset_tool = tools::build_reset_session_tool();

        // execute_python callback
        let sessions = Arc::clone(&self.sessions);
        let python_path = self.config.python_path.clone();
        let executor_script = self.executor_script_path.clone();
        let timeout = Duration::from_secs(self.config.timeout_secs);

        let execute_callback: Arc<mistralrs_mcp::MultimodalToolCallback> =
            Arc::new(move |func: &CalledFunction| {
                let sessions = Arc::clone(&sessions);
                let python_path = python_path.clone();
                let executor_script = executor_script.clone();

                // Parse the code argument.
                let args: serde_json::Value = serde_json::from_str(&func.arguments)?;
                let code = args
                    .get("code")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| anyhow::anyhow!("Missing 'code' argument"))?
                    .to_string();

                // Models sometimes emit LaTeX operators instead of Python ones.
                let code = sanitize_latex_operators(&code);

                // Run async code execution in the current runtime.
                let handle = tokio::runtime::Handle::current();
                tokio::task::block_in_place(|| {
                    handle.block_on(async {
                        Self::get_or_create_session(
                            &sessions,
                            &python_path,
                            &executor_script,
                            timeout,
                        )
                        .await?;

                        let tid = std::thread::current().id();
                        let mut map = sessions.lock().await;
                        let session = map.get_mut(&tid).unwrap();
                        let result = session.execute(&code).await;

                        Ok(ToolOutput::Multimodal {
                            text: result.text,
                            images: result.images,
                        })
                    })
                })
            });

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

        let reset_callback: Arc<mistralrs_mcp::ToolCallback> = Arc::new(
            move |_func: &CalledFunction| {
                let sessions = Arc::clone(&sessions);
                let python_path = python_path_reset.clone();
                let executor_script = executor_script_reset.clone();

                let handle = tokio::runtime::Handle::current();
                tokio::task::block_in_place(|| {
                    handle.block_on(async {
                        Self::get_or_create_session(
                            &sessions,
                            &python_path,
                            &executor_script,
                            timeout,
                        )
                        .await?;

                        let tid = std::thread::current().id();
                        let mut map = sessions.lock().await;
                        let session = map.get_mut(&tid).unwrap();
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

        callbacks
    }
}

/// Replace common LaTeX math operators that models sometimes emit with
/// their Python equivalents so the code actually runs.
fn sanitize_latex_operators(code: &str) -> String {
    code.replace("\\le", "<=")
        .replace("\\le", "<=")
        .replace("\\ge ", ">=")
        .replace("\\ge", ">=")
        .replace("\\ne ", "!=")
        .replace("\\ne", "!=")
        .replace("\\times ", "* ")
        .replace("\\cdot ", "* ")
}
