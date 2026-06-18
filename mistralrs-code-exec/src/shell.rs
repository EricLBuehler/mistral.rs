use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Context;
use mistralrs_mcp::{
    CalledFunction, ShellOptions, ToolCallbackKind, ToolCallbackWithTool, ToolCallbacksWithTools,
    ToolOutput,
};
use mistralrs_sandbox::{NetworkMode, Sandbox, SandboxPolicy};
use serde::{Deserialize, Serialize};
use tokio::process::Command;
use tokio::sync::Mutex;

use crate::files::{
    append_auto_output_files, collect_output_files, snapshot_output_files, surface_outputs_json,
};
use crate::protocol::ExecuteOutputSpec;
use crate::tools;

const REAP_INTERVAL: Duration = Duration::from_secs(300);
const SESSION_TTL: Duration = Duration::from_secs(3600);
const DEFAULT_MAX_OUTPUT_LENGTH: usize = 4096;

/// Shell execution config.
#[derive(Clone, Serialize, Deserialize)]
pub struct ShellConfig {
    #[serde(default = "default_shell_path")]
    pub shell_path: PathBuf,
    #[serde(default = "default_timeout_secs")]
    pub timeout_secs: u64,
    #[serde(default)]
    pub working_directory: Option<PathBuf>,
    #[serde(default)]
    pub sandbox_policy: Option<SandboxPolicy>,
    #[serde(default)]
    pub permission: mistralrs_mcp::AgentPermission,
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
            timeout_secs: default_timeout_secs(),
            working_directory: None,
            sandbox_policy: None,
            permission: mistralrs_mcp::AgentPermission::Auto,
        }
    }
}

fn default_shell_path() -> PathBuf {
    if cfg!(windows) {
        PathBuf::from("cmd")
    } else {
        PathBuf::from("/bin/sh")
    }
}

fn default_timeout_secs() -> u64 {
    30
}

struct ShellSession {
    work_dir: PathBuf,
    mounted_key: Option<String>,
    mounted_input_files_key: Option<String>,
    last_active: Instant,
}

impl ShellSession {
    fn new(working_directory: Option<&Path>, session_id: &str) -> anyhow::Result<Self> {
        let work_dir = if let Some(root) = working_directory {
            let path = root.join(session_id);
            std::fs::create_dir_all(&path)?;
            path
        } else {
            let dir = tempfile::Builder::new()
                .prefix("mistralrs-shell-")
                .tempdir()?;
            #[allow(deprecated)]
            dir.into_path()
        };
        Ok(Self {
            work_dir,
            mounted_key: None,
            mounted_input_files_key: None,
            last_active: Instant::now(),
        })
    }

    fn seconds_since_last_active(&self) -> u64 {
        self.last_active.elapsed().as_secs()
    }
}

#[derive(Clone)]
struct ShellSpawnCtx {
    shell_path: PathBuf,
    timeout: Duration,
    working_directory: Option<PathBuf>,
    sandbox: Arc<dyn Sandbox>,
    sandbox_policy: SandboxPolicy,
}

pub struct ShellManager {
    config: ShellConfig,
    sessions: Arc<Mutex<HashMap<String, ShellSession>>>,
    sandbox: Arc<dyn Sandbox>,
    sandbox_policy: SandboxPolicy,
}

#[derive(Debug, Deserialize)]
struct ShellArgs {
    commands: Vec<String>,
    #[serde(default)]
    outputs: Vec<ExecuteOutputSpec>,
    timeout_ms: Option<u64>,
    max_output_length: Option<usize>,
}

#[derive(Debug, Deserialize)]
struct SurfaceOutputsArgs {
    outputs: Vec<ExecuteOutputSpec>,
}

impl ShellManager {
    pub async fn new(config: ShellConfig) -> anyhow::Result<Self> {
        validate_shell(&config.shell_path).await?;
        let sessions = Arc::new(Mutex::new(HashMap::new()));
        spawn_reaper(Arc::clone(&sessions));
        let (sandbox, sandbox_policy) = sandbox_for_config(&config)?;
        let workdir = config
            .working_directory
            .as_ref()
            .map(|path| path.display().to_string())
            .unwrap_or_else(|| "per-session temp dir".to_string());
        tracing::info!(
            "shell sandbox: {} (workdir={}, network={:?}, strict={})",
            sandbox.name(),
            workdir,
            sandbox_policy.network,
            sandbox_policy.strict,
        );
        Ok(Self {
            config,
            sessions,
            sandbox,
            sandbox_policy,
        })
    }

    pub fn effective_protection(&self) -> mistralrs_sandbox::EffectiveProtection {
        self.sandbox.effective(&self.sandbox_policy)
    }

    pub fn network_mode(&self) -> Option<NetworkMode> {
        let policy = self.config.sandbox_policy.as_ref()?;
        if self.effective_protection().network_isolated {
            Some(policy.network)
        } else {
            None
        }
    }

    fn spawn_ctx(&self) -> ShellSpawnCtx {
        ShellSpawnCtx {
            shell_path: self.config.shell_path.clone(),
            timeout: Duration::from_secs(self.config.timeout_secs),
            working_directory: self.config.working_directory.clone(),
            sandbox: Arc::clone(&self.sandbox),
            sandbox_policy: self.sandbox_policy.clone(),
        }
    }

    pub fn get_tool_callbacks(&self) -> ToolCallbacksWithTools {
        let mut callbacks = ToolCallbacksWithTools::new();
        let sessions = Arc::clone(&self.sessions);
        let ctx = self.spawn_ctx();
        let tool = tools::build_shell_tool(
            self.config.timeout_secs,
            self.effective_protection(),
            self.network_mode(),
        );

        let callback: Arc<mistralrs_mcp::MultimodalToolCallback> = Arc::new(
            move |func: &CalledFunction, tc: &mistralrs_mcp::ToolCallContext| {
                let sessions = Arc::clone(&sessions);
                let ctx = ctx.clone();
                let session_id = tc
                    .session_id
                    .clone()
                    .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
                let args: ShellArgs = serde_json::from_str(&func.arguments)?;
                if args.commands.is_empty() {
                    anyhow::bail!("Missing shell commands");
                }

                let handle = tokio::runtime::Handle::current();
                tokio::task::block_in_place(|| {
                    handle.block_on(async {
                        let mut sessions = session_for(&sessions, &ctx, &session_id).await?;
                        let session = sessions
                            .get_mut(&session_id)
                            .ok_or_else(|| anyhow::anyhow!("missing shell session"))?;
                        mount_skills(session, tc.shell_options.as_ref())?;
                        mount_input_files(session, &tc.input_files)?;
                        session.last_active = Instant::now();
                        let work_dir = session.work_dir.clone();
                        let snapshot = snapshot_output_files(&work_dir);
                        drop(sessions);
                        let text = execute_commands(&ctx, &work_dir, &args).await;
                        let mut output_files = collect_output_files(&work_dir, &args.outputs);
                        append_auto_output_files(
                            &work_dir,
                            &snapshot,
                            &args.outputs,
                            &mut output_files,
                        );
                        let files = output_files
                            .iter()
                            .map(crate::execute_file_to_tool_file)
                            .collect();
                        Ok(ToolOutput::Multimodal {
                            text,
                            images: vec![],
                            video_frames: vec![],
                            files,
                        })
                    })
                })
            },
        );

        callbacks.insert(
            tools::SHELL_TOOL_NAME.to_string(),
            ToolCallbackWithTool {
                callback: ToolCallbackKind::Multimodal(callback),
                tool,
            },
        );

        let sessions = Arc::clone(&self.sessions);
        let ctx = self.spawn_ctx();
        let surface_outputs_callback: Arc<mistralrs_mcp::MultimodalToolCallback> = Arc::new(
            move |func: &CalledFunction, tc: &mistralrs_mcp::ToolCallContext| {
                let sessions = Arc::clone(&sessions);
                let ctx = ctx.clone();
                let session_id = tc
                    .session_id
                    .clone()
                    .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
                let args: SurfaceOutputsArgs = serde_json::from_str(&func.arguments)?;
                if args.outputs.is_empty() {
                    anyhow::bail!("Missing outputs");
                }

                let handle = tokio::runtime::Handle::current();
                tokio::task::block_in_place(|| {
                    handle.block_on(async {
                        let mut sessions = session_for(&sessions, &ctx, &session_id).await?;
                        let session = sessions
                            .get_mut(&session_id)
                            .ok_or_else(|| anyhow::anyhow!("missing shell session"))?;
                        mount_skills(session, tc.shell_options.as_ref())?;
                        mount_input_files(session, &tc.input_files)?;
                        session.last_active = Instant::now();
                        let work_dir = session.work_dir.clone();
                        drop(sessions);

                        let output_files = collect_output_files(&work_dir, &args.outputs);
                        let text = surface_outputs_json(&work_dir, &output_files);
                        let files = output_files
                            .iter()
                            .map(crate::execute_file_to_tool_file)
                            .collect();
                        Ok(ToolOutput::Multimodal {
                            text,
                            images: vec![],
                            video_frames: vec![],
                            files,
                        })
                    })
                })
            },
        );

        callbacks.insert(
            tools::SURFACE_OUTPUTS_TOOL_NAME.to_string(),
            ToolCallbackWithTool {
                callback: ToolCallbackKind::Multimodal(surface_outputs_callback),
                tool: tools::build_surface_outputs_tool(),
            },
        );

        callbacks
    }
}

async fn validate_shell(shell_path: &Path) -> anyhow::Result<()> {
    let mut command = Command::new(shell_path);
    if cfg!(windows) {
        command.args(["/C", "ver"]);
    } else {
        command.args(["-c", "true"]);
    }
    let output = command
        .output()
        .await
        .map_err(|e| anyhow::anyhow!("Shell not found at '{}': {e}", shell_path.display()))?;
    if !output.status.success() && !cfg!(windows) {
        anyhow::bail!(
            "Shell at '{}' returned non-zero status",
            shell_path.display()
        );
    }
    Ok(())
}

fn sandbox_for_config(config: &ShellConfig) -> anyhow::Result<(Arc<dyn Sandbox>, SandboxPolicy)> {
    let (sandbox, policy): (Arc<dyn Sandbox>, SandboxPolicy) = match config.sandbox_policy.clone() {
        Some(policy) => (Arc::from(mistralrs_sandbox::detect()), policy),
        None => (
            Arc::from(mistralrs_sandbox::null()),
            SandboxPolicy::default(),
        ),
    };
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

fn spawn_reaper(sessions: Arc<Mutex<HashMap<String, ShellSession>>>) {
    tokio::spawn(async move {
        loop {
            tokio::time::sleep(REAP_INTERVAL).await;
            let mut map = sessions.lock().await;
            let before = map.len();
            map.retain(|_, session| session.seconds_since_last_active() < SESSION_TTL.as_secs());
            let reaped = before - map.len();
            if reaped > 0 {
                tracing::debug!(
                    "Reaped {reaped} idle shell session(s) ({} remaining)",
                    map.len()
                );
            }
        }
    });
}

async fn session_for<'a>(
    sessions: &'a Mutex<HashMap<String, ShellSession>>,
    ctx: &ShellSpawnCtx,
    session_id: &str,
) -> anyhow::Result<tokio::sync::MutexGuard<'a, HashMap<String, ShellSession>>> {
    let mut map = sessions.lock().await;
    if !map.contains_key(session_id) {
        let session = ShellSession::new(ctx.working_directory.as_deref(), session_id)?;
        map.insert(session_id.to_string(), session);
    }
    Ok(map)
}

fn mount_input_files(
    session: &mut ShellSession,
    files: &[mistralrs_mcp::ToolInputFile],
) -> anyhow::Result<()> {
    let key = files
        .iter()
        .map(|f| format!("{}:{}:{}", f.id, f.name, f.size_bytes))
        .collect::<Vec<_>>()
        .join("\n");
    if session.mounted_input_files_key.as_deref() == Some(key.as_str()) {
        return Ok(());
    }
    crate::mount::mount_input_files(&session.work_dir, files)?;
    session.mounted_input_files_key = Some(key);
    Ok(())
}

fn mount_skills(
    session: &mut ShellSession,
    shell_options: Option<&ShellOptions>,
) -> anyhow::Result<()> {
    let Some(options) = shell_options else {
        return Ok(());
    };
    if options.skills.is_empty() {
        return Ok(());
    }
    let key = options
        .skills
        .iter()
        .map(|s| format!("{}={}", s.name, s.source_path.display()))
        .collect::<Vec<_>>()
        .join("\n");
    if session.mounted_key.as_deref() == Some(key.as_str()) {
        return Ok(());
    }
    let skills_dir = session.work_dir.join("skills");
    if skills_dir.exists() {
        std::fs::remove_dir_all(&skills_dir)?;
    }
    std::fs::create_dir_all(&skills_dir)?;
    for skill in &options.skills {
        let dest = skills_dir.join(safe_skill_dir_name(&skill.name));
        copy_dir_all(&skill.source_path, &dest).with_context(|| {
            format!(
                "copy skill {} from {}",
                skill.name,
                skill.source_path.display()
            )
        })?;
    }
    session.mounted_key = Some(key);
    Ok(())
}

fn safe_skill_dir_name(name: &str) -> String {
    name.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

fn copy_dir_all(src: &Path, dst: &Path) -> anyhow::Result<()> {
    std::fs::create_dir_all(dst)?;
    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        let file_type = entry.file_type()?;
        let dest = dst.join(entry.file_name());
        if file_type.is_dir() {
            copy_dir_all(&entry.path(), &dest)?;
        } else if file_type.is_file() {
            std::fs::copy(entry.path(), dest)?;
        }
    }
    Ok(())
}

async fn execute_commands(ctx: &ShellSpawnCtx, work_dir: &Path, args: &ShellArgs) -> String {
    let command = args.commands.join("\n");
    let timeout = args
        .timeout_ms
        .map(Duration::from_millis)
        .unwrap_or(ctx.timeout);
    let max_output = args.max_output_length.unwrap_or(DEFAULT_MAX_OUTPUT_LENGTH);

    let mut cmd = Command::new(&ctx.shell_path);
    if cfg!(windows) {
        cmd.arg("/C").arg(&command);
    } else {
        cmd.arg("-c").arg(&command);
    }
    cmd.current_dir(work_dir)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .kill_on_drop(true);

    let mut effective_policy = ctx.sandbox_policy.clone();
    effective_policy.session_workdir = Some(work_dir.to_path_buf());
    if let Err(e) = ctx.sandbox.harden(&mut cmd, &effective_policy) {
        return shell_error_json(
            &command,
            work_dir,
            None,
            "",
            &format!("sandbox harden failed: {e}"),
        );
    }

    let child = match cmd.spawn() {
        Ok(child) => child,
        Err(e) => return shell_error_json(&command, work_dir, None, "", &e.to_string()),
    };
    if let Some(pid) = child.id() {
        if let Err(e) = ctx.sandbox.attach(pid, &effective_policy) {
            tracing::warn!("sandbox attach failed for shell pid {pid}: {e}");
        }
    }

    match tokio::time::timeout(timeout, child.wait_with_output()).await {
        Ok(Ok(output)) => {
            let stdout = truncate_utf8(&String::from_utf8_lossy(&output.stdout), max_output);
            let stderr = truncate_utf8(&String::from_utf8_lossy(&output.stderr), max_output);
            let exit_code = output.status.code();
            serde_json::json!({
                "status": if output.status.success() { "success" } else { "error" },
                "commands": args.commands,
                "working_directory": work_dir.display().to_string(),
                "stdout": stdout,
                "stderr": stderr,
                "exit_code": exit_code,
                "timed_out": false,
            })
            .to_string()
        }
        Ok(Err(e)) => shell_error_json(&command, work_dir, None, "", &e.to_string()),
        Err(_) => shell_timeout_json(args, work_dir, timeout),
    }
}

fn truncate_utf8(text: &str, max: usize) -> String {
    text.chars().take(max).collect()
}

fn shell_error_json(
    command: &str,
    work_dir: &Path,
    exit_code: Option<i32>,
    stdout: &str,
    stderr: &str,
) -> String {
    serde_json::json!({
        "status": "error",
        "commands": [command],
        "working_directory": work_dir.display().to_string(),
        "stdout": stdout,
        "stderr": stderr,
        "exit_code": exit_code,
        "timed_out": false,
    })
    .to_string()
}

fn shell_timeout_json(args: &ShellArgs, work_dir: &Path, timeout: Duration) -> String {
    serde_json::json!({
        "status": "timeout",
        "commands": args.commands,
        "working_directory": work_dir.display().to_string(),
        "stdout": "",
        "stderr": format!("Shell command timed out after {} seconds.", timeout.as_secs()),
        "exit_code": null,
        "timed_out": true,
    })
    .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn surface_outputs_reports_existing_and_missing_files() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("deck.pptx"), b"pptx bytes").unwrap();

        let outputs = vec![
            ExecuteOutputSpec {
                name: "deck.pptx".to_string(),
                format: None,
            },
            ExecuteOutputSpec {
                name: "missing.html".to_string(),
                format: None,
            },
        ];

        let files = collect_output_files(dir.path(), &outputs);

        assert_eq!(files[0].name, "deck.pptx");
        assert_eq!(files[0].format, "pptx");
        assert_eq!(
            files[0].mime_type.as_deref(),
            Some("application/vnd.openxmlformats-officedocument.presentationml.presentation")
        );
        assert!(files[0].data_base64.is_some());
        assert!(files[0].error.is_none());
        assert_eq!(files[1].error.as_deref(), Some("not produced"));

        let json: serde_json::Value =
            serde_json::from_str(&surface_outputs_json(dir.path(), &files)).unwrap();
        assert_eq!(json["status"], "error");
        assert_eq!(json["files"][0]["status"], "success");
        assert_eq!(json["files"][1]["status"], "error");
    }
}
