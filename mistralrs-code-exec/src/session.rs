use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Context;
use mistralrs_sandbox::{Sandbox, SandboxPolicy};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader, BufWriter};
use tokio::process::{Child, ChildStdin, ChildStdout, Command};
use tokio::sync::Mutex;
use tokio::task::JoinHandle;

use crate::output::CodeExecResult;
use crate::protocol::{ExecuteOutputSpec, ExecuteResponse, ExecutorRequest, ResetResponse};

/// After SIGINT, wait this long for the child to return before SIGKILL.
const SIGINT_GRACE_WAIT: Duration = Duration::from_secs(3);
const STDERR_DRAIN_WAIT: Duration = Duration::from_millis(20);
const STDERR_TAIL_LINES: usize = 32;

type StderrTail = Arc<Mutex<VecDeque<String>>>;

struct SpawnedProcess {
    child: Child,
    stdin: BufWriter<ChildStdin>,
    stdout: BufReader<ChildStdout>,
    stderr_tail: StderrTail,
    stderr_pump: Option<JoinHandle<()>>,
}

pub struct PythonSession {
    child: Child,
    stdin: BufWriter<ChildStdin>,
    stdout: BufReader<ChildStdout>,
    work_dir: PathBuf,
    timeout: Duration,
    alive: bool,
    python_path: PathBuf,
    executor_script: PathBuf,
    last_active: Instant,
    sandbox: Arc<dyn Sandbox>,
    sandbox_policy: SandboxPolicy,
    stderr_tail: StderrTail,
    _stderr_pump: Option<JoinHandle<()>>,
}

impl PythonSession {
    pub async fn new(
        python_path: &Path,
        executor_script: &Path,
        timeout: Duration,
        working_directory: Option<&Path>,
        sandbox: Arc<dyn Sandbox>,
        sandbox_policy: SandboxPolicy,
    ) -> anyhow::Result<Self> {
        let work_dir = if let Some(dir) = working_directory {
            std::fs::create_dir_all(dir)?;
            dir.to_path_buf()
        } else {
            let dir = tempfile::Builder::new()
                .prefix("mistralrs-code-")
                .tempdir()?;
            #[allow(deprecated)]
            dir.into_path()
        };
        let python_path = python_path.to_path_buf();
        let executor_script = executor_script.to_path_buf();

        let spawned = Self::spawn_process(
            &python_path,
            &executor_script,
            &work_dir,
            sandbox.as_ref(),
            &sandbox_policy,
        )
        .await?;

        Ok(Self {
            child: spawned.child,
            stdin: spawned.stdin,
            stdout: spawned.stdout,
            work_dir,
            timeout,
            alive: true,
            python_path,
            executor_script,
            last_active: Instant::now(),
            sandbox,
            sandbox_policy,
            stderr_tail: spawned.stderr_tail,
            _stderr_pump: spawned.stderr_pump,
        })
    }

    async fn spawn_process(
        python_path: &Path,
        executor_script: &Path,
        work_dir: &Path,
        sandbox: &dyn Sandbox,
        sandbox_policy: &SandboxPolicy,
    ) -> anyhow::Result<SpawnedProcess> {
        let mut cmd = Command::new(python_path);
        cmd.arg("-u")
            .arg(executor_script)
            .arg(work_dir)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .kill_on_drop(true);

        let mut effective_policy = sandbox_policy.clone();
        effective_policy.session_workdir = Some(work_dir.to_path_buf());
        sandbox
            .harden(&mut cmd, &effective_policy)
            .map_err(|e| anyhow::anyhow!("sandbox harden failed: {e}"))?;

        let mut child = cmd.spawn().context("spawn sandboxed Python subprocess")?;

        if let Some(pid) = child.id() {
            if let Err(e) = sandbox.attach(pid, &effective_policy) {
                tracing::warn!("sandbox attach failed for pid {pid}: {e}");
            }
        }

        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| anyhow::anyhow!("Failed to capture stdin"))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| anyhow::anyhow!("Failed to capture stdout"))?;
        let stderr = child.stderr.take();
        let stderr_tail = Arc::new(Mutex::new(VecDeque::with_capacity(STDERR_TAIL_LINES)));

        let stderr_pump = stderr.map(|stderr| {
            let stderr_tail = Arc::clone(&stderr_tail);
            tokio::spawn(async move {
                let mut reader = BufReader::new(stderr);
                let mut buf = String::new();
                loop {
                    buf.clear();
                    match reader.read_line(&mut buf).await {
                        Ok(0) => break,
                        Ok(_) => {
                            let line = buf.trim_end();
                            if !line.is_empty() {
                                tracing::warn!(target: "code_exec.python", "{line}");
                                let mut tail = stderr_tail.lock().await;
                                if tail.len() == STDERR_TAIL_LINES {
                                    tail.pop_front();
                                }
                                tail.push_back(line.to_string());
                            }
                        }
                        Err(_) => break,
                    }
                }
            })
        });

        Ok(SpawnedProcess {
            child,
            stdin: BufWriter::new(stdin),
            stdout: BufReader::new(stdout),
            stderr_tail,
            stderr_pump,
        })
    }

    async fn respawn(&mut self) -> anyhow::Result<()> {
        let spawned = Self::spawn_process(
            &self.python_path,
            &self.executor_script,
            &self.work_dir,
            self.sandbox.as_ref(),
            &self.sandbox_policy,
        )
        .await?;
        self.child = spawned.child;
        self.stdin = spawned.stdin;
        self.stdout = spawned.stdout;
        self.stderr_tail = spawned.stderr_tail;
        self._stderr_pump = spawned.stderr_pump;
        self.alive = true;
        Ok(())
    }

    pub fn seconds_since_last_active(&self) -> u64 {
        self.last_active.elapsed().as_secs()
    }

    pub fn work_dir_str(&self) -> String {
        self.work_dir.display().to_string()
    }

    pub async fn execute_with_outputs(
        &mut self,
        code: &str,
        outputs: &[ExecuteOutputSpec],
    ) -> CodeExecResult {
        self.last_active = Instant::now();
        if !self.alive {
            if let Err(e) = self.respawn().await {
                return CodeExecResult::error(&format!("Failed to respawn Python session: {e}"));
            }
        }

        let request = ExecutorRequest::Execute {
            code: code.to_string(),
            outputs,
        };
        if let Err(e) = self.send(&request).await {
            self.alive = false;
            return CodeExecResult::error(&format!("Failed to send to Python: {e}"));
        }

        match tokio::time::timeout(self.timeout, self.read_response::<ExecuteResponse>()).await {
            Ok(Ok(response)) => CodeExecResult::from_response(response, &self.work_dir_str()),
            Ok(Err(e)) => {
                self.alive = false;
                CodeExecResult::error(&format!("Python subprocess error: {e}"))
            }
            Err(_) => {
                // Timeout: try SIGINT first.
                let interrupted = self.try_interrupt().await;
                if !interrupted {
                    // SIGKILL as last resort.
                    let _ = self.child.kill().await;
                    self.alive = false;
                }
                CodeExecResult::timeout(self.timeout.as_secs(), interrupted)
            }
        }
    }

    pub async fn reset(&mut self) -> anyhow::Result<()> {
        self.last_active = Instant::now();
        if !self.alive {
            self.respawn().await?;
            return Ok(());
        }

        self.send(&ExecutorRequest::Reset).await?;
        let _response: ResetResponse = self.read_response().await?;
        Ok(())
    }

    /// SIGINT then wait for a graceful response. Returns true if the session was preserved.
    async fn try_interrupt(&mut self) -> bool {
        #[cfg(unix)]
        {
            if let Some(pid) = self.child.id() {
                unsafe {
                    libc::kill(pid as i32, libc::SIGINT);
                }
            }
        }

        #[cfg(not(unix))]
        {
            // No SIGINT equivalent on this platform.
            return false;
        }

        #[cfg(unix)]
        matches!(
            tokio::time::timeout(SIGINT_GRACE_WAIT, self.read_response::<ExecuteResponse>()).await,
            Ok(Ok(_))
        )
    }

    async fn send(&mut self, request: &ExecutorRequest<'_>) -> anyhow::Result<()> {
        let json = serde_json::to_string(request)?;
        self.stdin.write_all(format!("{json}\n").as_bytes()).await?;
        self.stdin.flush().await?;
        Ok(())
    }

    async fn read_response<T: serde::de::DeserializeOwned>(&mut self) -> anyhow::Result<T> {
        let mut line = String::new();
        let n = self.stdout.read_line(&mut line).await?;
        if n == 0 {
            self.alive = false;
            let mut msg = "Python subprocess closed stdout (EOF)".to_string();
            if let Ok(Some(status)) = self.child.try_wait() {
                msg.push_str(&format!("; exit status: {status}"));
            }
            tokio::time::sleep(STDERR_DRAIN_WAIT).await;
            let tail = self.stderr_tail.lock().await;
            if !tail.is_empty() {
                msg.push_str("; recent stderr:\n");
                msg.push_str(&tail.iter().cloned().collect::<Vec<_>>().join("\n"));
            }
            anyhow::bail!(msg);
        }
        Ok(serde_json::from_str(line.trim())?)
    }
}
