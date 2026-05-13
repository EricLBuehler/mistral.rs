use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader, BufWriter};
use tokio::process::{Child, ChildStdin, ChildStdout, Command};
use tokio::task::JoinHandle;

use crate::output::CodeExecResult;
use crate::protocol::{ExecuteOutputSpec, ExecuteResponse, ExecutorRequest, ResetResponse};

/// After SIGINT, wait this long for the child to return before SIGKILL.
const SIGINT_GRACE_WAIT: Duration = Duration::from_secs(3);

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
    _stderr_pump: Option<JoinHandle<()>>,
}

impl PythonSession {
    pub async fn new(
        python_path: &Path,
        executor_script: &Path,
        timeout: Duration,
        working_directory: Option<&Path>,
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

        let (child, stdin, stdout, stderr_pump) =
            Self::spawn_process(&python_path, &executor_script, &work_dir).await?;

        Ok(Self {
            child,
            stdin,
            stdout,
            work_dir,
            timeout,
            alive: true,
            python_path,
            executor_script,
            last_active: Instant::now(),
            _stderr_pump: stderr_pump,
        })
    }

    async fn spawn_process(
        python_path: &Path,
        executor_script: &Path,
        work_dir: &Path,
    ) -> anyhow::Result<(
        Child,
        BufWriter<ChildStdin>,
        BufReader<ChildStdout>,
        Option<JoinHandle<()>>,
    )> {
        let mut child = Command::new(python_path)
            .arg("-u")
            .arg(executor_script)
            .arg(work_dir)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .kill_on_drop(true)
            .spawn()?;

        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| anyhow::anyhow!("Failed to capture stdin"))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| anyhow::anyhow!("Failed to capture stdout"))?;
        let stderr = child.stderr.take();

        let stderr_pump = stderr.map(|stderr| {
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
                            }
                        }
                        Err(_) => break,
                    }
                }
            })
        });

        Ok((
            child,
            BufWriter::new(stdin),
            BufReader::new(stdout),
            stderr_pump,
        ))
    }

    async fn respawn(&mut self) -> anyhow::Result<()> {
        let (child, stdin, stdout, stderr_pump) =
            Self::spawn_process(&self.python_path, &self.executor_script, &self.work_dir).await?;
        self.child = child;
        self.stdin = stdin;
        self.stdout = stdout;
        self._stderr_pump = stderr_pump;
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
            anyhow::bail!("Python subprocess closed stdout (EOF)");
        }
        Ok(serde_json::from_str(line.trim())?)
    }
}
