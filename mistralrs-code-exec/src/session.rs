use std::path::{Path, PathBuf};
use std::time::Duration;

use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader, BufWriter};
use tokio::process::{Child, ChildStdin, ChildStdout, Command};

use crate::output::CodeExecResult;
use crate::protocol::{ExecuteResponse, ExecutorRequest, ResetResponse};

pub struct PythonSession {
    child: Child,
    stdin: BufWriter<ChildStdin>,
    stdout: BufReader<ChildStdout>,
    work_dir: tempfile::TempDir,
    timeout: Duration,
    alive: bool,
    python_path: PathBuf,
    executor_script: PathBuf,
}

impl PythonSession {
    pub async fn new(
        python_path: &Path,
        executor_script: &Path,
        timeout: Duration,
    ) -> anyhow::Result<Self> {
        let work_dir = tempfile::tempdir()?;
        let python_path = python_path.to_path_buf();
        let executor_script = executor_script.to_path_buf();

        let (child, stdin, stdout) =
            Self::spawn_process(&python_path, &executor_script, work_dir.path()).await?;

        Ok(Self {
            child,
            stdin,
            stdout,
            work_dir,
            timeout,
            alive: true,
            python_path,
            executor_script,
        })
    }

    async fn spawn_process(
        python_path: &Path,
        executor_script: &Path,
        work_dir: &Path,
    ) -> anyhow::Result<(Child, BufWriter<ChildStdin>, BufReader<ChildStdout>)> {
        let mut child = Command::new(python_path)
            .arg("-u")
            .arg(executor_script)
            .arg(work_dir)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::null())
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

        Ok((child, BufWriter::new(stdin), BufReader::new(stdout)))
    }

    async fn respawn(&mut self) -> anyhow::Result<()> {
        let (child, stdin, stdout) =
            Self::spawn_process(&self.python_path, &self.executor_script, self.work_dir.path())
                .await?;
        self.child = child;
        self.stdin = stdin;
        self.stdout = stdout;
        self.alive = true;
        Ok(())
    }

    pub fn work_dir_str(&self) -> String {
        self.work_dir.path().display().to_string()
    }

    pub async fn execute(&mut self, code: &str) -> CodeExecResult {
        if !self.alive {
            if let Err(e) = self.respawn().await {
                return CodeExecResult::error(&format!("Failed to respawn Python session: {e}"));
            }
        }

        let request = ExecutorRequest::Execute {
            code: code.to_string(),
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
        if !self.alive {
            self.respawn().await?;
            return Ok(());
        }

        self.send(&ExecutorRequest::Reset).await?;
        let _response: ResetResponse = self.read_response().await?;
        Ok(())
    }

    /// Send SIGINT and wait briefly for a graceful response.
    /// Returns true if the process responded (session preserved).
    async fn try_interrupt(&mut self) -> bool {
        #[cfg(unix)]
        {
            if let Some(pid) = self.child.id() {
                unsafe {
                    libc::kill(pid as i32, libc::SIGINT);
                }
            }
        }

        // Wait briefly for the interrupted execution to return a result.
        match tokio::time::timeout(Duration::from_secs(3), self.read_response::<ExecuteResponse>())
            .await
        {
            Ok(Ok(_)) => true,
            _ => false,
        }
    }

    async fn send(&mut self, request: &ExecutorRequest) -> anyhow::Result<()> {
        let json = serde_json::to_string(request)?;
        self.stdin
            .write_all(format!("{json}\n").as_bytes())
            .await?;
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
