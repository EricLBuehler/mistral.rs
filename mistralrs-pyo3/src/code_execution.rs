use std::path::PathBuf;

use pyo3::exceptions::PyValueError;
use pyo3::{pyclass, pymethods, PyResult};

/// OS-level sandbox policy for the code-execution subprocess.
/// `None` for `sandbox_policy` on [`CodeExecutionConfig`] disables the
/// sandbox; constructing a `SandboxPolicy` enables it with the given limits.
/// See the [sandbox reference](/mistral.rs/reference/sandbox/) for layer
/// details.
#[pyclass]
#[derive(Clone, Debug)]
pub struct SandboxPolicy {
    pub(crate) max_memory_mb: u64,
    pub(crate) max_cpu_secs: u64,
    pub(crate) max_procs: u32,
    pub(crate) max_open_fds: u32,
    pub(crate) max_file_sz_mb: u64,
    pub(crate) network: mistralrs_core::NetworkMode,
    pub(crate) extra_fs_read: Vec<PathBuf>,
    pub(crate) extra_fs_write: Vec<PathBuf>,
    pub(crate) extra_env: Vec<String>,
    pub(crate) strict: bool,
}

#[pymethods]
impl SandboxPolicy {
    #[new]
    #[pyo3(signature = (
        max_memory_mb = 2048,
        max_cpu_secs = 300,
        max_procs = 64,
        max_open_fds = 1024,
        max_file_sz_mb = 256,
        network = "loopback".to_string(),
        extra_fs_read = Vec::new(),
        extra_fs_write = Vec::new(),
        extra_env = Vec::new(),
        strict = false,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        max_memory_mb: u64,
        max_cpu_secs: u64,
        max_procs: u32,
        max_open_fds: u32,
        max_file_sz_mb: u64,
        network: String,
        extra_fs_read: Vec<PathBuf>,
        extra_fs_write: Vec<PathBuf>,
        extra_env: Vec<String>,
        strict: bool,
    ) -> PyResult<Self> {
        let network = match network.to_ascii_lowercase().as_str() {
            "none" => mistralrs_core::NetworkMode::None,
            "loopback" => mistralrs_core::NetworkMode::Loopback,
            "full" => mistralrs_core::NetworkMode::Full,
            other => {
                return Err(PyValueError::new_err(format!(
                    "network must be one of 'none', 'loopback', 'full' (got {other:?})"
                )))
            }
        };
        Ok(Self {
            max_memory_mb,
            max_cpu_secs,
            max_procs,
            max_open_fds,
            max_file_sz_mb,
            network,
            extra_fs_read,
            extra_fs_write,
            extra_env,
            strict,
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "SandboxPolicy(max_memory_mb={}, max_cpu_secs={}, max_procs={}, max_open_fds={}, max_file_sz_mb={}, network={:?}, extra_fs_read={:?}, extra_fs_write={:?}, extra_env={:?}, strict={})",
            self.max_memory_mb,
            self.max_cpu_secs,
            self.max_procs,
            self.max_open_fds,
            self.max_file_sz_mb,
            self.network,
            self.extra_fs_read,
            self.extra_fs_write,
            self.extra_env,
            self.strict,
        )
    }
}

impl From<SandboxPolicy> for mistralrs_core::SandboxPolicy {
    fn from(p: SandboxPolicy) -> Self {
        mistralrs_core::SandboxPolicy {
            max_memory_mb: p.max_memory_mb,
            max_cpu_secs: p.max_cpu_secs,
            max_procs: p.max_procs,
            max_open_fds: p.max_open_fds,
            max_file_sz_mb: p.max_file_sz_mb,
            network: p.network,
            extra_fs_read: p.extra_fs_read,
            extra_fs_write: p.extra_fs_write,
            extra_env: p.extra_env,
            strict: p.strict,
            session_workdir: None,
        }
    }
}

/// Pass to `Runner(code_execution_config=...)` to enable the `execute_python` tool.
#[pyclass]
#[derive(Clone, Debug)]
pub struct CodeExecutionConfig {
    pub(crate) python_path: Option<PathBuf>,
    pub(crate) timeout_secs: Option<u64>,
    pub(crate) working_directory: Option<PathBuf>,
    pub(crate) sandbox_policy: Option<SandboxPolicy>,
}

#[pymethods]
impl CodeExecutionConfig {
    #[new]
    #[pyo3(signature = (
        python_path = None,
        timeout_secs = None,
        working_directory = None,
        sandbox_policy = None,
    ))]
    fn new(
        python_path: Option<PathBuf>,
        timeout_secs: Option<u64>,
        working_directory: Option<PathBuf>,
        sandbox_policy: Option<SandboxPolicy>,
    ) -> Self {
        Self {
            python_path,
            timeout_secs,
            working_directory,
            sandbox_policy,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "CodeExecutionConfig(python_path={:?}, timeout_secs={:?}, working_directory={:?}, sandbox_policy={:?})",
            self.python_path, self.timeout_secs, self.working_directory, self.sandbox_policy
        )
    }
}

impl From<CodeExecutionConfig> for mistralrs_core::CodeExecutionConfig {
    fn from(cfg: CodeExecutionConfig) -> Self {
        let default = mistralrs_core::CodeExecutionConfig::default();
        mistralrs_core::CodeExecutionConfig {
            python_path: cfg.python_path.unwrap_or(default.python_path),
            timeout_secs: cfg.timeout_secs.unwrap_or(default.timeout_secs),
            working_directory: cfg.working_directory.or(default.working_directory),
            sandbox_policy: cfg.sandbox_policy.map(Into::into),
        }
    }
}
