use std::path::PathBuf;

use pyo3::{pyclass, pymethods};

/// Pass to `Runner(code_execution_config=...)` to enable the `execute_python` tool.
#[pyclass]
#[derive(Clone, Debug)]
pub struct CodeExecutionConfig {
    pub(crate) python_path: Option<PathBuf>,
    pub(crate) timeout_secs: Option<u64>,
    pub(crate) working_directory: Option<PathBuf>,
}

#[pymethods]
impl CodeExecutionConfig {
    #[new]
    #[pyo3(signature = (python_path = None, timeout_secs = None, working_directory = None))]
    fn new(
        python_path: Option<PathBuf>,
        timeout_secs: Option<u64>,
        working_directory: Option<PathBuf>,
    ) -> Self {
        Self {
            python_path,
            timeout_secs,
            working_directory,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "CodeExecutionConfig(python_path={:?}, timeout_secs={:?}, working_directory={:?})",
            self.python_path, self.timeout_secs, self.working_directory
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
        }
    }
}
