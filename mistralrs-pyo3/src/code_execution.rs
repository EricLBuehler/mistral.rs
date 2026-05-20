use std::{path::PathBuf, sync::Arc};

use pyo3::exceptions::PyValueError;
use pyo3::{
    pyclass, pymethods,
    types::{PyAnyMethods, PyDict, PyDictMethods},
    Py, PyAny, PyRef, PyResult, Python,
};

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
    pub(crate) permission: Option<mistralrs_core::CodeExecutionPermission>,
    pub(crate) approval_callback: Option<Py<PyAny>>,
}

#[pymethods]
impl CodeExecutionConfig {
    #[new]
    #[pyo3(signature = (
        python_path = None,
        timeout_secs = None,
        working_directory = None,
        sandbox_policy = None,
        permission = None,
        approval_callback = None,
    ))]
    fn new(
        python_path: Option<PathBuf>,
        timeout_secs: Option<u64>,
        working_directory: Option<PathBuf>,
        sandbox_policy: Option<SandboxPolicy>,
        permission: Option<String>,
        approval_callback: Option<Py<PyAny>>,
    ) -> PyResult<Self> {
        Ok(Self {
            python_path,
            timeout_secs,
            working_directory,
            sandbox_policy,
            permission: parse_permission(permission.as_deref())?,
            approval_callback,
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "CodeExecutionConfig(python_path={:?}, timeout_secs={:?}, working_directory={:?}, sandbox_policy={:?}, permission={:?})",
            self.python_path,
            self.timeout_secs,
            self.working_directory,
            self.sandbox_policy,
            self.permission
        )
    }
}

pub(crate) fn parse_permission(
    value: Option<&str>,
) -> PyResult<Option<mistralrs_core::CodeExecutionPermission>> {
    value
        .map(|value| match value {
            "auto" => Ok(mistralrs_core::CodeExecutionPermission::Auto),
            "ask" => Ok(mistralrs_core::CodeExecutionPermission::Ask),
            "deny" => Ok(mistralrs_core::CodeExecutionPermission::Deny),
            other => Err(PyValueError::new_err(format!(
                "invalid code execution permission `{other}`; expected auto, ask, or deny"
            ))),
        })
        .transpose()
}

pub(crate) fn parse_agent_permission(
    value: Option<&str>,
) -> PyResult<Option<mistralrs_core::AgentPermission>> {
    value
        .map(|value| match value {
            "auto" => Ok(mistralrs_core::AgentPermission::Auto),
            "ask" => Ok(mistralrs_core::AgentPermission::Ask),
            "deny" => Ok(mistralrs_core::AgentPermission::Deny),
            other => Err(PyValueError::new_err(format!(
                "invalid agent permission `{other}`; expected auto, ask, or deny"
            ))),
        })
        .transpose()
}

fn source_str(source: &mistralrs_core::AgentToolSource) -> &'static str {
    match source {
        mistralrs_core::AgentToolSource::BuiltIn => "built_in",
        mistralrs_core::AgentToolSource::User => "user",
        mistralrs_core::AgentToolSource::Mcp => "mcp",
        mistralrs_core::AgentToolSource::External => "external",
    }
}

fn kind_str(kind: &mistralrs_core::AgentToolKind) -> &'static str {
    match kind {
        mistralrs_core::AgentToolKind::CodeExecution => "code_execution",
        mistralrs_core::AgentToolKind::WebSearch => "web_search",
        mistralrs_core::AgentToolKind::File => "file",
        mistralrs_core::AgentToolKind::Custom => "custom",
        mistralrs_core::AgentToolKind::External => "external",
    }
}

#[pyclass(name = "AgentToolMetadata", get_all)]
#[derive(Clone, Debug)]
pub struct AgentToolMetadataPy {
    source: String,
    kind: String,
    label: String,
}

#[pymethods]
impl AgentToolMetadataPy {
    fn __repr__(&self) -> String {
        format!(
            "AgentToolMetadata(source={:?}, kind={:?}, label={:?})",
            self.source, self.kind, self.label
        )
    }
}

impl From<&mistralrs_core::AgentToolMetadata> for AgentToolMetadataPy {
    fn from(tool: &mistralrs_core::AgentToolMetadata) -> Self {
        Self {
            source: source_str(&tool.source).to_string(),
            kind: kind_str(&tool.kind).to_string(),
            label: tool.label.clone(),
        }
    }
}

#[pyclass(name = "AgentToolApproval", get_all)]
#[derive(Clone, Debug)]
pub struct AgentToolApprovalPy {
    approval_id: String,
    session_id: String,
    round: usize,
    tool: AgentToolMetadataPy,
    arguments_json: String,
    code: Option<String>,
}

#[pymethods]
impl AgentToolApprovalPy {
    fn arguments(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let json = py.import("json")?;
        Ok(json.call_method1("loads", (&self.arguments_json,))?.into())
    }

    fn __repr__(&self) -> String {
        format!(
            "AgentToolApproval(approval_id={:?}, session_id={:?}, round={}, tool={:?})",
            self.approval_id, self.session_id, self.round, self.tool
        )
    }
}

impl From<&mistralrs_core::AgentToolApproval> for AgentToolApprovalPy {
    fn from(approval: &mistralrs_core::AgentToolApproval) -> Self {
        Self {
            approval_id: approval.approval_id.clone(),
            session_id: approval.session_id.clone(),
            round: approval.round,
            tool: AgentToolMetadataPy::from(&approval.tool),
            arguments_json: approval.arguments.to_string(),
            code: approval
                .arguments
                .get("code")
                .and_then(|value| value.as_str())
                .map(str::to_string),
        }
    }
}

#[pyclass(name = "AgentToolApprovalDecision", get_all)]
#[derive(Clone, Debug)]
pub struct AgentToolApprovalDecisionPy {
    decision: String,
    remember_for_session: bool,
    message: Option<String>,
}

#[pymethods]
impl AgentToolApprovalDecisionPy {
    #[new]
    #[pyo3(signature = (decision, remember_for_session = false, message = None))]
    fn new(
        decision: String,
        remember_for_session: bool,
        message: Option<String>,
    ) -> PyResult<Self> {
        match decision.as_str() {
            "approve" | "deny" => Ok(Self {
                decision,
                remember_for_session,
                message,
            }),
            other => Err(PyValueError::new_err(format!(
                "decision must be 'approve' or 'deny' (got {other:?})"
            ))),
        }
    }

    #[staticmethod]
    #[pyo3(signature = (remember_for_session = false))]
    fn approve(remember_for_session: bool) -> Self {
        Self {
            decision: "approve".to_string(),
            remember_for_session,
            message: None,
        }
    }

    #[staticmethod]
    #[pyo3(signature = (message = None))]
    fn deny(message: Option<String>) -> Self {
        Self {
            decision: "deny".to_string(),
            remember_for_session: false,
            message,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "AgentToolApprovalDecision(decision={:?}, remember_for_session={}, message={:?})",
            self.decision, self.remember_for_session, self.message
        )
    }
}

impl AgentToolApprovalDecisionPy {
    fn to_core(&self) -> mistralrs_core::AgentToolApprovalDecision {
        match self.decision.as_str() {
            "approve" => mistralrs_core::AgentToolApprovalDecision::approve()
                .with_remember_for_session(self.remember_for_session),
            "deny" => mistralrs_core::AgentToolApprovalDecision::deny(self.message.clone()),
            _ => mistralrs_core::AgentToolApprovalDecision::deny_with_message(
                "Invalid agent approval decision.",
            ),
        }
    }
}

fn parse_agent_approval_result(
    py: Python<'_>,
    result: Py<PyAny>,
) -> PyResult<mistralrs_core::AgentToolApprovalDecision> {
    if let Ok(approve) = result.extract::<bool>(py) {
        return Ok(if approve {
            mistralrs_core::AgentToolApprovalDecision::approve()
        } else {
            mistralrs_core::AgentToolApprovalDecision::deny(None)
        });
    }
    if let Ok(decision) = result.extract::<PyRef<'_, AgentToolApprovalDecisionPy>>(py) {
        return Ok(decision.to_core());
    }
    Err(PyValueError::new_err(
        "approval callback must return bool or AgentToolApprovalDecision",
    ))
}

pub(crate) fn build_agent_approval_callback(
    callback: Option<Py<PyAny>>,
) -> Option<mistralrs_core::AgentToolApprovalCallback> {
    callback.map(|callback| {
        Arc::new(move |approval: &mistralrs_core::AgentToolApproval| {
            Python::with_gil(|py| {
                let payload = AgentToolApprovalPy::from(approval);
                callback
                    .call1(py, (payload,))
                    .and_then(|result| parse_agent_approval_result(py, result))
                    .unwrap_or_else(|_| mistralrs_core::AgentToolApprovalDecision::deny(None))
            })
        }) as mistralrs_core::AgentToolApprovalCallback
    })
}

impl From<CodeExecutionConfig> for mistralrs_core::CodeExecutionConfig {
    fn from(cfg: CodeExecutionConfig) -> Self {
        let default = mistralrs_core::CodeExecutionConfig::default();
        let approval_callback = cfg.approval_callback.map(|callback| {
            Arc::new(move |approval: &mistralrs_core::CodeExecutionApproval| {
                Python::with_gil(|py| {
                    let payload = PyDict::new(py);
                    if payload
                        .set_item("approval_id", &approval.approval_id)
                        .is_err()
                    {
                        return false;
                    }
                    if payload
                        .set_item("session_id", &approval.session_id)
                        .is_err()
                    {
                        return false;
                    }
                    if payload.set_item("code", &approval.code).is_err() {
                        return false;
                    }
                    if payload.set_item("outputs", &approval.outputs).is_err() {
                        return false;
                    }
                    let workdir = approval
                        .working_directory
                        .as_ref()
                        .map(|path| path.display().to_string());
                    if payload.set_item("working_directory", workdir).is_err() {
                        return false;
                    }

                    callback
                        .call1(py, (payload,))
                        .and_then(|result| result.extract::<bool>(py))
                        .unwrap_or(false)
                })
            }) as mistralrs_core::CodeExecutionApprovalCallback
        });
        mistralrs_core::CodeExecutionConfig {
            python_path: cfg.python_path.unwrap_or(default.python_path),
            timeout_secs: cfg.timeout_secs.unwrap_or(default.timeout_secs),
            working_directory: cfg.working_directory.or(default.working_directory),
            sandbox_policy: cfg.sandbox_policy.map(Into::into),
            permission: cfg.permission.unwrap_or(default.permission),
            approval_callback,
        }
    }
}
