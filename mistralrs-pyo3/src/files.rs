use std::path::PathBuf;

use pyo3::{exceptions::PyOSError, pyclass, pymethods, PyResult};

/// Python wrapper for [`mistralrs_core::RequestedFile`].
#[pyclass]
#[derive(Clone, Debug)]
pub struct RequestedFile {
    pub(crate) name: String,
    pub(crate) format: Option<String>,
    pub(crate) description: Option<String>,
}

#[pymethods]
impl RequestedFile {
    #[new]
    #[pyo3(signature = (name, format = None, description = None))]
    fn new(name: String, format: Option<String>, description: Option<String>) -> Self {
        Self {
            name,
            format,
            description,
        }
    }

    #[getter]
    fn name(&self) -> &str {
        &self.name
    }

    #[getter]
    fn format(&self) -> Option<&str> {
        self.format.as_deref()
    }

    #[getter]
    fn description(&self) -> Option<&str> {
        self.description.as_deref()
    }

    fn __repr__(&self) -> String {
        format!(
            "RequestedFile(name={:?}, format={:?}, description={:?})",
            self.name, self.format, self.description
        )
    }
}

impl From<RequestedFile> for mistralrs_core::RequestedFile {
    fn from(f: RequestedFile) -> Self {
        mistralrs_core::RequestedFile {
            name: f.name,
            format: f.format,
            description: f.description,
        }
    }
}

#[pyclass]
#[derive(Clone, Debug)]
pub struct InputFile {
    pub(crate) file: mistralrs_core::File,
}

#[pymethods]
impl InputFile {
    #[new]
    #[pyo3(signature = (name, data, mime_type = None))]
    fn new(name: String, data: Vec<u8>, mime_type: Option<String>) -> Self {
        Self::from_bytes(name, data, mime_type)
    }

    #[staticmethod]
    #[pyo3(signature = (name, text, mime_type = "text/plain".to_string()))]
    fn from_text(name: String, text: String, mime_type: String) -> Self {
        Self::from_bytes(name, text.into_bytes(), Some(mime_type))
    }

    #[staticmethod]
    #[pyo3(signature = (path, mime_type = None, name = None))]
    fn from_path(path: PathBuf, mime_type: Option<String>, name: Option<String>) -> PyResult<Self> {
        let bytes = std::fs::read(&path).map_err(PyOSError::new_err)?;
        let name = name.unwrap_or_else(|| {
            path.file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("input_file")
                .to_string()
        });
        Ok(Self::from_bytes(name, bytes, mime_type))
    }

    #[getter]
    fn id(&self) -> &str {
        &self.file.id
    }

    #[getter]
    fn name(&self) -> &str {
        &self.file.name
    }

    #[getter]
    fn mime_type(&self) -> Option<&str> {
        self.file.mime_type.as_deref()
    }

    #[getter]
    fn bytes(&self) -> u64 {
        self.file.bytes
    }

    fn __repr__(&self) -> String {
        format!(
            "InputFile(id={:?}, name={:?}, mime_type={:?}, bytes={})",
            self.file.id, self.file.name, self.file.mime_type, self.file.bytes
        )
    }
}

impl InputFile {
    fn from_bytes(name: String, bytes: Vec<u8>, mime_type: Option<String>) -> Self {
        Self {
            file: mistralrs_core::File::from_bytes(
                mistralrs_core::File::make_upload_id(),
                name,
                mime_type,
                mistralrs_core::FILE_PURPOSE_USER_DATA.to_string(),
                mistralrs_core::FileSource {
                    tool: "sdk_input_file".to_string(),
                    round: 0,
                    turn: 0,
                },
                bytes,
            ),
        }
    }
}

impl From<InputFile> for mistralrs_core::File {
    fn from(f: InputFile) -> Self {
        f.file
    }
}
