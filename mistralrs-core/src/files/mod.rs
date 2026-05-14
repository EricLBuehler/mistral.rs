//! Typed file outputs from agentic runs.

use std::path::Path;

use base64::Engine;
#[cfg(feature = "pyo3_macros")]
use pyo3::{pyclass, pymethods};
use serde::{Deserialize, Serialize};

mod inject;
mod store;
pub use inject::{
    compose_tool_response_with_files, merge_required_outputs_into_args,
    required_files_tool_addendum, tool_file_to_file,
};
pub use store::{FileStore, DEFAULT_FILE_TTL};

/// Max text bytes shown to the model inline. Above this it gets a preview + id and calls `read_file`.
pub const MODEL_INLINE_BYTES: usize = 1024;

/// Max body size embedded in responses. Above this the file ships reference-only; clients fetch via `GET /v1/files/{id}`.
pub const WIRE_EMBED_LIMIT_BYTES: u64 = 8 * 1024 * 1024;

/// Per-call cap for `read_file`. Larger requests get truncated.
pub const READ_FILE_MAX_SLICE_CHARS: usize = 64 * 1024;

/// Where a file was produced.
#[cfg_attr(feature = "pyo3_macros", pyclass)]
#[cfg_attr(feature = "pyo3_macros", pyo3(get_all))]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileSource {
    pub tool: String,
    /// Zero-based round within the turn.
    pub round: usize,
    /// Zero-based turn within the session. 0 when there is no session.
    #[serde(default)]
    pub turn: usize,
}

/// File body. Serialized untagged so the wire shape is flat.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum FileContent {
    /// `text` is `None` when the body was elided over the wire; preview survives.
    Text {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        text: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        preview: Option<String>,
    },
    /// `data_base64` is `None` when the body was elided over the wire. Fetch by id.
    Binary {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        data_base64: Option<String>,
    },
    /// Placeholder for a required file the model never wrote.
    Error { code: String, message: String },
}

impl FileContent {
    pub fn is_error(&self) -> bool {
        matches!(self, Self::Error { .. })
    }

    pub fn is_text(&self) -> bool {
        matches!(self, Self::Text { .. })
    }

    pub fn is_binary(&self) -> bool {
        matches!(self, Self::Binary { .. })
    }
}

/// A file produced by an agentic run.
#[cfg_attr(feature = "pyo3_macros", pyclass)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct File {
    /// `file_<run>_r<round>_<idx>`.
    pub id: String,
    pub name: String,
    /// `csv`, `json`, `png`, `parquet`, etc. Inferred from the filename extension if not set.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub format: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mime_type: Option<String>,
    pub bytes: u64,
    /// Unix epoch seconds.
    #[serde(default)]
    pub created_at: u64,
    pub source: FileSource,
    #[serde(flatten)]
    pub content: FileContent,
}

impl File {
    pub(crate) fn make_id(run_id: &str, round: usize, idx: usize) -> String {
        format!("file_{run_id}_r{round}_{idx}")
    }

    /// Slice `s` to at most `n` bytes on a UTF-8 char boundary.
    pub(crate) fn truncate_utf8(s: &str, n: usize) -> &str {
        if s.len() <= n {
            return s;
        }
        let mut end = n;
        while end > 0 && !s.is_char_boundary(end) {
            end -= 1;
        }
        &s[..end]
    }

    /// Full text body if present.
    pub fn as_text(&self) -> Option<&str> {
        match &self.content {
            FileContent::Text { text, .. } => text.as_deref(),
            _ => None,
        }
    }

    /// Preview, falling back to the full text.
    pub fn preview_str(&self) -> Option<&str> {
        match &self.content {
            FileContent::Text { preview, text } => preview.as_deref().or(text.as_deref()),
            _ => None,
        }
    }

    /// Base64 body if present.
    pub fn binary_data(&self) -> Option<&str> {
        match &self.content {
            FileContent::Binary { data_base64 } => data_base64.as_deref(),
            _ => None,
        }
    }

    /// True when this `File` is the elided wire form (body stripped, bytes > 0).
    pub fn is_truncated(&self) -> bool {
        match &self.content {
            FileContent::Text { text, .. } => text.is_none() && self.bytes > 0,
            FileContent::Binary { data_base64 } => data_base64.is_none() && self.bytes > 0,
            FileContent::Error { .. } => false,
        }
    }

    /// Clone with the body dropped if it exceeds `WIRE_EMBED_LIMIT_BYTES`. Use before sending over the wire; the full body should already be in the `FileStore`.
    pub fn elide_for_wire(&self) -> File {
        if self.bytes <= WIRE_EMBED_LIMIT_BYTES {
            return self.clone();
        }
        let content = match &self.content {
            FileContent::Text { preview, .. } => FileContent::Text {
                text: None,
                preview: preview.clone(),
            },
            FileContent::Binary { .. } => FileContent::Binary { data_base64: None },
            FileContent::Error { code, message } => FileContent::Error {
                code: code.clone(),
                message: message.clone(),
            },
        };
        File {
            id: self.id.clone(),
            name: self.name.clone(),
            format: self.format.clone(),
            mime_type: self.mime_type.clone(),
            bytes: self.bytes,
            created_at: self.created_at,
            source: self.source.clone(),
            content,
        }
    }

    pub(crate) fn now_unix_secs() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0)
    }

    pub fn is_text(&self) -> bool {
        self.content.is_text()
    }

    pub fn is_binary(&self) -> bool {
        self.content.is_binary()
    }

    pub fn is_error(&self) -> bool {
        self.content.is_error()
    }

    pub fn is_image(&self) -> bool {
        self.mime_type
            .as_deref()
            .is_some_and(|m| m.to_ascii_lowercase().starts_with("image/"))
    }

    pub fn is_video(&self) -> bool {
        self.mime_type
            .as_deref()
            .is_some_and(|m| m.to_ascii_lowercase().starts_with("video/"))
    }

    /// Write the file to `path`. Errors if the body was elided or the file is an error placeholder.
    pub fn save<P: AsRef<Path>>(&self, path: P) -> std::io::Result<()> {
        match &self.content {
            FileContent::Text { text: Some(t), .. } => std::fs::write(path, t),
            FileContent::Binary {
                data_base64: Some(b64),
            } => {
                let bytes = base64::engine::general_purpose::STANDARD
                    .decode(b64)
                    .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
                std::fs::write(path, bytes)
            }
            FileContent::Text { text: None, .. } | FileContent::Binary { data_base64: None } => {
                Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!(
                        "file '{}' body was elided due to wire embed cap; fetch by id first",
                        self.id
                    ),
                ))
            }
            FileContent::Error { code, message } => Err(std::io::Error::other(format!(
                "file '{}' is an error placeholder: {code}: {message}",
                self.id
            ))),
        }
    }
}

#[cfg(feature = "pyo3_macros")]
#[pymethods]
impl File {
    #[getter]
    fn id(&self) -> &str {
        &self.id
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
    fn mime_type(&self) -> Option<&str> {
        self.mime_type.as_deref()
    }

    #[getter]
    fn bytes(&self) -> u64 {
        self.bytes
    }

    #[getter]
    fn source(&self) -> FileSource {
        self.source.clone()
    }

    #[getter]
    fn text(&self) -> Option<&str> {
        self.as_text()
    }

    #[getter]
    fn data_base64(&self) -> Option<&str> {
        self.binary_data()
    }

    #[getter]
    fn preview(&self) -> Option<&str> {
        self.preview_str()
    }

    #[pyo3(name = "is_text")]
    fn py_is_text(&self) -> bool {
        self.is_text()
    }

    #[pyo3(name = "is_binary")]
    fn py_is_binary(&self) -> bool {
        self.is_binary()
    }

    #[pyo3(name = "is_error")]
    fn py_is_error(&self) -> bool {
        self.is_error()
    }

    #[pyo3(name = "is_image")]
    fn py_is_image(&self) -> bool {
        self.is_image()
    }

    #[pyo3(name = "is_video")]
    fn py_is_video(&self) -> bool {
        self.is_video()
    }

    #[pyo3(name = "is_truncated")]
    fn py_is_truncated(&self) -> bool {
        self.is_truncated()
    }

    #[pyo3(name = "save")]
    fn py_save(&self, path: &str) -> pyo3::PyResult<()> {
        self.save(path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("failed to save file: {e}")))
    }

    fn __repr__(&self) -> String {
        format!("{self:#?}")
    }
}

/// A file the runtime asks the model to produce. Surfaces as `File` (or `FileContent::Error` if missing).
#[cfg_attr(feature = "pyo3_macros", pyclass)]
#[cfg_attr(feature = "pyo3_macros", pyo3(get_all))]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestedFile {
    pub name: String,
    /// Inferred from `name`'s extension if not set.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub format: Option<String>,
    /// Surfaced to the model in the system message.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

impl RequestedFile {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            format: None,
            description: None,
        }
    }

    pub fn with_format(mut self, format: impl Into<String>) -> Self {
        self.format = Some(format.into());
        self
    }

    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }
}

/// Lowercase extension without the leading dot, or `None` if there's no extension.
pub fn format_from_name(name: &str) -> Option<String> {
    name.rsplit_once('.')
        .map(|(_, ext)| ext.to_ascii_lowercase())
}

/// Mime type for a format string. Falls back to `application/octet-stream`.
pub fn mime_for_format(format: &str) -> String {
    let ext = match format.to_ascii_lowercase().as_str() {
        "markdown" => "md".to_string(),
        "yml" => "yaml".to_string(),
        "latex" | "tex" => "tex".to_string(),
        "python" => "py".to_string(),
        "rust" => "rs".to_string(),
        "vega-lite" | "vega" | "geojson" => "json".to_string(),
        "text" => "txt".to_string(),
        other => other.to_string(),
    };
    mime_guess::from_ext(&ext)
        .first_or_octet_stream()
        .essence_str()
        .to_string()
}

/// True for mime types we can ship as utf-8 text.
pub fn is_text_mime(mime: &str) -> bool {
    let m = mime.to_ascii_lowercase();
    m.starts_with("text/")
        || matches!(
            m.as_str(),
            "application/json"
                | "application/geo+json"
                | "application/xml"
                | "application/yaml"
                | "application/x-yaml"
                | "application/toml"
                | "application/sql"
                | "application/x-tex"
                | "image/svg+xml"
        )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn text_file(id: &str, body: &str) -> File {
        File {
            id: id.into(),
            name: "x.txt".into(),
            format: Some("txt".into()),
            mime_type: Some("text/plain".into()),
            bytes: body.len() as u64,
            created_at: 0,
            source: FileSource {
                tool: "execute_python".into(),
                round: 0,
                turn: 0,
            },
            content: FileContent::Text {
                text: Some(body.into()),
                preview: None,
            },
        }
    }

    #[test]
    fn id_format() {
        assert_eq!(File::make_id("abc", 0, 0), "file_abc_r0_0");
        assert_eq!(File::make_id("xyz", 3, 7), "file_xyz_r3_7");
    }

    #[test]
    fn truncate_respects_utf8() {
        let s = "héllo";
        // "héllo" is 6 bytes (h=1, é=2, l=1, l=1, o=1).
        assert_eq!(File::truncate_utf8(s, 10), "héllo");
        // Truncating to 2 bytes would split é; should back off to 1 byte.
        assert_eq!(File::truncate_utf8(s, 2), "h");
        assert_eq!(File::truncate_utf8(s, 3), "hé");
    }

    #[test]
    fn text_accessors() {
        let f = text_file("file_x_r0_0", "hello");
        assert_eq!(f.as_text(), Some("hello"));
        assert_eq!(f.preview_str(), Some("hello"));
        assert!(f.binary_data().is_none());
        assert!(!f.is_truncated());
    }

    #[test]
    fn truncated_binary_is_flagged() {
        let f = File {
            id: "file_x_r0_0".into(),
            name: "big.bin".into(),
            format: Some("bin".into()),
            mime_type: Some("application/octet-stream".into()),
            bytes: 64 * 1024 * 1024,
            created_at: 0,
            source: FileSource {
                tool: "execute_python".into(),
                round: 0,
                turn: 0,
            },
            content: FileContent::Binary { data_base64: None },
        };
        assert!(f.is_truncated());
    }

    #[test]
    fn format_from_name_extension() {
        assert_eq!(format_from_name("plot.png"), Some("png".into()));
        assert_eq!(format_from_name("data.csv"), Some("csv".into()));
        assert_eq!(format_from_name("report.MD"), Some("md".into()));
        assert_eq!(format_from_name("noext"), None);
    }

    #[test]
    fn mime_lookup() {
        assert_eq!(mime_for_format("csv"), "text/csv");
        assert_eq!(mime_for_format("PNG"), "image/png");
        assert_eq!(mime_for_format("markdown"), "text/markdown");
        assert_eq!(mime_for_format("geojson"), "application/json");
        assert_eq!(mime_for_format("unknown_"), "application/octet-stream");
    }

    #[test]
    fn text_mime_classifier() {
        assert!(is_text_mime("text/csv"));
        assert!(is_text_mime("application/json"));
        assert!(is_text_mime("image/svg+xml"));
        assert!(!is_text_mime("image/png"));
        assert!(!is_text_mime("application/octet-stream"));
    }
}
