//! First-class File outputs from agentic runs.
//!
//! A `File` is a typed output produced during a chat completion that ran
//! tools (typically code execution). Files exist independently of the
//! model's transcript: the app collects them by id, the model references
//! them by id, large bodies are reachable via the artifacts endpoint
//! without bloating the SSE stream.

use std::path::Path;

use base64::Engine;
#[cfg(feature = "pyo3_macros")]
use pyo3::{pyclass, pymethods};
use serde::{Deserialize, Serialize};

mod inject;
mod store;
pub use inject::{
    compact_tool_message_content, compose_tool_response_with_files,
    merge_required_outputs_into_args, prepend_system_message, system_message_for_required_files,
    tool_file_to_file,
};
pub use store::{FileStore, DEFAULT_FILE_TTL};

// ---------------------------------------------------------------------------
// Limits
// ---------------------------------------------------------------------------

/// Text shown to the model inline in tool responses up to this many bytes.
/// Above this, the model gets a preview plus an id and uses `read_file` to
/// read the rest. Sliced on a UTF-8 char boundary.
pub const MODEL_INLINE_BYTES: usize = 1024;

/// Maximum body size to embed directly in responses leaving
/// `mistralrs-core` (text in `text`, binary in `data_base64`). Above this
/// the file ships reference-only (`url` + metadata); clients fetch via
/// `GET /v1/files/{id}`.
pub const WIRE_EMBED_LIMIT_BYTES: u64 = 32 * 1024 * 1024;

// ---------------------------------------------------------------------------
// Source attribution
// ---------------------------------------------------------------------------

/// Where a file was produced.
#[cfg_attr(feature = "pyo3_macros", pyclass)]
#[cfg_attr(feature = "pyo3_macros", pyo3(get_all))]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileSource {
    /// Tool name that produced this file (e.g. `mistralrs_execute_python`).
    pub tool: String,
    /// Agentic loop round (zero-based) within this turn.
    pub round: usize,
    /// Conversation turn (zero-based) within the session. `0` if no
    /// session is in use.
    #[serde(default)]
    pub turn: usize,
}

// ---------------------------------------------------------------------------
// File body
// ---------------------------------------------------------------------------

/// Body of a [`File`]. Serializes untagged so the wire shape is flat:
/// `{"text": "..."}`, `{"data_base64": "..."}`, `{"error": {...}}`.
///
/// `Text` and `Binary` may have `None` bodies when the original content
/// exceeded [`WIRE_EMBED_LIMIT_BYTES`] — fetch via the artifacts endpoint
/// to retrieve the bytes.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum FileContent {
    /// Text file. `text` is the full content (or `None` if elided).
    /// `preview` is a UTF-8-safe slice up to [`MODEL_INLINE_BYTES`] for
    /// the model's context.
    Text {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        text: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        preview: Option<String>,
    },
    /// Binary file (images, videos, archives, parquet, anything that
    /// isn't utf-8 readable). `mime_type` on the parent `File`
    /// disambiguates — use [`File::is_image`] / [`File::is_video`] to
    /// branch. `data_base64` is `None` when the body exceeded the wire
    /// embed cap; fetch by id in that case.
    Binary {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        data_base64: Option<String>,
    },
    /// Placeholder for a file that was required but failed to materialize
    /// (e.g. the user requested it via `request.files` and the model
    /// didn't write it, or the model declared it in `outputs` and didn't
    /// write it).
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

// ---------------------------------------------------------------------------
// File
// ---------------------------------------------------------------------------

/// First-class output from an agentic run.
#[cfg_attr(feature = "pyo3_macros", pyclass)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct File {
    /// Stable id; `file_<run>_r<round>_<idx>` format.
    pub id: String,
    /// Filename as written to the working directory (or as declared by
    /// the producer).
    pub name: String,
    /// Open-ended format string: `csv`, `json`, `png`, `parquet`, etc.
    /// Inferred from the filename extension if not set explicitly.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub format: Option<String>,
    /// Content-Type, e.g. `text/csv`, `image/png`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mime_type: Option<String>,
    /// Size of the underlying body in bytes.
    pub bytes: u64,
    /// Where this file came from.
    pub source: FileSource,
    /// Body and content-aware fields.
    #[serde(flatten)]
    pub content: FileContent,
}

impl File {
    /// Format a deterministic file id. Stable within a run; unique across
    /// runs via `run_id`. Engine-internal — clients receive ids in
    /// responses.
    pub(crate) fn make_id(run_id: &str, round: usize, idx: usize) -> String {
        format!("file_{run_id}_r{round}_{idx}")
    }

    /// Slice a string on a UTF-8 char boundary, returning at most `n`
    /// bytes. Always returns valid UTF-8. Engine-internal helper.
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

    /// Full text body, if this is a text file with the body present.
    pub fn as_text(&self) -> Option<&str> {
        match &self.content {
            FileContent::Text { text, .. } => text.as_deref(),
            _ => None,
        }
    }

    /// Preview, falling back to the full text if present, else None.
    pub fn preview_str(&self) -> Option<&str> {
        match &self.content {
            FileContent::Text { preview, text } => preview.as_deref().or(text.as_deref()),
            _ => None,
        }
    }

    /// Base64 body, if this is a binary file with the body present.
    pub fn binary_data(&self) -> Option<&str> {
        match &self.content {
            FileContent::Binary { data_base64 } => data_base64.as_deref(),
            _ => None,
        }
    }

    /// True when the body would be elided on the wire (size exceeds
    /// [`WIRE_EMBED_LIMIT_BYTES`]). The body is still present in this
    /// `File` if it was loaded from the store; this flag tells clients
    /// whether they're seeing the elided wire form.
    pub fn is_truncated(&self) -> bool {
        match &self.content {
            FileContent::Text { text, .. } => text.is_none() && self.bytes > 0,
            FileContent::Binary { data_base64 } => data_base64.is_none() && self.bytes > 0,
            FileContent::Error { .. } => false,
        }
    }

    /// Returns a clone with the body elided if it exceeds
    /// [`WIRE_EMBED_LIMIT_BYTES`]. The preview survives. Use before
    /// emitting on the SSE stream or embedding in a response payload.
    /// The original (un-elided) `File` should live in the `FileStore`
    /// so `GET /v1/files/{id}/content` and SDK fetch can recover the
    /// bytes.
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
            source: self.source.clone(),
            content,
        }
    }

    /// Whether the body is text content.
    pub fn is_text(&self) -> bool {
        self.content.is_text()
    }

    /// Whether the body is binary content.
    pub fn is_binary(&self) -> bool {
        self.content.is_binary()
    }

    /// Whether this file is an error placeholder.
    pub fn is_error(&self) -> bool {
        self.content.is_error()
    }

    /// Whether the mime type indicates an image.
    pub fn is_image(&self) -> bool {
        self.mime_type
            .as_deref()
            .is_some_and(|m| m.to_ascii_lowercase().starts_with("image/"))
    }

    /// Whether the mime type indicates a video.
    pub fn is_video(&self) -> bool {
        self.mime_type
            .as_deref()
            .is_some_and(|m| m.to_ascii_lowercase().starts_with("video/"))
    }

    /// Persist this file to disk. Writes text bodies directly and decodes
    /// base64 for binary bodies. Errors when the body has been elided due
    /// to the wire embed cap (fetch by id first) or when the file is an
    /// error placeholder.
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
            FileContent::Error { code, message } => Err(std::io::Error::new(
                std::io::ErrorKind::Other,
                format!(
                    "file '{}' is an error placeholder: {code}: {message}",
                    self.id
                ),
            )),
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

// ---------------------------------------------------------------------------
// Request-side spec
// ---------------------------------------------------------------------------

/// A required output file declared on the request. The runtime tells the
/// model about these; if the file exists in the working directory after a
/// tool call, it surfaces as a [`File`] regardless of whether the model
/// listed it in its tool's `outputs` parameter. If missing, surfaces as a
/// [`File`] with [`FileContent::Error`].
#[cfg_attr(feature = "pyo3_macros", pyclass)]
#[cfg_attr(feature = "pyo3_macros", pyo3(get_all))]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestedFile {
    pub name: String,
    /// Optional format hint. When omitted, the runtime infers from the
    /// filename extension.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub format: Option<String>,
    /// Optional description to help the model choose what to put in this
    /// file. Surfaced in the system message contract.
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

// ---------------------------------------------------------------------------
// Format / mime helpers
// ---------------------------------------------------------------------------

/// Infer a format string from a filename. Returns the lowercase extension
/// without the leading dot, or `None` if there's no extension.
pub fn format_from_name(name: &str) -> Option<String> {
    name.rsplit_once('.')
        .map(|(_, ext)| ext.to_ascii_lowercase())
}

/// Look up a mime type from an open-ended format string. Falls back to
/// `application/octet-stream` for unknown formats. A handful of aliases
/// (`markdown` → `md`, `geojson` → `json`, etc.) are normalized first.
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

/// Whether a mime type is text-on-the-wire (utf-8 readable).
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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

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
