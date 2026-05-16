//! Helpers for surfacing `File`s to the model and adding the required-files contract to the code-exec tool.

use mistralrs_mcp::ToolFile;
use serde_json::Value;

use crate::tools::ToolCallResponse;

use super::{
    format_from_name, mime_for_format, File, FileContent, FileSource, RequestedFile,
    MODEL_INLINE_BYTES,
};

/// Convert a `ToolFile` to a `File` with full body. Elision happens later via `File::elide_for_wire`.
pub fn tool_file_to_file(
    tf: &ToolFile,
    run_id: &str,
    round: usize,
    turn: usize,
    idx: usize,
    tool_name: &str,
) -> File {
    let id = File::make_id(run_id, round, idx);
    let format = if tf.format.is_empty() {
        format_from_name(&tf.name)
    } else {
        Some(tf.format.clone())
    };
    let mime = tf
        .mime_type
        .clone()
        .or_else(|| format.as_deref().map(mime_for_format))
        .unwrap_or_else(|| "application/octet-stream".to_string());
    let source = FileSource {
        tool: tool_name.to_string(),
        round,
        turn,
    };

    let created_at = File::now_unix_secs();

    if let Some(err) = &tf.error {
        return File {
            id,
            name: tf.name.clone(),
            format,
            mime_type: Some(mime),
            bytes: 0,
            created_at,
            source,
            content: FileContent::Error {
                code: "not_produced".to_string(),
                message: err.clone(),
            },
        };
    }

    let content = if let Some(text) = &tf.text {
        FileContent::Text {
            text: Some(text.clone()),
            preview: Some(File::truncate_utf8(text, MODEL_INLINE_BYTES).to_string()),
        }
    } else if let Some(b64) = &tf.data_base64 {
        FileContent::Binary {
            data_base64: Some(b64.clone()),
        }
    } else {
        FileContent::Binary { data_base64: None }
    };

    File {
        id,
        name: tf.name.clone(),
        format,
        mime_type: Some(mime),
        bytes: tf.size_bytes,
        created_at,
        source,
        content,
    }
}

/// Append a metadata-only `Files:` summary. Bodies are never inlined; the model fetches via `read_file(id)`.
pub fn compose_tool_response_with_files(raw: &str, files: &[File]) -> String {
    if files.is_empty() {
        return raw.to_string();
    }
    let mut out = raw.trim_end().to_string();
    if !out.is_empty() {
        out.push('\n');
    }
    out.push_str("Files:\n");
    for f in files {
        let fmt = f.format.as_deref().unwrap_or("");
        match &f.content {
            FileContent::Text { .. } => {
                out.push_str(&format!(
                    "- {} ({}, text, {} bytes, id={}). Use read_file(file_id=\"{}\") to read.\n",
                    f.name, fmt, f.bytes, f.id, f.id,
                ));
            }
            FileContent::Binary { .. } => {
                out.push_str(&format!(
                    "- {} ({}, binary, {} bytes, id={}). User fetches via the SDK / files endpoint.\n",
                    f.name, fmt, f.bytes, f.id,
                ));
            }
            FileContent::Error { code, message } => {
                out.push_str(&format!(
                    "- {} ({}) failed: [{}] {}\n",
                    f.name, fmt, code, message
                ));
            }
        }
    }
    out
}

/// Text appended to the code-execution tool's `description` so the model sees the required-files contract. `None` if no required files.
pub fn required_files_tool_addendum(req_files: &[RequestedFile]) -> Option<String> {
    if req_files.is_empty() {
        return None;
    }
    let mut s = String::from(
        "\n\nThe runtime requires these output files for this request. Write each one to the \
         working directory and list it in the `outputs` parameter.\n\nRequired outputs:\n",
    );
    for r in req_files {
        let fmt = r
            .format
            .clone()
            .or_else(|| format_from_name(&r.name))
            .unwrap_or_else(|| "any".to_string());
        match &r.description {
            Some(d) => s.push_str(&format!("- {} ({}): {}\n", r.name, fmt, d)),
            None => s.push_str(&format!("- {} ({})\n", r.name, fmt)),
        }
    }
    s.push_str(
        "\nFiles you produce but do NOT list in `outputs` remain in the working directory \
         and are NOT surfaced to the user.",
    );
    Some(s)
}

/// Merge required files into the tool call's `outputs` arg so the executor reads them even if the model omitted them.
pub fn merge_required_outputs_into_args(
    tc: &ToolCallResponse,
    required: &[RequestedFile],
) -> ToolCallResponse {
    use std::collections::HashSet;
    let mut owned = tc.clone();
    let mut args: Value =
        serde_json::from_str(&owned.function.arguments).unwrap_or_else(|_| serde_json::json!({}));
    let Some(obj) = args.as_object_mut() else {
        return owned;
    };
    let arr = obj
        .entry("outputs".to_string())
        .or_insert_with(|| Value::Array(Vec::new()));
    let Some(arr) = arr.as_array_mut() else {
        return owned;
    };
    let existing: HashSet<String> = arr
        .iter()
        .filter_map(|v| v.get("name").and_then(|n| n.as_str()).map(String::from))
        .collect();
    for r in required {
        if !existing.contains(&r.name) {
            let mut entry = serde_json::Map::new();
            entry.insert("name".into(), Value::String(r.name.clone()));
            if let Some(fmt) = &r.format {
                entry.insert("format".into(), Value::String(fmt.clone()));
            }
            arr.push(Value::Object(entry));
        }
    }
    owned.function.arguments = serde_json::to_string(&args).unwrap_or(owned.function.arguments);
    owned
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::files::FileSource;

    fn text(id: &str, body: &str) -> File {
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
                preview: Some(body.into()),
            },
        }
    }

    #[test]
    fn compose_renders_metadata_only() {
        let raw = r#"{"status":"success","stdout":"hi"}"#;
        let composed = compose_tool_response_with_files(raw, &[text("file_a", "hello world")]);
        assert!(composed.contains("file_a"));
        assert!(composed.contains("read_file"));
        assert!(!composed.contains("hello world"));
    }

    #[test]
    fn addendum_lists_files() {
        let req = vec![RequestedFile::new("a.csv"), RequestedFile::new("plot.png")];
        let s = required_files_tool_addendum(&req).unwrap();
        assert!(s.contains("a.csv"));
        assert!(s.contains("plot.png"));
        assert!(s.contains("outputs"));
    }

    #[test]
    fn addendum_none_when_empty() {
        assert!(required_files_tool_addendum(&[]).is_none());
    }
}
