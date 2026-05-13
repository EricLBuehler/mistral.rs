//! Helpers for surfacing `File`s to the model and compacting them for session storage.

use either::Either;
use indexmap::IndexMap;
use mistralrs_mcp::ToolFile;
use serde_json::Value;

use crate::tools::ToolCallResponse;
use crate::MessageContent;

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

pub const REQUIRED_FILES_BLOCK_OPEN: &str = "<<<mistralrs:required_files>>>";
pub const REQUIRED_FILES_BLOCK_CLOSE: &str = "<<</mistralrs:required_files>>>";

/// System message telling the model which files to produce. `None` if there are none.
pub fn system_message_for_required_files(req_files: &[RequestedFile]) -> Option<String> {
    if req_files.is_empty() {
        return None;
    }
    let mut s = String::from(REQUIRED_FILES_BLOCK_OPEN);
    s.push('\n');
    s.push_str(
        "The runtime requires these output files for this request. Use code execution to write \
         each one to the working directory and list it in the `outputs` parameter of \
         `mistralrs_execute_python`.\n\nRequired outputs:\n",
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
        "\nFiles you produce but do NOT list in `outputs` remain in the working directory and \
         are NOT surfaced to the user.\n",
    );
    s.push_str(REQUIRED_FILES_BLOCK_CLOSE);
    Some(s)
}

pub fn strip_required_files_block(s: &str) -> String {
    let (Some(open), Some(close)) = (
        s.rfind(REQUIRED_FILES_BLOCK_OPEN),
        s.rfind(REQUIRED_FILES_BLOCK_CLOSE),
    ) else {
        return s.to_string();
    };
    if open >= close {
        return s.to_string();
    }
    let head = s[..open].trim_end_matches(['\n', ' ']).to_string();
    let tail = &s[close + REQUIRED_FILES_BLOCK_CLOSE.len()..];
    let tail = tail.trim_start_matches('\n');
    if head.is_empty() {
        tail.to_string()
    } else if tail.is_empty() {
        head
    } else {
        format!("{head}\n\n{tail}")
    }
}

/// Replaces any prior block bracketed by `REQUIRED_FILES_BLOCK_*` rather than stacking.
pub fn prepend_required_files_message(
    messages: &mut Vec<IndexMap<String, MessageContent>>,
    text: &str,
) {
    if let Some(first) = messages.first_mut() {
        let role = first
            .get("role")
            .and_then(|r| match r {
                Either::Left(s) => Some(s.as_str()),
                _ => None,
            })
            .unwrap_or("");
        if role == "system" {
            if let Some(Either::Left(s)) = first.get_mut("content") {
                let stripped = strip_required_files_block(s);
                *s = if stripped.is_empty() {
                    text.to_string()
                } else {
                    format!("{stripped}\n\n{text}")
                };
                return;
            }
        }
    }
    let mut sys = IndexMap::new();
    sys.insert("role".to_string(), Either::Left("system".to_string()));
    sys.insert("content".to_string(), Either::Left(text.to_string()));
    messages.insert(0, sys);
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
    fn required_files_block_is_replaceable() {
        let mut messages: Vec<IndexMap<String, MessageContent>> = vec![IndexMap::from([
            ("role".to_string(), Either::Left("system".to_string())),
            (
                "content".to_string(),
                Either::Left("You are helpful.".to_string()),
            ),
        ])];
        let req_a = vec![RequestedFile::new("a.csv")];
        let req_b = vec![RequestedFile::new("b.csv")];
        let block_a = system_message_for_required_files(&req_a).unwrap();
        let block_b = system_message_for_required_files(&req_b).unwrap();

        prepend_required_files_message(&mut messages, &block_a);
        prepend_required_files_message(&mut messages, &block_b);

        let Either::Left(content) = messages[0].get("content").unwrap() else {
            panic!()
        };
        assert!(content.contains("You are helpful."));
        assert!(content.contains("b.csv"));
        assert!(
            !content.contains("a.csv"),
            "block A should have been replaced, got: {content}"
        );
    }

    #[test]
    fn strip_required_files_leaves_user_text() {
        let req = vec![RequestedFile::new("a.csv")];
        let block = system_message_for_required_files(&req).unwrap();
        let combined = format!("You are helpful.\n\n{block}");
        let stripped = strip_required_files_block(&combined);
        assert_eq!(stripped, "You are helpful.");
    }
}
