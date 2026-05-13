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

    if let Some(err) = &tf.error {
        return File {
            id,
            name: tf.name.clone(),
            format,
            mime_type: Some(mime),
            bytes: 0,
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
        source,
        content,
    }
}

/// Append a `Files:` summary to a tool response so the model sees what was produced.
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
            FileContent::Text { text, preview } => {
                if let Some(t) = text {
                    if t.len() <= MODEL_INLINE_BYTES {
                        out.push_str(&format!(
                            "- {} ({}, text, {} bytes, id={}):\n{}\n",
                            f.name, fmt, f.bytes, f.id, t
                        ));
                        continue;
                    }
                }
                let pv = preview.as_deref().unwrap_or("");
                out.push_str(&format!(
                    "- {} ({}, text, {} bytes, id={}, truncated) preview:\n{}\n... call read_file(file_id=\"{}\") for the rest.\n",
                    f.name, fmt, f.bytes, f.id, pv, f.id,
                ));
            }
            FileContent::Binary { .. } => {
                out.push_str(&format!(
                    "- {} ({}, binary, {} bytes, id={}). Reference by id; user fetches via the SDK / files endpoint.\n",
                    f.name, fmt, f.bytes, f.id,
                ));
            }
            FileContent::Error { code, message } => {
                out.push_str(&format!("- {} ({}) failed: [{}] {}\n", f.name, fmt, code, message));
            }
        }
    }
    out
}

/// System message telling the model which files to produce. `None` if there are none.
pub fn system_message_for_required_files(req_files: &[RequestedFile]) -> Option<String> {
    if req_files.is_empty() {
        return None;
    }
    let mut s = String::from(
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
         are NOT surfaced to the user.",
    );
    Some(s)
}

/// Prepend a system message. Appends to an existing leading system message if there is one.
pub fn prepend_system_message(messages: &mut Vec<IndexMap<String, MessageContent>>, text: &str) {
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
                s.push_str("\n\n");
                s.push_str(text);
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

/// Strip inline file bodies (`text` / `data_base64` / `preview`) from a stored tool message. Bodies stay in the `FileStore`.
pub fn compact_tool_message_content(content: &str) -> String {
    let Ok(mut v) = serde_json::from_str::<Value>(content) else {
        return content.to_string();
    };
    if let Some(files) = v.get_mut("files").and_then(|f| f.as_array_mut()) {
        for f in files {
            if let Some(obj) = f.as_object_mut() {
                obj.remove("text");
                obj.remove("data_base64");
                obj.remove("preview");
            }
        }
    }
    serde_json::to_string(&v).unwrap_or_else(|_| content.to_string())
}
