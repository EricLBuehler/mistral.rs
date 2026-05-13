//! Engine-side dispatch for the `read_file` / `list_files` tools. Schemas live in `mistralrs-code-exec`.

use serde_json::Value;

use crate::files::{File, FileContent, FileStore, READ_FILE_MAX_SLICE_CHARS};
use crate::response::AgenticToolCallData;
use crate::tools::ToolCallResponse;
use crate::{NormalRequest, ToolChoice};

use super::agentic_loop::{append_assistant_tool_call, append_tool_response, get_messages_mut};

pub(super) fn do_read_file(
    mut request: NormalRequest,
    tc: &ToolCallResponse,
    store: &FileStore,
) -> (NormalRequest, AgenticToolCallData, Vec<File>) {
    let messages = get_messages_mut(&mut request);
    append_assistant_tool_call(messages, tc);

    let args: Value = serde_json::from_str(&tc.function.arguments).unwrap_or(Value::Null);
    let file_id = args.get("file_id").and_then(|v| v.as_str()).unwrap_or("");
    let start = args
        .get("start")
        .and_then(|v| v.as_u64())
        .and_then(|v| usize::try_from(v).ok())
        .unwrap_or(0);
    let end = args
        .get("end")
        .and_then(|v| v.as_u64())
        .and_then(|v| usize::try_from(v).ok());

    let response = match store.get(file_id) {
        Some(file) => match &file.content {
            FileContent::Text { text: Some(t), .. } => {
                let total = t.chars().count();
                let requested_end = end.unwrap_or(total).min(total);
                let real_start = start.min(requested_end);
                let real_end = requested_end.min(real_start + READ_FILE_MAX_SLICE_CHARS);
                let slice: String = t
                    .chars()
                    .skip(real_start)
                    .take(real_end - real_start)
                    .collect();
                serde_json::json!({
                    "file_id": file_id,
                    "name": file.name,
                    "format": file.format,
                    "start": real_start,
                    "end": real_start + slice.chars().count(),
                    "total_chars": total,
                    "max_slice_chars": READ_FILE_MAX_SLICE_CHARS,
                    "text": slice,
                })
            }
            FileContent::Text { text: None, .. } => serde_json::json!({
                "file_id": file_id,
                "error": "text body unavailable; refresh via the SDK or fetch endpoint.",
            }),
            FileContent::Binary { .. } => serde_json::json!({
                "file_id": file_id,
                "format": file.format,
                "error": "binary file; cannot read as text. Reference by id when discussing with the user.",
            }),
            FileContent::Error { code, message } => serde_json::json!({
                "file_id": file_id,
                "code": code,
                "error": message,
            }),
        },
        None => serde_json::json!({
            "file_id": file_id,
            "error": "file not found or expired.",
        }),
    }
    .to_string();

    let messages = get_messages_mut(&mut request);
    append_tool_response(messages, &tc.function.name, response.clone());

    request.tool_choice = Some(ToolChoice::Auto);
    (request, custom(response), Vec::new())
}

pub(super) fn do_list_files(
    mut request: NormalRequest,
    tc: &ToolCallResponse,
    store: &FileStore,
    session_id: &str,
) -> (NormalRequest, AgenticToolCallData, Vec<File>) {
    let messages = get_messages_mut(&mut request);
    append_assistant_tool_call(messages, tc);

    let listed = store.list_for_session(session_id);
    let files: Vec<Value> = listed
        .iter()
        .map(|f| {
            serde_json::json!({
                "id": f.id,
                "name": f.name,
                "format": f.format,
                "bytes": f.bytes,
                "round": f.source.round,
                "turn": f.source.turn,
            })
        })
        .collect();
    let response = serde_json::json!({ "files": files }).to_string();

    let messages = get_messages_mut(&mut request);
    append_tool_response(messages, &tc.function.name, response.clone());

    request.tool_choice = Some(ToolChoice::Auto);
    (request, custom(response), Vec::new())
}

fn custom(content: String) -> AgenticToolCallData {
    AgenticToolCallData::Custom {
        arguments: String::new(),
        content,
    }
}
