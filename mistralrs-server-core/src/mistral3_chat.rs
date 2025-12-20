//! Utilities for working with Devstral/Mistral3-style chat templates.
//!
//! Devstral's Jinja chat template enforces strict alternation of *counted* roles:
//! - Optional first `system` message.
//! - After that, counted roles must alternate `user`, `assistant`, `user`, `assistant`, ...
//! - Tool calls and tool results do not count toward alternation.
//!
//! Some clients (notably Codex CLI) may send multiple consecutive user messages for harness/context,
//! or include non-standard roles like `developer`. Without canonicalization, the template will raise
//! and requests will fail before generation starts.

use std::collections::HashMap;
use std::env;

use either::Either;
use serde_json::Value;
use tracing::warn;

use crate::openai::{FunctionCalled, Message, MessageContent, MessageInnerContent, ToolCall};

const TOOL_CALL_JSON_HINT: &str = r#"Tool-call formatting requirements (important):
- If you emit Codex-style tool calls like `[TOOL_CALLS]tool_name[ARGS]{...}`, the `{...}` MUST be valid JSON.
- Do NOT include unescaped `"` inside JSON strings. Prefer single quotes inside shell snippets (e.g. `-name '*.md'`).
- Only output the tool call when calling a tool; do not include additional prose inside tool arguments (e.g. `workdir`)."#;

fn looks_like_codex_tool_call_text(text: &str) -> bool {
    // Codex CLI often uses a text protocol for tool calls in model output, e.g.
    //   [TOOL_CALLS]shell {"command":[...], ...}
    // or
    //   [TOOL_CALLS]shell[ARGS]{...}
    // When this is present in an assistant message, Devstral's chat template expects it to be
    // treated as a tool-call message (i.e. excluded from the strict user/assistant alternation).
    text.contains("[TOOL_CALLS]")
}

fn message_content_to_text_lossy(content: &MessageContent) -> String {
    match &**content {
        Either::Left(s) => s.clone(),
        Either::Right(parts) => parts
            .iter()
            .filter_map(|p| {
                let tp = p.get("type")?.0.as_ref().left()?.as_str();
                if tp != "text" {
                    return None;
                }
                p.get("text")?.0.as_ref().left().cloned()
            })
            .collect::<Vec<_>>()
            .join("\n"),
    }
}

fn parse_codex_tool_calls_from_text(text: &str) -> Option<Vec<ToolCall>> {
    if !looks_like_codex_tool_call_text(text) {
        return None;
    }
    // If there is no JSON delimiter yet, treat as plain text to avoid blocking streaming.
    if !text.contains('{') && !text.contains('[') {
        return None;
    }

    // Repair common invalid JSON produced by agentic prompts when embedding shell snippets inside
    // JSON strings (unescaped `"` inside e.g. `find ... -name "*.md"`).
    //
    // This is intentionally conservative and only targets a small set of common `find`/`rg` flags.
    fn repair_codex_shell_unescaped_quotes(raw: &str) -> Option<String> {
        const FLAGS: [&str; 4] = ["-name \"", "-path \"", "-glob \"", "-g \""];
        let mut out = String::with_capacity(raw.len());
        let mut i = 0;
        let bytes = raw.as_bytes();

        while i < bytes.len() {
            let tail = &raw[i..];
            let mut matched: Option<&'static str> = None;
            for flag in FLAGS {
                if tail.starts_with(flag) {
                    matched = Some(flag);
                    break;
                }
            }

            if let Some(flag) = matched {
                // Copy the flag prefix without the opening quote, and use a single quote instead.
                out.push_str(&flag[..flag.len() - 1]); // drop the `"`
                out.push('\'');
                i += flag.len();

                // Copy until the next `"`, replacing it with `'`.
                if let Some(end_rel) = raw[i..].find('"') {
                    out.push_str(&raw[i..i + end_rel]);
                    out.push('\'');
                    i += end_rel + 1;
                    continue;
                } else {
                    // No closing quote; give up.
                    return None;
                }
            }

            // Default: copy one byte.
            out.push(bytes[i] as char);
            i += 1;
        }

        (out != raw).then_some(out)
    }

    fn sanitize_shell_args(arguments: &mut Value) {
        let Value::Object(map) = arguments else {
            return;
        };

        // Normalize `command` into the expected array form if the model emitted it as a string.
        if let Some(Value::String(cmd)) = map.get("command").cloned() {
            let cmd = cmd.trim().to_string();
            if !cmd.is_empty() {
                map.insert(
                    "command".to_string(),
                    Value::Array(vec![
                        Value::String("bash".to_string()),
                        Value::String("-lc".to_string()),
                        Value::String(cmd),
                    ]),
                );
            }
        }

        if !map.contains_key("workdir") {
            map.insert("workdir".to_string(), Value::String(".".to_string()));
        }
        let Some(Value::String(workdir)) = map.get("workdir").cloned() else {
            return;
        };
        let mut wd = workdir;
        for sentinel in ["\n", "\r", "\t", "[TOOL_CALLS]", "{", "}", "  "] {
            if let Some(pos) = wd.find(sentinel) {
                wd.truncate(pos);
            }
        }
        wd = wd.trim().to_string();
        if wd.is_empty() || wd.contains(' ') {
            map.insert("workdir".to_string(), Value::String(".".to_string()));
        } else {
            map.insert("workdir".to_string(), Value::String(wd));
        }
    }

    let mut rest = text;
    let mut calls = Vec::new();
    loop {
        let Some(tool_prefix_pos) = rest.find("[TOOL_CALLS]") else {
            break;
        };
        let after_prefix = &rest[tool_prefix_pos + "[TOOL_CALLS]".len()..];
        let after_prefix = after_prefix.trim_start();

        let json_start = after_prefix.find('{').or_else(|| after_prefix.find('['));
        let Some(json_start) = json_start else {
            break;
        };

        // Everything before the first JSON delimiter is the raw name region; Codex-style outputs
        // sometimes repeat `[ARGS]` tokens (e.g. `shell[ARGS]update_plan[ARGS]{...}`). Use the last
        // segment before JSON as the function name.
        let pre_json = &after_prefix[..json_start];
        let mut name = pre_json;
        if pre_json.contains("[ARGS]") {
            if let Some(seg) = pre_json.rsplit("[ARGS]").find(|s| !s.trim().is_empty()) {
                name = seg;
            }
        }
        let name = name.trim();

        let json_src = after_prefix[json_start..].trim_start();

        if name.is_empty() {
            break;
        }

        let try_parse_first_value = |src: &str| -> Option<(Value, usize)> {
            let mut stream = serde_json::Deserializer::from_str(src).into_iter::<Value>();
            let first = stream.next()?;
            match first {
                Ok(v) => Some((v, stream.byte_offset())),
                Err(_) => None,
            }
        };

        let mut parsed = try_parse_first_value(json_src);
        if parsed.is_none() {
            if let Some(repaired) = repair_codex_shell_unescaped_quotes(json_src) {
                parsed = try_parse_first_value(&repaired);
            }
        }
        let Some((mut arguments, offset)) = parsed else {
            if env::var("MISTRALRS_DEBUG_TOOL_CALL_PARSING").is_ok() {
                warn!("codex_tool_call_parse_failed assistant_text={:?}", text);
            }
            return None;
        };
        if name == "shell" {
            sanitize_shell_args(&mut arguments);
        }

        let parameters = serde_json::to_string(&arguments).ok()?;
        calls.push(ToolCall {
            id: None,
            tp: mistralrs_core::ToolType::Function,
            function: FunctionCalled {
                name: name.to_string(),
                arguments: parameters,
            },
        });

        rest = &json_src[offset..];
        rest = rest.trim_start();
    }

    if calls.is_empty() {
        None
    } else {
        Some(calls)
    }
}

fn is_counted_for_mistral3_template(msg: &Message) -> bool {
    if msg.role == "user" {
        return true;
    }
    if msg.role != "assistant" {
        return false;
    }
    // If this is an assistant tool message (wrong role, but some clients get this wrong),
    // never count it towards alternation.
    if msg.tool_call_id.is_some() {
        return false;
    }
    // If the assistant message looks like a tool-call marker encoded in plain text, treat it as
    // a tool call (not counted for alternation).
    if msg.tool_calls.is_none() {
        if let Some(content) = msg.content.as_ref() {
            let text = message_content_to_text_lossy(content);
            if looks_like_codex_tool_call_text(&text) {
                return false;
            }
        }
    }
    msg.tool_calls
        .as_ref()
        .map(|tc| tc.is_empty())
        .unwrap_or(true)
}

fn message_text_for_merge(msg: &Message) -> String {
    msg.content
        .as_ref()
        .map(message_content_to_text_lossy)
        .unwrap_or_default()
}

fn message_content_text_part(text: String) -> HashMap<String, MessageInnerContent> {
    let mut map = HashMap::new();
    map.insert(
        "type".to_string(),
        MessageInnerContent(Either::Left("text".to_string())),
    );
    map.insert("text".to_string(), MessageInnerContent(Either::Left(text)));
    map
}

fn merge_message_content(
    existing: Option<MessageContent>,
    added: Option<MessageContent>,
) -> Option<MessageContent> {
    match (existing, added) {
        (None, None) => None,
        (Some(c), None) => Some(c),
        (None, Some(c)) => Some(c),
        (Some(existing), Some(added)) => match (&*existing, &*added) {
            (Either::Left(a), Either::Left(b)) => {
                let merged = if a.is_empty() {
                    b.clone()
                } else if b.is_empty() {
                    a.clone()
                } else {
                    format!("{a}\n\n{b}")
                };
                Some(MessageContent::from_text(merged))
            }
            (Either::Right(a_parts), Either::Right(b_parts)) => {
                let mut merged = a_parts.clone();
                merged.extend(b_parts.clone());
                Some(MessageContent::from_parts(merged))
            }
            (Either::Left(a), Either::Right(b_parts)) => {
                let mut merged = Vec::with_capacity(1 + b_parts.len());
                merged.push(message_content_text_part(a.clone()));
                merged.extend(b_parts.clone());
                Some(MessageContent::from_parts(merged))
            }
            (Either::Right(a_parts), Either::Left(b)) => {
                let mut merged = a_parts.clone();
                merged.push(message_content_text_part(b.clone()));
                Some(MessageContent::from_parts(merged))
            }
        },
    }
}

pub fn looks_like_codex_preamble(text: &str) -> bool {
    // Heuristic: Codex CLI often sends a large "agent harness" preamble as a `user` message
    // (sometimes as multiple input_text parts), followed by the actual user prompt in the last part.
    // Treating the preamble as `system` dramatically improves instruction-following and reduces
    // "echo the harness" failures.
    let t = text;
    t.contains("Codex CLI")
        || t.contains("OpenAI Codex")
        || t.contains("<environment_context>")
        || t.contains("# AGENTS.md instructions")
        || t.contains("approval_policy")
        || t.contains("sandbox_mode")
}

pub fn looks_like_codex_context_message(text: &str) -> bool {
    let t = text.trim_start();
    t.starts_with("<environment_context>") || t.starts_with("# AGENTS.md instructions")
}

fn violates_mistral3_alternation(messages: &[Message]) -> bool {
    let mut idx = 0usize;
    let mut started = false;
    let start = messages.first().is_some_and(|m| m.role == "system") as usize;

    for msg in messages.iter().skip(start) {
        if !is_counted_for_mistral3_template(msg) {
            continue;
        }
        started = true;
        let expect_user = idx % 2 == 0;
        let is_user = msg.role == "user";
        if is_user != expect_user {
            return true;
        }
        idx += 1;
    }

    // If there are no counted messages at all, no alternation constraint to violate.
    !started
}

fn enforce_mistral3_alternation_by_insertion(mut messages: Vec<Message>) -> Vec<Message> {
    if messages.is_empty() {
        return messages;
    }

    let system_prefix_len = messages.iter().take_while(|m| m.role == "system").count();
    let mut out: Vec<Message> = Vec::with_capacity(messages.len() + 4);

    // Keep system messages as-is (there should be at most one at this point).
    out.extend(messages.drain(..system_prefix_len));

    // Devstral template expects first counted message to be user.
    let mut expect_user = true;

    for msg in messages {
        if !is_counted_for_mistral3_template(&msg) {
            out.push(msg);
            continue;
        }

        let is_user = msg.role == "user";
        if is_user != expect_user {
            // Insert an empty counted message to restore alternation without reordering.
            out.push(Message {
                content: Some(MessageContent::from_text(String::new())),
                role: if expect_user {
                    "user".to_string()
                } else {
                    "assistant".to_string()
                },
                name: None,
                tool_call_id: None,
                tool_calls: None,
            });
            expect_user = !expect_user;
        }

        out.push(msg);
        expect_user = !expect_user;
    }

    out
}

fn should_canonicalize(messages: &[Message]) -> bool {
    if messages.is_empty() {
        return false;
    }

    // Multiple system messages or system not first will violate Devstral's template.
    let system_count = messages.iter().filter(|m| m.role == "system").count();
    if system_count > 1 || (system_count == 1 && messages[0].role != "system") {
        return true;
    }

    if messages
        .iter()
        .any(|m| matches!(m.role.as_str(), "developer" | "function"))
    {
        return true;
    }

    // If an assistant message contains a Codex-style tool call in plain text, we must canonicalize
    // to convert it into structured `tool_calls` so Devstral/Mistral3 templates don't count it
    // as a normal assistant message (and so downstream tooling can treat it as a tool call).
    if messages.iter().any(|m| {
        m.role == "assistant"
            && m.tool_calls.is_none()
            && m.content
                .as_ref()
                .is_some_and(|c| message_content_to_text_lossy(c).contains("[TOOL_CALLS]"))
    }) {
        return true;
    }

    // Codex preambles / environment blocks should be treated as system context.
    if messages.iter().any(|m| {
        m.role == "user" && m.tool_call_id.is_none() && m.tool_calls.is_none() && {
            let text = message_text_for_merge(m);
            looks_like_codex_context_message(&text)
                || (text.len() > 1024 && looks_like_codex_preamble(&text))
        }
    }) {
        return true;
    }

    violates_mistral3_alternation(messages)
}

/// Best-effort canonicalization to satisfy Devstral/Mistral3 chat templates:
/// - Only one optional `system` message at the start.
/// - After that, counted roles must alternate `user` and `assistant` (tool/tool_call messages do not count).
/// - Non-standard roles (e.g. `developer`) are folded into the system prompt.
pub fn canonicalize_messages_for_mistral3_template(messages: Vec<Message>) -> Vec<Message> {
    let mut system_parts: Vec<String> = Vec::new();
    let mut out: Vec<Message> = Vec::new();
    let mut last_counted_role: Option<String> = None;

    for mut msg in messages {
        // Normalize roles.
        match msg.role.as_str() {
            "system" | "developer" => {
                let text = message_text_for_merge(&msg);
                if !text.is_empty() && !system_parts.iter().any(|s| s.trim() == text.trim()) {
                    system_parts.push(text);
                }
                continue;
            }
            // Some clients still use ChatCompletions' `function` role; treat it like tool output.
            "function" => msg.role = "tool".to_string(),
            "user" | "assistant" | "tool" => {}
            _ => {
                // Best-effort fallback: treat unknown roles as user input.
                msg.role = "user".to_string();
            }
        }

        // If the assistant message encodes a tool call using Codex's text protocol, convert it
        // to structured `tool_calls` so Devstral's chat template treats it as a tool call and
        // does not count it for strict alternation.
        if msg.role == "assistant" && msg.tool_calls.is_none() {
            if let Some(content) = msg.content.as_ref() {
                let text = message_content_to_text_lossy(content);
                if let Some(calls) = parse_codex_tool_calls_from_text(&text) {
                    msg.tool_calls = Some(calls);
                    msg.content = None;
                }
            }
        }

        // Codex frequently sends the repo AGENTS.md and sandbox environment context as user messages.
        // Treat these as system context to improve instruction-following and avoid accidental "respond to the harness".
        if msg.role == "user" && msg.tool_call_id.is_none() && msg.tool_calls.is_none() {
            let text = message_text_for_merge(&msg);
            let treat_as_system = looks_like_codex_context_message(&text)
                || (text.len() > 1024 && looks_like_codex_preamble(&text));
            if treat_as_system {
                if !text.is_empty() && !system_parts.iter().any(|s| s.trim() == text.trim()) {
                    system_parts.push(text);
                }
                continue;
            }
        }

        if is_counted_for_mistral3_template(&msg) {
            if let Some(prev_role) = last_counted_role.clone() {
                if prev_role == msg.role {
                    // Merge only if the immediately previous message is mergeable.
                    // Never merge across intervening tool/non-counted messages, as that would
                    // reorder the conversation (e.g. "assistant tool-call" + "assistant final").
                    if let Some(target) = out.last_mut() {
                        if target.role == msg.role && is_counted_for_mistral3_template(target) {
                            let merged =
                                merge_message_content(target.content.take(), msg.content.take());
                            target.content = merged;
                            continue;
                        }
                    }
                }
            }
            last_counted_role = Some(msg.role.clone());
        }

        out.push(msg);
    }

    // Ensure there is at most one system message and it is first.
    if !system_parts
        .iter()
        .any(|s| s.trim() == TOOL_CALL_JSON_HINT.trim())
    {
        system_parts.push(TOOL_CALL_JSON_HINT.to_string());
    }
    if !system_parts.is_empty() {
        let combined = system_parts.join("\n\n");
        let system_msg = Message {
            content: Some(MessageContent::from_text(combined)),
            role: "system".to_string(),
            name: None,
            tool_call_id: None,
            tool_calls: None,
        };

        if out.first().is_some_and(|m| m.role == "system") {
            // Merge into existing system message.
            if let Some(first) = out.first_mut() {
                first.content =
                    merge_message_content(first.content.take(), system_msg.content.clone());
            }
        } else {
            out.insert(0, system_msg);
        }
    }

    // If there are still alternation violations (e.g. duplicate assistant messages separated by
    // tool/tool-call messages), fix them by inserting empty counted messages.
    if violates_mistral3_alternation(&out) {
        out = enforce_mistral3_alternation_by_insertion(out);
    }

    out
}

/// Canonicalize only when it looks necessary (to avoid changing semantics for already-valid chats).
pub fn canonicalize_messages_for_mistral3_template_if_needed(
    messages: Vec<Message>,
) -> Vec<Message> {
    if should_canonicalize(&messages) {
        canonicalize_messages_for_mistral3_template(messages)
    } else {
        messages
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn assistant_tool_call_text_is_not_counted() {
        let msgs = vec![
            Message {
                content: Some(MessageContent::from_text("hi".to_string())),
                role: "user".to_string(),
                name: None,
                tool_call_id: None,
                tool_calls: None,
            },
            Message {
                content: Some(MessageContent::from_text(
                    "[TOOL_CALLS]shell {\"command\":[\"ls\",\"-la\"]}".to_string(),
                )),
                role: "assistant".to_string(),
                name: None,
                tool_call_id: None,
                tool_calls: None,
            },
            Message {
                content: Some(MessageContent::from_text("result".to_string())),
                role: "tool".to_string(),
                name: None,
                tool_call_id: Some("call_1".to_string()),
                tool_calls: None,
            },
            Message {
                content: Some(MessageContent::from_text("ok".to_string())),
                role: "assistant".to_string(),
                name: None,
                tool_call_id: None,
                tool_calls: None,
            },
        ];

        assert!(!violates_mistral3_alternation(&msgs));
    }

    #[test]
    fn canonicalize_if_needed_converts_plaintext_codex_tool_calls() {
        let msgs = vec![
            Message {
                content: Some(MessageContent::from_text("hi".to_string())),
                role: "user".to_string(),
                name: None,
                tool_call_id: None,
                tool_calls: None,
            },
            Message {
                content: Some(MessageContent::from_text(
                    "[TOOL_CALLS]shell[ARGS]{\"command\":[\"bash\",\"-lc\",\"ls -la\"],\"workdir\":\".\"}".to_string(),
                )),
                role: "assistant".to_string(),
                name: None,
                tool_call_id: None,
                tool_calls: None,
            },
            Message {
                content: Some(MessageContent::from_text("ok".to_string())),
                role: "assistant".to_string(),
                name: None,
                tool_call_id: None,
                tool_calls: None,
            },
        ];

        // This input alternates roles (user -> assistant -> assistant violates),
        // but the middle assistant message is a tool call; canonicalization should convert it
        // and (if necessary) fix alternation without leaking the raw tool-call text.
        let out = canonicalize_messages_for_mistral3_template_if_needed(msgs);
        assert!(out
            .iter()
            .any(|m| m.role == "assistant" && m.tool_calls.is_some()));
        assert!(!out.iter().any(|m| m.role == "assistant"
            && m.tool_calls.is_none()
            && m.content
                .as_ref()
                .is_some_and(|c| { message_content_to_text_lossy(c).contains("[TOOL_CALLS]") })));
    }

    #[test]
    fn canonicalize_does_not_merge_across_tool_messages() {
        let msgs = vec![
            Message {
                content: Some(MessageContent::from_text("hi".to_string())),
                role: "user".to_string(),
                name: None,
                tool_call_id: None,
                tool_calls: None,
            },
            Message {
                content: Some(MessageContent::from_text("a".to_string())),
                role: "assistant".to_string(),
                name: None,
                tool_call_id: None,
                tool_calls: None,
            },
            Message {
                content: Some(MessageContent::from_text("tool output".to_string())),
                role: "tool".to_string(),
                name: None,
                tool_call_id: Some("call_1".to_string()),
                tool_calls: None,
            },
            Message {
                content: Some(MessageContent::from_text("b".to_string())),
                role: "assistant".to_string(),
                name: None,
                tool_call_id: None,
                tool_calls: None,
            },
        ];

        let out = canonicalize_messages_for_mistral3_template(msgs);
        assert!(!violates_mistral3_alternation(&out));
        // out[0] is the inserted system prompt.
        // out[1] is User "hi"
        // out[2] is Assistant "a"
        assert_eq!(out[2].content.as_ref().unwrap().to_text().unwrap(), "a");
        assert_eq!(out[3].role, "tool");
        // We may insert an empty user message to satisfy alternation.
        assert_eq!(
            out.last()
                .unwrap()
                .content
                .as_ref()
                .unwrap()
                .to_text()
                .unwrap(),
            "b"
        );
    }

    #[test]
    fn canonicalize_converts_codex_tool_call_text_to_structured_tool_calls() {
        let msgs = vec![
            Message {
                content: Some(MessageContent::from_text("hi".to_string())),
                role: "user".to_string(),
                name: None,
                tool_call_id: None,
                tool_calls: None,
            },
            Message {
                content: Some(MessageContent::from_text(
                    "[TOOL_CALLS]shell {\"command\":[\"ls\",\"-la\"],\"workdir\":\".\"}"
                        .to_string(),
                )),
                role: "assistant".to_string(),
                name: None,
                tool_call_id: None,
                tool_calls: None,
            },
        ];

        let out = canonicalize_messages_for_mistral3_template(msgs);
        let assistant = out.iter().find(|m| m.role == "assistant").unwrap();
        assert!(assistant
            .tool_calls
            .as_ref()
            .is_some_and(|tc| !tc.is_empty()));
        assert!(assistant.content.is_none());
        assert!(!violates_mistral3_alternation(&out));
    }

    #[test]
    fn canonicalize_converts_codex_tool_call_text_no_space() {
        let msgs = vec![
            Message {
                content: Some(MessageContent::from_text("hi".to_string())),
                role: "user".to_string(),
                name: None,
                tool_call_id: None,
                tool_calls: None,
            },
            Message {
                content: Some(MessageContent::from_text(
                    "[TOOL_CALLS]shell[ARGS]{\"command\":[\"ls\",\"-la\"],\"workdir\":\".\"}"
                        .to_string(),
                )),
                role: "assistant".to_string(),
                name: None,
                tool_call_id: None,
                tool_calls: None,
            },
        ];

        let out = canonicalize_messages_for_mistral3_template(msgs);
        let assistant = out.iter().find(|m| m.role == "assistant").unwrap();
        assert!(assistant
            .tool_calls
            .as_ref()
            .is_some_and(|tc| !tc.is_empty()));
        assert!(assistant.content.is_none());
        assert!(!violates_mistral3_alternation(&out));
    }

    #[test]
    fn canonicalize_repairs_unescaped_quotes_in_shell_tool_call_text() {
        let msgs = vec![
            Message {
                content: Some(MessageContent::from_text("hi".to_string())),
                role: "user".to_string(),
                name: None,
                tool_call_id: None,
                tool_calls: None,
            },
            Message {
                content: Some(MessageContent::from_text(
                    r##"[TOOL_CALLS]shell[ARGS]{"command":["bash","-lc","find . -name ".md" -type f"],"workdir":"/weka/users/gu to see files"}"##
                        .to_string(),
                )),
                role: "assistant".to_string(),
                name: None,
                tool_call_id: None,
                tool_calls: None,
            },
        ];

        let out = canonicalize_messages_for_mistral3_template(msgs);
        let assistant = out.iter().find(|m| m.role == "assistant").unwrap();
        let tool_calls = assistant.tool_calls.as_ref().unwrap();
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].function.name, "shell");
        let params: Value = serde_json::from_str(&tool_calls[0].function.arguments).unwrap();
        assert_eq!(params["command"][2], "find . -name '.md' -type f");
        assert_eq!(params["workdir"], ".");
        assert!(!violates_mistral3_alternation(&out));
    }

    #[test]
    fn canonicalize_normalizes_shell_command_string_to_array() {
        let msgs = vec![
            Message {
                content: Some(MessageContent::from_text("hi".to_string())),
                role: "user".to_string(),
                name: None,
                tool_call_id: None,
                tool_calls: None,
            },
            Message {
                content: Some(MessageContent::from_text(
                    r#"[TOOL_CALLS]shell[ARGS]{"command":"ls -la","workdir":"/tmp"}"#.to_string(),
                )),
                role: "assistant".to_string(),
                name: None,
                tool_call_id: None,
                tool_calls: None,
            },
        ];

        let out = canonicalize_messages_for_mistral3_template(msgs);
        let assistant = out.iter().find(|m| m.role == "assistant").unwrap();
        let tool_calls = assistant.tool_calls.as_ref().unwrap();
        let params: Value = serde_json::from_str(&tool_calls[0].function.arguments).unwrap();
        assert_eq!(params["command"][0], "bash");
        assert_eq!(params["command"][1], "-lc");
        assert_eq!(params["command"][2], "ls -la");
        assert_eq!(params["workdir"], "/tmp");
    }

    #[test]
    fn canonicalize_inserts_empty_user_between_two_assistant_counted_messages_separated_by_tool() {
        let msgs = vec![
            Message {
                content: Some(MessageContent::from_text("u1".to_string())),
                role: "user".to_string(),
                name: None,
                tool_call_id: None,
                tool_calls: None,
            },
            Message {
                content: Some(MessageContent::from_text("a1".to_string())),
                role: "assistant".to_string(),
                name: None,
                tool_call_id: None,
                tool_calls: None,
            },
            // Non-counted tool output (e.g. tool result replayed by client).
            Message {
                content: Some(MessageContent::from_text("tool out".to_string())),
                role: "tool".to_string(),
                name: None,
                tool_call_id: Some("call_1".to_string()),
                tool_calls: None,
            },
            // A second assistant message with normal text (counted) without an intervening user.
            Message {
                content: Some(MessageContent::from_text("a2".to_string())),
                role: "assistant".to_string(),
                name: None,
                tool_call_id: None,
                tool_calls: None,
            },
        ];

        assert!(violates_mistral3_alternation(&msgs));
        let out = canonicalize_messages_for_mistral3_template(msgs);
        assert!(!violates_mistral3_alternation(&out));

        // Should insert an empty user message before the final assistant.
        let mut saw_inserted = false;
        for win in out.windows(2) {
            if win[0].role == "user"
                && win[0]
                    .content
                    .as_ref()
                    .and_then(|c| c.to_text())
                    .is_some_and(|t| t.is_empty())
                && win[1].role == "assistant"
                && win[1]
                    .content
                    .as_ref()
                    .and_then(|c| c.to_text())
                    .is_some_and(|t| t == "a2")
            {
                saw_inserted = true;
            }
        }
        assert!(saw_inserted);
    }
}
