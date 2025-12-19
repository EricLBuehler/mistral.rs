use serde_json::Value;
use uuid::Uuid;

use crate::tools::{ToolCallResponse, ToolCallType};
use mistralrs_mcp::CalledFunction;
use tokenizers::Tokenizer;

#[derive(Debug, Clone, Copy)]
pub struct MistralV11ToolTokenIds {
    pub tool_calls: u32,
    pub args: u32,
    pub call_id: Option<u32>,
}

impl MistralV11ToolTokenIds {
    pub fn from_tokenizer(tokenizer: &Tokenizer) -> Option<Self> {
        let tool_calls = tokenizer.token_to_id("[TOOL_CALLS]")?;
        let args = tokenizer.token_to_id("[ARGS]")?;
        let call_id = tokenizer.token_to_id("[CALL_ID]");
        Some(Self {
            tool_calls,
            args,
            call_id,
        })
    }
}

#[derive(Debug, Clone)]
struct PartialToolCall {
    name: String,
    args: String,
    started: bool,
    brace_depth: i32,
    bracket_depth: i32,
}

impl PartialToolCall {
    fn new() -> Self {
        Self {
            name: String::new(),
            args: String::new(),
            started: false,
            brace_depth: 0,
            bracket_depth: 0,
        }
    }

    fn feed_args_text(&mut self, s: &str) {
        self.args.push_str(s);
        for ch in s.chars() {
            match ch {
                '{' => {
                    self.started = true;
                    self.brace_depth += 1;
                }
                '}' => self.brace_depth -= 1,
                '[' => {
                    self.started = true;
                    self.bracket_depth += 1;
                }
                ']' => self.bracket_depth -= 1,
                _ => {}
            }
        }
    }

    fn is_args_complete(&self) -> bool {
        self.started && self.brace_depth <= 0 && self.bracket_depth <= 0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StreamingState {
    Content,
    ParsingToolName,
    ParsingCallId,
    ParsingArgs,
    /// Arguments are complete; await either a new `[TOOL_CALLS]` or end-of-generation.
    AfterArgs,
}

/// Token-ID-based streaming tool-call parser for Mistral v11+ style tool calls:
/// `[TOOL_CALLS]name[ARGS]{...}` (optionally with `[CALL_ID]id` between name and args).
///
/// This intentionally focuses on the v11+ control-token format and does not attempt to
/// support older text-only formats.
#[derive(Debug, Clone)]
pub struct MistralV11ToolStreamParser {
    ids: MistralV11ToolTokenIds,
    state: StreamingState,
    calls: Vec<ToolCallResponse>,
    current: Option<PartialToolCall>,
    /// Once we see `[TOOL_CALLS]`, we suppress streaming content until we finish parsing.
    in_tools: bool,
}

impl MistralV11ToolStreamParser {
    pub fn new(ids: MistralV11ToolTokenIds) -> Self {
        Self {
            ids,
            state: StreamingState::Content,
            calls: Vec::new(),
            current: None,
            in_tools: false,
        }
    }

    pub fn in_progress(&self) -> bool {
        self.in_tools && self.state != StreamingState::Content
    }

    pub fn saw_tools(&self) -> bool {
        self.in_tools
    }

    fn start_new_call(&mut self) {
        self.in_tools = true;
        self.state = StreamingState::ParsingToolName;
        self.current = Some(PartialToolCall::new());
    }

    fn reset_to_content(&mut self) {
        self.in_tools = false;
        self.state = StreamingState::Content;
        self.current = None;
    }

    fn finalize_current(&mut self) -> Option<ToolCallResponse> {
        let cur = self.current.take()?;
        let name = cur.name.replace('▁', " ").trim().to_string();
        if name.is_empty() {
            return None;
        }
        let args_raw = cur.args.trim();
        let args_val: Value = serde_json::from_str(args_raw).ok()?;
        let args = serde_json::to_string(&args_val).ok()?;

        Some(ToolCallResponse {
            index: self.calls.len(),
            id: format!("call-{}", Uuid::new_v4()),
            tp: ToolCallType::Function,
            function: CalledFunction {
                name,
                arguments: args,
            },
        })
    }

    /// Feed a single generated token (id + decoded text) into the parser.
    ///
    /// Returns `Some(tool_calls)` when a complete tool call has been parsed and tool calls
    /// should be emitted as a structured response.
    pub fn feed_token(
        &mut self,
        token_id: u32,
        token_text: &str,
        is_done: bool,
    ) -> Option<Vec<ToolCallResponse>> {
        if !self.in_tools {
            if token_id == self.ids.tool_calls {
                self.start_new_call();
            }
            return None;
        }

        // `[TOOL_CALLS]` always begins a new tool-call segment.
        // If we were mid-parse, we finalize best-effort and restart.
        if token_id == self.ids.tool_calls {
            if matches!(self.state, StreamingState::ParsingArgs) {
                if let Some(done) = self.finalize_current() {
                    self.calls.push(done);
                }
            } else {
                self.current = None;
            }
            self.state = StreamingState::ParsingToolName;
            self.current = Some(PartialToolCall::new());
            return None;
        }

        match self.state {
            StreamingState::Content => {}
            StreamingState::ParsingToolName => {
                if token_id == self.ids.args {
                    self.state = StreamingState::ParsingArgs;
                } else if self.ids.call_id.is_some_and(|id| id == token_id) {
                    self.state = StreamingState::ParsingCallId;
                } else if let Some(cur) = self.current.as_mut() {
                    cur.name.push_str(token_text);
                }
            }
            StreamingState::ParsingCallId => {
                if token_id == self.ids.args {
                    self.state = StreamingState::ParsingArgs;
                } else {
                    // Ignore call-id contents (we generate our own IDs).
                }
            }
            StreamingState::ParsingArgs => {
                if let Some(cur) = self.current.as_mut() {
                    cur.feed_args_text(token_text);
                    if cur.is_args_complete() {
                        if let Some(done) = self.finalize_current() {
                            self.calls.push(done);
                        }
                        if is_done {
                            let out = std::mem::take(&mut self.calls);
                            self.reset_to_content();
                            if !out.is_empty() {
                                return Some(out);
                            }
                        } else {
                            self.state = StreamingState::AfterArgs;
                        }
                    }
                }
            }
            StreamingState::AfterArgs => {
                // We've completed at least one tool call and are waiting to see whether another
                // `[TOOL_CALLS]` begins or generation ends. We intentionally suppress any leaked
                // assistant text here (clients expect finish_reason=tool_calls).
                if is_done {
                    let out = std::mem::take(&mut self.calls);
                    self.reset_to_content();
                    if !out.is_empty() {
                        return Some(out);
                    }
                    return None;
                }

                // Whitespace between tool calls / before EOS is ignored.
                if token_text.trim().is_empty() {
                    return None;
                }

                // Any other token means the model started emitting non-tool content; stop and
                // surface the parsed tool calls instead of leaking text.
                let out = std::mem::take(&mut self.calls);
                self.reset_to_content();
                if !out.is_empty() {
                    return Some(out);
                }
            }
        }

        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_single_tool_call_stream() {
        let ids = MistralV11ToolTokenIds {
            tool_calls: 9,
            args: 32,
            call_id: Some(33),
        };
        let mut p = MistralV11ToolStreamParser::new(ids);

        // [TOOL_CALLS]shell[ARGS]{"a":1}
        assert!(p.feed_token(9, "[TOOL_CALLS]", false).is_none());
        assert!(p.feed_token(999, "shell", false).is_none());
        assert!(p.feed_token(32, "[ARGS]", false).is_none());
        assert!(p.feed_token(1000, "{\"", false).is_none());
        assert!(p.feed_token(1001, "a", false).is_none());
        assert!(p.feed_token(1002, "\":", false).is_none());
        assert!(p.feed_token(1003, "1", false).is_none());
        let out = p.feed_token(1004, "}", true).unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].function.name, "shell");
        assert_eq!(out[0].function.arguments, "{\"a\":1}");
    }

    #[test]
    fn supports_optional_call_id_segment() {
        let ids = MistralV11ToolTokenIds {
            tool_calls: 9,
            args: 32,
            call_id: Some(33),
        };
        let mut p = MistralV11ToolStreamParser::new(ids);

        assert!(p.feed_token(9, "[TOOL_CALLS]", false).is_none());
        assert!(p.feed_token(999, "shell", false).is_none());
        assert!(p.feed_token(33, "[CALL_ID]", false).is_none());
        assert!(p.feed_token(999, "abcdefghi", false).is_none());
        assert!(p.feed_token(32, "[ARGS]", false).is_none());
        assert!(p.feed_token(1000, "{\"x\":", false).is_none());
        assert!(p.feed_token(1001, "2}", true).is_some());
    }

    #[test]
    fn parses_multiple_tool_calls_in_one_message() {
        let ids = MistralV11ToolTokenIds {
            tool_calls: 9,
            args: 32,
            call_id: Some(33),
        };
        let mut p = MistralV11ToolStreamParser::new(ids);

        // [TOOL_CALLS]a[ARGS]{"a":1}[TOOL_CALLS]b[ARGS]{"b":2}</s>
        assert!(p.feed_token(9, "[TOOL_CALLS]", false).is_none());
        assert!(p.feed_token(100, "a", false).is_none());
        assert!(p.feed_token(32, "[ARGS]", false).is_none());
        assert!(p.feed_token(101, "{\"a\":1}", false).is_none());

        assert!(p.feed_token(9, "[TOOL_CALLS]", false).is_none());
        assert!(p.feed_token(102, "b", false).is_none());
        assert!(p.feed_token(32, "[ARGS]", false).is_none());
        assert!(p.feed_token(103, "{\"b\":2}", false).is_none());

        let out = p
            .feed_token(2, "</s>", true)
            .expect("should emit tool calls");
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].function.name, "a");
        assert_eq!(out[0].function.arguments, "{\"a\":1}");
        assert_eq!(out[1].function.name, "b");
        assert_eq!(out[1].function.arguments, "{\"b\":2}");
    }
}
