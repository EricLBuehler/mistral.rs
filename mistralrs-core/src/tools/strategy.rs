use llguidance::api::TopLevelGrammar;

use crate::{
    reasoning_parsers::{HarmonyContext, HarmonyToolCall},
    tools::{parsers, ToolCallFormat, ToolCallResponse, ToolCallType},
    Tool,
};
use mistralrs_mcp::CalledFunction;
use std::{borrow::Cow, collections::HashMap};
use uuid::Uuid;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ToolCallBoundary {
    ContinueCurrentMessage,
    StartNewMessage,
}

pub(crate) trait ToolCallStrategy: Send + Sync {
    fn observe_token(&mut self, token: u32, bytes: &[u8]);
    fn continuation_grammar(
        &mut self,
        text: Option<&str>,
        tools: &[Tool],
    ) -> Option<TopLevelGrammar>;
    fn required_grammar(&self, tools: &[Tool], boundary: ToolCallBoundary) -> TopLevelGrammar;
    fn required_boundary(&self) -> ToolCallBoundary {
        ToolCallBoundary::ContinueCurrentMessage
    }
    fn has_reasoning(&self) -> bool {
        false
    }
    fn finalize(&mut self) {}
    fn content_delta(&mut self) -> Option<String> {
        None
    }
    fn reasoning_delta(&mut self) -> Option<String> {
        None
    }
    fn content(&self) -> Option<String> {
        None
    }
    fn reasoning_content(&self) -> Option<String> {
        None
    }
    fn has_tool_calls(&self) -> bool {
        false
    }
    fn stops_after_complete_tool_call(&self) -> bool {
        true
    }
    fn finalize_tool_calls(&mut self) -> Vec<ToolCallResponse> {
        Vec::new()
    }
}

pub(crate) struct TextToolCallStrategy {
    preferred_format: Option<ToolCallFormat>,
}

impl TextToolCallStrategy {
    pub(crate) fn new(preferred_format: Option<ToolCallFormat>) -> Self {
        Self { preferred_format }
    }
}

impl ToolCallStrategy for TextToolCallStrategy {
    fn observe_token(&mut self, _token: u32, _bytes: &[u8]) {}

    fn continuation_grammar(
        &mut self,
        text: Option<&str>,
        tools: &[Tool],
    ) -> Option<TopLevelGrammar> {
        parsers::build_tool_call_grammar(text?, tools)
    }

    fn required_grammar(&self, tools: &[Tool], _boundary: ToolCallBoundary) -> TopLevelGrammar {
        parsers::build_required_tool_call_grammar(self.preferred_format, tools)
    }
}

pub(crate) struct AtemToolCallStrategy {
    bytes: Vec<u8>,
    known_tools: HashMap<String, Tool>,
    emitted_content: String,
    emitted_reasoning: String,
    tool_calls_taken: bool,
}

impl AtemToolCallStrategy {
    pub(crate) fn new(tools: Option<&[Tool]>) -> Self {
        Self {
            bytes: Vec::new(),
            known_tools: tools
                .unwrap_or_default()
                .iter()
                .map(|tool| (tool.function.name.clone(), tool.clone()))
                .collect(),
            emitted_content: String::new(),
            emitted_reasoning: String::new(),
            tool_calls_taken: false,
        }
    }

    fn response(&self) -> parsers::atem::AtemResponse {
        let text = decode_streaming_utf8(&self.bytes);
        parsers::atem::parse_atem_response(&text).unwrap_or_default()
    }

    fn delta(emitted: &mut String, value: &str) -> Option<String> {
        let delta = value
            .strip_prefix(emitted.as_str())
            .unwrap_or(value)
            .to_string();
        *emitted = value.to_string();
        (!delta.is_empty()).then_some(delta)
    }

    fn required_prefix(&self) -> &'static str {
        let text = decode_streaming_utf8(&self.bytes);
        if text.ends_with("<|start|>assistant") {
            return "";
        }
        let Some(message_start) = text.rfind("<|message|>") else {
            return "";
        };
        let closed = ["<|eom|>", "<|eot|>"]
            .into_iter()
            .filter_map(|token| text.rfind(token))
            .max()
            .is_some_and(|end| end > message_start);
        if closed {
            "<|start|>assistant"
        } else {
            "<|eom|><|start|>assistant"
        }
    }
}

impl ToolCallStrategy for AtemToolCallStrategy {
    fn observe_token(&mut self, _token: u32, bytes: &[u8]) {
        self.bytes.extend_from_slice(bytes);
    }

    fn continuation_grammar(
        &mut self,
        _text: Option<&str>,
        tools: &[Tool],
    ) -> Option<TopLevelGrammar> {
        let text = decode_streaming_utf8(&self.bytes);
        parsers::build_tool_call_grammar(&text, tools)
    }

    fn required_grammar(&self, tools: &[Tool], _boundary: ToolCallBoundary) -> TopLevelGrammar {
        parsers::atem::required_tool_call_grammar(tools, self.required_prefix())
    }

    fn has_reasoning(&self) -> bool {
        true
    }

    fn content_delta(&mut self) -> Option<String> {
        let content = self.response().content;
        Self::delta(&mut self.emitted_content, &content)
    }

    fn reasoning_delta(&mut self) -> Option<String> {
        let reasoning = self.response().reasoning;
        Self::delta(&mut self.emitted_reasoning, &reasoning)
    }

    fn content(&self) -> Option<String> {
        let content = self.response().content;
        (!content.is_empty()).then_some(content)
    }

    fn reasoning_content(&self) -> Option<String> {
        let reasoning = self.response().reasoning;
        (!reasoning.is_empty()).then_some(reasoning)
    }

    fn has_tool_calls(&self) -> bool {
        self.response()
            .tool_calls
            .iter()
            .any(|call| self.known_tools.contains_key(&call.name))
    }

    fn stops_after_complete_tool_call(&self) -> bool {
        false
    }

    fn finalize_tool_calls(&mut self) -> Vec<ToolCallResponse> {
        if self.tool_calls_taken {
            return Vec::new();
        }
        self.tool_calls_taken = true;
        self.response()
            .tool_calls
            .into_iter()
            .filter_map(|mut call| {
                let tool = self.known_tools.get(&call.name)?;
                parsers::atem::normalize_atem_arguments(&mut call, tool);
                Some(call)
            })
            .enumerate()
            .map(|(index, call)| ToolCallResponse {
                index,
                id: format!("call-{}", Uuid::new_v4()),
                tp: ToolCallType::Function,
                function: CalledFunction {
                    name: call.name,
                    arguments: call.arguments.to_string(),
                },
            })
            .collect()
    }
}

fn decode_streaming_utf8(bytes: &[u8]) -> Cow<'_, str> {
    let first_error = match std::str::from_utf8(bytes) {
        Ok(text) => return Cow::Borrowed(text),
        Err(error) => error,
    };
    if first_error.error_len().is_none() {
        return Cow::Borrowed(
            std::str::from_utf8(&bytes[..first_error.valid_up_to()])
                .expect("valid_up_to must identify valid UTF-8"),
        );
    }

    let mut decoded = String::with_capacity(bytes.len());
    let mut remaining = bytes;
    loop {
        match std::str::from_utf8(remaining) {
            Ok(text) => {
                decoded.push_str(text);
                break;
            }
            Err(error) => {
                decoded.push_str(
                    std::str::from_utf8(&remaining[..error.valid_up_to()])
                        .expect("valid_up_to must identify valid UTF-8"),
                );
                let Some(error_len) = error.error_len() else {
                    break;
                };
                decoded.push('\u{fffd}');
                remaining = &remaining[error.valid_up_to() + error_len..];
            }
        }
    }
    Cow::Owned(decoded)
}

pub(crate) struct HarmonyToolCallStrategy {
    context: Option<HarmonyContext>,
}

impl HarmonyToolCallStrategy {
    pub(crate) fn new() -> anyhow::Result<Self> {
        Ok(Self { context: None })
    }

    fn context(&self) -> Option<&HarmonyContext> {
        self.context.as_ref()
    }

    fn context_mut(&mut self) -> Option<&mut HarmonyContext> {
        if self.context.is_none() {
            match HarmonyContext::new() {
                Ok(context) => self.context = Some(context),
                Err(e) => {
                    tracing::warn!("Failed to initialize Harmony parser: {e}");
                    return None;
                }
            }
        }
        self.context.as_mut()
    }
}

impl ToolCallStrategy for HarmonyToolCallStrategy {
    fn observe_token(&mut self, token: u32, _bytes: &[u8]) {
        if let Some(context) = self.context_mut() {
            context.process_token(token);
        }
    }

    fn continuation_grammar(
        &mut self,
        _text: Option<&str>,
        tools: &[Tool],
    ) -> Option<TopLevelGrammar> {
        let context = self.context_mut()?;
        if !context.take_needs_grammar_activation() {
            return None;
        }
        let recipient = context
            .get_current_tool_call()
            .map(|(recipient, _)| recipient.to_string());
        Some(parsers::harmony::tool_call_grammar_for_tool(
            recipient.as_deref(),
            Some(tools),
        ))
    }

    fn required_grammar(&self, tools: &[Tool], boundary: ToolCallBoundary) -> TopLevelGrammar {
        parsers::harmony::required_tool_call_grammar(
            tools,
            matches!(boundary, ToolCallBoundary::StartNewMessage),
        )
    }

    fn required_boundary(&self) -> ToolCallBoundary {
        if self
            .context()
            .is_some_and(|context| context.current_channel().is_some())
        {
            ToolCallBoundary::StartNewMessage
        } else {
            ToolCallBoundary::ContinueCurrentMessage
        }
    }

    fn has_reasoning(&self) -> bool {
        true
    }

    fn finalize(&mut self) {
        if let Some(context) = self.context.as_mut() {
            context.process_eos();
        }
    }

    fn content_delta(&mut self) -> Option<String> {
        self.context.as_mut()?.get_final_delta()
    }

    fn reasoning_delta(&mut self) -> Option<String> {
        self.context.as_mut()?.get_reasoning_delta()
    }

    fn content(&self) -> Option<String> {
        self.context()?.final_content()
    }

    fn reasoning_content(&self) -> Option<String> {
        self.context()?.reasoning_content()
    }

    fn has_tool_calls(&self) -> bool {
        self.context()
            .is_some_and(|context| context.has_tool_call())
    }

    fn finalize_tool_calls(&mut self) -> Vec<ToolCallResponse> {
        self.context
            .as_mut()
            .map(|context| harmony_tool_calls_to_responses(context.finalize_tool_calls()))
            .unwrap_or_default()
    }
}

fn harmony_tool_calls_to_responses(calls: Vec<HarmonyToolCall>) -> Vec<ToolCallResponse> {
    calls
        .into_iter()
        .enumerate()
        .map(|(index, call)| ToolCallResponse {
            index,
            id: call.id,
            tp: ToolCallType::Function,
            function: CalledFunction {
                name: call.name,
                arguments: call.arguments,
            },
        })
        .collect()
}
