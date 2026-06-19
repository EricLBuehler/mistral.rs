use llguidance::api::TopLevelGrammar;

use crate::{
    reasoning_parsers::{HarmonyContext, HarmonyToolCall},
    tools::{parsers, ToolCallFormat, ToolCallResponse, ToolCallType},
    Tool,
};
use mistralrs_mcp::CalledFunction;

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
