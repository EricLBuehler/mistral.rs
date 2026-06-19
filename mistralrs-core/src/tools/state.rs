use candle_core::Result;
use llguidance::api::TopLevelGrammar;

use crate::tools::{
    strategy::{HarmonyToolCallStrategy, TextToolCallStrategy, ToolCallStrategy},
    ToolCallFormat, ToolCallResponse, ToolCallingMatcher, ToolChoice,
};

const REQUIRED_TOOL_CALL_DEADLINE_DIVISOR: usize = 4;
const REQUIRED_TOOL_CALL_DEADLINE_MIN_TOKENS: usize = 1024;
const REQUIRED_TOOL_CALL_DEADLINE_MAX_TOKENS: usize = 4096;

#[derive(Clone, Copy, Debug, Default)]
struct ToolObligation {
    satisfied: bool,
    forced: bool,
}

impl ToolObligation {
    fn unsatisfied(self, requires_tool_call: bool) -> bool {
        requires_tool_call && !self.satisfied
    }

    fn mark_satisfied(&mut self, requires_tool_call: bool) {
        if requires_tool_call {
            self.satisfied = true;
        }
    }

    fn should_force(
        self,
        requires_tool_call: bool,
        max_generation_len: usize,
        remaining: usize,
    ) -> bool {
        if !self.unsatisfied(requires_tool_call) || self.forced {
            return false;
        }
        remaining <= required_tool_call_deadline_tokens(max_generation_len)
    }

    fn mark_forced(&mut self) {
        self.forced = true;
    }

    fn clear_forced(&mut self) {
        self.forced = false;
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub(crate) enum ToolGrammarState {
    #[default]
    Inactive,
    Active {
        forced: bool,
    },
}

pub(crate) struct ToolCallParse {
    pub content: Option<String>,
    pub reasoning_content: Option<String>,
    pub tool_calls: Vec<ToolCallResponse>,
    pub tool_use_still_possible: bool,
    pub tool_use_is_done: bool,
}

impl ToolCallParse {
    fn empty(content: Option<String>) -> Self {
        Self {
            content,
            reasoning_content: None,
            tool_calls: Vec::new(),
            tool_use_still_possible: false,
            tool_use_is_done: false,
        }
    }
}

pub(crate) struct ToolCallState {
    matcher: ToolCallingMatcher,
    strategy: Box<dyn ToolCallStrategy>,
    grammar: ToolGrammarState,
    obligation: ToolObligation,
}

impl ToolCallState {
    pub(crate) fn new(
        tool_choice: ToolChoice,
        tools: Option<&[crate::Tool]>,
        preferred_format: Option<ToolCallFormat>,
    ) -> anyhow::Result<Self> {
        let matcher = ToolCallingMatcher::new_with_format(tool_choice, tools, preferred_format)?;
        let strategy: Box<dyn ToolCallStrategy> =
            if preferred_format == Some(ToolCallFormat::Harmony) {
                Box::new(HarmonyToolCallStrategy::new()?)
            } else {
                Box::new(TextToolCallStrategy::new(preferred_format))
            };
        Ok(Self {
            matcher,
            strategy,
            grammar: ToolGrammarState::Inactive,
            obligation: ToolObligation::default(),
        })
    }

    pub(crate) fn observe_token(&mut self, token: u32, bytes: &[u8]) {
        self.strategy.observe_token(token, bytes);
    }

    pub(crate) fn requires_special_tokens(&self) -> bool {
        true
    }

    pub(crate) fn has_reasoning(&self) -> bool {
        self.strategy.has_reasoning()
    }

    pub(crate) fn required_tool_call_unsatisfied(&self) -> bool {
        self.obligation
            .unsatisfied(self.matcher.requires_tool_call())
    }

    pub(crate) fn content_delta(&mut self) -> Option<String> {
        self.strategy.content_delta()
    }

    pub(crate) fn reasoning_delta(&mut self) -> Option<String> {
        self.strategy.reasoning_delta()
    }

    pub(crate) fn content(&self) -> Option<String> {
        self.strategy.content()
    }

    pub(crate) fn reasoning_content(&self) -> Option<String> {
        self.strategy.reasoning_content()
    }

    pub(crate) fn finalize(&mut self) {
        self.strategy.finalize();
    }

    pub(crate) fn maybe_activate_continuation_grammar(
        &mut self,
        text: Option<&str>,
    ) -> Option<TopLevelGrammar> {
        if self.grammar != ToolGrammarState::Inactive || !self.matcher.allows_tool_call() {
            return None;
        }
        let tools = self.matcher.tools()?;
        self.strategy.continuation_grammar(text, tools)
    }

    pub(crate) fn maybe_force_required_grammar(
        &mut self,
        remaining: usize,
        max_generation_len: usize,
        force_now: bool,
    ) -> Option<TopLevelGrammar> {
        if self.grammar != ToolGrammarState::Inactive {
            return None;
        }
        let requires_tool_call = self.matcher.requires_tool_call();
        if !self.obligation.unsatisfied(requires_tool_call)
            || (!force_now
                && !self
                    .obligation
                    .should_force(requires_tool_call, max_generation_len, remaining))
        {
            return None;
        }
        let tools = self.matcher.tools()?;
        let boundary = self.strategy.required_boundary();
        Some(self.strategy.required_grammar(tools, boundary))
    }

    pub(crate) fn required_tool_call_deadline_status(
        generated: usize,
        max_generation_len: usize,
    ) -> (usize, usize, usize) {
        let deadline = required_tool_call_deadline_tokens(max_generation_len);
        let remaining = max_generation_len.saturating_sub(generated);
        (generated, remaining, deadline)
    }

    pub(crate) fn mark_grammar_active(&mut self, forced: bool) {
        self.grammar = ToolGrammarState::Active { forced };
        if forced {
            self.obligation.mark_forced();
        }
    }

    pub(crate) fn clear_active_grammar(&mut self) -> bool {
        if matches!(self.grammar, ToolGrammarState::Active { .. }) {
            self.grammar = ToolGrammarState::Inactive;
            self.obligation.clear_forced();
            true
        } else {
            false
        }
    }

    pub(crate) fn is_stop_token_blocked(
        &self,
        tok: u32,
        eos_tok: Option<&[u32]>,
        stop_tokens: &[u32],
    ) -> bool {
        self.obligation
            .unsatisfied(self.matcher.requires_tool_call())
            && (eos_tok.is_some_and(|tokens| tokens.contains(&tok)) || stop_tokens.contains(&tok))
    }

    pub(crate) fn prefix_status(&self, message_prefix: &str) -> Result<(bool, bool)> {
        self.matcher.prefix_could_be_tool(message_prefix)
    }

    pub(crate) fn complete_if_tool_call(
        &mut self,
        message: &str,
    ) -> anyhow::Result<Vec<ToolCallResponse>> {
        let calls = self.matcher.get_call(message)?;
        if !calls.is_empty() {
            self.obligation
                .mark_satisfied(self.matcher.requires_tool_call());
        }
        Ok(calls)
    }

    pub(crate) fn parse_streaming(
        &mut self,
        content_delta: Option<String>,
        raw_delta: &str,
        has_external_reasoning: bool,
        is_done: bool,
    ) -> Result<ToolCallParse> {
        if self.strategy.has_reasoning() {
            if is_done && self.strategy.has_tool_calls() {
                self.obligation
                    .mark_satisfied(self.matcher.requires_tool_call());
                return Ok(ToolCallParse {
                    content: content_delta,
                    reasoning_content: None,
                    tool_calls: self.strategy.finalize_tool_calls(),
                    tool_use_still_possible: false,
                    tool_use_is_done: true,
                });
            }
            return Ok(ToolCallParse::empty(content_delta));
        }

        let raw_text = match content_delta {
            Some(content_delta) => content_delta,
            None if has_external_reasoning => return Ok(ToolCallParse::empty(None)),
            None => raw_delta.to_string(),
        };
        let (tool_use_still_possible, tool_use_is_done) =
            self.matcher.prefix_could_be_tool(&raw_text)?;
        let (content, tool_calls) = self
            .matcher
            .get_call_with_content(&raw_text)
            .map_err(candle_core::Error::msg)?;
        if !tool_calls.is_empty() {
            self.obligation
                .mark_satisfied(self.matcher.requires_tool_call());
        }
        Ok(ToolCallParse {
            content,
            reasoning_content: None,
            tool_use_still_possible,
            tool_use_is_done: tool_use_is_done || !tool_calls.is_empty(),
            tool_calls,
        })
    }

    pub(crate) fn finalize_for_response(
        &mut self,
        raw_text: &str,
        parsed_content: Option<String>,
        reasoning_content: Option<String>,
    ) -> Result<ToolCallParse> {
        if self.strategy.has_reasoning() {
            let tool_calls = self.strategy.finalize_tool_calls();
            if !tool_calls.is_empty() {
                self.obligation
                    .mark_satisfied(self.matcher.requires_tool_call());
            }
            return Ok(ToolCallParse {
                content: self.strategy.content(),
                reasoning_content: self.strategy.reasoning_content(),
                tool_use_still_possible: false,
                tool_use_is_done: !tool_calls.is_empty(),
                tool_calls,
            });
        }

        let text = parsed_content.unwrap_or_else(|| raw_text.to_string());
        let (content, tool_calls) = self
            .matcher
            .get_call_with_content(&text)
            .map_err(candle_core::Error::msg)?;
        if !tool_calls.is_empty() {
            self.obligation
                .mark_satisfied(self.matcher.requires_tool_call());
        }
        Ok(ToolCallParse {
            content,
            reasoning_content,
            tool_use_still_possible: false,
            tool_use_is_done: !tool_calls.is_empty(),
            tool_calls,
        })
    }
}

pub(crate) fn required_tool_call_deadline_tokens(max_generation_len: usize) -> usize {
    (max_generation_len / REQUIRED_TOOL_CALL_DEADLINE_DIVISOR).clamp(
        REQUIRED_TOOL_CALL_DEADLINE_MIN_TOKENS,
        REQUIRED_TOOL_CALL_DEADLINE_MAX_TOKENS,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::NamedFunctionToolChoice;
    use crate::{Function, Tool, ToolType};
    use serde_json::json;

    fn tool(name: &str) -> Tool {
        Tool {
            tp: ToolType::Function,
            function: Function {
                name: name.to_string(),
                description: None,
                parameters: None,
                strict: None,
            },
        }
    }

    fn lark(grammar: &TopLevelGrammar) -> &str {
        grammar.grammars[0].lark_grammar.as_ref().unwrap()
    }

    #[test]
    fn auto_text_prefix_tool_call_activates_continuation_grammar() {
        let tools = vec![tool("get_weather")];
        let mut state = ToolCallState::new(ToolChoice::Auto, Some(&tools), None).unwrap();

        let grammar = state
            .maybe_activate_continuation_grammar(Some("<tool_call>"))
            .unwrap();

        assert!(lark(&grammar).contains("json_call"));
        assert_eq!(grammar.grammars.len(), 2);
    }

    #[test]
    fn tool_call_choice_none_does_not_activate_continuation_grammar() {
        let tools = vec![tool("get_weather")];
        let mut state = ToolCallState::new(ToolChoice::None, Some(&tools), None).unwrap();

        assert!(state
            .maybe_activate_continuation_grammar(Some("<tool_call>"))
            .is_none());
    }

    #[test]
    fn required_tool_call_deadline_forces_text_grammar() {
        let tools = vec![tool("get_weather")];
        let mut state = ToolCallState::new(
            ToolChoice::Required,
            Some(&tools),
            Some(ToolCallFormat::Gemma4),
        )
        .unwrap();

        let grammar = state
            .maybe_force_required_grammar(2048, 8192, false)
            .unwrap();

        assert!(lark(&grammar).contains("start: <|tool_call> tool_call_body"));
    }

    #[test]
    fn required_harmony_tool_call_deadline_forces_native_grammar() {
        let tools = vec![tool("get_weather")];
        let mut state = ToolCallState::new(
            ToolChoice::Required,
            Some(&tools),
            Some(ToolCallFormat::Harmony),
        )
        .unwrap();

        let grammar = state
            .maybe_force_required_grammar(8192, 8192, true)
            .unwrap();

        assert!(lark(&grammar).contains("<|channel|>"));
        assert!(lark(&grammar).contains("commentary to=functions.get_weather "));
    }

    #[test]
    fn stop_token_under_required_tool_call_is_blocked() {
        let tools = vec![tool("get_weather")];
        let state = ToolCallState::new(ToolChoice::Required, Some(&tools), None).unwrap();

        assert!(state.is_stop_token_blocked(2, Some(&[2]), &[]));
        assert!(state.is_stop_token_blocked(9, None, &[9]));
    }

    #[test]
    fn grammar_completion_deactivates_tool_grammar_state() {
        let tools = vec![tool("get_weather")];
        let mut state = ToolCallState::new(ToolChoice::Auto, Some(&tools), None).unwrap();

        state.mark_grammar_active(false);

        assert!(state.clear_active_grammar());
        assert!(!state.clear_active_grammar());
    }

    #[test]
    fn completed_tool_call_satisfies_required_obligation() {
        let tools = vec![tool("get_weather")];
        let mut state = ToolCallState::new(ToolChoice::Required, Some(&tools), None).unwrap();

        let calls = state
            .complete_if_tool_call(r#"{"name":"get_weather","parameters":{"city":"Paris"}}"#)
            .unwrap();

        assert_eq!(calls.len(), 1);
        assert!(!state.required_tool_call_unsatisfied());
    }

    #[test]
    fn named_tool_choice_narrows_allowed_tool_name_and_schema() {
        let tools = vec![tool("get_weather"), tool("search")];
        let choice = ToolChoice::NamedFunction(NamedFunctionToolChoice {
            tp: ToolType::Function,
            name: "search".to_string(),
        });
        let mut state = ToolCallState::new(choice, Some(&tools), None).unwrap();

        let grammar = state
            .maybe_force_required_grammar(8192, 8192, true)
            .unwrap();
        let schema = grammar.grammars[1].json_schema.as_ref().unwrap();
        let names = schema["properties"]["name"]["enum"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_str().unwrap())
            .collect::<Vec<_>>();

        assert_eq!(names, vec!["search"]);
    }

    #[test]
    fn freeform_text_before_forced_text_tool_call_is_preserved() {
        let tools = vec![tool("get_weather")];
        let mut state = ToolCallState::new(ToolChoice::Required, Some(&tools), None).unwrap();

        let parsed = state
            .finalize_for_response(
                r#"Before <tool_call>{"name":"get_weather","arguments":{"city":"Paris"}}</tool_call>"#,
                None,
                None,
            )
            .unwrap();

        assert_eq!(parsed.content, Some("Before ".to_string()));
        assert_eq!(parsed.tool_calls.len(), 1);
        assert_eq!(
            parsed.tool_calls[0].function.arguments,
            json!({"city":"Paris"}).to_string()
        );
    }
}
