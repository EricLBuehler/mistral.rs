//! OpenAI Harmony format parsing for GPT-OSS models.
//!
//! The Harmony format uses channels to separate different types of content:
//! - `analysis`: Chain-of-thought reasoning (internal, not for end users)
//! - `commentary`: Tool call preambles and explanations
//! - `final`: User-facing response content
//!
//! Tool calls in Harmony are indicated by the `recipient` field being set to:
//! - `functions.tool_name` for user-defined tools
//! - `browser.search`, `browser.open`, `browser.find` for browser tool
//! - `python` for python tool
//!
//! This module provides incremental parsing of Harmony-formatted token streams.

use openai_harmony::{
    chat::Role, load_harmony_encoding, HarmonyEncoding, HarmonyEncodingName, StreamableParser,
};
use std::sync::OnceLock;
use uuid::Uuid;

/// Extract the tool name from a recipient string.
/// - "functions.my_tool" -> "my_tool"
/// - "browser.search" -> "browser.search"
/// - "python" -> "python"
fn extract_tool_name(recipient: &str) -> String {
    if let Some(name) = recipient.strip_prefix("functions.") {
        name.to_string()
    } else {
        // For builtin tools like "browser.search" or "python", use the full recipient
        recipient.to_string()
    }
}

/// Channel types in Harmony format
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HarmonyChannel {
    /// Chain-of-thought reasoning (internal, not for end users)
    Analysis,
    /// Tool call preambles and explanations
    Commentary,
    /// User-facing response content
    Final,
}

impl HarmonyChannel {
    /// Parse a channel name string into a HarmonyChannel
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "analysis" => Some(Self::Analysis),
            "commentary" => Some(Self::Commentary),
            "final" => Some(Self::Final),
            _ => None,
        }
    }
}

/// Incremental delta from Harmony parsing
#[derive(Debug, Clone, Default)]
pub struct HarmonyDelta {
    /// New analysis/reasoning content since last delta
    pub analysis_delta: Option<String>,
    /// New commentary content since last delta
    pub commentary_delta: Option<String>,
    /// New final response content since last delta
    pub final_delta: Option<String>,
    /// Currently active channel
    pub current_channel: Option<HarmonyChannel>,
}

impl HarmonyDelta {
    /// Check if this delta has any content
    pub fn has_content(&self) -> bool {
        self.analysis_delta.is_some()
            || self.commentary_delta.is_some()
            || self.final_delta.is_some()
    }

    /// Get reasoning content (analysis + commentary without tool calls)
    pub fn reasoning_content(&self) -> Option<String> {
        match (&self.analysis_delta, &self.commentary_delta) {
            (Some(a), Some(c)) => Some(format!("{}{}", a, c)),
            (Some(a), None) => Some(a.clone()),
            (None, Some(c)) => Some(c.clone()),
            (None, None) => None,
        }
    }
}

/// Accumulated content for each channel
#[derive(Debug, Clone, Default)]
pub struct HarmonyAccumulated {
    /// Accumulated analysis content
    pub analysis: String,
    /// Accumulated commentary content
    pub commentary: String,
    /// Accumulated final content
    pub final_content: String,
}

/// A tool call parsed from Harmony format
#[derive(Debug, Clone)]
pub struct HarmonyToolCall {
    /// Unique ID for this tool call
    pub id: String,
    /// The function name (extracted from recipient like "functions.tool_name")
    pub name: String,
    /// The JSON arguments as a string
    pub arguments: String,
}

impl HarmonyAccumulated {
    /// Get all reasoning content (analysis + commentary)
    pub fn reasoning_content(&self) -> Option<String> {
        let combined = format!("{}{}", self.analysis, self.commentary);
        if combined.is_empty() {
            None
        } else {
            Some(combined)
        }
    }
}

/// Context for tracking Harmony parsing state within a sequence.
///
/// This wraps the openai-harmony crate's StreamableParser and provides
/// delta extraction for streaming responses.
pub struct HarmonyContext {
    parser: StreamableParser,
    // Track lengths for delta extraction (for parser content)
    last_analysis_len: usize,
    last_commentary_len: usize,
    last_final_len: usize,
    // Accumulated content
    accumulated: HarmonyAccumulated,
    // Track which channel we're currently in
    channel: Option<HarmonyChannel>,
    // Track positions for streaming deltas (what has been sent)
    sent_reasoning_len: usize,
    sent_final_len: usize,
    // Tool call tracking
    tool_calls: Vec<HarmonyToolCall>,
    // Track current tool call being built (recipient, accumulated_args)
    current_tool_call: Option<(String, String)>,
    // Track how much of current tool call args have been sent
    sent_tool_args_len: usize,
}

impl HarmonyContext {
    /// Create a new Harmony parsing context
    pub fn new() -> Result<Self, anyhow::Error> {
        let encoding = get_harmony_encoding().clone();
        let parser = StreamableParser::new(encoding, Some(Role::Assistant))
            .map_err(|e| anyhow::anyhow!("Failed to create Harmony parser: {:?}", e))?;
        Ok(Self {
            parser,
            last_analysis_len: 0,
            last_commentary_len: 0,
            last_final_len: 0,
            accumulated: HarmonyAccumulated::default(),
            channel: None,
            sent_reasoning_len: 0,
            sent_final_len: 0,
            tool_calls: Vec::new(),
            current_tool_call: None,
            sent_tool_args_len: 0,
        })
    }

    /// Process a token and return any new delta content
    pub fn process_token(&mut self, token_id: u32) -> HarmonyDelta {
        // process() returns Result, ignore errors for robustness
        let _ = self.parser.process(token_id);
        self.extract_delta()
    }

    /// Extract delta since last call
    fn extract_delta(&mut self) -> HarmonyDelta {
        let mut delta = HarmonyDelta::default();

        // Get current channel from parser
        if let Some(channel_str) = self.parser.current_channel() {
            if let Some(channel) = HarmonyChannel::parse(&channel_str) {
                self.channel = Some(channel);
                delta.current_channel = Some(channel);
            }
        }

        // Check for tool calls via recipient field
        // Recipient is set to "functions.tool_name" when making a tool call
        let current_recipient = self.parser.current_recipient();

        // Get current content and extract delta based on channel
        // current_content() returns Result<String>
        if let Ok(content) = self.parser.current_content() {
            // Check if this is a tool call
            // Tool calls have recipients like:
            // - "functions.tool_name" for user-defined tools
            // - "browser.search", "browser.open", "browser.find" for browser tool
            // - "python" for python tool
            if let Some(ref recipient) = current_recipient {
                let is_tool_call = recipient.starts_with("functions.")
                    || recipient.starts_with("browser.")
                    || recipient == "python";

                if is_tool_call {
                    // This is a tool call - track it
                    // Check if this is the same tool call or a different one
                    let is_same_tool_call = self
                        .current_tool_call
                        .as_ref()
                        .is_some_and(|(existing, _)| existing == recipient);

                    if is_same_tool_call {
                        // Same tool call, update arguments
                        if let Some((_, ref mut args)) = self.current_tool_call {
                            *args = content.clone();
                        }
                    } else {
                        // Different tool call or no current tool call
                        // Finalize previous tool call if any
                        if let Some((prev_recipient, prev_args)) = self.current_tool_call.take() {
                            let prev_name = extract_tool_name(&prev_recipient);
                            self.tool_calls.push(HarmonyToolCall {
                                id: format!("call_{}", Uuid::new_v4()),
                                name: prev_name,
                                arguments: prev_args,
                            });
                        }
                        // Start new tool call
                        self.current_tool_call = Some((recipient.clone(), content.clone()));
                        self.sent_tool_args_len = 0;
                    }
                    // Don't accumulate tool call content to final_content
                    return delta;
                }
            }

            // Not a tool call, handle normally by channel
            match self.channel {
                Some(HarmonyChannel::Analysis) => {
                    if content.len() > self.last_analysis_len {
                        let new_content = content[self.last_analysis_len..].to_string();
                        self.accumulated.analysis.push_str(&new_content);
                        delta.analysis_delta = Some(new_content);
                        self.last_analysis_len = content.len();
                    }
                }
                Some(HarmonyChannel::Commentary) => {
                    if content.len() > self.last_commentary_len {
                        let new_content = content[self.last_commentary_len..].to_string();
                        self.accumulated.commentary.push_str(&new_content);
                        delta.commentary_delta = Some(new_content);
                        self.last_commentary_len = content.len();
                    }
                }
                Some(HarmonyChannel::Final) | None => {
                    // Final channel OR no channel marker - treat content as final.
                    // This handles cases where the model responds without Harmony
                    // channel markers (e.g., after tool call results).
                    if content.len() > self.last_final_len {
                        let new_content = content[self.last_final_len..].to_string();
                        self.accumulated.final_content.push_str(&new_content);
                        delta.final_delta = Some(new_content);
                        self.last_final_len = content.len();
                    }
                }
            }
        }

        delta
    }

    /// Get the currently active channel
    pub fn current_channel(&self) -> Option<HarmonyChannel> {
        self.channel
    }

    /// Get all accumulated content
    pub fn accumulated(&self) -> &HarmonyAccumulated {
        &self.accumulated
    }

    /// Get accumulated reasoning content (analysis + commentary)
    pub fn reasoning_content(&self) -> Option<String> {
        self.accumulated.reasoning_content()
    }

    /// Get accumulated final content
    pub fn final_content(&self) -> Option<String> {
        if self.accumulated.final_content.is_empty() {
            None
        } else {
            Some(self.accumulated.final_content.clone())
        }
    }

    /// Get the reasoning delta since last call (for streaming).
    /// Returns new reasoning content that hasn't been sent yet.
    pub fn get_reasoning_delta(&mut self) -> Option<String> {
        let reasoning = format!(
            "{}{}",
            self.accumulated.analysis, self.accumulated.commentary
        );
        if reasoning.len() > self.sent_reasoning_len {
            let delta = reasoning[self.sent_reasoning_len..].to_string();
            self.sent_reasoning_len = reasoning.len();
            if delta.is_empty() {
                None
            } else {
                Some(delta)
            }
        } else {
            None
        }
    }

    /// Get the final content delta since last call (for streaming).
    /// Returns new final content that hasn't been sent yet.
    pub fn get_final_delta(&mut self) -> Option<String> {
        if self.accumulated.final_content.len() > self.sent_final_len {
            let delta = self.accumulated.final_content[self.sent_final_len..].to_string();
            self.sent_final_len = self.accumulated.final_content.len();
            if delta.is_empty() {
                None
            } else {
                Some(delta)
            }
        } else {
            None
        }
    }

    /// Signal end of stream to the parser
    pub fn process_eos(&mut self) {
        let _ = self.parser.process_eos();

        // Finalize any pending tool call
        if let Some((recipient, args)) = self.current_tool_call.take() {
            let name = extract_tool_name(&recipient);
            self.tool_calls.push(HarmonyToolCall {
                id: format!("call_{}", Uuid::new_v4()),
                name,
                arguments: args,
            });
        }
    }

    /// Get the recipient (for tool calls) if any
    pub fn current_recipient(&self) -> Option<String> {
        self.parser.current_recipient()
    }

    /// Check if there's a tool call in progress
    pub fn has_tool_call(&self) -> bool {
        self.current_tool_call.is_some() || !self.tool_calls.is_empty()
    }

    /// Get all completed tool calls
    pub fn get_tool_calls(&self) -> &[HarmonyToolCall] {
        &self.tool_calls
    }

    /// Get the current tool call being built (if any)
    /// Returns (recipient, arguments_so_far)
    pub fn get_current_tool_call(&self) -> Option<(&str, &str)> {
        self.current_tool_call
            .as_ref()
            .map(|(recipient, args)| (recipient.as_str(), args.as_str()))
    }

    /// Finalize any pending tool call and return all tool calls.
    /// This should be called when the sequence is done.
    /// Note: This takes ownership of the tool calls, so calling it twice
    /// will return an empty vector the second time.
    pub fn finalize_tool_calls(&mut self) -> Vec<HarmonyToolCall> {
        // Finalize any pending tool call
        if let Some((recipient, args)) = self.current_tool_call.take() {
            let name = extract_tool_name(&recipient);
            self.tool_calls.push(HarmonyToolCall {
                id: format!("call_{}", Uuid::new_v4()),
                name,
                arguments: args,
            });
        }
        // Take ownership to prevent duplicate returns if called multiple times
        std::mem::take(&mut self.tool_calls)
    }
}

/// Global harmony encoding (lazy loaded)
static HARMONY_ENCODING: OnceLock<HarmonyEncoding> = OnceLock::new();

/// Pre-initialize the Harmony encoding. This MUST be called from a non-async
/// context (e.g., during pipeline loading) before any async code runs.
/// The openai-harmony crate uses reqwest::blocking which creates its own
/// tokio runtime, so it cannot be called from within an existing async context.
pub fn prewarm_harmony_encoding() {
    let _ = HARMONY_ENCODING.get_or_init(|| {
        load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss)
            .expect("Failed to load Harmony encoding")
    });
}

/// Check if the Harmony encoding has been initialized.
pub fn is_harmony_encoding_ready() -> bool {
    HARMONY_ENCODING.get().is_some()
}

fn get_harmony_encoding() -> &'static HarmonyEncoding {
    HARMONY_ENCODING
        .get()
        .expect("Harmony encoding not initialized. Call prewarm_harmony_encoding() first.")
}

/// Check if a chat template uses Harmony format by looking for Harmony markers.
///
/// Returns true if the template contains Harmony-specific tokens like
/// `<|channel|>`, `<|start|>`, `<|message|>`, or `<|end|>`.
pub fn is_harmony_template(template: &str) -> bool {
    // Check for the most distinctive Harmony marker
    if template.contains("<|channel|>") {
        return true;
    }

    // Check for the combination of start/message/end which is characteristic of Harmony
    template.contains("<|start|>")
        && template.contains("<|message|>")
        && template.contains("<|end|>")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_harmony_template() {
        // Should detect Harmony templates
        assert!(is_harmony_template(
            "<|start|>system<|message|>content<|end|>"
        ));
        assert!(is_harmony_template(
            "some prefix <|channel|>analysis<|message|>thinking"
        ));

        // Should not detect non-Harmony templates
        assert!(!is_harmony_template("<|im_start|>system<|im_end|>"));
        assert!(!is_harmony_template("regular chat template"));
    }

    #[test]
    fn test_harmony_channel_from_str() {
        assert_eq!(
            HarmonyChannel::parse("analysis"),
            Some(HarmonyChannel::Analysis)
        );
        assert_eq!(
            HarmonyChannel::parse("commentary"),
            Some(HarmonyChannel::Commentary)
        );
        assert_eq!(HarmonyChannel::parse("final"), Some(HarmonyChannel::Final));
        assert_eq!(HarmonyChannel::parse("unknown"), None);
    }

    #[test]
    fn test_harmony_delta_has_content() {
        let empty = HarmonyDelta::default();
        assert!(!empty.has_content());

        let with_analysis = HarmonyDelta {
            analysis_delta: Some("thinking".to_string()),
            ..Default::default()
        };
        assert!(with_analysis.has_content());

        let with_final = HarmonyDelta {
            final_delta: Some("response".to_string()),
            ..Default::default()
        };
        assert!(with_final.has_content());
    }

    #[test]
    fn test_harmony_delta_reasoning_content() {
        let both = HarmonyDelta {
            analysis_delta: Some("thinking ".to_string()),
            commentary_delta: Some("about tools".to_string()),
            ..Default::default()
        };
        assert_eq!(
            both.reasoning_content(),
            Some("thinking about tools".to_string())
        );

        let only_analysis = HarmonyDelta {
            analysis_delta: Some("just thinking".to_string()),
            ..Default::default()
        };
        assert_eq!(
            only_analysis.reasoning_content(),
            Some("just thinking".to_string())
        );

        let none = HarmonyDelta::default();
        assert_eq!(none.reasoning_content(), None);
    }
}
