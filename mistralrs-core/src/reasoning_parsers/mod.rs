//! Unified reasoning/thinking parser framework.
//!
//! Supports multiple reasoning formats:
//! - **Think tags**: `<think>...</think>` (DeepSeek R1, QwQ, SmolLM3)
//! - **Channel tags**: `<|channel>thought\n...<channel|>` (Gemma 4)
//! - **Harmony**: Token-based channel format (GPT-OSS)

pub mod harmony;
pub mod tag_based;

pub use harmony::{HarmonyContext, HarmonyToolCall};
pub use tag_based::TagReasoningContext;

/// Trait for reasoning content parsers.
///
/// All reasoning parsers extract two streams from model output:
/// - **content**: The user-visible response
/// - **reasoning**: Internal chain-of-thought (hidden from user)
pub trait ReasoningParser: Send + Sync {
    /// Process incremental bytes (for text-based parsers like tag_based).
    fn process_bytes(&mut self, bytes: &[u8]);
    /// Process a token ID (for token-based parsers like Harmony).
    fn process_token(&mut self, _token_id: u32) {}
    /// Finalize at end of stream (flush buffers, handle unclosed blocks).
    fn finalize(&mut self);
    /// Get new content since last call (for streaming).
    fn get_content_delta(&mut self) -> Option<String>;
    /// Get new reasoning since last call (for streaming).
    fn get_reasoning_delta(&mut self) -> Option<String>;
    /// Get all accumulated content.
    fn content(&self) -> Option<String>;
    /// Get all accumulated reasoning.
    fn reasoning_content(&self) -> Option<String>;
    /// Check if there are any tool calls (Harmony-specific, defaults to false).
    fn has_tool_calls(&self) -> bool {
        false
    }
    /// Finalize and return all tool calls (Harmony-specific, defaults to empty).
    fn finalize_tool_calls(&mut self) -> Vec<harmony::HarmonyToolCall> {
        vec![]
    }
    /// Check if a tool call grammar should be activated mid-stream.
    /// Returns true once when a new tool call is detected, then auto-clears.
    /// Only meaningful for Harmony mode; defaults to false.
    fn take_needs_tool_grammar_activation(&mut self) -> bool {
        false
    }
    /// Return the recipient of the current in-progress tool call (e.g.
    /// `"functions.get_weather"`).  Only meaningful for Harmony mode.
    fn current_tool_recipient(&self) -> Option<String> {
        None
    }
}

/// The active reasoning format for a sequence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReasoningMode {
    /// OpenAI Harmony format (GPT-OSS models)
    Harmony,
    /// Tag-based reasoning (think tags or Gemma 4 channel tags)
    TagBased,
}

/// Check if a template uses any reasoning format (think tags or channel tags).
pub fn is_reasoning_template(template: &str) -> bool {
    tag_based::is_think_tag_template(template) || tag_based::is_channel_tag_template(template)
}
