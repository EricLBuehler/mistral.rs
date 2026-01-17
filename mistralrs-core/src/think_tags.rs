//! Parser for `<think>...</think>` tags in model output.
//!
//! Models like DeepSeek R1, QwQ, and SmolLM3 use `<think>` tags to wrap
//! chain-of-thought reasoning content. This module provides incremental
//! parsing to extract reasoning content separately from final response content.
//!
//! ## Format
//! ```text
//! <think>This is reasoning content</think>This is final content
//! ```
//!
//! ## Behavior
//! - Content before `<think>` goes to `content`
//! - Content inside `<think>...</think>` goes to `reasoning_content`
//! - Content after `</think>` goes to `content`
//! - Multiple `<think>` blocks are concatenated
//! - At EOS, treat unclosed `<think>` content as reasoning

const THINK_OPEN_TAG: &str = "<think>";
const THINK_CLOSE_TAG: &str = "</think>";

/// Context for tracking `<think>...</think>` tag parsing state within a sequence.
///
/// This provides incremental parsing of token streams containing think tags,
/// with delta extraction for streaming responses.
pub struct ThinkTagContext {
    /// Accumulated final content (outside think blocks)
    accumulated_content: String,
    /// Accumulated reasoning content (inside think blocks)
    accumulated_reasoning: String,
    /// Whether we're currently inside a `<think>` block
    in_think_block: bool,
    /// Buffer for handling partial tags split across token boundaries
    buffer: String,
    /// Length of content that has been sent (for delta tracking)
    sent_content_len: usize,
    /// Length of reasoning that has been sent (for delta tracking)
    sent_reasoning_len: usize,
    /// Buffer for handling incomplete UTF-8 byte sequences across token boundaries
    utf8_buffer: Vec<u8>,
}

impl ThinkTagContext {
    /// Create a new ThinkTagContext
    pub fn new() -> Self {
        Self {
            accumulated_content: String::new(),
            accumulated_reasoning: String::new(),
            in_think_block: false,
            buffer: String::new(),
            sent_content_len: 0,
            sent_reasoning_len: 0,
            utf8_buffer: Vec::new(),
        }
    }

    /// Create a new ThinkTagContext that starts inside a think block.
    ///
    /// Use this when the chat template hardcodes `<think>` in the generation prompt,
    /// so the model's output starts inside the think block without an opening tag.
    pub fn new_in_think_block() -> Self {
        Self {
            accumulated_content: String::new(),
            accumulated_reasoning: String::new(),
            in_think_block: true,
            buffer: String::new(),
            sent_content_len: 0,
            sent_reasoning_len: 0,
            utf8_buffer: Vec::new(),
        }
    }

    /// Process incremental bytes from the model output.
    ///
    /// This method handles bytes that may contain incomplete UTF-8 sequences
    /// (e.g., multi-byte characters like emojis split across token boundaries).
    pub fn process_bytes(&mut self, bytes: &[u8]) {
        // Combine with any leftover bytes from previous calls
        self.utf8_buffer.extend_from_slice(bytes);

        // Take ownership of buffer to avoid borrow issues
        let buffer = std::mem::take(&mut self.utf8_buffer);

        // Find the longest valid UTF-8 prefix
        let valid_up_to = match std::str::from_utf8(&buffer) {
            Ok(s) => {
                // All bytes are valid UTF-8
                self.process_text(s);
                return;
            }
            Err(e) => e.valid_up_to(),
        };

        // Process the valid portion
        if valid_up_to > 0 {
            // Safety: we just verified this portion is valid UTF-8
            let valid_str = unsafe { std::str::from_utf8_unchecked(&buffer[..valid_up_to]) };
            self.process_text(valid_str);
        }

        // Keep the incomplete bytes for next time
        self.utf8_buffer = buffer[valid_up_to..].to_vec();
    }

    /// Process incremental text from the model output.
    ///
    /// This method handles text that may contain partial tags split across
    /// token boundaries by buffering potentially incomplete tags.
    pub fn process_text(&mut self, text: &str) {
        // Add new text to buffer
        self.buffer.push_str(text);

        // Process the buffer, looking for complete tags
        loop {
            if self.in_think_block {
                // Looking for </think>
                if let Some(end_pos) = self.buffer.find(THINK_CLOSE_TAG) {
                    // Found closing tag - extract reasoning content before it
                    let reasoning = &self.buffer[..end_pos];
                    self.accumulated_reasoning.push_str(reasoning);
                    self.in_think_block = false;
                    // Remove processed part including the tag
                    self.buffer = self.buffer[end_pos + THINK_CLOSE_TAG.len()..].to_string();
                } else {
                    // No complete closing tag found
                    // Check if buffer might contain a partial closing tag at the end
                    let partial_len = self.potential_partial_tag_len(THINK_CLOSE_TAG);
                    if partial_len > 0 {
                        // Keep the potential partial tag in buffer, process the rest
                        let safe_len = self.buffer.len() - partial_len;
                        if safe_len > 0 {
                            let reasoning = &self.buffer[..safe_len];
                            self.accumulated_reasoning.push_str(reasoning);
                            self.buffer = self.buffer[safe_len..].to_string();
                        }
                    } else {
                        // No partial tag, process all content
                        self.accumulated_reasoning.push_str(&self.buffer);
                        self.buffer.clear();
                    }
                    break;
                }
            } else {
                // Looking for <think>
                if let Some(start_pos) = self.buffer.find(THINK_OPEN_TAG) {
                    // Found opening tag - extract content before it
                    let content = &self.buffer[..start_pos];
                    self.accumulated_content.push_str(content);
                    self.in_think_block = true;
                    // Remove processed part including the tag
                    self.buffer = self.buffer[start_pos + THINK_OPEN_TAG.len()..].to_string();
                } else {
                    // No complete opening tag found
                    // Check if buffer might contain a partial opening tag at the end
                    let partial_len = self.potential_partial_tag_len(THINK_OPEN_TAG);
                    if partial_len > 0 {
                        // Keep the potential partial tag in buffer, process the rest
                        let safe_len = self.buffer.len() - partial_len;
                        if safe_len > 0 {
                            let content = &self.buffer[..safe_len];
                            self.accumulated_content.push_str(content);
                            self.buffer = self.buffer[safe_len..].to_string();
                        }
                    } else {
                        // No partial tag, process all content
                        self.accumulated_content.push_str(&self.buffer);
                        self.buffer.clear();
                    }
                    break;
                }
            }
        }
    }

    /// Check how many characters at the end of the buffer could be the start of a tag.
    ///
    /// Returns the length of the potential partial tag, or 0 if none.
    fn potential_partial_tag_len(&self, tag: &str) -> usize {
        let buffer_bytes = self.buffer.as_bytes();
        let tag_bytes = tag.as_bytes();

        // Check each possible suffix of the buffer
        for suffix_len in 1..=tag.len().min(self.buffer.len()) {
            let suffix_start = self.buffer.len() - suffix_len;
            let suffix = &buffer_bytes[suffix_start..];
            let tag_prefix = &tag_bytes[..suffix_len];

            if suffix == tag_prefix {
                return suffix_len;
            }
        }
        0
    }

    /// Get the content delta since the last call.
    ///
    /// Returns new content that hasn't been sent yet, or None if no new content.
    pub fn get_content_delta(&mut self) -> Option<String> {
        if self.accumulated_content.len() > self.sent_content_len {
            let delta = self.accumulated_content[self.sent_content_len..].to_string();
            self.sent_content_len = self.accumulated_content.len();
            if delta.is_empty() {
                None
            } else {
                Some(delta)
            }
        } else {
            None
        }
    }

    /// Get the reasoning delta since the last call.
    ///
    /// Returns new reasoning content that hasn't been sent yet, or None if no new content.
    pub fn get_reasoning_delta(&mut self) -> Option<String> {
        if self.accumulated_reasoning.len() > self.sent_reasoning_len {
            let delta = self.accumulated_reasoning[self.sent_reasoning_len..].to_string();
            self.sent_reasoning_len = self.accumulated_reasoning.len();
            if delta.is_empty() {
                None
            } else {
                Some(delta)
            }
        } else {
            None
        }
    }

    /// Get the accumulated content (outside think blocks).
    ///
    /// Returns None if no content has been accumulated.
    pub fn content(&self) -> Option<String> {
        if self.accumulated_content.is_empty() {
            None
        } else {
            Some(self.accumulated_content.clone())
        }
    }

    /// Get the accumulated reasoning content (inside think blocks).
    ///
    /// Returns None if no reasoning content has been accumulated.
    pub fn reasoning_content(&self) -> Option<String> {
        if self.accumulated_reasoning.is_empty() {
            None
        } else {
            Some(self.accumulated_reasoning.clone())
        }
    }

    /// Finalize parsing at end of stream.
    ///
    /// If there's an unclosed `<think>` block, treat any remaining buffer
    /// content as reasoning. Otherwise, treat it as content.
    pub fn finalize(&mut self) {
        // First, flush any remaining UTF-8 bytes with replacement characters
        if !self.utf8_buffer.is_empty() {
            let text = String::from_utf8_lossy(&self.utf8_buffer);
            self.buffer.push_str(&text);
            self.utf8_buffer.clear();
        }

        if !self.buffer.is_empty() {
            if self.in_think_block {
                // Unclosed think block - remaining buffer is reasoning
                self.accumulated_reasoning.push_str(&self.buffer);
            } else {
                // Not in think block - remaining buffer is content
                self.accumulated_content.push_str(&self.buffer);
            }
            self.buffer.clear();
        }
    }

    /// Check if currently inside a think block
    pub fn is_in_think_block(&self) -> bool {
        self.in_think_block
    }
}

impl Default for ThinkTagContext {
    fn default() -> Self {
        Self::new()
    }
}

/// Check if a chat template uses `<think>...</think>` tags.
///
/// Returns true if the template contains both `<think>` and `</think>` markers.
pub fn is_think_tag_template(template: &str) -> bool {
    template.contains(THINK_OPEN_TAG) && template.contains(THINK_CLOSE_TAG)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_think_block() {
        let mut ctx = ThinkTagContext::new();
        ctx.process_text("<think>reasoning here</think>final content");
        ctx.finalize();

        assert_eq!(ctx.reasoning_content(), Some("reasoning here".to_string()));
        assert_eq!(ctx.content(), Some("final content".to_string()));
    }

    #[test]
    fn test_content_before_think() {
        let mut ctx = ThinkTagContext::new();
        ctx.process_text("prefix content<think>reasoning</think>suffix");
        ctx.finalize();

        assert_eq!(ctx.reasoning_content(), Some("reasoning".to_string()));
        assert_eq!(ctx.content(), Some("prefix contentsuffix".to_string()));
    }

    #[test]
    fn test_multiple_think_blocks() {
        let mut ctx = ThinkTagContext::new();
        ctx.process_text("<think>first</think>content<think>second</think>more");
        ctx.finalize();

        assert_eq!(ctx.reasoning_content(), Some("firstsecond".to_string()));
        assert_eq!(ctx.content(), Some("contentmore".to_string()));
    }

    #[test]
    fn test_incremental_parsing() {
        let mut ctx = ThinkTagContext::new();

        // Simulate tokens arriving one at a time
        ctx.process_text("<th");
        ctx.process_text("ink>");
        ctx.process_text("reas");
        ctx.process_text("oning");
        ctx.process_text("</th");
        ctx.process_text("ink>");
        ctx.process_text("final");
        ctx.finalize();

        assert_eq!(ctx.reasoning_content(), Some("reasoning".to_string()));
        assert_eq!(ctx.content(), Some("final".to_string()));
    }

    #[test]
    fn test_unclosed_think_block() {
        let mut ctx = ThinkTagContext::new();
        ctx.process_text("<think>reasoning without closing tag");
        ctx.finalize();

        assert_eq!(
            ctx.reasoning_content(),
            Some("reasoning without closing tag".to_string())
        );
        assert_eq!(ctx.content(), None);
    }

    #[test]
    fn test_no_think_blocks() {
        let mut ctx = ThinkTagContext::new();
        ctx.process_text("just regular content");
        ctx.finalize();

        assert_eq!(ctx.reasoning_content(), None);
        assert_eq!(ctx.content(), Some("just regular content".to_string()));
    }

    #[test]
    fn test_delta_tracking() {
        let mut ctx = ThinkTagContext::new();

        ctx.process_text("content1");
        assert_eq!(ctx.get_content_delta(), Some("content1".to_string()));
        assert_eq!(ctx.get_content_delta(), None); // Already sent

        ctx.process_text("content2");
        assert_eq!(ctx.get_content_delta(), Some("content2".to_string()));

        ctx.process_text("<think>reasoning");
        assert_eq!(ctx.get_reasoning_delta(), Some("reasoning".to_string()));
    }

    #[test]
    fn test_partial_tag_at_boundary() {
        let mut ctx = ThinkTagContext::new();

        // Tag split across multiple tokens
        ctx.process_text("content<");
        // The '<' should be buffered, not yet added to content
        ctx.process_text("think>reasoning</think>done");
        ctx.finalize();

        assert_eq!(ctx.reasoning_content(), Some("reasoning".to_string()));
        assert_eq!(ctx.content(), Some("contentdone".to_string()));
    }

    #[test]
    fn test_is_think_tag_template() {
        assert!(is_think_tag_template(
            "template with <think>content</think>"
        ));
        assert!(!is_think_tag_template("template without tags"));
        assert!(!is_think_tag_template("template with only <think>"));
        assert!(!is_think_tag_template("template with only </think>"));
    }

    #[test]
    fn test_empty_think_block() {
        let mut ctx = ThinkTagContext::new();
        ctx.process_text("<think></think>content");
        ctx.finalize();

        // Empty reasoning block results in empty string, which returns None
        assert_eq!(ctx.reasoning_content(), None);
        assert_eq!(ctx.content(), Some("content".to_string()));
    }

    #[test]
    fn test_nested_angle_brackets() {
        let mut ctx = ThinkTagContext::new();
        ctx.process_text("<think>some <xml> tags inside</think>final");
        ctx.finalize();

        assert_eq!(
            ctx.reasoning_content(),
            Some("some <xml> tags inside".to_string())
        );
        assert_eq!(ctx.content(), Some("final".to_string()));
    }

    #[test]
    fn test_emoji_in_content() {
        let mut ctx = ThinkTagContext::new();
        ctx.process_text("Hello 😀 world");
        ctx.finalize();

        assert_eq!(ctx.content(), Some("Hello 😀 world".to_string()));
    }

    #[test]
    fn test_emoji_split_across_bytes() {
        let mut ctx = ThinkTagContext::new();
        // Emoji 😀 is encoded as bytes: [0xF0, 0x9F, 0x98, 0x80]
        // Simulate it being split across multiple token boundaries
        ctx.process_bytes(b"Hi ");
        ctx.process_bytes(&[0xF0, 0x9F]); // First half of emoji
        ctx.process_bytes(&[0x98, 0x80]); // Second half of emoji
        ctx.process_bytes(b" there");
        ctx.finalize();

        assert_eq!(ctx.content(), Some("Hi 😀 there".to_string()));
    }

    #[test]
    fn test_emoji_in_think_block() {
        let mut ctx = ThinkTagContext::new();
        ctx.process_text("<think>thinking 🤔 hard</think>result ✨");
        ctx.finalize();

        assert_eq!(
            ctx.reasoning_content(),
            Some("thinking 🤔 hard".to_string())
        );
        assert_eq!(ctx.content(), Some("result ✨".to_string()));
    }

    #[test]
    fn test_emoji_split_across_think_boundary() {
        let mut ctx = ThinkTagContext::new();
        // Emoji at the boundary of think tag
        ctx.process_bytes(b"<think>reason");
        ctx.process_bytes(&[0xF0, 0x9F]); // First half of 😀
        ctx.process_bytes(&[0x98, 0x80]); // Second half of 😀
        ctx.process_bytes(b"</think>done");
        ctx.finalize();

        assert_eq!(ctx.reasoning_content(), Some("reason😀".to_string()));
        assert_eq!(ctx.content(), Some("done".to_string()));
    }

    #[test]
    fn test_multibyte_characters_incremental() {
        let mut ctx = ThinkTagContext::new();
        // Test with various multi-byte characters
        // 中 is 3 bytes: [0xE4, 0xB8, 0xAD]
        ctx.process_bytes(&[0xE4]);
        ctx.process_bytes(&[0xB8]);
        ctx.process_bytes(&[0xAD]);
        ctx.process_bytes(b" text");
        ctx.finalize();

        assert_eq!(ctx.content(), Some("中 text".to_string()));
    }

    #[test]
    fn test_new_in_think_block() {
        // Simulate when chat template hardcodes <think> in generation prompt
        // Model output starts inside think block without opening tag
        let mut ctx = ThinkTagContext::new_in_think_block();
        ctx.process_text("reasoning here</think>final content");
        ctx.finalize();

        assert_eq!(ctx.reasoning_content(), Some("reasoning here".to_string()));
        assert_eq!(ctx.content(), Some("final content".to_string()));
    }

    #[test]
    fn test_new_in_think_block_unclosed() {
        // Model output inside think block that never closes
        let mut ctx = ThinkTagContext::new_in_think_block();
        ctx.process_text("reasoning without closing tag");
        ctx.finalize();

        assert_eq!(
            ctx.reasoning_content(),
            Some("reasoning without closing tag".to_string())
        );
        assert_eq!(ctx.content(), None);
    }
}
