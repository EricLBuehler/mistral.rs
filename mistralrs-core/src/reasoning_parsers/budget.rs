use std::{borrow::Cow, sync::Arc};

use tokenizers::Tokenizer;

use super::TagReasoningContext;

const HARMONY_ASSISTANT_PREFIX: &str = "<|start|>assistant";
const HARMONY_START: &[u8] = b"<|start|>";
const HARMONY_MESSAGE: &str = "<|message|>";
const HARMONY_CHANNEL: &str = "<|channel|>";
const HARMONY_END: &[u8] = b"<|end|>";
const HARMONY_RETURN: &[u8] = b"<|return|>";
const HARMONY_CALL: &[u8] = b"<|call|>";
pub(crate) const HARMONY_FINAL_TRANSITION: &str =
    "<|end|><|start|>assistant<|channel|>final<|message|>";
pub(crate) const HARMONY_NONTERMINAL_TOKENS: &[&str] =
    &["<|end|>", "<|start|>", "<|message|>", "<|channel|>"];

fn encode_closing(tokenizer: &Tokenizer, text: &str) -> candle_core::Result<Vec<u32>> {
    let encoded = tokenizer
        .encode(text, false)
        .map_err(candle_core::Error::msg)?;
    let tokens = encoded.get_ids();
    if tokens.is_empty()
        || tokenizer
            .decode(tokens, false)
            .map_err(candle_core::Error::msg)?
            != text
    {
        candle_core::bail!("Cannot encode the end-of-thinking marker exactly");
    }
    Ok(tokens.to_vec())
}

pub(crate) fn supports_budget_tokenizer(
    tokenizer: &tokenizers::Tokenizer,
    closing: &str,
    harmony: bool,
) -> bool {
    let valid_harmony = !harmony
        || HARMONY_NONTERMINAL_TOKENS.iter().all(|marker| {
            tokenizer.token_to_id(marker).is_some_and(|id| {
                tokenizer
                    .encode(*marker, false)
                    .is_ok_and(|encoded| encoded.get_ids() == [id])
            })
        });
    valid_harmony && encode_closing(tokenizer, closing).is_ok()
}

enum BudgetParser {
    Tag(Box<TagReasoningContext>),
    Harmony(HarmonyBudgetState),
}

impl BudgetParser {
    fn force_text(&self, used: usize, limit: usize) -> Option<Cow<'_, str>> {
        if used < limit {
            return None;
        }
        match self {
            Self::Tag(parser) => parser.closing_suffix(),
            Self::Harmony(parser) => parser
                .is_reasoning()
                .then_some(Cow::Borrowed(HARMONY_FINAL_TRANSITION)),
        }
    }
}

enum HarmonyBudgetState {
    Header(Vec<u8>),
    Body { reasoning: bool },
    AwaitingStart,
}

impl HarmonyBudgetState {
    fn new(prompt: &str) -> Self {
        let suffix = prompt
            .rsplit_once(HARMONY_ASSISTANT_PREFIX)
            .map_or("", |(_, suffix)| suffix);
        if let Some((header, _)) = suffix.split_once(HARMONY_MESSAGE) {
            Self::body(header.as_bytes())
        } else {
            Self::Header(suffix.as_bytes().to_vec())
        }
    }

    fn body(header: &[u8]) -> Self {
        let header = String::from_utf8_lossy(header);
        let channel = header
            .split_once(HARMONY_CHANNEL)
            .and_then(|(_, rest)| rest.split(|c: char| c.is_whitespace() || c == '<').next());
        let has_recipient = header
            .split_whitespace()
            .any(|part| part.starts_with("to="));
        Self::Body {
            reasoning: matches!(channel, Some("analysis" | "commentary")) && !has_recipient,
        }
    }

    fn is_reasoning(&self) -> bool {
        matches!(self, Self::Body { reasoning: true })
    }

    fn observe(&mut self, bytes: &[u8], is_special: bool) -> bool {
        match self {
            Self::Header(header) if is_special && bytes == HARMONY_MESSAGE.as_bytes() => {
                *self = Self::body(header);
                false
            }
            Self::Header(header) => {
                header.extend_from_slice(bytes);
                false
            }
            Self::Body { .. }
                if is_special && matches!(bytes, HARMONY_END | HARMONY_RETURN | HARMONY_CALL) =>
            {
                *self = Self::AwaitingStart;
                false
            }
            Self::Body { reasoning } => *reasoning,
            Self::AwaitingStart => {
                if is_special && bytes == HARMONY_START {
                    *self = Self::Header(Vec::new());
                }
                false
            }
        }
    }
}

struct TokenBudget {
    max_tokens: usize,
    tokenizer: Arc<Tokenizer>,
    forced_tokens: std::vec::IntoIter<u32>,
    nonterminal_tokens: Vec<u32>,
}

pub(crate) struct ReasoningBudget {
    parser: BudgetParser,
    limit: Option<TokenBudget>,
    reasoning_tokens: usize,
    pending_delimiter_tokens: usize,
}

impl ReasoningBudget {
    pub(crate) fn new_tag(parser: TagReasoningContext) -> Self {
        Self {
            parser: BudgetParser::Tag(Box::new(parser)),
            limit: None,
            reasoning_tokens: 0,
            pending_delimiter_tokens: 0,
        }
    }

    pub(crate) fn new_harmony(prompt: &str) -> Self {
        Self {
            parser: BudgetParser::Harmony(HarmonyBudgetState::new(prompt)),
            limit: None,
            reasoning_tokens: 0,
            pending_delimiter_tokens: 0,
        }
    }

    pub(crate) fn set_limit(
        &mut self,
        max_tokens: usize,
        tokenizer: Arc<Tokenizer>,
    ) -> anyhow::Result<()> {
        let harmony = matches!(self.parser, BudgetParser::Harmony(_));
        if !supports_budget_tokenizer(&tokenizer, self.closing_text(), harmony) {
            anyhow::bail!(
                "reasoning_max_tokens / thinking.budget_tokens cannot encode this model's end-of-thinking marker"
            );
        }
        self.configure_special_tokens(&tokenizer);
        let nonterminal_tokens = if harmony {
            HARMONY_NONTERMINAL_TOKENS
                .iter()
                .map(|text| {
                    tokenizer
                        .token_to_id(text)
                        .expect("validated Harmony marker")
                })
                .collect()
        } else {
            Vec::new()
        };
        self.limit = Some(TokenBudget {
            max_tokens,
            tokenizer,
            forced_tokens: Vec::new().into_iter(),
            nonterminal_tokens,
        });
        Ok(())
    }

    pub(crate) fn next_token(&mut self) -> candle_core::Result<Option<u32>> {
        let used = self.reasoning_tokens();
        let Some(limit) = self.limit.as_mut() else {
            return Ok(None);
        };
        if limit.forced_tokens.as_slice().is_empty() {
            if let Some(closing) = self.parser.force_text(used, limit.max_tokens) {
                limit.forced_tokens = encode_closing(&limit.tokenizer, &closing)?.into_iter();
            }
        }
        Ok(limit.forced_tokens.next())
    }

    pub(crate) fn decode_token(&self, token: u32) -> candle_core::Result<String> {
        self.limit
            .as_ref()
            .expect("forced tokens require a reasoning budget")
            .tokenizer
            .decode(&[token], false)
            .map_err(candle_core::Error::msg)
    }

    pub(crate) fn ignores_eos(&self, token: u32) -> bool {
        self.limit
            .as_ref()
            .is_some_and(|limit| limit.nonterminal_tokens.contains(&token))
    }

    pub(crate) fn configure_special_tokens(&mut self, tokenizer: &tokenizers::Tokenizer) {
        if let BudgetParser::Tag(parser) = &mut self.parser {
            parser.configure_special_tokens(tokenizer);
        }
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn has_limit(&self) -> bool {
        self.limit.is_some()
    }

    fn closing_text(&self) -> &str {
        match &self.parser {
            BudgetParser::Tag(parser) => parser.closing_text(),
            BudgetParser::Harmony(_) => HARMONY_FINAL_TRANSITION,
        }
    }

    #[cfg(test)]
    fn force_text(&self, limit: usize) -> Option<Cow<'_, str>> {
        self.parser.force_text(self.reasoning_tokens(), limit)
    }

    pub(crate) fn observe_token(&mut self, bytes: &[u8], is_special: bool) {
        match &mut self.parser {
            BudgetParser::Tag(parser) => {
                let previous_bytes = parser.reasoning_bytes();
                let delimiter_fragment = (is_special || !parser.closing_is_special())
                    && parser
                        .closing_suffix()
                        .is_some_and(|closing| closing.as_bytes().starts_with(bytes));
                parser.process_token_bytes(bytes, is_special);
                if parser.reasoning_bytes() > previous_bytes {
                    self.reasoning_tokens +=
                        self.pending_delimiter_tokens + usize::from(!delimiter_fragment);
                    self.pending_delimiter_tokens = 0;
                } else if parser.has_pending_reasoning_delimiter() {
                    self.pending_delimiter_tokens += 1;
                } else {
                    self.pending_delimiter_tokens = 0;
                }
                parser.discard_output();
            }
            BudgetParser::Harmony(parser) => {
                if parser.observe(bytes, is_special) {
                    self.reasoning_tokens += 1;
                }
            }
        }
    }

    pub(crate) fn reasoning_tokens(&self) -> usize {
        self.reasoning_tokens + self.pending_delimiter_tokens
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn native_tag_tokenizer(tags: &[&str]) -> tokenizers::Tokenizer {
        let mut tokenizer =
            tokenizers::Tokenizer::new(tokenizers::models::wordlevel::WordLevel::default());
        tokenizer.add_special_tokens(
            &tags
                .iter()
                .map(|tag| tokenizers::AddedToken::from(*tag, true))
                .collect::<Vec<_>>(),
        );
        tokenizer
    }

    fn think_budget() -> ReasoningBudget {
        ReasoningBudget::new_tag(TagReasoningContext::new_in_think_block())
    }

    fn observe(budget: &mut ReasoningBudget, text: &str) {
        budget.observe_token(
            text.as_bytes(),
            text.starts_with("<|") && text.ends_with("|>"),
        );
    }

    #[test]
    fn partial_closing_marker_and_reopening_preserve_reasoning_count() {
        let mut budget = think_budget();
        assert_eq!(budget.force_text(0).as_deref(), Some("</think>"));
        observe(&mut budget, "thought");
        observe(&mut budget, "</th");
        assert_eq!(budget.force_text(2).as_deref(), Some("ink>"));
        observe(&mut budget, "in");
        observe(&mut budget, "k>");
        assert_eq!(budget.reasoning_tokens(), 1);
        assert_eq!(budget.force_text(2), None);
        observe(&mut budget, "answer");
        observe(&mut budget, "<think>");
        assert_eq!(budget.reasoning_tokens(), 1);
        assert_eq!(budget.force_text(1).as_deref(), Some("</think>"));
    }

    #[test]
    fn marker_like_reasoning_counts_every_token() {
        let mut budget = think_budget();
        for text in ["first", "<", "xml>"] {
            observe(&mut budget, text);
        }
        assert_eq!(budget.reasoning_tokens(), 3);
        assert_eq!(budget.force_text(3).as_deref(), Some("</think>"));
    }

    #[test]
    fn native_markers_require_special_token_identity() {
        let tokenizer = native_tag_tokenizer(&["<think>", "</think>"]);
        let mut budget = ReasoningBudget::new_tag(TagReasoningContext::new_think_tags());
        budget.configure_special_tokens(&tokenizer);
        budget.observe_token(b"<th", false);
        budget.observe_token(b"ink>", false);
        assert_eq!(budget.force_text(0), None);
        assert_eq!(budget.reasoning_tokens(), 0);

        budget.observe_token(b"<think>", true);
        assert_eq!(budget.force_text(0).as_deref(), Some("</think>"));
        budget.observe_token(b"</th", false);
        budget.observe_token(b"ink>", false);
        assert_eq!(budget.reasoning_tokens(), 2);
        assert_eq!(budget.force_text(2).as_deref(), Some("</think>"));
        budget.observe_token(b"</think>", true);
        budget.observe_token(b"answer", false);
        assert_eq!(budget.reasoning_tokens(), 2);
        assert_eq!(budget.force_text(0), None);
    }

    #[test]
    fn plain_multitoken_closer_works_with_a_native_opener() {
        let tokenizer = native_tag_tokenizer(&["<think>"]);
        let mut budget = ReasoningBudget::new_tag(TagReasoningContext::new_think_tags());
        budget.configure_special_tokens(&tokenizer);
        budget.observe_token(b"<think>", true);
        for text in ["thought", "</th", "ink>", "answer"] {
            observe(&mut budget, text);
        }
        assert_eq!(budget.reasoning_tokens(), 1);
        assert_eq!(budget.force_text(0), None);
    }

    #[test]
    fn split_utf8_counts_tokens_and_allows_forced_closure() {
        for complete_codepoint in [false, true] {
            let mut budget = think_budget();
            budget.observe_token(&[0xf0, 0x9f], false);
            if complete_codepoint {
                budget.observe_token(&[0x98, 0x80], false);
            }
            let tokens = 1 + usize::from(complete_codepoint);
            assert_eq!(budget.reasoning_tokens(), tokens);
            assert_eq!(budget.force_text(tokens).as_deref(), Some("</think>"));
            observe(&mut budget, "</think>");
            observe(&mut budget, "answer");
            assert_eq!(budget.reasoning_tokens(), tokens);
            assert_eq!(budget.force_text(0), None);
        }
    }

    #[test]
    fn gemma_headers_are_completed_without_spending_the_budget() {
        for (parser, open, close) in [
            (
                TagReasoningContext::new_gemma_channel(),
                "<|channel>",
                "<channel|>",
            ),
            (
                TagReasoningContext::new_gemma_thought(),
                "<unused94>",
                "<unused95>",
            ),
            (
                TagReasoningContext::new_gemma_channel_with_implicit_thinking(),
                "",
                "<channel|>",
            ),
        ] {
            let mut budget = ReasoningBudget::new_tag(parser);
            if !open.is_empty() {
                observe(&mut budget, open);
            }
            observe(&mut budget, "tho");
            let suffix = format!("ught\n{close}");
            assert_eq!(
                budget.force_text(0).as_deref(),
                (!open.is_empty()).then_some(suffix.as_str())
            );
            observe(&mut budget, "ught\n");
            assert_eq!(budget.reasoning_tokens(), 0);
            assert_eq!(budget.force_text(0).as_deref(), Some(close));
            observe(&mut budget, "reasoning");
            assert_eq!(budget.force_text(1).as_deref(), Some(close));
        }
    }

    #[test]
    fn harmony_prompt_and_generated_headers_do_not_count_as_reasoning() {
        for (prompt, header) in [
            ("<|start|>assistant<|channel|>analysis<|message|>", &[][..]),
            (
                "<|start|>assistant<|channel|>ana",
                &["lysis", "<|message|>"][..],
            ),
        ] {
            let mut budget = ReasoningBudget::new_harmony(prompt);
            for token in header {
                assert_eq!(budget.force_text(0), None);
                observe(&mut budget, token);
            }
            assert_eq!(budget.reasoning_tokens(), 0);
            assert_eq!(
                budget.force_text(0).as_deref(),
                Some(HARMONY_FINAL_TRANSITION)
            );
            observe(&mut budget, "reasoning");
            assert_eq!(budget.reasoning_tokens(), 1);
        }
    }

    #[test]
    fn harmony_native_end_and_reopening_preserve_reasoning_count() {
        for channel in ["analysis", "final"] {
            let mut budget =
                ReasoningBudget::new_harmony("<|start|>assistant<|channel|>analysis<|message|>");
            budget.observe_token(HARMONY_END, false);
            assert_eq!(budget.reasoning_tokens(), 1);
            assert_eq!(
                budget.force_text(1).as_deref(),
                Some(HARMONY_FINAL_TRANSITION)
            );
            budget.observe_token(HARMONY_END, true);
            assert_eq!(budget.force_text(1), None);
            for token in [
                "<|start|>",
                "assistant",
                "<|channel|>",
                channel,
                "<|message|>",
                "text",
            ] {
                observe(&mut budget, token);
            }
            let reasoning = channel == "analysis";
            assert_eq!(budget.reasoning_tokens(), 1 + usize::from(reasoning));
            assert_eq!(
                budget.force_text(1).as_deref(),
                reasoning.then_some(HARMONY_FINAL_TRANSITION)
            );
        }
    }

    #[test]
    fn harmony_tool_arguments_do_not_spend_reasoning_budget() {
        for header in [
            "<|start|>assistant to=functions.weather<|channel|>commentary<|message|>",
            "<|start|>assistant<|channel|>commentary to=functions.weather<|message|>",
        ] {
            let mut budget = ReasoningBudget::new_harmony(header);
            observe(&mut budget, "{}");
            assert_eq!(budget.reasoning_tokens(), 0);
            assert_eq!(budget.force_text(0), None);
        }
    }
}
