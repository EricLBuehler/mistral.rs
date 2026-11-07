use tokenizers::Tokenizer;

use crate::pipeline::chat_template::ChatTemplate;

use super::{
    budget::{supports_budget_tokenizer, HARMONY_FINAL_TRANSITION},
    tag_based::{CHANNEL_CLOSE_TAG, GEMMA_THOUGHT_CLOSE_TAG, THINK_CLOSE_TAG, THINK_OPEN_TAG},
    TagReasoningContext,
};

#[derive(Clone, Copy)]
pub(crate) enum ReasoningFormat {
    ThinkTags,
    GemmaChannel,
    GemmaThought,
    Harmony,
}

impl ReasoningFormat {
    pub(crate) fn from_template(template: &ChatTemplate) -> Option<Self> {
        if template.is_harmony_format() {
            Some(Self::Harmony)
        } else if template.uses_channel_tags() {
            Some(Self::GemmaChannel)
        } else if template.uses_think_tags() {
            Some(Self::ThinkTags)
        } else if template.uses_gemma_turns() {
            Some(Self::GemmaThought)
        } else {
            None
        }
    }

    pub(crate) fn tag_parser(self, prompt: &str) -> Option<TagReasoningContext> {
        match self {
            Self::ThinkTags if prompt.trim_end().ends_with(THINK_OPEN_TAG) => {
                Some(TagReasoningContext::new_in_think_block())
            }
            Self::ThinkTags => Some(TagReasoningContext::new_think_tags()),
            Self::GemmaChannel if prompt.contains("<|think|>") => {
                Some(TagReasoningContext::new_gemma_channel_with_implicit_thinking())
            }
            Self::GemmaChannel => Some(TagReasoningContext::new_gemma_channel()),
            Self::GemmaThought => Some(TagReasoningContext::new_gemma_thought()),
            Self::Harmony => None,
        }
    }

    pub(crate) fn supports_tokenizer(self, tokenizer: &Tokenizer) -> bool {
        let closing = match self {
            Self::ThinkTags => THINK_CLOSE_TAG,
            Self::GemmaChannel => CHANNEL_CLOSE_TAG,
            Self::GemmaThought => GEMMA_THOUGHT_CLOSE_TAG,
            Self::Harmony => HARMONY_FINAL_TRANSITION,
        };
        supports_budget_tokenizer(tokenizer, closing, matches!(self, Self::Harmony))
    }
}
